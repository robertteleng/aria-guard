"""
Observer para captura de frames desde Aria, webcam o video.
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import cv2
import numpy as np


class BaseObserver(ABC):
    """Interfaz base para observers."""

    # Horizontal field of view in radians (subclasses override)
    fov_h: float = 1.15  # ~66° default (typical webcam)

    @abstractmethod
    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """Obtiene el frame más reciente."""
        pass

    @abstractmethod
    def stop(self):
        """Detiene el observer."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del observer."""
        return {}


class MockObserver(BaseObserver):
    """Observer para webcam o video (desarrollo sin Aria)."""

    fov_h = 1.15  # ~66° typical webcam

    def __init__(self, source: str = "webcam", video_path: str = None, use_nvdec: bool = True):
        """
        Args:
            source: "webcam" o "video"
            video_path: Ruta al video si source="video"
            use_nvdec: Si True, intenta usar NVDEC para decodificación GPU
        """
        self.source = source
        self._stop = False
        self._lock = threading.Lock()
        self._current_frame = None
        self._frame_count = 0
        self._start_time = time.time()
        self._use_nvdec = False
        self._gpu_reader = None
        self._video_path = video_path

        # Intentar NVDEC para videos (no webcam)
        if source != "webcam" and use_nvdec and video_path:
            try:
                # Check if cudacodec is available
                if hasattr(cv2, 'cudacodec'):
                    self._gpu_reader = cv2.cudacodec.createVideoReader(video_path)
                    self._use_nvdec = True
                    # Get video info
                    format_info = self._gpu_reader.format()
                    self._target_fps = format_info.fps if hasattr(format_info, 'fps') else 30
                    print(f"[OBSERVER] ✓ NVDEC habilitado - Video FPS: {self._target_fps:.1f}")
            except Exception as e:
                msg = str(e)
                if "(-213" in msg or "disabled for current build or platform" in msg:
                    print(
                        "[OBSERVER] NVDEC no disponible en runtime. "
                        "Verifica NVIDIA_DRIVER_CAPABILITIES=compute,utility,video "
                        "y que libnvcuvid del host esté visible en el contenedor."
                    )
                print(f"[OBSERVER] NVDEC no disponible: {e}, usando CPU")
                self._use_nvdec = False

        # Fallback a CPU VideoCapture
        if not self._use_nvdec:
            if source == "webcam":
                self._cap = cv2.VideoCapture(0)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(1)
            else:
                self._cap = cv2.VideoCapture(video_path)

            if not self._cap.isOpened():
                raise RuntimeError(f"No se pudo abrir {source}: {video_path}")

            if source == "webcam":
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self._target_fps = 30
            else:
                self._target_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
                print(f"[OBSERVER] Video FPS: {self._target_fps:.1f} (CPU decode)")

        self._frame_interval = 1.0 / self._target_fps

        # Hilo de captura
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        decode_mode = "NVDEC GPU" if self._use_nvdec else "CPU"
        print(f"[OBSERVER] MockObserver iniciado ({source}, {decode_mode})")

    def _capture_loop(self):
        """Hilo de captura continua con timing controlado."""
        last_frame_time = time.time()

        while not self._stop:
            # Controlar timing para respetar FPS del video
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
            last_frame_time = time.time()

            if self._use_nvdec and self._gpu_reader:
                # NVDEC: decode on GPU, download to CPU
                ret, gpu_frame = self._gpu_reader.nextFrame()
                if ret:
                    frame = gpu_frame.download()
                    with self._lock:
                        self._current_frame = frame
                        self._frame_count += 1
                else:
                    # Reiniciar video - recrear reader
                    if self.source != "webcam":
                        try:
                            # cudacodec doesn't have seek, recreate reader
                            if self._video_path:
                                self._gpu_reader = cv2.cudacodec.createVideoReader(self._video_path)
                        except Exception:
                            pass
            else:
                # CPU decode
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._current_frame = frame
                        self._frame_count += 1
                else:
                    # Si es video, reiniciar
                    if self.source == "video":
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """
        Obtiene frame. Solo soporta 'rgb' (webcam/video no tiene eye tracking).
        """
        if camera == "eye":
            return None  # Mock no tiene eye tracking

        with self._lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "source": self.source,
            "frames": self._frame_count,
            "fps": self._frame_count / elapsed if elapsed > 0 else 0,
            "uptime": elapsed
        }

    def stop(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if hasattr(self, '_cap') and self._cap:
            self._cap.release()
        if self._gpu_reader:
            self._gpu_reader = None
        print("[OBSERVER] MockObserver detenido")


class RealSenseObserver(BaseObserver):
    """
    Observer para Intel RealSense D435 con depth nativo.

    Ventaja: El D435 tiene sensor de profundidad hardware, no necesita
    modelo de depth estimation (Depth Anything).
    """

    fov_h = 1.518  # ~87° D435 RGB sensor

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            fps: Frames por segundo
        """
        self._stop = False
        self._lock = threading.Lock()
        self._current_frame = None
        self._current_depth = None
        self._current_depth_raw = None  # uint16 en mm (para distancias reales)
        self._frame_count = 0
        self._start_time = time.time()

        try:
            import pyrealsense2 as rs
            self._rs = rs
        except ImportError:
            raise RuntimeError("pyrealsense2 no instalado. Instalar con: pip install pyrealsense2")

        # Configurar pipeline
        self._pipeline = rs.pipeline()
        config = rs.config()

        # Habilitar streams RGB y Depth
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Iniciar pipeline
        try:
            self._profile = self._pipeline.start(config)
            print(f"[OBSERVER] RealSense D435 iniciado ({width}x{height}@{fps}fps)")
        except Exception as e:
            raise RuntimeError(f"No se pudo iniciar RealSense: {e}")

        # Alinear depth a color
        self._align = rs.align(rs.stream.color)

        # Hilo de captura
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Hilo de captura continua."""
        while not self._stop:
            try:
                # Esperar frames
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)

                # Alinear depth a color
                aligned_frames = self._align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if color_frame and depth_frame:
                    # Convertir a numpy
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # Normalizar depth a 0-255 para visualización
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_normalized = depth_normalized.astype(np.uint8)

                    with self._lock:
                        self._current_frame = color_image
                        self._current_depth = depth_normalized
                        self._current_depth_raw = depth_image  # uint16 en mm
                        self._frame_count += 1

            except Exception as e:
                if not self._stop:
                    print(f"[OBSERVER] RealSense error: {e}")
                    time.sleep(0.1)

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """
        Obtiene frame RGB o depth.

        Args:
            camera: "rgb" para color, "depth" para profundidad
        """
        with self._lock:
            if camera == "depth":
                return self._current_depth.copy() if self._current_depth is not None else None
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def get_depth(self) -> Optional[np.ndarray]:
        """Obtiene el mapa de profundidad raw en mm (uint16)."""
        with self._lock:
            if self._current_depth_raw is not None:
                return self._current_depth_raw.copy()
        return None

    def get_depth_visual(self) -> Optional[np.ndarray]:
        """Obtiene el mapa de profundidad normalizado para visualización (uint8)."""
        with self._lock:
            if self._current_depth is not None:
                return self._current_depth.copy()
        return None

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "source": "realsense",
            "frames": self._frame_count,
            "fps": self._frame_count / elapsed if elapsed > 0 else 0,
            "uptime": elapsed,
            "has_depth": True
        }

    def stop(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._pipeline.stop()
        print("[OBSERVER] RealSenseObserver detenido")


class AriaDemoObserver(BaseObserver):
    """
    Observer para gafas Meta Aria con eye tracking.

    Soporta conexión USB y WiFi, captura de RGB + SLAM + Eye + IMU.

    Uso:
        # USB (por defecto)
        observer = AriaDemoObserver()

        # WiFi
        observer = AriaDemoObserver(interface="wifi", ip_address="<ARIA_IP>")
    """

    fov_h = 1.919  # ~110° Aria RGB camera

    # Streaming profiles
    PROFILE_USB = "profile28"   # 30 FPS USB
    PROFILE_WIFI = "profile18"  # 30 FPS WiFi

    def __init__(
        self,
        interface: str = "usb",
        ip_address: Optional[str] = None,
        enable_slam: bool = True,
        auto_subscribe: bool = True
    ):
        """
        Inicializa conexión con Aria.

        Args:
            interface: "usb" o "wifi"
            ip_address: IP para WiFi (ej: "<ARIA_IP>")
            enable_slam: Si True, captura también cámaras SLAM laterales
        """
        try:
            import aria.sdk as aria
            from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData
            from projectaria_tools.core.calibration import device_calibration_from_json_string
        except ImportError:
            raise ImportError(
                "Aria SDK no instalado. Instala con:\n"
                "pip install projectaria-tools aria-glasses"
            )

        self._aria = aria
        self._lock = threading.Lock()
        self._stop = False
        self._enable_slam = enable_slam

        # Storage de frames
        self._frames = {
            "rgb": None,
            "eye": None,
            "slam1": None,
            "slam2": None
        }
        self._frame_counts = {k: 0 for k in self._frames}
        self._start_time = time.time()

        # IMU data
        from collections import deque
        self._imu_history = deque(maxlen=50)
        self._motion_state = "unknown"

        # Calibraciones (para 3D si se necesita)
        self._rgb_calib = None
        self._slam1_calib = None
        self._slam2_calib = None

        # === CONEXIÓN ===
        print(f"[OBSERVER] Conectando con Aria ({interface.upper()})...")
        self._device_client = aria.DeviceClient()

        # Configurar IP si es WiFi
        if interface.lower() == "wifi":
            if not ip_address:
                ip_address = "<ARIA_IP>"  # Default
            client_config = aria.DeviceClientConfig()
            client_config.ip_v4_address = ip_address
            self._device_client.set_client_config(client_config)
            print(f"[OBSERVER] WiFi target: {ip_address}")

        self._device = self._device_client.connect()
        print("[OBSERVER] ✓ Conectado")

        # === STREAMING ===
        self._streaming_manager = self._device.streaming_manager

        config = aria.StreamingConfig()
        if interface.lower() == "wifi":
            config.profile_name = self.PROFILE_WIFI
            config.streaming_interface = aria.StreamingInterface.WifiStation
        else:
            config.profile_name = self.PROFILE_USB
            config.streaming_interface = aria.StreamingInterface.Usb

        config.security_options.use_ephemeral_certs = True
        self._streaming_manager.streaming_config = config

        self._streaming_manager.start_streaming()
        print(f"[OBSERVER] ✓ Streaming iniciado ({config.profile_name})")

        # === CALIBRACIONES ===
        try:
            sensors_json = self._streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_json)
            self._rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
            self._slam1_calib = sensors_calib.get_camera_calib("camera-slam-left")
            self._slam2_calib = sensors_calib.get_camera_calib("camera-slam-right")
            print("[OBSERVER] ✓ Calibraciones obtenidas")
        except Exception as e:
            print(f"[OBSERVER WARN] No se pudieron obtener calibraciones: {e}")

        # === REGISTRAR OBSERVER ===
        self._streaming_client = self._streaming_manager.streaming_client
        self._streaming_client.set_streaming_client_observer(self)
        if auto_subscribe:
            self._streaming_client.subscribe()

        print("[OBSERVER] ✓ AriaDemoObserver listo")
        print(f"[OBSERVER] Cámaras: RGB + Eye" + (" + SLAM1 + SLAM2" if enable_slam else ""))

    def pause_streaming(self):
        """Pausa la suscripción DDS (deja de recibir frames)."""
        try:
            self._streaming_client.unsubscribe()
        except Exception as e:
            print(f"[OBSERVER WARN] Pause failed: {e}")

    def resume_streaming(self):
        """Reanuda la suscripción DDS."""
        try:
            self._streaming_client.subscribe()
        except Exception as e:
            print(f"[OBSERVER WARN] Resume failed: {e}")

    def on_image_received(self, image: np.ndarray, record) -> None:
        """Callback del SDK para nuevas imágenes."""
        camera_id = record.camera_id

        if camera_id == self._aria.CameraId.Rgb:
            processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)  # Aria gives RGB, OpenCV needs BGR
            key = "rgb"
        elif camera_id == self._aria.CameraId.EyeTrack:
            processed = np.rot90(image, 2)  # 180 grados
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            key = "eye"
        elif camera_id == self._aria.CameraId.Slam1 and self._enable_slam:
            processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            key = "slam1"
        elif camera_id == self._aria.CameraId.Slam2 and self._enable_slam:
            processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            key = "slam2"
        else:
            return

        with self._lock:
            self._frames[key] = processed
            self._frame_counts[key] += 1

        # Log periódico
        if self._frame_counts[key] % 300 == 0:
            elapsed = time.time() - self._start_time
            fps = self._frame_counts[key] / elapsed if elapsed > 0 else 0
            print(f"[OBSERVER] {key.upper()}: {self._frame_counts[key]} frames ({fps:.1f} FPS)")

    def on_imu_received(self, samples, imu_idx: int) -> None:
        """Callback IMU para detección de movimiento."""
        if not samples or imu_idx != 0:
            return

        sample = samples[0]
        accel = sample.accel_msec2
        magnitude = (accel[0]**2 + accel[1]**2 + accel[2]**2)**0.5

        with self._lock:
            self._imu_history.append(magnitude)

            # Estimar estado de movimiento
            if len(self._imu_history) >= 10:
                std = np.std(list(self._imu_history)[-20:])
                if std < 0.3:
                    self._motion_state = "stationary"
                elif std > 0.6:
                    self._motion_state = "walking"
                # else: mantener estado anterior (histéresis)

    def on_streaming_client_failure(self, reason, message: str) -> None:
        """Callback de error del SDK."""
        print(f"[OBSERVER ERROR] Streaming failure: {reason} - {message}")

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """Obtiene frame de una cámara específica."""
        with self._lock:
            frame = self._frames.get(camera)
            return frame.copy() if frame is not None else None

    def get_motion_state(self) -> str:
        """Obtiene estado de movimiento estimado del IMU."""
        with self._lock:
            return self._motion_state

    def get_calibrations(self) -> tuple:
        """Obtiene calibraciones de cámaras (rgb, slam1, slam2)."""
        return self._rgb_calib, self._slam1_calib, self._slam2_calib

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        with self._lock:
            return {
                "source": "aria",
                "frames": dict(self._frame_counts),
                "fps": {k: v / elapsed for k, v in self._frame_counts.items()},
                "motion_state": self._motion_state,
                "uptime": elapsed
            }

    def stop(self):
        """Desconexión limpia de Aria."""
        self._stop = True
        try:
            if self._streaming_client:
                self._streaming_client.unsubscribe()
                print("[OBSERVER] ✓ Unsubscribed")
        except Exception as e:
            print(f"[OBSERVER WARN] Unsubscribe error: {e}")

        try:
            if self._streaming_manager:
                self._streaming_manager.stop_streaming()
                print("[OBSERVER] ✓ Streaming stopped")
        except Exception as e:
            print(f"[OBSERVER WARN] Stop streaming error: {e}")

        try:
            if self._device_client and self._device:
                self._device_client.disconnect(self._device)
                print("[OBSERVER] ✓ Disconnected")
        except Exception as e:
            print(f"[OBSERVER WARN] Disconnect error: {e}")

        print("[OBSERVER] AriaDemoObserver detenido")


class AriaDatasetObserver(BaseObserver):
    """Observer para datasets pregrabados de Aria (VRS + eye gaze CSV)."""

    fov_h = 1.919  # ~110° Aria RGB camera

    def __init__(self, vrs_path: str, eyegaze_csv: str = None, loop: bool = True, target_fps: float = 10.0):
        """
        Args:
            vrs_path: Ruta al archivo .vrs
            eyegaze_csv: Ruta al CSV con datos de eye gaze (opcional)
            loop: Si True, reinicia al terminar el video
            target_fps: FPS objetivo para playback
        """
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.stream_id import StreamId
        except ImportError:
            raise ImportError("projectaria_tools no instalado. pip install projectaria-tools")

        self._lock = threading.Lock()
        self._stop = False
        self._loop = loop
        self._target_fps = target_fps

        # Cargar VRS
        print(f"[OBSERVER] Cargando VRS: {vrs_path}")
        self._provider = data_provider.create_vrs_data_provider(vrs_path)

        # Stream IDs
        self._rgb_stream = StreamId('214-1')  # camera-rgb
        self._et_stream = StreamId('211-1')   # camera-et (eye tracking)

        self._num_rgb_frames = self._provider.get_num_data(self._rgb_stream)
        self._num_et_frames = self._provider.get_num_data(self._et_stream)
        print(f"[OBSERVER] RGB frames: {self._num_rgb_frames}, ET frames: {self._num_et_frames}")

        # Build timestamp index for synchronization
        self._rgb_timestamps = []
        for i in range(self._num_rgb_frames):
            ts = self._provider.get_image_data_by_index(self._rgb_stream, i)[1].capture_timestamp_ns
            self._rgb_timestamps.append(ts)

        self._et_timestamps = []
        for i in range(self._num_et_frames):
            ts = self._provider.get_image_data_by_index(self._et_stream, i)[1].capture_timestamp_ns
            self._et_timestamps.append(ts)

        # Cargar eye gaze CSV si existe
        self._gaze_data = None
        self._gaze_timestamps = None
        if eyegaze_csv:
            self._load_gaze_csv(eyegaze_csv)

        # Estado actual
        self._current_rgb_idx = 0
        self._current_frame = None
        self._current_et_frame = None
        self._current_gaze = None  # (yaw, pitch, depth) pre-computed
        self._frame_count = 0
        self._start_time = time.time()

        # Hilo de playback
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

        print(f"[OBSERVER] AriaDatasetObserver iniciado ({target_fps} FPS)")

    def _load_gaze_csv(self, csv_path: str):
        """Carga datos de eye gaze desde CSV."""
        import pandas as pd
        print(f"[OBSERVER] Cargando gaze CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Convertir timestamps de us a ns
        self._gaze_timestamps = (df['tracking_timestamp_us'].values * 1000).astype(np.int64)

        # Extraer yaw (promedio left/right) y pitch
        left_yaw = df['left_yaw_rads_cpf'].values
        right_yaw = df['right_yaw_rads_cpf'].values
        pitch = df['pitch_rads_cpf'].values
        depth = df['depth_m'].values

        # Promedio de yaw izq/der
        yaw = (left_yaw + right_yaw) / 2

        self._gaze_data = np.stack([yaw, pitch, depth], axis=1)
        print(f"[OBSERVER] Gaze samples: {len(self._gaze_data)}")

    def _find_nearest_idx(self, timestamps: list, target_ts: int) -> int:
        """Encuentra el índice más cercano al timestamp objetivo."""
        # Binary search simple
        import bisect
        idx = bisect.bisect_left(timestamps, target_ts)
        if idx == 0:
            return 0
        if idx == len(timestamps):
            return len(timestamps) - 1
        # Comparar con vecinos
        if abs(timestamps[idx] - target_ts) < abs(timestamps[idx-1] - target_ts):
            return idx
        return idx - 1

    def _playback_loop(self):
        """Hilo de playback que itera por los frames."""
        frame_interval = 1.0 / self._target_fps
        last_time = time.time()

        while not self._stop:
            # Controlar timing
            now = time.time()
            elapsed = now - last_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_time = time.time()

            with self._lock:
                if self._current_rgb_idx >= self._num_rgb_frames:
                    if self._loop:
                        self._current_rgb_idx = 0
                    else:
                        continue

                # Obtener RGB frame
                rgb_data = self._provider.get_image_data_by_index(
                    self._rgb_stream, self._current_rgb_idx
                )
                rgb_frame = rgb_data[0].to_numpy_array()
                rgb_ts = rgb_data[1].capture_timestamp_ns

                # Convertir a BGR y rotar (Aria dataset: COUNTERCLOCKWISE)
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                self._current_frame = cv2.rotate(bgr_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Obtener ET frame sincronizado
                et_idx = self._find_nearest_idx(self._et_timestamps, rgb_ts)
                et_data = self._provider.get_image_data_by_index(self._et_stream, et_idx)
                et_frame = et_data[0].to_numpy_array()
                # ET es grayscale, convertir a BGR
                if len(et_frame.shape) == 2:
                    self._current_et_frame = cv2.cvtColor(et_frame, cv2.COLOR_GRAY2BGR)
                else:
                    self._current_et_frame = et_frame

                # Obtener gaze pre-computado sincronizado
                if self._gaze_data is not None:
                    gaze_idx = self._find_nearest_idx(list(self._gaze_timestamps), rgb_ts)
                    self._current_gaze = self._gaze_data[gaze_idx]

                self._current_rgb_idx += 1
                self._frame_count += 1

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """Obtiene frame de RGB o Eye Tracking."""
        with self._lock:
            if camera == "rgb" and self._current_frame is not None:
                return self._current_frame.copy()
            elif camera == "eye" and self._current_et_frame is not None:
                return self._current_et_frame.copy()
        return None

    def get_precomputed_gaze(self) -> Optional[tuple]:
        """
        Obtiene el gaze pre-computado del CSV (yaw, pitch, depth).
        Returns None si no hay CSV cargado.
        """
        with self._lock:
            if self._current_gaze is not None:
                return tuple(self._current_gaze)
        return None

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        with self._lock:
            return {
                "source": "aria_dataset",
                "frames": self._frame_count,
                "total_frames": self._num_rgb_frames,
                "progress": self._current_rgb_idx / self._num_rgb_frames,
                "fps": self._frame_count / elapsed if elapsed > 0 else 0,
                "has_gaze_csv": self._gaze_data is not None,
                "uptime": elapsed
            }

    def stop(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[OBSERVER] AriaDatasetObserver detenido")
