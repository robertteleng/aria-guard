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

    def __init__(self, source: str = "webcam", video_path: str = None):
        """
        Args:
            source: "webcam" o "video"
            video_path: Ruta al video si source="video"
        """
        self.source = source
        self._stop = False
        self._lock = threading.Lock()
        self._current_frame = None
        self._frame_count = 0
        self._start_time = time.time()

        # Abrir captura
        if source == "webcam":
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                # Intentar con índice 1
                self._cap = cv2.VideoCapture(1)
        else:
            self._cap = cv2.VideoCapture(video_path)

        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir {source}: {video_path}")

        # Configurar resolución para webcam
        if source == "webcam":
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._target_fps = 30
        else:
            # Respetar FPS original del video
            self._target_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"[OBSERVER] Video FPS: {self._target_fps:.1f}")

        self._frame_interval = 1.0 / self._target_fps

        # Hilo de captura
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(f"[OBSERVER] MockObserver iniciado ({source})")

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
        self._cap.release()
        print("[OBSERVER] MockObserver detenido")


class AriaDemoObserver(BaseObserver):
    """Observer para gafas Meta Aria con eye tracking."""

    def __init__(self):
        """Inicializa conexión con Aria."""
        try:
            import aria.sdk as aria
            from projectaria_tools.core.sensor_data import ImageDataRecord
        except ImportError:
            raise ImportError(
                "Aria SDK no instalado. Instala con:\n"
                "pip install projectaria-tools aria-glasses"
            )

        self._aria = aria
        self._lock = threading.Lock()
        self._stop = False

        # Storage de frames
        self._frames = {
            "rgb": None,
            "eye": None,
            "slam1": None,
            "slam2": None
        }
        self._frame_counts = {k: 0 for k in self._frames}
        self._start_time = time.time()

        # Conectar con Aria
        print("[OBSERVER] Conectando con Aria...")
        self._device_client = aria.DeviceClient()
        self._device = self._device_client.connect()

        # Configurar streaming
        streaming_manager = self._device.streaming_manager
        streaming_client = streaming_manager.streaming_client

        config = aria.StreamingConfig()
        config.profile_name = "profile18"  # RGB + SLAM + Eye
        config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = config

        # Registrar callbacks
        streaming_client.set_streaming_client_observer(self)

        # Iniciar streaming
        streaming_manager.start_streaming()
        streaming_client.subscribe()

        self._streaming_manager = streaming_manager
        self._streaming_client = streaming_client

        print("[OBSERVER] AriaDemoObserver conectado")

    def on_image_received(self, image: np.ndarray, record) -> None:
        """Callback del SDK para nuevas imágenes."""
        camera_id = record.camera_id

        if camera_id == self._aria.CameraId.Rgb:
            processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            key = "rgb"
        elif camera_id == self._aria.CameraId.EyeTrack:
            processed = np.rot90(image, 2)  # 180 grados
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            key = "eye"
        elif camera_id == self._aria.CameraId.Slam1:
            processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            key = "slam1"
        elif camera_id == self._aria.CameraId.Slam2:
            processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            key = "slam2"
        else:
            return

        with self._lock:
            self._frames[key] = processed
            self._frame_counts[key] += 1

    def on_imu_received(self, samples, imu_idx: int) -> None:
        """Callback IMU (no usado en demo simple)."""
        pass

    def on_streaming_client_failure(self, reason, message: str) -> None:
        """Callback de error."""
        print(f"[OBSERVER ERROR] {reason}: {message}")

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """Obtiene frame de una cámara específica."""
        with self._lock:
            frame = self._frames.get(camera)
            return frame.copy() if frame is not None else None

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        with self._lock:
            return {
                "source": "aria",
                "frames": dict(self._frame_counts),
                "fps": {k: v / elapsed for k, v in self._frame_counts.items()},
                "uptime": elapsed
            }

    def stop(self):
        self._stop = True
        try:
            self._streaming_client.unsubscribe()
            self._streaming_manager.stop_streaming()
            self._device_client.disconnect(self._device)
        except Exception as e:
            print(f"[OBSERVER] Error al desconectar: {e}")
        print("[OBSERVER] AriaDemoObserver detenido")


class AriaDatasetObserver(BaseObserver):
    """Observer para datasets pregrabados de Aria (VRS + eye gaze CSV)."""

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
