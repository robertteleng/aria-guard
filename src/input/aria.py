"""
Observers para Meta Aria glasses: live streaming y dataset playback.
"""

import os
import threading
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .aria_frames import GravityEstimator, MotionClassifier, eye_to_bgr, nearest_index, rgb_to_bgr_upright
from .observer import BaseObserver


def resolve_wifi_ip(ip_address: Optional[str] = None) -> str:
    """IP of the glasses for WiFi streaming: the argument, else $ARIA_IP.

    There is no hard-coded default: the address depends on the local network.
    """
    ip = (ip_address or os.environ.get("ARIA_IP", "")).strip()
    if not ip:
        raise ValueError("WiFi streaming needs the glasses' IP: pass ip_address or set ARIA_IP")
    return ip


class AriaDemoObserver(BaseObserver):
    """
    Observer para gafas Meta Aria con eye tracking.

    Soporta conexion USB y WiFi, captura de RGB + SLAM + Eye + IMU.
    """

    fov_h = 1.919  # ~110 Aria RGB camera

    # Streaming profiles
    PROFILE_USB = "profile28"   # 30 FPS USB
    PROFILE_WIFI = "profile15"  # 30 FPS WiFi

    def __init__(
        self,
        interface: str = "usb",
        ip_address: Optional[str] = None,
        enable_slam: bool = True,
        auto_subscribe: bool = True
    ):
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
        self._motion = MotionClassifier()
        self._motion_state = "unknown"
        self._gravity = GravityEstimator(window_s=1.0)
        self._metric_inpath = None

        # Calibraciones
        self._rgb_calib = None
        self._slam1_calib = None
        self._slam2_calib = None

        # === CONEXION ===
        print(f"[OBSERVER] Conectando con Aria ({interface.upper()})...")
        self._device_client = aria.DeviceClient()

        if interface.lower() == "wifi":
            ip_address = resolve_wifi_ip(ip_address)
            client_config = aria.DeviceClientConfig()
            client_config.ip_v4_address = ip_address
            self._device_client.set_client_config(client_config)
            print(f"[OBSERVER] WiFi target: {ip_address}")

        self._device = self._device_client.connect()
        print("[OBSERVER] Conectado")

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
        print(f"[OBSERVER] Streaming iniciado ({config.profile_name})")

        # === CALIBRACIONES ===
        try:
            sensors_json = self._streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_json)
            self._rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
            self._slam1_calib = sensors_calib.get_camera_calib("camera-slam-left")
            self._slam2_calib = sensors_calib.get_camera_calib("camera-slam-right")
            print("[OBSERVER] Calibraciones obtenidas")
            try:
                from .metric_inpath import MetricInPath
                R_di = sensors_calib.get_imu_calib("imu-right").get_transform_device_imu().to_matrix()[:3, :3]
                self._metric_inpath = MetricInPath(self._rgb_calib, R_di)
            except Exception as e:
                print(f"[OBSERVER WARN] Metric in-path unavailable, heuristic threat model: {e}")
        except Exception as e:
            print(f"[OBSERVER WARN] No se pudieron obtener calibraciones: {e}")

        # === REGISTRAR OBSERVER ===
        self._streaming_client = self._streaming_manager.streaming_client

        # Configure subscription to only receive needed streams
        # CRITICAL: subscribing to SLAM/Audio causes DDS buffer overflow -> native segfault
        config = aria.StreamingSubscriptionConfig()
        if enable_slam:
            config.subscriber_data_type = (
                aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack |
                aria.StreamingDataType.Slam | aria.StreamingDataType.Imu
            )
        else:
            config.subscriber_data_type = (
                aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack |
                aria.StreamingDataType.Imu
            )
        config.security_options.use_ephemeral_certs = True
        self._streaming_client.subscription_config = config

        self._streaming_client.set_streaming_client_observer(self)
        if auto_subscribe:
            self._streaming_client.subscribe()

        streams = "RGB + Eye + IMU" + (" + SLAM" if enable_slam else "")
        print(f"[OBSERVER] AriaDemoObserver listo")
        print(f"[OBSERVER] Suscrito a: {streams}")

    def pause_streaming(self):
        """Pausa la suscripcion DDS (deja de recibir frames)."""
        try:
            self._streaming_client.unsubscribe()
        except Exception as e:
            print(f"[OBSERVER WARN] Pause failed: {e}")

    def resume_streaming(self):
        """Reanuda la suscripcion DDS."""
        try:
            self._streaming_client.subscribe()
        except Exception as e:
            print(f"[OBSERVER WARN] Resume failed: {e}")

    def on_image_received(self, image: np.ndarray, record) -> None:
        """Callback del SDK para nuevas imagenes."""
        camera_id = record.camera_id

        # Filter camera_id BEFORE copying — avoid unnecessary work in DDS callback
        if camera_id == self._aria.CameraId.Rgb:
            key = "rgb"
        elif camera_id == self._aria.CameraId.EyeTrack:
            key = "eye"
        elif camera_id == self._aria.CameraId.Slam1 and self._enable_slam:
            key = "slam1"
        elif camera_id == self._aria.CameraId.Slam2 and self._enable_slam:
            key = "slam2"
        else:
            return

        # Copy DDS buffer AFTER filtering — DDS can free it at any time
        try:
            image = image.copy()
        except Exception:
            return

        # Minimal transform — no cv2 heavy ops in DDS callback
        if key == "rgb":
            processed = rgb_to_bgr_upright(image)
        elif key == "eye":
            processed = eye_to_bgr(image)
        else:  # slam1, slam2
            processed = np.rot90(image, -1).copy()
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        with self._lock:
            self._frames[key] = processed
            self._frame_counts[key] += 1

        # Log periodico
        if self._frame_counts[key] % 300 == 0:
            elapsed = time.time() - self._start_time
            fps = self._frame_counts[key] / elapsed if elapsed > 0 else 0
            print(f"[OBSERVER] {key.upper()}: {self._frame_counts[key]} frames ({fps:.1f} FPS)")

    def on_imu_received(self, samples, imu_idx: int) -> None:
        """Callback IMU para deteccion de movimiento."""
        if not samples or imu_idx != 0:
            return

        sample = samples[0]
        with self._lock:
            self._motion_state = self._motion.update(sample.accel_msec2)
            self._gravity.update(sample.capture_timestamp_ns, sample.accel_msec2)

    def on_streaming_client_failure(self, reason, message: str) -> None:
        """Callback de error del SDK."""
        print(f"[OBSERVER ERROR] Streaming failure: {reason} - {message}")

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """Obtiene frame de una camara especifica."""
        with self._lock:
            frame = self._frames.get(camera)
            return frame.copy() if frame is not None else None

    def metric_inpath_inputs(self):
        """(MetricInPath, up vector in the IMU frame) when calibration and IMU are available."""
        with self._lock:
            up = self._gravity.up()
        if self._metric_inpath is None or up is None:
            return None
        return self._metric_inpath, up

    def get_motion_state(self) -> str:
        """Obtiene estado de movimiento estimado del IMU."""
        with self._lock:
            return self._motion_state

    def get_calibrations(self) -> tuple:
        """Obtiene calibraciones de camaras (rgb, slam1, slam2)."""
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
        """Desconexion limpia de Aria."""
        self._stop = True
        try:
            if self._streaming_client:
                self._streaming_client.unsubscribe()
                print("[OBSERVER] Unsubscribed")
        except Exception as e:
            print(f"[OBSERVER WARN] Unsubscribe error: {e}")

        try:
            if self._streaming_manager:
                self._streaming_manager.stop_streaming()
                print("[OBSERVER] Streaming stopped")
        except Exception as e:
            print(f"[OBSERVER WARN] Stop streaming error: {e}")

        try:
            if self._device_client and self._device:
                self._device_client.disconnect(self._device)
                print("[OBSERVER] Disconnected")
        except Exception as e:
            print(f"[OBSERVER WARN] Disconnect error: {e}")

        print("[OBSERVER] AriaDemoObserver detenido")


class AriaDatasetObserver(BaseObserver):
    """Observer para datasets pregrabados de Aria (VRS + eye gaze CSV)."""

    fov_h = 1.919  # ~110 Aria RGB camera

    def __init__(self, vrs_path: str, eyegaze_csv: str = None, loop: bool = True, target_fps: float = 10.0):
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.stream_id import StreamId
        except ImportError:
            raise ImportError("projectaria_tools no instalado. pip install projectaria-tools")

        self._lock = threading.Lock()
        self._stop = False
        self._loop = loop
        self._target_fps = target_fps

        print(f"[OBSERVER] Cargando VRS: {vrs_path}")
        self._provider = data_provider.create_vrs_data_provider(vrs_path)

        self._rgb_stream = StreamId('214-1')
        self._et_stream = StreamId('211-1')

        self._num_rgb_frames = self._provider.get_num_data(self._rgb_stream)
        self._num_et_frames = self._provider.get_num_data(self._et_stream)
        print(f"[OBSERVER] RGB frames: {self._num_rgb_frames}, ET frames: {self._num_et_frames}")

        # Timestamp index for synchronization, read from the index without
        # decoding every image (decoding took minutes on long recordings)
        from projectaria_tools.core.sensor_data import TimeDomain
        self._rgb_timestamps = list(self._provider.get_timestamps_ns(self._rgb_stream, TimeDomain.DEVICE_TIME))
        self._et_timestamps = list(self._provider.get_timestamps_ns(self._et_stream, TimeDomain.DEVICE_TIME))

        # Cargar eye gaze CSV si existe
        self._gaze_data = None
        self._gaze_timestamps = None
        if eyegaze_csv:
            self._load_gaze_csv(eyegaze_csv)

        # Estado actual
        self._current_rgb_idx = 0
        self._current_frame = None
        self._current_et_frame = None
        self._current_gaze = None
        self._frame_count = 0
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

        print(f"[OBSERVER] AriaDatasetObserver iniciado ({target_fps} FPS)")

    def _load_gaze_csv(self, csv_path: str):
        """Carga datos de eye gaze desde CSV."""
        import pandas as pd
        print(f"[OBSERVER] Cargando gaze CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        self._gaze_timestamps = (df['tracking_timestamp_us'].values * 1000).astype(np.int64)

        left_yaw = df['left_yaw_rads_cpf'].values
        right_yaw = df['right_yaw_rads_cpf'].values
        pitch = df['pitch_rads_cpf'].values
        depth = df['depth_m'].values

        yaw = (left_yaw + right_yaw) / 2
        self._gaze_data = np.stack([yaw, pitch, depth], axis=1)
        print(f"[OBSERVER] Gaze samples: {len(self._gaze_data)}")

    def _playback_loop(self):
        """Hilo de playback que itera por los frames."""
        frame_interval = 1.0 / self._target_fps
        last_time = time.time()

        while not self._stop:
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

                rgb_data = self._provider.get_image_data_by_index(
                    self._rgb_stream, self._current_rgb_idx
                )
                rgb_ts = rgb_data[1].capture_timestamp_ns
                # Same orientation as live streaming (AriaDemoObserver)
                self._current_frame = rgb_to_bgr_upright(rgb_data[0].to_numpy_array())

                et_idx = nearest_index(self._et_timestamps, rgb_ts)
                et_data = self._provider.get_image_data_by_index(self._et_stream, et_idx)
                self._current_et_frame = eye_to_bgr(et_data[0].to_numpy_array())

                if self._gaze_data is not None:
                    gaze_idx = nearest_index(self._gaze_timestamps, rgb_ts)
                    self._current_gaze = self._gaze_data[gaze_idx]

                self._current_rgb_idx += 1
                self._frame_count += 1

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        with self._lock:
            if camera == "rgb" and self._current_frame is not None:
                return self._current_frame.copy()
            elif camera == "eye" and self._current_et_frame is not None:
                return self._current_et_frame.copy()
        return None

    def get_precomputed_gaze(self) -> Optional[tuple]:
        """Obtiene el gaze pre-computado del CSV (yaw, pitch, depth)."""
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
