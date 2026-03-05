"""
Observer base + MockObserver para webcam/video.
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
        """Obtiene el frame mas reciente."""
        pass

    @abstractmethod
    def stop(self):
        """Detiene el observer."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Estadisticas del observer."""
        return {}


class MockObserver(BaseObserver):
    """Observer para webcam o video (desarrollo sin Aria)."""

    fov_h = 1.15  # ~66° typical webcam

    def __init__(self, source: str = "webcam", video_path: str = None, use_nvdec: bool = True):
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
                if hasattr(cv2, 'cudacodec'):
                    self._gpu_reader = cv2.cudacodec.createVideoReader(video_path)
                    self._use_nvdec = True
                    format_info = self._gpu_reader.format()
                    self._target_fps = format_info.fps if hasattr(format_info, 'fps') else 30
                    print(f"[OBSERVER] NVDEC habilitado - Video FPS: {self._target_fps:.1f}")
            except Exception as e:
                msg = str(e)
                if "(-213" in msg or "disabled for current build or platform" in msg:
                    print(
                        "[OBSERVER] NVDEC no disponible en runtime. "
                        "Verifica NVIDIA_DRIVER_CAPABILITIES=compute,utility,video "
                        "y que libnvcuvid del host este visible en el contenedor."
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

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        decode_mode = "NVDEC GPU" if self._use_nvdec else "CPU"
        print(f"[OBSERVER] MockObserver iniciado ({source}, {decode_mode})")

    def _capture_loop(self):
        """Hilo de captura continua con timing controlado."""
        last_frame_time = time.time()

        while not self._stop:
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
            last_frame_time = time.time()

            if self._use_nvdec and self._gpu_reader:
                ret, gpu_frame = self._gpu_reader.nextFrame()
                if ret:
                    frame = gpu_frame.download()
                    with self._lock:
                        self._current_frame = frame
                        self._frame_count += 1
                else:
                    if self.source != "webcam":
                        try:
                            if self._video_path:
                                self._gpu_reader = cv2.cudacodec.createVideoReader(self._video_path)
                        except Exception:
                            pass
            else:
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._current_frame = frame
                        self._frame_count += 1
                else:
                    if self.source == "video":
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        if camera == "eye":
            return None
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
