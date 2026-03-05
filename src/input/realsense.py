"""
Observer para Intel RealSense D435 con depth nativo.
"""

import threading
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .observer import BaseObserver


class RealSenseObserver(BaseObserver):
    """
    Observer para Intel RealSense D435 con depth nativo.

    Ventaja: El D435 tiene sensor de profundidad hardware, no necesita
    modelo de depth estimation (Depth Anything).
    """

    fov_h = 1.518  # ~87 D435 RGB sensor

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self._stop = False
        self._lock = threading.Lock()
        self._current_frame = None
        self._current_depth = None
        self._current_depth_raw = None
        self._frame_count = 0
        self._start_time = time.time()

        try:
            import pyrealsense2 as rs
            self._rs = rs
        except ImportError:
            raise RuntimeError("pyrealsense2 no instalado. Instalar con: pip install pyrealsense2")

        self._pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        try:
            self._profile = self._pipeline.start(config)
            print(f"[OBSERVER] RealSense D435 iniciado ({width}x{height}@{fps}fps)")
        except Exception as e:
            raise RuntimeError(f"No se pudo iniciar RealSense: {e}")

        self._align = rs.align(rs.stream.color)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Hilo de captura continua."""
        while not self._stop:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self._align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_normalized = depth_normalized.astype(np.uint8)

                    with self._lock:
                        self._current_frame = color_image
                        self._current_depth = depth_normalized
                        self._current_depth_raw = depth_image
                        self._frame_count += 1

            except Exception as e:
                if not self._stop:
                    print(f"[OBSERVER] RealSense error: {e}")
                    time.sleep(0.1)

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
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
        """Obtiene el mapa de profundidad normalizado para visualizacion (uint8)."""
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
