"""
Frame and IMU transforms shared by the live Aria observer and VRS playback.

Live streaming and recorded VRS files carry the same raw sensor images, so
both paths must orient them identically. Keeping the transforms here means a
replayed recording reaches the detector exactly as a live frame would.
"""

import bisect
from collections import deque
from typing import Optional, Sequence

import cv2
import numpy as np


def rgb_to_bgr_upright(image: np.ndarray) -> np.ndarray:
    """Raw Aria RGB (sensor orientation, RGB order) -> upright, contiguous BGR."""
    rotated = np.rot90(image, -1).copy()  # 90 CW
    return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)


def eye_to_bgr(image: np.ndarray) -> np.ndarray:
    """Raw Aria eye-tracking image -> the orientation the gaze model expects, BGR."""
    rotated = np.rot90(image, 2).copy()
    if rotated.ndim == 2:
        rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    return rotated


def upright_to_raw_pixel(x: float, y: float, size: int):
    """Pixel in the upright frame (np.rot90(raw, -1)) -> pixel in the raw sensor image.

    For a square image of side N, upright[r, c] = raw[N-1-c, r]; with x = column
    and y = row: raw_x = y, raw_y = N - 1 - x.
    """
    return y, size - 1 - x


def raw_to_upright_pixel(x: float, y: float, size: int):
    """Inverse of upright_to_raw_pixel."""
    return size - 1 - y, x


def nearest_index(timestamps: Sequence[int], target: int) -> int:
    """Index of the timestamp closest to target; timestamps must be sorted."""
    if len(timestamps) == 0:
        raise ValueError("timestamps is empty")
    idx = bisect.bisect_left(timestamps, target)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - target) < abs(timestamps[idx - 1] - target):
        return idx
    return idx - 1


class MotionClassifier:
    """Walking / stationary from the spread of accelerometer magnitude.

    Hysteresis: between the two thresholds the previous state is kept.
    """

    def __init__(self, window: int = 20, min_samples: int = 10,
                 stationary_std: float = 0.3, walking_std: float = 0.6):
        self._history = deque(maxlen=50)
        self._window = window
        self._min_samples = min_samples
        self._stationary_std = stationary_std
        self._walking_std = walking_std
        self.state = "unknown"

    def update(self, accel_msec2: Sequence[float]) -> str:
        magnitude = float(np.linalg.norm(accel_msec2))
        self._history.append(magnitude)
        if len(self._history) >= self._min_samples:
            std = float(np.std(list(self._history)[-self._window:]))
            if std < self._stationary_std:
                self.state = "stationary"
            elif std > self._walking_std:
                self.state = "walking"
        return self.state


class GravityEstimator:
    """Up direction from the accelerometer, averaged over a time window.

    At rest the accelerometer measures +g upward (specific force), so the mean
    over ~1 s of walking points up; acceleration from steps averages out.
    """

    def __init__(self, window_s: float = 1.0):
        self._window_ns = int(window_s * 1e9)
        self._samples = deque()
        self._sum = np.zeros(3)

    def update(self, t_ns: int, accel: Sequence[float]) -> None:
        a = np.asarray(accel, dtype=np.float64)
        self._samples.append((t_ns, a))
        self._sum += a
        while self._samples and self._samples[0][0] < t_ns - self._window_ns:
            self._sum -= self._samples.popleft()[1]

    def up(self) -> Optional[np.ndarray]:
        """Unit up vector in the accelerometer frame, or None before any sample."""
        n = np.linalg.norm(self._sum)
        if not self._samples or n < 1e-6:
            return None
        return self._sum / n
