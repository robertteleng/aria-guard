"""
Metric in-path threat for live Aria frames (candidate change 1, adopted).

Wraps the pure ground projection with the glasses' calibration: RGB camera
(unprojection and device pose) and the accelerometer frame. Used by the live
pipeline and by scripts/evaluate_alerts.py, so what was measured is what runs.
"""

from typing import Iterable, Optional

import numpy as np

from src.domain.ground_projection import ground_contact, horizontal_basis, metric_threat
from src.input.aria_frames import upright_to_raw_pixel


class MetricInPath:
    def __init__(self, rgb_calib, R_device_imu: np.ndarray, image_size: int = 1408):
        self._calib = rgb_calib
        T = rgb_calib.get_transform_device_camera().to_matrix()
        self._R_dc = T[:3, :3]
        self._R_di = np.asarray(R_device_imu, dtype=np.float64)
        self._size = image_size
        self._cam_forward = self._R_dc @ np.array([0.0, 0.0, 1.0])

    def ray_device(self, x: float, y: float) -> np.ndarray:
        """Unit ray in the device frame for an upright-frame pixel."""
        rx, ry = upright_to_raw_pixel(x, y, self._size)
        v = np.asarray(self._calib.unproject_no_checks(np.array([rx, ry], dtype=np.float64)), dtype=np.float64)
        return self._R_dc @ (v / np.linalg.norm(v))

    def contact(self, bbox, up_imu: Optional[np.ndarray]):
        """(forward_m, lateral_m) of the bbox bottom-centre on the ground, or None."""
        if up_imu is None:
            return None
        up = self._R_di @ np.asarray(up_imu, dtype=np.float64)
        basis = horizontal_basis(up, self._cam_forward)
        if basis is None:
            return None
        x, y, w, h = bbox
        return ground_contact(self.ray_device(x + w / 2, y + h), up, *basis)

    def annotate(self, detections: Iterable, up_imu: Optional[np.ndarray]) -> None:
        """Set metric_threat, forward_m and lateral_m on each detection in place."""
        for det in detections:
            c = self.contact(det.bbox, up_imu)
            det.metric_threat = metric_threat(c)
            det.forward_m, det.lateral_m = (c if c else (None, None))
