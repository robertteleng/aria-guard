"""
Metric "is it in my path" from data available live on the glasses.

A detection's bbox bottom-centre ray (camera calibration), the gravity
direction (accelerometer) and a fixed eye height give where the object touches
the ground, as forward distance F and lateral offset L in metres relative to
the camera heading. See docs/ALERT_EVALUATION.md, candidate change 1.

Pure numpy, no torch, no projectaria_tools: the device-frame ray is computed
by the caller.
"""

from typing import Optional, Tuple

import numpy as np

EYE_HEIGHT_M = 1.6
CORRIDOR_M = 0.75
DANGER_M, WARNING_M, ATTENTION_M = 1.5, 3.0, 5.0
MAX_RANGE_M = 15.0

# collision_risk given to each metric level, so the arbiter's top-1 choice
# keeps ranking the most urgent object first
LEVEL_RISK = {"DANGER": 0.9, "WARNING": 0.5, "ATTENTION": 0.2, "NONE": 0.0}


def horizontal_basis(up: np.ndarray, camera_forward: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Unit forward and right vectors on the horizontal plane (same frame as inputs).

    Returns None when the camera looks straight up or down.
    """
    u = np.asarray(up, dtype=np.float64)
    u = u / np.linalg.norm(u)
    f = np.asarray(camera_forward, dtype=np.float64)
    f = f - np.dot(f, u) * u
    n = np.linalg.norm(f)
    if n < 1e-6:
        return None
    f = f / n
    right = np.cross(f, u)
    return f, right


def ground_contact(ray: np.ndarray, up: np.ndarray, forward: np.ndarray, right: np.ndarray,
                   eye_height: float = EYE_HEIGHT_M, max_range: float = MAX_RANGE_M
                   ) -> Optional[Tuple[float, float]]:
    """(forward, lateral) metres where the ray meets flat ground eye_height below; None if it does not."""
    r = np.asarray(ray, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    down = -np.dot(r, u)
    if down <= 1e-6:
        return None
    p = r * (eye_height / down)
    F, L = float(np.dot(p, forward)), float(np.dot(p, right))
    if F <= 0 or np.hypot(F, L) > max_range:
        return None
    return F, L


def metric_threat(contact: Optional[Tuple[float, float]], corridor: float = CORRIDOR_M) -> str:
    """Threat level from a ground contact (pre-registered thresholds)."""
    if contact is None:
        return "NONE"
    F, L = contact
    if abs(L) > corridor or F <= 0:
        return "NONE"
    if F <= DANGER_M:
        return "DANGER"
    if F <= WARNING_M:
        return "WARNING"
    if F <= ATTENTION_M:
        return "ATTENTION"
    return "NONE"
