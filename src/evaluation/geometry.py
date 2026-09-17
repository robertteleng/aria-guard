"""
Pure geometry for the alert evaluation (docs/ALERT_EVALUATION.md).

World frame: MPS, metric and gravity-aligned (z up). No projectaria_tools here,
so everything is unit-testable; calibration is passed in as callables/matrices.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def upright_to_raw_pixel(x: float, y: float, size: int) -> Tuple[float, float]:
    """Pixel in the upright frame (np.rot90(raw, -1)) -> pixel in the raw sensor image.

    For a square image of side N, upright[r, c] = raw[N-1-c, r]; with x = column
    and y = row: raw_x = y, raw_y = N - 1 - x.
    """
    return y, size - 1 - x


def raw_to_upright_pixel(x: float, y: float, size: int) -> Tuple[float, float]:
    """Inverse of upright_to_raw_pixel."""
    return size - 1 - y, x


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def ray_ground_intersection(origin: np.ndarray, direction: np.ndarray, ground_z: float,
                            max_horizontal_m: float) -> Optional[np.ndarray]:
    """Point where the ray hits z = ground_z in front of the origin, within range; else None."""
    dz = direction[2]
    if dz >= -1e-9:  # parallel or pointing up: never reaches the ground ahead
        return None
    s = (ground_z - origin[2]) / dz
    if s <= 0:
        return None
    hit = origin + s * direction
    if np.hypot(*(hit[:2] - origin[:2])) > max_horizontal_m:
        return None
    return hit


def point_to_polyline_2d(point: np.ndarray, path: np.ndarray) -> Tuple[float, int, float]:
    """Horizontal distance from point to a polyline (K x 2).

    Returns (distance, index of the segment start, fraction along that segment),
    so the caller can interpolate the time at which the path passes closest.
    """
    p = np.asarray(point, dtype=np.float64)[:2]
    path = np.asarray(path, dtype=np.float64)[:, :2]
    if len(path) == 1:
        return float(np.linalg.norm(p - path[0])), 0, 0.0
    a, b = path[:-1], path[1:]
    ab = b - a
    denom = (ab * ab).sum(1)
    t = np.where(denom > 0, ((p - a) * ab).sum(1) / np.where(denom > 0, denom, 1), 0.0).clip(0, 1)
    closest = a + t[:, None] * ab
    d = np.linalg.norm(closest - p, axis=1)
    i = int(d.argmin())
    return float(d[i]), i, float(t[i])


def bearing_deg(from_xy: np.ndarray, heading_xy: np.ndarray, to_xy: np.ndarray) -> float:
    """Absolute angle (deg) between the heading and the direction to a point."""
    h = np.asarray(heading_xy, dtype=np.float64)
    v = np.asarray(to_xy, dtype=np.float64)[:2] - np.asarray(from_xy, dtype=np.float64)[:2]
    nh, nv = np.linalg.norm(h), np.linalg.norm(v)
    if nh < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(h, v) / (nh * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


@dataclass
class Episode:
    track_id: int
    start: float
    end: float
    closest_time: float
    closest_distance: float


def merge_episodes(samples: Sequence[Tuple[int, float, bool, float, float]],
                   max_gap_s: float) -> List[Episode]:
    """Group in-path samples into episodes per track.

    samples: (track_id, time, in_path, distance_to_path, closest_approach_time).
    Consecutive in-path samples of a track separated by < max_gap_s merge.
    """
    by_track = {}
    for tid, t, in_path, dist, t_closest in sorted(samples, key=lambda s: (s[0], s[1])):
        if not in_path:
            continue
        by_track.setdefault(tid, []).append((t, dist, t_closest))
    episodes = []
    for tid, rows in by_track.items():
        cur = None
        for t, dist, t_closest in rows:
            if cur is not None and t - cur.end < max_gap_s:
                cur.end = t
                if dist < cur.closest_distance:
                    cur.closest_distance, cur.closest_time = dist, t_closest
            else:
                if cur is not None:
                    episodes.append(cur)
                cur = Episode(tid, t, t, t_closest, dist)
        if cur is not None:
            episodes.append(cur)
    return sorted(episodes, key=lambda e: (e.start, e.track_id))
