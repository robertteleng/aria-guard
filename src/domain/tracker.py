"""
Simple object tracker for ARIA Guard.

Tracks objects across frames to detect approach and prioritize alerts.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import math
import numpy as np


@dataclass
class TrackedObject:
    """Object being tracked across frames."""
    id: int
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    zone: str
    distance: str
    depth_value: float
    confidence: float
    is_gazed: bool = False
    frame_width: int = 1280
    fov_h: float = 1.15  # horizontal FOV in radians
    traffic_light_state: Optional[str] = None  # "red", "green", "yellow"

    # Tracking data
    depth_history: deque = field(default_factory=lambda: deque(maxlen=10))
    bearing_history: deque = field(default_factory=lambda: deque(maxlen=10))  # bearing in radians (0 = center, negative = left, positive = right)
    height_history: deque = field(default_factory=lambda: deque(maxlen=10))  # bbox height (px) — looming cue
    frames_seen: int = 1
    frames_missing: int = 0

    # Computed
    is_approaching: bool = False
    approach_speed: float = 0.0  # positive = approaching
    looming_speed: float = 0.0  # bbox-height approach, in depth-slope units
    lateral_speed: float = 0.0  # rad/frame, positive = moving right
    enters_path: bool = False  # object moving laterally into user's path (center)
    bearing: float = 0.0  # current bearing in radians
    # H18: collision risk score (0.0–1.0) and threat level
    collision_risk: float = 0.0
    threat_level: str = "NONE"  # NONE, ATTENTION, WARNING, DANGER

    def __post_init__(self):
        if not self.depth_history:
            self.depth_history = deque(maxlen=10)
        if not self.bearing_history:
            self.bearing_history = deque(maxlen=10)
        if not self.height_history:
            self.height_history = deque(maxlen=10)
        self.depth_history.append(self.depth_value)
        self.bearing = self._pixel_to_bearing(self.bbox)
        self.bearing_history.append(self.bearing)
        self.height_history.append(self.bbox[3])

    def _pixel_to_bearing(self, bbox) -> float:
        """Convert bbox center_x to bearing angle in radians using FOV."""
        x, y, w, h = bbox
        center_x = (x + w / 2) / self.frame_width if self.frame_width > 0 else 0.5
        # Map pixel position to angle: atan((normalized - 0.5) * 2 * tan(fov/2))
        return math.atan((center_x - 0.5) * 2 * math.tan(self.fov_h / 2))


# H18: Collision risk weights by object class (0.0–1.0)
# Vehicles are lethal, people are unpredictable, static objects are low risk
CLASS_RISK = {
    "car": 1.0, "truck": 1.0, "bus": 1.0,
    "motorcycle": 0.9, "bicycle": 0.7,
    "person": 0.15, "dog": 0.15, "cat": 0.1,
    "chair": 0.15, "couch": 0.15, "bed": 0.1,
    "dining table": 0.1, "toilet": 0.1,
    "backpack": 0.05, "handbag": 0.05, "suitcase": 0.05,
}

# H18: Zone risk weights for collision_risk (center = user's path)
ZONE_RISK = {
    "center": 1.0,
    "left": 0.4,
    "right": 0.4,
}

# H18: Static proximity risk (objects not approaching but close)
STATIC_PROXIMITY = {
    "very_close": 0.8,
    "close": 0.4,
    "medium": 0.1,
    "far": 0.0,
    "unknown": 0.1,
}

# H18: Threat level thresholds (calibrated from ADAS literature)
THREAT_THRESHOLDS = {
    "DANGER": 0.7,     # ~TTC < 1s — fast vehicle or imminent collision
    "WARNING": 0.35,   # ~TTC < 3s (Mobileye FCW)
    "ATTENTION": 0.15, # ~TTC < 5s
}

# Ego-motion compensation: while the user WALKS, everything ahead appears to
# "approach" (depth grows) even when static — the dominant source of false
# DANGER alerts ("alertas mediocres"). We subtract a fixed walking bias from
# the apparent approach slope so a static object ahead nets to ~0 approach,
# while a genuinely fast approaching object (closing faster than the walk)
# keeps most of its signal. Subtraction (not a multiplicative factor) is
# deliberate: a blind pedestrian is almost always walking, so scaling the
# approach down would gut REAL threats in the common case. This is the v1
# proxy for subtracting an IMU-estimated forward velocity.
EGO_MOTION_WALKING_BIAS = 0.05

# depth_value is PROXIMITY (0 = far, 1 = close). Time-to-collision divides the
# gap still to close, 1 - proximity, by the approach speed. The floor keeps an
# object already at the camera from dividing zero by its speed.
MIN_REMAINING_GAP = 0.05


def remaining_gap(proximity: float) -> float:
    """Normalized distance left to close for a proximity value in [0, 1]."""
    return max(1.0 - proximity, MIN_REMAINING_GAP)


def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to x1, y1, x2, y2
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class SimpleTracker:
    """Tracks objects across frames using IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 5):
        """
        Args:
            iou_threshold: Minimum IoU to consider a match
            max_missing: Frames before removing a track
        """
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 0

    def update(self, detections: List, frame_width: int = 1280, fov_h: float = 1.15,
               motion_state: str = "unknown") -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects from detector
            frame_width: Width of the frame in pixels
            fov_h: Horizontal field of view in radians
            motion_state: User's ego-motion from the IMU ("walking" | "stationary"
                | "unknown"). When "walking", the apparent approach of objects is
                ego-compensated so static obstacles ahead don't read as collisions.
                "unknown" (default) applies no compensation (backwards-compatible).

        Returns:
            List of TrackedObject with tracking info
        """
        if not detections:
            # Increment missing count for all tracks
            for track in self.tracks.values():
                track.frames_missing += 1
            # Remove old tracks
            self._cleanup()
            return []

        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()

        # Sort tracks by collision_risk (process important ones first)
        track_ids = sorted(
            self.tracks.keys(),
            key=lambda tid: self.tracks[tid].collision_risk,
            reverse=True
        )

        for track_id in track_ids:
            track = self.tracks[track_id]
            best_iou = 0
            best_det_idx = -1

            for det_idx, det in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                if det.name != track.name:  # Must be same class
                    continue

                iou = _iou(track.bbox, det.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                # Update track with new detection
                det = detections[best_det_idx]
                track.bbox = det.bbox
                track.zone = det.zone
                track.distance = det.distance
                track.depth_value = det.depth_value
                track.confidence = det.confidence
                track.is_gazed = det.is_gazed
                track.traffic_light_state = getattr(det, 'traffic_light_state', None)
                track.frame_width = frame_width
                track.fov_h = fov_h
                track.depth_history.append(det.depth_value)
                track.bearing = track._pixel_to_bearing(det.bbox)
                track.bearing_history.append(track.bearing)
                track.height_history.append(det.bbox[3])
                track.frames_seen += 1
                track.frames_missing = 0

                # Calculate approach + lateral speed
                self._update_approach(track, motion_state)
                self._update_lateral(track)

                # Calculate collision risk (H18)
                self._update_collision_risk(track)

                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue

            new_track = TrackedObject(
                id=self.next_id,
                name=det.name,
                bbox=det.bbox,
                zone=det.zone,
                distance=det.distance,
                depth_value=det.depth_value,
                confidence=det.confidence,
                is_gazed=det.is_gazed,
                frame_width=frame_width,
                fov_h=fov_h,
                traffic_light_state=getattr(det, 'traffic_light_state', None),
            )
            self._update_collision_risk(new_track)
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # Increment missing count for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].frames_missing += 1

        # Cleanup old tracks
        self._cleanup()

        # Return sorted by collision_risk
        return sorted(
            self.tracks.values(),
            key=lambda t: t.collision_risk,
            reverse=True
        )

    def _update_approach(self, track: TrackedObject, motion_state: str = "unknown"):
        """Estimate approach by fusing depth slope with bbox-height looming.

        Two complementary cues:
        - Depth slope: DepthAnything value rising = object getting closer. But
          the value is relative (NORM_MINMAX) and noisy.
        - Looming: the bbox HEIGHT growing = the object subtends more FoV =
          closing in. Independent of the depth model.

        Looming is expressed in the SAME units as the depth slope via
        (Δheight/height)·remaining_gap, so the downstream TTC
        (remaining_gap/approach) of a looming-only approach equals the
        bbox time-to-contact height/Δheight, with no arbitrary scale factor. The two
        are fused with max() — either can flag an approach; when the bbox is
        static, looming is 0 and depth drives it (backwards-compatible).

        When the user is walking, part of any positive approach is self-motion,
        not the object closing in: we subtract EGO_MOTION_WALKING_BIAS from the
        fused signal so a STATIC object ahead nets to ~0 while a genuinely fast
        approacher keeps the residual.
        """
        if len(track.depth_history) < 3:
            track.is_approaching = False
            track.approach_speed = 0.0
            track.looming_speed = 0.0
            return

        # Depth Anything: higher value = closer
        history = list(track.depth_history)
        recent = history[-3:]  # Last 3 frames
        x = np.arange(len(recent))
        depth_slope = np.polyfit(x, recent, 1)[0]

        # Bbox-height looming, converted to depth-slope units.
        looming = 0.0
        if len(track.height_history) >= 3:
            hrecent = list(track.height_history)[-3:]
            mean_h = sum(hrecent) / len(hrecent)
            if mean_h > 0:
                h_rel_growth = np.polyfit(x, hrecent, 1)[0] / mean_h  # per frame
                looming = h_rel_growth * remaining_gap(track.depth_value)
        track.looming_speed = float(looming)

        # Fuse: either cue can flag an approach.
        slope = max(float(depth_slope), looming)

        # Ego-motion compensation: only the approaching (positive) component is
        # inflated by walking — recession is left untouched.
        if motion_state == "walking" and slope > 0:
            slope = slope - EGO_MOTION_WALKING_BIAS

        track.approach_speed = slope
        track.is_approaching = slope > 0.01  # Threshold for noise

    def _update_lateral(self, track: TrackedObject):
        """Calculate lateral speed and whether object enters user's path.

        Uses bearing (radians) instead of pixel position so the threshold
        is consistent across different camera FOVs.
        """
        if len(track.bearing_history) < 3:
            track.lateral_speed = 0.0
            track.enters_path = False
            return

        history = list(track.bearing_history)
        recent = history[-3:]

        # Slope of bearing over frames (rad/frame): positive = moving right
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        track.lateral_speed = slope

        # "enters_path" = object moving toward center (bearing=0) from either side
        # ~0.17 rad ≈ 10° — object is off-center but not at the edge
        current_bearing = recent[-1]
        # ~0.005 rad/frame threshold filters noise
        moving_toward_center = (
            (current_bearing < -0.17 and slope > 0.005) or  # on left, moving right
            (current_bearing > 0.17 and slope < -0.005)     # on right, moving left
        )
        close_enough = track.distance in ("very_close", "close")
        track.enters_path = moving_toward_center and close_enough

    def _update_collision_risk(self, track: TrackedObject):
        """Calculate collision risk score 0.0–1.0 and threat level (H18).

        Combines 4 weighted factors from ADAS literature:
        - TTC proxy (50%): time-to-collision from depth approach speed
        - CBDR (25%): constant bearing + decreasing range = collision course
        - Zone (15%): center of path = higher risk
        - Class (10%): vehicles are lethal, static objects are low risk
        """
        risk = 0.0

        # Factor 1: TTC proxy (50%)
        # TTC = remaining gap / approach_speed (in frames); depth_value is proximity
        # Normalize: TTC 0 = 1.0, TTC 150 frames (~5s @30fps) = 0.0
        if track.approach_speed > 0.01:
            ttc_frames = remaining_gap(track.depth_value) / track.approach_speed
            ttc_factor = max(0.0, 1.0 - ttc_frames / 150.0)
        else:
            # Static object: use proximity as TTC proxy
            ttc_factor = STATIC_PROXIMITY.get(track.distance, 0.1)

        risk += ttc_factor * 0.50

        # Factor 2: CBDR — bearing stable + approaching (25%)
        # If bearing doesn't change while distance decreases → collision course
        # Captures lateral bikes/motorcycles maintaining collision heading
        if track.approach_speed > 0.01:
            bearing_stability = max(0.0, 1.0 - abs(track.lateral_speed) / 0.02)
            approach_intensity = min(1.0, track.approach_speed / 0.03)
            cbdr_factor = bearing_stability * approach_intensity
        else:
            cbdr_factor = 0.0

        risk += cbdr_factor * 0.25

        # Factor 3: Zone (15%)
        risk += ZONE_RISK.get(track.zone, 0.3) * 0.15

        # Factor 4: Object class (10%)
        risk += CLASS_RISK.get(track.name, 0.1) * 0.10

        track.collision_risk = min(1.0, risk)

        # Map to threat level
        if track.collision_risk >= THREAT_THRESHOLDS["DANGER"]:
            track.threat_level = "DANGER"
        elif track.collision_risk >= THREAT_THRESHOLDS["WARNING"]:
            track.threat_level = "WARNING"
        elif track.collision_risk >= THREAT_THRESHOLDS["ATTENTION"]:
            track.threat_level = "ATTENTION"
        else:
            track.threat_level = "NONE"

    def _cleanup(self):
        """Remove tracks that have been missing too long."""
        to_remove = [
            tid for tid, track in self.tracks.items()
            if track.frames_missing > self.max_missing
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def get_approaching_objects(self) -> List[TrackedObject]:
        """Get objects that are approaching."""
        return [t for t in self.tracks.values() if t.is_approaching]

    def get_path_entering_objects(self) -> List[TrackedObject]:
        """Get objects moving laterally into the user's path."""
        return [t for t in self.tracks.values() if t.enters_path]
