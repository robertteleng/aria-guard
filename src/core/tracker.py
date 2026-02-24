"""
Simple object tracker for ARIA demo.

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
    frames_seen: int = 1
    frames_missing: int = 0

    # Computed
    is_approaching: bool = False
    approach_speed: float = 0.0  # positive = approaching
    lateral_speed: float = 0.0  # rad/frame, positive = moving right
    enters_path: bool = False  # object moving laterally into user's path (center)
    bearing: float = 0.0  # current bearing in radians
    priority: float = 0.0

    # H18: collision risk score (0.0–1.0) and threat level
    collision_risk: float = 0.0
    threat_level: str = "NONE"  # NONE, ATTENTION, WARNING, DANGER

    def __post_init__(self):
        if not self.depth_history:
            self.depth_history = deque(maxlen=10)
        if not self.bearing_history:
            self.bearing_history = deque(maxlen=10)
        self.depth_history.append(self.depth_value)
        self.bearing = self._pixel_to_bearing(self.bbox)
        self.bearing_history.append(self.bearing)

    def _pixel_to_bearing(self, bbox) -> float:
        """Convert bbox center_x to bearing angle in radians using FOV."""
        x, y, w, h = bbox
        center_x = (x + w / 2) / self.frame_width if self.frame_width > 0 else 0.5
        # Map pixel position to angle: atan((normalized - 0.5) * 2 * tan(fov/2))
        return math.atan((center_x - 0.5) * 2 * math.tan(self.fov_h / 2))


# Object type priority (higher = more dangerous)
OBJECT_PRIORITY = {
    # Vehicles - highest priority
    "car": 10, "truck": 10, "bus": 10,
    "motorcycle": 9, "bicycle": 8,

    # Traffic signals — high priority for crossing decisions
    "traffic light": 8, "stop sign": 7,

    # People/animals
    "person": 6, "dog": 5, "cat": 4,

    # Obstacles
    "chair": 3, "couch": 3, "bed": 2,
    "dining table": 2, "toilet": 2,

    # Objects
    "backpack": 1, "handbag": 1, "suitcase": 1,
}

# Distance priority multiplier
DISTANCE_PRIORITY = {
    "very_close": 4.0,
    "close": 2.0,
    "medium": 1.0,
    "far": 0.5,
    "unknown": 1.0,
}

# Zone priority multiplier (center = user walks into it)
ZONE_PRIORITY = {
    "center": 1.5,
    "left": 1.0,
    "right": 1.0,
}

# H18: Collision risk weights by object class (0.0–1.0)
# Vehicles are lethal, people are unpredictable, static objects are low risk
CLASS_RISK = {
    "car": 1.0, "truck": 1.0, "bus": 1.0,
    "motorcycle": 0.9, "bicycle": 0.7,
    "person": 0.3, "dog": 0.2, "cat": 0.15,
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
    "DANGER": 0.6,     # ~TTC < 1.5s (Euro NCAP AEB activation)
    "WARNING": 0.35,   # ~TTC < 3.0s (Mobileye FCW)
    "ATTENTION": 0.15, # ~TTC < 5.0s
}


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

    def update(self, detections: List, frame_width: int = 1280, fov_h: float = 1.15) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects from detector
            frame_width: Width of the frame in pixels
            fov_h: Horizontal field of view in radians

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

        # Sort tracks by priority (process important ones first)
        track_ids = sorted(
            self.tracks.keys(),
            key=lambda tid: self.tracks[tid].priority,
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
                track.frames_seen += 1
                track.frames_missing = 0

                # Calculate approach + lateral speed
                self._update_approach(track)
                self._update_lateral(track)

                # Calculate priority (v2) + collision risk (H18)
                self._update_priority(track)
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
            self._update_priority(new_track)
            self._update_collision_risk(new_track)
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # Increment missing count for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].frames_missing += 1

        # Cleanup old tracks
        self._cleanup()

        # Return sorted by priority
        return sorted(
            self.tracks.values(),
            key=lambda t: t.priority,
            reverse=True
        )

    def _update_approach(self, track: TrackedObject):
        """Calculate if object is approaching based on depth history."""
        if len(track.depth_history) < 3:
            track.is_approaching = False
            track.approach_speed = 0.0
            return

        # Depth Anything: higher value = closer
        # If depth is increasing, object is approaching
        history = list(track.depth_history)
        recent = history[-3:]  # Last 3 frames

        # Simple linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

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

    def _update_priority(self, track: TrackedObject):
        """Calculate priority score for the track (v2).

        v1 used binary approach (2x on/off). v2 uses continuous approach speed
        and adds zone factor so center objects rank higher.
        """
        # Base priority from object type
        type_priority = OBJECT_PRIORITY.get(track.name, 1)

        # Distance multiplier
        dist_mult = DISTANCE_PRIORITY.get(track.distance, 1.0)

        # Approach speed: continuous 1.0–3.0 (v2: replaces binary 2x)
        # approach_speed is depth slope/frame; 0.01 = barely moving, 0.05+ = fast
        if track.approach_speed > 0.01:
            approach_mult = min(3.0, 1.0 + track.approach_speed * 40.0)
        else:
            approach_mult = 1.0

        # Zone: center = higher risk (user walks straight into it)
        zone_mult = ZONE_PRIORITY.get(track.zone, 1.0)

        # Lateral intercept bonus (2.5x if entering user's path)
        path_mult = 2.5 if track.enters_path else 1.0

        # Not gazed bonus (1.5x if user not looking)
        gaze_mult = 1.5 if not track.is_gazed else 1.0

        # Combine
        track.priority = type_priority * dist_mult * approach_mult * zone_mult * path_mult * gaze_mult

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
        # TTC = depth / approach_speed (in frames)
        # Normalize: TTC 0 = 1.0, TTC 150 frames (~5s @30fps) = 0.0
        if track.approach_speed > 0.01:
            ttc_frames = track.depth_value / track.approach_speed
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

    def get_top_priority(self, n: int = 1) -> List[TrackedObject]:
        """Get top N priority objects."""
        sorted_tracks = sorted(
            self.tracks.values(),
            key=lambda t: t.priority,
            reverse=True
        )
        return sorted_tracks[:n]

    def get_approaching_objects(self) -> List[TrackedObject]:
        """Get objects that are approaching."""
        return [t for t in self.tracks.values() if t.is_approaching]

    def get_path_entering_objects(self) -> List[TrackedObject]:
        """Get objects moving laterally into the user's path."""
        return [t for t in self.tracks.values() if t.enters_path]

    def get_top_vehicle(self) -> Optional[TrackedObject]:
        """Get highest priority vehicle (car, truck, bus, motorcycle, bicycle)."""
        vehicles = {"car", "truck", "bus", "motorcycle", "bicycle"}
        vehicle_tracks = [t for t in self.tracks.values() if t.name in vehicles]
        if not vehicle_tracks:
            return None
        return max(vehicle_tracks, key=lambda t: t.priority)

    def get_top_non_vehicle(self) -> Optional[TrackedObject]:
        """Get highest priority non-vehicle (person, obstacle, etc).
        Excludes traffic lights and signs — they have independent alert channels."""
        exclude = {"car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "stop sign"}
        other_tracks = [t for t in self.tracks.values() if t.name not in exclude]
        if not other_tracks:
            return None
        return max(other_tracks, key=lambda t: t.priority)
