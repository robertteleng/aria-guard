"""
Simple object tracker for ARIA demo.

Tracks objects across frames to detect approach and prioritize alerts.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
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

    # Tracking data
    depth_history: deque = field(default_factory=lambda: deque(maxlen=10))
    frames_seen: int = 1
    frames_missing: int = 0

    # Computed
    is_approaching: bool = False
    approach_speed: float = 0.0  # positive = approaching
    priority: float = 0.0

    def __post_init__(self):
        if not self.depth_history:
            self.depth_history = deque(maxlen=10)
        self.depth_history.append(self.depth_value)


# Object type priority (higher = more dangerous)
OBJECT_PRIORITY = {
    # Vehicles - highest priority
    "car": 10, "truck": 10, "bus": 10,
    "motorcycle": 9, "bicycle": 8,

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

    def update(self, detections: List) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects from detector

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
                track.depth_history.append(det.depth_value)
                track.frames_seen += 1
                track.frames_missing = 0

                # Calculate approach speed
                self._update_approach(track)

                # Calculate priority
                self._update_priority(track)

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
            )
            self._update_priority(new_track)
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

    def _update_priority(self, track: TrackedObject):
        """Calculate priority score for the track."""
        # Base priority from object type
        type_priority = OBJECT_PRIORITY.get(track.name, 1)

        # Distance multiplier
        dist_mult = DISTANCE_PRIORITY.get(track.distance, 1.0)

        # Approach bonus (2x if approaching)
        approach_mult = 2.0 if track.is_approaching else 1.0

        # Not gazed bonus (1.5x if user not looking)
        gaze_mult = 1.5 if not track.is_gazed else 1.0

        # Combine
        track.priority = type_priority * dist_mult * approach_mult * gaze_mult

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
