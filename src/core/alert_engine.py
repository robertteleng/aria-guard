"""
Alert Decision Engine for ARIA demo.

Centralizes all alert logic: what to alert, when, and why.
Separates decision-making from audio playback.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

from src.core.tracker import TrackedObject, SimpleTracker


@dataclass
class AlertDecision:
    """Result of alert decision process."""
    should_alert: bool
    object: Optional[TrackedObject] = None
    reason: str = ""  # "close", "approaching", etc.


class AlertDecisionEngine:
    """
    Decides which objects deserve alerts based on:
    - Object type (vehicles > people > obstacles)
    - Distance (very_close, close trigger alerts)
    - Approach speed (approaching objects at medium distance)
    - User gaze (not looking = more urgent)
    """

    # Vehicles always get priority - they're more dangerous
    VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(
        self,
        vehicle_cooldown: float = 1.5,
        other_cooldown: float = 2.0,
        same_object_cooldown: float = 3.0
    ):
        """
        Args:
            vehicle_cooldown: Min seconds between vehicle alerts
            other_cooldown: Min seconds between non-vehicle alerts
            same_object_cooldown: Min seconds before re-alerting same object
        """
        self.vehicle_cooldown = vehicle_cooldown
        self.other_cooldown = other_cooldown
        self.same_object_cooldown = same_object_cooldown

        self._last_vehicle_alert = 0.0
        self._last_other_alert = 0.0
        self._last_alerted_id: Optional[int] = None
        self._last_alerted_time = 0.0

    def decide(self, tracker: SimpleTracker) -> Tuple[Optional[AlertDecision], Optional[AlertDecision]]:
        """
        Decide what to alert based on current tracked objects.

        Returns:
            (vehicle_alert, other_alert) - either can be None
            Both can have alerts if cooldowns allow
        """
        now = time.time()

        vehicle_decision = None
        other_decision = None

        # Get top candidates
        top_vehicle = tracker.get_top_vehicle()
        top_other = tracker.get_top_non_vehicle()

        # Vehicle alert check
        if top_vehicle and self._should_alert_vehicle(top_vehicle, now):
            vehicle_decision = AlertDecision(
                should_alert=True,
                object=top_vehicle,
                reason=self._get_alert_reason(top_vehicle)
            )
            self._last_vehicle_alert = now
            self._record_alert(top_vehicle.id, now)

        # Non-vehicle alert check (only if no vehicle alert)
        if not vehicle_decision and top_other and self._should_alert_other(top_other, now):
            other_decision = AlertDecision(
                should_alert=True,
                object=top_other,
                reason=self._get_alert_reason(top_other)
            )
            self._last_other_alert = now
            self._record_alert(top_other.id, now)

        return vehicle_decision, other_decision

    def _should_alert_vehicle(self, obj: TrackedObject, now: float) -> bool:
        """Check if vehicle should trigger alert."""
        # Cooldown check
        if now - self._last_vehicle_alert < self.vehicle_cooldown:
            return False

        # Same object cooldown
        if self._is_same_object_too_recent(obj.id, now):
            return False

        # Alert conditions for vehicles:
        # Vehicles are dangerous - alert if close OR approaching
        # Gaze = direction user will walk, so looking at vehicle = collision risk
        if obj.distance in ("very_close", "close"):
            return True  # Always alert for close vehicles

        if obj.is_approaching and obj.distance == "medium":
            return True  # Alert for approaching vehicles

        return False

    def _should_alert_other(self, obj: TrackedObject, now: float) -> bool:
        """Check if non-vehicle should trigger alert."""
        # Cooldown check
        if now - self._last_other_alert < self.other_cooldown:
            return False

        # Same object cooldown
        if self._is_same_object_too_recent(obj.id, now):
            return False

        # Alert conditions for non-vehicles:
        # Only close distance (they're less dangerous)
        if obj.distance in ("very_close", "close"):
            return not obj.is_gazed

        return False

    def _is_same_object_too_recent(self, obj_id: int, now: float) -> bool:
        """Check if we recently alerted about this same object."""
        if self._last_alerted_id == obj_id:
            return now - self._last_alerted_time < self.same_object_cooldown
        return False

    def _record_alert(self, obj_id: int, now: float):
        """Record that we alerted about this object."""
        self._last_alerted_id = obj_id
        self._last_alerted_time = now

    def _get_alert_reason(self, obj: TrackedObject) -> str:
        """Get human-readable reason for alert."""
        if obj.distance == "very_close":
            return "very_close"
        elif obj.distance == "close":
            return "close"
        elif obj.is_approaching:
            return "approaching"
        return "unknown"

    def get_zone_word(self, zone: str) -> str:
        """Convert zone to spoken word."""
        return {"left": "left", "right": "right", "center": "straight"}.get(zone, "")
