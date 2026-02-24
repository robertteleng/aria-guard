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

    # Signs that get their own independent alert channel
    SIGN_CLASSES = {"stop sign"}

    def __init__(
        self,
        vehicle_cooldown: float = 1.5,
        other_cooldown: float = 2.0,
        same_object_cooldown: float = 3.0,
        traffic_light_cooldown: float = 4.0,
        sign_cooldown: float = 5.0
    ):
        """
        Args:
            vehicle_cooldown: Min seconds between vehicle alerts
            other_cooldown: Min seconds between non-vehicle alerts
            same_object_cooldown: Min seconds before re-alerting same object
            traffic_light_cooldown: Min seconds between traffic light state alerts
            sign_cooldown: Min seconds between sign alerts
        """
        self.vehicle_cooldown = vehicle_cooldown
        self.other_cooldown = other_cooldown
        self.same_object_cooldown = same_object_cooldown
        self.traffic_light_cooldown = traffic_light_cooldown
        self.sign_cooldown = sign_cooldown

        self._last_vehicle_alert = 0.0
        self._last_other_alert = 0.0
        self._last_traffic_light_alert = 0.0
        self._last_tl_state: Optional[str] = None  # last announced state
        self._last_sign_alert = 0.0
        self._last_sign_id: Optional[int] = None
        self._last_alerted_id: Optional[int] = None
        self._last_alerted_time = 0.0

    def decide(self, tracker: SimpleTracker) -> Tuple[Optional[AlertDecision], Optional[AlertDecision], Optional[AlertDecision], Optional[AlertDecision]]:
        """
        Decide what to alert based on current tracked objects.

        Returns:
            (vehicle_alert, other_alert, traffic_light_alert, sign_alert) - any can be None
        """
        now = time.time()

        vehicle_decision = None
        other_decision = None
        tl_decision = None
        sign_decision = None

        # Get top candidates
        top_vehicle = tracker.get_top_vehicle()
        top_other = tracker.get_top_non_vehicle()
        top_tl = self._get_top_traffic_light(tracker)
        top_sign = self._get_top_sign(tracker)

        # Traffic light alert (independent channel — crossing info is always useful)
        if top_tl and self._should_alert_traffic_light(top_tl, now):
            tl_decision = AlertDecision(
                should_alert=True,
                object=top_tl,
                reason=f"traffic_light_{top_tl.traffic_light_state}"
            )
            self._last_traffic_light_alert = now
            self._last_tl_state = top_tl.traffic_light_state

        # Sign alert (independent channel — crossing context)
        if top_sign and self._should_alert_sign(top_sign, now):
            sign_decision = AlertDecision(
                should_alert=True,
                object=top_sign,
                reason=f"sign_{top_sign.name}"
            )
            self._last_sign_alert = now
            self._last_sign_id = top_sign.id

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

        return vehicle_decision, other_decision, tl_decision, sign_decision

    def _get_top_traffic_light(self, tracker: SimpleTracker) -> Optional[TrackedObject]:
        """Get highest-priority traffic light with a classified state."""
        tl_tracks = [
            t for t in tracker.tracks.values()
            if t.name == "traffic light" and t.traffic_light_state is not None
        ]
        if not tl_tracks:
            return None
        return max(tl_tracks, key=lambda t: t.priority)

    def _should_alert_traffic_light(self, obj: TrackedObject, now: float) -> bool:
        """Alert on traffic light if state changed or cooldown expired."""
        if now - self._last_traffic_light_alert < self.traffic_light_cooldown:
            # Still alert if state changed (e.g. red → green)
            if obj.traffic_light_state == self._last_tl_state:
                return False
        # Alert for any visible, classified traffic light within reasonable range
        return obj.distance in ("very_close", "close", "medium")

    def _get_top_sign(self, tracker: SimpleTracker) -> Optional[TrackedObject]:
        """Get highest-priority sign (stop sign, etc)."""
        sign_tracks = [
            t for t in tracker.tracks.values()
            if t.name in self.SIGN_CLASSES
        ]
        if not sign_tracks:
            return None
        return max(sign_tracks, key=lambda t: t.priority)

    def _should_alert_sign(self, obj: TrackedObject, now: float) -> bool:
        """Alert on sign if cooldown expired and not the same sign."""
        if now - self._last_sign_alert < self.sign_cooldown:
            # Still alert if it's a different sign instance
            if self._last_sign_id == obj.id:
                return False
        return obj.distance in ("very_close", "close", "medium")

    def _should_alert_vehicle(self, obj: TrackedObject, now: float) -> bool:
        """Check if vehicle should trigger alert (v2).

        v1: alert if close OR approaching+medium.
        v2: also alert at far distance if approach speed is high (fast vehicle).
        """
        # Cooldown check
        if now - self._last_vehicle_alert < self.vehicle_cooldown:
            return False

        # Same object cooldown
        if self._is_same_object_too_recent(obj.id, now):
            return False

        # Always alert for close vehicles
        if obj.distance in ("very_close", "close"):
            return True

        # Approaching at medium distance
        if obj.is_approaching and obj.distance == "medium":
            return True

        # v2: fast approach at any distance (high approach_speed = urgent)
        if obj.approach_speed > 0.03 and obj.distance == "far":
            return True

        return False

    def _should_alert_other(self, obj: TrackedObject, now: float) -> bool:
        """Check if non-vehicle should trigger alert (v2).

        v1: alert if close + not gazed.
        v2: also alert approaching objects at medium distance in center zone.
        """
        # Cooldown check
        if now - self._last_other_alert < self.other_cooldown:
            return False

        # Same object cooldown
        if self._is_same_object_too_recent(obj.id, now):
            return False

        # Close distance: alert if user not looking
        if obj.distance in ("very_close", "close"):
            return not obj.is_gazed

        # v2: approaching in center zone at medium distance
        if obj.is_approaching and obj.distance == "medium" and obj.zone == "center":
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
        elif obj.approach_speed > 0.03:
            return "approaching_fast"
        elif obj.is_approaching:
            return "approaching"
        return "unknown"

    def get_zone_word(self, zone: str) -> str:
        """Convert zone to spoken word."""
        return {"left": "left", "right": "right", "center": "straight"}.get(zone, "")
