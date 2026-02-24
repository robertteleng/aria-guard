"""
Alert Arbiter for ARIA (H19).

2-channel architecture based on collision_risk (H18):
- Channel A (threat): top-1 object by collision_risk, adaptive cooldowns
- Channel B (context): traffic lights, signs — only when A is silent

Replaces AlertDecisionEngine (v1/v2) which used object-type heuristics.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import time

from src.core.tracker import TrackedObject, SimpleTracker, THREAT_THRESHOLDS


# Context classes — handled by Channel B
CONTEXT_CLASSES = {"traffic light", "stop sign"}

# Cooldowns per threat level (seconds)
THREAT_COOLDOWNS = {
    "DANGER": 1.5,
    "WARNING": 3.0,
    "ATTENTION": 5.0,
}

# Rate limiting
MAX_ALERTS_WINDOW = 30.0   # seconds
MAX_ALERTS_COUNT = 6       # max alerts in window (~12/min peak)
SATURATION_WINDOW = 20.0   # seconds
SATURATION_THRESHOLD = 4   # alerts in window to trigger anti-saturation
CHANNEL_B_SILENCE = 3.0    # Channel A must be silent this long before B speaks


@dataclass
class AlertDecision:
    """Result of alert decision process."""
    should_alert: bool
    object: Optional[TrackedObject] = None
    reason: str = ""
    threat_level: str = "NONE"
    use_tts: bool = False  # Whether TTS should speak (vs beep only)


class AlertArbiter:
    """2-channel alert arbiter based on collision_risk scoring.

    Channel A — Threat (max 1 at a time):
        Selects top-1 object by collision_risk. Cooldown adapts by threat level.
        DANGER is never suppressed. Rate limit: max 6/30s.
        Gaze modulates urgency (TTS vs beep-only), NOT the risk score.

    Channel B — Context (informational):
        Traffic lights and signs. Only speaks when Channel A is silent >3s.
        Long cooldowns (5-8s). State changes bypass cooldown.
    """

    def __init__(self):
        # Channel A state
        self._last_a_alert_time = 0.0
        self._last_a_id: Optional[int] = None
        self._last_a_level: str = "NONE"

        # Channel B state
        self._last_tl_alert_time = 0.0
        self._last_tl_state: Optional[str] = None
        self._last_sign_alert_time = 0.0
        self._last_sign_id: Optional[int] = None

        # Rate limiting: ring buffer of recent alert timestamps
        self._alert_times: deque = deque(maxlen=MAX_ALERTS_COUNT * 2)

    def decide(self, tracker: SimpleTracker) -> Tuple[Optional[AlertDecision], Optional[AlertDecision]]:
        """Decide what to alert based on current tracked objects.

        Returns:
            (channel_a, channel_b) — either can be None.
            channel_a: threat alert (DANGER/WARNING/ATTENTION)
            channel_b: context alert (traffic light / sign)
        """
        now = time.time()

        channel_a = self._decide_channel_a(tracker, now)
        channel_b = self._decide_channel_b(tracker, now)

        return channel_a, channel_b

    # --- Channel A: Threat ---

    def _decide_channel_a(self, tracker: SimpleTracker, now: float) -> Optional[AlertDecision]:
        """Select top threat by collision_risk, apply cooldowns and rate limit."""
        # Get all non-context tracks with threat level > NONE
        candidates = [
            t for t in tracker.tracks.values()
            if t.name not in CONTEXT_CLASSES and t.threat_level != "NONE"
        ]
        if not candidates:
            return None

        # Select top-1 by collision_risk
        top = max(candidates, key=lambda t: t.collision_risk)

        # DANGER is never suppressed by cooldown or rate limit
        is_danger = top.threat_level == "DANGER"

        if not is_danger:
            # Cooldown check (adaptive by last alert level)
            cooldown = self._get_cooldown(now)
            if now - self._last_a_alert_time < cooldown:
                return None

            # Rate limit check
            if self._is_rate_limited(now):
                return None

            # Same object cooldown (don't nag about same object)
            if self._last_a_id == top.id:
                same_obj_cooldown = THREAT_COOLDOWNS.get(top.threat_level, 3.0) * 1.5
                if now - self._last_a_alert_time < same_obj_cooldown:
                    return None

        # Gaze modulates urgency: gazed = beep only, not gazed = beep + TTS
        # Exception: DANGER always gets full TTS
        use_tts = is_danger or not top.is_gazed

        # Record alert
        self._last_a_alert_time = now
        self._last_a_id = top.id
        self._last_a_level = top.threat_level
        self._alert_times.append(now)

        return AlertDecision(
            should_alert=True,
            object=top,
            reason=top.threat_level.lower(),
            threat_level=top.threat_level,
            use_tts=use_tts,
        )

    def _get_cooldown(self, now: float) -> float:
        """Get current cooldown, doubled if in saturation."""
        base = THREAT_COOLDOWNS.get(self._last_a_level, 3.0)

        # Anti-saturation: if 4+ alerts in 20s, double cooldowns
        recent = sum(1 for t in self._alert_times if now - t < SATURATION_WINDOW)
        if recent >= SATURATION_THRESHOLD:
            base *= 2.0

        return base

    def _is_rate_limited(self, now: float) -> bool:
        """Check if global rate limit is exceeded (max 6/30s)."""
        recent = sum(1 for t in self._alert_times if now - t < MAX_ALERTS_WINDOW)
        return recent >= MAX_ALERTS_COUNT

    # --- Channel B: Context ---

    def _decide_channel_b(self, tracker: SimpleTracker, now: float) -> Optional[AlertDecision]:
        """Handle traffic lights and signs — only when Channel A is silent."""
        # Channel B only speaks when A has been silent for >3s
        if now - self._last_a_alert_time < CHANNEL_B_SILENCE:
            return None

        # Try traffic light first (higher priority context)
        tl = self._check_traffic_light(tracker, now)
        if tl:
            return tl

        # Then signs
        sign = self._check_sign(tracker, now)
        if sign:
            return sign

        return None

    def _check_traffic_light(self, tracker: SimpleTracker, now: float) -> Optional[AlertDecision]:
        """Check for traffic light state to announce."""
        tl_tracks = [
            t for t in tracker.tracks.values()
            if t.name == "traffic light"
            and t.traffic_light_state is not None
            and t.distance in ("very_close", "close", "medium")
        ]
        if not tl_tracks:
            return None

        top_tl = max(tl_tracks, key=lambda t: t.collision_risk)

        # Cooldown: 5s, but state change bypasses
        state_changed = top_tl.traffic_light_state != self._last_tl_state
        if not state_changed and now - self._last_tl_alert_time < 5.0:
            return None

        self._last_tl_alert_time = now
        self._last_tl_state = top_tl.traffic_light_state

        return AlertDecision(
            should_alert=True,
            object=top_tl,
            reason=f"traffic_light_{top_tl.traffic_light_state}",
            threat_level="CONTEXT",
            use_tts=True,
        )

    def _check_sign(self, tracker: SimpleTracker, now: float) -> Optional[AlertDecision]:
        """Check for sign to announce."""
        sign_tracks = [
            t for t in tracker.tracks.values()
            if t.name == "stop sign"
            and t.distance in ("very_close", "close", "medium")
        ]
        if not sign_tracks:
            return None

        top_sign = max(sign_tracks, key=lambda t: t.collision_risk)

        # Cooldown: 8s, different sign bypasses
        different_sign = self._last_sign_id != top_sign.id
        if not different_sign and now - self._last_sign_alert_time < 8.0:
            return None

        self._last_sign_alert_time = now
        self._last_sign_id = top_sign.id

        return AlertDecision(
            should_alert=True,
            object=top_sign,
            reason=f"sign_{top_sign.name}",
            threat_level="CONTEXT",
            use_tts=True,
        )

    @staticmethod
    def get_zone_word(zone: str) -> str:
        """Convert zone to spoken word."""
        return {"left": "left", "right": "right", "center": "straight"}.get(zone, "")
