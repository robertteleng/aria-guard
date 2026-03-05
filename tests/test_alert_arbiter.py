"""Tests for AlertArbiter 2-channel architecture (H19).

Validates channel A (threat) and channel B (context) behavior,
cooldowns, rate limiting, and anti-saturation.
"""
import time
import pytest
from unittest.mock import patch
from src.domain.alert_engine import AlertArbiter, CHANNEL_B_SILENCE, MAX_ALERTS_COUNT
from src.domain.tracker import SimpleTracker, TrackedObject


class FakeDet:
    """Minimal detection for testing."""
    def __init__(self, **kw):
        kw.setdefault("traffic_light_state", None)
        self.__dict__.update(kw)


def build_tracker(dets_sequence, n_frames=5, frame_width=1280):
    """Feed detections and return the tracker with history."""
    tracker = SimpleTracker()
    for dets in dets_sequence:
        tracker.update(dets, frame_width=frame_width)
    return tracker


def make_car_approaching(n=5):
    """Create sequence of an approaching car."""
    return [
        [FakeDet(name="car", bbox=(600, 300, 100, 80), zone="center",
                 distance="medium" if i < 3 else "close",
                 depth_value=0.3 + i * 0.05, confidence=0.85, is_gazed=False)]
        for i in range(n)
    ]


def make_person_far(n=5):
    """Static person far away."""
    return [
        [FakeDet(name="person", bbox=(200, 200, 40, 80), zone="left",
                 distance="far", depth_value=0.15, confidence=0.8, is_gazed=False)]
    ] * n


def make_traffic_light(state="red", n=5):
    """Traffic light in medium distance."""
    return [
        [FakeDet(name="traffic light", bbox=(640, 100, 30, 60), zone="center",
                 distance="medium", depth_value=0.3, confidence=0.9, is_gazed=False,
                 traffic_light_state=state)]
    ] * n


class TestChannelA:
    """Channel A (threat) tests."""

    def test_danger_always_alerts(self):
        """DANGER is never suppressed."""
        tracker = build_tracker(make_car_approaching())
        arbiter = AlertArbiter()
        a, _ = arbiter.decide(tracker)
        assert a is not None
        assert a.threat_level == "DANGER"
        assert a.use_tts is True

    def test_none_threat_no_alert(self):
        """Objects with threat_level=NONE should not trigger."""
        tracker = build_tracker(make_person_far())
        arbiter = AlertArbiter()
        a, _ = arbiter.decide(tracker)
        assert a is None

    def test_cooldown_suppresses_warning(self):
        """WARNING should be suppressed within cooldown."""
        # Person close static = WARNING
        dets = [
            [FakeDet(name="person", bbox=(600, 300, 50, 100), zone="center",
                     distance="close", depth_value=0.6, confidence=0.8, is_gazed=False)]
        ] * 5
        tracker = build_tracker(dets)
        arbiter = AlertArbiter()

        a1, _ = arbiter.decide(tracker)
        assert a1 is not None  # First alert fires

        a2, _ = arbiter.decide(tracker)
        assert a2 is None  # Suppressed by cooldown

    def test_danger_respects_min_cooldown(self):
        """DANGER should respect its 1.5s cooldown (not fire immediately)."""
        tracker = build_tracker(make_car_approaching())
        arbiter = AlertArbiter()

        a1, _ = arbiter.decide(tracker)
        assert a1 is not None

        # Immediately again — DANGER now respects 1.5s cooldown
        a2, _ = arbiter.decide(tracker)
        assert a2 is None  # Suppressed by cooldown

    def test_danger_bypasses_rate_limit(self):
        """DANGER should fire even when rate limit is exceeded."""
        tracker = build_tracker(make_car_approaching())
        arbiter = AlertArbiter()

        # Fill rate limit buffer
        now = time.time()
        for i in range(MAX_ALERTS_COUNT):
            arbiter._alert_times.append(now - 5 - i)

        # Set last alert far enough back to pass cooldown
        arbiter._last_a_alert_time = now - 10

        a, _ = arbiter.decide(tracker)
        assert a is not None
        assert a.threat_level == "DANGER"

    def test_gazed_object_no_tts(self):
        """Gazed WARNING object should beep but not TTS."""
        dets = [
            [FakeDet(name="person", bbox=(600, 300, 50, 100), zone="center",
                     distance="close", depth_value=0.6, confidence=0.8, is_gazed=True)]
        ] * 5
        tracker = build_tracker(dets)
        arbiter = AlertArbiter()
        a, _ = arbiter.decide(tracker)
        assert a is not None
        assert a.use_tts is False  # Gazed = beep only

    def test_gazed_danger_still_tts(self):
        """DANGER always gets TTS even if gazed."""
        dets = [
            [FakeDet(name="car", bbox=(600, 300, 100, 80), zone="center",
                     distance="close", depth_value=0.3 + i * 0.05,
                     confidence=0.85, is_gazed=True)]
            for i in range(5)
        ]
        tracker = build_tracker(dets)
        arbiter = AlertArbiter()
        a, _ = arbiter.decide(tracker)
        assert a is not None
        assert a.use_tts is True  # DANGER = always TTS


class TestChannelB:
    """Channel B (context) tests."""

    def test_context_suppressed_when_a_active(self):
        """Channel B should not fire when A just alerted."""
        # Both car (threat) and traffic light (context)
        dets = [
            [FakeDet(name="car", bbox=(600, 300, 100, 80), zone="center",
                     distance="close", depth_value=0.3 + i * 0.05,
                     confidence=0.85, is_gazed=False),
             FakeDet(name="traffic light", bbox=(640, 100, 30, 60), zone="center",
                     distance="medium", depth_value=0.3, confidence=0.9, is_gazed=False,
                     traffic_light_state="red")]
            for i in range(5)
        ]
        tracker = build_tracker(dets)
        arbiter = AlertArbiter()
        a, b = arbiter.decide(tracker)
        assert a is not None  # Car DANGER fires
        assert b is None  # Traffic light suppressed (A just spoke)

    def test_context_fires_when_a_silent(self):
        """Channel B should fire when A has been silent."""
        tracker = build_tracker(make_traffic_light("red"))
        arbiter = AlertArbiter()
        # Force A to be silent long enough
        arbiter._last_a_alert_time = time.time() - CHANNEL_B_SILENCE - 1
        _, b = arbiter.decide(tracker)
        assert b is not None
        assert b.reason == "traffic_light_red"

    def test_traffic_light_state_change_bypasses_cooldown(self):
        """State change (red→green) should bypass cooldown."""
        arbiter = AlertArbiter()
        arbiter._last_a_alert_time = 0  # A is silent

        # First: red light
        tracker_red = build_tracker(make_traffic_light("red"))
        _, b1 = arbiter.decide(tracker_red)
        assert b1 is not None

        # Immediately: green light (state changed)
        tracker_green = build_tracker(make_traffic_light("green"))
        _, b2 = arbiter.decide(tracker_green)
        assert b2 is not None
        assert b2.reason == "traffic_light_green"


class TestRateLimiting:
    """Rate limiting and anti-saturation tests."""

    def test_rate_limit_blocks_non_danger(self):
        """Non-DANGER should be blocked after MAX_ALERTS_COUNT."""
        dets = [
            [FakeDet(name="person", bbox=(600, 300, 50, 100), zone="center",
                     distance="close", depth_value=0.6, confidence=0.8, is_gazed=False)]
        ] * 5
        tracker = build_tracker(dets)
        arbiter = AlertArbiter()

        # Simulate MAX_ALERTS_COUNT recent alerts
        now = time.time()
        for i in range(MAX_ALERTS_COUNT):
            arbiter._alert_times.append(now - i)

        a, _ = arbiter.decide(tracker)
        assert a is None  # Rate limited
