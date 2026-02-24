"""Tests for collision_risk scoring (H18).

Validates that the 4-factor model produces correct threat levels
for known scenarios.
"""
import pytest
from src.core.tracker import SimpleTracker, TrackedObject, THREAT_THRESHOLDS


class FakeDet:
    """Minimal detection for testing."""
    def __init__(self, **kw):
        kw.setdefault("traffic_light_state", None)
        self.__dict__.update(kw)


def run_scenario(dets_list, frame_width=1280):
    """Feed a sequence of detections and return the final tracked object."""
    tracker = SimpleTracker()
    for dets in dets_list:
        tracks = tracker.update(dets, frame_width=frame_width)
    return tracks[0] if tracks else None


def make_dets(n, **kw):
    """Create n identical detections (for building history)."""
    return [[FakeDet(**kw)] for _ in range(n)]


class TestThreatLevels:
    """Verify threat level classification for canonical scenarios."""

    def test_car_approaching_fast_center_is_danger(self):
        """Fast-approaching car in center → DANGER."""
        dets = [
            [FakeDet(name="car", bbox=(600, 300, 100, 80), zone="center",
                     distance="medium" if i < 3 else "close",
                     depth_value=0.3 + i * 0.05, confidence=0.85, is_gazed=False)]
            for i in range(5)
        ]
        t = run_scenario(dets)
        assert t.threat_level == "DANGER"
        assert t.collision_risk >= THREAT_THRESHOLDS["DANGER"]

    def test_person_far_static_is_none(self):
        """Static person far away → NONE."""
        dets = make_dets(5, name="person", bbox=(200, 200, 40, 80),
                         zone="left", distance="far", depth_value=0.15,
                         confidence=0.8, is_gazed=False)
        t = run_scenario(dets)
        assert t.threat_level == "NONE"
        assert t.collision_risk < THREAT_THRESHOLDS["ATTENTION"]

    def test_person_close_static_center_is_warning(self):
        """Static person close in center → WARNING (obstacle in path)."""
        dets = make_dets(5, name="person", bbox=(600, 300, 50, 100),
                         zone="center", distance="close", depth_value=0.6,
                         confidence=0.8, is_gazed=False)
        t = run_scenario(dets)
        assert t.threat_level == "WARNING"

    def test_bicycle_approaching_side_is_danger(self):
        """Bicycle approaching from side with CBDR → DANGER."""
        dets = [
            [FakeDet(name="bicycle", bbox=(900, 300, 40, 60), zone="right",
                     distance="medium", depth_value=0.3 + i * 0.04,
                     confidence=0.7, is_gazed=False)]
            for i in range(5)
        ]
        t = run_scenario(dets)
        assert t.threat_level == "DANGER"

    def test_car_static_far_is_attention(self):
        """Parked car far away → ATTENTION (class risk only)."""
        dets = make_dets(5, name="car", bbox=(100, 400, 80, 50),
                         zone="left", distance="far", depth_value=0.1,
                         confidence=0.9, is_gazed=True)
        t = run_scenario(dets)
        assert t.threat_level == "ATTENTION"

    def test_chair_very_close_is_warning(self):
        """Chair very close (static obstacle) → WARNING."""
        dets = make_dets(5, name="chair", bbox=(620, 400, 60, 80),
                         zone="center", distance="very_close", depth_value=0.85,
                         confidence=0.7, is_gazed=False)
        t = run_scenario(dets)
        assert t.threat_level == "WARNING"


class TestCollisionRiskRange:
    """Verify risk scores are in valid range."""

    def test_risk_between_0_and_1(self):
        """Risk must always be in [0.0, 1.0]."""
        tracker = SimpleTracker()
        # Extreme case: huge approach speed
        det = FakeDet(name="truck", bbox=(640, 360, 200, 150), zone="center",
                      distance="very_close", depth_value=0.99, confidence=0.99,
                      is_gazed=False)
        for _ in range(10):
            tracks = tracker.update([det], frame_width=1280)
        t = tracks[0]
        assert 0.0 <= t.collision_risk <= 1.0

    def test_risk_zero_for_distant_backpack(self):
        """Backpack far away on the side → near-zero risk."""
        dets = make_dets(5, name="backpack", bbox=(50, 50, 20, 30),
                         zone="left", distance="far", depth_value=0.05,
                         confidence=0.5, is_gazed=False)
        t = run_scenario(dets)
        assert t.collision_risk < 0.10


class TestThresholdConsistency:
    """Verify threshold constants are ordered correctly."""

    def test_thresholds_ordered(self):
        assert THREAT_THRESHOLDS["DANGER"] > THREAT_THRESHOLDS["WARNING"]
        assert THREAT_THRESHOLDS["WARNING"] > THREAT_THRESHOLDS["ATTENTION"]
        assert THREAT_THRESHOLDS["ATTENTION"] > 0.0
