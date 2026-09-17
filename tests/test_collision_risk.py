"""Tests for collision_risk scoring (H18).

Validates that the 4-factor model produces correct threat levels
for known scenarios.
"""
import pytest
from src.domain.tracker import SimpleTracker, TrackedObject, THREAT_THRESHOLDS


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


class TestTimeToCollisionUsesRemainingDistance:
    """depth_value is PROXIMITY (0 = far, 1 = close): the gap left to close is
    1 - depth_value. Dividing proximity itself by the approach speed made a
    nearly-touching object look further in time than a distant one."""

    @staticmethod
    def approaching(start, step=0.02, name="person", zone="center"):
        return [
            [FakeDet(name=name, bbox=(600, 300, 50, 100), zone=zone, distance="medium",
                     depth_value=start + i * step, confidence=0.8, is_gazed=False)]
            for i in range(5)
        ]

    def test_closer_object_has_higher_risk_at_same_approach_speed(self):
        near = run_scenario(self.approaching(0.80))
        far = run_scenario(self.approaching(0.10))
        assert near.approach_speed == pytest.approx(far.approach_speed)
        assert near.collision_risk > far.collision_risk

    def test_ttc_factor_matches_remaining_gap(self):
        # proximity 0.88 closing 0.02/frame -> 0.12 left -> TTC 6 frames -> factor 0.96
        t = run_scenario(self.approaching(0.80))
        from src.domain.tracker import SimpleTracker
        expected_ttc_factor = 1.0 - ((1.0 - t.depth_value) / t.approach_speed) / 150.0
        assert expected_ttc_factor == pytest.approx(0.96, abs=1e-6)
        # risk = ttc*0.5 + cbdr*0.25 + zone*0.15 + class*0.10, cbdr = 1 * min(1, 0.02/0.03)
        from src.domain.tracker import CLASS_RISK
        expected = 0.96 * 0.5 + (0.02 / 0.03) * 0.25 + 1.0 * 0.15 + CLASS_RISK["person"] * 0.10
        assert t.collision_risk == pytest.approx(min(1.0, expected), abs=1e-6)

    def test_looming_speed_gives_time_to_contact_of_bbox_growth(self):
        # Constant proximity, bbox height growing 10 %/frame of its mean: the
        # looming cue alone must yield TTC = 1 / relative growth = 10 frames.
        tracker = SimpleTracker()
        heights = [90, 100, 110]
        for h in heights:
            tracks = tracker.update([FakeDet(name="person", bbox=(600, 300, 50, h), zone="center",
                                             distance="close", depth_value=0.6, confidence=0.8,
                                             is_gazed=False)], frame_width=1280)
        t = tracks[0]
        assert t.looming_speed == pytest.approx(0.1 * (1.0 - 0.6))
        assert (1.0 - t.depth_value) / t.looming_speed == pytest.approx(10.0)


class TestRemainingGap:
    def test_complement_of_proximity(self):
        from src.domain.tracker import remaining_gap
        assert remaining_gap(0.0) == 1.0
        assert remaining_gap(0.7) == pytest.approx(0.3)

    def test_floor_at_camera(self):
        from src.domain.tracker import MIN_REMAINING_GAP, remaining_gap
        assert remaining_gap(1.0) == MIN_REMAINING_GAP
        assert remaining_gap(1.2) == MIN_REMAINING_GAP

    def test_object_touching_and_approaching_is_danger(self):
        dets = [
            [FakeDet(name="person", bbox=(600, 300, 50, 100), zone="center", distance="very_close",
                     depth_value=min(1.0, 0.92 + i * 0.03), confidence=0.8, is_gazed=False)]
            for i in range(5)
        ]
        assert run_scenario(dets).threat_level == "DANGER"
