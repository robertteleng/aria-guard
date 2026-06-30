"""Tests for ego-motion compensation in SimpleTracker.

While the user walks, every object ahead appears to "approach" (depth grows)
even when it is static — the dominant source of false DANGER alerts
("alertas mediocres"). The tracker subtracts EGO_MOTION_WALKING_BIAS from the
apparent approach when motion_state == "walking", so static obstacles don't
read as collisions while real fast approachers still trigger DANGER.
"""
import pytest

from src.domain.tracker import (
    SimpleTracker,
    THREAT_THRESHOLDS,
    EGO_MOTION_WALKING_BIAS,
)


class FakeDet:
    """Minimal detection for testing."""

    def __init__(self, **kw):
        kw.setdefault("traffic_light_state", None)
        self.__dict__.update(kw)


def run_scenario(dets_list, motion_state="unknown", frame_width=1280):
    tracker = SimpleTracker()
    tracks = []
    for dets in dets_list:
        tracks = tracker.update(dets, frame_width=frame_width, motion_state=motion_state)
    return tracks[0] if tracks else None


def car_approaching(depth0=0.3, step=0.05, n=5, zone="center"):
    """A car in `zone` whose depth grows `step`/frame (apparent approach).

    bbox height is constant (80px) so looming is 0 and depth drives the approach.
    """
    out = []
    for i in range(n):
        out.append([FakeDet(
            name="car", bbox=(600, 300, 100, 80), zone=zone,
            distance="close" if i >= 3 else "medium",
            depth_value=depth0 + i * step, confidence=0.85, is_gazed=False,
        )])
    return out


def car_looming(depth_value=0.5, h0=60, hstep=10, n=5, zone="center"):
    """A car with CONSTANT depth but a GROWING bbox height (pure looming cue)."""
    out = []
    for i in range(n):
        out.append([FakeDet(
            name="car", bbox=(600, 300, 100, h0 + i * hstep), zone=zone,
            distance="close" if i >= 3 else "medium",
            depth_value=depth_value, confidence=0.85, is_gazed=False,
        )])
    return out


class TestEgoMotionCompensation:
    def test_walking_toward_static_object_is_not_danger(self):
        """Case A: walking + apparent approach purely from self-motion → NOT DANGER."""
        t = run_scenario(car_approaching(step=0.05), motion_state="walking")
        assert t.threat_level != "DANGER"
        assert t.collision_risk < THREAT_THRESHOLDS["DANGER"]

    def test_stationary_object_approaching_is_danger(self):
        """Case B: stationary user + object closing in → DANGER (must still fire)."""
        t = run_scenario(car_approaching(step=0.05), motion_state="stationary")
        assert t.threat_level == "DANGER"
        assert t.collision_risk >= THREAT_THRESHOLDS["DANGER"]

    def test_walking_scores_strictly_lower_than_stationary(self):
        """Same scene: walking must reduce the collision risk."""
        scene = car_approaching(step=0.05)
        walking = run_scenario(scene, motion_state="walking")
        stationary = run_scenario(scene, motion_state="stationary")
        assert walking.collision_risk < stationary.collision_risk

    def test_fast_real_approach_survives_walking(self):
        """Safety: an approacher closing faster than the walk stays DANGER while walking."""
        t = run_scenario(car_approaching(depth0=0.2, step=0.12), motion_state="walking")
        assert t.threat_level == "DANGER"

    def test_unknown_motion_is_backwards_compatible(self):
        """Default motion_state applies no compensation (identical to stationary)."""
        scene = car_approaching(step=0.05)
        unknown = run_scenario(scene, motion_state="unknown")
        stationary = run_scenario(scene, motion_state="stationary")
        assert unknown.collision_risk == stationary.collision_risk
        assert unknown.threat_level == "DANGER"

    def test_approach_speed_reduced_by_exactly_the_bias(self):
        """Walking approach_speed = stationary slope minus the bias (positive slope only)."""
        scene = car_approaching(step=0.05)
        walking = run_scenario(scene, motion_state="walking")
        stationary = run_scenario(scene, motion_state="stationary")
        assert stationary.approach_speed == pytest.approx(0.05, abs=1e-6)
        assert walking.approach_speed == pytest.approx(0.05 - EGO_MOTION_WALKING_BIAS, abs=1e-6)

    def test_walking_does_not_invert_receding_object(self):
        """An object moving AWAY (negative slope) is untouched by the bias."""
        receding = []
        for i in range(5):
            receding.append([FakeDet(
                name="car", bbox=(600, 300, 100, 80), zone="center",
                distance="far", depth_value=0.6 - i * 0.05, confidence=0.85,
                is_gazed=False,
            )])
        walking = run_scenario(receding, motion_state="walking")
        # slope is negative; compensation only touches positive slopes
        assert walking.approach_speed < 0
        assert not walking.is_approaching


class TestBboxLooming:
    """bbox-height growth is a depth-independent approach (looming) cue."""

    def test_growing_bbox_detects_approach_with_flat_depth(self):
        """Constant depth but a growing bbox → approach detected via looming."""
        t = run_scenario(car_looming(), motion_state="stationary")
        assert t.looming_speed > 0.01
        assert t.is_approaching
        # depth is flat, so the fused approach comes from looming
        assert t.approach_speed == pytest.approx(t.looming_speed, abs=1e-9)

    def test_static_bbox_has_zero_looming(self):
        """Constant bbox height → no looming; depth alone drives the approach."""
        t = run_scenario(car_approaching(step=0.05), motion_state="stationary")
        assert t.looming_speed == pytest.approx(0.0, abs=1e-9)
        assert t.approach_speed == pytest.approx(0.05, abs=1e-6)

    def test_fusion_takes_the_stronger_cue(self):
        """When looming exceeds the depth slope, the fused approach uses looming."""
        # depth grows slowly (0.01/frame) but bbox looms strongly
        scene = []
        for i in range(5):
            scene.append([FakeDet(
                name="car", bbox=(600, 300, 100, 60 + i * 15), zone="center",
                distance="close", depth_value=0.4 + i * 0.01, confidence=0.85,
                is_gazed=False,
            )])
        t = run_scenario(scene, motion_state="stationary")
        assert t.looming_speed > 0.01
        assert t.approach_speed == pytest.approx(t.looming_speed, abs=1e-9)
