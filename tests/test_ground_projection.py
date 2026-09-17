"""Tests for candidate change 1: metric in-path threat (docs/ALERT_EVALUATION.md)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.ground_projection import (LEVEL_RISK, ground_contact, horizontal_basis, metric_threat)
from src.domain.tracker import SimpleTracker
from src.input.aria_frames import GravityEstimator

UP = np.array([0.0, 0.0, 1.0])


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class TestHorizontalBasis:
    def test_level_camera(self):
        f, r = horizontal_basis(UP, np.array([1.0, 0, 0]))
        np.testing.assert_allclose(f, [1, 0, 0])
        np.testing.assert_allclose(r, [0, -1, 0])     # facing +x with z up, right is -y

    def test_pitched_down_camera_keeps_horizontal_heading(self):
        f, _ = horizontal_basis(UP, unit([1.0, 0, -0.6]))
        np.testing.assert_allclose(f, [1, 0, 0], atol=1e-12)

    def test_looking_straight_down_has_no_heading(self):
        assert horizontal_basis(UP, np.array([0, 0, -1.0])) is None


class TestGroundContact:
    def setup_method(self):
        self.f, self.r = horizontal_basis(UP, np.array([1.0, 0, 0]))

    def test_forward_distance_from_depression_angle(self):
        F, L = ground_contact(unit([1, 0, -0.4]), UP, self.f, self.r, eye_height=1.6)
        assert F == pytest.approx(4.0) and L == pytest.approx(0.0)

    def test_lateral_offset_sign(self):
        F, L = ground_contact(unit([1, -0.2, -0.4]), UP, self.f, self.r, eye_height=1.6)
        assert F == pytest.approx(4.0) and L == pytest.approx(0.8)   # to the right

    def test_ray_above_horizon_has_no_contact(self):
        assert ground_contact(unit([1, 0, 0.1]), UP, self.f, self.r) is None

    def test_behind_or_out_of_range(self):
        assert ground_contact(unit([-1, 0, -0.5]), UP, self.f, self.r) is None
        assert ground_contact(unit([1, 0, -0.01]), UP, self.f, self.r) is None   # 160 m away

    def test_tilted_up_vector_is_respected(self):
        tilt = unit([0.1, 0, 1.0])                  # accelerometer frame slightly rolled
        f, r = horizontal_basis(tilt, np.array([1.0, 0, 0]))
        ray = unit(f * 4.0 - tilt * 1.6)            # exactly 4 m ahead on the tilted ground
        F, L = ground_contact(ray, tilt, f, r, eye_height=1.6)
        assert F == pytest.approx(4.0) and L == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("contact,level", [
    ((1.0, 0.0), "DANGER"), ((1.5, 0.75), "DANGER"), ((2.0, -0.5), "WARNING"),
    ((3.0, 0.0), "WARNING"), ((4.9, 0.1), "ATTENTION"), ((5.1, 0.0), "NONE"),
    ((1.0, 0.76), "NONE"), (None, "NONE"),
])
def test_metric_threat_rules(contact, level):
    assert metric_threat(contact) == level


class TestGravityEstimator:
    def test_mean_over_window_points_up(self):
        g = GravityEstimator(window_s=1.0)
        assert g.up() is None
        for i in range(200):                        # 2 s at 100 Hz, bouncing steps
            g.update(int(i * 1e7), [0.0, 2.0 * np.sin(i), 9.81])
        np.testing.assert_allclose(g.up(), [0, 0, 1], atol=0.05)

    def test_old_samples_leave_the_window(self):
        g = GravityEstimator(window_s=1.0)
        for i in range(100):
            g.update(int(i * 1e7), [9.81, 0, 0])    # first second: x up
        for i in range(100, 250):
            g.update(int(i * 1e7), [0, 0, 9.81])    # then z up
        np.testing.assert_allclose(g.up(), [0, 0, 1], atol=1e-9)


class Det:
    def __init__(self, **kw):
        kw.setdefault("traffic_light_state", None)
        self.__dict__.update(kw)


def test_tracker_uses_metric_threat_when_present():
    tracker = SimpleTracker()
    det = Det(name="car", bbox=(100, 400, 80, 50), zone="left", distance="far", depth_value=0.1,
              confidence=0.9, is_gazed=False, metric_threat="DANGER")
    tracks = tracker.update([det], frame_width=1408, fov_h=1.919)
    assert tracks[0].threat_level == "DANGER" and tracks[0].collision_risk == LEVEL_RISK["DANGER"]
    det.metric_threat = "NONE"
    tracks = tracker.update([det], frame_width=1408, fov_h=1.919)
    assert tracks[0].threat_level == "NONE" and tracks[0].collision_risk == 0.0


def test_tracker_without_metric_threat_keeps_heuristic():
    tracker = SimpleTracker()
    det = Det(name="car", bbox=(100, 400, 80, 50), zone="left", distance="far", depth_value=0.1,
              confidence=0.9, is_gazed=True)
    t = tracker.update([det], frame_width=1408, fov_h=1.919)[0]
    assert t.metric_threat is None and t.threat_level == "ATTENTION"   # same as the existing scenario
