"""Tests for src/evaluation/geometry.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.geometry import (bearing_deg, merge_episodes, point_to_polyline_2d, quat_to_matrix,
                                     raw_to_upright_pixel, ray_ground_intersection, upright_to_raw_pixel)


class TestPixelRotation:
    def test_matches_numpy_rot90_clockwise(self):
        n = 5
        raw = np.arange(n * n).reshape(n, n)
        upright = np.rot90(raw, -1)
        for (ux, uy) in [(0, 0), (4, 0), (1, 3), (2, 2)]:
            rx, ry = upright_to_raw_pixel(ux, uy, n)
            assert upright[uy, ux] == raw[int(ry), int(rx)]

    def test_roundtrip(self):
        for x, y in [(0.0, 0.0), (100.5, 1300.25), (1407, 3)]:
            assert raw_to_upright_pixel(*upright_to_raw_pixel(x, y, 1408), 1408) == pytest.approx((x, y))


class TestQuaternion:
    def test_identity(self):
        np.testing.assert_allclose(quat_to_matrix(0, 0, 0, 1), np.eye(3), atol=1e-12)

    def test_90deg_about_z(self):
        s = np.sqrt(0.5)
        r = quat_to_matrix(0, 0, s, s)
        np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


class TestRayGround:
    def test_hits_ground_ahead(self):
        hit = ray_ground_intersection(np.array([0, 0, 1.6]), np.array([1, 0, -0.4]), 0.0, 15)
        np.testing.assert_allclose(hit, [4.0, 0, 0.0])

    def test_upward_ray_misses(self):
        assert ray_ground_intersection(np.array([0, 0, 1.6]), np.array([1, 0, 0.1]), 0.0, 15) is None

    def test_beyond_range(self):
        assert ray_ground_intersection(np.array([0, 0, 1.6]), np.array([1, 0, -0.01]), 0.0, 15) is None

    def test_ground_above_origin_behind(self):
        assert ray_ground_intersection(np.array([0, 0, 1.0]), np.array([1, 0, -1]), 2.0, 15) is None


class TestPolyline:
    PATH = np.array([[0, 0], [2, 0], [4, 0]])

    def test_distance_and_position(self):
        d, i, t = point_to_polyline_2d(np.array([3, 0.5]), self.PATH)
        assert d == pytest.approx(0.5) and i == 1 and t == pytest.approx(0.5)

    def test_beyond_end_clamps(self):
        d, i, t = point_to_polyline_2d(np.array([5, 0]), self.PATH)
        assert d == pytest.approx(1.0) and i == 1 and t == 1.0

    def test_single_point_path(self):
        d, _, _ = point_to_polyline_2d(np.array([3, 4]), np.array([[0, 0]]))
        assert d == pytest.approx(5.0)

    def test_degenerate_segment(self):
        d, _, _ = point_to_polyline_2d(np.array([1, 1]), np.array([[0, 0], [0, 0], [2, 0]]))
        assert d == pytest.approx(1.0)


def test_bearing():
    assert bearing_deg([0, 0], [1, 0], [1, 1]) == pytest.approx(45)
    assert bearing_deg([0, 0], [1, 0], [-1, 0]) == pytest.approx(180)
    assert bearing_deg([0, 0], [0, 0], [1, 0]) == 0.0


class TestEpisodes:
    def test_merges_short_gaps_and_tracks_closest(self):
        samples = [
            (7, 0.0, True, 0.6, 1.5), (7, 0.1, True, 0.3, 1.6), (7, 0.3, True, 0.5, 1.7),
            (7, 2.0, True, 0.7, 3.0),                      # gap 1.7 s -> new episode
            (8, 0.2, False, 2.0, 0.0),                     # never in path
        ]
        eps = merge_episodes(samples, max_gap_s=0.5)
        assert [(e.track_id, e.start, e.end) for e in eps] == [(7, 0.0, 0.3), (7, 2.0, 2.0)]
        assert eps[0].closest_distance == 0.3 and eps[0].closest_time == 1.6

    def test_empty(self):
        assert merge_episodes([], 0.5) == []
