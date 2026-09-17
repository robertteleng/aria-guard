"""Tests for the frame and IMU transforms shared by live and VRS playback."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.input.aria_frames import MotionClassifier, eye_to_bgr, nearest_index, rgb_to_bgr_upright


class TestRgbToBgrUpright:
    def test_rotates_90_clockwise(self):
        # Mark the raw top-left pixel red; after 90 CW it lands top-right
        raw = np.zeros((4, 6, 3), dtype=np.uint8)
        raw[0, 0] = (255, 0, 0)  # RGB red
        out = rgb_to_bgr_upright(raw)
        assert out.shape == (6, 4, 3)
        assert tuple(out[0, -1]) == (0, 0, 255)  # BGR red, top-right
        assert out[0, 0].sum() == 0

    def test_output_is_contiguous(self):
        out = rgb_to_bgr_upright(np.zeros((8, 8, 3), dtype=np.uint8))
        assert out.flags["C_CONTIGUOUS"]

    def test_does_not_modify_input(self):
        raw = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        before = raw.copy()
        rgb_to_bgr_upright(raw)
        np.testing.assert_array_equal(raw, before)


class TestEyeToBgr:
    def test_gray_rotated_180_and_expanded(self):
        raw = np.zeros((2, 5), dtype=np.uint8)
        raw[0, 0] = 200
        out = eye_to_bgr(raw)
        assert out.shape == (2, 5, 3)
        assert tuple(out[-1, -1]) == (200, 200, 200)
        assert out[0, 0].sum() == 0

    def test_color_input_keeps_channels(self):
        raw = np.zeros((2, 3, 3), dtype=np.uint8)
        raw[0, 0] = (1, 2, 3)
        out = eye_to_bgr(raw)
        assert tuple(out[-1, -1]) == (1, 2, 3)


class TestNearestIndex:
    @pytest.mark.parametrize("target,expected", [(-5, 0), (0, 0), (4, 0), (6, 1), (10, 1), (29, 2), (100, 2)])
    def test_picks_closest(self, target, expected):
        assert nearest_index([0, 10, 30], target) == expected

    def test_tie_prefers_earlier(self):
        assert nearest_index([0, 10], 5) == 0

    def test_accepts_numpy_array(self):
        assert nearest_index(np.array([0, 10, 20], dtype=np.int64), 19) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nearest_index([], 1)


class TestMotionClassifier:
    def test_unknown_until_min_samples(self):
        m = MotionClassifier(min_samples=10)
        for _ in range(9):
            assert m.update((0.0, 0.0, 9.81)) == "unknown"
        assert m.update((0.0, 0.0, 9.81)) == "stationary"

    def test_walking_on_large_spread(self):
        m = MotionClassifier()
        for i in range(30):
            m.update((0.0, 0.0, 9.81 + (2.0 if i % 2 else -2.0)))
        assert m.state == "walking"

    def test_hysteresis_keeps_state_between_thresholds(self):
        m = MotionClassifier()
        for i in range(30):
            m.update((0.0, 0.0, 9.81 + (2.0 if i % 2 else -2.0)))
        # std 0.45: between 0.3 and 0.6, state must not change
        for i in range(20):
            m.update((0.0, 0.0, 9.81 + (0.45 if i % 2 else -0.45)))
        assert m.state == "walking"
