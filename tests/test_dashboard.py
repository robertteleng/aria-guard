"""Tests for Dashboard rendering (gaze overlay).

The gaze point is mirrored horizontally for DISPLAY only: the gaze model
returns the correct x, but on the dashboard the marker landed on the opposite
side (left<->right). These tests pin that horizontal flip and guard against a
regression where the mirror is removed.
"""
import numpy as np

from src.output.dashboard import Dashboard

# Gaze marker color drawn by _draw_gaze (BGR magenta)
_GAZE_BGR = (255, 0, 255)


def _gaze_mask(frame: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels painted with the gaze marker color."""
    return (
        (frame[:, :, 0] == _GAZE_BGR[0])
        & (frame[:, :, 1] == _GAZE_BGR[1])
        & (frame[:, :, 2] == _GAZE_BGR[2])
    )


def _marker_center(frame: np.ndarray):
    mask = _gaze_mask(frame)
    assert mask.any(), "no gaze marker was drawn"
    xs = np.where(mask.any(axis=0))[0]
    ys = np.where(mask.any(axis=1))[0]
    return (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2


class TestGazeMirror:
    def test_gaze_x_mirrored_to_opposite_half(self):
        """Model x=0.2 (left) must be drawn near the RIGHT (0.8*w)."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (0.2, 0.5))
        cx, _ = _marker_center(out)
        w = out.shape[1]
        assert cx > w * 0.5, f"marker not mirrored to right half (cx={cx})"
        assert abs(cx - 0.8 * w) < 20, f"marker not near 0.8*w (cx={cx})"

    def test_gaze_x_mirror_is_symmetric(self):
        """A right-side gaze (0.85) must land on the LEFT (~0.15*w)."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (0.85, 0.5))
        cx, _ = _marker_center(out)
        w = out.shape[1]
        assert cx < w * 0.5, f"marker not mirrored to left half (cx={cx})"
        assert abs(cx - 0.15 * w) < 20, f"marker not near 0.15*w (cx={cx})"

    def test_gaze_y_not_mirrored(self):
        """Vertical coordinate is NOT flipped — only the horizontal axis is."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (0.5, 0.25))
        _, cy = _marker_center(out)
        h = out.shape[0]
        assert abs(cy - 0.25 * h) < 20, f"y should track input (~0.25*h), got cy={cy}"
