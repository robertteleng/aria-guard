"""Tests for Dashboard rendering (gaze overlay).

The gaze point is mirrored horizontally for DISPLAY only: the gaze model
returns the correct x, but on the dashboard the marker landed on the opposite
side (left<->right). These tests pin that horizontal flip and guard against a
regression where the mirror is removed.
"""
import numpy as np

from src.output.dashboard import Dashboard, compose_dashboard_frame

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


class TestGazeRendering:
    def test_gaze_x_follows_model(self):
        """Model x=0.2 is drawn at ~0.2*w (no horizontal mirror)."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (0.2, 0.5))
        cx, _ = _marker_center(out)
        w = out.shape[1]
        assert cx < w * 0.5, f"marker should be on the left (cx={cx})"
        assert abs(cx - 0.2 * w) < 20, f"marker should be near 0.2*w (cx={cx})"

    def test_gaze_x_right_stays_right(self):
        """A right gaze (0.85) is drawn near 0.85*w (right half)."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (0.85, 0.5))
        cx, _ = _marker_center(out)
        w = out.shape[1]
        assert cx > w * 0.5, f"marker should be on the right (cx={cx})"
        assert abs(cx - 0.85 * w) < 20, f"marker should be near 0.85*w (cx={cx})"

    def test_gaze_y_follows_model(self):
        """Vertical coordinate tracks the model y directly (no flip)."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (0.5, 0.25))
        _, cy = _marker_center(out)
        h = out.shape[0]
        assert abs(cy - 0.25 * h) < 20, f"y should be ~0.25*h (cy={cy})"

    def test_gaze_out_of_range_is_clamped(self):
        """Out-of-range gaze (x>1, y<0) is clamped to the frame, no off-frame draw."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        out = Dashboard()._draw_gaze(frame, (1.5, -0.2))  # would draw off-frame unclamped
        cx, cy = _marker_center(out)
        w, h = out.shape[1], out.shape[0]
        assert cx > w * 0.5, f"x should clamp to the right edge (cx={cx})"
        assert cy < h * 0.5, f"y should clamp to the top (cy={cy})"


class TestDashboardComposite:
    """The /dashboard_feed composite frame (all panels tiled into one image)."""

    def test_full_grid_shape(self):
        out = compose_dashboard_frame(
            {
                "rgb": np.full((100, 200, 3), 50, dtype=np.uint8),
                "depth": np.full((80, 80, 3), 60, dtype=np.uint8),
                "eye": np.zeros((40, 120), dtype=np.uint8),   # grayscale → must convert
                "slam1": np.zeros((480, 640), dtype=np.uint8),
                "slam2": None,
            },
            status_lines=["FPS det: 8.1", "Latencia: 120 ms"], tw=320, th=240,
        )
        assert out.shape == (480, 960, 3)  # 2 rows x 3 cols of 240x320
        assert out.dtype == np.uint8

    def test_missing_panels_no_crash(self):
        out = compose_dashboard_frame({}, status_lines=None, tw=100, th=80)
        assert out.shape == (160, 300, 3)  # all-black placeholders, still a valid grid

    def test_grayscale_panel_is_converted_to_bgr(self):
        out = compose_dashboard_frame({"rgb": np.zeros((50, 50), dtype=np.uint8)}, tw=100, th=80)
        assert out.ndim == 3 and out.shape[2] == 3
