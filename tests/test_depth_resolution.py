"""Depth stays at the model resolution: same values as the old full-frame path.

Needs torch (skipped in the lightweight test environment); runs on CPU.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import ParallelDetector, normalize_depth_u8  # noqa: E402


def test_gpu_normalization_matches_opencv_minmax():
    rng = np.random.default_rng(0)
    depth = (rng.random((518, 518)) * 7.3 + 1.2).astype(np.float32)
    expected = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    got = normalize_depth_u8(torch.from_numpy(depth)).numpy()
    assert got.dtype == np.uint8
    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 1


def test_constant_map_does_not_divide_by_zero():
    got = normalize_depth_u8(torch.full((4, 4), 3.0)).numpy()
    assert (got == 0).all()


def test_bbox_depth_is_resolution_independent():
    # Smooth proximity gradient; sample the same frame-space bbox on a
    # 518x518 map and on its 1408x1408 upscale.
    yy, xx = np.mgrid[0:518, 0:518].astype(np.float32)
    small = ((xx + yy) / (2 * 517) * 255).astype(np.uint8)
    big = cv2.resize(small, (1408, 1408))
    sample = ParallelDetector._get_depth_in_bbox
    for x1, y1, x2, y2 in [(100, 200, 400, 700), (900, 50, 1300, 400), (600, 600, 800, 1400)]:
        a = sample(None, small, x1, y1, x2, y2, 1408, 1408)
        b = sample(None, big, x1, y1, x2, y2, 1408, 1408)
        assert a == pytest.approx(b, abs=0.01)
