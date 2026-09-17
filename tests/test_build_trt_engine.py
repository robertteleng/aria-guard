"""Tests for the shape parsing of scripts/build_trt_engine.py (no TensorRT needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_trt_engine import parse_shape


def test_parses_name_and_dims():
    assert parse_shape("pixel_values:1x3x518x518") == ("pixel_values", (1, 3, 518, 518))


def test_name_may_contain_colons():
    assert parse_shape("input:0:1x2") == ("input:0", (1, 2))


def test_uppercase_separator():
    assert parse_shape("eye_images:1X2X240X320") == ("eye_images", (1, 2, 240, 320))


@pytest.mark.parametrize("spec", ["1x3x518x518", "name:", ":1x3", "name:1xax3", "name:1x0x3", "name:1x-1"])
def test_rejects_malformed(spec):
    with pytest.raises(ValueError):
        parse_shape(spec)
