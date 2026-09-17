"""Tests for the TensorRT-free parts of src/detection/trt_yolo.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.trt_yolo import (PAD_VALUE, filter_end2end, letterbox, letterbox_params,
                                    read_ultralytics_engine, scale_boxes_to_frame)


class TestLetterbox:
    def test_square_aria_frame_has_no_padding(self):
        r, unpad, pad = letterbox_params((1408, 1408), (640, 640))
        assert r == pytest.approx(640 / 1408)
        assert unpad == (640, 640) and pad == (0, 0, 0, 0)

    def test_landscape_frame_pads_top_and_bottom(self):
        r, unpad, (left, top, right, bottom) = letterbox_params((720, 1280), (640, 640))
        assert unpad == (640, 360)
        assert (left, right) == (0, 0) and top + bottom == 280 and top == 140

    def test_output_shape_and_pad_value(self):
        img = letterbox(np.zeros((720, 1280, 3), np.uint8), (640, 640))
        assert img.shape == (640, 640, 3)
        assert (img[0, 0] == PAD_VALUE).all() and (img[320, 320] == 0).all()


class TestFilterEnd2End:
    PRED = np.array([
        [0, 0, 10, 10, 0.90, 2],
        [0, 0, 10, 10, 0.30, 0],   # below threshold
        [0, 0, 10, 10, 0.80, 0],
        [0, 0, 10, 10, 0.70, 5],
    ], dtype=np.float32)

    def test_confidence_then_max_det_then_classes(self):
        # Ultralytics order: max_det applies BEFORE the class filter
        out = filter_end2end(self.PRED, conf=0.4, max_det=2, classes=[0, 5])
        assert out[:, 5].tolist() == [0]

    def test_no_class_filter(self):
        out = filter_end2end(self.PRED, conf=0.4, max_det=20)
        assert out[:, 4].tolist() == pytest.approx([0.9, 0.8, 0.7])

    def test_strictly_greater_than_conf(self):
        assert len(filter_end2end(self.PRED, conf=0.9, max_det=20)) == 0


class TestScaleBoxes:
    def test_inverse_of_letterbox_for_landscape(self):
        frame_hw, input_hw = (720, 1280), (640, 640)
        r, _, (left, top, _, _) = letterbox_params(frame_hw, input_hw)
        frame_box = np.array([[100.0, 200.0, 500.0, 600.0]])
        input_box = frame_box * r + [left, top, left, top]
        det = np.hstack([input_box, [[0.9, 1]]]).astype(np.float32)
        back = scale_boxes_to_frame(det, input_hw, frame_hw)
        np.testing.assert_allclose(back[:, :4], frame_box, atol=1e-3)
        assert back[0, 4] == pytest.approx(0.9) and back[0, 5] == 1

    def test_clips_to_frame(self):
        det = np.array([[-20, -5, 700, 650, 0.5, 0]], dtype=np.float32)
        back = scale_boxes_to_frame(det, (640, 640), (1408, 1408))
        assert back[0, 0] == 0 and back[0, 1] == 0 and back[0, 2] == 1408 and back[0, 3] == 1408


class TestReadEngine:
    def test_splits_metadata_and_engine(self, tmp_path):
        meta = {"names": {"0": "person"}, "imgsz": [640, 640], "end2end": True}
        blob = json.dumps(meta).encode()
        f = tmp_path / "m.engine"
        f.write_bytes(len(blob).to_bytes(4, "little") + blob + b"ENGINEBYTES")
        got_meta, engine = read_ultralytics_engine(f)
        assert got_meta == meta and engine == b"ENGINEBYTES"

    def test_engine_without_metadata_is_rejected(self, tmp_path):
        f = tmp_path / "raw.engine"
        f.write_bytes(b"\xff\xfe\x00\x01" + b"\x80" * 64)
        with pytest.raises(ValueError):
            read_ultralytics_engine(f)
