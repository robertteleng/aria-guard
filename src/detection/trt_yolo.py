"""
YOLO26 (end-to-end, NMS-free) on a TensorRT engine exported by Ultralytics, without
the Ultralytics predictor.

The predictor wraps a ~4 ms engine in Python pre/post-processing that, on the
Jetson Orin Nano, cost ~20 ms per frame. This runs the same steps with the
image math on the GPU and returns plain (N, 6) arrays: x1, y1, x2, y2, conf, cls
in frame pixels. The letterbox, filtering and box scaling reproduce
Ultralytics 8.4 (LetterBox(center=True), nms end2end branch, scale_boxes).
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox_params(shape_hw: Tuple[int, int], new_hw: Tuple[int, int]):
    """Scale ratio, unpadded size (w, h) and (left, top, right, bottom) padding."""
    h, w = shape_hw
    r = min(new_hw[0] / h, new_hw[1] / w)
    new_unpad = (round(w * r), round(h * r))
    dw, dh = (new_hw[1] - new_unpad[0]) / 2, (new_hw[0] - new_unpad[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return r, new_unpad, (left, top, right, bottom)


def letterbox(image: np.ndarray, new_hw: Tuple[int, int]) -> np.ndarray:
    _, new_unpad, (left, top, right, bottom) = letterbox_params(image.shape[:2], new_hw)
    if (image.shape[1], image.shape[0]) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    if left or top or right or bottom:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=(PAD_VALUE,) * 3)
    return image


def filter_end2end(pred: np.ndarray, conf: float, max_det: int,
                   classes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Ultralytics end2end filtering: conf threshold, first max_det, then class filter."""
    out = pred[pred[:, 4] > conf][:max_det]
    if classes is not None:
        out = out[np.isin(out[:, 5], np.asarray(classes, dtype=out.dtype))]
    return out


def scale_boxes_to_frame(boxes: np.ndarray, input_hw: Tuple[int, int],
                         frame_hw: Tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from the letterboxed input back to frame pixels, clipped."""
    gain = min(input_hw[0] / frame_hw[0], input_hw[1] / frame_hw[1])
    pad_x = round((input_hw[1] - frame_hw[1] * gain) / 2 - 0.1)
    pad_y = round((input_hw[0] - frame_hw[0] * gain) / 2 - 0.1)
    b = boxes.copy()
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b[:, :4] /= gain
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, frame_hw[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, frame_hw[0])
    return b


def read_ultralytics_engine(path: Path) -> Tuple[dict, bytes]:
    """Split an Ultralytics .engine into its JSON metadata and the serialized engine."""
    data = Path(path).read_bytes()
    meta_len = int.from_bytes(data[:4], byteorder="little")
    try:
        meta = json.loads(data[4:4 + meta_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{path} has no embedded Ultralytics metadata (export it with Ultralytics)")
    return meta, data[4 + meta_len:]


class TrtYolo:
    """Callable detector: BGR frame -> (N, 6) float32 array in frame pixels."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import torch

        self._torch = torch
        meta, engine_bytes = read_ultralytics_engine(engine_path)
        if not meta.get("end2end", False):
            raise ValueError(f"{engine_path} is not an end-to-end (NMS-free) export")
        self.names: Dict[int, str] = {int(k): v for k, v in meta["names"].items()}
        self.input_hw = tuple(meta["imgsz"])

        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path} with TensorRT {trt.__version__}")
        self._context = self._engine.create_execution_context()

        self._inputs, self._outputs = {}, {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = torch.float16 if self._engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self._context.set_tensor_address(name, buf.data_ptr())
            target = self._inputs if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self._outputs
            target[name] = buf
        if len(self._inputs) != 1 or len(self._outputs) != 1:
            raise ValueError("expected one input and one output tensor")
        self._input = next(iter(self._inputs.values()))
        self._output = next(iter(self._outputs.values()))

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.25, max_det: int = 300,
                 classes: Optional[Sequence[int]] = None) -> np.ndarray:
        torch = self._torch
        img = letterbox(frame_bgr, self.input_hw)
        x = torch.from_numpy(img).cuda()                 # HWC uint8, BGR
        x = x[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)  # BCHW RGB
        self._input.copy_(x.to(self._input.dtype).div_(255.0))
        stream = torch.cuda.current_stream()
        self._context.execute_async_v3(stream.cuda_stream)
        pred = self._output[0].float().cpu().numpy()     # syncs this stream
        det = filter_end2end(pred, conf, max_det, classes)
        if len(det):
            det = scale_boxes_to_frame(det, self.input_hw, frame_bgr.shape[:2])
        return det.astype(np.float32, copy=False)
