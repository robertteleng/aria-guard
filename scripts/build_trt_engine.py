#!/usr/bin/env python3
"""
Build a TensorRT engine from an ONNX file on the machine that will run it.

Engines are not portable across GPUs or TensorRT versions, so the ONNX files
travel and each device builds its own engine (the RTX with TensorRT 10.16, the
Jetson with 10.3).

Usage:
    python scripts/build_trt_engine.py models/gaze.onnx models/gaze.engine
    python scripts/build_trt_engine.py models/depth_anything_v2_vits.onnx \\
        models/depth_anything_v2_vits.engine --shape pixel_values:1x3x518x518
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple


def parse_shape(spec: str) -> Tuple[str, Tuple[int, ...]]:
    """'name:1x3x518x518' -> ('name', (1, 3, 518, 518))."""
    name, sep, dims = spec.rpartition(":")
    if not sep or not name or not dims:
        raise ValueError(f"expected name:AxBxC, got {spec!r}")
    try:
        shape = tuple(int(d) for d in dims.lower().split("x"))
    except ValueError:
        raise ValueError(f"non-integer dimension in {spec!r}") from None
    if any(d <= 0 for d in shape):
        raise ValueError(f"dimensions must be positive in {spec!r}")
    return name, shape


def build(onnx_path: Path, engine_path: Path, shapes: Dict[str, Tuple[int, ...]],
          fp16: bool = True, workspace_mb: int = 1024) -> int:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves external weights (model.onnx.data) next to the file
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"cannot parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    dynamic = [network.get_input(i) for i in range(network.num_inputs)
               if any(d < 0 for d in network.get_input(i).shape)]
    if dynamic:
        missing = [t.name for t in dynamic if t.name not in shapes]
        if missing:
            raise ValueError(f"dynamic inputs need --shape: {', '.join(missing)}")
        profile = builder.create_optimization_profile()
        for t in dynamic:
            profile.set_shape(t.name, shapes[t.name], shapes[t.name], shapes[t.name])
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")
    data = bytes(serialized)
    engine_path.write_bytes(data)
    print(f"[TRT {trt.__version__}] {onnx_path.name} -> {engine_path} "
          f"({len(data) / 1e6:.1f} MB, fp16={fp16})")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--shape", action="append", default=[],
                        help="Static shape for a dynamic input, name:AxBxC (repeatable)")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    args = parser.parse_args()
    try:
        shapes = dict(parse_shape(s) for s in args.shape)
        build(args.onnx, args.engine, shapes, fp16=not args.fp32, workspace_mb=args.workspace_mb)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
