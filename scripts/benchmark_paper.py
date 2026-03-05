#!/usr/bin/env python3
"""
Comprehensive benchmark for ARIA Guard paper.

Tests all pipeline components individually and combined:
- YOLO26s object detection (TensorRT vs PyTorch)
- Depth Anything V2 Small (TensorRT vs FP16)
- Meta Eye Gaze estimation
- Full pipeline (YOLO + Depth + Gaze)
- TTS latency (NeMo FastPitch+HiFi-GAN)

Output: formatted table ready for paper.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path

# Ensure CUDA is available
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import cv2


def get_system_info():
    """Collect system information."""
    info = {
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}" if torch.cuda.is_available() else "N/A",
        "opencv": cv2.__version__,
        "opencv_cuda": hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0,
    }
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except ImportError:
        info["tensorrt"] = "N/A"
    return info


def load_test_frames(video_path=None, n_frames=200):
    """Load test frames from video or generate synthetic ones."""
    frames = []

    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames from {video_path} ({frames[0].shape})")
    else:
        # Generate synthetic indoor scene frames (realistic resolution)
        print("No video found, generating synthetic 1080p frames...")
        for i in range(n_frames):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            frames.append(frame)
        print(f"Generated {len(frames)} synthetic frames (1080x1920)")

    return frames


def benchmark_yolo_tensorrt(frames, n_warmup=10):
    """Benchmark YOLO26s with TensorRT engine."""
    from ultralytics import YOLO

    engine_path = PROJECT_ROOT / "models" / "yolo26s.engine"
    if not engine_path.exists():
        return None, "Engine not found"

    model = YOLO(str(engine_path), task="detect")

    # Warmup
    for i in range(n_warmup):
        model(frames[i % len(frames)], verbose=False, half=True, conf=0.4)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    total_detections = 0
    for frame in frames:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model(frame, verbose=False, half=True, conf=0.4)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
        if results and results[0].boxes is not None:
            total_detections += len(results[0].boxes)

    return {
        "fps": 1000.0 / statistics.mean(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "avg_detections": total_detections / len(frames),
        "n_frames": len(frames),
        "backend": "TensorRT FP16",
    }, None


def benchmark_yolo_pytorch(frames, n_warmup=10):
    """Benchmark YOLO26s with PyTorch (CUDA FP16)."""
    from ultralytics import YOLO

    pt_path = PROJECT_ROOT / "models" / "yolo26s.pt"
    if not pt_path.exists():
        return None, "PT model not found"

    model = YOLO(str(pt_path), task="detect")
    model.to("cuda")

    # Warmup
    for i in range(n_warmup):
        model(frames[i % len(frames)], verbose=False, half=True, conf=0.4)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    total_detections = 0
    for frame in frames:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model(frame, verbose=False, half=True, conf=0.4)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        if results and results[0].boxes is not None:
            total_detections += len(results[0].boxes)

    return {
        "fps": 1000.0 / statistics.mean(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "avg_detections": total_detections / len(frames),
        "n_frames": len(frames),
        "backend": "PyTorch CUDA FP16",
    }, None


def benchmark_depth_tensorrt(frames, n_warmup=10):
    """Benchmark Depth Anything V2 with TensorRT."""
    import tensorrt as trt

    engine_path = PROJECT_ROOT / "models" / "depth_anything_v2_vits.engine"
    if not engine_path.exists():
        return None, "Depth engine not found"

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        return None, "Failed to deserialize depth engine"

    context = engine.create_execution_context()

    # Get IO info
    inputs = []
    outputs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        # Handle dynamic shapes
        if any(d == -1 for d in shape):
            shape = (1, 3, 518, 518) if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else (1, 1, 518, 518)
            context.set_input_shape(name, shape)
        size = 1
        for d in shape:
            size *= d
        mem = torch.empty(size, dtype=torch.float32, device="cuda")
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({"name": name, "shape": shape, "mem": mem})
        else:
            outputs.append({"name": name, "shape": shape, "mem": mem})

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (518, 518))
        img = rgb_small.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, 0).astype(np.float32)

    # Warmup
    for i in range(n_warmup):
        img = preprocess(frames[i % len(frames)])
        input_tensor = torch.from_numpy(np.ascontiguousarray(img)).cuda()
        inputs[0]["mem"][:input_tensor.numel()].copy_(input_tensor.flatten())
        for inp in inputs:
            context.set_tensor_address(inp["name"], inp["mem"].data_ptr())
        for out in outputs:
            context.set_tensor_address(out["name"], out["mem"].data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for frame in frames:
        img = preprocess(frame)
        input_tensor = torch.from_numpy(np.ascontiguousarray(img)).cuda()
        inputs[0]["mem"][:input_tensor.numel()].copy_(input_tensor.flatten())

        for inp in inputs:
            context.set_tensor_address(inp["name"], inp["mem"].data_ptr())
        for out in outputs:
            context.set_tensor_address(out["name"], out["mem"].data_ptr())

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "fps": 1000.0 / statistics.mean(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "input_size": "518x518",
        "n_frames": len(frames),
        "backend": "TensorRT FP16",
    }, None


def benchmark_depth_fp16(frames, n_warmup=10):
    """Benchmark Depth Anything V2 with HuggingFace FP16."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image

    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).cuda().half().eval()

    # Warmup
    for i in range(n_warmup):
        rgb = cv2.cvtColor(frames[i % len(frames)], cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (518, 518))
        inputs = processor(images=Image.fromarray(rgb_small), return_tensors="pt")
        inputs = {k: v.cuda().half() for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (518, 518))
        inputs = processor(images=Image.fromarray(rgb_small), return_tensors="pt")
        inputs = {k: v.cuda().half() for k, v in inputs.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "fps": 1000.0 / statistics.mean(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "input_size": "518x518",
        "n_frames": len(frames),
        "backend": "HuggingFace FP16",
    }, None


def benchmark_gaze(n_frames=200, n_warmup=10):
    """Benchmark Meta Eye Gaze model."""
    # Monkey-patch torch.load for legacy model
    _original = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original(*args, **kwargs)
    torch.load = _patched

    try:
        from projectaria_eyetracking.inference.infer import EyeGazeInference
        import projectaria_eyetracking.inference.infer as infer_module

        weights_dir = Path(os.path.dirname(infer_module.__file__)) / "model" / "pretrained_weights" / "social_eyes_uncertainty_v1"
        if not (weights_dir / "weights.pth").exists():
            return None, "Gaze weights not found"

        model = EyeGazeInference(
            model_checkpoint_path=str(weights_dir / "weights.pth"),
            model_config_path=str(weights_dir / "config.yaml"),
            device="cuda"
        )

        # Simulate eye tracking images (240x640 grayscale, typical Aria ET camera)
        fake_eyes = [np.random.randint(0, 255, (240, 640), dtype=np.uint8) for _ in range(n_frames)]

        # Warmup
        for i in range(n_warmup):
            img_tensor = torch.tensor(fake_eyes[i % n_frames], device="cuda")
            model.predict(img_tensor)
        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for eye_img in fake_eyes:
            img_tensor = torch.tensor(eye_img, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(img_tensor)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        return {
            "fps": 1000.0 / statistics.mean(latencies),
            "latency_mean_ms": statistics.mean(latencies),
            "latency_median_ms": statistics.median(latencies),
            "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "input_size": "640x240",
            "n_frames": n_frames,
            "backend": "PyTorch CUDA",
        }, None

    except ImportError:
        return None, "projectaria_eyetracking not installed"
    except Exception as e:
        return None, str(e)
    finally:
        torch.load = _original


def benchmark_full_pipeline(frames, n_warmup=10):
    """Benchmark full detection pipeline (YOLO + Depth + Gaze combined)."""
    from src.detection.detector import ParallelDetector

    detector = ParallelDetector(enable_depth=True, device="cuda", depth_interval=3, mode="indoor")

    # Generate fake eye frames
    fake_eyes = [np.random.randint(0, 255, (240, 640), dtype=np.uint8) for _ in range(len(frames))]

    # Warmup
    for i in range(n_warmup):
        detector.process(frames[i % len(frames)], eye_frame=fake_eyes[i % len(frames)])
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    total_detections = 0
    for i, frame in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        detections, depth_map, gaze_point, tracked = detector.process(
            frame, eye_frame=fake_eyes[i]
        )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        total_detections += len(detections)

    yolo_backend = "TensorRT" if detector._yolo_tensorrt else "PyTorch"
    depth_backend = "TensorRT" if detector._depth_tensorrt else "FP16"

    return {
        "fps": 1000.0 / statistics.mean(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "avg_detections": total_detections / len(frames),
        "n_frames": len(frames),
        "yolo_backend": yolo_backend,
        "depth_backend": depth_backend,
        "depth_interval": 3,
        "has_gaze": detector.gaze_model is not None,
    }, None


def benchmark_full_no_depth(frames, n_warmup=10):
    """Benchmark pipeline without depth (YOLO + Gaze only)."""
    from src.detection.detector import ParallelDetector

    detector = ParallelDetector(enable_depth=False, device="cuda", mode="indoor")

    fake_eyes = [np.random.randint(0, 255, (240, 640), dtype=np.uint8) for _ in range(len(frames))]

    # Warmup
    for i in range(n_warmup):
        detector.process(frames[i % len(frames)], eye_frame=fake_eyes[i % len(frames)])
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    total_detections = 0
    for i, frame in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        detections, _, gaze_point, tracked = detector.process(
            frame, eye_frame=fake_eyes[i]
        )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        total_detections += len(detections)

    return {
        "fps": 1000.0 / statistics.mean(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "avg_detections": total_detections / len(frames),
        "n_frames": len(frames),
        "backend": "TensorRT" if detector._yolo_tensorrt else "PyTorch",
    }, None


def get_vram_usage():
    """Get current VRAM usage."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    return {}


def print_results(results):
    """Print formatted results table for paper."""
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS FOR PAPER")
    print("=" * 80)

    # System info
    sys_info = results.get("system", {})
    print(f"\n--- System Configuration ---")
    print(f"GPU: {sys_info.get('gpu', 'N/A')}")
    print(f"VRAM: {sys_info.get('vram_gb', 'N/A')} GB")
    print(f"PyTorch: {sys_info.get('pytorch', 'N/A')}")
    print(f"CUDA: {sys_info.get('cuda', 'N/A')}")
    print(f"TensorRT: {sys_info.get('tensorrt', 'N/A')}")
    print(f"OpenCV CUDA: {sys_info.get('opencv_cuda', 'N/A')}")

    # Table header
    print(f"\n--- Performance Results ---")
    print(f"{'Component':<35} {'Backend':<20} {'FPS':>8} {'Latency (ms)':>14} {'P95 (ms)':>10}")
    print("-" * 90)

    for key, label in [
        ("yolo_trt", "YOLO26s Detection"),
        ("yolo_pytorch", "YOLO26s Detection"),
        ("depth_trt", "Depth Anything V2-S"),
        ("depth_fp16", "Depth Anything V2-S"),
        ("gaze", "Meta Eye Gaze"),
        ("full_pipeline", "Full Pipeline (YOLO+Depth+Gaze)"),
        ("full_no_depth", "Pipeline (YOLO+Gaze only)"),
    ]:
        data = results.get(key)
        if data is None:
            continue
        backend = data.get("backend", "")
        if key == "full_pipeline":
            backend = f"Y:{data.get('yolo_backend','?')} D:{data.get('depth_backend','?')}"
        fps = data.get("fps", 0)
        lat = data.get("latency_mean_ms", 0)
        p95 = data.get("latency_p95_ms", 0)
        print(f"{label:<35} {backend:<20} {fps:>8.1f} {lat:>14.2f} {p95:>10.2f}")

    # Extra details
    full = results.get("full_pipeline")
    if full:
        print(f"\n--- Full Pipeline Details ---")
        print(f"Depth interval: every {full.get('depth_interval', '?')} frames")
        print(f"Gaze model active: {full.get('has_gaze', False)}")
        print(f"Avg detections/frame: {full.get('avg_detections', 0):.1f}")

    # VRAM
    vram = results.get("vram", {})
    if vram:
        print(f"\n--- VRAM Usage ---")
        print(f"Peak allocated: {vram.get('max_allocated_mb', 0):.0f} MB")
        print(f"Reserved: {vram.get('reserved_mb', 0):.0f} MB")

    # Model sizes
    print(f"\n--- Model Sizes ---")
    models = [
        ("yolo26s.engine", "YOLO26s TensorRT FP16"),
        ("yolo26s.pt", "YOLO26s PyTorch"),
        ("depth_anything_v2_vits.engine", "Depth Anything V2-S TensorRT FP16"),
        ("depth_anything_v2_vits.onnx", "Depth Anything V2-S ONNX"),
    ]
    for fname, label in models:
        fpath = PROJECT_ROOT / "models" / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            print(f"{label:<40} {size_mb:>8.1f} MB")

    print()
    print("=" * 80)


def main():
    print("=" * 60)
    print("ARIA Guard - Comprehensive Benchmark")
    print("=" * 60)

    results = {}

    # System info
    print("\n[1/8] System info...")
    results["system"] = get_system_info()
    print(f"  GPU: {results['system']['gpu']}")
    print(f"  TensorRT: {results['system']['tensorrt']}")

    # Load test frames
    print("\n[2/8] Loading test frames...")
    video_paths = [
        PROJECT_ROOT / "data" / "test_60fps.mp4",
        PROJECT_ROOT / "data" / "test_video.mp4",
        PROJECT_ROOT / "data" / "sample.mp4",
    ]
    video_path = None
    for vp in video_paths:
        if vp.exists():
            video_path = vp
            break
    frames = load_test_frames(video_path, n_frames=200)

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # YOLO TensorRT
    print("\n[3/8] Benchmarking YOLO26s (TensorRT)...")
    data, err = benchmark_yolo_tensorrt(frames)
    if data:
        results["yolo_trt"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # YOLO PyTorch
    print("\n[4/8] Benchmarking YOLO26s (PyTorch)...")
    data, err = benchmark_yolo_pytorch(frames)
    if data:
        results["yolo_pytorch"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # Depth TensorRT
    print("\n[5/8] Benchmarking Depth Anything V2 (TensorRT)...")
    data, err = benchmark_depth_tensorrt(frames)
    if data:
        results["depth_trt"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # Depth FP16
    print("\n[6/8] Benchmarking Depth Anything V2 (FP16)...")
    data, err = benchmark_depth_fp16(frames)
    if data:
        results["depth_fp16"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # Gaze
    print("\n[7/8] Benchmarking Meta Eye Gaze...")
    data, err = benchmark_gaze()
    if data:
        results["gaze"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # Full pipeline
    print("\n[8/8] Benchmarking full pipeline...")
    data, err = benchmark_full_pipeline(frames)
    if data:
        results["full_pipeline"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # Also test without depth
    print("\n[bonus] Benchmarking pipeline without depth...")
    data, err = benchmark_full_no_depth(frames)
    if data:
        results["full_no_depth"] = data
        print(f"  -> {data['fps']:.1f} FPS, {data['latency_mean_ms']:.2f}ms avg")
    else:
        print(f"  -> SKIPPED: {err}")

    # VRAM
    results["vram"] = get_vram_usage()

    # Print formatted results
    print_results(results)

    # Save raw JSON
    out_path = PROJECT_ROOT / "experiments" / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
