#!/usr/bin/env python3
"""
Replay an Aria VRS recording through the real aria-guard pipeline and measure it.

The recording reaches the detector the way live streaming would: RGB and eye
images get the live orientation, the eye frame is the one nearest in time, and
the IMU drives the same walking/stationary classifier. Detection runs in
process (ParallelDetector: YOLO + Depth Anything + gaze), then the tracker and
the alert arbiter run on recording time.

Pacing:
  realtime  after each frame, skip to the newest frame the glasses would have
            delivered by then (frames the pipeline is too slow for are dropped,
            as they are live). Measures effective FPS and alerts under real load.
  all       process every frame. Measures throughput and per-frame cost.

Usage:
    ARIA_YOLO_MODEL=yolo26n_nav python scripts/replay_vrs.py recording.vrs \\
        --mode outdoor --pace realtime --out benchmarks/replay/run.json

The JSON records the environment, the model hash and the command, so results
can be compared across machines.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_offline import FrameResult, run_benchmark  # noqa: E402
from src.input.aria_frames import MotionClassifier, eye_to_bgr, nearest_index, rgb_to_bgr_upright  # noqa: E402

ARIA_RGB_FOV_H = 1.919  # same value the Aria observers give the tracker
RGB_STREAM, EYE_STREAM, IMU_STREAM = "214-1", "211-1", "1202-1"


# --- pure helpers (unit-tested) -------------------------------------------------

def latency_summary(values_ms: Sequence[float]) -> Dict[str, float]:
    """mean / p50 / p95 / p99 / max of a list of milliseconds."""
    if len(values_ms) == 0:
        return {"n": 0}
    a = np.asarray(values_ms, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def next_realtime_index(timestamps_ns: Sequence[int], start_ts_ns: int,
                        elapsed_ns: int, current: int) -> int:
    """Index of the newest frame captured by start + elapsed, always > current.

    Returns len(timestamps_ns) when the recording is exhausted.
    """
    import bisect
    newest = bisect.bisect_right(timestamps_ns, start_ts_ns + elapsed_ns) - 1
    return max(newest, current + 1)


def wait_before_frame_ns(timestamps_ns: Sequence[int], start_ts_ns: int,
                        elapsed_ns: int, idx: int) -> int:
    """How long to wait until frame idx has been captured (0 if it already was).

    A pipeline faster than the camera must not read frames from the future.
    """
    if idx >= len(timestamps_ns):
        return 0
    return max(0, (timestamps_ns[idx] - start_ts_ns) - elapsed_ns)


def imu_decimation(imu_rate_hz: float, target_hz: float) -> int:
    """Keep one IMU sample every N so the classifier sees about target_hz."""
    if target_hz <= 0 or imu_rate_hz <= 0:
        raise ValueError("rates must be positive")
    return max(1, int(round(imu_rate_hz / target_hz)))


def detections_to_dicts(detections) -> List[dict]:
    return [{
        "name": d.name,
        "bbox": list(d.bbox),
        "zone": d.zone,
        "distance": d.distance,
        "depth_value": float(d.depth_value),
        "confidence": float(d.confidence),
        "is_gazed": bool(getattr(d, "is_gazed", False)),
        "traffic_light_state": getattr(d, "traffic_light_state", None),
    } for d in detections]


# --- recording --------------------------------------------------------------------

class VrsRecording:
    """RGB, eye and IMU access to one Aria VRS file, with live orientation."""

    def __init__(self, vrs_path: str, imu_target_hz: float):
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
        from projectaria_tools.core.stream_id import StreamId

        self.provider = data_provider.create_vrs_data_provider(vrs_path)
        if self.provider is None:
            raise RuntimeError(f"cannot open VRS: {vrs_path}")
        self._rgb, self._eye, self._imu = StreamId(RGB_STREAM), StreamId(EYE_STREAM), StreamId(IMU_STREAM)
        self.rgb_ts = list(self.provider.get_timestamps_ns(self._rgb, TimeDomain.DEVICE_TIME))
        self.eye_ts = list(self.provider.get_timestamps_ns(self._eye, TimeDomain.DEVICE_TIME))
        self.imu_ts = list(self.provider.get_timestamps_ns(self._imu, TimeDomain.DEVICE_TIME))
        imu_rate = self.provider.get_nominal_rate_hz(self._imu)
        self.imu_step = imu_decimation(imu_rate, imu_target_hz)
        self.imu_rate_hz = imu_rate
        self._motion = MotionClassifier()
        self._imu_next = 0

    def rgb_shape(self):
        cfg = self.provider.get_image_configuration(self._rgb)
        return (cfg.image_height, cfg.image_width, 3)

    def motion_state_at(self, ts_ns: int) -> str:
        """Feed the classifier every kept IMU sample up to ts_ns (monotonic calls)."""
        while self._imu_next < len(self.imu_ts) and self.imu_ts[self._imu_next] <= ts_ns:
            sample = self.provider.get_imu_data_by_index(self._imu, self._imu_next)
            self._motion.update(sample.accel_msec2)
            self._imu_next += self.imu_step
        return self._motion.state

    def read(self, idx: int):
        rgb_data = self.provider.get_image_data_by_index(self._rgb, idx)
        ts = rgb_data[1].capture_timestamp_ns
        rgb = rgb_to_bgr_upright(rgb_data[0].to_numpy_array())
        eye = None
        if self.eye_ts:
            eye_data = self.provider.get_image_data_by_index(self._eye, nearest_index(self.eye_ts, ts))
            eye = eye_to_bgr(eye_data[0].to_numpy_array())
        return ts, rgb, eye


class LiveFeed:
    """Decodes frames on their own thread at capture time, like the Aria SDK.

    Live, the SDK decodes each frame in its callback thread whether or not the
    detector is ready; the pipeline then takes the newest one. This thread owns
    the VRS provider for the whole run (it is not shared across threads): it
    also advances the IMU motion classifier to each frame's timestamp.
    """

    def __init__(self, rec: "VrsRecording", n_frames: int):
        import threading
        self._rec, self._n = rec, n_frames
        self._cond = threading.Condition()
        self._latest = None  # (idx, ts, rgb, eye, motion, decode_ms)
        self.decoded = 0
        self.skipped_by_decoder = 0
        self.finished = False
        self.start_ts = rec.rgb_ts[0]
        self.wall0 = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._run, name="vrs-feed", daemon=True)
        self._thread.start()

    def _run(self):
        idx = -1
        while True:
            nxt = next_realtime_index(self._rec.rgb_ts, self.start_ts, time.perf_counter_ns() - self.wall0, idx)
            if nxt >= self._n:
                break
            self.skipped_by_decoder += nxt - idx - 1
            wait = wait_before_frame_ns(self._rec.rgb_ts, self.start_ts, time.perf_counter_ns() - self.wall0, nxt)
            if wait:
                time.sleep(wait / 1e9)
            t0 = time.perf_counter()
            ts, rgb, eye = self._rec.read(nxt)
            motion = self._rec.motion_state_at(ts)
            decode_ms = (time.perf_counter() - t0) * 1000
            with self._cond:
                self._latest = (nxt, ts, rgb, eye, motion, decode_ms)
                self.decoded += 1
                self._cond.notify()
            idx = nxt
        with self._cond:
            self.finished = True
            self._cond.notify()

    def newest_after(self, last_idx: int):
        """Block until a frame newer than last_idx exists; None when the feed ended."""
        with self._cond:
            while not self.finished and (self._latest is None or self._latest[0] <= last_idx):
                self._cond.wait(timeout=1.0)
            if self._latest is not None and self._latest[0] > last_idx:
                return self._latest
            return None


# --- stage timing -----------------------------------------------------------------

def wrap_stage(obj, method: str, sink: Dict[str, List[float]], sync: Callable[[], None]):
    """Time obj.method into sink[method], synchronizing the GPU around the call.

    Synchronizing serializes the CUDA streams, so the breakdown slightly
    overstates the parallel pipeline; the end-to-end detector time is measured
    separately in a run without it.
    """
    original = getattr(obj, method)

    def timed(*args, **kwargs):
        sync()
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        sync()
        sink.setdefault(method, []).append((time.perf_counter() - t0) * 1000)
        return out

    setattr(obj, method, timed)


# --- environment ------------------------------------------------------------------

def _run(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_l4t_release(text: str) -> Optional[str]:
    """'# R36 (release), REVISION: 5.2, ...' -> 'R36.5.2'."""
    import re
    m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", text)
    return f"R{m.group(1)}.{m.group(2)}" if m else None


def package_version(module: str, distribution: str) -> Optional[str]:
    """Installed version: package metadata first, then module.__version__."""
    from importlib import metadata
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        pass
    try:
        return getattr(__import__(module), "__version__", None)
    except Exception:
        return None


def environment() -> dict:
    env = {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "commit": _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"]),
        "dirty": bool(_run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"])),
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "jetson_l4t": (parse_l4t_release(Path("/etc/nv_tegra_release").read_text())
                       if Path("/etc/nv_tegra_release").exists() else None),
        # nvpmodel is not available inside containers: the launcher passes it
        "nvpmodel": _run(["nvpmodel", "-q"]) or os.environ.get("JETSON_POWER_MODE"),
    }
    for mod, dist in (("torch", "torch"), ("tensorrt", "tensorrt"), ("ultralytics", "ultralytics"),
                      ("cv2", "opencv-python"), ("projectaria_tools", "projectaria-tools")):
        env[mod] = package_version(mod, dist)
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


# --- main -------------------------------------------------------------------------

def replay(args) -> dict:
    import torch
    from src.detection.detector import MODELS_DIR, ParallelDetector

    rec = VrsRecording(args.vrs, imu_target_hz=args.imu_hz)
    n_frames = len(rec.rgb_ts)
    if args.max_seconds:
        limit = rec.rgb_ts[0] + int(args.max_seconds * 1e9)
        n_frames = sum(1 for t in rec.rgb_ts if t <= limit)
    print(f"[REPLAY] {args.vrs}: {n_frames} RGB frames, {len(rec.eye_ts)} eye, IMU {rec.imu_rate_hz:.0f} Hz (1/{rec.imu_step})")

    model_name = os.environ.get("ARIA_YOLO_MODEL", "yolo26s")
    detector = ParallelDetector(enable_depth=not args.no_depth, mode=args.mode, fov_h=ARIA_RGB_FOV_H)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    stage_ms: Dict[str, List[float]] = {}
    if args.breakdown:
        for method in ("_run_yolo", "_run_depth", "estimate_gaze", "_create_detections"):
            wrap_stage(detector, method, stage_ms, sync)

    # Warm-up on the first frames (engine allocation, cudnn autotune) — not measured
    for i in range(min(args.warmup, n_frames)):
        _, rgb, eye = rec.read(i)
        detector.process(rgb, eye)
    sync()
    stage_ms.clear()
    detector._frame_idx = 0  # restart the depth/gaze cadence for the measured run

    frames: List[FrameResult] = []
    decode_ms, detect_ms, loop_ms, age_ms = [], [], [], []
    start_ts = rec.rgb_ts[0]
    feed = None

    def record_frame(idx, ts, motion, detections):
        frames.append(FrameResult(frame_idx=idx, timestamp=(ts - start_ts) / 1e9,
                                  detections=detections_to_dicts(detections), motion_state=motion))
        if len(frames) % 200 == 0:
            print(f"[REPLAY] {len(frames)} processed, frame {idx}/{n_frames}", flush=True)

    if args.pace == "realtime":
        feed = LiveFeed(rec, n_frames)
        wall0, last = feed.wall0, -1
        while True:
            item = feed.newest_after(last)
            if item is None:
                break
            idx, ts, rgb, eye, motion, dec_ms = item
            t1 = time.perf_counter()
            detections, _, _ = detector.process(rgb, eye)
            sync()
            done_ns = time.perf_counter_ns()
            detect_ms.append((done_ns / 1e9 - t1) * 1000)
            decode_ms.append(dec_ms)
            # capture -> detections ready, on the recording clock (transport excluded)
            age_ms.append(((done_ns - wall0) - (ts - start_ts)) / 1e6)
            record_frame(idx, ts, motion, detections)
            last = idx
    else:
        wall0 = time.perf_counter_ns()
        for idx in range(n_frames):
            t0 = time.perf_counter()
            ts, rgb, eye = rec.read(idx)
            t1 = time.perf_counter()
            motion = rec.motion_state_at(ts)
            detections, _, _ = detector.process(rgb, eye)
            sync()
            t2 = time.perf_counter()
            decode_ms.append((t1 - t0) * 1000)
            detect_ms.append((t2 - t1) * 1000)
            loop_ms.append((time.perf_counter() - t0) * 1000)
            record_frame(idx, ts, motion, detections)

    wall_s = (time.perf_counter_ns() - wall0) / 1e9
    recording_s = (rec.rgb_ts[n_frames - 1] - start_ts) / 1e9
    domain_ms: List[float] = []
    metrics = run_benchmark(frames, video_fps=n_frames / max(recording_s, 1e-9),
                            frame_width=rec.rgb_shape()[1], fov_h=ARIA_RGB_FOV_H, step_ms=domain_ms)
    metrics.video_path = Path(args.vrs).name
    try:
        detector.cleanup()
    except Exception:
        pass

    class_counts = Counter(d["name"] for fr in frames for d in fr.detections)
    model_file = next((MODELS_DIR / f"{model_name}{ext}" for ext in (".engine", ".pt")
                       if (MODELS_DIR / f"{model_name}{ext}").exists()), None)
    record = {
        "schema": "aria-guard/replay/2",
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "recording": {
            "file": Path(args.vrs).name,
            "sequence": Path(args.vrs).parent.name,
            "rgb_shape": list(rec.rgb_shape()),
            "rgb_frames": n_frames,
            "seconds": round(recording_s, 2),
        },
        "settings": {
            "pace": args.pace, "mode": args.mode, "depth": not args.no_depth,
            "depth_interval": detector.depth_interval, "breakdown": args.breakdown,
            "warmup_frames": args.warmup, "imu_target_hz": args.imu_hz,
            "fov_h": ARIA_RGB_FOV_H,
        },
        "model": {
            "yolo": model_name,
            "yolo_backend": getattr(detector, "_yolo_backend", None),
            "file": model_file.name if model_file else None,
            "sha256": sha256(model_file) if model_file else None,
            "depth": "tensorrt" if getattr(detector, "_depth_tensorrt", False) else
                     ("pytorch" if detector.depth_model is not None else None),
            "gaze": "tensorrt" if getattr(detector, "_gaze_trt_engine", None) is not None else
                    ("pytorch" if detector.gaze_model is not None else None),
        },
        "environment": environment(),
        "throughput": {
            "processed_frames": len(frames),
            "dropped_frames": n_frames - len(frames),
            **({"decoded_frames": feed.decoded, "skipped_by_decoder": feed.skipped_by_decoder}
               if feed else {}),
            "wall_seconds": round(wall_s, 2),
            "effective_fps": round(len(frames) / wall_s, 2) if wall_s > 0 else None,
        },
        "latency_ms": {
            "decode": latency_summary(decode_ms),
            "detector": latency_summary(detect_ms),
            "tracker_arbiter": latency_summary(domain_ms),
            "loop": latency_summary(loop_ms),
            "capture_to_detections": latency_summary(age_ms),
            **({f"stage.{m.lstrip('_')}": latency_summary(v) for m, v in stage_ms.items()} if args.breakdown else {}),
        },
        "alerts": asdict(metrics),
        "detections_per_class": dict(class_counts.most_common()),
        "motion_state": dict(Counter(fr.motion_state for fr in frames)),
    }
    if args.save_detections:
        out = Path(args.save_detections)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fps": metrics.fps, "frames": [asdict(f) for f in frames]}))
    return record


def main():
    parser = argparse.ArgumentParser(description="Replay an Aria VRS through aria-guard and measure it")
    parser.add_argument("vrs", help="Path to recording.vrs")
    parser.add_argument("--pace", choices=["realtime", "all"], default="realtime")
    parser.add_argument("--mode", default="outdoor", help="indoor | outdoor | all")
    parser.add_argument("--no-depth", action="store_true", help="Disable Depth Anything")
    parser.add_argument("--breakdown", action="store_true",
                        help="Per-stage times (synchronizes the GPU around each stage)")
    parser.add_argument("--max-seconds", type=float, help="Only replay the first N seconds")
    parser.add_argument("--warmup", type=int, default=30, help="Unmeasured warm-up frames")
    parser.add_argument("--imu-hz", type=float, default=100.0,
                        help="IMU samples per second fed to the motion classifier")
    parser.add_argument("--out", required=True, help="Output JSON record")
    parser.add_argument("--save-detections", help="Also save per-frame detections to this JSON")
    args = parser.parse_args()

    record = replay(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    t, lat, al = record["throughput"], record["latency_ms"], record["alerts"]
    print(f"[REPLAY] {t['processed_frames']} frames ({t['dropped_frames']} dropped), "
          f"{t['effective_fps']} FPS, detector p50 {lat['detector'].get('p50')} ms / "
          f"p95 {lat['detector'].get('p95')} ms, {al['alerts_per_min']} alerts/min -> {out}")


if __name__ == "__main__":
    main()
