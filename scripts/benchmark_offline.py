#!/usr/bin/env python3
"""
Offline benchmark for aria-guard alert pipeline (H21).

Processes a video file through tracker + arbiter (no audio) and measures:
- alerts_per_min: total alerts per minute
- danger_count: DANGER alerts (should be few, real)
- min_gap_s: minimum gap between consecutive alerts (target >1.5s)
- silent_ratio: % of time in silence (target >80%)
- max_concurrent: max alerts in same frame (target: 0, always 1 at a time)
- channel_b_count: context alerts (traffic light / sign)

Usage (inside Docker with CUDA):
    python scripts/benchmark_offline.py data/Tokyo_POV.mp4

Usage (CPU-only, with pre-recorded detections JSON):
    python scripts/benchmark_offline.py --from-json results/benchmark_detections.json

The script can also save per-frame detections to JSON for later analysis:
    python scripts/benchmark_offline.py data/Tokyo_POV.mp4 --save-json results/detections.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.tracker import SimpleTracker
from src.domain.alert_engine import AlertArbiter


@dataclass
class FrameResult:
    """Per-frame detection data for serialization."""
    frame_idx: int
    timestamp: float  # seconds into video
    detections: list  # list of dicts with name, bbox, zone, distance, depth_value, confidence, is_gazed
    motion_state: str = "unknown"  # user ego-motion from the IMU, as the live tracker receives it


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics from benchmark run."""
    video_path: str
    total_frames: int
    total_seconds: float
    fps: float

    # Alert counts
    total_alerts: int
    danger_count: int
    warning_count: int
    attention_count: int
    channel_b_count: int

    # Rates
    alerts_per_min: float
    silent_ratio: float  # % of seconds with no alert

    # Timing
    min_gap_s: float  # minimum gap between consecutive alerts
    max_gap_s: float  # maximum gap

    # Concurrency
    max_concurrent: int  # max channel_a + channel_b firing in same frame

    # Pass/fail
    pass_silent_ratio: bool   # target: >80%
    pass_alerts_per_min: bool # target: <12
    pass_min_gap: bool        # target: >1.0s (DANGER cooldown)
    pass_max_concurrent: bool # target: <=1


class FakeDet:
    """Lightweight detection object for feeding tracker."""
    def __init__(self, **kw):
        kw.setdefault("traffic_light_state", None)
        self.__dict__.update(kw)


def process_video(video_path: str, mode: str = "all", skip_frames: int = 1) -> List[FrameResult]:
    """Process video with real detector. Requires CUDA."""
    import cv2
    from src.detection import DetectorProcess

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[BENCH] Video: {video_path}")
    print(f"[BENCH] {frame_w}x{frame_h} @ {video_fps:.1f}fps, {total_frames} frames")
    print(f"[BENCH] Processing every {skip_frames} frame(s)...")

    # Start detector
    detector = DetectorProcess(mode=mode, enable_depth=True)
    if not detector.start(timeout=60, frame_shape=(frame_h, frame_w, 3)):
        print("[ERROR] DetectorProcess failed to start")
        sys.exit(1)

    results = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        detector.send_frame(frame, None, None)
        # Wait for result (blocking-ish)
        result = None
        for _ in range(50):  # max 0.5s wait
            result = detector.get_result()
            if result:
                break
            time.sleep(0.01)

        detections = []
        if result:
            for det in result.get("detections", []):
                detections.append({
                    "name": det.name,
                    "bbox": list(det.bbox),
                    "zone": det.zone,
                    "distance": det.distance,
                    "depth_value": float(det.depth_value),
                    "confidence": float(det.confidence),
                    "is_gazed": getattr(det, "is_gazed", False),
                    "traffic_light_state": getattr(det, "traffic_light_state", None),
                })

        results.append(FrameResult(
            frame_idx=frame_idx,
            timestamp=frame_idx / video_fps,
            detections=detections,
        ))

        processed += 1
        if processed % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"[BENCH] {processed} frames processed ({pct:.0f}%)")

        frame_idx += 1

    cap.release()
    detector.stop()
    print(f"[BENCH] Done. {processed} frames processed.")
    return results


def run_benchmark(frame_results: List[FrameResult], video_fps: float = 30.0,
                  frame_width: int = 1920, fov_h: float = 1.15,
                  step_ms: Optional[List[float]] = None) -> BenchmarkMetrics:
    """Run tracker + arbiter on frame results and compute metrics.

    frame_width and fov_h must match the source (Aria RGB: 1408 px, 1.919 rad):
    the tracker turns pixel positions into bearings with them. If step_ms is
    given, the tracker + arbiter time of each frame is appended to it.
    """
    tracker = SimpleTracker()
    arbiter = AlertArbiter()

    # Metrics tracking
    alert_timestamps = []  # (timestamp, threat_level, channel)
    frames_with_alert = set()
    concurrent_counts = []

    total_frames = len(frame_results)
    if total_frames == 0:
        print("[ERROR] No frames to process")
        sys.exit(1)

    total_seconds = frame_results[-1].timestamp - frame_results[0].timestamp
    if total_seconds <= 0:
        total_seconds = total_frames / video_fps

    for fr in frame_results:
        # Convert dicts to FakeDet objects
        dets = [FakeDet(**d) for d in fr.detections]
        # Fix bbox to tuple
        for d in dets:
            d.bbox = tuple(d.bbox)

        t0 = time.perf_counter()
        tracked = tracker.update(dets, frame_width=frame_width, fov_h=fov_h,
                                 motion_state=fr.motion_state)

        # Run arbiter on video time, not wall clock
        channel_a, channel_b = _decide_with_time(arbiter, tracker, fr.timestamp)
        if step_ms is not None:
            step_ms.append((time.perf_counter() - t0) * 1000)

        concurrent = 0
        if channel_a and channel_a.should_alert:
            alert_timestamps.append((fr.timestamp, channel_a.threat_level, "A"))
            frames_with_alert.add(fr.frame_idx)
            concurrent += 1

        if channel_b and channel_b.should_alert:
            alert_timestamps.append((fr.timestamp, "CONTEXT", "B"))
            frames_with_alert.add(fr.frame_idx)
            concurrent += 1

        concurrent_counts.append(concurrent)

    # Compute metrics
    total_alerts = len(alert_timestamps)
    danger_count = sum(1 for _, lvl, _ in alert_timestamps if lvl == "DANGER")
    warning_count = sum(1 for _, lvl, _ in alert_timestamps if lvl == "WARNING")
    attention_count = sum(1 for _, lvl, _ in alert_timestamps if lvl == "ATTENTION")
    channel_b_count = sum(1 for _, _, ch in alert_timestamps if ch == "B")

    # alerts_per_min counts only Channel A (threats) — Channel B is informational
    channel_a_count = total_alerts - channel_b_count
    alerts_per_min = (channel_a_count / total_seconds * 60) if total_seconds > 0 else 0

    # Silent ratio: % of 1-second windows with no alert
    n_windows = max(1, int(total_seconds))
    alert_seconds = set()
    for ts, _, _ in alert_timestamps:
        alert_seconds.add(int(ts))
    silent_ratio = (n_windows - len(alert_seconds)) / n_windows

    # Gap between consecutive alerts
    gaps = []
    channel_a_times = sorted([ts for ts, _, ch in alert_timestamps if ch == "A"])
    for i in range(1, len(channel_a_times)):
        gaps.append(channel_a_times[i] - channel_a_times[i - 1])
    min_gap = min(gaps) if gaps else float("inf")
    max_gap = max(gaps) if gaps else 0.0

    max_concurrent = max(concurrent_counts) if concurrent_counts else 0

    metrics = BenchmarkMetrics(
        video_path="",
        total_frames=total_frames,
        total_seconds=round(total_seconds, 1),
        fps=round(total_frames / total_seconds, 1) if total_seconds > 0 else 0,
        total_alerts=total_alerts,
        danger_count=danger_count,
        warning_count=warning_count,
        attention_count=attention_count,
        channel_b_count=channel_b_count,
        alerts_per_min=round(alerts_per_min, 1),
        silent_ratio=round(silent_ratio, 3),
        min_gap_s=round(min_gap, 2) if min_gap != float("inf") else -1,
        max_gap_s=round(max_gap, 2),
        max_concurrent=max_concurrent,
        pass_silent_ratio=silent_ratio >= 0.80,
        pass_alerts_per_min=round(alerts_per_min, 1) <= 12.0,
        pass_min_gap=min_gap >= 1.0 or min_gap == float("inf"),
        pass_max_concurrent=max_concurrent <= 1,
    )
    return metrics


def _decide_with_time(arbiter: AlertArbiter, tracker: SimpleTracker, timestamp: float):
    """Call arbiter.decide() using video timestamp instead of wall clock.

    Monkey-patches time.time for the duration of the call so cooldowns
    work relative to video time, not real time.
    """
    import unittest.mock
    with unittest.mock.patch("src.domain.alert_engine.time") as mock_time:
        mock_time.time.return_value = timestamp
        return arbiter.decide(tracker)


def print_metrics(m: BenchmarkMetrics):
    """Print benchmark results as a formatted table."""
    def status(passed):
        return "PASS" if passed else "FAIL"

    print()
    print("=" * 60)
    print("  ARIA GUARD — Benchmark Results (H21)")
    print("=" * 60)
    print(f"  Frames:     {m.total_frames}")
    print(f"  Duration:   {m.total_seconds}s ({m.total_seconds/60:.1f} min)")
    print(f"  FPS:        {m.fps}")
    print()
    print("  ALERTS")
    print(f"  Total:      {m.total_alerts}")
    print(f"  DANGER:     {m.danger_count}")
    print(f"  WARNING:    {m.warning_count}")
    print(f"  ATTENTION:  {m.attention_count}")
    print(f"  Context(B): {m.channel_b_count}")
    print()
    print("  METRICS                      VALUE     TARGET   STATUS")
    print(f"  alerts/min                   {m.alerts_per_min:>6.1f}     <12      [{status(m.pass_alerts_per_min)}]")
    print(f"  silent ratio                 {m.silent_ratio:>6.1%}     >80%     [{status(m.pass_silent_ratio)}]")
    gap_str = f"{m.min_gap_s:.2f}s" if m.min_gap_s >= 0 else "N/A"
    print(f"  min gap (Ch.A)               {gap_str:>6s}     >1.0s    [{status(m.pass_min_gap)}]")
    print(f"  max concurrent               {m.max_concurrent:>6d}     <=1      [{status(m.pass_max_concurrent)}]")
    print()

    all_pass = all([m.pass_silent_ratio, m.pass_alerts_per_min, m.pass_min_gap, m.pass_max_concurrent])
    if all_pass:
        print("  OVERALL: ALL PASS")
    else:
        fails = []
        if not m.pass_silent_ratio: fails.append("silent_ratio")
        if not m.pass_alerts_per_min: fails.append("alerts_per_min")
        if not m.pass_min_gap: fails.append("min_gap")
        if not m.pass_max_concurrent: fails.append("max_concurrent")
        print(f"  OVERALL: FAIL ({', '.join(fails)})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ARIA Guard Offline Benchmark (H21)")
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--from-json", help="Load pre-recorded detections from JSON")
    parser.add_argument("--save-json", help="Save per-frame detections to JSON")
    parser.add_argument("--save-metrics", help="Save metrics to JSON")
    parser.add_argument("--mode", default="all", help="Detection mode: indoor/outdoor/all")
    parser.add_argument("--skip", type=int, default=1, help="Process every N frames (1=all)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (for JSON input)")
    parser.add_argument("--frame-width", type=int, default=1920, help="Frame width in pixels")
    parser.add_argument("--fov-h", type=float, default=1.15, help="Horizontal FOV in radians")
    args = parser.parse_args()

    if args.from_json:
        # Load from pre-recorded JSON
        print(f"[BENCH] Loading detections from {args.from_json}")
        with open(args.from_json) as f:
            data = json.load(f)
        frame_results = [FrameResult(**fr) for fr in data["frames"]]
        video_fps = data.get("fps", args.fps)
        print(f"[BENCH] Loaded {len(frame_results)} frames")
    elif args.video:
        # Process video with detector
        frame_results = process_video(args.video, mode=args.mode, skip_frames=args.skip)
        video_fps = args.fps

        # Optionally save detections
        if args.save_json:
            out_path = Path(args.save_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "video": args.video,
                "fps": video_fps,
                "mode": args.mode,
                "frames": [asdict(fr) for fr in frame_results],
            }
            with open(out_path, "w") as f:
                json.dump(data, f)
            print(f"[BENCH] Saved detections to {out_path}")
    else:
        parser.error("Provide video path or --from-json")

    # Run benchmark
    metrics = run_benchmark(frame_results, video_fps=video_fps,
                            frame_width=args.frame_width, fov_h=args.fov_h)
    metrics.video_path = args.video or args.from_json or ""

    # Print results
    print_metrics(metrics)

    # Save metrics
    if args.save_metrics:
        out_path = Path(args.save_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"[BENCH] Saved metrics to {out_path}")

    # Exit code: 0 if all pass, 1 if any fail
    all_pass = all([metrics.pass_silent_ratio, metrics.pass_alerts_per_min,
                    metrics.pass_min_gap, metrics.pass_max_concurrent])
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
