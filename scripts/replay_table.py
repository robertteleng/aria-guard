#!/usr/bin/env python3
"""
Markdown tables from replay records (benchmarks/replay/*.json).

Tables are generated from the records, never written by hand. One row per
machine: medians across recordings of each per-recording statistic, plus the
worst recording, so a single bad sequence cannot hide.

Usage:
    python scripts/replay_table.py benchmarks/replay/*.json
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load(paths: List[Path]) -> List[dict]:
    records = []
    for p in paths:
        r = json.loads(p.read_text())
        if not str(r.get("schema", "")).startswith("aria-guard/replay/"):
            continue
        records.append(r)
    return records


def device_label(r: dict) -> str:
    env = r["environment"]
    gpu = env.get("gpu") or "cpu"
    trt = env.get("tensorrt")
    return f"{gpu} · TensorRT {trt}" if trt else gpu


def build_label(r: dict) -> str:
    """Code version and pipeline variant, so optimizations are compared row by row."""
    commit = _get(r, "environment", "commit") or "?"
    backend = _get(r, "model", "yolo_backend") or "ultralytics"
    depth = "async depth" if _get(r, "settings", "depth_async") else "sync depth"
    return f"{commit} · YOLO {backend} · {depth}"


def _get(r: dict, *path, default=None):
    cur = r
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def realtime_table(records: List[dict]) -> str:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if _get(r, "settings", "pace") == "realtime":
            groups[(device_label(r), build_label(r), _get(r, "model", "yolo"))].append(r)
    lines = [
        "| Device | Build | Model | Recordings | Effective FPS (median / worst) | Frames dropped | "
        "Capture → detections p50 / p95 (ms) | Alerts/min (median / max) | All alert criteria pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (dev, build, model), rs in sorted(groups.items()):
        fps = [_get(r, "throughput", "effective_fps") for r in rs]
        processed = sum(_get(r, "throughput", "processed_frames", default=0) for r in rs)
        dropped = sum(_get(r, "throughput", "dropped_frames", default=0) for r in rs)
        total = processed + dropped
        apm = [_get(r, "alerts", "alerts_per_min") for r in rs]
        passes = sum(all(_get(r, "alerts", k) for k in
                         ("pass_silent_ratio", "pass_alerts_per_min", "pass_min_gap", "pass_max_concurrent"))
                     for r in rs)
        lines.append(
            f"| {dev} | {build} | {model} | {len(rs)} | {_fmt(_median(fps))} / {_fmt(min(f for f in fps if f is not None))} | "
            f"{dropped / total * 100 if total else 0:.1f} % | "
            f"{_fmt(_median(_get(r, 'latency_ms', 'capture_to_detections', 'p50') for r in rs))} / "
            f"{_fmt(_median(_get(r, 'latency_ms', 'capture_to_detections', 'p95') for r in rs))} | "
            f"{_fmt(_median(apm))} / {_fmt(max(a for a in apm if a is not None))} | {passes}/{len(rs)} |"
        )
    return "\n".join(lines)


STAGES = [
    ("decode", "VRS decode"),
    ("stage.run_yolo", "YOLO"),
    ("stage.run_depth", "Depth (1 in 5 frames)"),
    ("stage.estimate_gaze", "Gaze (1 in 2 frames)"),
    ("stage.create_detections", "Detections + depth lookup"),
    ("tracker_arbiter", "Tracker + arbiter"),
    ("detector", "Detector total"),
]


def stage_table(records: List[dict]) -> str:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if _get(r, "settings", "pace") == "all" and _get(r, "settings", "breakdown"):
            groups[f"{device_label(r)} · {build_label(r)}"].append(r)
    devices = sorted(groups)
    head = "| Stage (ms, median of per-recording p50 / p95) | " + " | ".join(devices) + " |"
    lines = [head, "|---|" + "---|" * len(devices)]
    for key, label in STAGES:
        cells = []
        for dev in devices:
            rs = groups[dev]
            p50 = _median(_get(r, "latency_ms", key, "p50") for r in rs)
            p95 = _median(_get(r, "latency_ms", key, "p95") for r in rs)
            cells.append(f"{_fmt(p50, 2)} / {_fmt(p95, 2)}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Markdown tables from replay records")
    parser.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args()
    records = load(args.records)
    if not records:
        sys.exit("[ERROR] no replay records")
    print("### Realtime replay\n")
    print(realtime_table(records))
    print("\n### Per-stage cost (every frame, GPU synchronized per stage)\n")
    print(stage_table(records))


if __name__ == "__main__":
    main()
