"""Tests for scripts/replay_table.py with synthetic records."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.replay_table import load, realtime_table, stage_table


def record(pace, fps=30.0, dropped=0, apm=5.0, passing=True, gpu="Orin", p50=100.0, breakdown=False):
    lat = {"p50": p50, "p95": p50 * 2}
    return {
        "schema": "aria-guard/replay/2",
        "settings": {"pace": pace, "breakdown": breakdown},
        "model": {"yolo": "yolo26n_nav"},
        "environment": {"gpu": gpu, "tensorrt": "10.3.0"},
        "throughput": {"effective_fps": fps, "processed_frames": 100 - dropped, "dropped_frames": dropped},
        "latency_ms": {"capture_to_detections": lat, "decode": lat, "detector": lat,
                       "stage.run_yolo": lat, "tracker_arbiter": lat},
        "alerts": {"alerts_per_min": apm, "pass_silent_ratio": passing, "pass_alerts_per_min": True,
                   "pass_min_gap": True, "pass_max_concurrent": True},
    }


def test_realtime_row_reports_median_worst_and_drops():
    rs = [record("realtime", fps=30, dropped=0, apm=4), record("realtime", fps=10, dropped=50, apm=12, passing=False),
          record("realtime", fps=20, dropped=10, apm=6)]
    table = realtime_table(rs)
    row = table.splitlines()[2]
    assert "Orin · TensorRT 10.3.0" in row and "| 3 |" in row
    assert "YOLO ultralytics · sync depth" in row
    assert "20.0 / 10.0" in row          # median / worst FPS
    assert "20.0 %" in row               # 60 dropped of 300
    assert "6.0 / 12.0" in row           # alerts/min median / max
    assert "2/3" in row                  # one recording fails a criterion


def test_realtime_table_ignores_all_pace_records():
    table = realtime_table([record("all")])
    assert len(table.splitlines()) == 2  # header only


def test_stage_table_one_column_per_device_missing_stage_dash():
    rs = [record("all", gpu="Orin", p50=10, breakdown=True), record("all", gpu="RTX", p50=2, breakdown=True)]
    table = stage_table(rs)
    head = table.splitlines()[0]
    assert "Orin · TensorRT 10.3.0" in head and "RTX · TensorRT 10.3.0" in head and head.index("Orin") < head.index("RTX")
    yolo = next(l for l in table.splitlines() if l.startswith("| YOLO |"))
    assert "10.00 / 20.00" in yolo and "2.00 / 4.00" in yolo
    depth = next(l for l in table.splitlines() if l.startswith("| Depth"))
    assert "— / —" in depth


def test_load_skips_foreign_json(tmp_path):
    good, other = tmp_path / "a.json", tmp_path / "b.json"
    good.write_text(json.dumps(record("realtime")))
    other.write_text(json.dumps({"schema": "something-else"}))
    assert len(load([good, other])) == 1


def test_builds_are_separate_rows():
    base = record("realtime", fps=13)
    opt = record("realtime", fps=25)
    opt["environment"]["commit"] = "abc1234"
    opt["model"]["yolo_backend"] = "tensorrt"
    opt["settings"]["depth_async"] = True
    rows = realtime_table([base, opt]).splitlines()[2:]
    assert len(rows) == 2
    assert any("abc1234 · YOLO tensorrt · async depth" in r and "25.0 / 25.0" in r for r in rows)
