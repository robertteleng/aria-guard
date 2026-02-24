"""Tests for benchmark_offline script (H21).

Uses synthetic detections to verify metrics computation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_offline import FrameResult, run_benchmark, FakeDet


def make_scene_calm(n_frames=300, fps=30.0):
    """Calm scene: 5 people far away, static. Should produce 0 alerts."""
    results = []
    for i in range(n_frames):
        dets = [
            {"name": "person", "bbox": [100+j*200, 200, 40, 80], "zone": "left" if j < 2 else "right",
             "distance": "far", "depth_value": 0.15, "confidence": 0.7, "is_gazed": False,
             "traffic_light_state": None}
            for j in range(5)
        ]
        results.append(FrameResult(frame_idx=i, timestamp=i/fps, detections=dets))
    return results


def make_scene_one_danger(n_frames=900, fps=30.0):
    """Car appears far, approaches over 30s. Should trigger a few alerts."""
    results = []
    for i in range(n_frames):
        depth = 0.1 + i * 0.001  # slowly approaching over 30s
        dist = "far" if i < 300 else ("medium" if i < 600 else "close")
        dets = [
            {"name": "car", "bbox": [600, 300, 100, 80], "zone": "center",
             "distance": dist, "depth_value": depth, "confidence": 0.85, "is_gazed": False,
             "traffic_light_state": None}
        ]
        results.append(FrameResult(frame_idx=i, timestamp=i/fps, detections=dets))
    return results


def make_scene_chaos(n_frames=900, fps=30.0):
    """Busy intersection: 10 objects, some approaching. Tests rate limiting."""
    results = []
    for i in range(n_frames):
        dets = []
        # 3 cars at various distances, some approaching
        for j in range(3):
            depth = 0.2 + j * 0.15 + (i * 0.002 if j == 0 else 0)
            dets.append({
                "name": "car", "bbox": [200+j*300, 300, 100, 80],
                "zone": ["left", "center", "right"][j],
                "distance": "close" if depth > 0.6 else ("medium" if depth > 0.3 else "far"),
                "depth_value": depth, "confidence": 0.85, "is_gazed": False,
                "traffic_light_state": None,
            })
        # 5 people at various distances
        for j in range(5):
            dets.append({
                "name": "person", "bbox": [100+j*200, 200, 40, 80],
                "zone": "center" if j == 2 else ("left" if j < 2 else "right"),
                "distance": "medium", "depth_value": 0.35, "confidence": 0.7, "is_gazed": False,
                "traffic_light_state": None,
            })
        # 1 traffic light
        dets.append({
            "name": "traffic light", "bbox": [640, 80, 30, 60], "zone": "center",
            "distance": "medium", "depth_value": 0.3, "confidence": 0.9, "is_gazed": False,
            "traffic_light_state": "red",
        })
        results.append(FrameResult(frame_idx=i, timestamp=i/fps, detections=dets))
    return results


class TestBenchmarkCalm:
    """Calm scene should produce minimal alerts."""

    def test_calm_low_alerts(self):
        frames = make_scene_calm()
        m = run_benchmark(frames, video_fps=30.0)
        # Far static people should barely trigger
        assert m.alerts_per_min <= 12.0
        assert m.pass_alerts_per_min

    def test_calm_high_silence(self):
        frames = make_scene_calm()
        m = run_benchmark(frames, video_fps=30.0)
        assert m.silent_ratio >= 0.80
        assert m.pass_silent_ratio


class TestBenchmarkDanger:
    """Single approaching car should trigger limited alerts."""

    def test_danger_detected(self):
        frames = make_scene_one_danger()
        m = run_benchmark(frames, video_fps=30.0)
        # Should detect at least 1 alert
        assert m.total_alerts >= 1

    def test_danger_not_spammy(self):
        frames = make_scene_one_danger()
        m = run_benchmark(frames, video_fps=30.0)
        assert m.alerts_per_min <= 12.0


class TestBenchmarkChaos:
    """Busy scene should stay within rate limits."""

    def test_chaos_rate_limited(self):
        frames = make_scene_chaos()
        m = run_benchmark(frames, video_fps=30.0)
        assert m.pass_alerts_per_min, f"alerts_per_min={m.alerts_per_min} exceeds 12"

    def test_chaos_max_concurrent(self):
        frames = make_scene_chaos()
        m = run_benchmark(frames, video_fps=30.0)
        assert m.pass_max_concurrent, f"max_concurrent={m.max_concurrent} > 1"

    def test_chaos_min_gap(self):
        frames = make_scene_chaos()
        m = run_benchmark(frames, video_fps=30.0)
        assert m.pass_min_gap, f"min_gap={m.min_gap_s}s < 1.0s"
