"""Tests for the alert-matching logic of scripts/evaluate_alerts.py (fake camera and trajectory)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("pandas")
from scripts import evaluate_alerts as ev  # noqa: E402


class FakeCam:
    def __init__(self, n, fps=30):
        self.rgb_ts = np.array([10**12 + int(i * 1e9 / fps) for i in range(n)], dtype=np.int64)


class StraightWalk:
    """Walks along +x at 1 m/s starting at the origin."""

    def __init__(self, t0_ns):
        self.t0 = t0_ns

    def covers(self, t_ns):
        return True

    def position(self, t_ns):
        return np.array([(t_ns - self.t0) / 1e9, 0.0, 0.0])


def frames_with(det_fn, n):
    frames = []
    for i in range(n):
        frames.append({"frame_idx": i, "timestamp": i / 30, "motion_state": "walking",
                       "detections": det_fn(i)})
    return frames


def car(i, x=600, depth=None):
    return {"name": "car", "bbox": [x, 500, 200, 150], "zone": "center", "distance": "close",
            "depth_value": depth if depth is not None else min(0.95, 0.3 + i * 0.01), "confidence": 0.9,
            "is_gazed": False, "traffic_light_state": None}


def run(frames, geo_xy):
    cam = FakeCam(len(frames))
    traj = StraightWalk(int(cam.rgb_ts[0]))
    geo = {(f["frame_idx"], k): {"ground_xy": np.array(geo_xy), "distance": 1.0}
           for f in frames for k in range(len(f["detections"]))}
    return ev.evaluate(frames, geo, cam, traj, "ttc_fix")


def test_object_on_path_gives_episode_and_justified_alerts():
    frames = frames_with(lambda i: [car(i)], 150)
    r = run(frames, geo_xy=[4.0, 0.2])        # 0.2 m beside the path, reached in ~4 s
    assert r["episodes"] >= 1
    assert r["alerts"] >= 1
    assert r["alert_precision"] == 1.0
    assert r["episode_recall"] == 1.0
    assert r["unjustified_alerts_per_min"] == 0


def test_object_far_from_path_alerts_are_unjustified():
    frames = frames_with(lambda i: [car(i)], 150)
    r = run(frames, geo_xy=[3.0, 5.0])        # 5 m to the side, never in the corridor
    assert r["episodes"] == 0 and r["episode_recall"] is None
    assert r["alerts"] >= 1 and r["alert_precision"] == 0.0
    assert r["unjustified_alerts_per_min"] > 0


class Standing:
    """Does not move: the future path collapses to a single point."""

    def covers(self, t_ns):
        return True

    def position(self, t_ns):
        return np.array([0.0, 0.0, 0.0])


def test_within_near_radius_counts_when_standing_still():
    frames = frames_with(lambda i: [car(i)], 60)
    cam = FakeCam(len(frames))

    def episodes_at(xy):
        geo = {(f["frame_idx"], 0): {"ground_xy": np.array(xy), "distance": 1.0} for f in frames}
        return ev.evaluate(frames, geo, cam, Standing(), "ttc_fix")["episodes"]

    assert episodes_at([0.9, 0.0]) == 1    # 0.9 m away: outside the 0.75 m corridor, inside the 1 m radius
    assert episodes_at([1.2, 0.0]) == 0    # beyond both


def test_variants_are_restored():
    import src.domain.tracker as tracker_mod
    original = tracker_mod.remaining_gap
    with ev.tracker_variant("pre_ttc"):
        assert tracker_mod.remaining_gap(0.8) == 0.8
    assert tracker_mod.remaining_gap is original
    with pytest.raises(ValueError):
        with ev.tracker_variant("nope"):
            pass


def test_crosscheck_summary_excludes_tree_from_headline_but_reports_it():
    rows = [("car", 5.0, 5.2), ("car", 8.0, 8.4), ("Tree", 4.0, 7.5), ("Tree", 6.0, 9.0)]
    s = ev.summarize_crosscheck(rows)
    assert s["headline"]["n"] == 2 and s["headline"]["share_within_25pct"] == 1.0
    assert s["by_class"]["Tree"]["median_signed_diff_m"] == pytest.approx(-3.25)
    assert len(s["rows"]) == 4 and s["headline_excludes"] == ["Tree"]
    assert ev.summarize_crosscheck([]) == {"n": 0}


def test_demo_window_prefers_walking_segments():
    cv2 = pytest.importorskip("cv2")
    from scripts.render_demo import pick_window
    frames = []
    for i in range(0, 1200):
        t = i / 10
        moving = t >= 60                      # stands for 60 s, then walks at 1 m/s
        path = np.array([[0, 0], [3.0 if moving else 0.0, 0]])
        frames.append({"t": t, "path": path})
    episodes = [{"start": s} for s in (5, 6, 7, 8, 70, 75)]   # more episodes while standing
    t0, speed = pick_window(episodes, frames, 120.0, 40.0)
    assert t0 >= 60 - 2 and speed >= 0.5
