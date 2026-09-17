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


def test_hero_window_rule_scores_justified_minus_unjustified_and_skips_standing():
    pytest.importorskip("cv2")
    from scripts.render_demo import pick_hero_window

    def recording(walk_from, alerts, episodes=()):
        frames = [{"t": i / 10, "path": np.array([[0, 0], [3.0 if i / 10 >= walk_from else 0.0, 0]])}
                  for i in range(1200)]
        return {"frames": frames, "episodes": list(episodes),
                "alerts": [{"t": t, "justified": j, "track_id": k} for t, j, k in alerts]}

    # A: best score (2 - 0) but the wearer stands still there -> must be skipped
    a = recording(walk_from=100, alerts=[(20, True, 1), (22, True, 2)])
    # B: walking; window at 45 s holds 2 justified + 1 unjustified (score 1)
    b = recording(walk_from=0, alerts=[(50, True, 3), (52, True, 4), (55, False, 5)])
    # C: walking; same score 1, but one hazard episode warned -> wins the tie
    c = recording(walk_from=0, alerts=[(80, True, 6), (83, True, 7), (84, False, 8)],
                  episodes=[{"track_id": 6, "start": 79.0}])
    name, t0, score = pick_hero_window({"a": a, "b": b, "c": c})
    assert (name, score) == ("c", 1) and t0 == pytest.approx(75.0)


def test_hand_tracking_reads_confident_wrists_in_metres_within_time_window(tmp_path):
    import json
    rows = []
    for i, conf in enumerate((0.9, 0.2)):
        hand = {"existence_confidence": conf, "joint_angles": [],
                "T_wrist_device": {"translation": [100.0 * (i + 1), -250.0, 200.0], "quaternion": [0, 0, 0, 1]}}
        rows.append({"tracking_timestamp_us": 1_000_000 * (i + 1), "hand_poses": {"left": hand, "right": hand}})
    p = tmp_path / "hands.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    hands = ev.HandTracking(p)
    w = hands.wrists(1_000_000_000 + 30_000_000)            # 30 ms after the first row
    assert len(w) == 2 and w[0] == pytest.approx([0.1, -0.25, 0.2])
    assert hands.wrists(2_000_000_000) == []                 # low confidence
    assert hands.wrists(1_500_000_000) == []                 # no row within 60 ms


def test_wrist_in_box_uses_ten_percent_margin():
    assert ev.wrist_in_box((105, 50), (0, 0, 100, 100))
    assert not ev.wrist_in_box((115, 50), (0, 0, 100, 100))
    assert not ev.wrist_in_box(None, (0, 0, 100, 100))
