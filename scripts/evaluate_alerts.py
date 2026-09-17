#!/usr/bin/env python3
"""
Evaluate aria-guard alerts against the wearer's real path (docs/ALERT_EVALUATION.md).

Inputs per recording: the replay detections (scripts/replay_vrs.py
--save-detections), the VRS (timestamps and RGB calibration) and Meta's MPS
output (trajectory and semidense points). No manual labels.

Usage:
    python scripts/evaluate_alerts.py \\
        --detections benchmarks/replay/detections/<run>.json \\
        --recording ~/Datasets/aria/ritw/recording_XXXX --out benchmarks/alerts/<run>.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_offline import FrameResult, run_benchmark  # noqa: E402
from src.evaluation.geometry import (GridIndex2D, bearing_deg, merge_episodes,  # noqa: E402
                                     point_to_polyline_2d, quat_to_matrix, ray_ground_intersection,
                                     upright_to_raw_pixel)

# --- pre-registered parameters (docs/ALERT_EVALUATION.md) -------------------------
HORIZON_S = 3.0
CORRIDOR_M = 0.75
NEAR_M = 1.0
NEAR_BEARING_DEG = 30.0
EPISODE_GAP_S = 0.5
GROUND_RADIUS_M = 5.0
GROUND_PERCENTILE = 5
GROUND_MIN_POINTS = 200
MAX_GROUND_RANGE_M = 15.0
POINT_STD_MAX_M = 0.2
CROSSCHECK_EVERY = 10
PATH_STEP_S = 0.1
STATIC_CLASSES = {"car", "truck", "bus", "bench", "fire hydrant", "stop sign", "traffic light",
                  "potted plant", "Door", "Stairs", "Street light", "Traffic sign", "Tree"}
RGB_SIZE = 1408
FOV_H = 1.919


# --- MPS / VRS loading --------------------------------------------------------------

class Trajectory:
    def __init__(self, csv_path: Path):
        import pandas as pd
        cols = ["tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device",
                "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
        df = pd.read_csv(csv_path, usecols=cols)
        self.t_ns = df["tracking_timestamp_us"].to_numpy(np.int64) * 1000
        self.pos = df[cols[1:4]].to_numpy(np.float64)
        self.quat = df[cols[4:8]].to_numpy(np.float64)

    def index(self, t_ns: int) -> int:
        i = int(np.searchsorted(self.t_ns, t_ns))
        if i >= len(self.t_ns):
            return len(self.t_ns) - 1
        if i > 0 and abs(self.t_ns[i - 1] - t_ns) < abs(self.t_ns[i] - t_ns):
            return i - 1
        return i

    def covers(self, t_ns: int) -> bool:
        return self.t_ns[0] <= t_ns <= self.t_ns[-1]

    def pose(self, t_ns: int):
        i = self.index(t_ns)
        return quat_to_matrix(*self.quat[i]), self.pos[i]

    def position(self, t_ns: int) -> np.ndarray:
        return self.pos[self.index(t_ns)]


class Points:
    def __init__(self, csv_gz: Path):
        import pandas as pd
        df = pd.read_csv(csv_gz, usecols=["px_world", "py_world", "pz_world", "dist_std"])
        df = df[df["dist_std"] <= POINT_STD_MAX_M]
        self.xyz = df[["px_world", "py_world", "pz_world"]].to_numpy(np.float64)
        self.index = GridIndex2D(self.xyz)

    def ground_z(self, xy: np.ndarray):
        idx = self.index.query_radius(xy, GROUND_RADIUS_M)
        if len(idx) < GROUND_MIN_POINTS:
            return None
        return float(np.percentile(self.xyz[idx, 2], GROUND_PERCENTILE))

    def near(self, xy: np.ndarray, radius: float) -> np.ndarray:
        return self.xyz[self.index.query_radius(xy, radius)]


class ImuUp:
    """Up direction per timestamp from the VRS accelerometer (candidate change 1).

    Mean specific force over the preceding window, rotated to the device frame.
    """

    def __init__(self, vrs_path: Path, stream: str = "1202-1", label: str = "imu-right", window_s: float = 1.0):
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
        from projectaria_tools.core.stream_id import StreamId
        provider = data_provider.create_vrs_data_provider(str(vrs_path))
        sid = StreamId(stream)
        self.t_ns = np.asarray(provider.get_timestamps_ns(sid, TimeDomain.DEVICE_TIME), dtype=np.int64)
        accel = np.array([provider.get_imu_data_by_index(sid, i).accel_msec2 for i in range(len(self.t_ns))])
        self._cum = np.vstack([np.zeros(3), np.cumsum(accel, axis=0)])
        T = provider.get_device_calibration().get_imu_calib(label).get_transform_device_imu().to_matrix()
        self.R_di = T[:3, :3]
        self.window_ns = int(window_s * 1e9)

    def up_device(self, t_ns: int):
        hi = int(np.searchsorted(self.t_ns, t_ns, side="right"))
        lo = int(np.searchsorted(self.t_ns, t_ns - self.window_ns, side="left"))
        if hi <= lo:
            return None
        mean = (self._cum[hi] - self._cum[lo]) / (hi - lo)
        n = np.linalg.norm(mean)
        return None if n < 1e-6 else self.R_di @ (mean / n)


def live_metric_threats(frames: List[dict], cam: "RgbCamera", imu: ImuUp) -> Dict:
    """(frame_idx, det_index) -> {threat, forward_m, lateral_m} from live-available data only."""
    from src.domain.ground_projection import ground_contact, horizontal_basis, metric_threat
    cam_forward = cam.R_dc @ np.array([0.0, 0.0, 1.0])
    out = {}
    for fr in frames:
        up = imu.up_device(int(cam.rgb_ts[fr["frame_idx"]]))
        basis = horizontal_basis(up, cam_forward) if up is not None else None
        for k, d in enumerate(fr["detections"]):
            contact = None
            if basis is not None:
                x, y, w, h = d["bbox"]
                ray = cam.R_dc @ cam.ray_upright(x + w / 2, y + h)
                contact = ground_contact(ray, up, *basis)
            out[(fr["frame_idx"], k)] = {"threat": metric_threat(contact),
                                         "forward_m": contact[0] if contact else None,
                                         "lateral_m": contact[1] if contact else None}
    return out


class RgbCamera:
    def __init__(self, vrs_path: Path):
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
        from projectaria_tools.core.stream_id import StreamId
        provider = data_provider.create_vrs_data_provider(str(vrs_path))
        stream = StreamId("214-1")
        self.rgb_ts = np.asarray(provider.get_timestamps_ns(stream, TimeDomain.DEVICE_TIME), dtype=np.int64)
        self.calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
        T = self.calib.get_transform_device_camera().to_matrix()
        self.R_dc, self.t_dc = T[:3, :3], T[:3, 3]

    def ray_upright(self, x: float, y: float) -> np.ndarray:
        """Unit ray in the camera frame for an upright-frame pixel."""
        rx, ry = upright_to_raw_pixel(x, y, RGB_SIZE)
        v = np.asarray(self.calib.unproject_no_checks(np.array([rx, ry], dtype=np.float64)), dtype=np.float64)
        return v / np.linalg.norm(v)


# --- reference geometry per detection --------------------------------------------------

def reference_geometry(frames: List[dict], cam: RgbCamera, traj: Trajectory, pts: Points) -> Dict:
    """World ground position of every detection bottom-centre, plus the static cross-check."""
    geo = {}  # (frame_idx, det_index) -> dict
    stats = Counter()
    crosscheck = []
    static_frames_seen = 0
    for fr in frames:
        fi = fr["frame_idx"]
        t_ns = int(cam.rgb_ts[fi])
        if not traj.covers(t_ns):
            stats["frames_outside_trajectory"] += 1
            continue
        R_wd, t_wd = traj.pose(t_ns)
        cam_center = R_wd @ cam.t_dc + t_wd
        R_wc = R_wd @ cam.R_dc
        ground = pts.ground_z(cam_center)
        if ground is None:
            stats["frames_without_ground"] += 1
            continue
        do_cross = False
        if any(d["name"] in STATIC_CLASSES for d in fr["detections"]):
            do_cross = static_frames_seen % CROSSCHECK_EVERY == 0
            static_frames_seen += 1
        near_pts = pts.near(cam_center, 20.0) if do_cross else None
        for k, d in enumerate(fr["detections"]):
            x, y, w, h = d["bbox"]
            ray_w = R_wc @ cam.ray_upright(x + w / 2, y + h)
            hit = ray_ground_intersection(cam_center, ray_w, ground, MAX_GROUND_RANGE_M)
            stats["detections"] += 1
            if hit is None:
                stats["no_ground_contact"] += 1
                continue
            geo[(fi, k)] = {"ground_xy": hit[:2], "distance": float(np.hypot(*(hit[:2] - cam_center[:2])))}
            if do_cross and d["name"] in STATIC_CLASSES and len(near_pts):
                sd = semidense_distance(near_pts, cam_center, R_wc, cam, d["bbox"])
                if sd is not None:
                    crosscheck.append((d["name"], geo[(fi, k)]["distance"], sd))
    return {"geo": geo, "stats": stats, "crosscheck": crosscheck}


def semidense_distance(points_w, cam_center, R_wc, cam, bbox):
    """Median horizontal distance of converged points inside the central half of the bbox."""
    x, y, w, h = bbox
    corners = [(x + w / 4, y + h / 4), (x + 3 * w / 4, y + h / 4), (x + w / 4, y + 3 * h / 4), (x + 3 * w / 4, y + 3 * h / 4)]
    rays = np.array([cam.ray_upright(cx, cy) for cx, cy in corners])
    if (rays[:, 2] <= 0.05).any():
        return None
    nx, ny = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    p_cam = (points_w - cam_center) @ R_wc  # rows: R_wc^T (p - c)
    front = p_cam[:, 2] > 0.3
    if not front.any():
        return None
    p_cam, p_w = p_cam[front], points_w[front]
    px, py = p_cam[:, 0] / p_cam[:, 2], p_cam[:, 1] / p_cam[:, 2]
    inside = (px >= nx.min()) & (px <= nx.max()) & (py >= ny.min()) & (py <= ny.max())
    if inside.sum() < 5:
        return None
    return float(np.median(np.hypot(*(p_w[inside, :2] - cam_center[:2]).T)))


# --- tracker variants -----------------------------------------------------------------

@contextmanager
def tracker_variant(name: str):
    """'ttc_fix' = current heuristic; 'pre_ttc' = proximity used as distance (before
    3c56e50); 'metric_inpath' = candidate change 1 (needs metric threats injected)."""
    import src.domain.tracker as tracker_mod
    original = tracker_mod.remaining_gap
    if name == "pre_ttc":
        tracker_mod.remaining_gap = lambda proximity: proximity
    elif name not in ("ttc_fix", "metric_inpath"):
        raise ValueError(name)
    try:
        yield
    finally:
        tracker_mod.remaining_gap = original


# --- evaluation ------------------------------------------------------------------------

def evaluate(frames: List[dict], geo: Dict, cam: RgbCamera, traj: Trajectory, variant: str,
             details: bool = False, metric: Optional[Dict] = None) -> dict:
    """Metrics for one tracker variant. details=True also returns the per-frame
    matching (tracks with ground position and in-path flag, alerts with their
    verdict) used by scripts/render_demo.py."""
    if variant == "metric_inpath":
        if metric is None:
            raise ValueError("metric_inpath needs live metric threats")
        frames = [{**f, "detections": [{**d, "metric_threat": metric[(f["frame_idx"], k)]["threat"]}
                                       for k, d in enumerate(f["detections"])]} for f in frames]
    frame_results = [FrameResult(**{k: f[k] for k in ("frame_idx", "timestamp", "detections", "motion_state")})
                     for f in frames]
    log: List[dict] = []
    with tracker_variant(variant):
        metrics = run_benchmark(frame_results, video_fps=30.0, frame_width=RGB_SIZE, fov_h=FOV_H, frame_log=log)

    t0_ns = int(cam.rgb_ts[frames[0]["frame_idx"]])
    samples = []                               # (track_id, t, in_path, dist, t_closest)
    frame_details = []
    in_path_times = defaultdict(list)          # track_id -> [t]
    by_frame_dets = {f["frame_idx"]: f["detections"] for f in frames}
    for entry in log:
        fi = entry["frame_idx"]
        t_ns = int(cam.rgb_ts[fi])
        if not traj.covers(t_ns + int(HORIZON_S * 1e9)):
            continue
        t = (t_ns - t0_ns) / 1e9
        steps = np.arange(0.0, HORIZON_S + 1e-9, PATH_STEP_S)
        path = np.array([traj.position(t_ns + int(s * 1e9))[:2] for s in steps])
        here = path[0]
        heading = traj.position(t_ns + int(0.5e9))[:2] - here
        dets = by_frame_dets[fi]
        fd = {"frame_idx": fi, "t": t, "path": path, "tracks": []} if details else None
        for tr in entry["tracks"]:
            k = next((i for i, d in enumerate(dets) if d["bbox"] == tr["bbox"] and d["name"] == tr["name"]), None)
            g = geo.get((fi, k)) if k is not None else None
            if g is None:
                continue
            dist, seg, frac = point_to_polyline_2d(g["ground_xy"], path)
            t_closest = t + (seg + frac) * PATH_STEP_S
            near = (np.linalg.norm(g["ground_xy"] - here) <= NEAR_M
                    and bearing_deg(here, heading, g["ground_xy"]) <= NEAR_BEARING_DEG)
            in_path = dist <= CORRIDOR_M or near
            samples.append((tr["id"], t, in_path, dist, t_closest))
            if in_path:
                in_path_times[tr["id"]].append(t)
            if fd is not None:
                fd["tracks"].append({**tr, "ground_xy": g["ground_xy"], "in_path": bool(in_path),
                                     "path_distance": dist})
        if fd is not None:
            frame_details.append(fd)

    episodes = merge_episodes(samples, EPISODE_GAP_S)
    alerts = [(((int(cam.rgb_ts[e["frame_idx"]]) - t0_ns) / 1e9), a) for e in log for a in e["alerts"]
              if a["channel"] == "A"]

    def justified(t, a):
        ts = in_path_times.get(a["track_id"], [])
        return any(t <= s <= t + HORIZON_S for s in ts)

    level_total, level_just = Counter(), Counter()
    for t, a in alerts:
        level_total[a["level"]] += 1
        level_just[a["level"]] += justified(t, a)

    warned, leads, ep_level = 0, [], Counter()
    for ep in episodes:
        first = [(t, a) for t, a in alerts
                 if a["track_id"] == ep.track_id and ep.start - HORIZON_S <= t <= ep.closest_time]
        if first:
            warned += 1
            leads.append(ep.closest_time - first[0][0])
            ep_level[first[0][1]["level"]] += 1

    minutes = metrics.total_seconds / 60 if metrics.total_seconds else 0
    detail = None
    if details:
        detail = {
            "frames": frame_details,
            "alerts": [{"t": t, **a, "justified": justified(t, a)} for t, a in alerts],
            "episodes": [asdict(e) for e in episodes],
            "t0_ns": t0_ns,
        }
    n_alerts = len(alerts)
    n_just = sum(level_just.values())
    return {
        "variant": variant,
        "alerts": n_alerts,
        "justified_alerts": n_just,
        "alert_precision": round(n_just / n_alerts, 3) if n_alerts else None,
        "episodes": len(episodes),
        "warned_episodes": warned,
        "episode_recall": round(warned / len(episodes), 3) if episodes else None,
        "lead_time_s": {"median": round(float(np.median(leads)), 2) if leads else None,
                        "p10": round(float(np.percentile(leads, 10)), 2) if leads else None},
        "unjustified_alerts_per_min": round((n_alerts - n_just) / minutes, 2) if minutes else None,
        "by_level": {lvl: {"alerts": level_total[lvl], "justified": level_just[lvl],
                           "precision": round(level_just[lvl] / level_total[lvl], 3) if level_total[lvl] else None,
                           "first_alert_of_warned_episodes": ep_level[lvl]}
                     for lvl in ("ATTENTION", "WARNING", "DANGER")},
        "alert_benchmark": asdict(metrics),
        **({"details": detail} if details else {}),
    }


CROSSCHECK_EXCLUDED_FROM_HEADLINE = {"Tree"}  # bbox centre is canopy/background (amendment)


def _agreement(g: np.ndarray, s: np.ndarray) -> dict:
    rel = np.abs(g - s) / np.maximum(s, 1e-6)
    return {"n": int(len(g)),
            "median_abs_diff_m": round(float(np.median(np.abs(g - s))), 3),
            "median_signed_diff_m": round(float(np.median(g - s)), 3),
            "share_within_25pct": round(float((rel <= 0.25).mean()), 3)}


def live_reference_agreement(metric: Dict, geo: Dict) -> dict:
    """Candidate forward distance vs the reference's metric distance, same detections."""
    pairs = [(m["forward_m"], geo[key]["distance"]) for key, m in metric.items()
             if m["forward_m"] is not None and key in geo]
    if not pairs:
        return {"n": 0}
    live, ref = np.array(pairs).T
    return {**_agreement(live, ref), "live_only": sum(1 for k, m in metric.items() if m["forward_m"] is not None and k not in geo),
            "reference_only": sum(1 for k in geo if metric.get(k, {}).get("forward_m") is None)}


def summarize_crosscheck(rows) -> dict:
    """Ground-contact vs semidense distance: headline without excluded classes, plus per class."""
    if not rows:
        return {"n": 0}
    names = np.array([r[0] for r in rows])
    g = np.array([r[1] for r in rows], dtype=np.float64)
    s = np.array([r[2] for r in rows], dtype=np.float64)
    keep = ~np.isin(names, list(CROSSCHECK_EXCLUDED_FROM_HEADLINE))
    return {
        "headline": _agreement(g[keep], s[keep]) if keep.any() else {"n": 0},
        "headline_excludes": sorted(CROSSCHECK_EXCLUDED_FROM_HEADLINE),
        "by_class": {c: _agreement(g[names == c], s[names == c]) for c in sorted(set(names))},
        "rows": [[str(n), round(float(a), 3), round(float(b), 3)] for n, a, b in rows],
    }


def _git_commit():
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Evaluate alerts against the wearer's real path (MPS)")
    ap.add_argument("--detections", required=True, type=Path)
    ap.add_argument("--recording", required=True, type=Path, help="folder with recording.vrs and mps/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", nargs="+", default=["pre_ttc", "ttc_fix"],
                    help="pre_ttc, ttc_fix, metric_inpath")
    ap.add_argument("--details-out", type=Path,
                    help="Also pickle per-frame matching of every variant (input of render_demo.py)")
    args = ap.parse_args()

    frames = json.loads(args.detections.read_text())["frames"]
    cam = RgbCamera(args.recording / "recording.vrs")
    traj = Trajectory(args.recording / "mps" / "slam" / "closed_loop_trajectory.csv")
    pts = Points(args.recording / "mps" / "slam" / "semidense_points.csv.gz")
    ref = reference_geometry(frames, cam, traj, pts)
    metric = None
    if "metric_inpath" in args.variants:
        metric = live_metric_threats(frames, cam, ImuUp(args.recording / "recording.vrs"))
    record = {
        "schema": "aria-guard/alert-eval/1",
        "commit": _git_commit(),
        "recording": args.recording.name,
        "detections": args.detections.name,
        "parameters": {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")
                       and isinstance(v, (int, float))},
        "reference": {**dict(ref["stats"]), "crosscheck": summarize_crosscheck(ref["crosscheck"])},
        "results": [evaluate(frames, ref["geo"], cam, traj, v, details=args.details_out is not None, metric=metric)
                    for v in args.variants],
        **({"live_vs_reference_distance": live_reference_agreement(metric, ref["geo"])} if metric else {}),
    }
    if args.details_out:
        import pickle
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.details_out, "wb") as f:
            pickle.dump({r["variant"]: r for r in record["results"]}, f)
        for r in record["results"]:
            r.pop("details", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(json.dumps({"reference": record["reference"],
                      "results": [{k: r[k] for k in ("variant", "alerts", "alert_precision", "episodes",
                                                     "episode_recall", "lead_time_s", "unjustified_alerts_per_min")}
                                  for r in record["results"]]}, indent=1))


if __name__ == "__main__":
    main()
