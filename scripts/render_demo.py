#!/usr/bin/env python3
"""
Render "what aria-guard says vs what the wearer actually walked into".

Left: the glasses' RGB with aria-guard's tracks and alerts. Right: top-down view
from Meta's MPS (metric trajectory, 3D points), the 3 s path corridor and the
objects' ground positions. Bottom: timeline of hazard episodes vs alerts, each
alert marked justified or not. Everything drawn comes from
scripts/evaluate_alerts.py; the clip is chosen by a fixed rule (the 40 s window
with the most hazard episodes), not by hand.

Usage:
    python scripts/evaluate_alerts.py ... --details-out /tmp/details.pkl
    python scripts/render_demo.py --details /tmp/details.pkl \\
        --recording ~/Datasets/aria/ritw/recording_XXXX --out docs/media/demo.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_alerts as ev  # noqa: E402
from scripts.replay_vrs import VrsRecording  # noqa: E402

W, H = 1280, 720
CAM = 560                 # camera panel side
BEV_W, BEV = W - CAM, 560  # top-down panel
TL_H = H - CAM            # timeline height
PX_PER_M = 36.0
DEVICE_Y = BEV - 70       # device position in the top-down panel
MIN_WALK_MPS = 0.5
LEVEL_COLOR = {"ATTENTION": (0, 200, 255), "WARNING": (0, 140, 255), "DANGER": (40, 40, 230)}
OK, BAD, PATH_C, GREY = (90, 200, 90), (70, 70, 230), (255, 190, 90), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def walking_speed(frames, t0, length_s):
    """Mean speed (m/s) over [t0, t0+length] from each frame's 3 s future path."""
    speeds = [np.linalg.norm(f["path"][-1] - f["path"][0]) / ev.HORIZON_S
              for f in frames if t0 <= f["t"] <= t0 + length_s]
    return float(np.mean(speeds)) if speeds else 0.0


def pick_window(episodes, frames, duration_s, length_s):
    """Window with the most episode starts among those where the wearer walks
    (mean speed >= MIN_WALK_MPS); earliest on ties. Falls back to all windows."""
    starts = sorted(e["start"] for e in episodes)
    candidates = sorted({max(0.0, min(t - 2.0, duration_s - length_s)) for t in [0.0] + starts})
    scored = [(sum(t0 <= s < t0 + length_s for s in starts), walking_speed(frames, t0, length_s), t0)
              for t0 in candidates]
    walking = [c for c in scored if c[1] >= MIN_WALK_MPS] or scored
    best = max(walking, key=lambda c: (c[0], -c[2]))
    return best[2], best[1]


def text(img, s, org, scale=0.5, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def world_to_bev(xy, origin, heading):
    """World XY -> panel pixels, device at bottom-centre, heading up."""
    h = heading / (np.linalg.norm(heading) + 1e-9)
    right = np.array([h[1], -h[0]])
    d = np.atleast_2d(xy)[:, :2] - origin[:2]
    fwd, lat = d @ h, d @ right
    return np.stack([BEV_W / 2 + lat * PX_PER_M, DEVICE_Y - fwd * PX_PER_M], axis=1)


def draw_bev(fd, traj, cam, points_xyz, ground, t_ns, alerts_now):
    img = np.full((BEV, BEV_W, 3), 18, np.uint8)
    R_wd, pos = traj.pose(t_ns)
    origin = pos[:2]
    walk = fd["path"][min(10, len(fd["path"]) - 1)] - fd["path"][0]   # next 1 s
    heading = walk if np.linalg.norm(walk) > 0.3 else (R_wd @ cam.R_dc @ np.array([0, 0, 1.0]))[:2]
    for m in range(-20, 20):  # 1 m grid
        y = int(DEVICE_Y - m * PX_PER_M)
        cv2.line(img, (0, y), (BEV_W, y), (28, 28, 28), 1)
        x = int(BEV_W / 2 + m * PX_PER_M)
        cv2.line(img, (x, 0), (x, BEV), (28, 28, 28), 1)
    # obstacles' height band from the MPS points, faint, for context only
    near = points_xyz[np.hypot(*(points_xyz[:, :2] - origin).T) < 11]
    near = near[(near[:, 2] > ground + 0.4) & (near[:, 2] < ground + 1.6)][::6]
    for u, v in world_to_bev(near, origin, heading).astype(int):
        if 0 <= u < BEV_W and 0 <= v < BEV:
            img[v, u] = (72, 72, 72)
    past = np.array([traj.position(t_ns - int(s * 1e9))[:2] for s in np.arange(0, 10, 0.2)])
    cv2.polylines(img, [world_to_bev(past, origin, heading).astype(np.int32)], False, GREY, 2, cv2.LINE_AA)
    fut = world_to_bev(fd["path"], origin, heading).astype(np.int32)
    overlay = img.copy()
    cv2.polylines(overlay, [fut], False, PATH_C, int(2 * ev.CORRIDOR_M * PX_PER_M), cv2.LINE_AA)
    cv2.circle(overlay, tuple(fut[0]), int(ev.CORRIDOR_M * PX_PER_M), PATH_C, -1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
    cv2.polylines(img, [fut], False, PATH_C, 2, cv2.LINE_AA)
    alerted = {a["track_id"]: a for a in alerts_now}
    for tr in fd["tracks"]:
        u, v = world_to_bev(tr["ground_xy"], origin, heading)[0].astype(int)
        if not (-20 <= u < BEV_W + 20 and -20 <= v < BEV + 20):
            continue
        color = (60, 60, 240) if tr["in_path"] else (190, 190, 190)
        cv2.circle(img, (u, v), 7, color, -1, cv2.LINE_AA)
        text(img, tr["name"], (u + 10, v + 5), 0.45, color)
        if tr["id"] in alerted:
            a = alerted[tr["id"]]
            cv2.circle(img, (u, v), 15, OK if a["justified"] else BAD, 3, cv2.LINE_AA)
    cv2.circle(img, (BEV_W // 2, DEVICE_Y), 9, (255, 255, 255), -1, cv2.LINE_AA)
    text(img, "Truth: where the wearer actually walked (Meta MPS SLAM, 1 m grid)", (12, 24), 0.55, PATH_C)
    text(img, "blue band = next 3 s of path   red = object in that band   grey line = last 10 s", (12, 48), 0.45, (200, 200, 200))
    text(img, "ring on an object = aria-guard alerted about it (green justified, red not)", (12, 70), 0.45, (200, 200, 200))
    return img


def draw_cam(rgb, fd, alerts_now, alert_banner):
    img = cv2.resize(rgb, (CAM, CAM))
    s = CAM / ev.RGB_SIZE
    for tr in fd["tracks"] if fd else []:
        x, y, w, h = [int(v * s) for v in tr["bbox"]]
        if tr["threat"] == "NONE":
            cv2.rectangle(img, (x, y), (x + w, y + h), (170, 170, 170), 1)
            continue
        color = LEVEL_COLOR[tr["threat"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text(img, f'{tr["name"]} {tr["threat"].lower()}', (x, max(14, y - 5)), 0.45, color)
    text(img, "aria-guard (yolo26n_nav + depth + tracker + arbiter)", (10, 22), 0.5)
    if alert_banner:
        a = alert_banner
        verdict = "justified" if a["justified"] else "not in path"
        cv2.rectangle(img, (0, CAM - 44), (CAM, CAM), (20, 20, 20), -1)
        text(img, f'ALERT {a["level"]}: {a["name"]}', (10, CAM - 18), 0.7, LEVEL_COLOR[a["level"]], 2)
        text(img, verdict, (CAM - 150, CAM - 18), 0.6, OK if a["justified"] else BAD, 2)
    return img


def draw_timeline(t, t0, length, episodes, alerts, metrics):
    img = np.full((TL_H, W, 3), 16, np.uint8)
    x_of = lambda tt: int(20 + (tt - t0) / length * (W - 40))
    text(img, "hazard episodes (object entered the wearer's path)", (20, 22), 0.45, (60, 60, 240))
    for e in episodes:
        if e["end"] >= t0 and e["start"] <= t0 + length:
            cv2.rectangle(img, (x_of(max(e["start"], t0)), 32), (x_of(min(e["end"] + 0.1, t0 + length)), 58), (60, 60, 240), -1)
    text(img, "alerts (green = justified, red = object never in path)", (20, 86), 0.45)
    for a in alerts:
        if t0 <= a["t"] <= t0 + length:
            x = x_of(a["t"])
            cv2.line(img, (x, 94), (x, 124), OK if a["justified"] else BAD, 3)
    cv2.line(img, (x_of(t), 28), (x_of(t), 128), (255, 255, 255), 1)
    text(img, (f'Whole recording: alert precision {metrics["alert_precision"]:.0%} | '
               f'episodes warned {metrics["episode_recall"]:.0%} | '
               f'{metrics["unjustified_alerts_per_min"]:.1f} unjustified alerts/min'),
         (20, TL_H - 12), 0.5, (230, 230, 230))
    text(img, "Data: Reading in the Wild, Project Aria (CC BY-NC 4.0)", (W - 430, TL_H - 12), 0.42, GREY)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--details", required=True, type=Path, help="pickle from evaluate_alerts.py --details-out")
    ap.add_argument("--recording", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", default="ttc_fix")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--gif-seconds", type=float, default=10.0,
                    help="Also write a GIF of the N seconds of the clip with the most alerts (0 = none)")
    args = ap.parse_args()

    import pickle
    with open(args.details, "rb") as f:
        result = pickle.load(f)[args.variant]
    cam = ev.RgbCamera(args.recording / "recording.vrs")
    traj = ev.Trajectory(args.recording / "mps" / "slam" / "closed_loop_trajectory.csv")
    pts = ev.Points(args.recording / "mps" / "slam" / "semidense_points.csv.gz")
    det = result["details"]
    duration = det["frames"][-1]["t"]
    t0, speed = pick_window(det["episodes"], det["frames"], duration, args.seconds)
    print(f"[DEMO] window {t0:.1f}-{t0 + args.seconds:.1f} s of {duration:.0f} s (walking {speed:.2f} m/s), "
          f"precision {result['alert_precision']}, recall {result['episode_recall']}")

    rec = VrsRecording(str(args.recording / "recording.vrs"), 100)
    by_t = sorted(det["frames"], key=lambda f: f["t"])
    times = np.array([f["t"] for f in by_t])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out.with_suffix(".tmp.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    alert_times = [a["t"] - t0 for a in det["alerts"] if t0 <= a["t"] <= t0 + args.seconds]
    gif_start = max(np.arange(0, max(0.0, args.seconds - args.gif_seconds) + 0.5, 0.5),
                    key=lambda g: sum(g <= a < g + args.gif_seconds for a in alert_times)) if alert_times else 0.0
    banner, banner_until = None, -1.0
    for t in np.arange(t0, t0 + args.seconds, 1.0 / args.fps):
        fd = by_t[int(np.abs(times - t).argmin())]
        t_ns = det["t0_ns"] + int(fd["t"] * 1e9)
        alerts_now = [a for a in det["alerts"] if abs(a["t"] - fd["t"]) < 1.0]
        for a in det["alerts"]:
            if fd["t"] - 1.0 / args.fps < a["t"] <= fd["t"]:
                banner, banner_until = a, fd["t"] + 1.5
        if fd["t"] > banner_until:
            banner = None
        _, rgb, _ = rec.read(fd["frame_idx"])
        R_wd, pos = traj.pose(t_ns)
        ground = pts.ground_z(R_wd @ cam.t_dc + pos) or (pos[2] - 1.5)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:CAM, :CAM] = draw_cam(rgb, fd, alerts_now, banner)
        canvas[:CAM, CAM:] = draw_bev(fd, traj, cam, pts.xyz, ground, t_ns, alerts_now)
        canvas[CAM:, :] = draw_timeline(fd["t"], t0, args.seconds, det["episodes"], det["alerts"], result)
        writer.write(canvas)
    writer.release()
    tmp = args.out.with_suffix(".tmp.mp4")
    # H.264 for browsers and GitHub; keep mp4v if ffmpeg is missing
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "30", "-preset", "slow", "-movflags", "+faststart", str(args.out)], check=True)
        tmp.unlink()
        if args.gif_seconds > 0:
            gif = args.out.with_suffix(".gif")
            vf = ("fps=6,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{gif_start:.1f}", "-t", str(args.gif_seconds),
                            "-i", str(args.out), "-vf", vf, str(gif)], check=True)
            print(f"[DEMO] {gif}")
    except (OSError, subprocess.CalledProcessError):
        tmp.rename(args.out)
    print(f"[DEMO] {args.out}")


if __name__ == "__main__":
    main()
