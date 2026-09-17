#!/usr/bin/env python3
"""
Render "what aria-guard says vs what the wearer actually walked into".

Left: the glasses' RGB with aria-guard's tracks and alerts. Right: top-down view
from Meta's MPS (metric trajectory, 3D points), the 3 s path corridor and the
objects' ground positions. Bottom: timeline of hazard episodes vs alerts, each
alert marked justified or not. Everything drawn comes from
scripts/evaluate_alerts.py; windows are chosen by fixed rules written in
docs/media/README.md, not by hand.

Usage:
    python scripts/evaluate_alerts.py ... --details-out /tmp/details.pkl
    python scripts/render_demo.py --details /tmp/details.pkl \\
        --recording ~/Datasets/aria/ritw/recording_XXXX --out docs/media/demo.mp4
    # README hero clip: rule applied over every recording's details
    python scripts/render_demo.py --hero --details-dir DIR --recordings-root ~/Datasets/aria/ritw \\
        --records benchmarks/alerts/candidate2/*.json --out docs/media/aria-guard-hero.mp4
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_alerts as ev  # noqa: E402

W, H = 1280, 720
CAM = 560                 # camera panel side
BEV_W, BEV = W - CAM, 560  # top-down panel
TL_H = H - CAM            # timeline height
PX_PER_M = 36.0
DEVICE_Y = BEV - 70       # device position in the top-down panel
MIN_WALK_MPS = 0.5
HERO_SECONDS, HERO_LEAD_S = 15.0, 5.0
HERO_VARIANT, HERO_DETAILS_PREFIX = "metric_inpath_selfbody", "c2_"  # adopted model, corrected reference
LEVEL_COLOR = {"ATTENTION": (0, 200, 255), "WARNING": (0, 140, 255), "DANGER": (40, 40, 230)}
OK, BAD, PATH_C, GREY = (90, 200, 90), (70, 70, 230), (255, 190, 90), (150, 150, 150)
HAZARD = (60, 60, 240)
FONT_FILES = {False: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              True: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}
VARIANT_LABEL = "heuristic"
VARIANT_LABELS = {"ttc_fix": "heuristic (image thirds)", "pre_ttc": "heuristic, pre TTC fix",
                  "metric_inpath": "metric in-path (metres)",
                  "metric_inpath_selfbody": "metric in-path + wearer filter"}


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


def warned_in_window(det, t0, length_s):
    """Hazard episodes starting in the window with a justified alert on the same track inside it."""
    justified = {a["track_id"] for a in det["alerts"] if a["justified"] and t0 <= a["t"] < t0 + length_s}
    return sum(1 for e in det["episodes"] if t0 <= e["start"] < t0 + length_s and e["track_id"] in justified)


def pick_hero_window(details_by_recording, length_s=HERO_SECONDS, lead_s=HERO_LEAD_S, min_speed=MIN_WALK_MPS):
    """README hero clip rule (docs/media/README.md): windows starting lead_s before each
    justified alert, walking only, score = justified - unjustified alerts; ties: more
    episodes warned, recording name order, earliest start. Returns (recording, t0, score)."""
    best, best_key = None, None
    for order, name in enumerate(sorted(details_by_recording)):
        det = details_by_recording[name]
        duration = det["frames"][-1]["t"]
        for a in det["alerts"]:
            if not a["justified"]:
                continue
            t0 = max(0.0, min(a["t"] - lead_s, duration - length_s))
            if walking_speed(det["frames"], t0, length_s) < min_speed:
                continue
            inside = [x for x in det["alerts"] if t0 <= x["t"] < t0 + length_s]
            score = sum(x["justified"] for x in inside) - sum(not x["justified"] for x in inside)
            key = (score, warned_in_window(det, t0, length_s), -order, -t0)
            if best_key is None or key > best_key:
                best, best_key = (name, t0, score), key
    return best


_fonts = {}


def _font(px, bold):
    from PIL import ImageFont
    if (px, bold) not in _fonts:
        _fonts[(px, bold)] = ImageFont.truetype(FONT_FILES[bold], px)
    return _fonts[(px, bold)]


def draw_texts(img, items):
    """items: (text, (x, y) top-left, size px, BGR color, bold). DejaVu via Pillow; OpenCV Hershey fallback."""
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(np.ascontiguousarray(img[..., ::-1]))
        d = ImageDraw.Draw(pil)
        for s, (x, y), px, color, bold in items:
            d.text((x, y), s, font=_font(px, bold), fill=tuple(int(c) for c in color[::-1]),
                   stroke_width=2, stroke_fill=(0, 0, 0))
        img[:] = np.asarray(pil)[..., ::-1]
    except (ImportError, OSError):
        for s, (x, y), px, color, bold in items:
            scale, thick = px / 32.0, 2 if bold else 1
            cv2.putText(img, s, (x, y + px), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(img, s, (x, y + px), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return img


def world_to_bev(xy, origin, heading):
    """World XY -> panel pixels, device at bottom-centre, heading up."""
    h = heading / (np.linalg.norm(heading) + 1e-9)
    right = np.array([h[1], -h[0]])
    d = np.atleast_2d(xy)[:, :2] - origin[:2]
    fwd, lat = d @ h, d @ right
    return np.stack([BEV_W / 2 + lat * PX_PER_M, DEVICE_Y - fwd * PX_PER_M], axis=1)


def draw_bev(fd, traj, cam, points_xyz, ground, t_ns, alerts_now):
    img = np.full((BEV, BEV_W, 3), 18, np.uint8)
    texts = []
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
            img[v, u] = (58, 58, 58)
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
        color = HAZARD if tr["in_path"] else (190, 190, 190)
        cv2.circle(img, (u, v), 8, color, -1, cv2.LINE_AA)
        texts.append((tr["name"], (u + 13, v - 10), 17, color, False))
        if tr["id"] in alerted:
            cv2.circle(img, (u, v), 17, OK if alerted[tr["id"]]["justified"] else BAD, 4, cv2.LINE_AA)
    cv2.circle(img, (BEV_W // 2, DEVICE_Y), 10, (255, 255, 255), -1, cv2.LINE_AA)
    texts += [("You are here", (BEV_W // 2 + 16, DEVICE_Y + 8), 16, (235, 235, 235), False),
              ("Where the wearer really walked (Meta SLAM, 1 m grid)", (14, 10), 21, PATH_C, True),
              ("blue band: path walked in the next 3 s   ·   grey: last 10 s", (14, 40), 16, (215, 215, 215), False),
              ("red dot: object in that path   ·   ring: alert (green = justified, red = not)",
               (14, 62), 16, (215, 215, 215), False)]
    return draw_texts(img, texts)


def draw_cam(rgb, fd, alerts_now, alert_banner):
    img = cv2.resize(rgb, (CAM, CAM))
    texts = [(f"What aria-guard sees · {VARIANT_LABEL}", (12, 10), 20, (240, 240, 240), True)]
    s = CAM / ev.RGB_SIZE
    for tr in fd["tracks"] if fd else []:
        x, y, w, h = [int(v * s) for v in tr["bbox"]]
        if tr["threat"] == "NONE":
            cv2.rectangle(img, (x, y), (x + w, y + h), (170, 170, 170), 1)
            continue
        color = LEVEL_COLOR[tr["threat"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        texts.append((f'{tr["name"]} · {tr["threat"].lower()}', (x, max(36, y - 24)), 17, color, True))
    if alert_banner:
        a = alert_banner
        cv2.rectangle(img, (0, CAM - 58), (CAM, CAM), (20, 20, 20), -1)
        texts.append((f'ALERT {a["level"]}: {a["name"]}', (14, CAM - 46), 25, LEVEL_COLOR[a["level"]], True))
        texts.append(("✓ justified" if a["justified"] else "✗ not in path", (CAM - 175, CAM - 43), 21,
                      OK if a["justified"] else BAD, True))
    return draw_texts(img, texts)


def summary_line(metrics, pooled):
    if pooled:
        return (f'All six recordings ({pooled["minutes"]:.1f} min): {pooled["precision"]:.0%} of alerts justified  ·  '
                f'{pooled["recall"]:.0%} of hazards warned  ·  {pooled["unjustified_per_min"]:.1f} unjustified alerts/min')
    return (f'This recording: {metrics["alert_precision"]:.0%} of alerts justified  ·  '
            f'{metrics["episode_recall"]:.0%} of hazards warned  ·  '
            f'{metrics["unjustified_alerts_per_min"]:.1f} unjustified alerts/min')


def draw_timeline(t, t0, length, episodes, alerts, metrics, pooled=None):
    img = np.full((TL_H, W, 3), 16, np.uint8)
    x_of = lambda tt: int(20 + (tt - t0) / length * (W - 40))
    for e in episodes:
        if e["end"] >= t0 and e["start"] <= t0 + length:
            cv2.rectangle(img, (x_of(max(e["start"], t0)), 34), (x_of(min(e["end"] + 0.1, t0 + length)), 58), HAZARD, -1)
    for a in alerts:
        if t0 <= a["t"] <= t0 + length:
            x = x_of(a["t"])
            cv2.line(img, (x, 94), (x, 122), OK if a["justified"] else BAD, 4)
    cv2.line(img, (x_of(t), 30), (x_of(t), 126), (255, 255, 255), 1)
    return draw_texts(img, [
        ("hazards: an object was in the path the wearer walked", (20, 8), 16, HAZARD, True),
        ("alerts: green = justified, red = about an object never in the path", (20, 68), 16, (225, 225, 225), True),
        (summary_line(metrics, pooled), (20, TL_H - 28), 18, (240, 240, 240), True),
        ("Data: Reading in the Wild, Project Aria (CC BY-NC 4.0)", (W - 400, 8), 14, GREY, False)])


def render_comparison(details_path: Path, variants, t0: float, length: float, records, out: Path):
    """Static before/after: the same window's timeline for each variant, plus pooled metrics."""
    from scripts.alert_table import by_variant, pooled
    with open(details_path, "rb") as f:
        data = pickle.load(f)
    groups = by_variant(records)
    rows = [draw_texts(np.full((56, W, 3), 12, np.uint8), [
        (f"Same {length:.0f} s of walking, two threat models (hazards from the wearer's real path)",
         (20, 16), 24, (235, 235, 235), True)])]
    for v in variants:
        res = data[v]
        det = res["details"]
        tl = draw_timeline(t0 - 1, t0, length, det["episodes"], det["alerts"], res)
        p = pooled(groups[v])
        band = draw_texts(np.full((38, W, 3), 12, np.uint8), [
            (f"{VARIANT_LABELS.get(v, v)}  ·  six recordings: {p['precision']:.0%} of alerts justified, "
             f"{p['recall']:.0%} of hazards warned, {p['unjustified_per_min']:.1f} unjustified alerts/min",
             (20, 8), 20, OK if v == variants[-1] else (200, 200, 200), True)])
        rows += [band, tl[:TL_H - 34]]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"[DEMO] {out}")


def load_hero_details(details_dir: Path, variant: str, prefix: str = HERO_DETAILS_PREFIX):
    """{recording name: (path, details)} from evaluate_alerts pickles named <prefix><recording>.pkl."""
    out = {}
    for p in sorted(details_dir.glob(f"{prefix}*.pkl")):
        with open(p, "rb") as f:
            out[p.stem[len(prefix):]] = (p, pickle.load(f)[variant]["details"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--details", type=Path, help="pickle from evaluate_alerts.py --details-out")
    ap.add_argument("--recording", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", default="ttc_fix")
    ap.add_argument("--compare", nargs=2, metavar="VARIANT", help="Write a before/after PNG for two variants instead of video")
    ap.add_argument("--records", nargs="*", type=Path, default=[],
                    help="alert records: pooled numbers for --compare and on the video's summary line")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--start", type=float, help="Window start in seconds (default: the fixed selection rule); "
                                                   "use it to render another variant on the same window")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--gif-seconds", type=float, default=10.0,
                    help="Also write a GIF of the N seconds of the clip with the most alerts (0 = none)")
    ap.add_argument("--gif-width", type=int, default=800)
    ap.add_argument("--gif-fps", type=int, default=6)
    ap.add_argument("--hero", action="store_true",
                    help="README hero clip: pick recording and window with the rule in docs/media/README.md "
                         f"(sets --variant {HERO_VARIANT}, --seconds 15, GIF of the whole clip)")
    ap.add_argument("--details-dir", type=Path, help=f"--hero: folder with {HERO_DETAILS_PREFIX}<recording>.pkl files")
    ap.add_argument("--recordings-root", type=Path, help="--hero: folder with the recordings")
    args = ap.parse_args()

    global VARIANT_LABEL
    from scripts.alert_table import by_variant, load, pooled
    if args.compare:
        start = args.start if args.start is not None else 0.0
        render_comparison(args.details, args.compare, start, args.seconds, load(args.records), args.out)
        return
    if args.hero:
        args.variant, args.seconds, args.gif_seconds = HERO_VARIANT, HERO_SECONDS, HERO_SECONDS
        loaded = load_hero_details(args.details_dir, args.variant)
        name, args.start, score = pick_hero_window({k: v[1] for k, v in loaded.items()})
        args.details, args.recording = loaded[name][0], args.recordings_root / name
        print(f"[DEMO] hero rule -> {name} at {args.start:.1f} s (justified - unjustified = {score})")
    if args.details is None or args.recording is None:
        ap.error("--details and --recording are required (or --hero with --details-dir and --recordings-root)")
    VARIANT_LABEL = VARIANT_LABELS.get(args.variant, args.variant)
    pooled_metrics = pooled(by_variant(load(args.records))[args.variant]) if args.records else None
    with open(args.details, "rb") as f:
        result = pickle.load(f)[args.variant]
    from scripts.replay_vrs import VrsRecording
    cam = ev.RgbCamera(args.recording / "recording.vrs")
    traj = ev.Trajectory(args.recording / "mps" / "slam" / "closed_loop_trajectory.csv")
    pts = ev.Points(args.recording / "mps" / "slam" / "semidense_points.csv.gz")
    det = result["details"]
    duration = det["frames"][-1]["t"]
    t0, speed = pick_window(det["episodes"], det["frames"], duration, args.seconds)
    if args.start is not None:
        t0, speed = args.start, walking_speed(det["frames"], args.start, args.seconds)
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
        canvas[CAM:, :] = draw_timeline(fd["t"], t0, args.seconds, det["episodes"], det["alerts"], result, pooled_metrics)
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
            vf = (f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=max_colors=64:stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{gif_start:.1f}", "-t", str(args.gif_seconds),
                            "-i", str(args.out), "-vf", vf, str(gif)], check=True)
            print(f"[DEMO] {gif}")
    except (OSError, subprocess.CalledProcessError):
        tmp.rename(args.out)
    print(f"[DEMO] {args.out}")


if __name__ == "__main__":
    main()
