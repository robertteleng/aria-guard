#!/usr/bin/env python3
"""
Where do wrong and missing alerts come from? Exploratory breakdown of the
per-frame evaluation details (evaluate_alerts.py --details-out), no thresholds
tuned and no decision taken here.

Unjustified alerts are split by what the reference says about the alerted
object at alert time; missed hazard episodes by what the tracker did during
them. Output: Markdown tables on stdout.

Usage:
    python scripts/alert_error_analysis.py --details <dir>/c2_*.pkl --variant metric_inpath_selfbody
"""
import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BOTTOM_EDGE_PX = 1380          # box reaching the image bottom (1408 px)
NEAR_PATH_M = 1.5              # beside the path, not in it
LATER_S = 6.0                  # "in the path later": between 3 and 6 s after the alert
WARN_LOOKBACK_S = 3.0          # same window evaluate_alerts uses to count a warning
MATCH_S = 0.5                  # look for the alerted track this close to the alert time
SHORT_EPISODE_S = 0.3


def track_near(frames, times, t, track_id):
    """The alerted track's record in the closest frame within MATCH_S that contains it."""
    lo, hi = np.searchsorted(times, t - MATCH_S), np.searchsorted(times, t + MATCH_S, side="right")
    cands = [(abs(frames[i]["t"] - t), tr) for i in range(lo, hi) for tr in frames[i]["tracks"] if tr["id"] == track_id]
    return min(cands, key=lambda c: c[0])[1] if cands else None


def classify_unjustified(a, det, frames, times):
    """One category per unjustified alert."""
    x, y, w, h = a["bbox"]
    if a["name"] == "person" and y + h >= BOTTOM_EDGE_PX:
        return "wearer's body (person at bottom edge)"
    tr = track_near(frames, times, a["t"], a["track_id"])
    if tr is None:
        return "no reference position within 0.5 s"
    later = [t for f in frames if a["t"] + 3.0 < f["t"] <= a["t"] + LATER_S
             for t in f["tracks"] if t["id"] == a["track_id"] and t["in_path"]]
    if later:
        return "in the path, but more than 3 s later"
    if tr["path_distance"] <= NEAR_PATH_M:
        return f"beside the path (≤ {NEAR_PATH_M} m)"
    return f"away from the path (> {NEAR_PATH_M} m)"


def classify_missed(ep, det, frames, times):
    """One category per hazard episode without a warning."""
    lo, hi = ep["start"] - WARN_LOOKBACK_S, ep["closest_time"]
    seen = [t for f in frames if lo <= f["t"] <= hi for t in f["tracks"] if t["id"] == ep["track_id"]]
    alerts_other = [a for a in det["alerts"] if lo <= a["t"] <= hi and a["track_id"] != ep["track_id"]]
    if ep["end"] - ep["start"] < SHORT_EPISODE_S:
        return f"very short episode (< {SHORT_EPISODE_S} s)"
    if not seen or all(t["threat"] == "NONE" for t in seen):
        return "model never rated it a threat"
    if alerts_other:
        return "rated a threat, but the alert went to another object"
    return "rated a threat, no alert (cooldown or level too low)"


def table(counter: Counter, total: int, title: str, head: str) -> str:
    lines = [f"### {title}", "", f"| {head} | Count | Share |", "|---|---|---|"]
    for k, v in counter.most_common():
        lines.append(f"| {k} | {v} | {v / total:.0%} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--details", nargs="+", type=Path, required=True)
    ap.add_argument("--variant", default="metric_inpath_selfbody")
    args = ap.parse_args()

    unj_cat, unj_class, unj_level, just_level = Counter(), Counter(), Counter(), Counter()
    miss_cat, miss_class, warned_class = Counter(), Counter(), Counter()
    short_eps = 0
    per_rec = defaultdict(Counter)
    n_alerts = n_unj = n_eps = n_missed = 0
    for path in sorted(args.details):
        det = pickle.load(open(path, "rb"))[args.variant]["details"]
        frames = sorted(det["frames"], key=lambda f: f["t"])
        times = np.array([f["t"] for f in frames])
        name = path.stem.split("recording_")[-1][-6:]
        for a in det["alerts"]:
            n_alerts += 1
            if a["justified"]:
                just_level[a["level"]] += 1
                continue
            n_unj += 1
            cat = classify_unjustified(a, det, frames, times)
            unj_cat[cat] += 1
            unj_class[a["name"]] += 1
            unj_level[a["level"]] += 1
            per_rec[name][cat] += 1
        for ep in det["episodes"]:
            n_eps += 1
            short_eps += ep["end"] - ep["start"] < SHORT_EPISODE_S
            cls = next((t["name"] for f in frames for t in f["tracks"] if t["id"] == ep["track_id"]), "?")
            lo, hi = ep["start"] - WARN_LOOKBACK_S, ep["closest_time"]
            if any(a["track_id"] == ep["track_id"] and lo <= a["t"] <= hi for a in det["alerts"]):
                warned_class[cls] += 1
                continue
            n_missed += 1
            miss_cat[classify_missed(ep, det, frames, times)] += 1
            miss_class[cls] += 1

    print(f"## Alert error analysis · {args.variant}\n")
    print(f"{n_alerts} alerts, {n_unj} unjustified ({n_unj / n_alerts:.0%}); "
          f"{n_eps} hazard episodes, {n_missed} without warning ({n_missed / n_eps:.0%}).\n")
    print(table(unj_cat, n_unj, "Unjustified alerts: what the reference says about the object", "Category"))
    print(table(unj_class, n_unj, "Unjustified alerts by class", "Class"))
    print("### Alerts by level\n\n| Level | Justified | Unjustified | Precision |\n|---|---|---|---|")
    for lvl in ("DANGER", "WARNING", "ATTENTION"):
        j, u = just_level[lvl], unj_level[lvl]
        print(f"| {lvl} | {j} | {u} | {j / (j + u):.0%} |" if j + u else f"| {lvl} | 0 | 0 | — |")
    print()
    print(table(miss_cat, n_missed, "Missed hazard episodes: what the tracker did", "Category"))
    print("### Hazard episodes by class\n\n| Class | Warned | Missed | Recall |\n|---|---|---|---|")
    for cls in sorted(set(miss_class) | set(warned_class), key=lambda c: -(miss_class[c] + warned_class[c])):
        w, m = warned_class[cls], miss_class[cls]
        print(f"| {cls} | {w} | {m} | {w / (w + m):.0%} |")
    print()
    print(f"Hazard episodes shorter than {SHORT_EPISODE_S} s: {short_eps} of {n_eps} ({short_eps / n_eps:.0%}).\n")
    print("### Unjustified alerts per recording\n")
    cats = [c for c, _ in unj_cat.most_common()]
    print("| Recording | " + " | ".join(cats) + " |")
    print("|---|" + "---|" * len(cats))
    for rec in sorted(per_rec):
        print(f"| `{rec}` | " + " | ".join(str(per_rec[rec][c]) for c in cats) + " |")


if __name__ == "__main__":
    main()
