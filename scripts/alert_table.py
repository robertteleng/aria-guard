#!/usr/bin/env python3
"""
Markdown tables from alert evaluation records (benchmarks/alerts/*.json).

Pooled metrics sum counts over recordings (not an average of ratios). The
variant comparison also reports in how many recordings each metric moved, so a
pooled difference driven by one recording is visible.

Usage:
    python scripts/alert_table.py benchmarks/alerts/*.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def pooled(results: List[dict]) -> dict:
    alerts = sum(r["alerts"] for r in results)
    just = sum(r["justified_alerts"] for r in results)
    eps = sum(r["episodes"] for r in results)
    warned = sum(r["warned_episodes"] for r in results)
    minutes = sum(r["alert_benchmark"]["total_seconds"] for r in results) / 60
    return {
        "alerts": alerts, "justified": just, "episodes": eps, "warned": warned, "minutes": minutes,
        "precision": just / alerts if alerts else None,
        "recall": warned / eps if eps else None,
        "unjustified_per_min": (alerts - just) / minutes if minutes else None,
    }


def pct(v):
    return "—" if v is None else f"{v * 100:.1f} %"


def num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def load(paths: List[Path]) -> List[dict]:
    recs = [json.loads(p.read_text()) for p in paths]
    return [r for r in recs if str(r.get("schema", "")).startswith("aria-guard/alert-eval/")]


def by_variant(records: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for rec in records:
        for res in rec["results"]:
            out.setdefault(res["variant"], []).append({**res, "recording": rec["recording"]})
    return out


def recording_table(records: List[dict], variant: str) -> str:
    lines = ["| Recording | Channel-A alerts | Alert precision | Hazard episodes | Episode recall | "
             "Lead time median (s) | Unjustified alerts/min | Reference check (headline) |",
             "|---|---|---|---|---|---|---|---|"]
    for rec in sorted(records, key=lambda r: r["recording"]):
        res = next(r for r in rec["results"] if r["variant"] == variant)
        cc = rec["reference"]["crosscheck"].get("headline", {})
        check = (f'{cc["median_abs_diff_m"]:.2f} m, {cc["share_within_25pct"] * 100:.0f} % within 25 % (n={cc["n"]})'
                 if cc.get("n") else "—")
        lines.append(f'| `{rec["recording"][-6:]}` | {res["alerts"]} | {pct(res["alert_precision"])} | {res["episodes"]} | '
                     f'{pct(res["episode_recall"])} | {num(res["lead_time_s"]["median"])} | '
                     f'{num(res["unjustified_alerts_per_min"])} | {check} |')
    return "\n".join(lines)


def variant_table(groups: Dict[str, List[dict]], order: List[str]) -> str:
    lines = ["| Variant | Channel-A alerts | Alert precision | Hazard episodes | Episode recall | Unjustified alerts/min |",
             "|---|---|---|---|---|---|"]
    for v in order:
        if v not in groups:
            continue
        p = pooled(groups[v])
        lines.append(f'| {v} | {p["alerts"]} | {pct(p["precision"])} ({p["justified"]}) | {p["episodes"]} | '
                     f'{pct(p["recall"])} ({p["warned"]}) | {num(p["unjustified_per_min"])} |')
    if len(order) == 2 and all(v in groups for v in order):
        a = {r["recording"]: r for r in groups[order[0]]}
        b = {r["recording"]: r for r in groups[order[1]]}
        common = sorted(set(a) & set(b))

        def moved(key):
            up = sum((b[c][key] or 0) > (a[c][key] or 0) for c in common)
            down = sum((b[c][key] or 0) < (a[c][key] or 0) for c in common)
            return f"up in {up}, down in {down}, equal in {len(common) - up - down} of {len(common)}"
        lines.append("")
        lines.append(f"{order[1]} vs {order[0]} per recording: precision {moved('alert_precision')}; "
                     f"recall {moved('episode_recall')}; unjustified alerts/min {moved('unjustified_alerts_per_min')}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Tables from alert evaluation records")
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--variants", nargs="+", default=["pre_ttc", "ttc_fix"])
    args = ap.parse_args()
    records = load(args.records)
    if not records:
        sys.exit("[ERROR] no alert evaluation records")
    groups = by_variant(records)
    print("### Pooled over recordings\n")
    print(variant_table(groups, args.variants))
    for v in args.variants:
        print(f"\n### Per recording · {v}\n")
        print(recording_table(records, v))


if __name__ == "__main__":
    main()
