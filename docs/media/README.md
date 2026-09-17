# Media

All files are generated from `scripts/evaluate_alerts.py --details-out` with
`scripts/render_demo.py`, on recording `recording_968813465288489`, window
57.2–97.2 s (the 40 s with the most hazard episodes while the wearer walks).

| File | What it shows |
|---|---|
| `aria-guard-metric-inpath.mp4` / `.gif` | Adopted threat model: aria-guard's alerts next to the wearer's real path from Meta's MPS SLAM |
| `aria-guard-heuristic.mp4` | Previous threat model on the same window |
| `before-after.png` | Both models' alerts on the same window, with pooled metrics over the six recordings |

```bash
python scripts/render_demo.py --details <details.pkl> --variant metric_inpath --start 57.2 \
    --recording ~/Datasets/aria/ritw/recording_968813465288489 --out docs/media/aria-guard-metric-inpath.mp4
python scripts/render_demo.py --details <details.pkl> --compare ttc_fix metric_inpath --start 57.2 \
    --records benchmarks/alerts/candidate1/*.json --recording ... --out docs/media/before-after.png
```

**License of these files:** they contain frames from the Project Aria *Reading in
the Wild* dataset, licensed **CC BY-NC 4.0** (non-commercial, attribution). They
are not covered by this repository's AGPL-3.0 license. Attribution: Project
Aria, Meta, Reading in the Wild dataset.
