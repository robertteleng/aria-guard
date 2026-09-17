#!/usr/bin/env bash
# Replay every recording.vrs under DATA_DIR twice and write one JSON per run:
#   realtime pacing (effective FPS, drops, alerts as live) and
#   all frames with per-stage breakdown (per-frame cost).
# PYTHON is the interpreter command for this machine, e.g.
#   PYTHON="uv run python" (x86)  or  PYTHON=python3 (inside the Jetson container)
set -euo pipefail
DATA_DIR=${1:?usage: replay_benchmark.sh DATA_DIR OUT_DIR}
OUT_DIR=${2:?usage: replay_benchmark.sh DATA_DIR OUT_DIR}
PYTHON=${PYTHON:-python3}
MODE=${MODE:-outdoor}
export ARIA_YOLO_MODEL=${ARIA_YOLO_MODEL:-yolo26n_nav}
host=$(hostname -s)
mkdir -p "$OUT_DIR"
cd "$(dirname "$0")/.."
for vrs in "$DATA_DIR"/*/recording.vrs; do
  seq=$(basename "$(dirname "$vrs")")
  for pace in realtime all; do
    extra=()
    [ "$pace" = all ] && extra=(--breakdown)
    out="$OUT_DIR/${host}_${seq}_${ARIA_YOLO_MODEL}_${pace}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "=== $seq $pace"
    $PYTHON scripts/replay_vrs.py "$vrs" --pace "$pace" --mode "$MODE" "${extra[@]}" --out "$out" 2>&1 \
      | grep -E '^\[REPLAY\]|Traceback|Error' || true
    [ -f "$out" ] || { echo "FAILED $seq $pace"; exit 1; }
  done
done
echo "DONE"
