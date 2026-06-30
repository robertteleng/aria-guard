# Depth & approach audit — how the near/far critical alerts are computed

> Audit 2026-06-30 (session: ego-motion + bbox looming). What is involved in
> computing object depth and the near/far alerts today, and the structural
> limits that make them unreliable.

## Pipeline: RGB → depth → distance → threat → sound

1. **Depth model** — Depth Anything V2 Small (HF
   `depth-anything/Depth-Anything-V2-Small-hf`, or TensorRT engine
   `models/depth_anything_v2_vits.engine`) in the CUDA worker
   (`src/detection/process.py`). Frame resized to **518×518**, output resized
   back to frame size. **Relative depth only**, normalized **per frame** with
   `cv2.normalize(..., NORM_MINMAX)` → uint8 0–255 (higher = closer). Run every
   `depth_interval=5` frames and cached (`src/detection/detector.py`). RealSense
   hardware depth bypasses the AI model.
2. **Per-object depth** — `_get_depth_in_bbox()`: **mean of the central 50%** of
   the bbox (25% margin each side), `/255` → `depth_value ∈ [0,1]`. RealSense
   path uses the **median** of valid mm in the central crop (more robust).
3. **Distance buckets** — `_depth_to_distance()` (AI, relative): `>0.7`
   very_close, `>0.5` close, `>0.3` medium, else far. `_depth_mm_to_distance()`
   (RealSense, metric): `<800mm` very_close, `<2000` close, `<4000` medium, else
   far.
4. **Collision risk / threat** (`src/domain/tracker.py` `_update_collision_risk`)
   — weighted sum capped at 1.0: **TTC 50%** (`ttc = depth_value/approach_speed`,
   normalized over 150 frames), **CBDR 25%**, **zone 15%**, **class 10%**.
   Thresholds: DANGER ≥0.7, WARNING ≥0.35, ATTENTION ≥0.15. Static objects use
   `STATIC_PROXIMITY` (very_close 0.8 / close 0.4 / medium 0.1 / far 0.0).
5. **Arbiter** (`src/domain/alert_engine.py`) — Channel A (threat, top-1 by
   risk; cooldowns DANGER 2s / WARNING 4s / ATTENTION 6s; DANGER bypasses the
   rate limit; global ≤6 alerts/30s; ≥4 in 20s doubles cooldowns). Channel B
   (traffic lights / signs, only when A silent ≥3s). Gaze suppresses TTS (beep
   only); DANGER always speaks.

## Structural limitations (near/far reliability)

- **Per-frame NORM_MINMAX makes `depth_value` non-comparable across frames.**
  When the scene changes (objects enter/leave, camera pans) the same physical
  distance maps to a different value → corrupts `approach_speed` (the slope of
  `depth_history`) and the distance buckets. Root cause of "alertas mediocres"
  alongside ego-motion. (Acknowledged in `tracker.py` `_update_approach`.)
- **TTC is unitless** (`depth_value / slope`, both NORM_MINMAX artifacts); the
  150-frame constant has no metric grounding.
- **Distance thresholds 0.7/0.5/0.3 are not metric-calibrated** — 0.7 could be
  0.5 m or 5 m depending on scene content.
- **Stale depth** up to 5 frames (~167 ms @30 fps) from the cache.

## Mitigations landed this session

- **bbox-height looming** (`_update_approach`): a depth-model-independent
  approach cue `(Δheight/height)·depth_value`, fused with the depth slope via
  `max()`. Survives NORM_MINMAX noise on the depth channel; expressed in
  depth-slope units so TTC stays consistent (no arbitrary scale constant).
- **Ego-motion compensation**: subtract `EGO_MOTION_WALKING_BIAS = 0.05` from
  the fused approach when `motion_state == "walking"` so static obstacles ahead
  don't read as collisions. Motion state from IMU accel-std
  (`AriaDemoObserver.get_motion_state`; `AriaBridgeObserver.get_motion_state`).

## Not done (future, higher ROI)

- **Metric depth**: a metric model (or anchoring NORM_MINMAX to a stable
  reference) so `depth_value`, TTC and the buckets become physically meaningful.
  Tracked separately — do **not** conflate with the ego-motion fix.
- **IMU forward-velocity subtraction** instead of the fixed walking bias.
