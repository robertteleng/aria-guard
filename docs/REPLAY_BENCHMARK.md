# Replay benchmark: real Aria recordings through the full pipeline

This benchmark answers one question: **on real egocentric walking footage, how
fast and how noisy is aria-guard on an RTX desktop and on a Jetson Orin Nano?**
It replaces the earlier performance table, which was measured on an RTX 2060
with the base COCO YOLO and synthetic input, and whose raw results were not kept.

## Data

Six outdoor recordings from **Reading in the Wild** (Project Aria, Gen 1
glasses, CC BY-NC 4.0), 36.9 minutes in total. They were chosen so the input
matches aria-guard's use: a person walking outdoors among parked cars,
crosswalks, curbs, stairs and pedestrians.

How they were selected, so the choice can be checked:

1. All 11 Project Aria open datasets were screened with the metadata the
   Dataset Explorer publishes per recording (`light_intensity_lux_median`,
   `speed_mps_mean`, `trajectory_length_m`).
2. Candidates: median illuminance > 5,000 lux (daylight), mean speed ≥ 0.35 m/s
   and trajectory ≥ 60 m.
3. Every candidate was then reviewed visually from its thumbnails. That review
   removed datasets the metadata alone would have kept:
   - Aria Digital Twin: indoor only (apartment and office).
   - Nymeria "Fresh_air": mostly **cycling** on a campus.
   - Nymeria "Hike": forest trails.
   - Aria Scenes: 12 short scenes at 10 FPS.
   - Aria Gen 2 Pilot: a different device, with different stream IDs.
4. From the 64 Reading in the Wild candidates, six were kept to cover stairs
   with handrails, crosswalks, parking lots, sidewalks and a fire hydrant.

| Recording | Duration | What it contains |
|---|---|---|
| `recording_1446119809363548` | 6 min 13 s | Stairs with handrails, crosswalk, cars, pedestrian |
| `recording_1723524325073193` | 6 min 03 s | Parking lot, stairs, curb |
| `recording_2874281686067074` | 6 min 29 s | Path, parking lot (longest trajectory, 422 m) |
| `recording_547476497950519` | 6 min 57 s | Crosswalk, parked cars |
| `recording_2263006570727110` | 6 min 04 s | Parking lot with many cars |
| `recording_968813465288489` | 5 min 07 s | Sidewalk, fire hydrant, parked cars |

## What runs

`scripts/replay_vrs.py` reads the VRS file and runs the **same code as live
streaming**:

- RGB (1408×1408, 30 FPS) and the eye-tracking image nearest in time, with the
  live orientation (`src/input/aria_frames.py`, shared with the live observer).
- `ParallelDetector`: `yolo26n_nav` TensorRT FP16, Depth Anything V2 Small
  TensorRT FP16 every 5th frame, and Meta eye gaze TensorRT FP16 every 2nd frame.
  Each record stores the YOLO backend (`ultralytics` predictor or the engine
  called directly) and whether depth ran asynchronously.
- The IMU drives the same walking/stationary classifier the tracker receives live.
- `SimpleTracker` and `AlertArbiter`, run on recording time.

Engines are built on each device from the same ONNX files
(`scripts/build_trt_engine.py`); the YOLO engine comes from vision-fine-tuning.

## Two pacing modes

| Mode | How | Answers |
|---|---|---|
| `realtime` | A decoder thread delivers each frame at its capture time, as the SDK does; the detector always takes the newest frame, so frames it is too slow for are dropped | Effective FPS, drop rate, capture → detections latency and alerts **under real load** |
| `all` | Every frame, one after another, with the GPU synchronized around each stage | Cost per frame and per stage |

`scripts/replay_benchmark.sh` runs both modes on every recording (`PACES`
selects modes, `TAG` labels a variant such as an optimization pass) and writes
one JSON record per run: environment (GPU, driver, TensorRT, torch, L4T and power
mode on Jetson), commit, model file hash, settings and results. Realtime runs
also keep per-frame detections, so alert logic can be re-evaluated offline with
`scripts/benchmark_offline.py --from-json` without the GPU. Tables are generated
with `scripts/replay_table.py`, never written by hand.

## Alert criteria

Fixed in `scripts/benchmark_offline.py` before this benchmark:

| Criterion | Target |
|---|---|
| Seconds without any alert | ≥ 80 % |
| Threat alerts per minute | ≤ 12 |
| Minimum gap between threat alerts | ≥ 1 s |
| Alerts firing in the same frame | ≤ 1 |

These measure **alert fatigue, not correctness**, and one is met by
construction: the arbiter itself caps non-DANGER alerts at 6 per 30 s (12 per
minute), and the recordings sat at 9–11. A pipeline that never alerts passes
all four. Whether alerts are deserved is measured separately against the
wearer's real path: [ALERT_EVALUATION.md](ALERT_EVALUATION.md).

## What this does not measure

- **Transport.** Frames come from disk, not over WiFi. The live WiFi and USB
  numbers are in aria-arm64-bridge's findings.
- **Audio.** No sound is played; the latency from alert to sound is not included.
- **The detector subprocess and shared memory.** Detection runs in process.
- **IMU batching.** The SDK delivers IMU samples in batches and the live
  callback reads only the first sample of each batch. The replay feeds the
  classifier 100 samples per second, an assumption recorded in each JSON
  (`imu_target_hz`).
- **Per-stage times in `all` mode are serialized.** Synchronizing the GPU around
  each stage removes the overlap between CUDA streams, so the stages add up to
  more than the end-to-end detector time.
