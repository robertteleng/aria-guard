# ARIA Guard

> **L1 — Critical Reactive Layer** | Detección de colisiones en tiempo real para personas con discapacidad visual.

**Phase 1-3 ✅** · **Phase 4 ⏳ (ARM64/Jetson)**

## Problem

Pedestrians who are blind or have low vision face constant collision risks — especially from silent electric vehicles, aerial obstacles, and complex intersections. Existing solutions (white cane, guide dogs) don't detect fast-moving threats or provide directional awareness.

## Solution

Wearable real-time collision detection using YOLO + depth estimation + eye tracking, with spatial audio alerts calibrated by academic research (Gao et al. 2025, Nature Communications).

Supports multiple input sources:
- **Meta Aria Glasses** — RGB + Eye Tracking + Gaze (x86_64)
- **Intel RealSense D435** — RGB + Hardware Depth (x86_64 + ARM64/Jetson)
- **Webcam/Video** — RGB + AI Depth

## Architecture

```
Main Process (NO CUDA)        DetectorProcess (spawn)       TTSProcess (spawn)
├─ Observer (Aria/RS/Webcam)  ├─ YOLO26s TensorRT FP16     ├─ NeMo FastPitch+HiFiGAN
├─ Flask :5000 (MJPEG)       ├─ Depth Anything V2 TRT     └─ Audio playback
├─ Dashboard                  ├─ Meta Eye Gaze
└─ AudioFeedback (BRR+TTS)   └─ SimpleTracker + collision_risk()
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| **ParallelDetector** | YOLO + Depth + Gaze in parallel CUDA streams |
| **SimpleTracker** | IoU matching, depth history, approach speed |
| **collision_risk()** | 0.0–1.0 score: TTC (50%) + CBDR (25%) + zone (15%) + class (10%) |
| **AlertArbiter** | 2 channels: threat (top-1 risk) + context (traffic lights, signs) |
| **AudioFeedback** | BRR bursts (3/2/1 beeps) + pitch 400–1100Hz + spatial pan L/R |

## Quick Start

### UV (recommended)

```bash
uv sync                              # Base dependencies
uv sync --extra dev                   # + pytest
uv sync --extra cuda                  # + NeMo TTS
uv sync --extra aria                  # + Aria SDK

./run.sh webcam outdoor                      # Webcam
./run.sh realsense indoor                    # RealSense D435
./run.sh data/video.mp4 all                  # Video file
./run.sh aria:usb                            # Aria glasses USB
./run.sh aria:wifi:<ARIA_IP>             # Aria WiFi
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

See [docs/deploy/DOCKER.md](docs/deploy/DOCKER.md) for full Docker documentation.

### Options

```bash
./run.sh <source> <mode> [--no-tts]
```

- **Sources:** `webcam`, `realsense`, video file path, `aria:usb`, `aria:wifi:IP`, `aria:bridge`
- **Modes:** `indoor`, `outdoor`, `all` (24 nav classes)
- **Flags:** `--no-tts` for development without TTS

Open http://localhost:5000 for the MJPEG dashboard.

## Tests

```bash
uv run pytest                    # All tests (27)
uv run python scripts/benchmark_offline.py --from-json data/benchmark_detections.json
```

## Performance

| Component | Backend | FPS | Latency |
|---|---|---:|---:|
| YOLO26s | TensorRT FP16 | 188 | 5.3ms |
| Depth Anything V2-S | TensorRT FP16 | 127 | 7.9ms |
| Meta Eye Gaze | PyTorch CUDA | 261 | 3.8ms |
| **Full pipeline** | **All TensorRT** | **67** | **15.0ms** |

VRAM: ~2.5 GB total. Tested on RTX 2060 (6 GB) and RTX 5060 Ti.

## Project Structure

```
aria-guard/
├── run.sh                       # Launch wrapper (jemalloc + cleanup)
├── pyproject.toml               # UV dependencies and config
├── src/
│   ├── main.py                  # Entry point (spawn, CUDA hide, CLI)
│   ├── domain/                  # Pure business logic (no I/O, no CUDA)
│   │   ├── types.py             # Detection dataclass, CLASS_FILTERS
│   │   ├── tracker.py           # SimpleTracker + collision_risk
│   │   └── alert_engine.py      # AlertArbiter (2 channels)
│   ├── input/                   # Frame sources (sensors)
│   │   ├── observer.py          # BaseObserver + MockObserver
│   │   ├── aria.py              # AriaDemoObserver + AriaDatasetObserver
│   │   └── realsense.py         # RealSenseObserver
│   ├── detection/               # ML inference (CUDA workers)
│   │   ├── detector.py          # ParallelDetector (YOLO+Depth+Gaze)
│   │   └── process.py           # DetectorProcess (spawn wrapper)
│   ├── output/                  # Audio + visual
│   │   ├── audio.py             # BRR + spatial beeps
│   │   ├── tts.py               # TTSProcess (NeMo spawn)
│   │   └── dashboard.py         # Rendering overlays
│   └── web/                     # HTTP server
│       ├── server.py            # Flask app + routes
│       └── pipeline.py          # process_loop (orchestration)
├── scripts/                     # Export, benchmark, utilities
├── models/                      # TensorRT engines, weights
├── data/                        # Videos, datasets
├── tests/                       # Tests
├── docker/                      # Dockerfiles
└── docs/                        # Documentation
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [CLAUDE.md](CLAUDE.md) | LLM/AI rules, coding conventions, how to run |
| [IMPLEMENTATION_PLAN.md](docs/project/IMPLEMENTATION_PLAN.md) | Roadmap by phases (H1–H26) |
| [RESEARCH.md](docs/project/RESEARCH.md) | Academic papers, threat model design, audio research |
| [DOCKER.md](docs/deploy/DOCKER.md) | Docker images, build workflow, troubleshooting |
| [JETSON.md](docs/deploy/JETSON.md) | Jetson Orin Nano deployment |
| [REALSENSE.md](docs/deploy/REALSENSE.md) | Intel RealSense D435 integration |
| [DEVELOPER_DIARY.md](docs/DEVELOPER_DIARY.md) | Feature design diary (5-question framework) |

## Related Projects

- [aria-arm64-bridge](https://github.com/robertteleng/aria-arm64-bridge) — FEX-Emu bridge for Aria SDK on ARM64
- [aria-core](https://github.com/aria-core) — C++ production port (future)

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Meta Project Aria](https://www.projectaria.com/)
- [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking)
