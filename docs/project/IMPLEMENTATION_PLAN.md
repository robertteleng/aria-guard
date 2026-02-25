# IMPLEMENTATION_PLAN.md

Roadmap del proyecto organizado por fases. Cada fase se valida con uso real antes de avanzar.

> **Regla:** No empezar Fase N+1 hasta que Fase N esté validada.

---

## Phase 1 — MVP ✅

Pipeline completo de detección en tiempo real con alertas de audio.

| # | Milestone | Descripción | Estado |
|---|-----------|-------------|--------|
| H1 | Project Setup + Docker | Dockerfile base CUDA, GPU auto-detect, multi-GPU | ✅ |
| H2 | YOLO TensorRT | YOLO26s FP16, 188 FPS, 5.3ms | ✅ |
| H3 | Depth Estimation TRT | Depth Anything V2-S FP16, 127 FPS, 7.9ms | ✅ |
| H4 | Object Tracking | SimpleTracker IoU matching entre frames | ✅ |
| H5 | Alert System | AlertDecisionEngine, priorización, cooldowns | ✅ |
| H6 | Spatial Audio | Beeps 3D estéreo, 4 zonas distancia + panning L/R | ✅ |
| H7 | TTS (NeMo) | Proceso separado, aislamiento CUDA, pre-cache | ✅ |
| H8 | Meta Aria Glasses | RGB + gaze-aware filtering | ✅ |
| H9 | Intel RealSense D435 | Hardware depth mm, align depth-to-color | ✅ |
| H10 | Shared Memory IPC | Zero-copy RGB + depth entre procesos | ✅ |
| H11 | NVDEC Video Decode | Hardware video decoding | ✅ |
| H12 | Benchmarks Paper | benchmark_paper.py reproducible + results JSON | ✅ |

---

## Phase 2 — Advanced Detection ✅

| # | Milestone | Descripción | Estado |
|---|-----------|-------------|--------|
| H13 | YOLO Fine-tune Navigation | Clases custom: doors, stairs, curbs, traffic_light, signs | ✅ |
| H14 | Traffic Light Classification | HSV sobre crop YOLO: red/yellow/green | ✅ |
| H15 | Key Sign Detection | Stop sign: alerta independiente con TTS | ✅ |
| H16 | Risk Prioritization v2 | Approach continuo, zone factor, fast vehicle alert | ✅ |
| H17 | ~~Haptic Feedback Prototype~~ | Nice to have — requiere hardware custom, bajo ROI | ⚪ |

---

## Phase 3 — Threat Model & Audio ✅

Rediseño del sistema de alertas basado en evidencia académica (ver [RESEARCH.md](RESEARCH.md)).

| # | Milestone | Descripción | Estado |
|---|-----------|-------------|--------|
| H18 | Collision Risk Score | `collision_risk()` 0.0–1.0: TTC + CBDR + zone + class | ✅ |
| H19 | Alert Arbiter (2 canales) | Canal A: top-1 risk. Canal B: contexto. Rate limit 6/30s | ✅ |
| H20 | Audio BRR + Pitch | BRR burst 3/2/1 beeps. Pitch 400–1100Hz. TTS "danger left" | ✅ |
| H21 | Benchmark Offline | benchmark_offline.py: alerts/min, silent ratio, min gap | ✅ |
| H22 | Calibration | Tokyo_POV.mp4 benchmark ALL PASS: 11.8 alerts/min, 80.3% silence | ✅ |

---

## Phase 4 — ARM64 / Jetson ⏳

Despliegue standalone en Jetson Orin Nano con RealSense.

| # | Milestone | Descripción | Estado |
|---|-----------|-------------|--------|
| H23 | UV nativo | Migrar de Docker a UV (pyproject.toml) | ✅ |
| H24 | Aria on ARM64 | FEX-Emu bridge para Aria SDK en Jetson | ⏳ |
| H25 | Single-process mode | DetectorLite sin multiprocessing para Jetson + RealSense | Pending |
| H26 | Ego-motion compensation | Aria IMU para compensar movimiento del usuario | Pending |

> Ver [aria-arm64-bridge](https://github.com/robertteleng/aria-arm64-bridge) para H24.

---

## Future Ideas

- Ego-motion compensation via Aria IMU
- Haptic feedback (H17, nice-to-have)
- SLAM cameras (deferred, only front RGB for now)
- Ground plane anomaly detection (holes, curbs via depth map)
- DeepStream pipeline for Jetson (if Python is too slow)

---

## Notas

- Benchmark: 27 FPS pipeline completo en RTX 5060 Ti
- 27 tests pass (9 collision risk + 11 arbiter + 7 benchmark)
- Legacy priority system removed (tracker uses collision_risk only)
- DANGER threshold 0.7, cooldown 2.0s
