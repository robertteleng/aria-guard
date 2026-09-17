# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); this project
versions loosely.

## [Unreleased]

### Added
- **Wearer-body filter** (`is_wearer_body` in `src/domain/ground_projection.py`,
  on in the live metric model): a `person` box that reaches the bottom edge with
  its top at least 15° below horizontal is the wearer's own hand or arm and gets
  no threat. Pre-registered as candidate change 2; precision 20.4 → 24.0 %,
  unjustified alerts 6.89 → 5.67 per minute, recall unchanged.
- **VRS replay benchmark** (`scripts/replay_vrs.py`, `replay_benchmark.sh`,
  `replay_table.py`): real Aria recordings through the full pipeline (detector,
  depth, gaze, IMU motion state, tracker, arbiter). Realtime pacing decodes on
  its own thread at capture time and drops what the pipeline cannot keep up
  with; `all` pacing gives per-stage cost. One JSON record per run with
  environment, commit and model hash. Methodology: `docs/REPLAY_BENCHMARK.md`.
- **`scripts/build_trt_engine.py`**: builds each TensorRT engine on its device
  from ONNX (external weights, static shapes for dynamic inputs).

### Fixed
- **Alert evaluation counted the wearer's hands as obstacles.** The reference now
  uses MPS hand tracking: `person` detections containing a tracked wrist get no
  ground position. The published candidate-1 numbers (37.6 % precision, 33.1 %
  recall) were inflated; corrected they are 20.4 % and 29.1 %.
- **Time-to-collision was inverted**: TTC divided proximity (`depth_value`, 1 =
  close) by the approach speed as if it were distance, so a person almost at the
  camera scored WARNING and the same person far away DANGER. TTC and the looming
  conversion now use the remaining gap `1 - depth_value`. On the six replay
  recordings (RTX, same detections) DANGER alerts go from 9 to 18 and WARNING
  from 56 to 88; all alert criteria still pass.
- **Dataset playback orientation**: `AriaDatasetObserver` rotated RGB the
  opposite way to live streaming and left the eye image unrotated. Frame and IMU
  transforms are now shared (`src/input/aria_frames.py`); VRS timestamps are read
  from the index instead of decoding every image at start-up.
- **Offline benchmark** assumed 1920 px frames and never passed the ego-motion
  state; width, FOV and per-frame motion state are now parameters.
- Gaze ONNX export on torch >= 2.6 (`weights_only` default).

### Removed
- **Docker setup** (`docker/`, `docs/deploy/DOCKER.md`, `scripts/validate-docker.sh`): the images
  needed a proprietary NVIDIA SDK archive or files no longer in the repo, and none was verified.
  `docs/deploy/JETSON.md` now documents the container path actually used for the Jetson benchmark.
- `docs/project/IMPLEMENTATION_PLAN.md` (stale roadmap).

### Changed
- **Metric in-path threat model adopted** (`src/input/metric_inpath.py`): alert precision 14.6 % ->
  37.6 %, episode recall 15.7 % -> 33.1 % against the wearer's real path (docs/ALERT_EVALUATION.md).
- **Jetson performance**: YOLO engine called directly, depth normalized on the GPU and optionally
  asynchronous, OpenCV threads restored after `import ultralytics`.
- **Depth map kept at model resolution (518x518), normalized on the GPU**: it was
  upscaled to the full frame on the CPU only to be sampled inside bboxes.
  Identical detections on 300 frames, depth stage 9.8 -> 3.4 ms on the RTX.
- Glasses IP, headset MAC and bridge path come from the environment (`ARIA_IP`,
  `ARIA_BT_DEV`, `ARIA_BRIDGE_SRC`), no network-specific defaults in code.

- **Ego-motion compensation + bbox-height looming (fewer false DANGER alerts)**:
  while the user walks, every object ahead "approaches" (depth grows) even when
  static — the dominant source of "alertas mediocres". The tracker now subtracts
  a walking bias (`EGO_MOTION_WALKING_BIAS = 0.05`) from the apparent approach
  when `motion_state == "walking"`, so static obstacles don't read as collisions
  while genuinely fast approachers still trigger DANGER. A complementary,
  depth-model-independent **looming** cue from bbox-height growth
  (`(Δheight/height)·(1 - depth_value)` since the TTC fix, fused via `max`) survives the relative-depth
  (`NORM_MINMAX`) noise; it stays in depth-slope units so TTC is unchanged.
  Motion state comes from IMU accel variance (`AriaDemoObserver` and, on Jetson,
  `AriaBridgeObserver.get_motion_state`). Tests: `tests/test_ego_motion.py`
  (+ bridge `tests/test_motion_state.py`). Depth/near-far path audited in
  `docs/research/depth-and-approach-audit.md`. (Offline benchmark regression not
  re-run — its detection JSON isn't in the repo; the change only reduces false
  approaches, the safe direction for the alert-rate constraint.)
- **Real-time audio test loop**: the command dashboard now shows, per audio
  event, whether what the system detects actually reaches the user as sound —
  `played` / `dropped` / `failed`, the reason, and detection→sound latency,
  plus device-health. New "Audio en vivo" panel + `POST /tts/test` (speak
  arbitrary text / by-ear voice A/B) and `POST /audio/sim` (trigger an exact
  beep scenario). Engine-agnostic, local-only.
- **Audio reaches Bluetooth headphones from the full Docker pipeline**: the
  container's beeps route to the host PulseAudio sink via an ALSA→pulse bridge
  (`libasound2-plugins` + `/etc/asound.conf`) and a mounted pulse socket.
  Validated on device: glasses → FEX receiver → Docker detector (9.1 FPS, real
  person/chair detections) → beeps heard on Shokz OpenRun Pro 2.
- **Spoken alerts via Piper TTS (on-device, es_ES)**: `ARIA_TTS_ENGINE=piper`
  loads a light, in-process CPU voice (`models/piper/es_ES-davefx-medium.onnx`)
  — no GPU, no NeMo. Validated on ARM64: synth + playback of a Spanish phrase
  (2.2 s) on the Shokz. The bridge's `launch_pipeline.sh` enables it by default
  when the voice model is present (`VOICE=0` to disable).
- **A2DP stereo restored on Jetson** (was HFP mono): root-caused to NVIDIA's
  `nv-bluetooth-service.conf` starting `bluetoothd --noplugin=audio,a2dp,avrcp`,
  which disables BlueZ's A2DP plugin. A `/etc` systemd override re-enables it;
  the Shokz sink is now `s16le 2ch 44100Hz`. Full write-up in
  `docs/research/bluetooth-a2dp-jetson-shokz.md`.
- **`scripts/bt-audio.sh`**: host BT audio control — `status` / `a2dp` (stereo)
  / `mic [secs]` (A2DP↔HFP switch + record) / `fix` (print the A2DP plugin fix).

### Changed
- **Beeps decoupled from NeMo TTS**: spatial beeps (the dominant guidance
  channel) now run without loading NeMo on the GPU. NeMo stays opt-in via
  `ARIA_TTS_ENGINE=nemo`; **Piper (es_ES) is the selected on-device voice**
  (`ARIA_TTS_ENGINE=piper`) after the by-ear A/B (cloud rejected as non-local).

### Fixed
- **A2DP stereo on the Shokz** — see Added. The earlier "HFP mono only"
  limitation was a disabled BlueZ plugin, not a missing capability.

### Known limitations
- **Mic capture blocked on the Jetson's onboard Realtek BT**: the Shokz HFP
  source streams but delivers silence — measured 1002 bytes of SCO in 8 s while
  speaking (rms=2). The controller advertises CVSD (HV1/2/3) but no eSCO, so
  mSBC can't transport and CVSD SCO doesn't deliver mic packets either —
  a known `rtk_btusb` SCO weakness. The A2DP↔HFP switch (`bt-audio.sh mic`)
  works; the SCO transport does not. **Chosen path: keep the Shokz mic via a
  USB BT dongle (CSR8510)** that handles HFP/SCO on Linux (a USB wired mic is the
  fallback). Aria glasses' mics rejected (crash under FEX + steal RGB DDS
  bandwidth). Setup + evidence in `docs/research/bluetooth-a2dp-jetson-shokz.md`
  and `aria-scene/docs/development/voice-input-and-vlm-feeding.md`.

### Tests
- `tests/test_audio_loop.py` (10) — event lifecycle, latency, no-device /
  no-engine / cooldown drops, TTS return-channel drain, ring-buffer bound.
- `tests/test_audio_endpoints.py` (9) — endpoint validation (400) + rate-limit
  (429) + audio-not-ready (503).
