# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); this project
versions loosely.

## [Unreleased]

### Added
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
