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

### Changed
- **Beeps decoupled from NeMo TTS**: spatial beeps (the dominant guidance
  channel) now run without loading NeMo on the GPU. NeMo is opt-in via
  `ARIA_TTS_ENGINE=nemo`; the voice engine choice is deferred to a by-ear A/B
  (Piper rejected as robotic, cloud rejected as non-local).

### Known limitations
- BT output is **HFP mono** — PulseAudio 15 on the Jetson does not expose the
  A2DP profile for the Shokz, so stereo left/right panning is lost. Distance
  (pitch) and threat level (beep count) are conveyed; the per-side cue is a
  pending audio-design task (a non-spatial pattern is the right fix for
  bone-conduction headphones, not A2DP).

### Tests
- `tests/test_audio_loop.py` (10) — event lifecycle, latency, no-device /
  no-engine / cooldown drops, TTS return-channel drain, ring-buffer bound.
- `tests/test_audio_endpoints.py` (9) — endpoint validation (400) + rate-limit
  (429) + audio-not-ready (503).
