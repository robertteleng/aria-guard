"""
Audio feedback system for ARIA (H20).

BRR (beep repetition rate) + pitch by distance + spatial pan.
TTS speaks threat level ("danger left") not object type ("car left").
NeMo TTS runs in a separate process to avoid CUDA conflicts with detector.
"""

import threading
import time
from typing import Optional, Dict, Tuple

import numpy as np

try:
    import sounddevice as sd
    try:
        sd.query_devices()
        _audio_available = True
    except Exception:
        _audio_available = False
        print("[AUDIO WARN] No audio devices found. Beeps disabled.")
except ImportError:
    sd = None
    _audio_available = False
    print("[AUDIO WARN] sounddevice not installed. Beeps disabled.")

_pyttsx3_available = False
try:
    import pyttsx3
    _pyttsx3_available = True
except ImportError:
    pass


# --- H20: BRR burst config per threat level ---
# count: beeps in burst, gap_ms: silence between beeps, duration_ms: each beep
BRR_CONFIG = {
    "DANGER":    {"count": 3, "gap_ms": 80,  "duration_ms": 60},
    "WARNING":   {"count": 2, "gap_ms": 150, "duration_ms": 80},
    "ATTENTION": {"count": 1, "gap_ms": 0,   "duration_ms": 100},
}

# --- H20: Pitch by distance (higher = closer) ---
PITCH_MAP = {
    "very_close": 1100,
    "close": 800,
    "medium": 600,
    "far": 400,
    "unknown": 600,
}

# Volume by distance (unchanged from v1)
VOLUME_MAP = {
    "very_close": 1.0,
    "close": 0.7,
    "medium": 0.45,
    "far": 0.25,
    "unknown": 0.5,
}

# Spatial pan (L, R) per zone
PAN_MAP = {
    "left":   (1.0, 0.15),
    "center": (0.7, 0.7),
    "right":  (0.15, 1.0),
}

# Zone to spoken word
ZONE_WORDS = {"left": "left", "right": "right", "center": "straight"}


class AudioFeedback:
    """Spatial audio feedback with BRR bursts, pitch-by-distance, and TTS."""

    def __init__(self, enabled: bool = True, use_nemo: bool = True):
        self.enabled = enabled and sd is not None and _audio_available
        self.beep_sample_rate = 44100
        self.base_volume = 0.6

        # TTS engine selection
        self.tts_engine = None
        self.tts_type = None
        self.tts_speaking = False
        self._tts_process = None

        if use_nemo:
            try:
                from src.core.tts_process import TTSProcess
                self._tts_process = TTSProcess()
                self._tts_process.start()
                if self._tts_process.ready:
                    self.tts_type = "nemo"
                    print("[AUDIO] NeMo TTS ready (separate process)")
                else:
                    self._tts_process = None
            except Exception as e:
                print(f"[AUDIO WARN] NeMo process failed: {e}")
                self._tts_process = None

        if self.tts_type is None and _pyttsx3_available:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_type = "pyttsx3"
                print("[AUDIO] pyttsx3 TTS initialized (fallback)")
            except Exception as e:
                print(f"[AUDIO WARN] pyttsx3 TTS init failed: {e}")

        # Cooldowns
        self.last_beep_time = 0
        self.beep_cooldown = 0.3
        self.last_tts_time = 0
        self.tts_cooldown = 2.0

        if self.enabled:
            print("[AUDIO] BRR + pitch audio feedback initialized (H20)")

    def _generate_beep(self, freq: float, duration_s: float, volume: float,
                       pan: Tuple[float, float]) -> np.ndarray:
        """Generate a single beep tone with fade and spatial pan."""
        n_samples = int(self.beep_sample_rate * duration_s)
        t = np.linspace(0, duration_s, n_samples, False)
        tone = np.sin(2 * np.pi * freq * t)

        # Fade in/out (10ms)
        fade = int(self.beep_sample_rate * 0.01)
        if len(tone) > fade * 2:
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)

        tone *= volume
        left = tone * pan[0]
        right = tone * pan[1]
        return np.column_stack((left, right)).astype(np.float32)

    def _generate_burst(self, threat_level: str, distance: str,
                        zone: str) -> np.ndarray:
        """Generate a BRR burst: multiple beeps with gaps."""
        config = BRR_CONFIG.get(threat_level, BRR_CONFIG["ATTENTION"])
        freq = PITCH_MAP.get(distance, 600)
        volume = self.base_volume * VOLUME_MAP.get(distance, 0.5)
        pan = PAN_MAP.get(zone, (0.7, 0.7))

        beep_dur = config["duration_ms"] / 1000.0
        gap_dur = config["gap_ms"] / 1000.0
        count = config["count"]

        beep = self._generate_beep(freq, beep_dur, volume, pan)
        gap_samples = int(self.beep_sample_rate * gap_dur)
        gap = np.zeros((gap_samples, 2), dtype=np.float32)

        parts = []
        for i in range(count):
            parts.append(beep)
            if i < count - 1 and gap_samples > 0:
                parts.append(gap)

        return np.concatenate(parts)

    def play_spatial_beep(
        self,
        zone: str,
        distance: str = "medium",
        threat_level: str = "ATTENTION",
    ) -> None:
        """Play a BRR burst based on threat level, distance, and zone."""
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_beep_time < self.beep_cooldown:
            return
        self.last_beep_time = now

        def _play():
            try:
                burst = self._generate_burst(threat_level, distance, zone)
                sd.play(burst, samplerate=self.beep_sample_rate, blocking=False)
            except Exception as e:
                print(f"[AUDIO ERROR] Beep: {e}")

        threading.Thread(target=_play, daemon=True).start()

    def speak(self, message: str, force: bool = False) -> bool:
        """Speak a message using TTS (NeMo process or pyttsx3)."""
        if self.tts_type is None:
            return False

        now = time.time()
        if not force and (now - self.last_tts_time) < self.tts_cooldown:
            return False

        self.last_tts_time = now

        if self.tts_type == "nemo" and self._tts_process:
            self._tts_process.speak(message)
            return True

        elif self.tts_type == "pyttsx3":
            def _speak_pyttsx3():
                try:
                    self.tts_speaking = True
                    print(f"[AUDIO TTS] {message}")
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"[AUDIO ERROR] pyttsx3 TTS: {e}")
                finally:
                    self.tts_speaking = False

            threading.Thread(target=_speak_pyttsx3, daemon=True).start()
            return True

        return False

    def alert_danger(
        self,
        object_name: str,
        zone: str,
        distance: str,
        user_looking: bool,
        force_tts: bool = False,
        threat_level: str = "WARNING",
    ) -> None:
        """Alert user about a dangerous object.

        H20: Uses BRR burst by threat_level, pitch by distance.
        TTS says "danger left" instead of "car left".
        """
        self.play_spatial_beep(
            zone=zone,
            distance=distance,
            threat_level=threat_level,
        )

        # TTS: speak threat+direction if forced or if DANGER
        if force_tts:
            zone_word = ZONE_WORDS.get(zone, "")
            tts_level = threat_level.lower() if threat_level != "ATTENTION" else ""
            if tts_level:
                self.speak(f"{tts_level} {zone_word}")

    def alert_traffic_light(self, state: str, zone: str) -> None:
        """Alert user about traffic light state (Channel B context)."""
        self.play_spatial_beep(
            zone=zone,
            distance="medium",
            threat_level="ATTENTION",
        )
        self.speak(f"{state} light", force=True)

    def alert_sign(self, sign_name: str, zone: str, distance: str) -> None:
        """Alert user about a road sign (Channel B context)."""
        self.play_spatial_beep(
            zone=zone,
            distance=distance,
            threat_level="ATTENTION",
        )
        self.speak(f"{sign_name} ahead", force=True)

    def shutdown(self):
        """Clean shutdown of TTS process."""
        if self._tts_process:
            self._tts_process.stop()
