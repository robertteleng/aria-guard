"""
Audio feedback system for ARIA demo.

Spatial beeps + TTS based on object position, distance, and gaze.
NeMo TTS runs in a separate process to avoid CUDA conflicts with detector.
"""

import threading
import time
from typing import Optional, Dict

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[AUDIO WARN] sounddevice not installed. Beeps disabled.")

# Check for pyttsx3 fallback
_pyttsx3_available = False
try:
    import pyttsx3
    _pyttsx3_available = True
except ImportError:
    pass


class AudioFeedback:
    """Spatial audio feedback with beeps and TTS."""

    # Frequencies (from aria-nav)
    FREQ_CRITICAL = 1000  # Hz - danger/very close
    FREQ_NORMAL = 500     # Hz - info/medium distance

    # Volume by distance
    VOLUME_MAP = {
        "very_close": 1.0,
        "close": 0.7,
        "medium": 0.45,
        "far": 0.25,
        "unknown": 0.5
    }

    def __init__(self, enabled: bool = True, use_nemo: bool = True):
        self.enabled = enabled and sd is not None
        self.beep_sample_rate = 44100  # Fixed for beeps
        self.base_volume = 0.6

        # TTS engine selection
        self.tts_engine = None
        self.tts_type = None
        self.tts_speaking = False
        self._tts_process = None

        # Try NeMo in separate process (avoids CUDA conflicts)
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

        # Fallback to pyttsx3
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
            print("[AUDIO] Spatial audio feedback initialized")

    def play_spatial_beep(
        self,
        zone: str,
        distance: str = "medium",
        is_critical: bool = False,
        is_unnoticed_danger: bool = False
    ) -> None:
        """Play a spatial beep based on object position and distance."""
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_beep_time < self.beep_cooldown:
            return
        self.last_beep_time = now

        def _play():
            try:
                # Select frequency
                if is_critical or is_unnoticed_danger:
                    freq = self.FREQ_CRITICAL
                    duration = 0.25 if is_unnoticed_danger else 0.15
                else:
                    freq = self.FREQ_NORMAL
                    duration = 0.1

                # Volume based on distance
                volume = self.base_volume * self.VOLUME_MAP.get(distance, 0.5)
                if is_unnoticed_danger:
                    volume = min(1.0, volume * 1.5)

                # Generate tone
                t = np.linspace(0, duration, int(self.beep_sample_rate * duration), False)
                tone = np.sin(2 * np.pi * freq * t)

                # Fade in/out
                fade_samples = int(self.beep_sample_rate * 0.01)
                if len(tone) > fade_samples * 2:
                    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                tone *= volume

                # Spatial panning
                if zone == "left":
                    left, right = tone, tone * 0.2
                elif zone == "right":
                    left, right = tone * 0.2, tone
                else:
                    left, right = tone, tone

                audio_data = np.column_stack((left, right)).astype(np.float32)
                sd.play(audio_data, samplerate=self.beep_sample_rate, blocking=False)

                if is_unnoticed_danger:
                    time.sleep(0.08)
                    sd.play(audio_data, samplerate=self.beep_sample_rate, blocking=False)

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
            # Non-blocking: send to separate process
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
        force_tts: bool = False
    ) -> None:
        """
        Alert user about a dangerous/close object.

        Args:
            force_tts: If True, always speak TTS (for vehicles)
        """
        is_critical = distance in ("very_close", "close")

        self.play_spatial_beep(
            zone=zone,
            distance=distance,
            is_critical=is_critical,
            is_unnoticed_danger=is_critical and not user_looking
        )

        # TTS: speak if forced (vehicles) OR if close and not looking
        if force_tts or (is_critical and not user_looking):
            zone_word = {"left": "left", "right": "right", "center": "straight"}.get(zone, "")
            self.speak(f"{object_name} {zone_word}")

    def shutdown(self):
        """Clean shutdown of TTS process."""
        if self._tts_process:
            self._tts_process.stop()
