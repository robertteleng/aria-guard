"""
Audio feedback system for ARIA demo.

Spatial beeps + TTS based on object position, distance, and gaze.
"""

import threading
import time
from typing import Optional, List

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[AUDIO WARN] sounddevice not installed. Beeps disabled.")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("[AUDIO WARN] pyttsx3 not installed. TTS disabled.")


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

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sd is not None
        self.sample_rate = 44100
        self.base_volume = 0.6

        # TTS
        self.tts_engine = None
        self.tts_speaking = False
        if pyttsx3:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                print("[AUDIO] TTS initialized")
            except Exception as e:
                print(f"[AUDIO WARN] TTS init failed: {e}")

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
                t = np.linspace(0, duration, int(self.sample_rate * duration), False)
                tone = np.sin(2 * np.pi * freq * t)

                # Fade in/out
                fade_samples = int(self.sample_rate * 0.01)
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
                sd.play(audio_data, samplerate=self.sample_rate, blocking=False)

                if is_unnoticed_danger:
                    time.sleep(0.08)
                    sd.play(audio_data, samplerate=self.sample_rate, blocking=False)

            except Exception as e:
                print(f"[AUDIO ERROR] Beep: {e}")

        threading.Thread(target=_play, daemon=True).start()

    def speak(self, message: str, force: bool = False) -> bool:
        """Speak a message using TTS."""
        if not self.tts_engine:
            return False

        now = time.time()
        if not force and (now - self.last_tts_time) < self.tts_cooldown:
            return False
        if self.tts_speaking:
            return False

        self.last_tts_time = now

        def _speak():
            try:
                self.tts_speaking = True
                print(f"[AUDIO TTS] {message}")
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"[AUDIO ERROR] TTS: {e}")
            finally:
                self.tts_speaking = False

        threading.Thread(target=_speak, daemon=True).start()
        return True

    def alert_danger(
        self,
        object_name: str,
        zone: str,
        distance: str,
        user_looking: bool
    ) -> None:
        """Alert user about a dangerous/close object."""
        is_critical = distance in ("very_close", "close")

        self.play_spatial_beep(
            zone=zone,
            distance=distance,
            is_critical=is_critical,
            is_unnoticed_danger=is_critical and not user_looking
        )

        if is_critical and not user_looking:
            zone_word = {"left": "left", "right": "right", "center": "ahead"}.get(zone, "")
            self.speak(f"Warning, {object_name} {zone_word}", force=True)

    def announce_scene(self, detections: List) -> None:
        """Announce summary of detected objects."""
        if not detections:
            self.speak("No objects detected", force=True)
            return

        zones = {"center": [], "left": [], "right": []}
        for det in detections[:5]:
            zones[det.zone].append(det.name)

        parts = ["Scanning."]
        zone_names = {"center": "Ahead", "left": "Left", "right": "Right"}
        for zone in ["center", "left", "right"]:
            if zones[zone]:
                objects = ", ".join(zones[zone][:3])
                parts.append(f"{zone_names[zone]}: {objects}.")

        self.speak(" ".join(parts), force=True)

    def close(self) -> None:
        """Cleanup."""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        print("[AUDIO] Closed")
