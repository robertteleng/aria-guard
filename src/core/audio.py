"""
Audio feedback system for ARIA demo.

Spatial beeps + TTS based on object position, distance, and gaze.
Supports NVIDIA NeMo TTS (FastPitch + HiFiGAN) for high-quality voice.
"""

import threading
import time
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[AUDIO WARN] sounddevice not installed. Beeps disabled.")

# Try NeMo TTS first (better quality), fallback to pyttsx3
_nemo_available = False
_pyttsx3_available = False

try:
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    import torch
    _nemo_available = True
except ImportError:
    pass

if not _nemo_available:
    try:
        import pyttsx3
        _pyttsx3_available = True
    except ImportError:
        pass

if not _nemo_available and not _pyttsx3_available:
    print("[AUDIO WARN] No TTS engine available. Install nemo_toolkit[tts] or pyttsx3.")


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
        self._nemo_spec_gen = None
        self._nemo_vocoder = None
        self._tts_cache: Dict[str, np.ndarray] = {}  # Cache for common phrases
        self._tts_sample_rate = 22050  # NeMo default

        # Try NeMo first (better quality)
        if use_nemo and _nemo_available:
            try:
                print("[AUDIO] Loading NeMo TTS models...")
                self._nemo_spec_gen = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
                self._nemo_vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan")
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._nemo_spec_gen = self._nemo_spec_gen.cuda()
                    self._nemo_vocoder = self._nemo_vocoder.cuda()
                self._nemo_spec_gen.eval()
                self._nemo_vocoder.eval()
                self.tts_type = "nemo"
                print("[AUDIO] NeMo TTS initialized (FastPitch + HiFiGAN)")
            except Exception as e:
                print(f"[AUDIO WARN] NeMo TTS failed: {e}")

        # Fallback to pyttsx3
        if self.tts_type is None and _pyttsx3_available:
            try:
                import pyttsx3
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
                    sd.play(audio_data, samplerate=self.sample_rate, blocking=False)

            except Exception as e:
                print(f"[AUDIO ERROR] Beep: {e}")

        threading.Thread(target=_play, daemon=True).start()

    def _generate_nemo_audio(self, text: str) -> np.ndarray:
        """Generate audio using NeMo FastPitch + HiFiGAN."""
        import torch
        with torch.no_grad():
            parsed = self._nemo_spec_gen.parse(text)
            spectrogram = self._nemo_spec_gen.generate_spectrogram(tokens=parsed)
            audio = self._nemo_vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        return audio.squeeze().cpu().numpy()

    def speak(self, message: str, force: bool = False) -> bool:
        """Speak a message using TTS (NeMo or pyttsx3)."""
        if self.tts_type is None:
            return False

        now = time.time()
        if not force and (now - self.last_tts_time) < self.tts_cooldown:
            return False
        if self.tts_speaking:
            return False

        self.last_tts_time = now

        def _speak_nemo():
            try:
                self.tts_speaking = True
                print(f"[AUDIO TTS] {message}")

                # Check cache first
                if message in self._tts_cache:
                    audio = self._tts_cache[message]
                else:
                    audio = self._generate_nemo_audio(message)
                    # Cache short common phrases
                    if len(message) < 50:
                        self._tts_cache[message] = audio

                # Play audio
                sd.play(audio, samplerate=self._tts_sample_rate, blocking=True)
            except Exception as e:
                print(f"[AUDIO ERROR] NeMo TTS: {e}")
            finally:
                self.tts_speaking = False

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

        if self.tts_type == "nemo":
            threading.Thread(target=_speak_nemo, daemon=True).start()
        else:
            threading.Thread(target=_speak_pyttsx3, daemon=True).start()
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
