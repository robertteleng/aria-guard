"""
NeMo TTS running in a separate process with isolated CUDA context.
This allows the detector to use CUDA streams without interference.
"""
import os
import tempfile
from pathlib import Path
import multiprocessing as mp
from typing import Optional, List

# Use spawn context for clean CUDA initialization in child process
_ctx = mp.get_context('spawn')

# Fix /tmp issue before importing NeMo
_tmp_dir = Path.home() / "tmp"
_tmp_dir.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_tmp_dir)
os.environ["TEMP"] = str(_tmp_dir)
os.environ["TMP"] = str(_tmp_dir)
tempfile.tempdir = str(_tmp_dir)

# Common phrases to pre-cache at startup (short: object + direction)
PRECACHE_PHRASES = [
    "person left", "person right", "person straight",
    "car left", "car right", "car straight",
    "bicycle left", "bicycle right", "bicycle straight",
    "motorcycle left", "motorcycle right", "motorcycle straight",
    "bus left", "bus right", "bus straight",
    "truck left", "truck right", "truck straight",
    "chair left", "chair right", "chair straight",
    "dog left", "dog right", "dog straight",
    "backpack left", "backpack right", "backpack straight",
    "handbag left", "handbag right", "handbag straight",
]


def _tts_worker(queue, sample_rate_out):
    """Worker process for NeMo TTS. Has its own CUDA context."""
    import os
    # Restore CUDA visibility BEFORE importing torch (main process hides it for FastDDS)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("NUMBA_DISABLE_CUDA", None)

    import numpy as np
    import sounddevice as sd
    import torch

    print("[TTS PROCESS] Starting...")

    # Load NeMo models in this process
    try:
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel

        spec_gen = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
        vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan")

        if torch.cuda.is_available():
            spec_gen = spec_gen.cuda().eval()
            vocoder = vocoder.cuda().eval()
            print("[TTS PROCESS] NeMo loaded on CUDA")
        else:
            spec_gen = spec_gen.eval()
            vocoder = vocoder.eval()
            print("[TTS PROCESS] NeMo loaded on CPU")

        sample_rate = 22050

    except Exception as e:
        print(f"[TTS PROCESS] Failed to load NeMo: {e}")
        sample_rate_out.put(None)
        return

    # Audio cache
    cache = {}

    def generate_audio(text):
        """Generate audio for text, using cache if available."""
        if text in cache:
            return cache[text].copy()

        with torch.no_grad():
            parsed = spec_gen.parse(text)
            spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        audio = audio.squeeze().cpu().numpy()

        if len(text) < 50:
            cache[text] = audio.copy()

        return audio

    # Pre-cache common phrases for instant playback
    print("[TTS PROCESS] Pre-caching common phrases...")
    for phrase in PRECACHE_PHRASES:
        generate_audio(phrase)
    print(f"[TTS PROCESS] Cached {len(PRECACHE_PHRASES)} phrases")

    # Signal ready
    sample_rate_out.put(sample_rate)

    # Process messages
    while True:
        try:
            msg = queue.get()

            if msg is None:  # Shutdown signal
                print("[TTS PROCESS] Shutting down")
                break

            # Skip old messages - only process the latest
            while not queue.empty():
                try:
                    newer_msg = queue.get_nowait()
                    if newer_msg is None:
                        print("[TTS PROCESS] Shutting down")
                        return
                    msg = newer_msg  # Use the newer message
                except:
                    break

            audio = generate_audio(msg)
            print(f"[TTS] {msg}")
            sd.play(audio, samplerate=sample_rate, blocking=True)

        except Exception as e:
            print(f"[TTS PROCESS ERROR] {e}")


class TTSProcess:
    """Manages a separate process for NeMo TTS."""

    def __init__(self):
        self.queue = None
        self.process = None
        self.sample_rate: int = 22050
        self._ready = False

    def start(self):
        """Start the TTS process."""
        self.queue = _ctx.Queue()
        sample_rate_out = _ctx.Queue()

        self.process = _ctx.Process(
            target=_tts_worker,
            args=(self.queue, sample_rate_out),
            daemon=True
        )
        self.process.start()

        # Wait for initialization (longer timeout for pre-caching)
        result = sample_rate_out.get(timeout=120)
        if result is None:
            print("[TTS] Process failed to initialize")
            self._ready = False
        else:
            self.sample_rate = result
            self._ready = True
            print("[TTS] Process ready")

    def speak(self, text: str):
        """Send text to be spoken (non-blocking)."""
        if self._ready and self.queue:
            self.queue.put(text)

    def stop(self):
        """Stop the TTS process."""
        if self.queue:
            self.queue.put(None)
        if self.process:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()

    @property
    def ready(self) -> bool:
        return self._ready
