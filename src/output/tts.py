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
    # H20: threat-level TTS (replaces object-name TTS)
    "danger left", "danger right", "danger straight",
    "warning left", "warning right", "warning straight",
    # Channel B context
    "red light", "green light", "yellow light",
    "stop sign ahead",
]


def _tts_worker(queue, sample_rate_out, result_queue):
    """Worker process for NeMo TTS. Has its own CUDA context.

    result_queue: return channel — after each utterance actually plays (or
    fails) the worker reports {text, requested_ts, played_ts, status, reason}
    so the main process can confirm audio reached the user and measure latency.
    """
    import os
    import time
    # Restore CUDA visibility BEFORE importing torch (main process hides it for FastDDS)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ.pop("NUMBA_DISABLE_CUDA", None)
    # Remove jemalloc LD_PRELOAD - only needed for main process (Aria SDK/FastDDS)
    os.environ.pop("LD_PRELOAD", None)

    import numpy as np
    import sounddevice as sd
    import torch

    print("[TTS PROCESS] Starting...")

    # Load NeMo models in this process
    try:
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel

        spec_gen = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
        vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan")

        use_amp = False
        if torch.cuda.is_available():
            # Use AMP FP16 only on Ampere+ (compute 8.0+)
            # RTX 2060 (Turing, 7.5) has cuBLAS FP16 issues with FastPitch
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                use_amp = True
                spec_gen = spec_gen.cuda().eval()
                vocoder = vocoder.cuda().eval()
                print("[TTS PROCESS] NeMo loaded on CUDA (AMP FP16)")
            else:
                # Force FP32 — NeMo loads weights in FP16 by default, convert explicitly
                spec_gen = spec_gen.float().cuda().eval()
                vocoder = vocoder.float().cuda().eval()
                print("[TTS PROCESS] NeMo loaded on CUDA (FP32)")

            # Warm up CUDA
            torch.cuda.synchronize()
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

        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    parsed = spec_gen.parse(text)
                    spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
                    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            else:
                parsed = spec_gen.parse(text)
                spectrogram = spec_gen.generate_spectrogram(tokens=parsed)
                audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        audio = audio.squeeze().cpu().numpy()

        # Ensure float32 for sounddevice
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

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

    # Process messages. Each message is (text, requested_ts); None = shutdown.
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

            text, requested_ts = msg
            try:
                audio = generate_audio(text)
                print(f"[TTS] {text}")
                # played_ts = instant playback STARTS (so latency = detection ->
                # sound-start incl. synthesis, not incl. the utterance duration)
                played_ts = time.time()
                sd.play(audio, samplerate=sample_rate, blocking=True)
                result_queue.put({
                    "text": text, "requested_ts": requested_ts,
                    "played_ts": played_ts, "status": "played", "reason": "",
                })
            except Exception as e:
                print(f"[TTS PROCESS ERROR] play: {e}")
                result_queue.put({
                    "text": text, "requested_ts": requested_ts,
                    "played_ts": time.time(), "status": "failed", "reason": str(e),
                })

        except Exception as e:
            print(f"[TTS PROCESS ERROR] {e}")


class TTSProcess:
    """Manages a separate process for NeMo TTS."""

    def __init__(self):
        self.queue = None
        self.result_queue = None
        self.process = None
        self.sample_rate: int = 22050
        self._ready = False

    def start(self):
        """Start the TTS process."""
        self.queue = _ctx.Queue()
        self.result_queue = _ctx.Queue()
        sample_rate_out = _ctx.Queue()

        self.process = _ctx.Process(
            target=_tts_worker,
            args=(self.queue, sample_rate_out, self.result_queue),
            daemon=True
        )
        self.process.start()

        # Wait for initialization (longer timeout for model download + pre-caching)
        result = sample_rate_out.get(timeout=90)
        if result is None:
            print("[TTS] Process failed to initialize")
            self._ready = False
        else:
            self.sample_rate = result
            self._ready = True
            print("[TTS] Process ready")

    def speak(self, text: str, requested_ts: Optional[float] = None):
        """Send text to be spoken (non-blocking).

        requested_ts: detection/request timestamp (time.time()), echoed back
        in the result so the main process can compute detection->sound latency.
        """
        if self._ready and self.queue:
            self.queue.put((text, requested_ts))

    def poll_results(self) -> List[dict]:
        """Drain the worker's return channel (non-blocking).

        Returns a list of {text, requested_ts, played_ts, status, reason} for
        each utterance that played or failed since the last poll.
        """
        results: List[dict] = []
        if not self.result_queue:
            return results
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Exception:
                break
        return results

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


class PiperProcess:
    """Piper TTS engine (CPU, local, es_ES). Same interface as TTSProcess, but
    thread-based (in-process) — Piper is light and needs no CUDA isolation.

    Voice path: ARIA_PIPER_VOICE env, else /app/models/piper/es_ES-davefx-medium.onnx.
    """

    def __init__(self, voice_path: Optional[str] = None):
        self.voice_path = voice_path or os.environ.get(
            "ARIA_PIPER_VOICE",
            "/app/models/piper/es_ES-davefx-medium.onnx",
        )
        self._voice = None
        self.sample_rate: int = 22050
        self._queue = None
        self._results = None
        self._thread = None
        self._stop = None
        self._ready = False

    def start(self):
        """Load the voice and start the synth/play worker thread."""
        import queue as _q
        import threading as _t
        try:
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.voice_path)
            self.sample_rate = self._voice.config.sample_rate
        except Exception as e:
            print(f"[PIPER] load failed ({self.voice_path}): {e}")
            self._ready = False
            return
        self._queue = _q.Queue()
        self._results = _q.Queue()
        self._stop = _t.Event()
        self._thread = _t.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready = True
        print(f"[PIPER] ready (voice={self.voice_path}, {self.sample_rate} Hz)")

    def _worker(self):
        import time as _time
        import queue as _q
        import numpy as np
        import sounddevice as sd
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except _q.Empty:
                continue
            if item is None:  # shutdown
                break
            text, requested_ts = item
            try:
                chunks = [c.audio_float_array for c in self._voice.synthesize(text)]
                audio = (np.concatenate(chunks).astype(np.float32)
                         if chunks else np.zeros(1, dtype=np.float32))
                # played_ts = playback start (latency = detection -> sound-start)
                played_ts = _time.time()
                sd.play(audio, samplerate=self.sample_rate, blocking=True)
                self._results.put({
                    "text": text, "requested_ts": requested_ts,
                    "played_ts": played_ts, "status": "played", "reason": "",
                })
            except Exception as e:
                print(f"[PIPER] synth/play error: {e}")
                self._results.put({
                    "text": text, "requested_ts": requested_ts,
                    "played_ts": _time.time(), "status": "failed", "reason": str(e),
                })

    def speak(self, text: str, requested_ts: Optional[float] = None):
        """Queue text for synthesis + playback (non-blocking)."""
        if self._ready and self._queue is not None:
            self._queue.put((text, requested_ts))

    def poll_results(self) -> List[dict]:
        """Drain played/failed results (non-blocking) — same shape as TTSProcess."""
        import queue as _q
        results: List[dict] = []
        if self._results is None:
            return results
        while True:
            try:
                results.append(self._results.get_nowait())
            except _q.Empty:
                break
        return results

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def ready(self) -> bool:
        return self._ready
