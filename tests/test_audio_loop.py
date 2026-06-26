"""Tests for the real-time audio test loop (dashboard <-> sound correspondence).

Validates the event lifecycle recorded by AudioFeedback: every beep/utterance
is logged as played / dropped / failed, with detection->sound latency, plus the
device-health snapshot and the TTS worker return-channel drain. These assertions
fail if the instrumentation is removed.
"""
import queue
import threading
import time

import pytest
from unittest.mock import MagicMock

import src.output.audio as audio_mod
from src.output.audio import AudioFeedback
from src.output.tts import TTSProcess


def _wait_for(predicate, timeout=1.0):
    """Poll until predicate() is true (threaded audio paths) or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def silent_audio(monkeypatch):
    """AudioFeedback with NO device and NO TTS engine (disabled output)."""
    monkeypatch.setattr(audio_mod, "_pyttsx3_available", False)
    monkeypatch.setattr(audio_mod, "_audio_available", False)
    monkeypatch.setattr(audio_mod, "sd", None)
    return AudioFeedback(enabled=True, use_nemo=False)


@pytest.fixture
def playing_audio(monkeypatch):
    """AudioFeedback with a MOCKED device — beeps reach the 'played' path."""
    fake_sd = MagicMock()
    monkeypatch.setattr(audio_mod, "sd", fake_sd)
    monkeypatch.setattr(audio_mod, "_audio_available", True)
    monkeypatch.setattr(audio_mod, "_pyttsx3_available", False)
    return AudioFeedback(enabled=True, use_nemo=False), fake_sd


# --- device health snapshot ---------------------------------------------------

def test_get_stats_shape_and_health(silent_audio):
    s = silent_audio.get_stats()
    assert set(s) >= {"device_ok", "enabled", "engine", "counters", "events"}
    assert s["device_ok"] is False        # no device patched in
    assert s["enabled"] is False          # => beeps disabled
    assert s["engine"] == "none"          # no TTS engine
    assert s["counters"] == {"played": 0, "dropped": 0, "failed": 0}
    assert s["events"] == []


def test_device_ok_true_when_available(playing_audio):
    audio, _ = playing_audio
    s = audio.get_stats()
    assert s["device_ok"] is True
    assert s["enabled"] is True


# --- beep lifecycle -----------------------------------------------------------

def test_beep_dropped_when_no_device(silent_audio):
    silent_audio.play_spatial_beep("left", "very_close", "DANGER", detected_ts=time.time())
    events = silent_audio.get_stats()["events"]
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "beep"
    assert ev["status"] == "dropped"
    assert ev["reason"] == "no_device"
    assert silent_audio.get_stats()["counters"]["dropped"] == 1


def test_beep_played_records_latency(playing_audio):
    audio, fake_sd = playing_audio
    t0 = time.time() - 0.10           # pretend detection was 100ms ago
    audio.play_spatial_beep("left", "very_close", "DANGER", detected_ts=t0)

    assert _wait_for(lambda: any(e["status"] == "played"
                                 for e in audio.get_stats()["events"]))
    ev = [e for e in audio.get_stats()["events"] if e["status"] == "played"][-1]
    assert ev["kind"] == "beep"
    assert "1100Hz" in ev["detail"]    # very_close -> 1100 Hz, encoded in detail
    assert ev["latency_ms"] is not None
    assert ev["latency_ms"] >= 100.0   # at least the 100ms we backdated
    assert fake_sd.play.called


def test_beep_cooldown_drops_second(playing_audio):
    audio, _ = playing_audio
    audio.play_spatial_beep("center", "medium", "WARNING", detected_ts=time.time())
    # immediate second call is inside beep_cooldown (0.3s) -> dropped synchronously
    audio.play_spatial_beep("center", "medium", "WARNING", detected_ts=time.time())
    dropped = [e for e in audio.get_stats()["events"]
               if e["status"] == "dropped" and e["reason"] == "cooldown"]
    assert len(dropped) == 1


def test_beep_played_without_timestamp_has_no_latency(playing_audio):
    audio, _ = playing_audio
    audio.play_spatial_beep("right", "far", "ATTENTION")  # no detected_ts
    assert _wait_for(lambda: any(e["status"] == "played"
                                 for e in audio.get_stats()["events"]))
    ev = [e for e in audio.get_stats()["events"] if e["status"] == "played"][-1]
    assert ev["latency_ms"] is None


# --- speech lifecycle ---------------------------------------------------------

def test_speak_dropped_when_no_engine(silent_audio):
    ok = silent_audio.speak("peligro izquierda", detected_ts=time.time())
    assert ok is False
    ev = silent_audio.get_stats()["events"][-1]
    assert ev["kind"] == "speech"
    assert ev["status"] == "dropped"
    assert ev["reason"] == "no_engine"


# --- TTS worker return channel (drain) ---------------------------------------

def test_ttsprocess_poll_results_drains_queue():
    p = TTSProcess()
    p.result_queue = queue.Queue()     # stand in for the mp queue (same API)
    p.result_queue.put({"text": "red light", "requested_ts": 1.0,
                        "played_ts": 1.05, "status": "played", "reason": ""})
    out = p.poll_results()
    assert len(out) == 1
    assert out[0]["text"] == "red light"
    assert p.poll_results() == []      # drained, nothing left


def test_drain_thread_records_speech_with_latency(silent_audio):
    audio = silent_audio
    fake_proc = MagicMock()
    t0 = time.time()
    calls = {"n": 0}

    def fake_poll():
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"text": "green light", "requested_ts": t0,
                     "played_ts": t0 + 0.08, "status": "played", "reason": ""}]
        return []

    fake_proc.poll_results.side_effect = fake_poll
    audio._tts_process = fake_proc

    t = threading.Thread(target=audio._drain_tts_results, daemon=True)
    t.start()
    try:
        assert _wait_for(lambda: any(e["kind"] == "speech"
                                     for e in audio.get_stats()["events"]))
    finally:
        audio._tts_drain_stop.set()
        t.join(timeout=1)

    ev = [e for e in audio.get_stats()["events"] if e["kind"] == "speech"][-1]
    assert ev["status"] == "played"
    assert ev["detail"] == "green light"
    assert ev["latency_ms"] == pytest.approx(80.0, abs=1.0)  # 0.08s -> 80ms


# --- counters + ring buffer ---------------------------------------------------

def test_engine_selection_piper_wires_through(monkeypatch):
    """tts_engine='piper' selects PiperProcess and routes speak() to it."""
    class FakePiper:
        ready = True
        def __init__(self): self.spoken = []
        def start(self): pass
        def speak(self, text, requested_ts=None): self.spoken.append((text, requested_ts))
        def poll_results(self): return []
        def stop(self): pass

    import src.output.tts as tts_mod
    monkeypatch.setattr(tts_mod, "PiperProcess", FakePiper, raising=False)
    monkeypatch.setattr(audio_mod, "_audio_available", False)
    monkeypatch.setattr(audio_mod, "sd", None)
    monkeypatch.setattr(audio_mod, "_pyttsx3_available", False)

    a = AudioFeedback(enabled=True, tts_engine="piper")
    assert a.tts_type == "piper"
    assert a.speak("peligro izquierda", detected_ts=1.0) is True
    assert a._tts_process.spoken == [("peligro izquierda", 1.0)]


def test_engine_piper_load_failure_is_graceful(monkeypatch):
    """A Piper that fails to load leaves tts_type=None and speak() drops cleanly."""
    class FakePiperFail:
        ready = False
        def start(self): pass

    import src.output.tts as tts_mod
    monkeypatch.setattr(tts_mod, "PiperProcess", FakePiperFail, raising=False)
    monkeypatch.setattr(audio_mod, "_audio_available", False)
    monkeypatch.setattr(audio_mod, "sd", None)
    monkeypatch.setattr(audio_mod, "_pyttsx3_available", False)

    a = AudioFeedback(enabled=True, tts_engine="piper")
    assert a.tts_type is None
    assert a.speak("x") is False
    assert a.get_stats()["events"][-1]["reason"] == "no_engine"


def test_counters_accumulate_and_events_bounded(silent_audio):
    for _ in range(40):                 # all dropped (no device)
        silent_audio.play_spatial_beep("left", "close", "DANGER")
    s = silent_audio.get_stats()
    assert s["counters"]["dropped"] == 40      # counter is unbounded
    assert len(s["events"]) <= 25              # ring buffer caps the log
