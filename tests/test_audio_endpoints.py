"""Tests for the audio dashboard control endpoints (/tts/test, /audio/sim).

Covers the input validation + anti-spam guard added after code review:
invalid scenario params -> 400, oversize/empty text -> 400, repeated speak
within the cooldown -> 429, and audio-not-ready -> 503.
"""
import sys
from unittest.mock import MagicMock

import pytest

# server.py imports cv2 at module top; stub it (not needed for these routes).
sys.modules.setdefault("cv2", MagicMock())

import src.web.server as server


@pytest.fixture
def client():
    server.app.config["TESTING"] = True
    server._last_tts_test["t"] = 0.0          # reset anti-spam between tests
    yield server.app.test_client()
    server.state.pop("audio_ref", None)


@pytest.fixture
def audio_mock():
    return MagicMock()


# --- /audio/sim ---------------------------------------------------------------

def test_sim_valid_params_triggers_beep(client, audio_mock):
    server.state["audio_ref"] = audio_mock
    r = client.post("/audio/sim", json={"zone": "left", "distance": "very_close",
                                        "threat_level": "DANGER"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    audio_mock.play_spatial_beep.assert_called_once()
    kw = audio_mock.play_spatial_beep.call_args.kwargs
    assert kw["zone"] == "left" and kw["distance"] == "very_close"
    assert kw["threat_level"] == "DANGER"


@pytest.mark.parametrize("bad", [
    {"zone": "up", "distance": "medium", "threat_level": "DANGER"},
    {"zone": "left", "distance": "teleport", "threat_level": "DANGER"},
    {"zone": "left", "distance": "medium", "threat_level": "PANIC"},
])
def test_sim_invalid_params_rejected(client, audio_mock, bad):
    server.state["audio_ref"] = audio_mock
    r = client.post("/audio/sim", json=bad)
    assert r.status_code == 400
    assert "allowed" in r.get_json()
    audio_mock.play_spatial_beep.assert_not_called()  # never reaches the audio


def test_sim_audio_not_ready(client):
    server.state.pop("audio_ref", None)
    r = client.post("/audio/sim", json={"zone": "left", "distance": "medium",
                                        "threat_level": "DANGER"})
    assert r.status_code == 503


# --- /tts/test ----------------------------------------------------------------

def test_tts_test_valid_speaks(client, audio_mock):
    server.state["audio_ref"] = audio_mock
    r = client.post("/tts/test", json={"text": "peligro derecha"})
    assert r.status_code == 200
    audio_mock.speak.assert_called_once()
    assert audio_mock.speak.call_args.kwargs.get("force") is True


def test_tts_test_empty_text_rejected(client, audio_mock):
    server.state["audio_ref"] = audio_mock
    r = client.post("/tts/test", json={"text": "   "})
    assert r.status_code == 400
    audio_mock.speak.assert_not_called()


def test_tts_test_oversize_text_rejected(client, audio_mock):
    server.state["audio_ref"] = audio_mock
    r = client.post("/tts/test", json={"text": "x" * 201})
    assert r.status_code == 400
    audio_mock.speak.assert_not_called()


def test_tts_test_rate_limited(client, audio_mock):
    server.state["audio_ref"] = audio_mock
    r1 = client.post("/tts/test", json={"text": "uno"})
    r2 = client.post("/tts/test", json={"text": "dos"})   # within cooldown
    assert r1.status_code == 200
    assert r2.status_code == 429
    assert audio_mock.speak.call_count == 1               # second never spoke
