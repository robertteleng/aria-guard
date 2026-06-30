"""Tests for the Spanish TTS alert phrases (es_ES Piper voice).

Guards against the regression where spoken alerts were English ("warning left")
under a Spanish voice.
"""
from src.output.audio import (
    es_alert_phrase,
    TL_STATE_WORDS_ES,
    SIGN_WORDS_ES,
)


class TestSpanishAlertPhrase:
    def test_danger_left(self):
        assert es_alert_phrase("DANGER", "left") == "peligro izquierda"

    def test_danger_right(self):
        assert es_alert_phrase("DANGER", "right") == "peligro derecha"

    def test_warning_center(self):
        assert es_alert_phrase("WARNING", "center") == "precaución al frente"

    def test_attention_is_silent(self):
        # ATTENTION → solo beep, sin voz
        assert es_alert_phrase("ATTENTION", "left") == ""

    def test_unknown_level_is_silent(self):
        assert es_alert_phrase("NONE", "left") == ""

    def test_unknown_zone_yields_just_level(self):
        # zona desconocida → solo el nivel, sin sufijo colgando ni espacios
        assert es_alert_phrase("DANGER", "???") == "peligro"

    def test_no_english_words_anywhere(self):
        """Regresión: ninguna frase hablada contiene palabras en inglés."""
        english = ("warning", "danger", "left", "right", "straight", "center")
        for lvl in ("DANGER", "WARNING"):
            for zone in ("left", "center", "right"):
                phrase = es_alert_phrase(lvl, zone)
                assert phrase, "DANGER/WARNING deben hablar"
                low = phrase.lower()
                assert not any(w in low for w in english), f"frase en inglés: {phrase!r}"


class TestContextMaps:
    def test_traffic_light_states_spanish(self):
        assert TL_STATE_WORDS_ES == {"red": "rojo", "green": "verde", "yellow": "amarillo"}

    def test_stop_sign_spanish(self):
        assert SIGN_WORDS_ES.get("stop sign") == "stop"
