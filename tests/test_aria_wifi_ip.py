"""WiFi streaming takes the glasses' IP from the caller or ARIA_IP, never from a hard-coded address."""

import pytest

from src.input.aria import resolve_wifi_ip


def test_explicit_ip_wins(monkeypatch):
    monkeypatch.setenv("ARIA_IP", "10.0.0.9")
    assert resolve_wifi_ip("10.0.0.5") == "10.0.0.5"


def test_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("ARIA_IP", " 10.0.0.9 ")
    assert resolve_wifi_ip(None) == "10.0.0.9"


def test_no_ip_anywhere_is_an_error(monkeypatch):
    monkeypatch.delenv("ARIA_IP", raising=False)
    with pytest.raises(ValueError, match="ARIA_IP"):
        resolve_wifi_ip("")
