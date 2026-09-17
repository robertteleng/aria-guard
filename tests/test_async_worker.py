"""Tests for LatestOnlyWorker (no GPU)."""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.async_worker import LatestOnlyWorker


def test_result_becomes_latest():
    w = LatestOnlyWorker(lambda x: x * 2)
    assert w.latest() is None
    assert w.submit(21)
    assert w.wait_idle(2)
    assert w.latest() == 42 and w.completed == 1
    w.stop()


def test_submissions_while_busy_are_dropped():
    gate = threading.Event()
    started = threading.Event()

    def slow(x):
        started.set()
        gate.wait(2)
        return x

    w = LatestOnlyWorker(slow)
    assert w.submit(1)
    assert started.wait(2)
    assert not w.submit(2)          # busy
    assert not w.submit(3)
    assert w.dropped == 2
    gate.set()
    assert w.wait_idle(2)
    assert w.latest() == 1
    assert w.submit(4) and w.wait_idle(2) and w.latest() == 4
    w.stop()


def test_error_keeps_previous_result():
    def fn(x):
        if x == "boom":
            raise RuntimeError("bad frame")
        return x

    w = LatestOnlyWorker(fn)
    w.submit("ok")
    w.wait_idle(2)
    w.submit("boom")
    w.wait_idle(2)
    assert w.latest() == "ok" and w.errors == 1 and w.completed == 1
    w.stop()


def test_none_result_does_not_replace_latest():
    results = iter(["map", None])
    w = LatestOnlyWorker(lambda _: next(results))
    w.submit(0); w.wait_idle(2)
    w.submit(0); w.wait_idle(2)
    assert w.latest() == "map"
    w.stop()


def test_stop_rejects_new_work():
    w = LatestOnlyWorker(lambda x: x)
    w.stop()
    assert not w.submit(1)
