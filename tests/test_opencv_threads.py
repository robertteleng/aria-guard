"""Tests for restore_opencv_threads."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.opencv_threads import restore_opencv_threads


def test_restores_all_cpus_after_single_thread(monkeypatch):
    monkeypatch.delenv("ARIA_CV2_THREADS", raising=False)
    cv2.setNumThreads(0)                       # what `import ultralytics` does
    assert restore_opencv_threads() == cv2.getNumberOfCPUs()


def test_env_override(monkeypatch):
    monkeypatch.setenv("ARIA_CV2_THREADS", "2")
    assert restore_opencv_threads() == 2
    monkeypatch.setenv("ARIA_CV2_THREADS", "0")
    assert restore_opencv_threads() == 1       # never below one
