"""Tests for the hardware-free parts of scripts/replay_vrs.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_offline import FrameResult, run_benchmark
from scripts.replay_vrs import (detections_to_dicts, imu_decimation, latency_summary,
                                next_realtime_index, wait_before_frame_ns, wrap_stage)
from src.domain.types import Detection


class TestLatencySummary:
    def test_percentiles(self):
        s = latency_summary(list(range(1, 101)))
        assert s["n"] == 100
        assert s["mean"] == 50.5
        assert s["p50"] == 50.5
        assert s["p95"] == pytest.approx(95.05)
        assert s["max"] == 100

    def test_empty(self):
        assert latency_summary([]) == {"n": 0}


class TestNextRealtimeIndex:
    TS = [i * 33_333_333 for i in range(10)]  # 30 FPS

    def test_fast_pipeline_advances_one(self):
        assert next_realtime_index(self.TS, 0, elapsed_ns=1_000, current=0) == 1

    def test_slow_pipeline_skips_to_newest_captured(self):
        # 100 ms after start the glasses have delivered frames 0..3
        assert next_realtime_index(self.TS, 0, elapsed_ns=100_000_000, current=0) == 3

    def test_never_goes_backwards(self):
        assert next_realtime_index(self.TS, 0, elapsed_ns=0, current=5) == 6

    def test_exhausted_recording(self):
        assert next_realtime_index(self.TS, 0, elapsed_ns=10**12, current=2) == 9
        assert next_realtime_index(self.TS, 0, elapsed_ns=10**12, current=9) == 10


class TestWaitBeforeFrame:
    TS = [i * 33_333_333 for i in range(10)]

    def test_fast_pipeline_waits_for_capture(self):
        assert wait_before_frame_ns(self.TS, 0, elapsed_ns=20_000_000, idx=1) == 13_333_333

    def test_no_wait_when_frame_already_captured(self):
        assert wait_before_frame_ns(self.TS, 0, elapsed_ns=50_000_000, idx=1) == 0

    def test_uses_recording_start_offset(self):
        ts = [t + 10**9 for t in self.TS]
        assert wait_before_frame_ns(ts, 10**9, elapsed_ns=0, idx=2) == 66_666_666

    def test_past_end(self):
        assert wait_before_frame_ns(self.TS, 0, elapsed_ns=0, idx=10) == 0


class TestImuDecimation:
    def test_1khz_to_100hz(self):
        assert imu_decimation(1000.0, 100.0) == 10

    def test_never_below_one(self):
        assert imu_decimation(50.0, 100.0) == 1

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            imu_decimation(1000.0, 0)


def test_detections_to_dicts_roundtrip_through_benchmark():
    det = Detection(name="car", confidence=0.9, bbox=(600, 300, 100, 80), zone="center",
                    distance="close", depth_value=0.8)
    d = detections_to_dicts([det])[0]
    assert d["bbox"] == [600, 300, 100, 80]
    assert d["is_gazed"] is False and d["traffic_light_state"] is None
    frames = [FrameResult(frame_idx=i, timestamp=i / 30, detections=[d], motion_state="walking")
              for i in range(60)]
    step_ms = []
    metrics = run_benchmark(frames, video_fps=30, frame_width=1408, fov_h=1.919, step_ms=step_ms)
    assert metrics.total_frames == 60
    assert len(step_ms) == 60


def test_wrap_stage_times_and_syncs():
    calls = []

    class Obj:
        def work(self, x):
            calls.append("work")
            return x * 2

    obj, sink = Obj(), {}
    wrap_stage(obj, "work", sink, sync=lambda: calls.append("sync"))
    assert obj.work(21) == 42
    assert calls == ["sync", "work", "sync"]
    assert len(sink["work"]) == 1 and sink["work"][0] >= 0


class FakeRecording:
    """Stands in for VrsRecording: 20 frames 5 ms apart, tiny images."""

    def __init__(self, n=20, step_ns=5_000_000, read_delay_s=0.0):
        import numpy as np
        self.rgb_ts = [10**9 + i * step_ns for i in range(n)]
        self._np, self._delay, self.reads = np, read_delay_s, []

    def read(self, idx):
        import time
        time.sleep(self._delay)
        self.reads.append(idx)
        return self.rgb_ts[idx], self._np.zeros((2, 2, 3), "uint8"), None

    def motion_state_at(self, ts):
        return "walking"


def _drain(feed):
    got, last = [], -1
    while (item := feed.newest_after(last)) is not None:
        got.append(item[0])
        last = item[0]
    return got


def test_live_feed_delivers_every_frame_when_decoder_keeps_up():
    from scripts.replay_vrs import LiveFeed
    rec = FakeRecording()
    feed = LiveFeed(rec, n_frames=20)
    got = _drain(feed)
    assert rec.reads == list(range(20))
    assert feed.decoded == 20 and feed.skipped_by_decoder == 0 and feed.finished
    assert got == sorted(set(got)) and got[-1] == 19  # newest, increasing, reaches the end


def test_live_feed_skips_frames_when_decoding_is_slower_than_capture():
    from scripts.replay_vrs import LiveFeed
    rec = FakeRecording(n=20, step_ns=5_000_000, read_delay_s=0.02)  # 20 ms decode vs 5 ms capture
    feed = LiveFeed(rec, n_frames=20)
    _drain(feed)
    assert feed.skipped_by_decoder > 0
    assert feed.decoded + feed.skipped_by_decoder <= 20
    assert rec.reads == sorted(rec.reads)


def test_live_feed_respects_n_frames_limit():
    from scripts.replay_vrs import LiveFeed
    rec = FakeRecording(n=20)
    feed = LiveFeed(rec, n_frames=5)
    assert _drain(feed)[-1] == 4
    assert max(rec.reads) == 4
