"""
Run a slow per-frame model (depth) on its own thread, latest result only.

Synchronously, the frame that also runs Depth Anything waits for it: on the
Jetson that frame took ~110 ms instead of ~30 ms. With this worker the
detector submits a frame when the worker is idle and keeps using the newest
finished map, so no frame waits for depth. The map is at most one depth
inference older than in the synchronous path.
"""

import threading
from typing import Any, Callable, Optional


class LatestOnlyWorker:
    """Single background thread; submissions while busy are dropped."""

    def __init__(self, fn: Callable[[Any], Any], name: str = "worker",
                 cuda_stream: Optional[Any] = None):
        self._fn = fn
        self._stream = cuda_stream
        self._cond = threading.Condition()
        self._pending = None
        self._has_pending = False
        self._busy = False
        self._stopped = False
        self._latest = None
        self.completed = 0
        self.dropped = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item: Any) -> bool:
        """Queue item unless a job is running or queued. Returns whether it was accepted."""
        with self._cond:
            if self._stopped or self._busy or self._has_pending:
                self.dropped += 1
                return False
            self._pending, self._has_pending = item, True
            self._cond.notify()
            return True

    def latest(self) -> Any:
        with self._cond:
            return self._latest

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until nothing is running or queued (tests, warm-up)."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._busy and not self._has_pending, timeout)

    def stop(self, timeout: float = 2.0):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout)

    def _run(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._has_pending or self._stopped)
                if self._stopped:
                    return
                item, self._pending, self._has_pending = self._pending, None, False
                self._busy = True
            try:
                result = self._call(item)
            except Exception as e:  # keep the last good map; report and continue
                print(f"[ASYNC {self._thread.name}] error: {e}", flush=True)
                result, failed = None, True
            else:
                failed = False
            with self._cond:
                if failed:
                    self.errors += 1
                elif result is not None:
                    self._latest = result
                    self.completed += 1
                self._busy = False
                self._cond.notify_all()

    def _call(self, item):
        if self._stream is None:
            return self._fn(item)
        import torch
        with torch.cuda.stream(self._stream):
            out = self._fn(item)
        self._stream.synchronize()
        return out
