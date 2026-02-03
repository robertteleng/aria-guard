"""
CUDA Detection running in a separate process.

All CUDA operations (YOLO, Depth, Eye Gaze) run here, isolated from Aria SDK.
This follows the aria-nav CentralWorker pattern.

Architecture:
    Main Process          DetectorProcess (spawn)
    - Aria SDK      -->   - YOLO detection
    - Flask         <--   - Depth estimation
    - NO CUDA             - Eye gaze model
"""
import multiprocessing as mp
import time
from typing import Optional, Dict, Any, List
import numpy as np

# Use spawn context for clean CUDA initialization
_ctx = mp.get_context('spawn')


def _detector_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event,
    mode: str = "all",
    enable_depth: bool = True
):
    """
    Worker process that runs CUDA models.

    Args:
        input_queue: Queue receiving frames {"rgb": np.array, "eye": np.array}
        output_queue: Queue sending results {"detections": [...], "depth": np.array, "gaze": {...}}
        ready_event: Event to signal when models are loaded
        mode: Detection mode ("indoor", "outdoor", "all")
        enable_depth: Whether to run depth estimation
    """
    import os
    # Restore CUDA visibility in worker process (main process hides it to avoid FastDDS conflict)
    # Check if we're in Docker or if GPU is available before forcing device 0
    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        # Docker with nvidia-container-toolkit sets this
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # Let nvidia runtime handle it
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("NUMBA_DISABLE_CUDA", None)

    print("[DETECTOR PROCESS] Starting...", flush=True)
    print(f"[DETECTOR PROCESS] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
    print(f"[DETECTOR PROCESS] NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}", flush=True)

    try:
        # Import CUDA modules here, in the worker process
        import torch
        print(f"[DETECTOR PROCESS] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[DETECTOR PROCESS] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("[DETECTOR PROCESS] WARNING: CUDA not available, using CPU (will be slower)", flush=True)

        from src.core.detector import ParallelDetector

        print(f"[DETECTOR PROCESS] Loading models (mode={mode}, depth={enable_depth})...", flush=True)
        detector = ParallelDetector(enable_depth=enable_depth, mode=mode)
        print("[DETECTOR PROCESS] ✓ Models loaded", flush=True)

    except Exception as e:
        print(f"[DETECTOR PROCESS] Failed to load models: {e}", flush=True)
        import traceback
        traceback.print_exc()
        ready_event.set()
        return

    ready_event.set()
    print("[DETECTOR PROCESS] ✓ Ready", flush=True)

    frame_count = 0
    last_log = time.time()

    while True:
        try:
            # Get frame from queue (blocking with timeout)
            try:
                frame_data = input_queue.get(timeout=1.0)
            except:
                continue

            if frame_data is None:
                # Poison pill - stop signal
                print("[DETECTOR PROCESS] Stop signal received", flush=True)
                break

            rgb = frame_data.get("rgb")
            eye = frame_data.get("eye")

            if rgb is None:
                continue

            # Process frame (returns: detections, depth_map, gaze_point, tracked_objects)
            detections, depth_colored, gaze_info, _ = detector.process(rgb, eye)

            # Send results back
            result = {
                "detections": detections,
                "depth": depth_colored,
                "gaze": gaze_info,
                "timestamp": time.time()
            }

            # Non-blocking put (drop if queue full)
            try:
                output_queue.put_nowait(result)
            except:
                pass  # Queue full, drop result

            frame_count += 1

            # Periodic logging
            now = time.time()
            if now - last_log >= 5.0:
                fps = frame_count / (now - last_log)
                print(f"[DETECTOR PROCESS] {frame_count} frames, {fps:.1f} FPS", flush=True)
                frame_count = 0
                last_log = now

        except Exception as e:
            print(f"[DETECTOR PROCESS] Error: {e}", flush=True)

    # Cleanup
    print("[DETECTOR PROCESS] Cleaning up...", flush=True)
    try:
        detector.cleanup()
    except:
        pass
    print("[DETECTOR PROCESS] Stopped", flush=True)


class DetectorProcess:
    """
    Wrapper that manages CUDA detection in a separate process.

    Usage:
        detector = DetectorProcess(mode="outdoor")
        detector.start()

        # Send frame
        detector.send_frame(rgb, eye)

        # Get results (non-blocking)
        result = detector.get_result()
        if result:
            detections = result["detections"]
            depth = result["depth"]

        detector.stop()
    """

    def __init__(self, mode: str = "all", enable_depth: bool = True):
        self._mode = mode
        self._enable_depth = enable_depth
        self._process: Optional[mp.Process] = None
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._ready_event = None
        self._started = False
        self._latest_result: Dict[str, Any] = {}

    def start(self, timeout: float = 60.0) -> bool:
        """
        Start the detector process.

        Args:
            timeout: Max seconds to wait for models to load

        Returns:
            True if started successfully
        """
        if self._started:
            return True

        print("[DetectorProcess] Starting CUDA process...", flush=True)

        self._input_queue = _ctx.Queue(maxsize=2)
        self._output_queue = _ctx.Queue(maxsize=2)
        self._ready_event = _ctx.Event()

        self._process = _ctx.Process(
            target=_detector_worker,
            args=(
                self._input_queue,
                self._output_queue,
                self._ready_event,
                self._mode,
                self._enable_depth
            ),
            daemon=True
        )
        self._process.start()

        # Wait for models to load
        print("[DetectorProcess] Waiting for models to load...", flush=True)
        if self._ready_event.wait(timeout=timeout):
            self._started = True
            print("[DetectorProcess] ✓ Ready", flush=True)
            return True
        else:
            print("[DetectorProcess] ✗ Timeout waiting for models", flush=True)
            self.stop()
            return False

    def send_frame(self, rgb: np.ndarray, eye: Optional[np.ndarray] = None) -> bool:
        """
        Send a frame for processing.

        Args:
            rgb: RGB frame
            eye: Eye tracking frame (optional)

        Returns:
            True if frame was queued
        """
        if not self._started or self._input_queue is None:
            return False

        frame_data = {
            "rgb": rgb,
            "eye": eye
        }

        try:
            # Non-blocking put, drop old frame if queue full
            if self._input_queue.full():
                try:
                    self._input_queue.get_nowait()
                except:
                    pass
            self._input_queue.put_nowait(frame_data)
            return True
        except:
            return False

    def get_result(self) -> Optional[Dict[str, Any]]:
        """
        Get latest detection result (non-blocking).

        Returns:
            Dict with detections, depth, gaze, or None
        """
        if not self._started or self._output_queue is None:
            return None

        # Drain queue to get latest result
        latest = None
        try:
            while True:
                latest = self._output_queue.get_nowait()
        except:
            pass

        if latest:
            self._latest_result = latest

        return self._latest_result if self._latest_result else None

    def stop(self):
        """Stop the detector process."""
        if self._input_queue:
            try:
                self._input_queue.put(None)  # Poison pill
            except:
                pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()

        self._started = False
        print("[DetectorProcess] Stopped", flush=True)

    def __del__(self):
        self.stop()
