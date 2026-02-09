"""
CUDA Detection running in a separate process.

All CUDA operations (YOLO, Depth, Eye Gaze) run here, isolated from Aria SDK.
This follows the aria-nav CentralWorker pattern.

Architecture:
    Main Process          DetectorProcess (spawn)
    - Aria SDK      -->   - YOLO detection
    - Flask         <--   - Depth estimation
    - NO CUDA             - Eye gaze model

Optimizations:
    - Shared memory for frame transfer (avoids queue serialization)
    - Zero-copy frame passing between processes
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import Optional, Dict, Any, List
import numpy as np

# Use spawn context for clean CUDA initialization
_ctx = mp.get_context('spawn')

# Shared memory configuration
SHM_FRAME_NAME = "aria_frame"
SHM_DEPTH_NAME = "aria_depth"
SHM_HW_DEPTH_NAME = "aria_hw_depth"
MAX_FRAME_SIZE = 1920 * 1080 * 3  # Max 1080p BGR
MAX_DEPTH_SIZE = 1920 * 1080      # Max 1080p grayscale
MAX_HW_DEPTH_SIZE = 1920 * 1080 * 2  # Max 1080p uint16 (RealSense, mm)


def _detector_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event,
    mode: str = "all",
    enable_depth: bool = True,
    use_shared_memory: bool = False,
    shm_frame_name: str = None,
    shm_depth_name: str = None,
    frame_shape: tuple = None,
    frame_ready_event = None,
    result_ready_event = None,
    shm_hw_depth_name: str = None,
    has_hardware_depth: bool = False
):
    """
    Worker process that runs CUDA models.

    Args:
        input_queue: Queue receiving frames or control signals
        output_queue: Queue sending results
        ready_event: Event to signal when models are loaded
        mode: Detection mode ("indoor", "outdoor", "all")
        enable_depth: Whether to run depth estimation
        use_shared_memory: If True, use shared memory for frame transfer
        shm_frame_name: Name of shared memory block for input frames
        shm_depth_name: Name of shared memory block for depth output
        frame_shape: Shape of input frame (H, W, C)
        frame_ready_event: Event signaling new frame is ready
        result_ready_event: Event signaling result is ready
    """
    import os
    # Restore CUDA visibility in worker process (main process hides it to avoid FastDDS conflict)
    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("NUMBA_DISABLE_CUDA", None)

    print("[DETECTOR PROCESS] Starting...", flush=True)
    print(f"[DETECTOR PROCESS] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
    print(f"[DETECTOR PROCESS] NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}", flush=True)

    # Shared memory setup
    shm_frame = None
    shm_depth = None
    shm_hw_depth = None
    if use_shared_memory and shm_frame_name and frame_shape:
        try:
            shm_frame = shared_memory.SharedMemory(name=shm_frame_name)
            if shm_depth_name:
                shm_depth = shared_memory.SharedMemory(name=shm_depth_name)
            if shm_hw_depth_name:
                shm_hw_depth = shared_memory.SharedMemory(name=shm_hw_depth_name)
            print(f"[DETECTOR PROCESS] ✓ Shared memory attached", flush=True)
        except Exception as e:
            print(f"[DETECTOR PROCESS] Shared memory failed: {e}, using queues", flush=True)
            use_shared_memory = False

    try:
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
    detector_fps = 0.0

    while True:
        try:
            rgb = None
            eye = None
            hardware_depth = None

            if use_shared_memory and frame_ready_event:
                # Wait for new frame signal
                if frame_ready_event.wait(timeout=1.0):
                    frame_ready_event.clear()
                    # Read from shared memory (zero-copy)
                    rgb = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_frame.buf)
                    rgb = rgb.copy()  # Make a copy to release the buffer
                    # Read hardware depth if available (RealSense D435)
                    if has_hardware_depth and shm_hw_depth:
                        hw_depth_shape = (frame_shape[0], frame_shape[1])
                        hardware_depth = np.ndarray(hw_depth_shape, dtype=np.uint16, buffer=shm_hw_depth.buf)
                        hardware_depth = hardware_depth.copy()
                else:
                    continue
            else:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_data = input_queue.get(timeout=1.0)
                except:
                    continue

                if frame_data is None:
                    print("[DETECTOR PROCESS] Stop signal received", flush=True)
                    break

                rgb = frame_data.get("rgb")
                eye = frame_data.get("eye")
                hardware_depth = frame_data.get("hardware_depth")

            if rgb is None:
                continue

            # Process frame
            detections, depth_colored, gaze_info, _ = detector.process(rgb, eye, hardware_depth)

            # Send results
            if use_shared_memory and shm_depth and depth_colored is not None and result_ready_event:
                # Write depth to shared memory
                depth_flat = depth_colored.flatten()
                shm_depth.buf[:len(depth_flat)] = depth_flat.tobytes()

            result = {
                "detections": detections,
                "depth": depth_colored,  # Always send depth via queue (small overhead)
                "gaze": gaze_info,
                "timestamp": time.time(),
                "detector_fps": detector_fps
            }

            try:
                output_queue.put_nowait(result)
                if result_ready_event:
                    result_ready_event.set()
            except:
                pass

            frame_count += 1

            # Periodic logging
            now = time.time()
            if now - last_log >= 5.0:
                detector_fps = frame_count / (now - last_log)
                print(f"[DETECTOR PROCESS] {frame_count} frames, {detector_fps:.1f} FPS", flush=True)
                frame_count = 0
                last_log = now

        except Exception as e:
            print(f"[DETECTOR PROCESS] Error: {e}", flush=True)

    # Cleanup shared memory
    if shm_frame:
        shm_frame.close()
    if shm_depth:
        shm_depth.close()

    print("[DETECTOR PROCESS] Cleaning up...", flush=True)
    try:
        detector.cleanup()
    except:
        pass
    print("[DETECTOR PROCESS] Stopped", flush=True)


class DetectorProcess:
    """
    Wrapper that manages CUDA detection in a separate process.

    Supports two modes:
    1. Queue-based (default): Uses multiprocessing queues for frame transfer
    2. Shared memory: Zero-copy frame transfer (faster, less CPU)

    Usage:
        detector = DetectorProcess(mode="outdoor", use_shared_memory=True)
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

    def __init__(self, mode: str = "all", enable_depth: bool = True, use_shared_memory: bool = True, has_hardware_depth: bool = False):
        self._mode = mode
        self._enable_depth = enable_depth
        self._use_shared_memory = use_shared_memory
        self._has_hardware_depth = has_hardware_depth
        self._process: Optional[mp.Process] = None
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._ready_event = None
        self._started = False
        self._latest_result: Dict[str, Any] = {}

        # Shared memory
        self._shm_frame = None
        self._shm_depth = None
        self._shm_hw_depth = None
        self._frame_shape = None
        self._frame_ready_event = None
        self._result_ready_event = None

    def start(self, timeout: float = 60.0, frame_shape: tuple = (1080, 1920, 3)) -> bool:
        """
        Start the detector process.

        Args:
            timeout: Max seconds to wait for models to load
            frame_shape: Expected frame shape (H, W, C) for shared memory allocation

        Returns:
            True if started successfully
        """
        if self._started:
            return True

        print("[DetectorProcess] Starting CUDA process...", flush=True)

        self._input_queue = _ctx.Queue(maxsize=2)
        self._output_queue = _ctx.Queue(maxsize=2)
        self._ready_event = _ctx.Event()

        # Initialize shared memory if enabled
        shm_frame_name = None
        shm_depth_name = None
        shm_hw_depth_name = None
        self._frame_shape = frame_shape

        if self._use_shared_memory:
            try:
                # Create shared memory for input frame
                frame_size = int(np.prod(frame_shape))
                self._shm_frame = shared_memory.SharedMemory(
                    create=True,
                    size=frame_size,
                    name=f"{SHM_FRAME_NAME}_{id(self)}"
                )
                shm_frame_name = self._shm_frame.name

                # Create shared memory for depth output
                depth_shape = (frame_shape[0], frame_shape[1])
                depth_size = int(np.prod(depth_shape))
                self._shm_depth = shared_memory.SharedMemory(
                    create=True,
                    size=depth_size,
                    name=f"{SHM_DEPTH_NAME}_{id(self)}"
                )
                shm_depth_name = self._shm_depth.name

                # Create shared memory for hardware depth input (RealSense uint16 mm)
                if self._has_hardware_depth:
                    hw_depth_size = frame_shape[0] * frame_shape[1] * 2  # uint16
                    self._shm_hw_depth = shared_memory.SharedMemory(
                        create=True,
                        size=hw_depth_size,
                        name=f"{SHM_HW_DEPTH_NAME}_{id(self)}"
                    )
                    shm_hw_depth_name = self._shm_hw_depth.name

                # Events for synchronization
                self._frame_ready_event = _ctx.Event()
                self._result_ready_event = _ctx.Event()

                print(f"[DetectorProcess] ✓ Shared memory allocated ({frame_size + depth_size} bytes)", flush=True)
            except Exception as e:
                print(f"[DetectorProcess] Shared memory failed: {e}, using queues", flush=True)
                self._use_shared_memory = False
                self._shm_frame = None
                self._shm_depth = None
                self._shm_hw_depth = None

        self._process = _ctx.Process(
            target=_detector_worker,
            args=(
                self._input_queue,
                self._output_queue,
                self._ready_event,
                self._mode,
                self._enable_depth,
                self._use_shared_memory,
                shm_frame_name,
                shm_depth_name,
                frame_shape,
                self._frame_ready_event,
                self._result_ready_event,
                shm_hw_depth_name,
                self._has_hardware_depth
            ),
            daemon=True
        )
        self._process.start()

        # Wait for models to load
        print("[DetectorProcess] Waiting for models to load...", flush=True)
        if self._ready_event.wait(timeout=timeout):
            self._started = True
            mode_str = "shared memory" if self._use_shared_memory else "queues"
            print(f"[DetectorProcess] ✓ Ready ({mode_str})", flush=True)
            return True
        else:
            print("[DetectorProcess] ✗ Timeout waiting for models", flush=True)
            self.stop()
            return False

    def send_frame(
        self,
        rgb: np.ndarray,
        eye: Optional[np.ndarray] = None,
        hardware_depth: Optional[np.ndarray] = None
    ) -> bool:
        """
        Send a frame for processing.

        Args:
            rgb: RGB frame
            eye: Eye tracking frame (optional)
            hardware_depth: Hardware depth map from RealSense D435 (optional).
                           If provided, skips AI depth estimation.

        Returns:
            True if frame was queued/written
        """
        if not self._started:
            return False

        # Shared memory path (zero-copy)
        if self._use_shared_memory and self._shm_frame and self._frame_ready_event:
            try:
                # Resize frame if needed to fit shared memory
                if rgb.shape != self._frame_shape:
                    import cv2
                    rgb = cv2.resize(rgb, (self._frame_shape[1], self._frame_shape[0]))

                # Write to shared memory (direct copy)
                np.ndarray(self._frame_shape, dtype=np.uint8, buffer=self._shm_frame.buf)[:] = rgb

                # Write hardware depth if available (RealSense D435)
                if hardware_depth is not None and self._shm_hw_depth:
                    hw_depth_shape = (self._frame_shape[0], self._frame_shape[1])
                    if hardware_depth.shape[:2] != hw_depth_shape:
                        import cv2
                        hardware_depth = cv2.resize(hardware_depth, (hw_depth_shape[1], hw_depth_shape[0]))
                    np.ndarray(hw_depth_shape, dtype=np.uint16, buffer=self._shm_hw_depth.buf)[:] = hardware_depth

                # Signal frame is ready
                self._frame_ready_event.set()
                return True
            except Exception as e:
                # Fallback to queue
                pass

        # Queue path (with serialization)
        if self._input_queue is None:
            return False

        frame_data = {
            "rgb": rgb,
            "eye": eye,
            "hardware_depth": hardware_depth
        }

        try:
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

        # Cleanup shared memory
        if self._shm_frame:
            try:
                self._shm_frame.close()
                self._shm_frame.unlink()
            except:
                pass
            self._shm_frame = None

        if self._shm_depth:
            try:
                self._shm_depth.close()
                self._shm_depth.unlink()
            except:
                pass
            self._shm_depth = None

        if self._shm_hw_depth:
            try:
                self._shm_hw_depth.close()
                self._shm_hw_depth.unlink()
            except:
                pass
            self._shm_hw_depth = None

        self._started = False
        print("[DetectorProcess] Stopped", flush=True)

    def __del__(self):
        self.stop()
