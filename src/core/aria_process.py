"""
Aria SDK running in a separate process to avoid CUDA context conflicts.

The Aria SDK (FastDDS) cannot coexist with CUDA in the same process.
This module isolates Aria in its own process and sends frames via Queue.
"""
import multiprocessing as mp
import threading
import time
from typing import Optional, Dict, Any
import numpy as np

# Use spawn context for clean process (no inherited CUDA context)
_ctx = mp.get_context('spawn')


def _aria_worker(
    frame_queue: mp.Queue,
    control_queue: mp.Queue,
    ready_event,
    interface: str,
    ip_address: Optional[str]
):
    """
    Worker process that runs Aria SDK and sends frames to main process.

    Args:
        frame_queue: Queue to send frames (dict with rgb, eye, slam1, slam2)
        control_queue: Queue for control commands (stop, etc)
        ready_event: Event to signal when ready
        interface: "usb" or "wifi"
        ip_address: IP for WiFi connection
    """
    import cv2

    print("[ARIA PROCESS] Starting...")

    try:
        import aria.sdk as aria
        from projectaria_tools.core.calibration import device_calibration_from_json_string
    except ImportError as e:
        print(f"[ARIA PROCESS] Failed to import Aria SDK: {e}")
        ready_event.set()
        return

    # Frame storage (updated by callbacks)
    frames = {
        "rgb": None,
        "eye": None,
        "slam1": None,
        "slam2": None
    }
    frame_lock = threading.Lock()
    frame_counts = {k: 0 for k in frames}
    start_time = time.time()

    class AriaObserver:
        """Observer that receives callbacks from Aria SDK."""

        def on_image_received(self, image: np.ndarray, record) -> None:
            camera_id = record.camera_id

            if camera_id == aria.CameraId.Rgb:
                processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                key = "rgb"
            elif camera_id == aria.CameraId.EyeTrack:
                processed = np.rot90(image, 2)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                key = "eye"
            elif camera_id == aria.CameraId.Slam1:
                processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                key = "slam1"
            elif camera_id == aria.CameraId.Slam2:
                processed = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                key = "slam2"
            else:
                return

            with frame_lock:
                frames[key] = processed
                frame_counts[key] += 1

        def on_imu_received(self, samples, imu_idx: int) -> None:
            pass  # IMU not needed for now

    # Connect to Aria
    try:
        print(f"[ARIA PROCESS] Connecting ({interface.upper()})...")
        device_client = aria.DeviceClient()

        if interface.lower() == "wifi":
            if not ip_address:
                ip_address = "<ARIA_IP>"
            client_config = aria.DeviceClientConfig()
            client_config.ip_v4_address = ip_address
            device_client.set_client_config(client_config)
            print(f"[ARIA PROCESS] WiFi target: {ip_address}")

        device = device_client.connect()
        print("[ARIA PROCESS] ✓ Connected")

        # Start streaming
        streaming_manager = device.streaming_manager
        config = aria.StreamingConfig()

        if interface.lower() == "wifi":
            config.profile_name = "profile18"
            config.streaming_interface = aria.StreamingInterface.WifiStation
        else:
            config.profile_name = "profile28"
            config.streaming_interface = aria.StreamingInterface.Usb

        config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = config
        streaming_manager.start_streaming()
        print(f"[ARIA PROCESS] ✓ Streaming started ({config.profile_name})")

        # Get calibrations
        calibrations = None
        try:
            sensors_json = streaming_manager.sensors_calibration()
            calibrations = sensors_json  # Send raw JSON to main process
            print("[ARIA PROCESS] ✓ Calibrations obtained")
        except Exception as e:
            print(f"[ARIA PROCESS] Warning: Could not get calibrations: {e}")

        # Register observer
        streaming_client = streaming_manager.streaming_client
        observer = AriaObserver()
        streaming_client.set_streaming_client_observer(observer)
        streaming_client.subscribe()

        print("[ARIA PROCESS] ✓ Ready")
        ready_event.set()

        # Main loop: send frames to queue
        last_send = 0
        send_interval = 1.0 / 30  # Target 30 FPS

        while True:
            # Check for stop command
            try:
                cmd = control_queue.get_nowait()
                if cmd == "stop":
                    print("[ARIA PROCESS] Stop command received")
                    break
            except:
                pass

            # Send frames at controlled rate
            now = time.time()
            if now - last_send >= send_interval:
                with frame_lock:
                    # Only send if we have RGB frame
                    if frames["rgb"] is not None:
                        # Copy frames to avoid race conditions
                        frame_data = {
                            "rgb": frames["rgb"].copy() if frames["rgb"] is not None else None,
                            "eye": frames["eye"].copy() if frames["eye"] is not None else None,
                            "slam1": frames["slam1"].copy() if frames["slam1"] is not None else None,
                            "slam2": frames["slam2"].copy() if frames["slam2"] is not None else None,
                            "timestamp": now
                        }

                        # Non-blocking put (drop if queue full)
                        try:
                            frame_queue.put_nowait(frame_data)
                        except:
                            pass  # Queue full, drop frame

                        last_send = now

            time.sleep(0.001)  # Small sleep to prevent busy loop

        # Cleanup
        print("[ARIA PROCESS] Cleaning up...")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)

    except Exception as e:
        print(f"[ARIA PROCESS] Error: {e}")
        import traceback
        traceback.print_exc()
        ready_event.set()

    print("[ARIA PROCESS] Stopped")


class AriaProcess:
    """
    Wrapper that manages Aria SDK in a separate process.

    Usage:
        aria = AriaProcess(interface="usb")
        aria.start()

        while True:
            frames = aria.get_frames()
            if frames:
                rgb = frames["rgb"]
                eye = frames["eye"]
                ...

        aria.stop()
    """

    def __init__(self, interface: str = "usb", ip_address: Optional[str] = None):
        self._interface = interface
        self._ip_address = ip_address
        self._process: Optional[mp.Process] = None
        self._frame_queue: Optional[mp.Queue] = None
        self._control_queue: Optional[mp.Queue] = None
        self._ready_event = None
        self._latest_frames: Dict[str, Any] = {}
        self._started = False

    def start(self, timeout: float = 30.0) -> bool:
        """
        Start the Aria process.

        Args:
            timeout: Max seconds to wait for Aria to be ready

        Returns:
            True if started successfully
        """
        if self._started:
            return True

        print("[AriaProcess] Starting separate process for Aria SDK...")

        self._frame_queue = _ctx.Queue(maxsize=5)
        self._control_queue = _ctx.Queue()
        self._ready_event = _ctx.Event()

        self._process = _ctx.Process(
            target=_aria_worker,
            args=(
                self._frame_queue,
                self._control_queue,
                self._ready_event,
                self._interface,
                self._ip_address
            ),
            daemon=True
        )
        self._process.start()

        # Wait for ready
        print("[AriaProcess] Waiting for Aria to be ready...")
        if self._ready_event.wait(timeout=timeout):
            self._started = True
            print("[AriaProcess] ✓ Aria process ready")
            return True
        else:
            print("[AriaProcess] ✗ Timeout waiting for Aria")
            self.stop()
            return False

    def get_frames(self) -> Optional[Dict[str, Any]]:
        """
        Get latest frames from Aria.

        Returns:
            Dict with rgb, eye, slam1, slam2 frames, or None if no frames
        """
        if not self._started or self._frame_queue is None:
            return None

        # Drain queue to get latest frames
        latest = None
        try:
            while True:
                latest = self._frame_queue.get_nowait()
        except:
            pass

        if latest:
            self._latest_frames = latest

        return self._latest_frames if self._latest_frames else None

    def get_frame(self, camera: str = "rgb") -> Optional[np.ndarray]:
        """
        Get a specific camera frame (compatible with BaseObserver interface).

        Args:
            camera: "rgb", "eye", "slam1", or "slam2"

        Returns:
            Frame as numpy array, or None
        """
        frames = self.get_frames()
        if frames:
            return frames.get(camera)
        return None

    def stop(self):
        """Stop the Aria process."""
        if self._control_queue:
            try:
                self._control_queue.put("stop")
            except:
                pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()

        self._started = False
        print("[AriaProcess] Stopped")

    def __del__(self):
        self.stop()
