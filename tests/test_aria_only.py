#!/usr/bin/env python3
"""
Minimal test: Aria SDK only, no CUDA, no multiprocessing.
If this crashes, the problem is Aria SDK alone.
"""
import os
# Disable ALL GPU stuff
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NUMBA_DISABLE_CUDA"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time

print("=== TEST: Aria SDK Only (no CUDA) ===")
print()

# Import Aria SDK
print("[1] Importing Aria SDK...")
import aria.sdk as aria
from projectaria_tools.core.calibration import device_calibration_from_json_string
print("    OK")

# Connect
print("[2] Connecting to Aria (USB)...")
device_client = aria.DeviceClient()
device = device_client.connect()
print("    OK - Connected")

# Start streaming
print("[3] Starting streaming...")
streaming_manager = device.streaming_manager
config = aria.StreamingConfig()
config.profile_name = "profile28"
config.streaming_interface = aria.StreamingInterface.Usb
config.security_options.use_ephemeral_certs = True
streaming_manager.streaming_config = config
streaming_manager.start_streaming()
print("    OK - Streaming started")

# Get calibrations
print("[4] Getting calibrations...")
sensors_json = streaming_manager.sensors_calibration()
sensors_calib = device_calibration_from_json_string(sensors_json)
print("    OK")

# Simple observer that just counts frames
class SimpleObserver:
    def __init__(self):
        self.frame_count = 0

    def on_image_received(self, image, record):
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            print(f"    Received {self.frame_count} frames")

    def on_imu_received(self, samples, imu_idx):
        pass

    def on_streaming_client_failure(self, reason, message):
        print(f"    ERROR: {reason} - {message}")

# Subscribe
print("[5] Subscribing...")
observer = SimpleObserver()
streaming_client = streaming_manager.streaming_client
streaming_client.set_streaming_client_observer(observer)
streaming_client.subscribe()
print("    OK - Subscribed")

print()
print("=== SUCCESS! Aria SDK works without CUDA ===")
print("Receiving frames for 10 seconds...")
print()

try:
    for i in range(10):
        time.sleep(1)
        print(f"  {i+1}s - {observer.frame_count} total frames")
except KeyboardInterrupt:
    print("\nInterrupted")

# Cleanup
print()
print("[6] Cleaning up...")
streaming_client.unsubscribe()
streaming_manager.stop_streaming()
device_client.disconnect(device)
print("    OK - Disconnected")
print()
print("=== TEST PASSED ===")
