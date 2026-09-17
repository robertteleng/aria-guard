"""
Keep OpenCV multithreaded in inference processes.

`import ultralytics` calls cv2.setNumThreads(0) (single thread, to avoid a clash
with PyTorch DataLoader workers during training). aria-guard does not train; in
its pipeline that made every resize and colour conversion single-threaded. On
the Jetson Orin Nano a 1408->640 resize went from 1.2 to 6.6 ms, and the image
work of each frame paid it several times.
"""

import os

import cv2


def restore_opencv_threads() -> int:
    """Set OpenCV threads to ARIA_CV2_THREADS, or to all CPUs. Returns the value set."""
    requested = os.environ.get("ARIA_CV2_THREADS")
    n = int(requested) if requested else cv2.getNumberOfCPUs()
    cv2.setNumThreads(max(1, n))
    return cv2.getNumThreads()
