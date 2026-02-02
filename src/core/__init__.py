# Non-CUDA imports (safe for main process with Aria SDK)
from .observer import BaseObserver, MockObserver, AriaDemoObserver, AriaDatasetObserver
from .dashboard import Dashboard
from .detector_process import DetectorProcess

# NOTE: ParallelDetector, AudioFeedback use CUDA - import only in DetectorProcess
# from .detector import ParallelDetector, Detection, CLASS_FILTERS
# from .audio import AudioFeedback

__all__ = [
    "BaseObserver",
    "MockObserver",
    "AriaDemoObserver",
    "AriaDatasetObserver",
    "Dashboard",
    "DetectorProcess",
]
