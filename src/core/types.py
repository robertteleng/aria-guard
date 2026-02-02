"""
Core types for ARIA demo.

This file contains data structures that are shared between modules.
IMPORTANT: This file must NOT import CUDA/torch to keep the main process clean.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Objeto detectado con distancia."""
    name: str           # "chair", "person", etc.
    confidence: float   # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    zone: str           # "left", "center", "right"
    distance: str       # "very_close", "close", "medium", "far"
    depth_value: float  # 0.0 - 1.0 (normalizado)
    is_gazed: bool = False  # True if user is looking at this object


# Filtros de clases por modo
CLASS_FILTERS = {
    "indoor": {"person", "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "door", "refrigerator", "oven", "sink", "backpack", "handbag", "suitcase"},
    "outdoor": {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "dog", "cat", "backpack", "handbag", "suitcase"},
    "all": None  # Sin filtro
}
