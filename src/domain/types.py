"""
Core types for ARIA Guard.

This file contains data structures that are shared between modules.
IMPORTANT: This file must NOT import CUDA/torch to keep the main process clean.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


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
    traffic_light_state: Optional[str] = None  # "red", "green", "yellow", None


# YOLO26s Nav model classes (24):
# 0:person 1:bicycle 2:car 3:motorcycle 4:bus 5:truck 6:traffic light
# 7:fire hydrant 8:stop sign 9:bench 10:chair 11:dog 12:cat
# 13:backpack 14:umbrella 15:handbag 16:suitcase 17:potted plant
# 18:Door 19:Stairs 20:Street light 21:Traffic sign 22:Tree 23:Wheelchair

# Filtros de clases por modo
CLASS_FILTERS = {
    "indoor": {
        "person", "chair", "backpack", "handbag", "suitcase", "umbrella",
        "potted plant", "Door", "Stairs", "Wheelchair", "bench",
    },
    "outdoor": {
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "traffic light", "stop sign", "fire hydrant", "dog", "cat",
        "backpack", "handbag", "suitcase", "umbrella",
        "Door", "Stairs", "Street light", "Traffic sign", "Tree", "Wheelchair",
    },
    "all": None  # Sin filtro — las 24 clases
}
