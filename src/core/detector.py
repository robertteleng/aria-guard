"""
Detector: YOLO + Depth + Eye Gaze con CUDA streams.

YOLO detecta objetos, Depth calcula distancia, Meta model estima gaze.
Soporta TensorRT y OpenCV CUDA para máximo rendimiento.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import math
import os

import cv2
import numpy as np
import torch

# Optimizar convs para tamaños fijos
torch.backends.cudnn.benchmark = True

# OpenCV CUDA disponible?
_OPENCV_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0

# Paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = _PROJECT_ROOT / "models"


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


class ParallelDetector:
    """YOLO + Depth en paralelo con CUDA streams."""

    def __init__(self, enable_depth: bool = True, device: str = "cuda", depth_interval: int = 3, mode: str = "all"):
        """
        Args:
            enable_depth: Activar estimación de profundidad
            device: "cuda" o "cpu"
            depth_interval: Procesar depth cada N frames (para performance)
            mode: "indoor", "outdoor" o "all" - filtra clases relevantes
        """
        self.filter_classes = CLASS_FILTERS.get(mode, None)
        if self.filter_classes:
            print(f"[DETECTOR] Modo {mode}: filtrando a {len(self.filter_classes)} clases")
        # Detectar dispositivo
        if device == "cuda" and not torch.cuda.is_available():
            print("[DETECTOR] CUDA no disponible, usando CPU")
            device = "cpu"

        self.device = device
        self.enable_depth = enable_depth
        self.depth_interval = depth_interval
        self._frame_idx = 0
        self._cached_depth = None

        # CUDA streams (solo si hay GPU)
        if device == "cuda":
            self.yolo_stream = torch.cuda.Stream()
            self.depth_stream = torch.cuda.Stream()
            self.gaze_stream = torch.cuda.Stream()
            print(f"[DETECTOR] CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.yolo_stream = None
            self.depth_stream = None
            self.gaze_stream = None

        # Cargar YOLO
        self._load_yolo()

        # Cargar Depth (opcional)
        self.depth_model = None
        self.depth_processor = None
        if enable_depth:
            self._load_depth()

        # Cargar Eye Gaze model (Meta)
        self.gaze_model = None
        self._load_gaze()

        print(f"[DETECTOR] Inicializado (device={device}, depth={enable_depth}, depth_interval={depth_interval})")

    def _load_yolo(self):
        """Carga YOLO26s con TensorRT si está disponible."""
        try:
            from ultralytics import YOLO

            model_name = "yolo26s"
            engine_path = MODELS_DIR / f"{model_name}.engine"
            pt_path = MODELS_DIR / f"{model_name}.pt"

            # Try TensorRT first (NeMo now runs in separate process)
            if self.device == "cuda" and engine_path.exists():
                try:
                    self.yolo = YOLO(str(engine_path), task="detect")
                    self._yolo_tensorrt = True
                    print(f"[DETECTOR] {model_name} cargado (TensorRT)")
                except Exception as e:
                    print(f"[DETECTOR WARN] TensorRT failed: {e}")
                    self.yolo = YOLO(str(pt_path), task="detect")
                    self.yolo.to("cuda")
                    self._yolo_tensorrt = False
                    print(f"[DETECTOR] {model_name} cargado (CUDA PyTorch)")
            elif self.device == "cuda":
                self.yolo = YOLO(str(pt_path), task="detect")
                self.yolo.to("cuda")
                self._yolo_tensorrt = False
                print(f"[DETECTOR] {model_name} cargado (CUDA PyTorch)")
            else:
                self.yolo = YOLO(str(pt_path), task="detect")
                self._yolo_tensorrt = False
                print(f"[DETECTOR] {model_name} cargado (CPU)")
        except Exception as e:
            print(f"[DETECTOR ERROR] No se pudo cargar YOLO: {e}")
            self.yolo = None
            self._yolo_tensorrt = False

    def _load_depth(self):
        """Carga Depth Anything V2. Intenta TensorRT, luego torch.compile, luego FP16."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            self.depth_processor = AutoImageProcessor.from_pretrained(model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self._depth_tensorrt = False

            if self.device == "cuda":
                self.depth_model = self.depth_model.cuda().half().eval()
                # Try torch.compile for faster inference (NeMo now in separate process)
                try:
                    self.depth_model = torch.compile(self.depth_model, mode="reduce-overhead")
                    print("[DETECTOR] Depth Anything V2 (FP16 + torch.compile)")
                except Exception as e:
                    print(f"[DETECTOR WARN] torch.compile failed: {e}")
                    print("[DETECTOR] Depth Anything V2 (FP16)")
            else:
                print("[DETECTOR] Depth Anything V2 (CPU)")
        except Exception as e:
            print(f"[DETECTOR WARN] No se pudo cargar Depth: {e}")
            print("[DETECTOR] Continuando sin profundidad")
            self.depth_model = None
            self.enable_depth = False

    def _load_gaze(self):
        """Carga Meta Eye Gaze model (projectaria_eyetracking)."""
        # PyTorch 2.6+ changed default to weights_only=True, which breaks legacy models
        # using EasyDict. Monkey-patch torch.load before importing the library.
        _original_torch_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _original_torch_load(*args, **kwargs)

        try:
            torch.load = _patched_load
            from projectaria_eyetracking.inference.infer import EyeGazeInference
            import os

            # Path to pretrained weights
            pkg_base = os.path.dirname(__file__)
            venv_path = os.path.join(pkg_base, ".venv/lib/python3.12/site-packages")
            weights_dir = "projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1"

            # Try multiple paths
            possible_paths = [
                os.path.join(venv_path, weights_dir),
                os.path.expanduser(f"~/.cache/projectaria/{weights_dir}"),
            ]

            # Also check site-packages directly
            import site
            for sp in site.getsitepackages():
                possible_paths.append(os.path.join(sp, weights_dir))

            weights_path = None
            for path in possible_paths:
                if os.path.exists(os.path.join(path, "weights.pth")):
                    weights_path = path
                    break

            if weights_path is None:
                print("[DETECTOR WARN] Meta gaze weights not found, using fallback")
                return

            self.gaze_model = EyeGazeInference(
                model_checkpoint_path=os.path.join(weights_path, "weights.pth"),
                model_config_path=os.path.join(weights_path, "config.yaml"),
                device=self.device
            )
            print(f"[DETECTOR] Meta Eye Gaze model loaded ({self.device})")

        except ImportError:
            print("[DETECTOR WARN] projectaria_eyetracking not installed")
        except Exception as e:
            print(f"[DETECTOR WARN] Could not load gaze model: {e}")
        finally:
            # Restore original torch.load
            torch.load = _original_torch_load

    def process(
        self,
        frame: np.ndarray,
        eye_frame: Optional[np.ndarray] = None
    ) -> Tuple[List[Detection], Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """
        Procesa frame: detecta objetos + estima profundidad + gaze (todo en paralelo).

        Args:
            frame: Imagen BGR
            eye_frame: Imagen de eye tracking (opcional)

        Returns:
            (detecciones, depth_map, gaze_point)
        """
        if frame is None:
            return [], None, None

        detections = []
        depth_map = None
        gaze_point = None
        self._frame_idx += 1

        # Decidir si calcular depth este frame
        should_compute_depth = (
            self.enable_depth and
            self.depth_model is not None and
            self._frame_idx % self.depth_interval == 0
        )

        # Ejecutar en paralelo con CUDA streams
        if self.device == "cuda" and self.yolo_stream:
            # Stream 1: YOLO
            with torch.cuda.stream(self.yolo_stream):
                yolo_results = self._run_yolo(frame)

            # Stream 2: Depth (solo cada N frames)
            if should_compute_depth:
                with torch.cuda.stream(self.depth_stream):
                    self._cached_depth = self._run_depth(frame)

            # Stream 3: Gaze (si hay eye frame)
            if eye_frame is not None and self.gaze_stream:
                with torch.cuda.stream(self.gaze_stream):
                    gaze_point = self.estimate_gaze(eye_frame)

            # Sincronizar todos los streams
            torch.cuda.synchronize()
        else:
            # CPU: secuencial
            yolo_results = self._run_yolo(frame)
            if should_compute_depth:
                self._cached_depth = self._run_depth(frame)
            if eye_frame is not None:
                gaze_point = self.estimate_gaze(eye_frame)

        # Usar depth cacheado
        depth_map = self._cached_depth

        # Combinar YOLO + Depth para crear detecciones con distancia
        if yolo_results:
            detections = self._create_detections(yolo_results, depth_map, frame.shape)

        # Marcar objetos que el usuario está mirando
        if gaze_point:
            for det in detections:
                det.is_gazed = self.check_gaze_on_detection(gaze_point, det, frame.shape)

        return detections, depth_map, gaze_point

    def _run_yolo(self, frame: np.ndarray):
        """Ejecuta YOLO con FP16."""
        if self.yolo is None:
            return None
        try:
            results = self.yolo(frame, verbose=False, half=(self.device == "cuda"))
            return results[0] if results else None
        except Exception as e:
            print(f"[DETECTOR ERROR] YOLO: {e}")
            return None

    def _run_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Ejecuta Depth Anything. Usa OpenCV CUDA si está disponible."""
        if self.depth_model is None:
            return None
        try:
            h, w = frame.shape[:2]

            # Preprocesar con OpenCV CUDA si está disponible
            if _OPENCV_CUDA:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                gpu_small = cv2.cuda.resize(gpu_rgb, (384, 384))
                rgb_small = gpu_small.download()
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_small = cv2.resize(rgb, (384, 384))

            # Procesar con modelo
            from PIL import Image
            inputs = self.depth_processor(images=Image.fromarray(rgb_small), return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.cuda().half() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                depth_np = outputs.predicted_depth.squeeze().float().cpu().numpy()

            # Resize a tamaño original (OpenCV CUDA no soporta float32 resize fácilmente)
            depth_resized = cv2.resize(depth_np, (w, h))

            # Normalizar a 0-255
            depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
            return depth_normalized.astype(np.uint8)
        except Exception as e:
            print(f"[DETECTOR ERROR] Depth: {e}")
            return None

    def _create_detections(
        self, yolo_result, depth_map: Optional[np.ndarray], frame_shape
    ) -> List[Detection]:
        """Combina YOLO + Depth para crear detecciones con distancia."""
        detections = []
        h, w = frame_shape[:2]

        if yolo_result.boxes is None:
            return detections

        for box in yolo_result.boxes:
            # Extraer info del bbox
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_result.names[cls_id]

            # Filtrar por clase si hay filtro activo
            if self.filter_classes and name not in self.filter_classes:
                continue

            # Calcular bbox en formato (x, y, w, h)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Calcular zona (izq/centro/der)
            center_x = (x1 + x2) / 2 / w
            if center_x < 0.33:
                zone = "left"
            elif center_x > 0.66:
                zone = "right"
            else:
                zone = "center"

            # Calcular distancia desde depth map
            distance = "unknown"
            depth_value = 0.5
            if depth_map is not None:
                depth_value = self._get_depth_in_bbox(
                    depth_map, int(x1), int(y1), int(x2), int(y2), h, w
                )
                distance = self._depth_to_distance(depth_value)

            detections.append(Detection(
                name=name,
                confidence=conf,
                bbox=bbox,
                zone=zone,
                distance=distance,
                depth_value=depth_value
            ))

        # Ordenar por relevancia (distancia cercana primero)
        distance_order = {"very_close": 0, "close": 1, "medium": 2, "far": 3, "unknown": 4}
        detections.sort(key=lambda d: distance_order.get(d.distance, 4))

        return detections

    def _get_depth_in_bbox(
        self, depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int, h: int, w: int
    ) -> float:
        """Obtiene valor de profundidad promedio en bbox."""
        # Escalar coordenadas al tamaño del depth map
        dh, dw = depth_map.shape[:2]
        scale_x = dw / w
        scale_y = dh / h

        dx1 = max(0, int(x1 * scale_x))
        dy1 = max(0, int(y1 * scale_y))
        dx2 = min(dw, int(x2 * scale_x))
        dy2 = min(dh, int(y2 * scale_y))

        # Región central del bbox (más precisa)
        margin_x = (dx2 - dx1) // 4
        margin_y = (dy2 - dy1) // 4
        cx1 = dx1 + margin_x
        cy1 = dy1 + margin_y
        cx2 = dx2 - margin_x
        cy2 = dy2 - margin_y

        if cx2 > cx1 and cy2 > cy1:
            region = depth_map[cy1:cy2, cx1:cx2]
            if region.size > 0:
                # Valor medio normalizado (0=lejos, 1=cerca en depth anything)
                return float(np.mean(region)) / 255.0

        return 0.5

    def _depth_to_distance(self, depth_value: float) -> str:
        """Convierte valor de profundidad a categoría de distancia."""
        # Depth Anything: valores altos = cerca, valores bajos = lejos
        if depth_value > 0.7:
            return "very_close"
        elif depth_value > 0.5:
            return "close"
        elif depth_value > 0.3:
            return "medium"
        else:
            return "far"

    def estimate_gaze(self, eye_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Estima punto de mirada desde imagen de eye tracking.

        Uses Meta's projectaria_eyetracking model if available,
        falls back to simple pupil detection otherwise.

        Args:
            eye_frame: Imagen de ambos ojos (240x640)

        Returns:
            (x, y) normalizado (0-1, 0-1) o None
        """
        if eye_frame is None:
            return None

        # Try Meta model first
        if self.gaze_model is not None:
            return self._estimate_gaze_meta(eye_frame)

        # Fallback to simple method
        return self._estimate_gaze_simple(eye_frame)

    def _estimate_gaze_meta(self, eye_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Use Meta's projectaria_eyetracking model."""
        try:
            # Convert to tensor
            if len(eye_frame.shape) == 3:
                gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = eye_frame

            img_tensor = torch.tensor(gray, device=self.device)

            # Run inference
            preds, lower, upper = self.gaze_model.predict(img_tensor)
            yaw = float(preds[0][0])  # radians
            pitch = float(preds[0][1])  # radians

            # Check for valid output
            if math.isnan(yaw) or math.isnan(pitch):
                return None

            # Convert yaw/pitch to normalized screen coordinates
            # Aria CPF: yaw is horizontal (-pi/4 to pi/4), pitch is vertical
            # Map to 0-1 range
            gaze_x = 0.5 + (yaw / (math.pi / 4)) * 0.5  # Clamp to 0-1
            gaze_y = 0.5 + (pitch / (math.pi / 4)) * 0.5

            gaze_x = max(0.0, min(1.0, gaze_x))
            gaze_y = max(0.0, min(1.0, gaze_y))

            return (gaze_x, gaze_y)

        except Exception as e:
            print(f"[DETECTOR ERROR] Meta gaze: {e}")
            return None

    def _estimate_gaze_simple(self, eye_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Simple pupil detection fallback."""
        try:
            if len(eye_frame.shape) == 3:
                gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = eye_frame

            h, w = gray.shape
            left_eye = gray[:, :w//2]
            right_eye = gray[:, w//2:]

            left_pos = self._find_pupil(left_eye)
            right_pos = self._find_pupil(right_eye)

            if left_pos and right_pos:
                gaze_x = (left_pos[0] + right_pos[0] + 0.5) / 2
                gaze_y = (left_pos[1] + right_pos[1]) / 2
                return (gaze_x, gaze_y)

            return None
        except Exception as e:
            print(f"[DETECTOR ERROR] Simple gaze: {e}")
            return None

    def _find_pupil(self, eye_region: np.ndarray) -> Optional[Tuple[float, float]]:
        """Encuentra pupila en región del ojo."""
        h, w = eye_region.shape
        blurred = cv2.GaussianBlur(eye_region, (7, 7), 0)
        min_val, _, min_loc, _ = cv2.minMaxLoc(blurred)

        if min_val < 100:
            x, y = min_loc
            return (x / w, y / h)
        return None

    def check_gaze_on_detection(
        self,
        gaze_point: Tuple[float, float],
        detection: Detection,
        frame_shape: Tuple[int, int],
        tolerance: float = 0.1
    ) -> bool:
        """
        Check if gaze point falls on a detection.

        Args:
            gaze_point: (x, y) normalized 0-1
            detection: Detection object
            frame_shape: (height, width)
            tolerance: Extra margin around bbox (fraction of frame)

        Returns:
            True if user is looking at the detection
        """
        h, w = frame_shape[:2]
        gx, gy = gaze_point[0] * w, gaze_point[1] * h

        x, y, bw, bh = detection.bbox
        margin_x = w * tolerance
        margin_y = h * tolerance

        return (x - margin_x <= gx <= x + bw + margin_x and
                y - margin_y <= gy <= y + bh + margin_y)
