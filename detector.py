"""
Detector: YOLO + Depth con CUDA streams.

KISS: YOLO detecta objetos, Depth calcula distancia.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class Detection:
    """Objeto detectado con distancia."""
    name: str           # "chair", "person", etc.
    confidence: float   # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    zone: str           # "left", "center", "right"
    distance: str       # "very_close", "close", "medium", "far"
    depth_value: float  # 0.0 - 1.0 (normalizado)


class ParallelDetector:
    """YOLO + Depth en paralelo con CUDA streams."""

    def __init__(self, enable_depth: bool = True, device: str = "cuda", depth_interval: int = 3):
        """
        Args:
            enable_depth: Activar estimación de profundidad
            device: "cuda" o "cpu"
            depth_interval: Procesar depth cada N frames (para performance)
        """
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
            print(f"[DETECTOR] CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.yolo_stream = None
            self.depth_stream = None

        # Cargar YOLO
        self._load_yolo()

        # Cargar Depth (opcional)
        self.depth_model = None
        self.depth_processor = None
        if enable_depth:
            self._load_depth()

        print(f"[DETECTOR] Inicializado (device={device}, depth={enable_depth}, depth_interval={depth_interval})")

    def _load_yolo(self):
        """Carga YOLO26s con PyTorch."""
        try:
            from ultralytics import YOLO

            model_name = "yolo26s"
            self.yolo = YOLO(f"{model_name}.pt")

            if self.device == "cuda":
                self.yolo.to("cuda")
                print(f"[DETECTOR] {model_name} cargado (CUDA)")
            else:
                print(f"[DETECTOR] {model_name} cargado (CPU)")
        except Exception as e:
            print(f"[DETECTOR ERROR] No se pudo cargar YOLO: {e}")
            self.yolo = None

    def _load_depth(self):
        """Carga Depth Anything V2 con PyTorch FP16."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            self.depth_processor = AutoImageProcessor.from_pretrained(model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)

            if self.device == "cuda":
                self.depth_model = self.depth_model.cuda().half()
                self.depth_model.eval()
                # Verificar que está en GPU
                param = next(self.depth_model.parameters())
                print(f"[DETECTOR] Depth Anything V2 cargado (device={param.device}, dtype={param.dtype})")
            else:
                print("[DETECTOR] Depth Anything V2 cargado (CPU)")
        except Exception as e:
            print(f"[DETECTOR WARN] No se pudo cargar Depth: {e}")
            print("[DETECTOR] Continuando sin profundidad")
            self.depth_model = None
            self.enable_depth = False

    def process(self, frame: np.ndarray) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Procesa frame: detecta objetos + estima profundidad.

        Args:
            frame: Imagen BGR

        Returns:
            (detecciones, depth_map)
        """
        if frame is None:
            return [], None

        detections = []
        depth_map = None
        self._frame_idx += 1

        # Decidir si calcular depth este frame
        should_compute_depth = (
            self.enable_depth and
            self.depth_model and
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

            # Sincronizar
            torch.cuda.synchronize()
        else:
            # CPU: secuencial
            yolo_results = self._run_yolo(frame)
            if should_compute_depth:
                self._cached_depth = self._run_depth(frame)

        # Usar depth cacheado
        depth_map = self._cached_depth

        # Combinar YOLO + Depth para crear detecciones con distancia
        if yolo_results:
            detections = self._create_detections(yolo_results, depth_map, frame.shape)

        return detections, depth_map

    def _run_yolo(self, frame: np.ndarray):
        """Ejecuta YOLO."""
        if self.yolo is None:
            return None
        try:
            results = self.yolo(frame, verbose=False)
            return results[0] if results else None
        except Exception as e:
            print(f"[DETECTOR ERROR] YOLO: {e}")
            return None

    def _run_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Ejecuta Depth Anything con PyTorch FP16."""
        if self.depth_model is None:
            return None
        try:
            h, w = frame.shape[:2]

            # Preprocesar
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (384, 384))

            # PyTorch FP16
            from PIL import Image
            pil_img = Image.fromarray(rgb_small)
            inputs = self.depth_processor(images=pil_img, return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.cuda().half() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                depth_np = outputs.predicted_depth.squeeze().float().cpu().numpy()

            # Resize a tamaño original
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

        Args:
            eye_frame: Imagen de ambos ojos (240x640)

        Returns:
            (x, y) normalizado (0-1, 0-1) o None
        """
        if eye_frame is None:
            return None

        try:
            # Convertir a grayscale si es necesario
            if len(eye_frame.shape) == 3:
                gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = eye_frame

            # Dividir en ojo izquierdo y derecho (imagen tiene ambos)
            h, w = gray.shape
            left_eye = gray[:, :w//2]
            right_eye = gray[:, w//2:]

            # Detectar pupila en cada ojo (zona más oscura)
            left_pos = self._find_pupil(left_eye)
            right_pos = self._find_pupil(right_eye)

            if left_pos and right_pos:
                # Promediar posiciones
                gaze_x = (left_pos[0] + right_pos[0] + 0.5) / 2  # +0.5 para right eye offset
                gaze_y = (left_pos[1] + right_pos[1]) / 2
                return (gaze_x, gaze_y)

            return None
        except Exception as e:
            print(f"[DETECTOR ERROR] Gaze: {e}")
            return None

    def _find_pupil(self, eye_region: np.ndarray) -> Optional[Tuple[float, float]]:
        """Encuentra pupila en región del ojo."""
        h, w = eye_region.shape

        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(eye_region, (7, 7), 0)

        # Encontrar el punto más oscuro (pupila)
        min_val, _, min_loc, _ = cv2.minMaxLoc(blurred)

        # Verificar que sea suficientemente oscuro
        if min_val < 100:  # Threshold para pupila
            x, y = min_loc
            return (x / w, y / h)  # Normalizado

        return None
