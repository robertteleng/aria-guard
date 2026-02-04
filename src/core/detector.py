"""
Detector: YOLO + Depth + Eye Gaze con CUDA streams.

YOLO detecta objetos, Depth calcula distancia, Meta model estima gaze.
Soporta TensorRT y OpenCV CUDA para máximo rendimiento.
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path
import math
import os

import cv2
import numpy as np
import torch

from src.core.tracker import SimpleTracker, TrackedObject
from src.core.types import Detection, CLASS_FILTERS

# Optimizar convs para tamaños fijos
torch.backends.cudnn.benchmark = True

# OpenCV CUDA disponible?
_OPENCV_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0

# Paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = _PROJECT_ROOT / "models"


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

        # Tracker para seguimiento temporal
        self.tracker = SimpleTracker()

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
        """Carga Depth Anything V2. Intenta TensorRT, luego FP16."""
        self._depth_tensorrt = False
        self._depth_trt_context = None

        # Try TensorRT first (vits = small model from fabio-sim/Depth-Anything-ONNX)
        if self.device == "cuda":
            engine_path = MODELS_DIR / "depth_anything_v2_vits.engine"
            if engine_path.exists():
                try:
                    self._load_depth_tensorrt(engine_path)
                    return
                except Exception as e:
                    print(f"[DETECTOR WARN] TensorRT depth failed: {e}")

        # Fallback to HuggingFace FP16
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            self.depth_processor = AutoImageProcessor.from_pretrained(model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)

            if self.device == "cuda":
                self.depth_model = self.depth_model.cuda().half().eval()
                print("[DETECTOR] Depth Anything V2 (FP16)")
            else:
                print("[DETECTOR] Depth Anything V2 (CPU)")
        except Exception as e:
            print(f"[DETECTOR WARN] No se pudo cargar Depth: {e}")
            print("[DETECTOR] Continuando sin profundidad")
            self.depth_model = None
            self.enable_depth = False

    def _load_depth_tensorrt(self, engine_path: Path):
        """Carga Depth Anything V2 con TensorRT."""
        import tensorrt as trt

        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(logger)
        self._depth_trt_engine = runtime.deserialize_cuda_engine(engine_data)
        self._depth_trt_context = self._depth_trt_engine.create_execution_context()

        # Allocate buffers
        self._depth_trt_inputs = []
        self._depth_trt_outputs = []
        self._depth_trt_bindings = []

        for i in range(self._depth_trt_engine.num_io_tensors):
            name = self._depth_trt_engine.get_tensor_name(i)
            shape = self._depth_trt_engine.get_tensor_shape(name)
            dtype = trt.nptype(self._depth_trt_engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            # Allocate device memory
            device_mem = torch.empty(size, dtype=torch.float32, device="cuda")
            self._depth_trt_bindings.append(device_mem.data_ptr())

            if self._depth_trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._depth_trt_inputs.append({"name": name, "shape": shape, "mem": device_mem})
            else:
                self._depth_trt_outputs.append({"name": name, "shape": shape, "mem": device_mem})

        self._depth_tensorrt = True
        self.depth_model = True  # Mark as loaded (non-None)
        self.depth_processor = None  # Not needed for TensorRT
        print(f"[DETECTOR] Depth Anything V2 (TensorRT: {engine_path.name})")

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
            import projectaria_eyetracking.inference.infer as infer_module
            import os
            from pathlib import Path

            # Search for weights in multiple locations
            weights_locations = [
                # 1. Local project models directory
                Path(__file__).parent.parent.parent / "models" / "gaze_weights" / "social_eyes_uncertainty_v1",
                # 2. Inside installed pip package
                Path(os.path.dirname(infer_module.__file__)) / "model" / "pretrained_weights" / "social_eyes_uncertainty_v1",
            ]

            weights_path = None
            for loc in weights_locations:
                if (loc / "weights.pth").exists():
                    weights_path = str(loc)
                    break

            if weights_path is None:
                print(f"[DETECTOR WARN] Meta gaze weights not found. Searched:")
                for loc in weights_locations:
                    print(f"    - {loc}")
                return

            self.gaze_model = EyeGazeInference(
                model_checkpoint_path=os.path.join(weights_path, "weights.pth"),
                model_config_path=os.path.join(weights_path, "config.yaml"),
                device=self.device
            )
            print(f"[DETECTOR] Meta Eye Gaze model loaded from {weights_path}")

        except ImportError:
            print("[DETECTOR WARN] projectaria_eyetracking not installed")
        except Exception as e:
            print(f"[DETECTOR WARN] Could not load gaze model: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original torch.load
            torch.load = _original_torch_load

    def process(
        self,
        frame: np.ndarray,
        eye_frame: Optional[np.ndarray] = None,
        hardware_depth: Optional[np.ndarray] = None
    ) -> Tuple[List[Detection], Optional[np.ndarray], Optional[Tuple[float, float]], List[TrackedObject]]:
        """
        Procesa frame: detecta objetos + estima profundidad + gaze + tracking.

        Args:
            frame: Imagen BGR
            eye_frame: Imagen de eye tracking (opcional)
            hardware_depth: Mapa de profundidad de hardware (ej: RealSense D435).
                           Si se proporciona, no se ejecuta el modelo de depth.

        Returns:
            (detecciones, depth_map, gaze_point, tracked_objects)
        """
        if frame is None:
            return [], None, None

        detections = []
        depth_map = None
        gaze_point = None
        self._frame_idx += 1

        # Si hay depth por hardware, usarlo directamente (RealSense D435)
        if hardware_depth is not None:
            self._cached_depth = hardware_depth
            should_compute_depth = False
        else:
            # Decidir si calcular depth este frame con el modelo de IA
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

        # Update tracker and get tracked objects with priority
        tracked_objects = self.tracker.update(detections)

        return detections, depth_map, gaze_point, tracked_objects

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
        """Ejecuta Depth Anything. Usa TensorRT o FP16 según disponibilidad."""
        if self.depth_model is None:
            return None
        try:
            h, w = frame.shape[:2]

            # Preprocesar con OpenCV CUDA si está disponible
            if _OPENCV_CUDA:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                gpu_small = cv2.cuda.resize(gpu_rgb, (518, 518))  # Depth Anything V2 uses 518x518
                rgb_small = gpu_small.download()
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_small = cv2.resize(rgb, (518, 518))

            # TensorRT path
            if self._depth_tensorrt:
                depth_np = self._run_depth_tensorrt(rgb_small)
            else:
                # HuggingFace path
                from PIL import Image
                inputs = self.depth_processor(images=Image.fromarray(rgb_small), return_tensors="pt")

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

    def _run_depth_tensorrt(self, rgb_image: np.ndarray) -> np.ndarray:
        """Ejecuta Depth Anything V2 con TensorRT."""
        import tensorrt as trt

        # Preprocess: normalize to [0, 1] and convert to NCHW
        img = rgb_image.astype(np.float32) / 255.0
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        img = np.ascontiguousarray(img)

        # Copy input to GPU
        input_tensor = torch.from_numpy(img).cuda()
        self._depth_trt_inputs[0]["mem"].copy_(input_tensor.flatten())

        # Set tensor addresses
        for inp in self._depth_trt_inputs:
            self._depth_trt_context.set_tensor_address(inp["name"], inp["mem"].data_ptr())
        for out in self._depth_trt_outputs:
            self._depth_trt_context.set_tensor_address(out["name"], out["mem"].data_ptr())

        # Execute
        self._depth_trt_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Get output
        output_shape = self._depth_trt_outputs[0]["shape"]
        depth = self._depth_trt_outputs[0]["mem"][:np.prod(output_shape)].cpu().numpy()
        depth = depth.reshape(output_shape).squeeze()

        return depth

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
