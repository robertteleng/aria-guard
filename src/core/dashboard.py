"""
Dashboard para ARIA Demo.

Renderiza RGB + Depth overlay + Radar + Status.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import Detection


class Dashboard:
    """Dashboard visual."""

    def __init__(self):
        self._colors = {
            "very_close": (0, 0, 255),    # Rojo
            "close": (0, 165, 255),       # Naranja
            "medium": (0, 255, 255),      # Amarillo
            "far": (0, 255, 0),           # Verde
            "unknown": (128, 128, 128)    # Gris
        }
        self._radar_size = 200
        self._depth_size = 200

    def render(
        self,
        rgb_frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        eye_frame: Optional[np.ndarray],
        detections: List[Detection],
        gaze_point: Optional[Tuple[float, float]],
        fps: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Renderiza todos los componentes del dashboard.

        Returns:
            (rgb_annotated, depth_colored, eye_annotated, status_text)
        """
        # 1. RGB con detecciones
        rgb_out = self._draw_detections(rgb_frame.copy(), detections)
        if gaze_point:
            rgb_out = self._draw_gaze(rgb_out, gaze_point)

        # 2. Depth map en pseudocolor
        if depth_map is not None:
            depth_out = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        else:
            depth_out = np.full_like(rgb_frame, 50)

        # 3. Overlay depth (200x200) en esquina superior derecha del RGB
        if depth_map is not None:
            depth_small = cv2.resize(depth_out, (self._depth_size, self._depth_size))
            x_depth = rgb_out.shape[1] - self._depth_size - 10
            rgb_out = self._overlay_image(rgb_out, depth_small, x_depth, 10)

        # 4. Overlay radar en esquina inferior derecha del RGB
        radar = self._draw_radar(detections, rgb_frame.shape[1])
        x_radar = rgb_out.shape[1] - self._radar_size - 10
        y_radar = rgb_out.shape[0] - self._radar_size - 10
        rgb_out = self._overlay_image(rgb_out, radar, x_radar, y_radar)

        # 5. Eye tracking (raw frame, gaze is shown on RGB)
        if eye_frame is not None:
            eye_out = eye_frame.copy()
        else:
            eye_out = np.full((120, 640, 3), 30, dtype=np.uint8)
            cv2.putText(eye_out, "No Eye Tracking", (220, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        # 6. Status text
        status = self._format_status(detections, fps, gaze_point)

        return rgb_out, depth_out, eye_out, status

    def _draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Dibuja bounding boxes con colores por distancia.

        Objects the user is looking at (is_gazed=True) get a filled
        semi-transparent overlay to highlight them.
        """
        for det in detections:
            x, y, w, h = det.bbox
            color = self._colors.get(det.distance, (128, 128, 128))

            # If user is looking at this object, fill with semi-transparent color
            if getattr(det, 'is_gazed', False):
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                # Thicker border for gazed objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            else:
                # Normal bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label con fondo
            label = f"{det.name} ({det.distance})"
            if getattr(det, 'is_gazed', False):
                label = f"[LOOKING] {label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _draw_gaze(
        self, frame: np.ndarray, gaze_point: Tuple[float, float]
    ) -> np.ndarray:
        """Dibuja punto de mirada."""
        h, w = frame.shape[:2]
        gx = int(gaze_point[0] * w)
        gy = int(gaze_point[1] * h)

        # Círculo grande con crosshair
        cv2.circle(frame, (gx, gy), 20, (255, 0, 255), 2)
        cv2.line(frame, (gx - 30, gy), (gx + 30, gy), (255, 0, 255), 1)
        cv2.line(frame, (gx, gy - 30), (gx, gy + 30), (255, 0, 255), 1)

        return frame

    def _format_status(
        self,
        detections: List[Detection],
        fps: float,
        gaze_point: Optional[Tuple[float, float]]
    ) -> str:
        """Formatea texto de status."""
        lines = [
            f"FPS: {fps:.1f}",
            f"Objetos: {len(detections)}",
            ""
        ]

        if detections:
            lines.append("Detectados:")
            for det in detections[:5]:  # Max 5
                lines.append(f"  - {det.name}: {det.zone}, {det.distance}")

        if gaze_point:
            lines.append("")
            lines.append(f"Mirada: ({gaze_point[0]:.2f}, {gaze_point[1]:.2f})")

        return "\n".join(lines)

    def _draw_radar(
        self, detections: List[Detection], frame_width: int
    ) -> np.ndarray:
        """Dibuja radar 2D top-down (semicírculo)."""
        size = self._radar_size
        radar = np.zeros((size, size, 3), dtype=np.uint8)
        center_x = size // 2
        center_y = size - 15  # Usuario en la parte inferior

        # Arcos de referencia (distancia)
        for i, radius in enumerate([size // 4, size // 2, size * 3 // 4 - 10]):
            alpha = 40 + i * 20
            cv2.ellipse(radar, (center_x, center_y), (radius, radius),
                       0, 180, 360, (alpha, alpha, alpha), 1)

        # Líneas de zona (izq/centro/der)
        cv2.line(radar, (size // 3, center_y), (size // 3, 20), (40, 40, 40), 1)
        cv2.line(radar, (2 * size // 3, center_y), (2 * size // 3, 20), (40, 40, 40), 1)

        # Usuario (punto blanco abajo)
        cv2.circle(radar, (center_x, center_y), 6, (255, 255, 255), -1)

        # Objetos detectados
        for det in detections:
            # Posición X: centro del bbox normalizado
            bbox_center_x = (det.bbox[0] + det.bbox[2] / 2) / frame_width
            # Mapear a posición en radar (0=izq, 1=der)
            px = int(bbox_center_x * (size - 20) + 10)

            # Distancia: invertir depth (1=cerca=abajo, 0=lejos=arriba)
            dist_normalized = 1.0 - det.depth_value
            max_radius = size - 30
            py = int(center_y - dist_normalized * max_radius)

            # Color según distancia
            color = self._colors.get(det.distance, (128, 128, 128))

            # Dibujar punto (más grande si is_gazed)
            point_size = 10 if getattr(det, 'is_gazed', False) else 6
            cv2.circle(radar, (px, py), point_size, color, -1)

            # Borde blanco si is_gazed
            if getattr(det, 'is_gazed', False):
                cv2.circle(radar, (px, py), point_size + 2, (255, 255, 255), 2)

        # Etiqueta
        cv2.putText(radar, "RADAR", (size // 2 - 25, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return radar

    def _overlay_image(
        self, base: np.ndarray, overlay: np.ndarray,
        x: int, y: int, alpha: float = 0.85
    ) -> np.ndarray:
        """Superpone una imagen sobre otra con transparencia."""
        h, w = overlay.shape[:2]
        # Asegurar que no se sale del frame
        if x + w > base.shape[1]:
            w = base.shape[1] - x
        if y + h > base.shape[0]:
            h = base.shape[0] - y
        if w <= 0 or h <= 0:
            return base

        overlay_crop = overlay[:h, :w]
        roi = base[y:y+h, x:x+w]
        cv2.addWeighted(overlay_crop, alpha, roi, 1 - alpha, 0, roi)
        return base


class SimpleDashboard:
    """Dashboard OpenCV simple para visualización local."""

    def __init__(self):
        self._dashboard = Dashboard()

    def show(
        self,
        rgb_frame: np.ndarray,
        depth_map: Optional[np.ndarray],
        eye_frame: Optional[np.ndarray],
        detections: List[Detection],
        gaze_point: Optional[Tuple[float, float]],
        fps: float
    ) -> int:
        """
        Muestra dashboard con OpenCV.

        Returns:
            Key presionada (para detectar 'q')
        """
        rgb_out, depth_out, eye_out, status = self._dashboard.render(
            rgb_frame, depth_map, eye_frame, detections, gaze_point, fps
        )

        # Redimensionar para layout
        h_target = 360
        rgb_resized = self._resize_height(rgb_out, h_target)
        depth_resized = self._resize_height(depth_out, h_target)
        eye_resized = self._resize_height(eye_out, h_target // 2)

        # Combinar en layout
        top_row = np.hstack([rgb_resized, depth_resized])
        cv2.imshow("ARIA Demo", top_row)

        return cv2.waitKey(1) & 0xFF

    def _resize_height(self, img: np.ndarray, height: int) -> np.ndarray:
        h, w = img.shape[:2]
        new_w = int(w * height / h)
        return cv2.resize(img, (new_w, height))
