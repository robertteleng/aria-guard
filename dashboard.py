"""
Dashboard Gradio para ARIA Demo.

UI web simple: RGB + Depth + Eye + Status.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import gradio as gr

from detector import Detection


class Dashboard:
    """Dashboard visual con Gradio."""

    def __init__(self):
        self._colors = {
            "very_close": (0, 0, 255),    # Rojo
            "close": (0, 165, 255),       # Naranja
            "medium": (0, 255, 255),      # Amarillo
            "far": (0, 255, 0),           # Verde
            "unknown": (128, 128, 128)    # Gris
        }

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
            # Placeholder gris
            depth_out = np.full_like(rgb_frame, 50)

        # 3. Eye tracking con anotaciones
        if eye_frame is not None:
            eye_out = eye_frame.copy()
        else:
            # Placeholder
            eye_out = np.full((240, 640, 3), 30, dtype=np.uint8)
            cv2.putText(eye_out, "No Eye Tracking", (200, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # 4. Status text
        status = self._format_status(detections, fps, gaze_point)

        return rgb_out, depth_out, eye_out, status

    def _draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """Dibuja bounding boxes con colores por distancia."""
        for det in detections:
            x, y, w, h = det.bbox
            color = self._colors.get(det.distance, (128, 128, 128))

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label con fondo
            label = f"{det.name} ({det.distance})"
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


def create_gradio_app(process_callback, scan_callback=None):
    """
    Crea aplicación Gradio.

    Args:
        process_callback: Función que procesa un frame y retorna los outputs
        scan_callback: Función para scan de escena (opcional)

    Returns:
        Gradio Blocks app
    """
    with gr.Blocks(title="ARIA Demo") as app:
        gr.Markdown("# ARIA Demo")
        gr.Markdown("Detección de objetos + Profundidad + Eye Tracking")

        with gr.Row():
            with gr.Column(scale=2):
                rgb_output = gr.Image(label="RGB + Detecciones", type="numpy")
            with gr.Column(scale=1):
                depth_output = gr.Image(label="Profundidad", type="numpy")

        with gr.Row():
            with gr.Column(scale=2):
                eye_output = gr.Image(label="Eye Tracking", type="numpy")
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            if scan_callback:
                scan_btn = gr.Button("Scan Scene")

        # Estado de ejecución
        is_running = gr.State(False)

        def toggle_start():
            return True

        def toggle_stop():
            return False

        start_btn.click(toggle_start, outputs=is_running)
        stop_btn.click(toggle_stop, outputs=is_running)

        # Timer para actualizar frames
        timer = gr.Timer(0.1)  # 10 FPS en UI

        def update_display(running):
            if not running:
                return None, None, None, "Parado"
            return process_callback()

        timer.tick(
            update_display,
            inputs=[is_running],
            outputs=[rgb_output, depth_output, eye_output, status_output]
        )

        if scan_callback:
            scan_btn.click(scan_callback)

    return app


# Versión simple sin Gradio (OpenCV fallback)
class SimpleDashboard:
    """Dashboard OpenCV simple (fallback si no hay Gradio)."""

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
