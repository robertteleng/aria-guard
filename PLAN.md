# Plan: ARIA Demo

## Objetivo
Demo simple: ponerse gafas → detectar objetos → feedback visual (Gradio) + auditivo.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                       demo.py                               │
├─────────────────────────────────────────────────────────────┤
│  Observer (observer.py)                                     │
│    ├─ AriaDemoObserver: RGB + Eye + SLAM (gafas reales)    │
│    └─ MockObserver: Webcam o Video (desarrollo)            │
├─────────────────────────────────────────────────────────────┤
│  Detector (detector.py) - CUDA Streams paralelos           │
│    ├─ YOLO11: Detección de objetos                         │
│    ├─ Depth Anything V2: Mapa de profundidad               │
│    └─ GazeEstimator: Punto de mirada desde eye tracking    │
├─────────────────────────────────────────────────────────────┤
│  Dashboard (dashboard.py) - Gradio Web UI                  │
│    ┌──────────────────┬─────────────┐                      │
│    │   RGB + Boxes    │  Depth Map  │                      │
│    │   + Gaze Point   │ (magma)     │                      │
│    ├──────────────────┼─────────────┤                      │
│    │   Eye Tracking   │  Detections │                      │
│    │   + Pupil        │  + FPS      │                      │
│    └──────────────────┴─────────────┘                      │
│    [Start] [Stop] [Scan Scene]                             │
├─────────────────────────────────────────────────────────────┤
│  Audio (audio.py)                                           │
│    ├─ Beeps espaciales (izq/centro/der)                    │
│    └─ TTS: "Silla a tu derecha, cerca"                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Archivos a Crear

| Archivo | Estado | Descripción |
|---------|--------|-------------|
| `demo.py` | CREADO | Script principal |
| `observer.py` | CREADO | Captura Aria/Webcam/Video |
| `detector.py` | PENDIENTE | YOLO + Depth + Gaze |
| `dashboard.py` | PENDIENTE | UI Gradio |
| `audio.py` | PENDIENTE | Beeps + TTS |

---

## Siguiente: detector.py

```python
class ParallelDetector:
    """YOLO + Depth en paralelo con CUDA streams."""

    def __init__(self, enable_depth=True, device="cuda"):
        # CUDA streams para paralelismo
        self.yolo_stream = torch.cuda.Stream()
        self.depth_stream = torch.cuda.Stream()

        # YOLO (descarga automática)
        self.yolo = YOLO("yolo11n.pt")

        # Depth Anything V2
        if enable_depth:
            self.depth_model = load_depth_anything()

    def process(self, frame):
        # Stream 1: YOLO
        with torch.cuda.stream(self.yolo_stream):
            detections = self.yolo(frame)

        # Stream 2: Depth (paralelo)
        with torch.cuda.stream(self.depth_stream):
            depth_map = self.depth_model(frame)

        # Sincronizar
        torch.cuda.synchronize()
        return detections, depth_map

    def estimate_gaze(self, eye_frame):
        # Detectar pupila con OpenCV
        # Retornar (x, y) normalizado
        pass
```

---

## Siguiente: dashboard.py

```python
import gradio as gr

def create_dashboard():
    with gr.Blocks(title="ARIA Demo") as demo:
        gr.Markdown("# ARIA Demo")

        with gr.Row():
            rgb_output = gr.Image(label="RGB + Detecciones")
            depth_output = gr.Image(label="Profundidad")

        with gr.Row():
            eye_output = gr.Image(label="Eye Tracking")
            status = gr.Textbox(label="Status", lines=5)

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            scan_btn = gr.Button("Scan Scene")

    return demo
```

---

## Siguiente: audio.py

```python
class AudioFeedback:
    """Beeps espaciales + TTS."""

    def __init__(self):
        self.cooldowns = {}  # Evitar spam
        self._setup_tts()

    def announce(self, detections, depth_map):
        # Seleccionar objeto más relevante
        # Emitir beep direccional
        # TTS: "Silla a tu derecha"
        pass

    def scan_scene(self, detections):
        # Anunciar todos los objetos
        pass

    def play_beep(self, direction, distance):
        # Beep espacial (izq/centro/der)
        # Volumen según distancia
        pass
```

---

## Verificación

```bash
cd ~/Projects/aria/aria-demo

# Test 1: Webcam
python demo.py --webcam

# Test 2: Video
python demo.py --video /path/to/video.mp4

# Test 3: Aria (cuando tengas las gafas)
python demo.py --aria
```

**Checklist:**
- [ ] Gradio abre en localhost:7860
- [ ] Webcam muestra video en RGB panel
- [ ] YOLO detecta objetos (bounding boxes)
- [ ] Depth map en pseudocolor
- [ ] Audio: beeps cuando detecta objetos
- [ ] FPS > 15 con CUDA streams

---

## Dependencias

```bash
pip install ultralytics opencv-python torch gradio sounddevice numpy pyttsx3
```

---

## Referencia: aria-nav

El proyecto original `~/Projects/aria/aria-nav/` tiene implementaciones completas que sirven de referencia:

| Componente | Archivo de referencia en aria-nav |
|------------|-----------------------------------|
| Observer SDK | `src/core/observer.py` |
| YOLO Processor | `src/core/vision/yolo_processor.py` |
| Depth Estimator | `src/core/vision/depth_estimator.py` |
| Audio System | `src/core/audio/audio_system.py` |
| TTS + Beeps | `src/core/audio/navigation_audio_router.py` |
| Frame Renderer | `src/presentation/renderers/frame_renderer.py` |
| Config tipada | `src/utils/config_sections.py` |

**Consultar estos archivos cuando haya dudas de implementación.**
