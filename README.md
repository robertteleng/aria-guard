# ARIA Demo

Demo de asistencia visual para gafas Meta Aria: detección de objetos en tiempo real + estimación de profundidad + eye tracking + feedback audio espacial.

## Características

- **YOLO26s** - Detección de objetos con GPU
- **Depth Anything V2** - Estimación de profundidad monocular (FP16)
- **Meta Eye Gaze** - Modelo oficial de Meta para estimación de mirada (`projectaria_eyetracking`)
- **Audio Espacial** - Beeps direccionales (izq/centro/der) + TTS
- **CUDA Streams** - YOLO y Depth ejecutándose en paralelo
- **Gaze-Aware Alerts** - Alertas más intensas para objetos no vistos por el usuario

## Arquitectura

```mermaid
graph TB
    subgraph Input
        ARIA[Meta Aria Glasses]
        WEB[Webcam]
        VID[Video File]
    end

    subgraph Observer
        OBS[observer.py<br/>Frame Capture]
    end

    subgraph Detector
        DET[detector.py]
        subgraph CUDA Streams
            YOLO[YOLO26s<br/>Object Detection]
            DEPTH[Depth Anything V2<br/>Depth Estimation]
        end
        GAZE[Meta Eye Gaze<br/>Gaze Estimation]
    end

    subgraph Output
        DASH[dashboard.py<br/>Visual Rendering]
        AUD[audio.py<br/>Spatial Audio + TTS]
        SRV[server.py<br/>MJPEG Streaming]
    end

    ARIA --> OBS
    WEB --> OBS
    VID --> OBS

    OBS -->|RGB Frame| DET
    OBS -->|Eye Frame| GAZE

    DET --> YOLO
    DET --> DEPTH

    YOLO -->|Detections| DASH
    DEPTH -->|Depth Map| DASH
    GAZE -->|Gaze Point| DASH

    YOLO -->|Danger Objects| AUD
    GAZE -->|User Looking?| AUD
```

## Pipeline de Procesamiento

```mermaid
sequenceDiagram
    participant O as Observer
    participant D as Detector
    participant G as Gaze Model
    participant A as Audio
    participant V as Dashboard

    loop Every Frame
        O->>D: RGB Frame
        O->>G: Eye Frame

        par CUDA Stream 1
            D->>D: YOLO Detection
        and CUDA Stream 2
            D->>D: Depth Estimation
        end

        G->>G: Estimate Gaze (yaw, pitch)

        D->>D: Combine Detections + Depth
        D->>D: Check Gaze on Detections

        alt Danger Object Detected
            D->>A: Alert (object, zone, distance, is_gazed)
            A->>A: Spatial Beep
            alt User Not Looking
                A->>A: Double Beep + TTS Warning
            end
        end

        D->>V: Render Frame
    end
```

## Sistema de Audio

```mermaid
graph LR
    subgraph Distance
        VC[very_close<br/>100% vol]
        CL[close<br/>70% vol]
        MD[medium<br/>45% vol]
        FR[far<br/>25% vol]
    end

    subgraph Frequency
        CRIT[Critical<br/>1000 Hz]
        NORM[Normal<br/>500 Hz]
    end

    subgraph Panning
        LEFT[Left<br/>L:100% R:20%]
        CENTER[Center<br/>L:100% R:100%]
        RIGHT[Right<br/>L:20% R:100%]
    end

    VC --> CRIT
    CL --> CRIT
    MD --> NORM
    FR --> NORM
```

## Estructura del Proyecto

```
aria-demo/
├── demo.py          # Script principal (Gradio/OpenCV UI)
├── server.py        # Servidor MJPEG para streaming web
├── observer.py      # Captura de frames (Aria/Webcam/Video)
├── detector.py      # YOLO + Depth + Gaze (CUDA streams)
├── dashboard.py     # Renderizado visual con OpenCV
├── audio.py         # Feedback auditivo (beeps + TTS)
├── benchmark.py     # Benchmarks de rendimiento
├── requirements.txt # Dependencias
└── README.md
```

## Instalación

```bash
# Clonar/crear entorno
cd aria-demo
python -m venv .venv
source .venv/bin/activate

# Dependencias base
pip install -r requirements.txt

# Meta Eye Gaze Model (opcional, para gafas Aria)
pip install git+https://github.com/facebookresearch/projectaria_eyetracking.git

# Audio en Linux
sudo apt-get install -y libportaudio2 portaudio19-dev espeak-ng
```

## Uso

### Modo Interactivo (Gradio)

```bash
python demo.py                    # Menú interactivo
python demo.py --webcam           # Webcam directa
python demo.py --video video.mp4  # Video
python demo.py --aria             # Gafas Aria
```

### Servidor Web (MJPEG Streaming)

```bash
python server.py test_video.mp4   # Streaming en http://localhost:5000
```

### Opciones

| Opción | Descripción |
|--------|-------------|
| `--aria` | Conectar con gafas Meta Aria |
| `--webcam` | Usar webcam del sistema |
| `--video PATH` | Usar archivo de video |
| `--no-audio` | Desactivar feedback de audio |
| `--no-depth` | Desactivar estimación de profundidad |
| `--opencv` | Usar OpenCV en vez de Gradio |
| `--share` | Compartir Gradio públicamente |

## Controles

- `q` - Salir
- `s` - Scan de escena (anuncia todos los objetos)

## Detecciones

Cada objeto detectado incluye:

| Campo | Descripción |
|-------|-------------|
| `name` | Clase del objeto (person, chair, etc.) |
| `confidence` | Confianza de detección (0.0-1.0) |
| `bbox` | Bounding box (x, y, w, h) |
| `zone` | Zona espacial (left, center, right) |
| `distance` | Categoría de distancia (very_close, close, medium, far) |
| `depth_value` | Valor de profundidad normalizado (0.0-1.0) |
| `is_gazed` | True si el usuario está mirando el objeto |

## Rendimiento

Probado en RTX 3090:
- **15-19 FPS** con YOLO + Depth + Gaze
- **~25 FPS** sin profundidad
- **~30 FPS** solo YOLO

## Dependencias

```
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.0.0
ultralytics>=8.0.0
transformers>=4.35.0
gradio>=4.0.0
sounddevice>=0.4.6
pyttsx3>=2.90
Pillow>=10.0.0
```

## Roadmap: VLM + Control por Voz

Próxima integración: FastVLM para descripciones de escena + comandos de voz.

### Arquitectura Completa (Planned)

```mermaid
graph TB
    subgraph Input["Entrada"]
        CAM[Camera/Video/Aria]
        MIC[Micrófono]
    end

    subgraph RealTime["Real-Time Loop (~20 FPS)"]
        YOLO[YOLO26s]
        DEPTH[Depth Anything V2]
        GAZE[Meta Eye Gaze]
        BEEP[Spatial Beeps]
    end

    subgraph OnDemand["On-Demand (~400ms)"]
        VLM[FastVLM-0.5B]
        TTS[TTS Descripción]
    end

    subgraph Voice["Control de Voz"]
        WHISPER[Whisper/Vosk]
        CMD{Comando}
    end

    CAM --> YOLO
    CAM --> DEPTH
    CAM --> GAZE
    CAM --> VLM

    YOLO --> BEEP
    DEPTH --> BEEP
    GAZE --> BEEP

    MIC --> WHISPER
    WHISPER --> CMD

    CMD -->|"describe"| VLM
    CMD -->|"stop"| BEEP
    CMD -->|"resume"| BEEP

    VLM --> TTS
```

### Flujo de Voz

```mermaid
sequenceDiagram
    participant U as Usuario
    participant W as Whisper
    participant S as Sistema
    participant V as FastVLM
    participant T as TTS

    U->>W: "describe"
    W->>S: Comando reconocido
    S->>V: Frame actual
    V->>V: Genera descripción (~400ms)
    V->>T: "Office space with person at desk..."
    T->>U: Audio descripción

    Note over S: Loop real-time continúa<br/>sin interrupción
```

### Modelos y VRAM

```mermaid
pie title VRAM Usage (~2.7GB total)
    "YOLO26s" : 0.5
    "Depth Anything V2" : 0.8
    "Meta Eye Gaze" : 0.2
    "FastVLM-0.5B" : 1.2
```

### Comandos de Voz (Planned)

| Comando | Acción |
|---------|--------|
| "describe" / "scan" | Descripción VLM detallada |
| "stop" | Pausar alertas de audio |
| "resume" | Reanudar alertas |
| "help" | Listar comandos disponibles |

## Créditos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Meta Project Aria](https://www.projectaria.com/)
- [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking)
- [FastVLM](https://github.com/apple/ml-fastvlm) (planned)
