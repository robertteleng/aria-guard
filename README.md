# ARIA Demo

Demo de asistencia visual con detección de objetos en tiempo real + estimación de profundidad + eye tracking + tracking temporal + alertas inteligentes.

Soporta múltiples fuentes de entrada:
- **Meta Aria Glasses** - RGB + Eye Tracking + Gaze (x86_64)
- **Intel RealSense D435** - RGB + Depth por hardware (x86_64 + ARM64/Jetson)
- **Webcam/Video** - RGB + Depth por IA

## Características

- **YOLO26s** - Detección de objetos (TensorRT FP16)
- **Depth Anything V2** - Estimación de profundidad monocular (TensorRT FP16)
- **NVDEC** - Decodificación de video por hardware (OpenCV 4.13.0 + Video Codec SDK 13.0)
- **Meta Eye Gaze** - Modelo oficial de Meta para estimación de mirada
- **SimpleTracker** - Tracking de objetos entre frames con IoU matching
- **AlertDecisionEngine** - Sistema de decisión de alertas con priorización
- **NeMo TTS** - Síntesis de voz en proceso separado (aislamiento CUDA)
- **Audio Espacial** - Beeps direccionales (izq/centro/der) según posición
- **Gaze-Aware Alerts** - Alertas solo para objetos no vistos por el usuario

## Arquitectura General

```mermaid
graph TB
    subgraph Input["Entrada"]
        ARIA[Meta Aria Glasses]
        VRS[VRS Dataset]
        WEB[Webcam/Video]
        RS[Intel RealSense D435]
    end

    subgraph Observer["Observer Layer"]
        OBS[MockObserver<br/>Webcam/Video]
        AOBS[AriaDemoObserver<br/>Aria Glasses]
        DOBS[AriaDatasetObserver<br/>VRS + Gaze CSV]
        ROBS[RealSenseObserver<br/>RGB + HW Depth]
    end

    subgraph Detection["Detection Layer (GPU)"]
        DET[ParallelDetector]
        subgraph Models["CUDA Streams"]
            YOLO[YOLO26s<br/>TensorRT FP16]
            DEPTH[Depth Anything V2<br/>TensorRT FP16]
            GAZE[Meta Eye Gaze]
        end
    end

    subgraph Tracking["Tracking Layer"]
        TRK[SimpleTracker<br/>IoU Matching]
        PRI[Priority Calculation<br/>type × distance × approaching × gaze]
    end

    subgraph Decision["Decision Layer"]
        ADE[AlertDecisionEngine]
        VEH[Vehicle Priority<br/>car, bus, truck, bike]
        OTH[Non-Vehicle<br/>person, chair, etc]
    end

    subgraph Audio["Audio Layer (Separate Process)"]
        AUD[AudioFeedback]
        TTS[TTSProcess<br/>NeMo FastPitch+HiFiGAN]
        BEEP[Spatial Beeps<br/>Stereo Panning]
    end

    subgraph Output["Output Layer"]
        DASH[Dashboard<br/>Visual Rendering]
        SRV[Flask Server<br/>MJPEG Streaming]
    end

    ARIA --> AOBS
    VRS --> DOBS
    WEB --> OBS
    RS --> ROBS

    OBS --> DET
    AOBS --> DET
    DOBS --> DET
    ROBS --> DET

    DET --> YOLO
    DET --> DEPTH
    DET --> GAZE

    YOLO --> TRK
    DEPTH --> TRK
    GAZE --> TRK

    TRK --> PRI
    PRI --> ADE

    ADE --> VEH
    ADE --> OTH

    VEH --> AUD
    OTH --> AUD

    AUD --> TTS
    AUD --> BEEP

    TRK --> DASH
    DASH --> SRV
```

## Pipeline de Procesamiento

```mermaid
sequenceDiagram
    participant O as Observer
    participant D as Detector
    participant T as Tracker
    participant E as AlertEngine
    participant A as Audio
    participant TTS as TTS Process

    loop Every Frame (~15-30 FPS)
        O->>D: RGB Frame + Eye Frame

        par CUDA Parallel
            D->>D: YOLO Detection
            D->>D: Depth Estimation
            D->>D: Gaze Estimation
        end

        D->>D: Combine Detections + Depth + Gaze
        D->>T: List[Detection]

        T->>T: IoU Match with existing tracks
        T->>T: Update depth history
        T->>T: Calculate approach speed
        T->>T: Calculate priority scores

        T->>E: TrackedObjects (sorted by priority)

        E->>E: Get top vehicle
        E->>E: Get top non-vehicle
        E->>E: Check cooldowns

        alt Vehicle close OR approaching
            E->>A: Alert vehicle
            A->>TTS: "car left" (async)
            A->>A: Spatial beep
        else No vehicle, person close
            E->>A: Alert person
            A->>TTS: "person right" (async)
            A->>A: Spatial beep
        end
    end
```

## Sistema de Tracking

### SimpleTracker

Tracking de objetos entre frames usando IoU (Intersection over Union):

```mermaid
graph LR
    subgraph Frame_N["Frame N"]
        D1[Detection 1<br/>person @ 100,200]
        D2[Detection 2<br/>car @ 300,150]
    end

    subgraph Tracker["SimpleTracker"]
        IOU[IoU Matching<br/>threshold=0.3]
        HIST[Depth History<br/>deque maxlen=10]
        SPEED[Approach Speed<br/>linear regression]
    end

    subgraph Frame_N1["Frame N+1"]
        T1[Track 1<br/>person ID=0<br/>frames_seen=5]
        T2[Track 2<br/>car ID=1<br/>is_approaching=true]
    end

    D1 --> IOU
    D2 --> IOU
    IOU --> HIST
    HIST --> SPEED
    SPEED --> T1
    SPEED --> T2
```

### TrackedObject

```python
@dataclass
class TrackedObject:
    id: int                    # Unique track ID
    name: str                  # Object class (person, car, etc)
    bbox: Tuple[int,int,int,int]  # x, y, w, h
    zone: str                  # left, center, right
    distance: str              # very_close, close, medium, far
    depth_value: float         # Normalized depth (0-1)
    confidence: float          # Detection confidence
    is_gazed: bool             # User looking at object?

    # Tracking state
    depth_history: deque       # Last 10 depth values
    frames_seen: int           # Consecutive frames tracked
    frames_missing: int        # Frames since last detection

    # Computed
    is_approaching: bool       # Depth increasing = approaching
    approach_speed: float      # Rate of approach
    priority: float            # Alert priority score
```

### Detección de Aproximación

```mermaid
graph TB
    subgraph DepthHistory["Depth History (últimos 10 frames)"]
        F1[Frame 1: 0.3]
        F2[Frame 2: 0.32]
        F3[Frame 3: 0.35]
        F4[Frame 4: 0.38]
        F5[Frame 5: 0.42]
    end

    subgraph Analysis["Análisis"]
        REG[Linear Regression<br/>slope = polyfit]
        THR[Threshold<br/>slope > 0.01]
    end

    subgraph Result["Resultado"]
        APP[is_approaching = True<br/>approach_speed = 0.03]
    end

    F1 --> REG
    F2 --> REG
    F3 --> REG
    F4 --> REG
    F5 --> REG
    REG --> THR
    THR --> APP

    style APP fill:#f96
```

**Nota**: Depth Anything V2 usa profundidad inversa (mayor valor = más cerca). Si `depth_value` aumenta entre frames, el objeto se acerca.

### Cálculo de Prioridad

```python
priority = type_priority × distance_mult × approach_mult × gaze_mult
```

```mermaid
graph LR
    subgraph TypePriority["Tipo de Objeto"]
        CAR[car/truck/bus: 10]
        MOTO[motorcycle: 9]
        BIKE[bicycle: 8]
        PERSON[person: 6]
        DOG[dog: 5]
        CHAIR[chair: 3]
    end

    subgraph DistanceMult["Distancia ×"]
        VC[very_close: 4.0]
        CL[close: 2.0]
        MD[medium: 1.0]
        FR[far: 0.5]
    end

    subgraph ApproachMult["Aproximación ×"]
        YES[approaching: 2.0]
        NO[static: 1.0]
    end

    subgraph GazeMult["Gaze ×"]
        NL[not looking: 1.5]
        LK[looking: 1.0]
    end

    TypePriority --> DistanceMult
    DistanceMult --> ApproachMult
    ApproachMult --> GazeMult
```

**Ejemplo**: Coche acercándose, no visto:

```
10 (car) × 2.0 (close) × 2.0 (approaching) × 1.5 (not gazed) = 60
```

**Ejemplo**: Persona muy cerca, vista:

```
6 (person) × 4.0 (very_close) × 1.0 (static) × 1.0 (gazed) = 24
```

## Sistema de Decisión de Alertas

### AlertDecisionEngine

Centraliza toda la lógica de alertas:

```mermaid
flowchart TD
    START[Tracked Objects] --> VEH{Top Vehicle?}

    VEH -->|Yes| VCLOSE{Close OR<br/>Approaching?}
    VEH -->|No| OTHER

    VCLOSE -->|Yes| VGAZE{User Looking?}
    VCLOSE -->|No| OTHER

    VGAZE -->|No| VCOOL{Cooldown OK?}
    VGAZE -->|Yes| OTHER

    VCOOL -->|Yes| VALERT[ALERT VEHICLE]
    VCOOL -->|No| OTHER

    OTHER{Top Non-Vehicle?} -->|Yes| OCLOSE{Close?}
    OTHER -->|No| NONE[No Alert]

    OCLOSE -->|Yes| OGAZE{User Looking?}
    OCLOSE -->|No| NONE

    OGAZE -->|No| OCOOL{Cooldown OK?}
    OGAZE -->|Yes| NONE

    OCOOL -->|Yes| OALERT[ALERT OTHER]
    OCOOL -->|No| NONE

    style VALERT fill:#f66
    style OALERT fill:#fa0
    style NONE fill:#6f6
```

### Priorización Vehículo vs No-Vehículo

**Problema resuelto**: Un coche a distancia media era ignorado porque personas muy cercanas tenían más prioridad numérica.

**Solución**: Los vehículos tienen **prioridad absoluta** sobre no-vehículos:

```mermaid
graph TB
    subgraph Scene["Escena"]
        P1[Person 1<br/>very_close<br/>priority=24]
        P2[Person 2<br/>close<br/>priority=12]
        CAR[Car<br/>medium, approaching<br/>priority=20]
    end

    subgraph OldLogic["Lógica Antigua"]
        OLD[Top = Person 1<br/>Alerta: person]
    end

    subgraph NewLogic["Lógica Nueva"]
        NEW1[Top Vehicle = Car<br/>approaching = true]
        NEW2[Alerta: car]
    end

    Scene --> OldLogic
    Scene --> NewLogic

    style OLD fill:#f66
    style NEW2 fill:#6f6
```

### Cooldowns

```python
vehicle_cooldown = 1.5s      # Entre alertas de vehículos
other_cooldown = 2.0s        # Entre alertas de no-vehículos
same_object_cooldown = 3.0s  # Antes de re-alertar mismo objeto
```

## Sistema de Audio

### Aislamiento CUDA: Process Isolation Pattern

**Problema**: Aria SDK (FastDDS) y CUDA (PyTorch/TensorRT) **no pueden coexistir** en el mismo proceso - causa "double free or corruption" y segfaults.

**Solución** (patrón de aria-nav): Ocultar CUDA del proceso principal y ejecutar modelos en procesos separados:

```
Main Process (NO CUDA)        DetectorProcess (spawn)       TTSProcess (spawn)
├─ Aria SDK (FastDDS)         ├─ YOLO TensorRT             ├─ NeMo TTS
├─ Flask server               ├─ Depth Anything V2         └─ Audio playback
├─ Dashboard rendering        └─ Eye Gaze model
└─ AudioFeedback wrapper
```

**Implementación**:

```python
# run.py - CRÍTICO: Ocultar CUDA ANTES de cualquier import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Ocultar GPU del proceso principal
os.environ["NUMBA_DISABLE_CUDA"] = "1"    # Desactivar numba CUDA

if __name__ == '__main__':
    import multiprocessing as mp  # NO torch.multiprocessing (importa torch)
    mp.set_start_method('spawn', force=True)

    # Ahora seguro importar módulos
    from src.web.main import app, process_loop
```

```python
# detector_process.py - Worker restaura CUDA
def _detector_worker(input_queue, output_queue, ...):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Restaurar CUDA
    os.environ.pop("NUMBA_DISABLE_CUDA", None)

    # AHORA importar torch (con CUDA visible)
    from src.core.detector import ParallelDetector
    detector = ParallelDetector(...)
```

**Por qué funciona**:

- `spawn` crea procesos hijos desde cero (sin heredar estado)
- El proceso principal nunca inicializa CUDA (invisible)
- Cada worker restaura CUDA antes de importar torch
- Aria SDK y CUDA nunca se encuentran en el mismo proceso

**Patrón heredado de aria-nav** donde se descubrió este conflicto.

### Arquitectura Multi-Proceso (Aislamiento CUDA)

**Problema resuelto**: CUDA y Aria SDK (FastDDS) crasheaban con "double free or corruption".

**Solución**: Tres procesos separados con contextos CUDA aislados:

```mermaid
graph TB
    subgraph MainProcess["Proceso Principal (NO CUDA)"]
        ARIA[Aria SDK<br/>FastDDS]
        FLASK[Flask Server<br/>MJPEG Streaming]
        DASH[Dashboard<br/>Rendering]
        AUDIO[AudioFeedback<br/>Wrapper]
    end

    subgraph DetectorProcess["DetectorProcess (spawn)"]
        YOLO[YOLO TensorRT<br/>CUDA]
        DEPTH[Depth Anything V2<br/>CUDA FP16]
        GAZE[Eye Gaze Model<br/>CUDA]
    end

    subgraph TTSProcess["TTSProcess (spawn)"]
        NEMO[NeMo TTS<br/>FastPitch + HiFiGAN]
        CACHE[Audio Cache<br/>30 phrases]
    end

    ARIA -->|frames| DetectorProcess
    DetectorProcess -->|detections, depth| DASH
    AUDIO -->|text| TTSProcess
    DASH --> FLASK

    style MainProcess fill:#e8f4ea
    style DetectorProcess fill:#f4e8ea
    style TTSProcess fill:#e8e4f4
```

### Pre-caching de Frases

Para latencia mínima, las frases comunes se pre-generan al iniciar:

```python
PRECACHE_PHRASES = [
    "person left", "person right", "person straight",
    "car left", "car right", "car straight",
    "bicycle left", "bicycle right", "bicycle straight",
    "motorcycle left", "motorcycle right", "motorcycle straight",
    "bus left", "bus right", "bus straight",
    "truck left", "truck right", "truck straight",
    # ... 30 frases total
]
```

**Latencia**:

- Frase cacheada: **<10ms** (solo playback)
- Frase nueva: **~200ms** (generación + playback)

### Skip de Mensajes Antiguos

Si hay mensajes acumulados en la cola, solo se reproduce el **más reciente**:

```python
# En _tts_worker:
while not queue.empty():
    newer_msg = queue.get_nowait()
    msg = newer_msg  # Usar el más reciente
```

### Beeps Espaciales

```mermaid
graph LR
    subgraph Distance["Volumen por Distancia"]
        VC[very_close<br/>100%]
        CL[close<br/>70%]
        MD[medium<br/>45%]
        FR[far<br/>25%]
    end

    subgraph Frequency["Frecuencia"]
        CRIT[Critical<br/>1000 Hz]
        NORM[Normal<br/>500 Hz]
    end

    subgraph Panning["Stereo Panning"]
        LEFT[Left Zone<br/>L:100% R:20%]
        CENTER[Center Zone<br/>L:100% R:100%]
        RIGHT[Right Zone<br/>L:20% R:100%]
    end

    VC --> CRIT
    CL --> CRIT
    MD --> NORM
    FR --> NORM
```

## Fuentes de Entrada

| Fuente | RGB | Depth | Gaze | Eye Tracking | Plataforma |
|--------|-----|-------|------|--------------|------------|
| Meta Aria Glasses | ✓ | IA (Depth Anything) | ✓ | ✓ | x86_64 |
| Intel RealSense D435 | ✓ | Hardware (instantáneo) | ✗ | ✗ | x86_64 + ARM64 |
| Webcam | ✓ | IA (Depth Anything) | ✗ | ✗ | Todas |
| Video/VRS | ✓ | IA (Depth Anything) | Precomputed | Precomputed | Todas |

**Ventajas RealSense D435:**
- Depth por hardware = ~0.8GB menos VRAM
- ~30% más rápido (sin modelo de depth IA)
- Funciona en Jetson Orin Nano (ARM64)

## Estructura del Proyecto

```
aria-demo/
├── run.py                      # Entry point
├── Dockerfile                  # Docker básico (desarrollo)
├── Dockerfile.tensorrt         # OpenCV CUDA + TensorRT (producción x86_64)
├── Dockerfile.jetson           # Jetson Orin Nano + RealSense (ARM64)
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── observer.py         # Frame capture (Aria/Webcam/VRS/RealSense)
│   │   ├── detector.py         # YOLO + Depth + Gaze (CUDA)
│   │   ├── detector_process.py # CUDA en proceso separado
│   │   ├── tracker.py          # SimpleTracker + TrackedObject
│   │   ├── alert_engine.py     # AlertDecisionEngine
│   │   ├── audio.py            # AudioFeedback + Beeps
│   │   ├── tts_process.py      # NeMo en proceso separado
│   │   └── dashboard.py        # Visual rendering
│   └── web/
│       ├── main.py             # Flask + MJPEG streaming
│       └── templates/
│           └── index.html
├── scripts/
│   └── export_depth_tensorrt.py  # Exportar Depth Anything a TensorRT
├── data/
│   └── aria_sample/            # VRS recordings + gaze CSV
├── models/                     # YOLO weights (.pt, .engine)
├── docs/
│   └── DOCKER.md               # Documentación Docker completa
└── requirements.txt
```

## Instalación via Docker (Recomendada)

> **📖 Documentación completa**: [docs/DOCKER.md](docs/DOCKER.md) - Arquitectura de imágenes, workflow de desarrollo, troubleshooting, diagramas detallados.

Esta es la **forma más segura** de ejecutar el proyecto, ya que aísla todas las dependencias y evita conflictos de librerías del sistema (como `glibc` vs Aria SDK).

```mermaid
flowchart LR
    subgraph "Quick Start"
        A[./docker-build.sh all] --> B[Primera vez ~25 min]
        C[./docker-build.sh dev] --> D[Desarrollo 0 min]
        E[./docker-build.sh app] --> F[Deps ~3 min]
    end
```

### Requisitos Previos

1. **Drivers NVIDIA** instalados en el sistema host.
2. **Docker Desktop** (Linux/Windows) o **Docker Engine**.
3. **NVIDIA Container Toolkit** (Crítico en Linux):
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

### Ejecución Rápida

Clona el proyecto y simplemente ejecuta:

```bash
# Construye la imagen y levanta el contenedor con GPU
docker compose up --build
```

El sistema descargará automáticamente la imagen base de Ubuntu 22.04, instalará las versiones correctas de Python, PyTorch y Aria SDK, compilará todo y lanzará la aplicación.

Si necesitas lanzar opciones personalizadas (como un video específico), entra en el contenedor:

```bash
docker exec -it aria-demo bash
# Dentro:
python run.py video.mp4
```

---

## Instalación Manual (Legacy)

```bash
cd aria-demo
python -m venv .venv
source .venv/bin/activate

# Dependencias base
pip install -r requirements.txt

# NeMo TTS (requiere CUDA)
pip install nemo_toolkit[tts]

# Meta Eye Gaze (opcional)
pip install projectaria-tools
pip install git+https://github.com/facebookresearch/projectaria_eyetracking.git

# Audio en Linux
sudo apt-get install -y libportaudio2 portaudio19-dev espeak-ng
```

## Uso

```bash
source .venv/bin/activate

# Desarrollo (sin gafas)
python run.py webcam           # Webcam (depth por IA)
python run.py video.mp4        # Video file
python run.py dataset          # VRS sample (data/aria_sample/)

# Intel RealSense D435 (RGB + depth por hardware)
python run.py realsense        # Depth instantáneo, sin modelo IA

# Gafas Aria reales (x86_64 only)
python run.py aria             # USB (por defecto)
python run.py aria:usb         # USB explícito
python run.py aria:wifi        # WiFi (IP por defecto: <ARIA_IP>)
python run.py aria:wifi:<ARIA_IP>  # WiFi con IP específica
```

Selecciona modo de detección:

- **[1] Indoor** - persona, silla, sofá, mesa, tv...
- **[2] Outdoor** - persona, coche, bici, moto, bus...
- **[3] All** - 80 clases COCO

Abre http://localhost:5000

## Conexión Meta Aria Glasses

### Requisitos

```bash
pip install projectaria-client-sdk projectaria-tools
```

### Streaming Profiles

| Interfaz | Profile   | FPS | Notas                       |
| -------- | --------- | --- | --------------------------- |
| USB      | profile28 | 30  | Recomendado, más estable    |
| WiFi     | profile18 | 30  | Requiere IP del dispositivo |

### Cámaras Disponibles

| Cámara            | Resolución | Stream         |
| ----------------- | ---------- | -------------- |
| RGB (centro)      | 1408×1408  | Siempre activo |
| Eye Track         | -          | Siempre activo |
| SLAM1 (izquierda) | 640×480    | Opcional       |
| SLAM2 (derecha)   | 640×480    | Opcional       |

### Troubleshooting

**"Connection refused"**: Verifica que Aria esté en modo streaming

```bash
# En el móvil: Aria App → Streaming → Start
```

**WiFi lento**: Usa USB si es posible, más estable y menor latencia

**"No se pudo obtener calibraciones"**: Normal si las gafas no están en modo correcto, continúa funcionando

## Rendimiento

Probado en RTX 5060 Ti (Blackwell):

| Configuración | FPS | CPU |
|---------------|-----|-----|
| YOLO + Depth TensorRT + Gaze | **42** | ~100% |
| Solo YOLO TensorRT | ~70 | ~50% |
| Con RealSense (sin Depth IA) | ~70 | ~50% |

### Optimizaciones

- **NVDEC** - Decodificación de video en GPU (OpenCV 4.13.0 + Video Codec SDK 13.0)
- **TensorRT FP16** para YOLO y Depth Anything V2
- **Server throttle** - Limita loop principal a 30 FPS (suficiente para asistencia visual)
- **CUDA Streams** para ejecución paralela
- **NeMo en proceso separado** (evita conflictos CUDA)
- **Pre-caching TTS** para latencia mínima

### GPUs Soportadas

- RTX 20xx (Turing)
- RTX 30xx (Ampere)
- RTX 40xx (Ada Lovelace)
- **RTX 50xx (Blackwell)** - Requiere CUDA 12.8+ y Video Codec SDK 13.0

## VRAM Usage

```mermaid
pie title VRAM (~2.5GB total)
    "YOLO26s TensorRT" : 0.4
    "Depth Anything V2 TensorRT" : 0.5
    "Meta Eye Gaze" : 0.2
    "NeMo TTS (proceso separado)" : 1.1
    "OpenCV CUDA buffers" : 0.3
```

## Roadmap

### Completado

- ✅ Conexión Meta Aria Glasses (USB + WiFi)
- ✅ Intel RealSense D435 (RGB + Depth hardware)
- ✅ Sistema de tracking con priorización
- ✅ AlertDecisionEngine para alertas inteligentes
- ✅ NeMo TTS en proceso separado
- ✅ TensorRT FP16 para YOLO y Depth Anything V2
- ✅ NVDEC para decodificación de video en GPU
- ✅ Server throttle 30 FPS para reducir CPU

### Próximo: Jetson + RealSense + IMU

Ver [docs/JETSON.md](docs/JETSON.md) para detalles.

- Dockerfile.jetson para ARM64 (Orin/NX/Nano)
- Intel RealSense D435 (RGB + Depth hardware)
- BNO086 IMU (orientación 9-DOF para alertas direccionales)
- TensorRT engine específico para Jetson GPU
- Modo "lite" single-process (sin multiprocessing)
- Evaluar DeepStream si Python no es suficiente

### Futuro

- Ajuste fino de umbrales de alerta con usuarios reales
- FastVLM para descripciones de escena
- Control por voz (Whisper)
- Detección de semáforos y señales

## Créditos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Meta Project Aria](https://www.projectaria.com/)
- [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking)
