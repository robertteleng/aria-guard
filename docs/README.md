# ARIA Demo

Demo de asistencia visual para gafas Meta Aria: detección de objetos en tiempo real + estimación de profundidad + eye tracking + tracking temporal + alertas inteligentes.

## Características

- **YOLO26s** - Detección de objetos con GPU (TensorRT/PyTorch)
- **Depth Anything V2** - Estimación de profundidad monocular (FP16/torch.compile)
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
    end

    subgraph Observer["Observer Layer"]
        OBS[MockObserver<br/>Webcam/Video]
        AOBS[AriaDemoObserver<br/>Aria Glasses]
        DOBS[AriaDatasetObserver<br/>VRS + Gaze CSV]
    end

    subgraph Detection["Detection Layer (GPU)"]
        DET[ParallelDetector]
        subgraph Models["CUDA Streams"]
            YOLO[YOLO26s<br/>TensorRT]
            DEPTH[Depth Anything V2<br/>FP16 + torch.compile]
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

    OBS --> DET
    AOBS --> DET
    DOBS --> DET

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

### Arquitectura TTS (Aislamiento CUDA)

**Problema resuelto**: NeMo TTS crasheaba con "double free or corruption" al usar CUDA junto con YOLO TensorRT.

**Solución**: NeMo corre en un **proceso separado** con `mp.get_context('spawn')` para aislar contextos CUDA:

```mermaid
graph TB
    subgraph MainProcess["Proceso Principal"]
        DET[Detector]
        YOLO[YOLO TensorRT<br/>CUDA Context A]
        DEPTH[Depth FP16<br/>CUDA Context A]
        AUDIO[AudioFeedback]
        QUEUE[Queue<br/>multiprocessing]
    end

    subgraph TTSProcess["Proceso TTS (spawn)"]
        TTS[TTSProcess]
        NEMO[NeMo<br/>FastPitch + HiFiGAN]
        CUDA_B[CUDA Context B]
        CACHE[Audio Cache<br/>30 phrases]
    end

    DET --> YOLO
    DET --> DEPTH
    AUDIO -->|text| QUEUE
    QUEUE -->|text| TTS
    TTS --> NEMO
    NEMO --> CUDA_B
    TTS --> CACHE

    style MainProcess fill:#e8f4ea
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

## Estructura del Proyecto

```
aria-demo/
├── run.py                      # Entry point
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── observer.py         # Frame capture (Aria/Webcam/VRS)
│   │   ├── detector.py         # YOLO + Depth + Gaze (CUDA)
│   │   ├── tracker.py          # SimpleTracker + TrackedObject
│   │   ├── alert_engine.py     # AlertDecisionEngine
│   │   ├── audio.py            # AudioFeedback + Beeps
│   │   ├── tts_process.py      # NeMo en proceso separado
│   │   └── dashboard.py        # Visual rendering
│   └── web/
│       ├── main.py             # Flask + MJPEG streaming
│       └── templates/
│           └── index.html
├── data/
│   └── aria_sample/            # VRS recordings + gaze CSV
├── models/                     # YOLO weights (.pt, .engine)
├── docs/
│   └── README.md
└── requirements.txt
```

## Instalación

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

python run.py webcam      # Webcam
python run.py video.mp4   # Video file
python run.py dataset     # VRS sample
```

Selecciona modo de detección:
- **[1] Indoor** - persona, silla, sofá, mesa, tv...
- **[2] Outdoor** - persona, coche, bici, moto, bus...
- **[3] All** - 80 clases COCO

Abre http://localhost:5000

## Rendimiento

Probado en RTX 3090:

| Configuración | FPS |
|--------------|-----|
| YOLO + Depth + Gaze + Tracking | 15-19 |
| Con TensorRT | ~30 |
| Sin profundidad | ~25 |
| Solo YOLO TensorRT | ~40 |

### Optimizaciones

- **TensorRT** para YOLO (auto-exporta .engine)
- **torch.compile** para Depth Anything V2
- **FP16** para todos los modelos
- **CUDA Streams** para ejecución paralela
- **NeMo en proceso separado** (evita conflictos CUDA)
- **Pre-caching TTS** para latencia mínima

## VRAM Usage

```mermaid
pie title VRAM (~2.5GB total)
    "YOLO26s TensorRT" : 0.4
    "Depth Anything V2" : 0.8
    "Meta Eye Gaze" : 0.2
    "NeMo TTS (proceso separado)" : 1.1
```

## Roadmap

### Próximo
- Conexión Meta Aria Glasses en tiempo real
- Ajuste fino de umbrales de alerta

### Futuro
- FastVLM para descripciones de escena
- Control por voz (Whisper)
- Detección de semáforos y señales

## Créditos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Meta Project Aria](https://www.projectaria.com/)
- [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking)
