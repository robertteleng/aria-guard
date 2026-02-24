# ARIA Guard

> **L1 — Critical Reactive Layer** | Prototipo Python de detección de colisiones en tiempo real.
> La versión C++ de producción se porta a [aria-core](https://github.com/aria-core).

**Phase 1 ✅ (12 milestones)** · **Phase 2 ⏳ (advanced detection)**

Detección de objetos en tiempo real + estimación de profundidad + eye tracking + tracking temporal + alertas inteligentes.

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
aria-guard/
├── run.py                      # Entry point
├── docker/
│   ├── Dockerfile              # Docker básico (desarrollo)
│   ├── Dockerfile.base         # OpenCV+CUDA+NVDEC (base image)
│   ├── Dockerfile.app          # App sobre base (builds rápidos)
│   ├── Dockerfile.tensorrt     # Todo-en-uno (legacy)
│   ├── Dockerfile.jetson       # Jetson Orin Nano + RealSense (ARM64)
│   ├── docker-build.sh         # Helper script (auto-detecta GPU)
│   ├── docker-compose.yml      # Compose x86_64
│   └── docker-compose.jetson.yml
├── tests/                      # Test scripts
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
        A[./docker/docker-build.sh all] --> B[Primera vez ~25 min]
        C[./docker/docker-build.sh dev] --> D[Desarrollo 0 min]
        E[./docker/docker-build.sh app] --> F[Deps ~3 min]
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
docker compose -f docker/docker-compose.yml up --build
```

El sistema descargará automáticamente la imagen base de Ubuntu 22.04, instalará las versiones correctas de Python, PyTorch y Aria SDK, compilará todo y lanzará la aplicación.

Si necesitas lanzar opciones personalizadas (como un video específico), entra en el contenedor:

```bash
docker exec -it aria-guard bash
# Dentro:
python run.py video.mp4
```

---

## Instalación Manual (Legacy)

```bash
cd aria-guard
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

### Metodología de Benchmark

Las mediciones se realizaron dentro del contenedor Docker de producción (`aria-guard:tensorrt`) para garantizar reproducibilidad. El procedimiento:

1. **Carga de frames**: 200 frames de video real (768x432, escena indoor) cargados en memoria RAM antes de medir, eliminando I/O del benchmark.
2. **Warmup**: 10 frames de calentamiento por componente para estabilizar caches de GPU, JIT de TensorRT y memory pools de CUDA.
3. **Medición individual**: Cada componente (YOLO, Depth, Gaze) se mide aislado con `torch.cuda.synchronize()` antes y después de cada frame para capturar la latencia real en GPU (no solo el tiempo de enqueue).
4. **Medición de pipeline**: El pipeline completo (`ParallelDetector`) ejecuta YOLO + Depth + Gaze en CUDA streams paralelos, midiendo el tiempo wall-clock incluyendo sincronización.
5. **Estadísticas**: Se reportan media, mediana, P95 y P99 sobre los 200 frames. Depth se ejecuta cada 3 frames (configurable) para balancear precisión y throughput.
6. **Comparativa de backends**: Cada modelo se mide con TensorRT FP16 (optimizado) y PyTorch/HuggingFace FP16 (baseline) para cuantificar el speedup de TensorRT.

El script de benchmark está en `experiments/benchmark_paper.py` y genera resultados en JSON (`experiments/benchmark_results.json`).

**Entorno**: NVIDIA GeForce RTX 2060 (6 GB VRAM), CUDA 12.8, TensorRT 10.8.0.43, PyTorch 2.10.0, OpenCV 4.13.0 (CUDA).

### Componentes Individuales

| Componente | Backend | FPS | Latencia media | Mediana | P95 |
|---|---|---:|---:|---:|---:|
| YOLO26s (9.5M params, 20.7 GFLOPs) | TensorRT FP16 | **188.2** | 5.31 ms | 4.90 ms | 8.17 ms |
| YOLO26s | PyTorch CUDA FP16 | 95.9 | 10.42 ms | 10.40 ms | 13.55 ms |
| Depth Anything V2-S (518x518) | TensorRT FP16 | **126.8** | 7.88 ms | 7.80 ms | 9.61 ms |
| Depth Anything V2-S | HuggingFace FP16 | 71.6 | 13.98 ms | 13.92 ms | 15.16 ms |
| Meta Eye Gaze (640x240) | PyTorch CUDA | **261.0** | 3.83 ms | 3.74 ms | 4.70 ms |

### Pipeline Completo

| Configuración | FPS | Latencia media | Mediana | P95 |
|---|---:|---:|---:|---:|
| YOLO + Depth + Gaze (todo TensorRT) | **66.7** | 14.99 ms | 10.01 ms | 26.32 ms |
| YOLO + Gaze sin Depth (TensorRT) | **108.2** | 9.24 ms | 9.27 ms | 10.36 ms |

> Depth se ejecuta cada 3 frames para mantener >60 FPS en el pipeline completo.

### Speedup TensorRT vs PyTorch

| Componente | Speedup |
|---|---:|
| YOLO26s | **1.96x** (188 vs 96 FPS) |
| Depth Anything V2-S | **1.77x** (127 vs 72 FPS) |

### Tamaños de Modelo

| Modelo | Formato | Tamaño |
|---|---|---:|
| YOLO26s | TensorRT FP16 (.engine) | 22.9 MB |
| YOLO26s | PyTorch (.pt) | 20.4 MB |
| Depth Anything V2-S | TensorRT FP16 (.engine) | 53.7 MB |
| Depth Anything V2-S | ONNX | 99.2 MB |

### Detección

- **YOLO26s** con threshold de confianza `conf=0.4` (reduce falsos positivos)
- Los engines de TensorRT son específicos por GPU y versión de TRT
- Si cambia la versión de TensorRT, regenerar: `python scripts/export_tensorrt.py`
- Benchmark reproducible: `python experiments/benchmark_paper.py`

### Optimizaciones

- **NVDEC** - Decodificación de video en GPU (OpenCV 4.13.0 + Video Codec SDK 13.0)
- **TensorRT FP16** para YOLO y Depth Anything V2
- **RealSense hardware depth** - Usa depth nativo (mm), sin modelo IA. Ver [docs/REALSENSE.md](docs/REALSENSE.md)
- **Shared memory IPC** - Zero-copy entre procesos (RGB + depth)
- **CUDA Streams** para ejecución paralela
- **NeMo en proceso separado** (evita conflictos CUDA)
- **Pre-caching TTS** para latencia mínima
- **GPU auto-detection** - `docker-build.sh` detecta compute capability y solo compila para la GPU local

### GPUs Soportadas

- RTX 20xx (Turing)
- RTX 30xx (Ampere)
- RTX 40xx (Ada Lovelace)
- **RTX 50xx (Blackwell)** - Requiere CUDA 12.8+ y Video Codec SDK 13.0

## VRAM Usage

Peak allocated: **1,236 MB** | Reserved: **449 MB**

```mermaid
pie title VRAM (~2.5GB total)
    "YOLO26s TensorRT" : 0.4
    "Depth Anything V2 TensorRT" : 0.5
    "Meta Eye Gaze" : 0.2
    "NeMo TTS (proceso separado)" : 1.1
    "OpenCV CUDA buffers" : 0.3
```

## Project Milestones

> Fuente de verdad: [Notion](https://notion.so) — estas tablas se sincronizan manualmente.

### Phase 1 — MVP ✅

| # | Milestone | Descripción |
|---|-----------|-------------|
| H1 | Project Setup + Docker | Dockerfile base CUDA, GPU auto-detect, multi-GPU (RTX 20xx/30xx/40xx/50xx) |
| H2 | YOLO TensorRT | YOLO26s FP16, 188 FPS, 5.3ms |
| H3 | Depth Estimation TRT | Depth Anything V2-S FP16, 127 FPS, 7.9ms (monocular, para Aria/webcam) |
| H4 | Object Tracking | SimpleTracker IoU matching entre frames |
| H5 | Alert System | AlertDecisionEngine, priorización por riesgo, anti-spam cooldowns, zone filtering |
| H6 | Spatial Audio | Beeps 3D estéreo, 4 zonas distancia (very_close/close/medium/far) + panning L/R |
| H7 | TTS (NeMo) | Proceso separado, aislamiento CUDA, cola de prioridad |
| H8 | Meta Aria Glasses | RGB + gaze-aware filtering (alerta solo objetos no vistos por el usuario) |
| H9 | Intel RealSense D435 | Hardware depth mm, align depth-to-color, distancias absolutas |
| H10 | Shared Memory IPC | Zero-copy RGB + depth entre procesos |
| H11 | NVDEC Video Decode | Hardware video decoding |
| H12 | Benchmarks Paper | benchmark_paper.py reproducible + results JSON (IWINAC 2026) |

### Phase 2 — Advanced Detection ⏳

| # | Milestone | Prioridad | Descripción |
|---|-----------|-----------|-------------|
| H13 | YOLO Fine-tune Navigation | ✅ Done | Clases custom: doors, stairs, curbs, traffic_light, signs |
| H14 | Traffic Light Classification | ✅ Done | HSV sobre crop YOLO: red/yellow/green, canal alerta independiente |
| H15 | Key Sign Detection | ✅ Done | Stop sign: alerta independiente con TTS. Yield/crosswalk requieren re-entrenamiento YOLO |
| H16 | Risk Prioritization v2 | ✅ Done | Approach continuo (1–3x), zone factor (center 1.5x), fast vehicle alert at far (speed>0.03) |
| H17 | ~~Haptic Feedback Prototype~~ | ⚪ Nice to have | Vibración via BLE/serial — requiere hardware custom, bajo ROI vs audio |

> **Nota:** Una vez completada Phase 2 en Python, todo se porta a aria-core (C++) como parte de H22 (Obstacle Avoidance).

### Phase 3 — Threat Model & Audio (basado en papers) ⏳

Rediseño del sistema de alertas basado en evidencia académica (ver [docs/RESEARCH.md](docs/RESEARCH.md)).

| # | Milestone | Prioridad | Descripción |
|---|-----------|-----------|-------------|
| H18 | Collision Risk Score | ✅ | `collision_risk()` 0.0–1.0: TTC (50%) + CBDR bearing (25%) + zone (15%) + class (10%). 4 factores ADAS |
| H19 | Alert Arbiter (2 canales) | ✅ | Canal A: top-1 por risk (DANGER/WARNING/ATTENTION). Canal B: contexto. Rate limit 6/30s, anti-saturación |
| H20 | Audio BRR + Pitch | ✅ | BRR burst 3/2/1 beeps. Pitch 400–1100Hz por distancia. TTS "danger left". Pan mejorado |
| H21 | Benchmark Offline | 🔴 Alta | Script que procesa Tokyo_POV.mp4 sin audio: mide alerts/min, silent ratio, false alerts. Target: >80% silencio, 0 alertas simultáneas |

> **Principio (Gao 2025, Nature):** Al usuario no le importa si es coche o bus. Le importa cuánto peligro hay y de dónde viene.

```mermaid
gantt
    title aria-guard Milestones
    dateFormat YYYY-MM-DD
    axisFormat %b %Y

    section Phase 1 — MVP ✅
    H1  Project Setup + Docker        :done, h1, 2025-01-01, 14d
    H2  YOLO TensorRT                 :done, h2, after h1, 14d
    H3  Depth Estimation TRT          :done, h3, after h2, 14d
    H4  Object Tracking               :done, h4, after h3, 7d
    H5  Alert System                  :done, h5, after h4, 7d
    H6  Spatial Audio                 :done, h6, after h5, 7d
    H7  TTS (NeMo)                    :done, h7, after h6, 7d
    H8  Meta Aria Glasses             :done, h8, after h7, 14d
    H9  Intel RealSense D435          :done, h9, after h8, 14d
    H10 Shared Memory IPC             :done, h10, after h9, 7d
    H11 NVDEC Video Decode            :done, h11, after h10, 7d
    H12 Benchmarks Paper              :done, h12, after h11, 7d

    section Phase 2 — Advanced Detection ✅
    H13 YOLO Fine-tune Navigation     :done, h13, 2026-02-18, 21d
    H14 Traffic Light Classification  :done, h14, after h13, 14d
    H15 Key Sign Detection            :done, h15, after h14, 7d
    H16 Risk Prioritization v2        :done, h16, after h15, 7d
    H17 Haptic Feedback (nice to have) :h17, after h16, 14d

    section Phase 3 — Threat Model & Audio ⏳
    H18 Collision Risk Score          :done, h18, 2026-02-25, 7d
    H19 Alert Arbiter (2 canales)     :done, h19, after h18, 7d
    H20 Audio BRR + Pitch             :done, h20, after h19, 7d
    H21 Benchmark Offline             :h21, after h18, 14d
```

## Créditos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Meta Project Aria](https://www.projectaria.com/)
- [projectaria_eyetracking](https://github.com/facebookresearch/projectaria_eyetracking)
