# Intel RealSense D435 - Integración

## Hardware

| Especificación | Valor |
|---|---|
| Modelo | Intel RealSense D435 |
| Sensor depth | Stereo IR (proyector + 2 cámaras IR) |
| Sensor color | RGB rolling shutter |
| Rango depth | 0.2m - 10m |
| Resolución | 1280x720 @ 30fps (configurable) |
| Interfaz | USB 3.0 Type-C |
| Firmware | 5.17.0.10 |

## Streams disponibles

| Stream | Tipo de dato | Uso |
|---|---|---|
| **RGB (Color)** | uint8 BGR | Entrada para YOLO, visualización |
| **Depth** | uint16 (mm) | Distancias reales por hardware |
| **Infrared** | uint8 grayscale | Visión nocturna (no usado actualmente) |

## Arquitectura en el pipeline

```
RealSense D435
    ├── RGB frame ──────────> DetectorProcess (YOLO TensorRT)
    ├── Depth raw (mm) ─────> DetectorProcess (distancias reales)
    └── Depth visual (uint8) > Dashboard (visualización)
```

### Ventajas vs Depth Anything (IA)

- **Distancias absolutas**: RealSense da milímetros reales, no valores relativos
- **Sin carga GPU**: El depth es por hardware, la GPU solo ejecuta YOLO
- **Mayor FPS**: Al no calcular depth por IA, el detector es más rápido
- **Funciona en oscuridad**: El proyector IR permite depth sin luz visible

### Limitaciones

- Rango máximo ~10m (IA no tiene límite teórico)
- Superficies reflectantes/transparentes pueden dar lecturas incorrectas
- El depth tiene "agujeros" (píxeles sin dato, valor 0)

## Alineación (Alignment)

El sensor RGB y el sensor depth están en posiciones físicas distintas. Para que el píxel
(x,y) del color corresponda al mismo punto 3D que el píxel (x,y) del depth, se usa
`rs.align(rs.stream.color)`.

Esto se hace automáticamente en `RealSenseObserver._capture_loop()`.

## Clasificación de distancias

Con RealSense se usan distancias absolutas:

| Categoría | Rango | Uso |
|---|---|---|
| `very_close` | < 0.8m | Alerta inmediata |
| `close` | 0.8m - 2m | Alerta normal |
| `medium` | 2m - 4m | Informativo |
| `far` | > 4m | Baja prioridad |

Con Depth Anything (IA) se usan valores relativos normalizados (0-1).

## Instalación

### Host (Ubuntu)

```bash
# SDK
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev

# Viewer para verificar/actualizar firmware
realsense-viewer

# Python
pip install pyrealsense2
```

**Nota**: `librealsense2-dkms` requiere patches para el kernel. En kernel 6.17+ no
compila y no es necesario (la cámara funciona sin él).

### Docker

Dependencias en `Dockerfile.tensorrt`:

```dockerfile
# Libs nativas
RUN apt-get install -y librealsense2-dev librealsense2-utils libusb-1.0-0

# Python
RUN pip install pyrealsense2
```

El contenedor necesita acceso USB:

```yaml
# docker-compose.yml
devices:
  - /dev/bus/usb:/dev/bus/usb
privileged: true
```

## Ejecución

```bash
# Interactivo (elige opción 4)
docker compose -f docker/docker-compose.yml run --rm --service-ports aria-demo

# Directo
docker compose -f docker/docker-compose.yml run --rm --service-ports aria-demo \
  python run.py realsense indoor
```

## Shared Memory (IPC)

El depth de RealSense se pasa al `DetectorProcess` por shared memory para evitar
serialización:

| Buffer | Tipo | Tamaño (1280x720) |
|---|---|---|
| `aria_frame` | uint8 BGR | 2.76 MB |
| `aria_depth` | uint8 grayscale | 0.92 MB (salida del detector) |
| `aria_hw_depth` | uint16 mm | 1.84 MB (entrada de RealSense) |

## Troubleshooting

### Cámara no detectada

```bash
# Verificar USB
lsusb | grep Intel
# Debería mostrar: Intel Corp. RealSense D435

# Verificar permisos
ls -la /dev/bus/usb/
```

### `libusb-1.0.so.0: cannot open shared object file`

```bash
apt-get install libusb-1.0-0
```

### Depth con muchos agujeros (píxeles en 0)

- Verificar que no hay superficies reflectantes/transparentes frente a la cámara
- Reducir resolución (640x480 es más estable que 1280x720)
- El código usa mediana en vez de media para ser más robusto ante agujeros

### Firmware desactualizado

```bash
realsense-viewer
# Aparecerá notificación de actualización si hay nueva versión
# La cámara se reinicia tras actualizar (~30 seg)
```
