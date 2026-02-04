# Docker Setup para ARIA Demo

## Dockerfiles Disponibles

| Dockerfile | Plataforma | GPU | Uso |
|------------|------------|-----|-----|
| `Dockerfile` | x86_64 | CUDA (pip) | Desarrollo rápido, webcam |
| `Dockerfile.tensorrt` | x86_64 | CUDA + TensorRT + OpenCV CUDA | Producción, máximo rendimiento |
| `Dockerfile.jetson` | ARM64 | Jetson (L4T) | Jetson Orin Nano + RealSense |

## Por qué Docker?

El Aria SDK (FastDDS) tiene incompatibilidades con ciertas versiones de glibc.
En particular, Ubuntu 24.04 con glibc 2.39-0ubuntu8.7 causa crashes al conectar
con las gafas Aria:

```
munmap_chunk(): invalid pointer
free(): invalid size
```

Docker permite usar Ubuntu 22.04 con una versión compatible de glibc.

## Requisitos

- Docker con soporte GPU (nvidia-container-toolkit)
- NVIDIA GPU con compute capability >= 7.5 (RTX 20xx o superior)
- Para Jetson: JetPack 6.x / L4T R36.x

### Instalar nvidia-container-toolkit (si no lo tienes)

```bash
# Añadir repositorio
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configurar Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Dockerfile.tensorrt (Producción - x86_64)

Incluye OpenCV compilado con CUDA para máximo rendimiento.

### Configuración

- **CUDA**: 12.6 + cuDNN 8.9.7
- **OpenCV**: 4.10.0 compilado con CUDA
- **PyTorch**: cu126 + TensorRT
- **GPUs soportadas**: RTX 20xx (7.5), RTX 30xx (8.6), RTX 40xx (8.9), RTX 50xx (12.0)

### Build y uso

```bash
# Build (tarda ~30-60 min por compilación de OpenCV)
docker build -f Dockerfile.tensorrt -t aria-demo:tensorrt .

# Run con Aria glasses
docker run -it --rm --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -p 5000:5000 \
  --gpus all \
  aria-demo:tensorrt

# Run con webcam
docker run -it --rm \
  --device /dev/video0 \
  -p 5000:5000 \
  --gpus all \
  aria-demo:tensorrt python run.py webcam

# Run con RealSense D435
docker run -it --rm \
  -v /dev/bus/usb:/dev/bus/usb \
  -p 5000:5000 \
  --gpus all \
  aria-demo:tensorrt python run.py realsense
```

---

## Dockerfile.jetson (Jetson Orin Nano)

Para dispositivos ARM64 con Jetson. Usa RealSense D435 en lugar de Aria (el SDK de Aria no soporta ARM).

### Configuración

- **Base**: nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
- **Cámara**: Intel RealSense D435 (RGB + Depth por hardware)
- **Compute Capability**: 8.7 (Orin Nano)

### Características RealSense vs Aria

| Característica | Aria Glasses | RealSense D435 |
|----------------|--------------|----------------|
| RGB | ✓ | ✓ |
| Depth | ✓ (modelo IA) | ✓ (hardware) |
| Eye Tracking | ✓ | ✗ |
| Gaze Estimation | ✓ | ✗ |
| Plataforma | x86_64 | x86_64 + ARM64 |

### Build en Jetson

```bash
# Build (ejecutar EN el Jetson)
docker build -f Dockerfile.jetson -t aria-demo:jetson .

# Run con RealSense
docker run -it --rm \
  --runtime nvidia \
  -v /dev/bus/usb:/dev/bus/usb \
  --device /dev/video0 \
  -p 5000:5000 \
  aria-demo:jetson
```

### Ventajas del depth por hardware (RealSense)

- **Sin modelo de IA**: No ejecuta Depth Anything V2, ahorra ~0.8GB VRAM
- **Más rápido**: Depth instantáneo del sensor, ~30% mejora en FPS
- **Ideal para Jetson**: Maximiza recursos limitados del Orin Nano

---

## Dockerfile (Desarrollo rápido)

Para pruebas rápidas sin compilar OpenCV.

```bash
# Build
docker build -t aria-demo .

# Run con compose
docker compose up
```

---

## Fuentes de Video Soportadas

```bash
# Webcam
python run.py webcam

# Aria Glasses (USB)
python run.py aria

# Aria Glasses (WiFi)
python run.py aria:wifi:<ARIA_IP>

# Aria Dataset (VRS)
python run.py dataset
python run.py /path/to/recording.vrs

# Intel RealSense D435
python run.py realsense

# Archivo de video
python run.py /path/to/video.mp4
```

---

## Modos de Detección

```bash
python run.py <source> indoor   # Objetos de interior
python run.py <source> outdoor  # Objetos de exterior
python run.py <source> all      # Todos (80 clases COCO)
```

---

## Exportar Modelos a TensorRT

Para máximo rendimiento, exporta los modelos a TensorRT:

```bash
# Dentro del contenedor
python scripts/export_depth_tensorrt.py   # Depth Anything V2

# YOLO se exporta automáticamente la primera vez
```

---

## Troubleshooting

### Puerto 5000 ocupado

```bash
lsof -i :5000
docker compose down
docker rm -f aria-demo
```

### No detecta las gafas Aria

```bash
lsusb | grep -i aria
docker compose exec aria-demo lsusb
```

### Error de GPU / CUDA no disponible

```bash
# Verificar GPU
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi

# Verificar PyTorch
docker run --rm --gpus all aria-demo:tensorrt python -c "import torch; print(torch.cuda.is_available())"
```

### OpenCV CUDA no funciona

```bash
# Verificar OpenCV CUDA
docker run --rm --gpus all aria-demo:tensorrt python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### RealSense no detecta cámara

```bash
# Verificar USB
lsusb | grep -i intel

# Permisos
sudo usermod -a -G video $USER
```

### TTS tarda mucho en cargar

El modelo NeMo TTS tarda ~60s la primera vez. Para desarrollo:

```bash
python run.py webcam outdoor --no-tts
```

---

## Transferir imagen a otro equipo

```bash
# Guardar
docker save aria-demo:tensorrt | gzip > aria-demo-tensorrt.tar.gz

# Copiar
scp aria-demo-tensorrt.tar.gz usuario@destino:/ruta/

# Cargar
gunzip -c aria-demo-tensorrt.tar.gz | docker load
```

**IMPORTANTE**: Las imágenes x86_64 NO funcionan en ARM64 (Jetson). Debes usar `Dockerfile.jetson` y construir en el dispositivo.
