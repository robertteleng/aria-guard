# =============================================================================
# Dockerfile para ARIA Demo - Visual Assistance System
# =============================================================================
#
# IMPORTANTE: Usa Ubuntu 22.04 porque el Aria SDK (FastDDS) requiere una
# version especifica de glibc. Ubuntu 24.04 con glibc 2.39-0ubuntu8.7 causa
# crashes (munmap_chunk(): invalid pointer) al conectar con las gafas.
#
# Build:
#   docker build -t aria-demo .
#   docker compose build
#
# Run (con acceso USB y GPU):
#   docker run -it --rm --privileged \
#     -v /dev/bus/usb:/dev/bus/usb \
#     -p 5000:5000 \
#     --gpus all \
#     aria-demo
#
#   O con docker compose:
#   docker compose up
#
# Para Jetson (ARM64):
#   Cambiar imagen base a: nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
#
# =============================================================================

FROM ubuntu:22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
# - software-properties-common: para add-apt-repository
# - python3.11: version de Python compatible con Aria SDK
# - libgl1-mesa-glx, libglib2.0-0, etc: dependencias de OpenCV
# - libportaudio2, libasound2-dev: para audio (sounddevice)
# - usbutils: para lsusb (debug de conexion USB)
# - libstdc++6: C++ runtime actualizado para PyTorch
# - espeak-ng: text-to-speech para asistencia visual
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libportaudio2 \
    libasound2-dev \
    usbutils \
    libstdc++6 \
    espeak-ng \
    bash-completion \
    vim \
    nano \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Activar autocompletado y colores
RUN echo "source /etc/bash_completion" >> /root/.bashrc && \
    echo "alias ls='ls --color=auto'" >> /root/.bashrc

# Crear directorio de trabajo
WORKDIR /app

# Crear y activar venv
RUN python3.11 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Instalar dependencias Python principales
# - projectaria-client-sdk: SDK oficial para streaming de Aria glasses
# - projectaria-tools: herramientas para VRS y calibracion
# - aria-glasses: utilidades adicionales de Aria
# - opencv-python-headless: procesamiento de imagen (sin GUI)
# - flask: servidor web para dashboard
# - torch, torchvision: PyTorch para modelos de IA
# - ultralytics: YOLO para deteccion de objetos
RUN pip install --upgrade pip && \
    pip install \
    projectaria-client-sdk \
    projectaria-tools \
    aria-glasses \
    opencv-python-headless \
    numpy \
    flask \
    sounddevice \
    torch \
    torchvision \
    ultralytics

# Copiar código del proyecto
COPY . /app/

# Instalar dependencias adicionales si hay requirements.txt
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Puerto para Flask
EXPOSE 5000

# Comando por defecto
CMD ["python", "run.py"]
