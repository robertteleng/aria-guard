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
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libportaudio2 \
    libasound2-dev \
    libsndfile1 \
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

# Crear y activar venv fuera de /app para evitar que el volumen lo oculte
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# =============================================================================
# LAYER CACHING OPTIMIZATION
# =============================================================================
# Copiar SOLO requirements.txt primero para cachear pip install
# Si requirements.txt no cambia, esta capa pesada se reutiliza
# =============================================================================

# Actualizar pip
RUN pip install --upgrade pip && pip install Cython

# Copiar requirements.txt primero (si existe)
COPY requirements.txt* /app/

# Instalar dependencias de requirements.txt (si existe)
# Esta capa se cachea mientras requirements.txt no cambie
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Instalar PyTorch con CUDA 12.4 primero (usando extra-index-url para no bloquear PyPI)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalar resto de dependencias desde PyPI
RUN pip install \
    projectaria-client-sdk \
    projectaria-tools \
    git+https://github.com/facebookresearch/projectaria_eyetracking.git \
    opencv-python-headless \
    numpy \
    flask \
    sounddevice \
    ultralytics \
    transformers \
    "nemo_toolkit[tts]"

# =============================================================================
# CÓDIGO FUENTE (última capa - cambios frecuentes)
# =============================================================================
# Esta capa se invalida con cada cambio de código, pero las dependencias
# ya están cacheadas arriba, así que el rebuild es rápido.
# =============================================================================

COPY . /app/

# Puerto para Flask
EXPOSE 5000

# Comando por defecto
CMD ["python", "run.py"]
