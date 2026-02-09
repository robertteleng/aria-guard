# =============================================================================
# Dockerfile.app - Aplicación ARIA (builds rápidos ~2-3 min)
# =============================================================================
# Usa aria-base:opencv-nvdec como base (ya tiene OpenCV+CUDA+NVDEC)
#
# Build: docker build -f Dockerfile.app -t aria-demo:tensorrt .
# Run:   docker run --gpus all -p 5000:5000 aria-demo:tensorrt
#
# Para desarrollo con hot-reload de código:
#   docker run --gpus all -p 5000:5000 \
#     -v $(pwd)/src:/app/src:ro \
#     -v $(pwd)/models:/app/models \
#     aria-demo:tensorrt python run.py webcam
# =============================================================================

FROM aria-base:opencv-nvdec

# Build dependencies for NeMo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install TensorRT
RUN pip install --no-cache-dir \
    tensorrt==10.8.0.43 \
    tensorrt-cu12==10.8.0.43 \
    tensorrt_lean==10.8.0.43 \
    tensorrt_dispatch==10.8.0.43

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ultralytics (YOLO)
RUN pip install --no-cache-dir ultralytics>=8.3.0

# Install NeMo TTS
RUN pip install --no-cache-dir \
    Cython packaging \
    nemo_toolkit[tts]==2.2.1 \
    huggingface_hub

# Install projectaria_eyetracking with Git LFS weights
RUN git lfs install && \
    git clone --depth 1 https://github.com/facebookresearch/projectaria_eyetracking.git /tmp/aria_gaze && \
    cd /tmp/aria_gaze && \
    git lfs pull --include="projectaria_eyetracking/inference/model/pretrained_weights/**" && \
    pip install /tmp/aria_gaze && \
    mkdir -p /opt/venv/lib/python3.11/site-packages/projectaria_eyetracking/inference/model/pretrained_weights && \
    # Copy LFS weights manually (pip install does not include them) \
    cp -r /tmp/aria_gaze/projectaria_eyetracking/inference/model/pretrained_weights/* \
        /opt/venv/lib/python3.11/site-packages/projectaria_eyetracking/inference/model/pretrained_weights/ && \
    mkdir -p /app/models/gaze_weights && \
    cp -r /tmp/aria_gaze/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1 /app/models/gaze_weights/ && \
    rm -rf /tmp/aria_gaze

# Copy application code
COPY run.py .
COPY src/ src/
COPY scripts/ scripts/

# Create directories
RUN mkdir -p /app/models /app/data

# Default command
CMD ["python", "run.py", "webcam"]
