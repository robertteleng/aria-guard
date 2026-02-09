#!/bin/bash
# =============================================================================
# ARIA Demo - Docker Build Helper
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Auto-detect GPU compute capability
detect_gpu_arch() {
    if [ -n "${CUDA_ARCH_BIN:-}" ]; then
        echo -e "${YELLOW}Using manual CUDA_ARCH_BIN=${CUDA_ARCH_BIN}${NC}"
        return
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        CUDA_ARCH_BIN="7.5,8.6,8.9,12.0"
        echo -e "${YELLOW}nvidia-smi not found, using all architectures: ${CUDA_ARCH_BIN}${NC}"
        return
    fi

    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    local compute_cap
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

    if [ -z "$compute_cap" ]; then
        CUDA_ARCH_BIN="7.5,8.6,8.9,12.0"
        echo -e "${YELLOW}Could not detect GPU arch, using all: ${CUDA_ARCH_BIN}${NC}"
        return
    fi

    CUDA_ARCH_BIN="$compute_cap"
    echo -e "${GREEN}Detected GPU: ${gpu_name} (compute ${compute_cap})${NC}"
    echo -e "${GREEN}Building only for CUDA_ARCH_BIN=${CUDA_ARCH_BIN}${NC}"
}

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  base     Build base image (OpenCV+CUDA+NVDEC) - ~20 min, solo una vez"
    echo "  app      Build app image (requiere base) - ~3 min"
    echo "  all      Build base + app"
    echo "  run      Run container con video test"
    echo "  dev      Run container en modo desarrollo (mount src/)"
    echo ""
    echo "Workflow típico:"
    echo "  1. Primera vez:  $0 all"
    echo "  2. Cambios código: usa 'dev' (no rebuild)"
    echo "  3. Cambios deps:   $0 app"
    echo "  4. Cambios OpenCV: $0 base && $0 app"
}

build_base() {
    detect_gpu_arch
    echo -e "${YELLOW}Building base image (OpenCV+CUDA+NVDEC)...${NC}"
    echo "This will take ~20 minutes..."
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base \
        --build-arg CUDA_ARCH_BIN="$CUDA_ARCH_BIN" \
        -t aria-base:opencv-nvdec .
    echo -e "${GREEN}✓ Base image built: aria-base:opencv-nvdec${NC}"
}

build_app() {
    # Check if base exists
    if ! docker image inspect aria-base:opencv-nvdec >/dev/null 2>&1; then
        echo -e "${RED}Error: Base image not found. Run '$0 base' first.${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Building app image...${NC}"
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.app -t aria-demo:tensorrt .
    echo -e "${GREEN}✓ App image built: aria-demo:tensorrt${NC}"
}

run_container() {
    VIDEO="${1:-/app/data/test_60fps.mp4}"
    MODE="${2:-outdoor}"

    echo -e "${YELLOW}Running aria-demo with video: $VIDEO, mode: $MODE${NC}"
    docker run --rm -it --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -p 5000:5000 \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/data:/app/data" \
        aria-demo:tensorrt \
        python run.py "$VIDEO" "$MODE"
}

run_dev() {
    VIDEO="${1:-/app/data/test_60fps.mp4}"
    MODE="${2:-outdoor}"

    echo -e "${YELLOW}Running in DEV mode (src mounted)${NC}"
    docker run --rm -it --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -p 5000:5000 \
        -v "$(pwd)/src:/app/src:ro" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/run.py:/app/run.py:ro" \
        aria-demo:tensorrt \
        python run.py "$VIDEO" "$MODE"
}

case "${1:-}" in
    base)
        build_base
        ;;
    app)
        build_app
        ;;
    all)
        build_base
        build_app
        ;;
    run)
        run_container "$2" "$3"
        ;;
    dev)
        run_dev "$2" "$3"
        ;;
    *)
        usage
        ;;
esac
