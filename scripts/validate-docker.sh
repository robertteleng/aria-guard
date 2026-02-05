#!/bin/bash
# =============================================================================
# Validate Dockerfiles before building (catch errors early, save 20+ min)
# Usage: ./scripts/validate-docker.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

error() { echo -e "${RED}ERROR: $1${NC}"; ((ERRORS++)); }
warn()  { echo -e "${YELLOW}WARN:  $1${NC}"; ((WARNINGS++)); }
ok()    { echo -e "${GREEN}OK:    $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Validating Docker build files ==="
echo ""

# --- Check files exist ---
for f in Dockerfile.base Dockerfile.app docker-compose.yml docker-build.sh requirements.txt; do
    if [ -f "$ROOT/$f" ]; then
        ok "$f exists"
    else
        error "$f not found"
    fi
done

echo ""

# --- Dockerfile.base checks ---
echo "--- Dockerfile.base ---"
BASE="$ROOT/Dockerfile.base"

# Check NVDEC stubs (the key fix)
if grep -q "libnvcuvid.so" "$BASE"; then
    ok "NVDEC stub library created for cmake"
else
    error "Missing libnvcuvid.so stub - cmake won't find NVDEC"
fi

# Check nv-codec-headers
if grep -q "nv-codec-headers" "$BASE"; then
    ok "nv-codec-headers included"
else
    error "Missing nv-codec-headers (NVDEC/NVENC API headers)"
fi

# Check cmake NVCUVID flags
if grep -q "WITH_NVCUVID=ON" "$BASE"; then
    ok "WITH_NVCUVID=ON set"
else
    error "WITH_NVCUVID=ON not found in cmake flags"
fi

# Check libprotobuf in runtime stage
if grep -q "libprotobuf" "$BASE"; then
    ok "libprotobuf in runtime dependencies"
else
    error "Missing libprotobuf in runtime stage (cv2 import will fail)"
fi

# Check numpy in venv
if grep -q "pip install.*numpy" "$BASE"; then
    ok "numpy installed in venv"
else
    error "Missing numpy in venv (cv2 requires numpy)"
fi

# Check no build-time cv2 verification requiring GPU
if grep -q 'python.*import cv2.*getCudaEnabledDeviceCount' "$BASE"; then
    error "Build-time cv2 check uses CUDA (will fail without GPU in docker build)"
fi

echo ""

# --- Dockerfile.app checks ---
echo "--- Dockerfile.app ---"
APP="$ROOT/Dockerfile.app"

# Check base image reference
if grep -q "FROM aria-base:opencv-nvdec" "$APP"; then
    ok "Uses aria-base:opencv-nvdec"
else
    error "Wrong base image reference"
fi

# Check build-essential for NeMo C extensions
if grep -q "build-essential" "$APP"; then
    ok "build-essential for C extensions (cdifflib, texterrors)"
else
    error "Missing build-essential (NeMo deps cdifflib/texterrors need gcc)"
fi

# Check TensorRT version format (must be X.Y.Z.W)
TRT_VER=$(grep -oP 'tensorrt==\K[0-9.]+' "$APP" | head -1)
if [ -n "$TRT_VER" ]; then
    DOTS=$(echo "$TRT_VER" | tr -cd '.' | wc -c)
    if [ "$DOTS" -ge 3 ]; then
        ok "TensorRT version format correct: $TRT_VER"
    else
        error "TensorRT version needs 4-part format (e.g. 10.8.0.43), got: $TRT_VER"
    fi
fi

# Check line continuations (\ before &&)
# Look for lines ending with && but no backslash (inside RUN blocks)
BROKEN=$(awk '/^RUN /{in_run=1} in_run && /&&$/ && !/\\$/{print NR": "$0} /^[A-Z]/ && !/^RUN /{in_run=0}' "$APP")
if [ -n "$BROKEN" ]; then
    error "Missing backslash after && (Dockerfile will break):\n$BROKEN"
else
    ok "Line continuations look correct"
fi

echo ""

# --- docker-compose.yml checks ---
echo "--- docker-compose.yml ---"
COMPOSE="$ROOT/docker-compose.yml"

if grep -q "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" "$COMPOSE"; then
    ok "NVIDIA_DRIVER_CAPABILITIES includes video"
else
    error "NVIDIA_DRIVER_CAPABILITIES missing 'video' (NVDEC needs it at runtime)"
fi

echo ""

# --- docker-build.sh checks ---
echo "--- docker-build.sh ---"
BUILD="$ROOT/docker-build.sh"

NVDEC_RUNS=$(grep -c "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" "$BUILD")
if [ "$NVDEC_RUNS" -ge 2 ]; then
    ok "NVIDIA_DRIVER_CAPABILITIES in run + dev commands"
elif [ "$NVDEC_RUNS" -ge 1 ]; then
    warn "NVIDIA_DRIVER_CAPABILITIES only in one run command (check run + dev)"
else
    error "NVIDIA_DRIVER_CAPABILITIES missing from docker-build.sh run commands"
fi

echo ""

# --- Check base image exists (for app build) ---
echo "--- Docker images ---"
if docker image inspect aria-base:opencv-nvdec >/dev/null 2>&1; then
    ok "aria-base:opencv-nvdec exists"
else
    warn "aria-base:opencv-nvdec not found (run: ./docker-build.sh base)"
fi

echo ""
echo "=== Results: $ERRORS errors, $WARNINGS warnings ==="

if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}Fix errors before building.${NC}"
    exit 1
else
    echo -e "${GREEN}Ready to build.${NC}"
    exit 0
fi
