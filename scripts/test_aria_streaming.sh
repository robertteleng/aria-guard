#!/bin/bash
# =============================================================================
# Test Aria SDK Streaming - Diagnóstico de crash
# =============================================================================
# Prueba múltiples configuraciones para encontrar cuál funciona:
#   1. Docker (glibc 2.35) - debería funcionar
#   2. Host con MALLOC_CHECK_=0 - desactiva validación de heap
#   3. Host con LD_PRELOAD tcmalloc - reemplaza allocator
#   4. Host con LD_PRELOAD jemalloc - reemplaza allocator
#
# Uso:
#   ./scripts/test_aria_streaming.sh docker    # Test en Docker
#   ./scripts/test_aria_streaming.sh host      # Test directo en host
#   ./scripts/test_aria_streaming.sh malloc0   # Host + MALLOC_CHECK_=0
#   ./scripts/test_aria_streaming.sh tcmalloc  # Host + tcmalloc
#   ./scripts/test_aria_streaming.sh jemalloc  # Host + jemalloc
#   ./scripts/test_aria_streaming.sh all       # Todas las opciones
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# Check system info
show_system() {
    echo ""
    echo "============================================="
    echo " System Info"
    echo "============================================="
    echo "Kernel:  $(uname -r)"
    echo "glibc:   $(ldd --version 2>&1 | head -1 | grep -oP '[\d.]+$')"
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        echo "Driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    fi
    echo "============================================="
    echo ""
}

# Test 1: Docker container (glibc 2.35)
test_docker() {
    info "=== Test: Docker (Ubuntu 22.04, glibc 2.35) ==="

    if ! docker image inspect aria-demo:tensorrt &>/dev/null; then
        fail "Image aria-demo:tensorrt not found. Build with: docker/docker-build.sh all"
        return 1
    fi

    info "Running test_aria_only.py inside Docker..."
    docker compose -f docker/docker-compose.yml run --rm \
        -v "$PROJECT_ROOT/tests:/app/tests:ro" \
        aria-demo \
        python tests/test_aria_only.py 2>&1 | tee /tmp/aria_test_docker.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        ok "Docker test PASSED"
        return 0
    else
        fail "Docker test FAILED"
        return 1
    fi
}

# Test 2: Host direct
test_host() {
    info "=== Test: Host (direct, no workarounds) ==="
    info "glibc: $(ldd --version 2>&1 | head -1)"

    if [ ! -d ".venv" ]; then
        fail "No .venv found. Create with: python3 -m venv .venv && pip install -r requirements.txt"
        return 1
    fi

    .venv/bin/python tests/test_aria_only.py 2>&1 | tee /tmp/aria_test_host.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        ok "Host test PASSED"
        return 0
    else
        fail "Host test FAILED (expected with glibc 2.39)"
        return 1
    fi
}

# Test 3: Host + MALLOC_CHECK_=0
test_malloc_check() {
    info "=== Test: Host + MALLOC_CHECK_=0 ==="
    info "This disables glibc malloc validation checks"

    MALLOC_CHECK_=0 .venv/bin/python tests/test_aria_only.py 2>&1 | tee /tmp/aria_test_malloc0.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        ok "MALLOC_CHECK_=0 test PASSED"
        return 0
    else
        fail "MALLOC_CHECK_=0 test FAILED"
        return 1
    fi
}

# Test 4: Host + tcmalloc
test_tcmalloc() {
    info "=== Test: Host + LD_PRELOAD tcmalloc ==="

    # Find tcmalloc library
    local tcmalloc_lib=""
    for path in \
        /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
        /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so \
        /usr/lib/libtcmalloc_minimal.so.4 \
        /usr/lib/libtcmalloc_minimal.so; do
        if [ -f "$path" ]; then
            tcmalloc_lib="$path"
            break
        fi
    done

    if [ -z "$tcmalloc_lib" ]; then
        warn "tcmalloc not found. Install with: sudo apt install libtcmalloc-minimal4"
        return 1
    fi

    info "Using: $tcmalloc_lib"
    LD_PRELOAD="$tcmalloc_lib" .venv/bin/python tests/test_aria_only.py 2>&1 | tee /tmp/aria_test_tcmalloc.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        ok "tcmalloc test PASSED"
        return 0
    else
        fail "tcmalloc test FAILED"
        return 1
    fi
}

# Test 5: Host + jemalloc
test_jemalloc() {
    info "=== Test: Host + LD_PRELOAD jemalloc ==="

    local jemalloc_lib=""
    for path in \
        /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
        /usr/lib/x86_64-linux-gnu/libjemalloc.so \
        /usr/lib/libjemalloc.so.2 \
        /usr/lib/libjemalloc.so; do
        if [ -f "$path" ]; then
            jemalloc_lib="$path"
            break
        fi
    done

    if [ -z "$jemalloc_lib" ]; then
        warn "jemalloc not found. Install with: sudo apt install libjemalloc2"
        return 1
    fi

    info "Using: $jemalloc_lib"
    LD_PRELOAD="$jemalloc_lib" .venv/bin/python tests/test_aria_only.py 2>&1 | tee /tmp/aria_test_jemalloc.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        ok "jemalloc test PASSED"
        return 0
    else
        fail "jemalloc test FAILED"
        return 1
    fi
}

# Run all tests
test_all() {
    show_system

    local passed=0
    local failed=0
    local results=()

    for test_name in docker host malloc_check tcmalloc jemalloc; do
        echo ""
        echo "---------------------------------------------"
        if "test_$test_name"; then
            results+=("${GREEN}PASS${NC} - $test_name")
            ((passed++))
        else
            results+=("${RED}FAIL${NC} - $test_name")
            ((failed++))
        fi
        echo "---------------------------------------------"
    done

    echo ""
    echo "============================================="
    echo " Results Summary"
    echo "============================================="
    for r in "${results[@]}"; do
        echo -e "  $r"
    done
    echo ""
    echo -e "  Passed: ${GREEN}${passed}${NC}  Failed: ${RED}${failed}${NC}"
    echo "============================================="
}

case "${1:-all}" in
    docker)     show_system; test_docker ;;
    host)       show_system; test_host ;;
    malloc0)    show_system; test_malloc_check ;;
    tcmalloc)   show_system; test_tcmalloc ;;
    jemalloc)   show_system; test_jemalloc ;;
    all)        test_all ;;
    *)
        echo "Usage: $0 {docker|host|malloc0|tcmalloc|jemalloc|all}"
        exit 1
        ;;
esac
