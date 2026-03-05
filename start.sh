#!/bin/bash
# Wrapper que carga jemalloc antes de Python para evitar heap corruption con Aria SDK / FastDDS
JEMALLOC_X86="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
JEMALLOC_ARM="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"

# Cleanup: matar procesos huerfanos de sesiones anteriores (puerto + GPU)
lsof -i :5000 -t 2>/dev/null | xargs -r kill -9 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 0.5

if [ -f "$JEMALLOC_X86" ]; then
    export LD_PRELOAD="$JEMALLOC_X86"
elif [ -f "$JEMALLOC_ARM" ]; then
    export LD_PRELOAD="$JEMALLOC_ARM"
fi

exec uv run python run.py "$@"
