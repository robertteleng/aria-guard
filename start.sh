#!/bin/bash
# Wrapper que carga jemalloc antes de Python para evitar heap corruption con Aria SDK / FastDDS
JEMALLOC_X86="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
JEMALLOC_ARM="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"

if [ -f "$JEMALLOC_X86" ]; then
    export LD_PRELOAD="$JEMALLOC_X86"
elif [ -f "$JEMALLOC_ARM" ]; then
    export LD_PRELOAD="$JEMALLOC_ARM"
fi

# Deshabilitar SHM transport en FastDDS — evita heap corruption con SHM local
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

exec uv run python run.py "$@"
