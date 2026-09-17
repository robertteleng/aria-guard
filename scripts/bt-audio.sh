#!/usr/bin/env bash
# bt-audio.sh — control del audio Bluetooth (Shokz OpenRun Pro 2) en el Jetson.
#
# A2DP (estéreo, sin micro) y HFP (mono + micro) son EXCLUYENTES en Bluetooth clásico:
# para capturar el micro hay que cambiar de perfil y volver. Este script encapsula eso.
#
# Uso:
#   ./bt-audio.sh status            # perfil activo, sinks y sources
#   ./bt-audio.sh a2dp              # salida estéreo (default para TTS/beeps)
#   ./bt-audio.sh mic [seg] [out]   # cambia a HFP, graba, vuelve a A2DP
#   ./bt-audio.sh fix               # imprime el fix del plugin A2DP (necesita sudo)
#
# NOTA (Jetson): el SCO sobre HCI del BT onboard puede entregar SILENCIO (rms=0) por el
# micro HFP. Si `mic` graba silencio, usa un micro USB (captura ALSA directa, sin switch).
# Ver docs/research/bluetooth-a2dp-jetson-shokz.md.
set -euo pipefail

DEV="${ARIA_BT_DEV:?set ARIA_BT_DEV to the Bluetooth MAC of the headset (e.g. AA:BB:CC:DD:EE:FF)}"
CARD="bluez_card.${DEV//:/_}"
SINK_A2DP="bluez_sink.${DEV//:/_}.a2dp_sink"
SRC_HFP="bluez_source.${DEV//:/_}.handsfree_head_unit"

cmd="${1:-status}"

case "$cmd" in
  status)
    echo "Card:   $CARD"
    pactl list cards 2>/dev/null | sed -n "/$CARD/,/Active Profile/p" | grep -iE "Active Profile|a2dp_sink|handsfree" || true
    echo "Default sink: $(pactl get-default-sink 2>/dev/null)"
    echo "--- sinks ---";   pactl list short sinks   2>/dev/null | grep "${DEV//:/_}" || echo "  (sin sink BT)"
    echo "--- sources ---"; pactl list short sources 2>/dev/null | grep "${DEV//:/_}" || echo "  (sin source BT)"
    ;;

  a2dp)
    pactl set-card-profile "$CARD" a2dp_sink
    pactl set-default-sink "$SINK_A2DP" 2>/dev/null || true
    echo "A2DP estéreo activo (default sink=$SINK_A2DP)"
    pactl list short sinks | grep "${DEV//:/_}" || true
    ;;

  mic)
    secs="${2:-3}"; out="${3:-/tmp/bt_mic.wav}"
    echo "[mic] cambiando a HFP..."
    pactl set-card-profile "$CARD" handsfree_head_unit
    sleep 2
    echo "[mic] grabando ${secs}s de $SRC_HFP -> $out"
    timeout "$((secs+2))" parecord -d "$SRC_HFP" --channels=1 --rate=16000 \
        --format=s16le --file-format=wav "$out" 2>/dev/null || true
    # Aviso si el SCO entregó silencio (problema conocido del BT onboard del Jetson)
    python3 - "$out" <<'PY' || true
import sys, wave, audioop
try:
    w = wave.open(sys.argv[1], 'rb'); n = w.getnframes(); d = w.readframes(n)
    rms = audioop.rms(d, w.getsampwidth()) if n else 0
    print(f"[mic] frames={n} dur={n/16000:.2f}s rms={rms}")
    if rms == 0:
        print("[mic] AVISO: rms=0 (silencio). SCO sobre HCI del Jetson no entrega audio; usa micro USB.")
except Exception as e:
    print("[mic] no se pudo leer el wav:", e)
PY
    echo "[mic] volviendo a A2DP estéreo..."
    pactl set-card-profile "$CARD" a2dp_sink
    pactl set-default-sink "$SINK_A2DP" 2>/dev/null || true
    ;;

  fix)
    cat <<'EOF'
# El plugin A2DP de BlueZ viene DESACTIVADO en Jetson (nv-bluetooth-service.conf:
#   ExecStart=/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp)
# Reactívalo con (necesita sudo una vez):
sudo install -d /etc/systemd/system/bluetooth.service.d
printf '[Service]\nExecStart=\nExecStart=/usr/lib/bluetooth/bluetoothd\n' | \
  sudo tee /etc/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf >/dev/null
sudo systemctl daemon-reload && sudo systemctl restart bluetooth
# luego: bluetoothctl connect <DEV> && ./bt-audio.sh a2dp
EOF
    ;;

  *)
    echo "uso: $0 {status|a2dp|mic [seg] [out]|fix}"; exit 1;;
esac
