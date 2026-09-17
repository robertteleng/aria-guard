#!/usr/bin/env bash
# mic-test.sh — test AISLADO e INTERACTIVO del micro del Shokz (HFP), con cue de
# "habla ahora" y un medidor de nivel EN VIVO. Pensado para correrlo tú con `!`.
#
#   ! bash ~/Projects/aria-guard/scripts/mic-test.sh 6
#
# Cambia a HFP, mide el nivel mientras hablas, y vuelve a A2DP estéreo al salir.
set -uo pipefail
DEV="${ARIA_BT_DEV:?set ARIA_BT_DEV to the Bluetooth MAC of the headset (e.g. AA:BB:CC:DD:EE:FF)}"
CARD="bluez_card.${DEV//:/_}"
SRC="bluez_source.${DEV//:/_}.handsfree_head_unit"
SINK_A2DP="bluez_sink.${DEV//:/_}.a2dp_sink"
SECS="${1:-6}"

cleanup() {
  pactl set-card-profile "$CARD" a2dp_sink >/dev/null 2>&1 || true
  pactl set-default-sink "$SINK_A2DP" >/dev/null 2>&1 || true
  echo; echo "[mic-test] A2DP estéreo restaurado."
}
trap cleanup EXIT INT TERM

echo "[mic-test] cambiando a HFP (micro)..."
pactl set-card-profile "$CARD" handsfree_head_unit >/dev/null 2>&1
sleep 2
if ! pactl list short sources | grep -q "$SRC"; then
  echo "[mic-test] ERROR: no aparece la fuente del micro ($SRC)"; exit 1
fi

echo "[mic-test] Cuando veas 'HABLA AHORA', habla normal. La barra SUBE si te capta."
for i in 3 2 1; do echo "  empezando en $i..."; sleep 1; done
echo ">>>>>>>>>>  HABLA AHORA durante ${SECS}s  <<<<<<<<<<"

timeout "$((SECS+1))" parec -d "$SRC" --channels=1 --rate=16000 --format=s16le 2>/dev/null \
  | python3 - "$SECS" <<'PY'
import sys, time, audioop
secs = int(sys.argv[1]); rate = 16000
chunk = rate * 2 // 4          # 0.25 s de audio s16le mono
start = time.time(); peak = 0; got = 0
while time.time() - start < secs:
    data = sys.stdin.buffer.read(chunk)
    if not data:
        break
    got += len(data)
    rms = audioop.rms(data, 2)
    peak = max(peak, rms)
    bars = min(40, rms // 50)
    print(f"\r  nivel |{'#'*bars}{' '*(40-bars)}| rms={rms:5d}", end="", flush=True)
print("\n")
print(f"  bytes recibidos = {got}   pico rms = {peak}")
if peak > 300:
    print("  ✅ EL MIC CAPTA TU VOZ — funciona, no hace falta dongle.")
elif got < 8000:
    print("  ❌ casi no llega audio (stream SCO muerto) -> toca dongle USB BT (CSR).")
else:
    print("  ⚠️ llega algo pero muy flojo -> revisar códec/volumen del mic.")
PY
