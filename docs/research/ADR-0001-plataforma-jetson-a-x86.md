# ADR-0001 — Migrar la plataforma de Jetson (ARM+FEX) a x86 + GPU discreta

- **Estado:** Aceptado; **revisión pendiente** (2026-09-17). Su causa principal, emular con FEX un SDK
  solo x86_64, ya no existe: `projectaria-client-sdk` 2.5.0 publica wheels nativas aarch64. El replay
  benchmark medido después (`docs/REPLAY_BENCHMARK.md`) muestra el Jetson a ~19 FPS sin emulación, con
  la CPU como cuello; la decisión se revisará con una sesión en vivo con SDK nativo.
- **Fecha:** 2026-06-30
- **Afecta a:** `aria-guard` (producto) y `aria-arm64-bridge` (queda obsoleto)
- **Decisores:** Robert (+ análisis de esta sesión, con medidas en device)

## Contexto

`aria-guard` hace detección de colisiones en tiempo real para peatones ciegos con
gafas Meta Aria. La plataforma actual es una **Jetson Orin Nano (ARM64)**. El
`projectaria-client-sdk` es un **binario x86_64 cerrado** → en ARM **no puede
ejecutarse nativo**, solo **emulado bajo FEX-Emu**. Para eso existe el repo entero
`aria-arm64-bridge` (receiver FEX + puente ZMQ hacia el consumidor ARM).

## Problema (evidencia medida 2026-06-30)

| Métrica | Valor en Jetson | Causa |
|---|---|---|
| RGB receiver (solo) | ~9,4-10 FPS | techo de `profile12` **bajo emulación FEX** |
| Pipeline completo | ~8 FPS | + contención de RAM unificada |
| **Latencia percepción** | **~486 ms** | FEX + backlog de colas a 8 FPS + salto ZMQ |
| Uso GPU bajo carga | **~18%** (7,3 W, 51 °C) | **input-bound**, NO compute-bound |

Diagnóstico: el cuello de botella **no es la potencia de la GPU** (está ociosa) ni
el calor (51 °C, sobra). Es **(1) emular el SDK x86 con FEX** (techo arquitectónico
~10 FPS) y **(2) la contención de ancho de banda de la RAM unificada** (las ráfagas
de GPU dejan sin oxígeno al receiver, 9,4→8). No es optimizable: el binario es
cerrado (no se puede recompilar) y un modelo más ligero no ayuda (la GPU sobra).

Descomposición de los 486 ms: cómputo solo ~22 ms; el resto es FEX + ~4 frames de
backlog en colas a 8 FPS + serializar 6 MB/frame por ZMQ. **Casi todo eliminable.**

## Decisión

**Migrar a x86_64 + GPU NVIDIA discreta.** Hardware objetivo: **Intel NUC 11
Enthusiast "Phantom Canyon" (RTX 2060 Mobile, 6 GB VRAM)** o portátil con **RTX 2060
(6 GB)**. Se **descarta** la ruta Jetson+FEX. Se evita la RTX 3050 (4 GB): VRAM
justa para depth métrico + detección + VIO simultáneos.

## Consecuencias

**Positivas:**
- **SDK Aria nativo** (sin emulación) → objetivo ~30 FPS; latencia estimada
  **~60-120 ms** (≈4-8× menos; a confirmar midiendo `/stats` en el NUC).
- **Desaparece toda la capa bridge** (FEX + receiver + ZMQ): `aria-guard` usa su
  ruta ya existente `aria:usb` (`AriaDemoObserver`), en proceso. Menos piezas.
- **VRAM dedicada** → sin contención de RAM unificada; entra **profundidad métrica**
  y modelos mayores (lo que desbloquea la calibración de umbrales).
- **Unifica plataforma con `aria-nav`** (cuyo GATE ya es NUC x86) → el VIO produce
  pose/escala métrica que retroalimenta la profundidad de `aria-guard`.

**Negativas / costes:**
- Se pierde el factor wearable de bajo consumo del Jetson — pero esa ventaja ya
  estaba anulada por el peaje de FEX.
- Hay que **reconstruir los engines TensorRT** para la GPU nueva (son por-dispositivo).
- NUC/portátil es más voluminoso y consume más que el Jetson.

**Neutro / importante:**
- La lógica de `aria-guard` (ego-motion, tracker, looming de bbox, AlertArbiter,
  audio, dashboard) es **Python/PyTorch puro → porta a x86 sin cambios.** El trabajo
  hecho NO se tira.
- `aria-arm64-bridge` queda **obsoleto**, conservado como el experimento que
  **demostró con datos** el techo del Jetson.

## Alternativas consideradas (y rechazadas)

1. **Seguir en Jetson y optimizar** — rechazada: el techo es la emulación FEX de un
   binario cerrado (no recompilable); la GPU ya está ociosa (modelo ligero no ayuda);
   ~10 FPS / 486 ms son arquitectónicos, no de tuning.
2. **WiFi en vez de USB** — no quita el peaje de FEX (la emulación del SDK sigue ahí).

## Notas de migración (cuando se haga)

- Usar `./run.sh aria:usb all` (`AriaDemoObserver`), NO el bridge.
- Imagen Docker base x86+CUDA estándar → se eliminan los líos del base L4T del Jetson
  (numpy<2, ABI de cv2, etc.).
- Reconstruir engines TensorRT en la GPU nueva.
- **Validar latencia y FPS** con `/stats` (`latency_ms`, `detector_fps`) y la
  validación controlada (clip + `benchmark_offline.py`) directamente en x86.

## Referencias
- `docs/research/fps-and-streaming-stability.md` (medidas FPS/estabilidad)
- `docs/research/depth-and-approach-audit.md` (por qué hace falta depth métrico)
