# Bluetooth A2DP (estéreo) + mic bajo demanda en Jetson — caso Shokz OpenRun Pro 2

**Fecha:** 2026-06-26
**Plataforma:** NVIDIA Jetson Orin Nano, JetPack 6.x (Ubuntu, ARM64)
**Stack audio:** PulseAudio 15.99.1 (no PipeWire), `pulseaudio-module-bluetooth` 15.99.1, BlueZ 5.64
**Dispositivo:** Shokz OpenRun Pro 2 (conducción ósea, con micro)

## Síntoma

El casco empareja y conecta, y `bluetoothctl info` muestra que **sí** anuncia A2DP:

```
UUID: Audio Sink   (0000110b-...)   <- A2DP Sink
UUID: Handsfree    (0000111e-...)   <- HFP
UUID: A/V Remote Control / Target   <- AVRCP
```

Pero el único perfil que expone PulseAudio es `handsfree_head_unit` (HFP, **mono 16 kHz**).
El perfil `a2dp_sink` **no aparece** en `pactl list cards` (ni siquiera como "available: no"),
y `pactl set-card-profile <card> a2dp_sink` devuelve `Failure: No such entity`.

## Diagnóstico (qué se descartó)

| Hipótesis | Prueba | Resultado |
|-----------|--------|-----------|
| Carrera de registro de endpoints en PA | `pulseaudio -k` + reconectar | Sigue HFP-only |
| Módulo BT mal cargado | `unload/load module-bluetooth-discover` | Sigue HFP-only |
| Multipoint (A2DP ocupado por un móvil) | Confirmado por el usuario: no hay móvil | Descartado |
| `ofono` forzando HFP | `pgrep ofono` | No corre |
| App grabando del micro fuerza HFP | `pactl list source-outputs` | Vacío |
| El device no soporta A2DP | `bluetoothctl info` → Audio Sink UUID | Soporta A2DP |

## Causa raíz (confirmada)

`bluetoothd` arranca con los plugins de audio **deshabilitados**:

```
$ ps -o args= -C bluetoothd
/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp
```

El origen es un **drop-in de NVIDIA** que viene con JetPack:

```
/usr/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf
  [Service]
  ExecStart=
  ExecStart=/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp
```

`--noplugin=a2dp,avrcp` desactiva en BlueZ los plugins que registran los SEP de A2DP
y el `org.bluez.Media1` que PulseAudio necesita para crear el endpoint estéreo. Por eso:

- **HFP sí funciona**: usa el backend nativo de PA vía `org.bluez.Profile1`, que no depende del plugin `a2dp`.
- **A2DP no existe**: sin el plugin `a2dp` de BlueZ, PA nunca recibe el endpoint → el perfil ni se lista.

NVIDIA lo deja así por defecto en las imágenes de Jetson (histórico, para no interferir con
su pila). Es un quirk conocido del Jetson, no un fallo de PulseAudio.

## Por qué NO ayuda cambiar de framework

PipeWire/WirePlumber (la alternativa típica) usan **el mismo** `org.bluez.Media1` / plugin `a2dp`
de BlueZ por debajo. Con el plugin desactivado, PipeWire se quedaría **igual de cojo** (solo HFP).
El cuello de botella está en **BlueZ (el daemon)**, no en el servidor de audio. La solución es
reactivar el plugin, no migrar de stack. Migrar a PipeWire añadiría trabajo sin resolver nada.

## Fix (re-activar el plugin A2DP de BlueZ)

Crear un override en `/etc` que enmascara el de NVIDIA (mismo nombre = lo reemplaza por completo):

`/etc/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf`
```ini
[Service]
ExecStart=
ExecStart=/usr/lib/bluetooth/bluetoothd
```

Aplicar (requiere sudo una vez):
```bash
sudo systemctl daemon-reload
sudo systemctl restart bluetooth
sleep 3
bluetoothctl connect <BT_MAC>
sleep 5
pactl set-card-profile bluez_card.A8_F5_E1_CB_06_21 a2dp_sink
pactl set-default-sink bluez_sink.A8_F5_E1_CB_06_21.a2dp_sink
```

Verificación: `pactl list short sinks` debe mostrar
`bluez_sink.A8_F5_E1_CB_06_21.a2dp_sink  ...  s16le 2ch 44100Hz` (2ch = estéreo).

Nota: reiniciar `bluetooth.service` **no afecta a la Aria** (va por USB-NCM/DDS, no por el
controlador BT clásico), así que el pipeline de aria-guard sigue corriendo.

## Micrófono bajo demanda (A2DP ↔ HFP)

En **Bluetooth clásico, A2DP y HFP son mutuamente excluyentes** sobre una conexión: o tienes
salida **estéreo sin micro** (A2DP), o **mono + micro** (HFP/HSP). No hay las dos a la vez.
"Mic bajo demanda" = **cambiar de perfil** cuando se necesita capturar y volver a A2DP después:

```bash
CARD=bluez_card.A8_F5_E1_CB_06_21
# capturar micro (mono, ~16 kHz mSBC/CVSD):
pactl set-card-profile $CARD handsfree_head_unit
#   ... fuente disponible: bluez_source.A8_F5_E1_CB_06_21.handsfree_head_unit
# volver a estéreo:
pactl set-card-profile $CARD a2dp_sink
```

- El cambio de perfil cuesta ~0.5–1 s y produce un pequeño corte de audio (re-negocia AVDTP/SCO).
- `module-bluetooth-policy` puede hacerlo **automático** (`auto_switch=2`, por defecto): al abrir
  un stream de captura sobre la fuente BT, salta a HFP; al cerrarlo, vuelve a A2DP.
- Calidad de micro: **mSBC** (wideband, ~16 kHz) si el casco y BlueZ lo negocian; si no, **CVSD**
  (narrowband, 8 kHz). El OpenRun Pro 2 soporta mSBC.
- Implicación para aria-guard: el TTS/beeps salen en estéreo (A2DP); si en el futuro se quiere
  entrada de voz (comandos), hay que aceptar el corte del switch a HFP durante la captura.

### Hallazgo empírico (2026-06-26) y decisión de fuente de micro

Test definitivo (hablando al casco 8 s, monitorizando el contador SCO de `hciconfig`):

```
$ ./scripts/bt-audio.sh mic 8     # + hablando fuerte
SCO RX delta = 1002 bytes en 8s   # audio real serían DECENAS DE MILES
frames=143640 dur=8.98s rms=2 max=297   # silencio: PA rellena con ceros
```

El perfil HFP cambia y la fuente aparece, pero el **transporte SCO no entrega el audio del
micro al host** (~1000 bytes en 8 s = nada). Causa: el controlador es un **Realtek USB
(`rtk_btusb`)** que anuncia paquetes **CVSD (HV1/2/3) pero NO eSCO (EV3/4/5)** — mSBC necesita
eSCO transparente, así que no puede transportar; y el SCO CVSD tampoco entrega los paquetes del
micro. Es una debilidad conocida del SCO de Realtek en Linux. (Por qué Siri sí: en un móvil el
chip BT + firmware implementan HFP/SCO completo por hardware.)

Contexto adicional: usar **los micros de las propias gafas Aria** tampoco es viable aquí — la SDK
de audio bajo FEX-Emu **crasheaba**, y además añadir un stream de audio DDS roba ancho de banda al
RGB (ya hay `CRITICAL DDS: sample lost`), degradando la detección de colisiones (seguridad crítica).

**Decisión (fuente de micro) — elegida: Shokz vía dongle USB BT (CSR):**

| Fuente | Veredicto | Motivo |
|--------|-----------|--------|
| **Shokz vía dongle USB BT (CSR8510)** | ✅ **Elegido** | Mismo casco, pero un adaptador BT que SÍ hace SCO/HFP bien en Linux. No es 2º micro. ~5-10 €. |
| **Shokz vía BT onboard (Realtek)** | ❌ No funciona aquí | SCO no transporta el micro (1002 bytes/8s). Limitación del `rtk_btusb`. |
| **Micro USB cableado** | 🔁 Fallback | Captura ALSA directa, fiable, pero es un micro aparte. Solo si el dongle fallara. |
| **Micros Aria** | ❌ Rechazado | Crashean bajo FEX y roban ancho de banda al RGB de seguridad. |

El mecanismo de switch (`bt-audio.sh mic`) queda implementado y probado (el perfil cambia y vuelve);
el bloqueo es el SCO del **adaptador onboard**, no el código. La integración de voz (aria-scene) debe
alimentar su `SpeechRecognizer` desde una **captura de micro del host** (el `bluez_source...` del casco
emparejado al dongle), nunca desde `aria_camera.get_audio_samples` (la vía que crasheaba).

### Setup del dongle USB BT (cuando llegue)

1. Enchufar el dongle → aparece `hci1` (`hciconfig -a` debe listar dos controladores).
2. (Opcional, recomendado) bloquear el BT Realtek onboard para que el casco use solo el dongle:
   `sudo rfkill block <idx-realtek>` o `hciconfig hci0 down`.
3. Emparejar el Shokz al dongle: `bluetoothctl` → `select <BD_ADDR-dongle>` → `scan on` →
   `pair`/`connect <BT_MAC>`.
4. Verificar: `./scripts/bt-audio.sh mic 5` hablando → `rms` debe subir a >1000 y el `SCO RX delta`
   a decenas de miles. A2DP estéreo sigue por el dongle (el fix del plugin a2dp es global, ya aplicado).
5. Ojo con clones CSR falsos: si el `mic` sigue dando silencio, probar otro dongle (CSR8510 genuino).

## Fuentes

- [Fix disabled A2DP profile for bluetooth headset in Linux — A. Zaharia](https://alexandra-zaharia.github.io/posts/fix-disabled-a2dp-profile-for-bluetooth-headset-in-linux/)
- [BluetoothUser/a2dp — Debian Wiki](https://wiki.debian.org/BluetoothUser/a2dp)
- [Bluetooth headset not working as microphone (A2DP sink, HSP/HFP) — Arch Linux Forums](https://bbs.archlinux.org/viewtopic.php?id=256167)
- BlueZ `--noplugin` y el drop-in `nv-bluetooth-service.conf` son específicos de las imágenes Jetson de NVIDIA.
