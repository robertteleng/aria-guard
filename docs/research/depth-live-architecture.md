# Arquitectura de Profundidad Métrica Densa en Vivo — Aria Gen 1 / Jetson Orin

> **Estado:** investigación + auditoría completadas (2026-06-27). Plan de fases propuesto.
> **Decisión central:** profundidad métrica vía **VIO mono-inercial (escala IMU) + DepthAnything anclado**, NO estéreo.
> **Hardware sin Orin esta sesión:** los benchmarks de FPS en Orin quedan como TODO en hardware real.

Documentos prompt asociados:
- [PROMPT_stereo-slam-aria-gen1.md](PROMPT_stereo-slam-aria-gen1.md) — research del problema estéreo SLAM-SLAM.
- [PROMPT_gpt-validate-vio-architecture.md](PROMPT_gpt-validate-vio-architecture.md) — validación cruzada con GPT.

---

## 0. Resumen ejecutivo

El objetivo es profundidad **métrica** (metros reales) densa, en vivo, on-device en Jetson Orin,
usando Meta Aria Gen 1. Tres fuentes candidatas, una sola ganadora:

| Fuente | Veredicto | Razón |
|--------|-----------|-------|
| **Estéreo SLAM-SLAM** | ❌ Descartado | 35° de solape (Meta lo confirma como insuficiente; subió a 80° en Gen 2) |
| **FoundationStereo (NVIDIA)** | ❌ Descartado para Gen 1 | Mismo bloqueo de solape + ~1 FPS en Orin (modelo grande). Es para Gen 2 |
| **DepthAnything monocular solo** | ⚠️ Insuficiente | Funciona, pero sin escala métrica fiable (falla pasado ~1 m) |
| **VIO mono-inercial + DepthAnything anclado** | ✅ **Arquitectura elegida** | IMU da escala; mapa disperso da metros; DepthAnything rellena forma |

**La frase clave:** DepthAnything **va**, pero **falta VIO**. El VIO no reemplaza a DepthAnything —
le da los **metros reales** que al monocular le faltan, y aporta ego-motion.

---

## 1. Separar dos problemas ortogonales (no mezclar)

| Problema | Pregunta | Lo resuelve |
|----------|----------|-------------|
| **Profundidad** | "¿a qué distancia está lo que veo?" | VIO (escala métrica) + densificación monocular |
| **Cobertura de campo** | "¿qué porción del mundo cubro?" | Cámaras a lados distintos + mapa 3D persistente que crece con el movimiento |

El estéreo quiere **solape**. Aquí queremos lo contrario: **poco solape = más campo**. Las dos cámaras
SLAM NO son un par estéreo — son dos VIO monoculares independientes que cubren lados distintos.

---

## 2. Hardware verificado — Aria Gen 1

Fuentes: [hardware_spec](https://facebookresearch.github.io/projectaria_tools/docs/tech_spec/hardware_spec),
[arXiv 2308.13561](https://arxiv.org/abs/2308.13561), [Meta blog Gen 2](https://www.meta.com/blog/aria-gen-2-research-glasses-under-the-hood-reality-labs/).

| Componente | Spec | Relevancia |
|------------|------|------------|
| 2× cámara SLAM | 640×480, monocroma, **global shutter**, fisheye **150° HFOV**, anguladas hacia fuera | VIO corre aquí (global shutter obligatorio) |
| Solape SLAM-SLAM | **35°** (Gen 2 = 80°) | Insuficiente para estéreo |
| Baseline SLAM-SLAM | ~130 mm (no publicado por Meta) | — |
| 1× cámara RGB | hasta 2880², **rolling shutter**, fisheye 110° HFOV | DepthAnything aquí. NO sirve para VIO (rolling shutter) |
| 2× IMU | 800 Hz / 1000 Hz, calibradas y sincronizadas (reloj común ns) | **Fuente de escala métrica** |
| Calibración | intrínsecas Fisheye624 + extrínsecas cámara-cámara y cámara-IMU en VRS/JSON | Frame device = cámara SLAM izquierda |

---

## 3. Por qué se descarta el estéreo (con evidencia, no opinión)

Tres fuentes independientes convergen (mi research empírico + geométrico + validación de GPT):

1. **Meta, oficial** ([issue #103](https://github.com/facebookresearch/projectaria_tools/issues/103)):
   *"The camera orientation [is] optimized for field of view rather than stereo depth recovery."*
   La única profundidad de MPS en Gen 1 es la nube semi-densa del SLAM, no disparidad estéreo.
2. **Geometría** (estimación con B≈130 mm, IFOV 0.26°/px, error matching 0.5 px):
   error estéreo ≈ **0.6 m a 5 m**, **2.5 m a 10 m**. Útil solo <3 m, en una franja de ~149 px.
3. **Ningún paper ni repo** usa las SLAM de Gen 1 para estéreo. InCrowd-VI las usa para VIO;
   Aria Digital Twin da depth por raycasting sintético, no por estéreo.

**FoundationStereo** ([arXiv 2501.09898](https://arxiv.org/abs/2501.09898)) no cambia esto: es un matcher
estéreo, requiere solape; solo cubriría ~23% de la imagen en Gen 1, y el modelo grande va a **~1 FPS en
AGX Orin**. Es la herramienta de **Gen 2** (80° solape, repo `projectaria_gen2_depth_from_stereo`).
Único uso teórico en Gen 1 (estéreo temporal de 1 cámara) **no lo soporta** — eso es VIO.

---

## 4. La arquitectura elegida (runtime en vivo)

```
Cámara SLAM izquierda (global shutter) + IMU
   └─ VIO mono-inercial (ORB-SLAM3 mono-inertial / VINS) en C++
        └─► trayectoria 6DOF + ESCALA métrica (IMU) + nube de puntos 3D dispersa métrica

Cámara SLAM derecha (global shutter)   [FASE POSTERIOR]
   └─ 2º VIO monocular INDEPENDIENTE → mismo mapa
        └─► NO estéreo. Cobertura de campo (cada cámara mira a un lado).

Cámara RGB central (rolling shutter)
   └─ DepthAnything V2 → profundidad densa RELATIVA (forma, sin escala)
        └─ ANCLADA a la escala del VIO: se proyectan los puntos métricos del mapa sobre el
           frame RGB (extrínsecas de fábrica) y se resuelve scale+shift por mínimos cuadrados.

Distancia a obstáculo (YOLO bbox):
   └─ mediana de los puntos del mapa VIO dentro del bbox (métrica directa).
      Para colisión casi nunca hace falta mapa denso: basta distancia a objetos + plano de suelo.
```

### 4.1 Fundamentos (el "por qué", para aprender)

- **Por qué la IMU da escala:** mide aceleración real en m/s²; la gravedad es referencia métrica.
  El monocular puro reconstruye a escala arbitraria; cámara+IMU rompe esa ambigüedad. (VINS-Mono.)
- **Por qué global shutter es obligatorio para VIO:** el VIO asume que todos los píxeles de un frame
  son del mismo instante. El rolling shutter de la RGB da a cada fila un tiempo distinto → error
  geométrico dependiente de la velocidad. Por eso VIO va en las SLAM, no en la RGB.
- **Por qué el mapa persistente sustituye al estéreo para cobertura:** estéreo necesita solape
  **simultáneo**; el mapa necesita paralaje **temporal** (movimiento). Al girar la cabeza, el mapa
  crece. El poco solape deja de ser defecto y pasa a ser ventaja de campo.
- **La nube del VIO NO es un LiDAR:** es **dispersa** (solo donde hay textura — una pared lisa da
  cero puntos) y **se acumula con el movimiento** (no es una foto 3D instantánea). DepthAnything
  rellena la forma donde el VIO tiene agujeros. Lo más cercano a un "LiDAR pobre" sin sensor de
  profundidad en Gen 1.

### 4.2 Precisión esperada (verificado en benchmarks)

- ORB-SLAM3 mono-inercial: ATE **4.3 cm** medio en EuRoC, **1.1 cm** en TUM-VI (movimiento a mano).
  Supera VINS-Mono (~11 cm) 2.5×. Fuente: [arXiv 2007.11898](https://arxiv.org/abs/2007.11898).
- DepthAnything V2 Metric + TensorRT en Orin NX 16GB: ~43 FPS (referencia hackster, **a verificar en
  tu Orin concreto**).

---

## 5. Estado del workspace (auditoría 4 repos, 2026-06-27)

### aria-core (C++/CUDA — destino del runtime VIO)
- **Tiene esqueleto VIO propio** (no ORB-SLAM3): `EkfFusion` (EKF 15-state funcional),
  `SlamPipeline::estimatePose` (8-point Eigen), `OrbCudaExtractor`, `BoWLoopDetector`, `G2oMapper`
  (solo triangula, NO optimiza pose graph pese al nombre).
- **Gaps:** ningún adapter de hardware real (IAriaDevice sin implementar), sin preintegración IMU,
  sin inicialización VIO, sin depth adapter (H18 pendiente), calibración hardcodeada (fx=700…),
  IMU nunca llega al pipeline (`processIMU` sin caller).
- **No tiene ORB-SLAM3 ni VINS vendorizados.** La arquitectura hexagonal permite añadirlos como adapter.

### aria-nav (Python — prototipo, validación offline)
- Esqueleto SLAM en `src/core/slam/` (visual_odometry, ekf_fusion, imu_preintegration, loop_closure,
  mapper) — mayormente stub. Deps: opencv-contrib, g2opy, projectaria-tools.
- **Estéreo:** NO implementado. Solo 1 fila aspiracional en README + 1 checkbox sin marcar en PLAN.
  No hay StereoBM/SGBM/rectify. **El "estéreo como primer paso" nunca tuvo base en código.**

### aria-arm64-bridge (FEX + ZMQ — fuente de datos)
- Expone RGB funcional (11 FPS profile12). **IMU NO suscrito en producción** (`receiver.py:140` solo Rgb).
  SLAM cams: profile28 da 49 FPS pero desactivado en consumidor. Audio = crash bajo FEX.
- **Calibración extrínseca NO expuesta.** Timestamps NO sincronizados host↔device.
- **Riesgo FEX en Orin:** atomics lentos (issue #4120) → profile15 colapsa a 0.6 FPS. Riesgo de
  latencia para runtime en vivo.

### aria-guard (Python — L1, donde estamos)
- DepthAnything V2 Small integrado (TRT + HF fallback), pero `NORM_MINMAX` destruye la escala.
  Tracker consume `depth_value` adimensional. RealSense D435 da mm reales pero no estará en Aria.
- **Reusable:** wrapper TRT de DepthAnything, export ONNX→TRT, extracción bbox-aware con mediana,
  IPC shared-memory zero-copy. Bug #1 conocido: sin ego-motion, `approach_speed` confunde
  "yo me acerco" con "objeto se acerca".

---

## 6. Plan de implementación por fases

> Leyenda: **[P]** prototipo Python validable offline contra MPS · **[R]** runtime C++/CUDA.

### FASE 0 — Gate de viabilidad (BLOQUEA todo lo demás)
- **0.1 [P]** Benchmark DepthAnything V2 Small FP16 TensorRT **en Orin real**: FPS end-to-end,
  VRAM, latencia. Variantes (Small/Metric, input 518 vs menor). *Sin Orin → TODO en hardware.*
- **0.2 [R]** Verificar que aria-arm64-bridge puede entregar **cámara SLAM + IMU sincronizada +
  calibración extrínseca**. Sin esto, NINGÚN VIO arranca. Es el gate real. (Hoy: IMU no suscrito,
  calibración no expuesta → trabajo previo obligatorio.)

### FASE 1 — VIO mínimo con escala métrica
- **1.1 [P]** Prototipo en aria-nav: alimentar ORB-SLAM3 mono-inertial (o VINS) con VRS de Aria
  (offline) → validar trayectoria + escala contra MPS ground truth.
- **1.2 [R]** Integrar el VIO elegido en aria-core como adapter de `ISensorFusion`/pipeline,
  alimentado por el bridge (FASE 0.2).
- **1.3 [R]** Eliminar `NORM_MINMAX` de DepthAnything; retener float32 raw.

### FASE 2 — Fusión escala(VIO) × forma(DepthAnything)
- **2.1** Calibración extrínseca SLAM↔RGB (ver §7): proyectar puntos métricos del mapa al frame RGB.
- **2.2** Anclar DepthAnything: scale+shift por mínimos cuadrados contra puntos VIO proyectados.
- **2.3** Distancia por objeto: mediana de puntos VIO dentro de cada bbox YOLO.

### FASE 3 — Cobertura + ego-motion
- **3.1 [R]** 2º VIO sobre cámara SLAM derecha → mismo mapa (cobertura de campo).
- **3.2** Ego-motion al tracker: restar velocidad del usuario de `approach_speed` (bug #1 de aria-guard).
- **3.3** Mapa 3D persistente que acumula periferia.

### FASE 4 — Validación
- Contra MPS offline (trayectoria + nube semi-densa). Benchmark de alertas en vivo.

---

## 7. Calibración extrínseca SLAM↔RGB (tarea 4)

Aria guarda intrínsecas + extrínsecas por sensor en el VRS / device_calibration JSON.
`get_transform_device_sensor(label)` devuelve `T_Device_Sensor`. El frame device por defecto está
alineado con `camera-slam-left`.

Proyectar un punto del frame SLAM al RGB:
```
p_rgb = inverse(T_Device_RGB) · T_Device_SLAM · p_slam
```
Para `camera-slam-left`, `T_Device_SLAM` ≈ identidad (es el frame device). Luego se proyecta con la
intrínseca RGB **Fisheye624** (NO pinhole, salvo que se cree una calibración rectificada).

**Estado:** aria-guard YA recupera `_rgb_calib`/`_slam1_calib`/`_slam2_calib`
(`src/input/aria.py:107`) vía `device_calibration_from_json_string`, pero **`get_calibrations()`
no tiene ningún caller** — el punto de entrada existe, falta conectarlo.

---

## 7.bis Estrategia de prototipado y testeo (decidido 2026-06-28)

**Dónde se prototipa:** Intel NUC 11 (x86). El SDK de Aria corre **nativo** en x86
(`AriaDemoObserver`, `aria:usb`/`aria:wifi`) — **sin bridge ni FEX**. El bridge
(aria-arm64-bridge) existe SOLO porque el SDK no tiene binarios ARM64; es problema de
**despliegue en Orin**, no de prototipado. Verificado: `aria-arm64-bridge/README.md:3-6`.

**Modo de desarrollo: tiempo real + grabación VRS automática.** Se desarrolla con las gafas
en vivo (ver el mapa/trayectoria crecer en el dashboard), y el SDK graba la sesión a `.vrs`
de fondo (un flag, coste cero). La grabación es la red de seguridad: permite **reproducir**
un fallo exacto y construir **tests de regresión** repetibles. Análogo a git: desarrollas
normal, el VRS está ahí por si hay que volver atrás.

**Cómo se testea cada capa de forma independiente:** cada repo lee de una **interfaz**
(Observer / IAriaDevice), no de las gafas. La interfaz se rellena con:
- gafas en vivo (runtime / demo)
- VRS grabado (offline, repetible — banco de regresión)
- datos sintéticos (CI / pytest)

Las gafas reales son **una fuente más, no una dependencia**. aria-guard ya hace esto
(MockObserver / AriaDatasetObserver / AriaDemoObserver / RealSenseObserver, 27 tests sin hardware).

**Dónde vive el VIO y cómo se comparte el código de Aria:**
- VIO = **L2 → vive en aria-nav** (Python prototipo) → luego **aria-core** (C++ runtime).
  NO en aria-guard (L1, reactivo <33ms). NO juntar aria-guard + aria-nav (rompe el test
  independiente por capa). La unificación es tarea de **aria-core**.
- **Lector de Aria:** aria-nav lee VRS/streams **por su cuenta** con `projectaria-tools`
  (ya en su pyproject: `aria = ["projectaria-tools>=1.5"]`). NO se copia `aria.py` de
  aria-guard (la copia diverge — ya pasó con el `receiver.py` duplicado del bridge). NO se
  crea paquete compartido `aria-io` todavía (sobre-ingeniería prematura): se extrae solo
  *si* el solape duele de verdad más adelante.
- **Dashboard:** aria-nav necesita el suyo (trayectoria 6DOF + nube 3D, ya hay
  `src/core/visualization/trajectory_viewer.py`), distinto del de aria-guard (vídeo+alertas MJPEG).

**Cómo se hace VIO mono-inercial en Python (verificado):** NO hay wrapper de ORB-SLAM3
mantenido que exponga mono-inertial (todos abandonados o mono puro). La ruta real es
**gtsam** (API Python nativa, wheels ARM64): `PreintegratedImuMeasurements` (IMU) +
`SmartProjectionPoseFactor` (visual) + `ISAM2` (backend). El **frontend** visual (tracking
OpenCV + lectura IMU, ~300-500 LOC) se escribe a mano — y eso **es el curso
perception-from-scratch** (features sem5, epipolar sem6, EKF/preintegración sem7-8). El
prototipo de aria-nav y el curso son la misma actividad. ORB-SLAM3 C++ directo queda para
producción (aria-core), no para el prototipo.

## 8. Decisiones cerradas vs abiertas

**Cerradas (con evidencia):**
- Estéreo SLAM-SLAM y FoundationStereo: descartados para Gen 1.
- Escala métrica = IMU vía VIO.
- VIO sobre SLAM global-shutter, NO sobre RGB rolling shutter.

**Abiertas (bloqueadas por FASE 0):**
- Densidad: ¿basta nube dispersa + distancia a objetos, o hace falta densificar? → decide tras ver
  densidad real del mapa VIO en escena de calle (FASE 1.1).
- ORB-SLAM3 vs VINS en Orin ARM64 → decide tras compilar ambos.
- FPS de DepthAnything en Orin → decide tras benchmark 0.1.
