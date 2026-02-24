# Research — Navegación Asistida para Personas Ciegas/Baja Visión

Evidencia de papers académicos para guiar decisiones de diseño en aria-guard.
Última actualización: 2026-02-24.

---

## 1. Peligros Reales para Peatones Ciegos

### ¿Qué causa accidentes?

| Peligro | Severidad | Fuente |
|---|---|---|
| **Vehículos** (especialmente híbridos/eléctricos silenciosos) | Fatal | PMC3046409 — Toyota híbrido más difícil de detectar |
| **Hoyos/canales descubiertos** | Lesión grave | PMC4909643 — encuesta Irán |
| **Obstáculos aéreos** (barras, andamios, ramas) | Lesión craneal | PMC4909643 — no detectables con bastón |
| **Bordillos/desniveles** | Caída | PMC4909643 |
| **Intersecciones complejas** | Atropello | PMC3358127 — 13/163 encuestados habían sido atropellados |
| **Mobiliario urbano** (postes, bicicletas aparcadas) | Golpe | PMC4909643 |

**Dato clave:** Muertes de peatones +78% desde 2009 en EE.UU. (NHTSA).

**Implicación para aria-guard:** Los vehículos silenciosos son el peligro #1. Obstáculos aéreos y hoyos son críticos pero YOLO no los detecta bien → futuro trabajo con depth map analysis.

### Modelo de 3 Niveles de Peligro (Gao et al., 2025 — Nature Communications)

Paper más relevante encontrado. Dispositivo wearable (gafas + smartphone) con cámaras + radar ToF.

**Define 3 niveles: Attention → Warning → Danger**

Justificación:
1. El nivel de peligro representa la amenaza mejor que el tipo de objeto solo
2. 3 niveles reducen carga cognitiva vs. procesar múltiples tipos de obstáculos
3. Un coche aparcado vs. un coche a 50km/h son niveles de amenaza completamente diferentes

**Resultados:** 100% evasión de colisiones, tiempo de respuesta <320ms, 11h batería. Testado 7 meses con 12 voluntarios ciegos.

- Fuente: [Nature Communications PMC11933268](https://pmc.ncbi.nlm.nih.gov/articles/PMC11933268/)

**Implicación para aria-guard:** Nuestro sistema anuncia tipo de objeto ("car left"). La evidencia sugiere que deberíamos comunicar **nivel de peligro** ("danger left", "warning straight") en vez del tipo. El tipo es ruido cognitivo — al usuario no le importa si es un coche o un bus, le importa cuánto peligro hay.

### Otros Modelos de Priorización

- **Cuadrícula adaptativa** (IEEE 2023): Prioriza obstáculo más inmediato dentro de proximidad. Detecta <20m, prioriza <2m con 95% precisión.
- **Lógica difusa** (PMC8466919): Inputs de profundidad + velocidad humana → evaluación de riesgo de ruta.
- **Multi-factor** (PMC10781372): Distancia + clase + trayectoria + posición angular.

---

## 2. Audio Feedback — Qué Funciona

### Beeps vs TTS vs Earcons vs Audio Espacial

| Método | Pros | Contras | Evidencia |
|---|---|---|---|
| **BRR (Beep Repetition Rate)** | Intuitivo, sin latencia, no interrumpe habla | No comunica tipo de peligro | EyeCane (PMC8070041) — usuarios estimaron distancia correctamente |
| **TTS conciso** | Información semántica rica | Latencia (~200ms), interrumpe audio ambiente | Usuarios prefieren 5-7 palabras máx. Latencia >1s reduce confianza |
| **Spearcons** (TTS acelerado) | Velocidad + precisión superiores a earcons | Requiere aprendizaje | PubMed 23516800 — superaron earcons y auditory icons |
| **Audio espacial** | Dirección intuitiva | Requiere bone conduction o auriculares buenos | PMC7909643 — efectivo sin entrenamiento largo |
| **Earcons** | Compactos, rápidos | Requieren aprendizaje, menos intuitivos | Inferiores a spearcons en estudios |

### Codificación de Distancia por BRR (Beep Repetition Rate)

Esquema probado (inspirado en asistentes de estacionamiento):

```
400ms entre beeps = "far"
300ms entre beeps = "medium"
200ms entre beeps = "close"
100ms entre beeps = "very_close"
```

Adicionalmente:
- **Pitch más alto = más cerca** (correlación negativa con distancia)
- **Volumen = tamaño del objeto** (más fuerte = más grande)

### Hallazgo Crítico: Carga Cognitiva vs Audio Espacial

> "Cuando la carga cognitiva era **baja**, el audio espacial era **preferido**. Cuando la carga cognitiva **aumentaba**, las preferencias se **invertían** hacia un único flujo de audio interrumpible."
> — ACM doi:10.1145/1978942.1979258

**Implicación:** Audio espacial sí, pero con fallback a flujo único cuando hay muchas alertas simultáneas.

### TTS: Preferencias de Usuarios

- **Conciso: 5-7 palabras máximo** — usuarios prefieren brevedad
- **Latencia <500ms** percibida como "inmediata"
- **Latencia >1s** reduce confianza en el sistema
- No anunciar todo lo detectado — solo lo peligroso

### Fatiga de Alertas

Problema documentado: usuarios reportaron **sobrecarga cognitiva y ansiedad** cuando el sistema anunciaba todas las detecciones.

Parámetros de filtrado probados:
- Ancho de zona de colisión
- Cooldown entre alertas del mismo obstáculo
- **Máximo de anuncios concurrentes** (lo más efectivo)

**Solución Gao et al.:** 3 niveles de peligro en vez de tipo de objeto → reduce carga cognitiva drásticamente.

### Bone Conduction > Auriculares Normales

- Localización del sonido **prácticamente idéntica** a auriculares tradicionales
- Hasta **5 ubicaciones de fuente** con alta precisión
- **No obstruyen el oído externo** → el usuario sigue oyendo tráfico, personas, etc.
- Tiempo de navegación **menor** con binaural vs voz sola

Fuente: MDPI Applied Sciences 11(8), 3356

### Microsoft Soundscape (2018-2022, ahora open-source)

Usó binaural 3D audio donde los sonidos se perciben como provenientes de puntos de interés. Permitía construir imagen mental del entorno. Premiado por NCBI Irlanda.

---

## 3. Implicaciones para aria-guard

### Lo que estamos haciendo bien ✅

1. **Cooldowns entre alertas** — evidencia confirma que filtrar es crítico
2. **Canales independientes** (vehículos, semáforos, señales) — no se pisan entre sí
3. **Priorización por distancia × tipo × approach** — alineado con literatura
4. **Beep espacial + TTS** — combinación validada en múltiples estudios
5. **Gaze como factor** — usuario mirando = menos urgente (validado)

### Lo que deberíamos cambiar/mejorar ⚠️

1. **Comunicar nivel de peligro, no tipo de objeto**
   - Actual: "car left", "person straight"
   - Evidencia: "warning left", "danger straight"
   - El tipo es ruido cognitivo. El nivel de peligro es lo que necesitan.
   - **Cambio propuesto:** Mapear priority score → 3 niveles (attention/warning/danger)

2. **BRR (Beep Repetition Rate) para distancia**
   - Actual: un beep con pitch variable
   - Evidencia: repetición de beeps más rápida = más cerca (400ms→100ms)
   - Más intuitivo que un solo beep con pitch diferente

3. **Limitar anuncios concurrentes**
   - Actual: decide() puede devolver 4 alertas simultáneas
   - Evidencia: máximo 1-2 alertas simultáneas
   - **Cambio propuesto:** Priorizar la más peligrosa, suprimir el resto

4. **Bone conduction como hardware recomendado**
   - No bloquea oídos → el usuario sigue oyendo tráfico
   - Localización del sonido funciona igual de bien

5. **Obstáculos aéreos y hoyos** — peligros críticos que no detectamos
   - YOLO no los ve, pero depth map analysis podría detectar suelo faltante
   - Futuro: análisis de depth map para "ground plane anomalies"

### Lo que NO deberíamos hacer ❌

1. Anunciar todo lo detectado — causa ansiedad y sobrecarga
2. Descripciones largas — máximo 5-7 palabras
3. Audio espacial complejo cuando hay muchas alertas — simplificar a flujo único
4. Ignorar vehículos eléctricos/híbridos — son el peligro #1 y el usuario no los oye

---

## 4. Papers Clave (para referencia)

| Paper | Año | Relevancia | Fuente |
|---|---|---|---|
| Gao et al. — Wearable obstacle avoidance with cross-modal learning | 2025 | Modelo 3 niveles, 100% evasión | Nature Comm. PMC11933268 |
| Hybrid vehicle detection by blind pedestrians | 2011 | Vehículos silenciosos | PMC3046409 |
| Outdoor difficulties visually impaired | 2016 | Encuesta de peligros reales | PMC4909643 |
| Street crossing decisions accuracy | 2012 | Intersecciones, atropellos | PMC3358127 |
| EyeCane — distance sonification | 2021 | BRR para distancia | PMC8070041 |
| Spearcons navigation performance | 2013 | Spearcons > earcons | PubMed 23516800 |
| WatchOut obstacle sonification | 2019 | 85% evasión con sonificación | ACM 3308561.3353779 |
| Cognitive load + spatial audio | 2011 | Espacial OK si baja carga | ACM 1978942.1979258 |
| Bone conduction navigation | 2021 | Bone conduction = tradicional | MDPI AppSci 11(8) 3356 |
| Sonification review | 2024 | Review sistemático | PubMed 38469665 |
| Guiding blind pedestrians (TTC) | 2020 | Ajuste velocidad, 14 ciegos | IMWUT 2020 |
| Wearable ETAs systematic review | 2023 | 89 estudios | IEEE 10148956 |
| Fuzzy risk assessment | 2021 | Lógica difusa para riesgo | PMC8466919 |

---

## 5. Diseño del Algoritmo v3 — Threat Model

### El problema

En una calle concurrida (Tokyo, Barcelona, NYC) hay 30+ objetos detectados por frame.
El sistema actual tiene 4 canales de alerta independientes que pueden disparar a la vez.
Resultado: caos auditivo, fatiga de alertas, usuario ignora el sistema.

### Principio fundamental (Gao 2025, Nature)

> "Al usuario no le importa si es un coche o un bus. Le importa cuánto peligro hay y de dónde viene."

### Arquitectura propuesta: 2 canales, 1 amenaza

```
┌─────────────────────────────────────────────────────────┐
│                    TRACKED OBJECTS                       │
│                  (SimpleTracker, N objetos)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              THREAT SCORING (por objeto)                 │
│                                                         │
│  Para cada TrackedObject:                               │
│    1. Calcular threat_level: NONE / ATTENTION /         │
│       WARNING / DANGER                                  │
│    2. Basado en collision_risk (0.0 – 1.0)              │
│                                                         │
│  collision_risk = f(TTC, bearing_stability, zone,       │
│                     object_type, is_gazed)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ALERT ARBITER (decide qué alertar)         │
│                                                         │
│  Canal A — AMENAZA (máx 1 a la vez):                    │
│    • Selecciona top-1 por collision_risk                 │
│    • Cooldown adaptativo según nivel:                   │
│      DANGER=1.5s, WARNING=3.0s, ATTENTION=5.0s          │
│    • Rate limit global: máx 4 alertas/30s               │
│    • DANGER nunca se suprime                            │
│                                                         │
│  Canal B — CONTEXTO (informativo, baja prioridad):      │
│    • Semáforos (rojo/verde) y señales (stop)            │
│    • Solo si Canal A está en silencio                   │
│    • Cooldown largo (5-8s)                              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    AUDIO OUTPUT                          │
│                                                         │
│  DANGER:  beep rápido (BRR 100ms) + "danger left"      │
│  WARNING: beep medio (BRR 300ms) + "warning straight"  │
│  ATTENTION: solo beep suave (BRR 500ms), sin TTS       │
│  CONTEXT: TTS informativo ("red light", "stop sign")   │
└─────────────────────────────────────────────────────────┘
```

### collision_risk: cómo se calcula

El score de riesgo combina 4 señales que la investigación identifica como las más
importantes para predecir colisión:

```python
def collision_risk(track: TrackedObject) -> float:
    """Score 0.0 (safe) to 1.0 (imminent collision).

    Basado en:
    - TTC (Time-to-Collision) — factor dominante (Mobileye, Euro NCAP)
    - CBDR (Constant Bearing, Decreasing Range) — bearing estable = colisión
    - Zone — centro de trayectoria = mayor riesgo
    - Object class — vehículos > personas > obstáculos estáticos
    """
    risk = 0.0

    # --- Factor 1: TTC (50% del peso) ---
    # El factor más importante según la literatura (NHTSA, Mobileye, Euro NCAP)
    # TTC = distancia / velocidad_acercamiento
    # Con depth relativo: TTC_proxy = depth_value / approach_speed
    #
    # Umbrales (de ADAS comerciales, adaptados a peatón):
    #   TTC < 1.5s → DANGER (Euro NCAP AEB activation)
    #   TTC < 3.0s → WARNING (Mobileye FCW)
    #   TTC < 5.0s → ATTENTION (si en trayectoria)
    #   TTC > 5.0s → safe

    if track.approach_speed > 0.01:  # Se acerca
        ttc = track.depth_value / track.approach_speed  # frames
        # Normalizar a 0-1: TTC 0 frames = 1.0, TTC 150 frames (5s@30fps) = 0.0
        ttc_factor = max(0.0, 1.0 - ttc / 150.0)
    else:
        ttc_factor = 0.0

    # Objetos estáticos cercanos también son peligro (bordillo, poste)
    # Distancia pura como proxy de TTC para estáticos
    static_proximity = {
        "very_close": 0.8,  # Ya estás encima
        "close": 0.4,
        "medium": 0.1,
        "far": 0.0,
    }
    if track.approach_speed <= 0.01:
        ttc_factor = static_proximity.get(track.distance, 0.0)

    risk += ttc_factor * 0.50

    # --- Factor 2: CBDR — bearing estable + acercamiento (25% del peso) ---
    # Principio de navegación: si el ángulo no cambia y la distancia baja,
    # colisión es inevitable. Esto captura bicis/motos laterales que
    # mantienen curso de colisión.
    #
    # lateral_speed ~0 + approach_speed > 0 = CBDR
    # lateral_speed alto = objeto se va, no es peligro

    if track.approach_speed > 0.01:
        bearing_stable = max(0.0, 1.0 - abs(track.lateral_speed) / 0.02)
        cbdr_factor = bearing_stable * min(1.0, track.approach_speed / 0.03)
    else:
        cbdr_factor = 0.0

    risk += cbdr_factor * 0.25

    # --- Factor 3: Zone — centro = trayectoria del usuario (15% del peso) ---
    zone_risk = {"center": 1.0, "left": 0.4, "right": 0.4}
    risk += zone_risk.get(track.zone, 0.3) * 0.15

    # --- Factor 4: Object class — vehículos son letales (10% del peso) ---
    class_risk = {
        "car": 1.0, "truck": 1.0, "bus": 1.0,
        "motorcycle": 0.9, "bicycle": 0.7,
        "person": 0.3, "dog": 0.2,
        "chair": 0.15, "backpack": 0.05,
    }
    risk += class_risk.get(track.name, 0.1) * 0.10

    return min(1.0, risk)
```

### Mapeo risk → threat_level

```python
# Umbrales calibrados con la literatura
if collision_risk >= 0.6:
    threat_level = DANGER      # ~TTC < 1.5s equivalente
elif collision_risk >= 0.35:
    threat_level = WARNING     # ~TTC < 3.0s equivalente
elif collision_risk >= 0.15:
    threat_level = ATTENTION   # ~TTC < 5.0s equivalente
else:
    threat_level = NONE        # safe, no alertar
```

### Gaze como modificador (NO como factor en risk)

```
La investigación dice: gaze indica dirección de marcha del usuario.
Si el usuario mira hacia un objeto, es consciente de él.

NO reducir el risk score por gaze — el objeto sigue siendo peligroso.
SÍ usar gaze para modular la URGENCIA de la alerta:
  - not gazed: alerta full (beep + TTS)
  - gazed: solo beep suave (el usuario ya lo sabe)

Excepción: DANGER nunca se suprime, gazed o no.
```

### Alert Arbiter — supresión adaptativa

```python
# Rate limiting global
MAX_ALERTS_30S = 6  # Máximo 6 alertas en 30 segundos (~12/min)
                     # Papers sugieren 3-4/min normal, hasta 12/min en pico

# Cooldowns por nivel
COOLDOWN = {
    DANGER: 1.5,     # Re-alertar rápido si sigue en peligro
    WARNING: 3.0,    # Tiempo para reaccionar
    ATTENTION: 5.0,  # Solo referencia, no urgente
}

# Regla anti-saturación:
# Si 4+ alertas en últimos 20s → duplicar cooldowns de WARNING y ATTENTION
# DANGER nunca se duplica — el peligro real siempre pasa

# Regla de Canal B (contexto):
# Solo hablar si Canal A lleva >3s en silencio
# Así el semáforo no pisa la alerta de un coche
```

### TTS: qué decir

```
Sistema actual: "car left", "person straight"
Propuesto:      "danger left", "warning straight"

¿Por qué?
- El tipo de objeto es ruido cognitivo (Gao 2025)
- "danger left" son 2 palabras, procesables en <300ms
- El usuario necesita saber: ¿cuánto peligro? ¿de dónde?
- NO necesita saber: ¿qué tipo de objeto es?

Excepciones que SÍ merecen tipo:
- "red light" / "green light" — info de cruce, no peligro
- "stop sign" — info de cruce
- Estos van por Canal B (contexto)
```

### Audio: BRR + Pitch + Espacialización

El sistema actual (audio.py) tiene 2 frecuencias fijas (500/1000Hz) y 1 beep
por alerta. No hay relación continua entre sonido y peligro.

#### Sistema actual vs propuesto

```
ACTUAL:
  - 2 frecuencias: 500Hz (normal), 1000Hz (critical)
  - 1 beep por alerta, duración 0.1-0.25s
  - Volume: 0.25 (far) → 1.0 (very_close)
  - Pan L/R: 100%/20% — burdo pero funcional
  - Cooldown fijo 0.3s entre beeps

PROPUESTO (basado en papers):
  - BRR como canal primario de urgencia
  - Pitch como canal secundario de distancia
  - Pan L/R se mantiene (funciona bien sin HRTF)
  - Envelope suave (fade in/out ya existe)
```

#### BRR (Beep Repetition Rate) — canal primario

Inspirado en sensor de aparcamiento (EyeCane, PMC8070041).
La repetición comunica urgencia mejor que un solo beep.

```python
# Beeps por ráfaga según threat_level:
BRR = {
    DANGER:    {"count": 3, "gap_ms": 80,  "duration_ms": 60},   # bip-bip-bip rápido
    WARNING:   {"count": 2, "gap_ms": 150, "duration_ms": 80},   # bip-bip
    ATTENTION: {"count": 1, "gap_ms": 0,   "duration_ms": 100},  # bip suave
}

# ¿Por qué ráfagas y no beeps continuos?
# - Ráfaga corta (< 400ms total) no interfiere con TTS que viene después
# - El número de beeps es distinguible sin pensar: 3=peligro, 2=aviso, 1=info
# - Beep continuo causa fatiga (papers de UCI: alarmas continuas ignoradas)
```

#### Pitch — canal secundario (refuerzo)

```python
# Frecuencia correlaciona con distancia (más agudo = más cerca)
# Rango perceptualmente uniforme: 400-1200Hz (evitar >2kHz que es molesto)
PITCH = {
    "very_close": 1100,  # Hz — agudo, urgente
    "close":       800,
    "medium":      600,
    "far":         400,  # Hz — grave, suave
}

# ¿Por qué no usar pitch como canal primario?
# - Los usuarios ciegos distinguen mejor BRR que pitch (EyeCane study)
# - Pitch requiere referencia auditiva (¿1000Hz es alto o bajo?)
# - BRR es intuitivo sin entrenamiento: más rápido = más urgente
```

#### Volumen — refuerzo terciario

```python
# Se mantiene el VOLUME_MAP actual — funciona bien
VOLUME = {
    "very_close": 1.0,
    "close":      0.7,
    "medium":     0.45,
    "far":        0.25,
}
```

#### Espacialización (Pan L/R)

```python
# El pan actual (100%/20%) es burdo pero distinguible.
# 3 posiciones es suficiente para peatón (left/center/right).
# Los papers dicen que 5 posiciones son distinguibles, pero
# 3 bastan para "de dónde viene el peligro".
#
# Mejora propuesta: pan proporcional al bearing en vez de discreto
PAN = {
    "left":   (1.0, 0.15),   # (left_vol, right_vol)
    "center": (0.7, 0.7),    # Ligeramente atenuado vs mono para evitar confusión
    "right":  (0.15, 1.0),
}

# Futuro: si el tracker da bearing en radianes, mapear a pan continuo:
# pan = 0.5 + bearing / fov_h  (0=full left, 0.5=center, 1=full right)
```

#### Secuencia temporal de una alerta completa

```
Ejemplo: WARNING, left, close

  t=0ms     t=80ms    t=230ms   t=380ms        t=800ms
  |─beep─|  |─beep─|  |──gap──|  |───TTS───────|
  800Hz     800Hz      silencio   "warning left"
  L=0.7     L=0.7
  R=0.15    R=0.15

  Duración total: ~800ms (< 1s target)

Ejemplo: DANGER, center, very_close

  t=0ms   t=60ms  t=140ms t=200ms t=280ms       t=600ms
  |beep|  |beep|  |beep|  |─gap─| |──TTS──────|
  1100Hz  1100Hz  1100Hz  silencio "danger straight"
  L=0.7   L=0.7   L=0.7
  R=0.7   R=0.7   R=0.7

  Duración total: ~600ms (urgente, rápido)

Ejemplo: ATTENTION, right, medium

  t=0ms      t=100ms
  |──beep──|
  600Hz
  L=0.15
  R=0.45

  Sin TTS (ATTENTION no habla — solo beep informativo)
  Duración total: 100ms
```

#### Bone conduction (nota para hardware)

```
Los papers recomiendan bone conduction headphones porque:
- No tapan los oídos → el usuario sigue oyendo tráfico
- Localización del sonido igual de buena que auriculares normales
- Shokz OpenRun (~$130) o AfterShokz son los más usados en estudios

Nuestro audio funciona con cualquier salida — bone conduction es una
recomendación de hardware, no requiere cambio de código.
```

### Escenarios de validación con Tokyo_POV.mp4

```
Escenario 1: Calle concurrida, 20 personas, 3 coches
  Esperado: NINGUNA alerta si todos están lejos y estáticos
  Métrica: 0 alertas/min en "calma"

Escenario 2: Bici acercándose por lateral
  Esperado: WARNING cuando TTC < 3s, DANGER cuando TTC < 1.5s
  Métrica: 1 alerta, bien cronometrada

Escenario 3: Cruce con semáforo
  Esperado: Canal B dice "red light" solo si Canal A está en silencio
  Métrica: TTS no se pisan

Escenario 4: 30 segundos de caos total (intersección Tokyo)
  Esperado: máx 6 alertas en 30s, solo las más peligrosas
  Métrica: rate < 12/min, 0 alertas simultáneas

Escenario 5: Coche aparcado vs coche acercándose
  Esperado: coche aparcado = NONE, coche acercándose = WARNING/DANGER
  Métrica: 0 falsas alertas por estáticos lejanos
```

### Benchmark script: qué medir

```
1. alerts_per_minute — total de alertas emitidas por minuto
2. danger_alerts — cuántas DANGER (deben ser pocas, reales)
3. alert_gap_min — gap mínimo entre alertas consecutivas (>1.5s)
4. silent_ratio — % del tiempo en silencio (target: >80%)
5. false_alerts — alertas sobre objetos estáticos lejanos (target: 0)
6. concurrent_alerts — alertas simultáneas (target: 0, siempre 1 a la vez)
7. context_vs_threat — ratio alertas contexto / amenaza
```

---

## 6. Próximos Pasos

1. **Implementar collision_risk()** en tracker.py — reemplaza _update_priority()
2. **Nuevo AlertArbiter** — reemplaza AlertDecisionEngine con 2 canales
3. **BRR en audio.py** — beeps repetitivos en vez de beep único
4. **TTS de nivel** — "danger left" en vez de "car left"
5. **Benchmark offline** — correr Tokyo_POV.mp4 sin audio, contar métricas
6. **Iterar thresholds** — ajustar con datos reales del benchmark
