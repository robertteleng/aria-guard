# DEVELOPER_DIARY.md — Bitácora de Diseño de Features

Registro detallado de cómo se pensó y construyó cada feature, aplicando el [Framework de las 5 Preguntas](PAIR_WORKFLOW.md#framework-las-5-preguntas).

Se llena **después** de completar cada feature. El objetivo es ver cómo pensaste, qué decidiste, y por qué — para que el patrón se vuelva natural.

---

<!-- TODO: Rellenar con las features ya completadas en este repo. Usa la plantilla de abajo. -->

## Plantilla por Feature

Copia esto para cada feature nueva:

```markdown
## Feature: [nombre]
**Fecha:** YYYY-MM-DD
**Branch:** `feature/...`
**Estado:** Completada / En progreso

---

### P1: Historia del Usuario
> "Yo [acción en primera persona]..."

### P2: Estados y Transiciones

¿Cuándo cambia algo? Dibuja cajas con flechas:
```
[A] ──qué pasa──▶ [B] ──qué pasa──▶ [C]
```

**¿Por qué estos estados?**
- ...

**¿Qué descarté?**
- ...

### P3: Veo / Necesito

| Estado | Lo que ve el usuario | Datos que necesito | De dónde vienen |
|---|---|---|---|
| ... | ... | ... | ... |

### P4: Inventario

| Necesito | ¿Existe? | Decisión | Por qué |
|---|---|---|---|
| ... | ... | Reusar / Extender / Crear | ... |

### P5: Diagrama de Pegamento

```
[Servicio A] ──evento──▶ [pegamento] ──acción──▶ [Servicio B]
```

**¿Por qué esta conexión?**
- ...

### Implementación

**Archivos creados:**
- ...

**Archivos modificados:**
- ...

**Decisiones clave:**
- ...

**Diagrama de arquitectura final:**
(cómo quedaron conectados los componentes)

### Reflexión
- **Lo que funcionó:** ...
- **Lo que costó:** ...
- **Lo que haría diferente:** ...
- **Patrón reutilizable:** ...
```
