# Piper voice (es_ES) — voz on-device para los alertas habladas

`ARIA_TTS_ENGINE=piper` carga esta voz (CPU, en proceso, sin GPU). El binario `.onnx`
**no se versiona** (63 MB, está en `.gitignore`). Descárgalo aquí:

```bash
cd models/piper
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium
curl -L -o es_ES-davefx-medium.onnx      "$BASE/es_ES-davefx-medium.onnx"
curl -L -o es_ES-davefx-medium.onnx.json "$BASE/es_ES-davefx-medium.onnx.json"
```

El `launch_pipeline.sh` del bridge activa la voz por defecto si el `.onnx` está presente
(`VOICE=0` lo desactiva). Un solo quant por modelo (disciplina de disco): si pruebas otra
voz, borra la anterior al decidir.
