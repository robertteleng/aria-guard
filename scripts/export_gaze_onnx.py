"""Exporta el modelo de gaze de Meta (projectaria_eyetracking) a ONNX.

El modelo SocialEyeModel devuelve un dict {"main","lower","upper"} — ONNX no
soporta dicts, asi que se envuelve concatenando a (1,6) en el orden que espera
el detector: [main_yaw, main_pitch, lower_yaw, lower_pitch, upper_yaw, upper_pitch].

Batch estatico=1 a proposito: el SplitAndConcat interno duplica el batch y
confunde la inferencia de shapes de TensorRT con ejes dinamicos.

La normalizacion [-0.5, 0.5] (min-max por muestra) NO va dentro del grafo —
el preprocesado del detector debe aplicarla antes de la inferencia.

Uso (dentro del contenedor, CPU):
    pip3 install --no-deps https://github.com/facebookresearch/projectaria_eyetracking/archive/refs/heads/main.zip
    pip3 install easydict pyyaml
    python3 scripts/export_gaze_onnx.py models/gaze.onnx
Despues, en el Orin:
    trtexec --onnx=models/gaze.onnx --saveEngine=models/gaze.engine --fp16 \
            --shapes=eye_images:1x2x240x320
"""
import sys
from pathlib import Path

import torch
from projectaria_eyetracking.inference import infer as infer_module
from projectaria_eyetracking.inference.infer import EyeGazeInference


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.cat([out["main"], out["lower"], out["upper"]], dim=1)


def main(out_path: str, weights_dir: str = None):
    # pip install del zip NO incluye los .pth/.yaml (MANIFEST.in solo aplica a
    # sdist) — pasar el dir de pesos del clon del repo como 2o argumento
    if weights_dir:
        weights_dir = Path(weights_dir)
    else:
        weights_dir = (Path(infer_module.__file__).parent
                       / "model" / "pretrained_weights"
                       / "social_eyes_uncertainty_v1")
    assert (weights_dir / "weights.pth").exists(), f"sin pesos en {weights_dir}"
    inferencer = EyeGazeInference(
        str(weights_dir / "weights.pth"),
        str(weights_dir / "config.yaml"),
        device="cpu",
    )
    model = inferencer.model
    model.eval()

    wrapper = ExportWrapper(model)
    dummy = torch.zeros(1, 2, 240, 320)

    with torch.no_grad():
        reference = wrapper(dummy)
    assert reference.shape == (1, 6), f"salida inesperada: {reference.shape}"

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["eye_images"],
        output_names=["gaze_predictions"],
        opset_version=17,
    )
    print(f"ONNX exportado: {out_path} (salida verificada {tuple(reference.shape)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/gaze.onnx",
         sys.argv[2] if len(sys.argv) > 2 else None)
