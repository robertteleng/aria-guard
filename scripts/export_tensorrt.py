#!/usr/bin/env python3
"""
Exporta modelos a TensorRT para máximo rendimiento.

Modelos:
1. YOLO26s (detección de objetos)
2. Depth Anything V2 Small (estimación de profundidad)
3. Meta Eye Gaze (estimación de mirada - projectaria_eyetracking)

Uso:
    python scripts/export_tensorrt.py          # Exporta todos
    python scripts/export_tensorrt.py yolo     # Solo YOLO
    python scripts/export_tensorrt.py depth    # Solo Depth
    python scripts/export_tensorrt.py gaze     # Solo Eye Gaze
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def export_yolo():
    """Exporta YOLO26s a TensorRT usando Ultralytics."""
    from ultralytics import YOLO

    print("=" * 60)
    print("Exportando YOLO26s a TensorRT")
    print("=" * 60)

    pt_path = MODELS_DIR / "yolo26s.pt"
    engine_path = MODELS_DIR / "yolo26s.engine"

    if engine_path.exists():
        print(f"Engine ya existe: {engine_path}")
        return engine_path

    # Download model if not present
    print("[1/2] Cargando modelo YOLO26s...")
    model = YOLO(str(pt_path))

    # Export to TensorRT
    print("[2/2] Exportando a TensorRT (puede tardar varios minutos)...")
    model.export(
        format="engine",
        imgsz=640,
        half=True,  # FP16
        device=0,
        simplify=True,
        workspace=4,  # GB
    )

    # Ultralytics saves next to .pt file
    exported = pt_path.with_suffix(".engine")
    if exported.exists():
        print(f"✓ Engine guardado: {exported} ({exported.stat().st_size / 1e6:.1f} MB)")
        return exported
    else:
        print("✗ Error: Engine no se generó")
        return None


def export_depth():
    """Exporta Depth Anything V2 a TensorRT."""
    import tensorrt as trt

    print()
    print("=" * 60)
    print("Exportando Depth Anything V2 Small a TensorRT")
    print("=" * 60)

    # Use pre-exported ONNX from fabio-sim/Depth-Anything-ONNX (static shapes, TensorRT compatible)
    onnx_path = MODELS_DIR / "depth_anything_v2_vits.onnx"
    engine_path = MODELS_DIR / "depth_anything_v2_vits.engine"
    input_size = 518

    if engine_path.exists():
        print(f"Engine ya existe: {engine_path}")
        return engine_path

    # Check for pre-downloaded ONNX
    if not onnx_path.exists():
        print(f"[ERROR] ONNX no encontrado: {onnx_path}")
        print("Descarga el modelo pre-exportado:")
        print("  curl -L -o models/depth_anything_v2_vits.onnx \\")
        print("    https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx")
        return None

    print(f"[1/2] ONNX encontrado: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Step 2: Convert to TensorRT
    print(f"[2/2] Convirtiendo a TensorRT...")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("    FP16 habilitado")

    print("    Construyendo engine (puede tardar unos minutos)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"✓ Engine guardado: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return engine_path


class GazeModelWrapper(nn.Module):
    """Wrapper para exportar Meta Eye Gaze a ONNX/TensorRT.

    El modelo original usa SplitAndConcat en el head que opera sobre dim=0 (batch),
    lo cual funciona en PyTorch pero es problemático para ONNX con batch estático.
    Este wrapper encapsula el modelo completo para que la exportación ONNX sea limpia.

    Input:  (1, 2, 240, 320) — imagen preprocesada (ojo izq + derecho como canales)
    Output: (1, 6) — [main_yaw, main_pitch, lower_yaw, lower_pitch, upper_yaw, upper_pitch]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        preds = self.model(x)
        # Concatenar los 3 outputs en un solo tensor para ONNX/TensorRT
        return torch.cat([preds["main"], preds["lower"], preds["upper"]], dim=1)


def _load_gaze_pytorch():
    """Carga el modelo PyTorch de Meta Eye Gaze y lo devuelve como wrapper ONNX-ready."""
    weights_dir = MODELS_DIR / "gaze_weights" / "social_eyes_uncertainty_v1"
    weights_path = weights_dir / "weights.pth"
    config_path = weights_dir / "config.yaml"

    if not weights_path.exists():
        print(f"[ERROR] Pesos no encontrados: {weights_path}")
        print("Descarga desde projectaria_eyetracking o copia manualmente.")
        return None

    # Monkey-patch torch.load para compatibilidad con PyTorch 2.6+ (EasyDict en checkpoint)
    _original_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(*args, **kwargs)

    try:
        torch.load = _patched_load
        from projectaria_eyetracking.inference.infer import EyeGazeInference

        gaze = EyeGazeInference(
            model_checkpoint_path=str(weights_path),
            model_config_path=str(config_path),
            device="cpu",
        )
        wrapper = GazeModelWrapper(gaze.model)
        wrapper.eval()
        print(f"    Modelo cargado desde {weights_dir.name}")
        return wrapper
    finally:
        torch.load = _original_torch_load


def _gaze_to_onnx(wrapper):
    """Exporta el wrapper gaze a ONNX. Devuelve path del ONNX."""
    onnx_path = MODELS_DIR / "gaze.onnx"

    if onnx_path.exists():
        print(f"    ONNX ya existe: {onnx_path}")
        return onnx_path

    # Input: imagen preprocesada (1, 2, 240, 320) — 2 canales: ojo izq/derecho
    dummy_input = torch.randn(1, 2, 240, 320)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    print(f"    ONNX guardado: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def _onnx_to_tensorrt(onnx_path, engine_path, workspace_gb=1):
    """Convierte un ONNX a TensorRT engine con FP16."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("    FP16 habilitado")

    print("    Construyendo engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"    Engine guardado: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return engine_path


def export_gaze():
    """Exporta Meta Eye Gaze model a ONNX y TensorRT.

    El modelo viene de projectaria_eyetracking (Meta) y usa ResNet-18
    para estimar yaw/pitch de la mirada a partir de imágenes de eye tracking
    de las gafas Aria.

    Pipeline: .pth → ONNX (PyTorch) → TensorRT engine (requiere tensorrt)
    Si tensorrt no está instalado, se exporta solo a ONNX.
    """
    print()
    print("=" * 60)
    print("Exportando Meta Eye Gaze a TensorRT")
    print("=" * 60)

    onnx_path = MODELS_DIR / "gaze.onnx"
    engine_path = MODELS_DIR / "gaze.engine"

    if engine_path.exists():
        print(f"Engine ya existe: {engine_path}")
        return engine_path

    # Step 1: PyTorch → ONNX
    if not onnx_path.exists():
        print("[1/2] Cargando modelo PyTorch y exportando a ONNX...")
        wrapper = _load_gaze_pytorch()
        if wrapper is None:
            return None
        onnx_path = _gaze_to_onnx(wrapper)
    else:
        print(f"[1/2] ONNX ya existe: {onnx_path}")

    # Step 2: ONNX → TensorRT
    print("[2/2] Convirtiendo a TensorRT...")
    try:
        engine_path = _onnx_to_tensorrt(onnx_path, engine_path)
    except ImportError:
        print("    [WARN] tensorrt no instalado. ONNX listo para convertir en Docker:")
        print(f"    trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")
        return onnx_path

    # Limpiar ONNX intermedio
    onnx_path.unlink()
    print(f"    ONNX intermedio eliminado")

    print(f"✓ Engine guardado: {engine_path}")
    return engine_path


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         ARIA Guard - TensorRT Export                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    results = {}

    if "all" in args or "yolo" in args:
        try:
            results["yolo"] = export_yolo()
        except Exception as e:
            print(f"✗ Error exportando YOLO: {e}")
            results["yolo"] = None

    if "all" in args or "depth" in args:
        try:
            results["depth"] = export_depth()
        except Exception as e:
            print(f"✗ Error exportando Depth: {e}")
            results["depth"] = None

    if "all" in args or "gaze" in args:
        try:
            results["gaze"] = export_gaze()
        except Exception as e:
            print(f"✗ Error exportando Gaze: {e}")
            import traceback
            traceback.print_exc()
            results["gaze"] = None

    # Summary
    print()
    print("=" * 60)
    print("Resumen:")
    print("=" * 60)
    for name, path in results.items():
        status = f"✓ {path}" if path else "✗ Error"
        print(f"  {name.upper()}: {status}")
    print()


if __name__ == "__main__":
    main()
