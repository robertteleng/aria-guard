#!/usr/bin/env python3
"""
Exporta modelos a TensorRT para máximo rendimiento.

Modelos:
1. YOLO26s (detección de objetos)
2. Depth Anything V2 Small (estimación de profundidad)

Uso:
    python scripts/export_tensorrt.py          # Exporta ambos
    python scripts/export_tensorrt.py yolo     # Solo YOLO
    python scripts/export_tensorrt.py depth    # Solo Depth
"""

import sys
from pathlib import Path

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


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         ARIA Demo - TensorRT Export                      ║")
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
