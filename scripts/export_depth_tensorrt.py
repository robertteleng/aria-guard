#!/usr/bin/env python3
"""
Exporta Depth Anything V2 Small a TensorRT.

Pasos:
1. Descarga modelo de HuggingFace
2. Exporta a ONNX
3. Convierte a TensorRT engine

Uso:
    python scripts/export_depth_tensorrt.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

ONNX_PATH = MODELS_DIR / "depth_anything_v2_vits.onnx"
ENGINE_PATH = MODELS_DIR / "depth_anything_v2_vits.engine"

# Input size for Depth Anything V2
INPUT_SIZE = 518


def export_to_onnx():
    """Export HuggingFace model to ONNX."""
    import torch
    from transformers import AutoModelForDepthEstimation

    print("[1/3] Cargando modelo de HuggingFace...")
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()
    model.cuda()

    print(f"[2/3] Exportando a ONNX ({ONNX_PATH})...")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).cuda()

    # Use dynamo=False to avoid external data files and onnxscript issues
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "predicted_depth": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"    ONNX guardado: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


def convert_to_tensorrt():
    """Convert ONNX to TensorRT engine."""
    import tensorrt as trt

    print(f"[3/3] Convirtiendo a TensorRT ({ENGINE_PATH})...")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Add optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", (1, 3, INPUT_SIZE, INPUT_SIZE), (1, 3, INPUT_SIZE, INPUT_SIZE), (1, 3, INPUT_SIZE, INPUT_SIZE))
    config.add_optimization_profile(profile)

    # Enable FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("    FP16 habilitado")

    # Build engine
    print("    Construyendo engine (puede tardar unos minutos)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)

    print(f"    Engine guardado: {ENGINE_PATH} ({ENGINE_PATH.stat().st_size / 1e6:.1f} MB)")


def verify_engine():
    """Verify the TensorRT engine works."""
    import numpy as np
    import tensorrt as trt
    import torch

    print("\n[Verificación] Probando engine...")

    logger = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Create dummy input
    dummy_input = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    input_tensor = torch.from_numpy(dummy_input).cuda()

    # Get output shape
    output_name = engine.get_tensor_name(1)
    output_shape = tuple(engine.get_tensor_shape(output_name))
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device="cuda")

    # Set addresses
    context.set_tensor_address("pixel_values", input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    # Execute
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {tuple(output_tensor.shape)}")
    print("    Engine funciona correctamente!")


def main():
    print("=" * 60)
    print("Exportando Depth Anything V2 Small a TensorRT")
    print("=" * 60)
    print()

    if not ONNX_PATH.exists():
        export_to_onnx()
    else:
        print(f"[1/3] ONNX ya existe: {ONNX_PATH}")

    if not ENGINE_PATH.exists():
        convert_to_tensorrt()
    else:
        print(f"[3/3] Engine ya existe: {ENGINE_PATH}")

    verify_engine()

    print()
    print("=" * 60)
    print("Exportación completada!")
    print(f"Engine: {ENGINE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
