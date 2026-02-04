#!/usr/bin/env python3
"""
Entry point for ARIA Demo.

CRITICAL: This wrapper sets multiprocessing spawn method BEFORE importing
any torch/CUDA modules. This prevents CUDA context conflicts with Aria SDK.
Pattern borrowed from aria-nav.

The order is critical:
1. Set mp.set_start_method('spawn') FIRST
2. Only then import modules that load torch/CUDA
3. Then run the application
"""
import os
import sys
import tempfile
from pathlib import Path

# CRITICAL: Disable ALL CUDA in main process BEFORE any imports
# cv2.cuda and numba.cuda conflict with Aria SDK's FastDDS
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from main process
os.environ["NUMBA_DISABLE_CUDA"] = "1"   # Disable numba CUDA

# Force TMPDIR to /home to avoid disk space issues on /
# NeMo uses tempfile module directly, so we must patch tempfile.tempdir too
_tmp_dir = Path.home() / "tmp"
_tmp_dir.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_tmp_dir)
os.environ["TEMP"] = str(_tmp_dir)
os.environ["TMP"] = str(_tmp_dir)
tempfile.tempdir = str(_tmp_dir)

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == '__main__':
    # CRITICAL: Set spawn method BEFORE any torch/CUDA imports
    # Use standard multiprocessing (NOT torch.multiprocessing) to avoid loading torch in main process
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Match aria-nav pattern for Qt
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # NOW safe to import modules that load torch/CUDA
    from src.web.main import app, process_loop
    import threading

    # Parsear source desde argumentos
    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    # Check for dataset source (VRS files)
    if source == "dataset":
        # Use default sample dataset
        vrs_path = PROJECT_ROOT / "data" / "aria_sample" / "sample.vrs"
        gaze_csv = PROJECT_ROOT / "data" / "aria_sample" / "eye_gaze" / "general_eye_gaze.csv"
        if not vrs_path.exists():
            print(f"[ERROR] Dataset no encontrado: {vrs_path}")
            print("        Descarga con: curl -L -o data/aria_sample/sample.vrs ...")
            sys.exit(1)
        source = f"dataset:{vrs_path}:{gaze_csv}"
    elif source.endswith(".vrs"):
        # Direct VRS path
        vrs_path = Path(source)
        if not vrs_path.is_absolute():
            data_path = PROJECT_ROOT / "data" / source
            if data_path.exists():
                vrs_path = data_path
        # Look for matching gaze CSV
        gaze_csv = vrs_path.parent / "eye_gaze" / "general_eye_gaze.csv"
        if gaze_csv.exists():
            source = f"dataset:{vrs_path}:{gaze_csv}"
        else:
            source = f"dataset:{vrs_path}:"
    elif source not in ("webcam", "aria", "aria:usb", "realsense") and not source.startswith("aria:wifi"):
        # Regular video file
        video_path = Path(source)
        if not video_path.is_absolute():
            data_path = PROJECT_ROOT / "data" / source
            if data_path.exists():
                source = str(data_path)
            elif not video_path.exists():
                print(f"[ERROR] Video no encontrado: {source}")
                print(f"        Busqué en: {video_path.absolute()}")
                print(f"        Y en: {data_path}")
                sys.exit(1)

    print()
    print("╔══════════════════════════════════════╗")
    print("║          ARIA DEMO v1.0              ║")
    print("║   Visual Assistance System           ║")
    print("╚══════════════════════════════════════╝")
    print()

    if source.startswith("dataset:"):
        print(f"  Fuente: Aria Dataset (VRS + Eye Gaze)")
    elif source == "aria" or source == "aria:usb":
        print(f"  Fuente: Aria Glasses (USB)")
    elif source.startswith("aria:wifi"):
        print(f"  Fuente: Aria Glasses (WiFi)")
    elif source == "realsense":
        print(f"  Fuente: Intel RealSense D435 (RGB + Depth)")
    else:
        print(f"  Fuente: {source}")
    print()
    print("  Selecciona el modo de detección:")
    print()
    print("    [1] Indoor  - persona, silla, sofá, mesa, tv, puerta...")
    print("    [2] Outdoor - persona, coche, bici, moto, bus, semáforo...")
    print("    [3] All     - todas las clases (80 objetos)")
    print()

    # Check for --no-tts flag
    enable_audio = "--no-tts" not in sys.argv
    if "--no-tts" in sys.argv:
        sys.argv.remove("--no-tts")
        print("  [TTS desactivado]")

    mode = None
    if len(sys.argv) > 2:
        m = sys.argv[2].lower()
        if m in ["1", "indoor"]: mode = "indoor"
        elif m in ["2", "outdoor"]: mode = "outdoor"
        elif m in ["3", "all"]: mode = "all"

    if mode is None:
        while True:
            choice = input("  Modo [1/2/3]: ").strip()
            if choice == "1":
                mode = "indoor"
                break
            elif choice == "2":
                mode = "outdoor"
                break
            elif choice == "3":
                mode = "all"
                break
            else:
                print("  Opción no válida. Introduce 1, 2 o 3.")

    print()
    print(f"  → Modo seleccionado: {mode.upper()}")
    print()

    # Iniciar procesamiento en background
    thread = threading.Thread(target=process_loop, args=(source, mode, enable_audio), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
