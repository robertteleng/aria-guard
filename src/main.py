#!/usr/bin/env python3
"""
Entry point for ARIA Guard.

CRITICAL: This wrapper sets multiprocessing spawn method BEFORE importing
any torch/CUDA modules. This prevents CUDA context conflicts with Aria SDK.

The order is critical:
1. Set mp.set_start_method('spawn') FIRST
2. Only then import modules that load torch/CUDA
3. Then run the application
"""
import faulthandler
import os
import sys
import tempfile
from pathlib import Path

# Capturar segfaults (FastDDS/CUDA crashes) con backtrace completo
faulthandler.enable()

# CRITICAL: Disable ALL CUDA in main process BEFORE any imports
# cv2.cuda and numba.cuda conflict with Aria SDK's FastDDS
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs from main process
os.environ["NUMBA_DISABLE_CUDA"] = "1"   # Disable numba CUDA

# Remove LD_PRELOAD so child processes (detector/TTS) don't inherit jemalloc
# jemalloc is already loaded in main process memory by ld.so, stays active for Aria/FastDDS
# But CUDA/PyTorch in child processes can conflict with jemalloc
os.environ.pop("LD_PRELOAD", None)

# Force TMPDIR to /home to avoid disk space issues on /
# NeMo uses tempfile module directly, so we must patch tempfile.tempdir too
_tmp_dir = Path.home() / "tmp"
_tmp_dir.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_tmp_dir)
os.environ["TEMP"] = str(_tmp_dir)
os.environ["TMP"] = str(_tmp_dir)
tempfile.tempdir = str(_tmp_dir)

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == '__main__':
    # CRITICAL: Set spawn method BEFORE any torch/CUDA imports
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # NOW safe to import modules that load torch/CUDA
    from src.web.server import app, state
    from src.web.pipeline import process_loop
    import threading

    # Parsear source desde argumentos
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        print()
        print("====================================")
        print("       ARIA GUARD v1.0")
        print("   Visual Assistance System")
        print("====================================")
        print()
        print("  Selecciona la fuente de entrada:")
        print()
        print("    [1] Webcam")
        print("    [2] Video file")
        print("    [3] Aria Glasses")
        print("    [4] RealSense D435")
        print("    [5] Aria Bridge (Jetson ARM64 via ZMQ)")
        print()
        while True:
            choice = input("  Fuente [1/2/3/4/5]: ").strip()
            if choice == "1":
                source = "webcam"
                break
            elif choice == "2":
                video = input("  Ruta al video (ej: /app/data/test_60fps.mp4): ").strip()
                source = video if video else "/app/data/test_60fps.mp4"
                break
            elif choice == "3":
                print()
                print("    [a] USB")
                print("    [b] WiFi")
                print()
                while True:
                    aria_choice = input("  Conexion [a/b]: ").strip().lower()
                    if aria_choice == "a":
                        source = "aria:usb"
                        break
                    elif aria_choice == "b":
                        default_ip = os.environ.get("ARIA_IP", "")
                        hint = f" (Enter = {default_ip})" if default_ip else ""
                        ip = input(f"  IP de Aria{hint}: ").strip() or default_ip
                        if not ip:
                            print("  Hace falta la IP de las gafas (o define ARIA_IP).")
                            continue
                        source = f"aria:wifi:{ip}"
                        break
                    else:
                        print("  Opcion no valida. Introduce a o b.")
                break
            elif choice == "4":
                source = "realsense"
                break
            elif choice == "5":
                source = "aria:bridge"
                break
            else:
                print("  Opcion no valida. Introduce 1, 2, 3, 4 o 5.")

    # Check for dataset source (VRS files)
    if source == "dataset":
        vrs_path = PROJECT_ROOT / "data" / "aria_sample" / "sample.vrs"
        gaze_csv = PROJECT_ROOT / "data" / "aria_sample" / "eye_gaze" / "general_eye_gaze.csv"
        if not vrs_path.exists():
            print(f"[ERROR] Dataset no encontrado: {vrs_path}")
            sys.exit(1)
        source = f"dataset:{vrs_path}:{gaze_csv}"
    elif source.endswith(".vrs"):
        vrs_path = Path(source)
        if not vrs_path.is_absolute():
            data_path = PROJECT_ROOT / "data" / source
            if data_path.exists():
                vrs_path = data_path
        gaze_csv = vrs_path.parent / "eye_gaze" / "general_eye_gaze.csv"
        if gaze_csv.exists():
            source = f"dataset:{vrs_path}:{gaze_csv}"
        else:
            source = f"dataset:{vrs_path}:"
    elif source not in ("webcam", "aria", "aria:usb", "realsense") and not source.startswith("aria:wifi") and not source.startswith("aria:bridge"):
        video_path = Path(source)
        if not video_path.is_absolute():
            data_path = PROJECT_ROOT / "data" / source
            if data_path.exists():
                source = str(data_path)
            elif not video_path.exists():
                print(f"[ERROR] Video no encontrado: {source}")
                print(f"        Busque en: {video_path.absolute()}")
                print(f"        Y en: {data_path}")
                sys.exit(1)

    # Banner
    if len(sys.argv) > 1:
        print()
        print("====================================")
        print("       ARIA GUARD v1.0")
        print("   Visual Assistance System")
        print("====================================")
        print()

    if source.startswith("dataset:"):
        print(f"  Fuente: Aria Dataset (VRS + Eye Gaze)")
    elif source == "aria" or source == "aria:usb":
        print(f"  Fuente: Aria Glasses (USB)")
    elif source.startswith("aria:wifi"):
        print(f"  Fuente: Aria Glasses (WiFi)")
    elif source == "realsense":
        print(f"  Fuente: Intel RealSense D435 (RGB + Depth)")
    elif source.startswith("aria:bridge"):
        print(f"  Fuente: Aria Bridge (Jetson ARM64)")
    else:
        print(f"  Fuente: {source}")
    print()
    print("  Selecciona el modo de deteccion:")
    print()
    print("    [1] Indoor  - persona, silla, sofa, mesa, tv, puerta...")
    print("    [2] Outdoor - persona, coche, bici, moto, bus, semaforo...")
    print("    [3] All     - todas las clases (24 objetos nav)")
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
                print("  Opcion no valida. Introduce 1, 2 o 3.")

    print()
    print(f"  -> Modo seleccionado: {mode.upper()}")
    print()

    # Iniciar procesamiento en background
    thread = threading.Thread(target=process_loop, args=(source, mode, enable_audio, state), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
