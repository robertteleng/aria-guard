#!/usr/bin/env python3
"""Entry point for ARIA Demo."""
import os
import sys
import tempfile
from pathlib import Path

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

from src.web.main import app, process_loop
import threading


if __name__ == '__main__':
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
    elif source not in ("webcam", "aria"):
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
    else:
        print(f"  Fuente: {source}")
    print()
    print("  Selecciona el modo de detección:")
    print()
    print("    [1] Indoor  - persona, silla, sofá, mesa, tv, puerta...")
    print("    [2] Outdoor - persona, coche, bici, moto, bus, semáforo...")
    print("    [3] All     - todas las clases (80 objetos)")
    print()

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
    thread = threading.Thread(target=process_loop, args=(source, mode), daemon=True)
    thread.start()

    print("Servidor en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
