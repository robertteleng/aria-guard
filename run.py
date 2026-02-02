#!/usr/bin/env python3
"""Entry point for ARIA Demo."""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.web.main import app, process_loop
import threading


if __name__ == '__main__':
    # Parsear source desde argumentos
    source = sys.argv[1] if len(sys.argv) > 1 else "webcam"

    # Resolve video path if relative
    if source not in ("webcam", "aria"):
        video_path = Path(source)
        if not video_path.is_absolute():
            # Check in data/ directory first
            data_path = Path(__file__).parent / "data" / source
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
