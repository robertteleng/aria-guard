#!/usr/bin/env python3
"""
Descarga los pesos del modelo Meta Eye Gaze para projectaria_eyetracking.

El paquete pip no incluye los pesos .pth porque GitHub usa LFS para archivos grandes.
Este script descarga los pesos directamente desde el repositorio.

Uso:
    python scripts/download_gaze_weights.py
"""

import os
import sys
import urllib.request
from pathlib import Path

# URLs de GitHub raw para los archivos de pesos
GITHUB_RAW_BASE = "https://github.com/facebookresearch/projectaria_eyetracking/raw/main"
WEIGHTS_PATH = "projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1"

FILES = [
    ("config.yaml", 247),      # bytes
    ("weights.pth", 11_300_000),  # ~11.3 MB
]


def find_package_path():
    """Encuentra la ruta de instalación del paquete projectaria_eyetracking."""
    try:
        import projectaria_eyetracking.inference.infer as infer_module
        infer_dir = os.path.dirname(infer_module.__file__)
        return Path(infer_dir) / "model" / "pretrained_weights" / "social_eyes_uncertainty_v1"
    except ImportError:
        print("ERROR: projectaria_eyetracking no está instalado")
        print("Instala con: pip install git+https://github.com/facebookresearch/projectaria_eyetracking.git")
        return None


def download_file(url: str, dest: Path, expected_size: int = 0):
    """Descarga un archivo con barra de progreso."""
    print(f"  Descargando {dest.name}...", end=" ", flush=True)

    try:
        urllib.request.urlretrieve(url, dest)
        actual_size = dest.stat().st_size

        # Verificar que no sea un archivo LFS pointer
        if actual_size < 1000 and expected_size > 1000:
            with open(dest, 'r') as f:
                content = f.read(100)
                if 'git-lfs' in content or 'oid sha256' in content:
                    print("ERROR: Archivo LFS pointer descargado")
                    print(f"  El archivo real está en Git LFS, no en raw GitHub")
                    dest.unlink()
                    return False

        print(f"OK ({actual_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_via_git_lfs(dest_dir: Path):
    """Clona el repo con LFS para obtener los pesos reales."""
    import subprocess
    import tempfile
    import shutil

    print("  Intentando clonar con Git LFS...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Clone con LFS
            subprocess.run([
                "git", "lfs", "install"
            ], check=True, capture_output=True)

            subprocess.run([
                "git", "clone", "--depth", "1", "--filter=blob:none",
                "https://github.com/facebookresearch/projectaria_eyetracking.git",
                tmpdir
            ], check=True, capture_output=True)

            # Descargar solo los archivos LFS necesarios
            subprocess.run([
                "git", "-C", tmpdir, "lfs", "pull",
                "--include", f"{WEIGHTS_PATH}/*"
            ], check=True, capture_output=True)

            # Copiar archivos
            src_path = Path(tmpdir) / WEIGHTS_PATH
            if src_path.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                for file in src_path.iterdir():
                    shutil.copy2(file, dest_dir / file.name)
                    print(f"  Copiado: {file.name} ({(dest_dir / file.name).stat().st_size / 1e6:.1f} MB)")
                return True

        except subprocess.CalledProcessError as e:
            print(f"  Git LFS falló: {e}")
        except FileNotFoundError:
            print("  Git LFS no está instalado")

    return False


def main():
    print()
    print("=" * 60)
    print("Descarga de pesos Meta Eye Gaze")
    print("=" * 60)
    print()

    # Encontrar directorio de destino
    dest_dir = find_package_path()
    if dest_dir is None:
        return 1

    print(f"Directorio destino: {dest_dir}")

    # Verificar si ya existen
    weights_file = dest_dir / "weights.pth"
    if weights_file.exists() and weights_file.stat().st_size > 1_000_000:
        print(f"Los pesos ya existen ({weights_file.stat().st_size / 1e6:.1f} MB)")
        return 0

    # Crear directorio
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Método 1: Intentar descarga directa (puede fallar si es LFS)
    print("\nMétodo 1: Descarga directa desde GitHub...")
    all_ok = True
    for filename, expected_size in FILES:
        url = f"{GITHUB_RAW_BASE}/{WEIGHTS_PATH}/{filename}"
        dest = dest_dir / filename
        if not download_file(url, dest, expected_size):
            all_ok = False
            break

    if all_ok:
        # Verificar que weights.pth no sea un LFS pointer
        weights_size = (dest_dir / "weights.pth").stat().st_size
        if weights_size > 1_000_000:
            print("\n✓ Pesos descargados correctamente")
            return 0

    # Método 2: Git LFS clone
    print("\nMétodo 2: Clonando repositorio con Git LFS...")
    if download_via_git_lfs(dest_dir):
        print("\n✓ Pesos descargados via Git LFS")
        return 0

    # Método 3: Instrucciones manuales
    print("\n" + "=" * 60)
    print("DESCARGA MANUAL REQUERIDA")
    print("=" * 60)
    print(f"""
Los pesos están almacenados en Git LFS y no se pueden descargar automáticamente.

Opción A - Clonar el repositorio:
    git lfs install
    git clone https://github.com/facebookresearch/projectaria_eyetracking.git /tmp/aria_gaze
    cp /tmp/aria_gaze/{WEIGHTS_PATH}/* {dest_dir}/

Opción B - Descargar desde navegador:
    1. Visita: https://github.com/facebookresearch/projectaria_eyetracking
    2. Navega a: {WEIGHTS_PATH}/
    3. Descarga weights.pth y config.yaml
    4. Copia a: {dest_dir}/
""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
