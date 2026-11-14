"""
Utilidad para calcular y verificar digests SHA-256 de los modelos ONNX locales.
"""
import argparse
import hashlib
import sys
from pathlib import Path

from app.config import MODELS_DIR
from app.core.model_fetcher import MODEL_SHA256, MODEL_URLS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calcula o verifica SHA-256 de modelos ONNX en MODELS_DIR.")
    parser.add_argument("--dir", default=str(MODELS_DIR), help="Directorio de modelos a inspeccionar.")
    parser.add_argument("--verify", action="store_true", help="Verifica los archivos presentes contra MODEL_SHA256.")
    args = parser.parse_args()

    models_dir = Path(args.dir)
    if not models_dir.is_dir():
        print(f"Directorio no encontrado: {models_dir}")
        return 1

    had_mismatch = False
    for fname in MODEL_URLS:
        target = models_dir / fname
        if not target.is_file():
            print(f"[MISSING] {fname}")
            continue
        digest = sha256_file(target)
        expected = MODEL_SHA256.get(fname)
        if args.verify and expected:
            ok = digest == expected.lower()
            status = "OK" if ok else "MISMATCH"
            if not ok:
                had_mismatch = True
            print(f"[{status}] {fname}: {digest}")
        else:
            print(f"{fname}: {digest}")

    return 1 if had_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
