import argparse
import sys
from pathlib import Path

from app.config import MODELS_DIR
from app.core.quantizer import quantize_model as _quantize_model_core, validate_quantized_model


def quantize_model(model_path: str) -> bool:
    """Cuantiza un modelo ONNX e imprime resumen en consola para CLI."""
    path = Path(model_path)
    if not path.exists():
        print(f"Error: No se encontro el modelo: {path}")
        return False
    if path.stem.endswith("_int8"):
        print(f"Aviso: El modelo ya parece estar cuantizado: {path.name}")
        return False

    size_before = path.stat().st_size / (1024 * 1024)
    print(f"Cuantizando {path.name}...")
    ok = _quantize_model_core(path)
    if not ok:
        print(f"Error al cuantizar o validar {path.name}")
        return False

    out_path = path.with_name(f"{path.stem}_int8.onnx")
    size_after = out_path.stat().st_size / (1024 * 1024)
    print(f"Exito! Guardado y validado como {out_path.name}")
    print(f"Reduccion de tamano: {size_before:.1f} MB -> {size_after:.1f} MB")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cuantiza los modelos ONNX de TimoitaseBG a INT8.")
    parser.add_argument("--dir", type=str, default=str(MODELS_DIR), help="Directorio donde buscar archivos .onnx")
    args = parser.parse_args()

    models_dir = Path(args.dir)
    if not models_dir.exists():
        print(f"Directorio {models_dir} no encontrado.")
        sys.exit(0)

    for file in models_dir.glob("*.onnx"):
        if not file.stem.endswith("_int8"):
            quantize_model(str(file))
