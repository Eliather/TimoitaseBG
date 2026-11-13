import argparse
import sys
from pathlib import Path

from app.config import MODELS_DIR

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Por favor instala onnxruntime: pip install onnxruntime")
    sys.exit(1)


def validate_quantized_model(output_path: Path) -> bool:
    """Valida que el archivo cuantizado pueda inicializarse en ONNX Runtime CPU."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(str(output_path), opts, providers=["CPUExecutionProvider"])
        return True
    except Exception as exc:
        print(f"Validación fallida para {output_path.name}: {exc}")
        return False


def quantize_model(model_path: str):
    """
    Cuantiza dinámicamente un modelo ONNX de float32 a uint8.
    Reduce el tamaño a 1/4 y mejora la velocidad en CPU sin requerir calibración.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: No se encontro el modelo: {model_path}")
        return False

    if model_path.stem.endswith("_int8"):
        print(f"Aviso: El modelo ya parece estar cuantizado: {model_path.name}")
        return False

    output_path = model_path.with_name(f"{model_path.stem}_int8.onnx")

    print(f"Cuantizando {model_path.name}...")
    size_before = model_path.stat().st_size / (1024 * 1024)

    try:
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8
        )
        if not validate_quantized_model(output_path):
            output_path.unlink(missing_ok=True)
            return False

        size_after = output_path.stat().st_size / (1024 * 1024)
        print(f"Exito! Guardado y validado como {output_path.name}")
        print(f"Reduccion de tamano: {size_before:.1f} MB -> {size_after:.1f} MB")
        return True
    except Exception as e:
        print(f"Error al cuantizar {model_path.name}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cuantiza modelos ONNX locales a INT8")
    parser.add_argument("--dir", default=str(MODELS_DIR), help="Directorio donde están los modelos .onnx")
    args = parser.parse_args()

    models_dir = Path(args.dir)
    if not models_dir.exists():
        print(f"El directorio {models_dir} no existe.")
        sys.exit(1)

    print(f"Buscando modelos en {models_dir}...")
    for file in models_dir.glob("*.onnx"):
        if not file.name.endswith("_int8.onnx"):
            quantize_model(str(file))
