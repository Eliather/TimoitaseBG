"""
Cuantización dinámica a INT8 y validación de sesiones ONNX Runtime.
"""
import logging
from pathlib import Path

logger = logging.getLogger("TimoitaseBG.Quantizer")


def validate_quantized_model(output_path: Path) -> bool:
    """Valida que el archivo cuantizado pueda inicializarse en ONNX Runtime CPU."""
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(str(output_path), opts, providers=["CPUExecutionProvider"])
        return True
    except Exception as exc:
        logger.warning(f"Validación fallida para {output_path.name}: {exc}")
        return False


def quantize_model(model_path: str | Path) -> bool:
    """
    Cuantiza dinámicamente un modelo ONNX de float32 a uint8 y valida su carga.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.error("onnxruntime.quantization no está disponible.")
        return False

    path = Path(model_path)
    if not path.exists():
        logger.warning(f"Modelo no encontrado para cuantizar: {path}")
        return False
    if path.stem.endswith("_int8"):
        logger.warning(f"El modelo ya está cuantizado: {path.name}")
        return False

    output_path = path.with_name(f"{path.stem}_int8.onnx")
    try:
        quantize_dynamic(
            model_input=str(path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
        )
        if not validate_quantized_model(output_path):
            output_path.unlink(missing_ok=True)
            return False
        return True
    except Exception as exc:
        logger.error(f"Error al cuantizar {path.name}: {exc}")
        output_path.unlink(missing_ok=True)
        return False
