"""
TimoitaseBG - Configuración global y rutas del sistema.
"""
import os
import sys
from typing import Optional
from pathlib import Path

# Directorio raíz de la aplicación
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Directorio para modelos locales autocontenidos
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SEGURIDAD Y ACELERACIÓN POR HARDWARE (GPU)
# ==============================================================================
# Flag maestro de seguridad para aceleración GPU.
# Validado de forma aislada y seguro contra cuelgues y fugas de VRAM.
ENABLE_GPU_ACCELERATION: bool = True

# Control dinámico de modo: False permite GPU (si está disponible); True fuerza CPU.
FORCE_CPU_ONLY: bool = False

import threading
import logging

logger = logging.getLogger("TimoitaseBG.Config")

_best_dml_device_id: Optional[int] = None
_dml_probe_lock = threading.Lock()


def _probe_dml_device_worker(model_path_str: str, device_id: int) -> None:
    """
    Sonda aislada en subproceso independiente para verificar si DirectML
    responde en un device_id específico.
    Se ejecuta fuera del proceso principal para proteger el sistema ante cuelgues D3D12/TDR.
    Libera explícitamente la sesión y fuerza recolección de basura antes de salir.
    """
    import sys
    import gc
    session = None
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(
            model_path_str,
            providers=[("DmlExecutionProvider", {"device_id": device_id})],
        )
        # Éxito: liberar sesión y memoria VRAM de inmediato
        del session
        gc.collect()
        sys.exit(0)
    except Exception:
        if session is not None:
            del session
        gc.collect()
        sys.exit(1)


_cached_gpu_name: Optional[str] = None

def get_gpu_name() -> str:
    """Obtiene el nombre de la GPU dedicada activa (ej. NVIDIA GeForce RTX 5050) si está disponible."""
    global _cached_gpu_name
    if _cached_gpu_name is not None:
        return _cached_gpu_name
    # 1. Probar nvidia-smi con timeout prudente de 3 segundos
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        if out:
            _cached_gpu_name = out.splitlines()[0].strip()
            return _cached_gpu_name
    except Exception:
        pass

    # 2. Respaldo directo en Windows mediante Win32_VideoController
    try:
        import subprocess
        cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        for line in out.splitlines():
            line = line.strip()
            name_lower = line.lower()
            if any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon")):
                _cached_gpu_name = line
                return _cached_gpu_name
    except Exception:
        pass

    _cached_gpu_name = ""
    return ""


def get_best_dml_device_id() -> int:
    """
    En portátiles con GPU dual (Intel integrada + NVIDIA dedicada),
    DirectML suele asignar device_id=0 a la integrada y device_id=1 a la dedicada.
    Detecta automáticamente el device_id correspondiente a la GPU dedicada
    inspeccionando el orden en Win32_VideoController o la presencia de NVIDIA.
    """
    global _best_dml_device_id

    # Si la aceleración GPU está desactivada o forzada a CPU, retornar 0
    if FORCE_CPU_ONLY or not ENABLE_GPU_ACCELERATION:
        return 0

    if _best_dml_device_id is not None:
        return _best_dml_device_id

    with _dml_probe_lock:
        if _best_dml_device_id is not None:
            return _best_dml_device_id

        # 1. Intentar detectar directamente por el orden en Win32_VideoController
        try:
            import subprocess
            cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            names = [line.strip() for line in out.splitlines() if line.strip()]
            for idx, name in enumerate(names):
                name_lower = name.lower()
                is_igpu = any(k in name_lower for k in ("intel", "uhd", "iris"))
                is_dgpu = any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon"))
                if is_dgpu and not is_igpu:
                    logger.info(f"GPU dedicada detectada en Win32_VideoController: '{name}' (DirectML device_id={idx})")
                    _best_dml_device_id = idx
                    return _best_dml_device_id
        except Exception as e:
            logger.warning(f"Error consultando Win32_VideoController para DirectML: {e}")

        # 2. Si nvidia-smi responde pero WMI falló, usar 1 por defecto en laptops duales
        gpu_name = get_gpu_name()
        if gpu_name:
            _best_dml_device_id = 1
            return _best_dml_device_id

        _best_dml_device_id = 0
        return _best_dml_device_id


# Configuración de rendimiento y aceleración
def is_gpu_available() -> bool:
    """Retorna True si el sistema cuenta con GPU y proveedor compatible (CUDA o DirectML)."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return ("CUDAExecutionProvider" in providers or "DmlExecutionProvider" in providers)
    except Exception:
        return False


def set_gpu_acceleration(enabled: bool) -> bool:
    """Activa o desactiva dinámicamente la aceleración por GPU."""
    global FORCE_CPU_ONLY
    FORCE_CPU_ONLY = not enabled
    return not FORCE_CPU_ONLY


def detect_device() -> str:
    """Detecta el dispositivo de aceleración GPU (NVIDIA / DirectML / CPU)."""
    if FORCE_CPU_ONLY or not ENABLE_GPU_ACCELERATION:
        return "CPU"

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_name = get_gpu_name()
        if "CUDAExecutionProvider" in providers or "DmlExecutionProvider" in providers:
            if gpu_name:
                clean_name = gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
                return f"GPU: {clean_name}"
            return "GPU (DirectML)"
    except Exception:
        pass
    return "CPU"


def get_inference_providers():
    """Retorna la lista ordenada de proveedores de ejecución para ONNX Runtime priorizando la GPU dedicada."""
    if FORCE_CPU_ONLY or not ENABLE_GPU_ACCELERATION:
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        selected = []
        if "CUDAExecutionProvider" in avail:
            selected.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in avail:
            dml_id = get_best_dml_device_id()
            selected.append(("DmlExecutionProvider", {"device_id": dml_id}))
        selected.append("CPUExecutionProvider")
        return selected
    except Exception:
        return ["CPUExecutionProvider"]

# Parámetros por defecto de la aplicación
APP_NAME = "TimoitaseBG"
APP_VERSION = "1.0.0"
APP_DEVELOPER = "Eliather"
DEFAULT_WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION} - by {APP_DEVELOPER}"
WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 700

# Parámetros del canvas y pincel
MAX_HISTORY_STATES = 25
DEFAULT_BRUSH_SIZE = 30
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 200
CHECKER_SIZE = 16

# Opciones de modelos para Quitar Fondo
BG_REMOVER_MODELS = [
    ("isnet-anime", "IS-Net Anime (Óptimo para Anime, Manga e Ilustración)"),
    ("u2net_human_seg", "U2-Net Human (Detector de Personas y Ropa — Especial Playeras Blancas)"),
    ("birefnet-general", "BiRefNet (Alta Definición General y Bordes Complejos)"),
    ("inspyrenet", "InSPyReNet Swin-B (Fotografía y Retratos Reales — Anti-aliasing Subpíxel)"),
]
DEFAULT_BG_MODEL = "isnet-anime"

# Opciones de escala para Restaurar
RESTORE_SCALES = [
    ("x2", "Escalar 2x"),
    ("x4", "Escalar 4x (Máxima resolución)"),
    ("enhance_only", "Solo nitidez (Sin cambiar tamaño)"),
]
