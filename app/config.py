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

def is_arm_architecture() -> bool:
    """Retorna True si el procesador anfitrión es de arquitectura ARM (ARM64, aarch64, armv8, etc.)."""
    import platform
    mach = platform.machine().lower()
    return mach in ("arm64", "aarch64", "armv8l", "armv7l", "arm") or mach.startswith("arm")


def get_gpu_name() -> str:
    """Obtiene el nombre de la GPU dedicada o SoC activo (Apple Silicon, Snapdragon Adreno, Jetson, NVIDIA, AMD, etc.)."""
    global _cached_gpu_name
    if _cached_gpu_name is not None:
        return _cached_gpu_name

    # 1. En macOS (Darwin): Apple Silicon SoC o system_profiler
    if sys.platform == "darwin":
        # a) Probar sysctl machdep.cpu.brand_string (Apple M1-M5 SoC integra CPU/GPU/Neural Engine)
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode().strip()
            if out and "apple" in out.lower():
                _cached_gpu_name = out
                return _cached_gpu_name
        except Exception:
            pass

        # b) Respaldo en Mac Intel / AMD Radeon Pro mediante system_profiler
        try:
            import subprocess
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode().strip()
            for line in out.splitlines():
                if "chipset model:" in line.lower():
                    chip_model = line.split(":")[-1].strip()
                    if chip_model:
                        _cached_gpu_name = chip_model
                        return _cached_gpu_name
        except Exception:
            pass

    # 2. Probar nvidia-smi con timeout prudente de 3 segundos (NVIDIA en Linux y Windows, incluyendo Jetson si está disponible)
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

    # 3. Probar rocm-smi en Linux si nvidia-smi no detectó tarjeta (AMD ROCm en Linux)
    if sys.platform != "win32" and sys.platform != "darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            for line in out.splitlines():
                if any(k in line.lower() for k in ("card", "series", "radeon", "amd")):
                    _cached_gpu_name = line.split(":")[-1].strip()
                    return _cached_gpu_name
        except Exception:
            pass

    # 4. En Windows (x86_64 y ARM64 Snapdragon / Adreno)
    if sys.platform == "win32":
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
                if any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd", "arc", "intel", "adreno", "qualcomm", "snapdragon")):
                    _cached_gpu_name = line
                    return _cached_gpu_name
        except Exception:
            pass

        # Respaldo en Windows ARM64: consultar procesador Snapdragon si la GPU no retornó o para QNN
        if is_arm_architecture():
            try:
                import platform
                proc = platform.processor()
                if proc and any(k in proc.lower() for k in ("snapdragon", "qualcomm")):
                    _cached_gpu_name = proc.strip()
                    return _cached_gpu_name
            except Exception:
                pass
            try:
                import subprocess
                cmd = "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).decode().strip()
                for line in out.splitlines():
                    if any(k in line.lower() for k in ("snapdragon", "qualcomm")):
                        _cached_gpu_name = line.strip()
                        return _cached_gpu_name
            except Exception:
                pass

    elif sys.platform != "darwin":
        # 5. En Linux x86_64 y ARM: lspci
        try:
            import subprocess
            out = subprocess.check_output(
                ["lspci"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            for line in out.splitlines():
                line_lower = line.lower()
                if any(term in line_lower for term in ("vga", "3d controller", "display controller")):
                    if any(k in line_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd", "intel", "arc", "adreno", "qualcomm")):
                        _cached_gpu_name = line.split(":")[-1].strip()
                        return _cached_gpu_name
        except Exception:
            pass

        # 6. En Linux ARM64 (aarch64): SoCs como NVIDIA Jetson, Raspberry Pi, Rockchip
        if is_arm_architecture():
            # a) /proc/device-tree/model
            try:
                from pathlib import Path
                dt_model = Path("/proc/device-tree/model")
                if dt_model.exists():
                    raw = dt_model.read_bytes().decode("utf-8", errors="ignore").replace("\x00", "").strip()
                    if raw:
                        _cached_gpu_name = raw
                        return _cached_gpu_name
            except Exception:
                pass

            # b) /proc/cpuinfo
            try:
                from pathlib import Path
                cpuinfo = Path("/proc/cpuinfo")
                if cpuinfo.exists():
                    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line_lower = line.lower()
                        if line_lower.startswith("model") or line_lower.startswith("hardware"):
                            val = line.split(":")[-1].strip()
                            if val and not val.isdigit():
                                _cached_gpu_name = val
                                return _cached_gpu_name
            except Exception:
                pass

    _cached_gpu_name = ""
    return ""


def is_integrated_gpu() -> bool:
    """
    Retorna True si la GPU detectada corresponde a arquitectura de memoria unificada/compartida
    (Apple Silicon UMA, Windows ARM64 Snapdragon/Adreno, Linux ARM64 SoCs, APU AMD Ryzen o Intel iGPU)
    que comparten la memoria RAM del sistema como VRAM.
    """
    # Toda arquitectura ARM (Apple Silicon, Windows on ARM Snapdragon, Linux aarch64 Jetson/SBC)
    # opera con arquitectura de memoria unificada / compartida con el sistema.
    if is_arm_architecture():
        return True

    # En macOS Intel: verificar si usa gráficos integrados Intel Iris/UHD o dedicada AMD
    if sys.platform == "darwin":
        name = get_gpu_name().lower()
        if not name or any(k in name for k in ("intel", "iris", "uhd", "integrated")):
            return True
        if any(d in name for d in ("radeon pro", "discrete", "rx ")):
            return False
        return True

    name = get_gpu_name().lower()
    if not name:
        return False
    integrated_keywords = (
        "radeon(tm) graphics",
        "radeon graphics",
        "vega",
        "apu",
        "uhd graphics",
        "iris",
        "hd graphics",
        "adreno",
    )
    dedicated_keywords = ("rx ", "xt", "geforce", "rtx", "gtx", "arc a", "discrete", "pro w")
    if any(k in name for k in integrated_keywords) and not any(d in name for d in dedicated_keywords):
        return True
    return False


def get_best_dml_device_id() -> int:
    """
    En portátiles o PCs con GPU dual (APU AMD/Intel integrada + GPU dedicada NVIDIA/AMD),
    DirectML suele asignar device_id=0 a la integrada y device_id=1 a la dedicada.
    Detecta automáticamente el device_id correspondiente a la GPU dedicada
    inspeccionando el orden en Win32_VideoController. Si solo hay APU integrada, retorna su id.
    En Linux retorna 0 de inmediato al no utilizar DirectML.
    """
    global _best_dml_device_id

    # Si la aceleración GPU está desactivada, forzada a CPU o no es Windows, retornar 0
    if FORCE_CPU_ONLY or not ENABLE_GPU_ACCELERATION or sys.platform != "win32":
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
            
            # Buscar primero GPU dedicada
            for idx, name in enumerate(names):
                name_lower = name.lower()
                is_igpu = any(k in name_lower for k in ("intel", "uhd", "iris", "radeon(tm) graphics", "radeon graphics", "vega", "adreno"))
                is_dgpu = any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon rx", "radeon pro", "arc a")) or (
                    any(k in name_lower for k in ("radeon", "amd")) and not is_igpu
                )
                if is_dgpu and not is_igpu:
                    logger.info(f"GPU dedicada detectada en Win32_VideoController: '{name}' (DirectML device_id={idx})")
                    _best_dml_device_id = idx
                    return _best_dml_device_id

            # Si no hay dedicada, verificar si hay APU o iGPU utilizable
            for idx, name in enumerate(names):
                name_lower = name.lower()
                if any(k in name_lower for k in ("radeon", "amd", "intel", "iris", "uhd", "adreno", "qualcomm")):
                    logger.info(f"Gráficos integrados detectados en Win32_VideoController: '{name}' (DirectML device_id={idx})")
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
    """Verifica si hay una GPU o NPU compatible disponible (CUDA, ROCm, QNN, CoreML o DirectML)."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        return any(p in avail for p in [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "QNNExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
        ])
    except Exception:
        return False


def set_gpu_acceleration(enabled: bool) -> bool:
    """Activa o desactiva dinámicamente la aceleración por GPU."""
    global FORCE_CPU_ONLY
    FORCE_CPU_ONLY = not enabled
    return not FORCE_CPU_ONLY


def detect_device() -> str:
    """Retorna el nombre descriptivo del dispositivo activo para la interfaz (NVIDIA, AMD, Apple Silicon, Qualcomm QNN, DirectML ARM, Intel o CPU)."""
    import platform
    is_intel_mac = (sys.platform == "darwin" and platform.machine() == "x86_64")
    is_arm = is_arm_architecture()

    if is_intel_mac:
        cpu_fallback_name = "Intel Mac (CPU)"
    elif is_arm and sys.platform.startswith("linux"):
        soc_name = get_gpu_name()
        cpu_fallback_name = f"Linux ARM64 NEON ({soc_name})" if soc_name else "Linux ARM64 NEON (CPU)"
    else:
        cpu_fallback_name = "CPU"

    if FORCE_CPU_ONLY or not ENABLE_GPU_ACCELERATION:
        return cpu_fallback_name

    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        gpu_name = get_gpu_name()

        # 1. NVIDIA CUDA
        if "CUDAExecutionProvider" in avail:
            if gpu_name:
                clean_name = gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
                return f"GPU: {clean_name}"
            return "NVIDIA CUDA (GPU)"
        # 2. AMD ROCm
        elif "ROCMExecutionProvider" in avail:
            if gpu_name:
                clean_name = gpu_name.replace("AMD ", "").replace("Advanced Micro Devices, Inc. ", "")
                return f"GPU: {clean_name}"
            return "AMD ROCm (GPU)"
        # 3. Qualcomm QNN (NPU Hexagon)
        elif "QNNExecutionProvider" in avail:
            if gpu_name and any(k in gpu_name.lower() for k in ("snapdragon", "qualcomm", "hexagon")):
                return f"Qualcomm QNN (NPU: {gpu_name})"
            return "Qualcomm QNN (NPU / Snapdragon)"
        # 4. Apple Silicon / macOS CoreML
        elif "CoreMLExecutionProvider" in avail:
            if sys.platform == "darwin":
                if platform.machine() == "arm64":
                    if gpu_name:
                        return f"Apple Silicon CoreML ({gpu_name} - GPU / Neural Engine)"
                    return "Apple Silicon CoreML (GPU / Neural Engine)"
                else:
                    if gpu_name:
                        return f"macOS CoreML (GPU: {gpu_name})"
                    return "macOS CoreML (GPU)"
            return "CoreML (Apple GPU / NPU)"
        # 5. DirectML (Windows x64 y Windows ARM64 Adreno)
        elif "DmlExecutionProvider" in avail:
            if sys.platform == "win32" and is_arm:
                if gpu_name:
                    return f"DirectML ARM64 (GPU: {gpu_name})"
                return "DirectML ARM64 (GPU: Qualcomm Adreno)"
            if gpu_name:
                clean_name = gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
                return f"GPU: {clean_name}"
            return "DirectML (GPU / AMD / Intel / NVIDIA)"
    except Exception:
        pass
    return cpu_fallback_name


def get_inference_providers():
    """Retorna la lista ordenada de proveedores de ejecución para ONNX Runtime priorizando la GPU/NPU dedicada/acelerada."""
    if FORCE_CPU_ONLY or not ENABLE_GPU_ACCELERATION:
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        selected = []
        # 1. CUDA (NVIDIA Desktop y Jetson en Linux ARM64)
        if "CUDAExecutionProvider" in avail:
            selected.append("CUDAExecutionProvider")
        # 2. AMD ROCm
        if "ROCMExecutionProvider" in avail:
            selected.append(("ROCMExecutionProvider", {"device_id": 0}))
        # 3. Qualcomm QNN (Windows ARM64)
        if "QNNExecutionProvider" in avail:
            selected.append("QNNExecutionProvider")
        # 4. Apple Silicon / macOS CoreML
        if "CoreMLExecutionProvider" in avail:
            selected.append("CoreMLExecutionProvider")
        # 5. DirectML (Windows x64 y Windows ARM64 Adreno)
        if "DmlExecutionProvider" in avail:
            dml_id = get_best_dml_device_id()
            selected.append(("DmlExecutionProvider", {"device_id": dml_id}))
        selected.append("CPUExecutionProvider")
        return selected
    except Exception:
        return ["CPUExecutionProvider"]


def get_optimized_session_options():
    """
    Retorna SessionOptions de ONNX Runtime optimizado para alto rendimiento
    sin saturar la CPU ni congelar la interfaz de usuario.
    Limita intra_op_num_threads para reservar núcleos al hilo principal y al sistema operativo.
    En procesadores ARM (big.LITTLE), desactiva el spinning para evitar sobrecalentamiento y drenaje de batería.
    """
    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = os.cpu_count() or 4
        threads = max(1, min(4, cpu_count // 2 if cpu_count > 2 else cpu_count))
        sess_options.intra_op_num_threads = threads
        sess_options.inter_op_num_threads = 1

        # Optimización para arquitecturas ARM big.LITTLE (Snapdragon, Jetson, Raspberry Pi, Apple Silicon)
        if is_arm_architecture():
            try:
                sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass

        return sess_options
    except Exception:
        return None

# Parámetros por defecto de la aplicación
APP_NAME = "TimoitaseBG"
APP_VERSION = "1.1.0"
APP_DEVELOPER = "Eliather"
DEFAULT_WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION}"
WINDOW_MIN_WIDTH = 1200
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
    ("rmbg-2.0", "RMBG 2.0 (Bria AI SOTA — Máxima Precisión General y Cabello)"),
    ("birefnet-general", "BiRefNet (Alta Definición General y Bordes Complejos)"),
    ("inspyrenet", "InSPyReNet Swin-B (Fotografía y Retratos Reales — Anti-aliasing Subpíxel)"),
    ("u2net_human_seg", "U2-Net Human (Detector de Personas y Ropa — Especial Playeras Blancas)"),
]
DEFAULT_BG_MODEL = "isnet-anime"

# Opciones de escala para Restaurar
RESTORE_SCALES = [
    ("x2", "Escalar 2x"),
    ("x4", "Escalar 4x (Máxima resolución)"),
    ("enhance_only", "Solo nitidez (Sin cambiar tamaño)"),
]
