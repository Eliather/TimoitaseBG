"""
TimoitaseBG - Detección de hardware, aceleración por GPU/NPU y opciones de sesión ONNX Runtime.
"""
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("TimoitaseBG.Hardware")

ENABLE_GPU_ACCELERATION: bool = True
FORCE_CPU_ONLY: bool = False

_best_dml_device_id: Optional[int] = None
_dml_probe_lock = threading.Lock()
_cached_gpu_name: Optional[str] = None
_UNSET = object()

_VIRTUAL_DISPLAY_KEYWORDS = (
    "parsec",
    "remote desktop",
    "microsoft basic",
    "virtual",
    "indirect display",
    "citrix",
    "spacedesk",
    "amyuni",
    "luminon",
    "splashtop",
    "teamviewer",
    "anydesk",
)


def _cfg_module():
    return sys.modules.get("app.config")


def _is_force_cpu() -> bool:
    cfg = _cfg_module()
    if cfg is not None and hasattr(cfg, "FORCE_CPU_ONLY"):
        return bool(cfg.FORCE_CPU_ONLY)
    return FORCE_CPU_ONLY


def _is_gpu_accel_enabled() -> bool:
    cfg = _cfg_module()
    if cfg is not None and hasattr(cfg, "ENABLE_GPU_ACCELERATION"):
        return bool(cfg.ENABLE_GPU_ACCELERATION)
    return ENABLE_GPU_ACCELERATION


def _resolve_gpu_name() -> str:
    cfg = _cfg_module()
    if cfg is not None:
        fn = getattr(cfg, "get_gpu_name", None)
        if callable(fn) and fn is not get_gpu_name:
            return fn()
    return get_gpu_name()


def _probe_dml_device_worker(model_path_str: str, device_id: int) -> None:
    """
    Sonda aislada en subproceso independiente para verificar si DirectML
    responde en un device_id específico.
    Se ejecuta fuera del proceso principal para proteger el sistema ante cuelgues D3D12/TDR.
    Libera explícitamente la sesión y fuerza recolección de basura antes de salir.
    """
    import gc

    session = None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path_str,
            providers=[("DmlExecutionProvider", {"device_id": device_id})],
        )
        del session
        gc.collect()
        sys.exit(0)
    except Exception:
        if session is not None:
            del session
        gc.collect()
        sys.exit(1)


def is_arm_architecture() -> bool:
    """Retorna True si el procesador anfitrión es de arquitectura ARM (ARM64, aarch64, armv8, etc.)."""
    import platform

    mach = platform.machine().lower()
    return mach in ("arm64", "aarch64", "armv8l", "armv7l", "arm") or mach.startswith("arm")


def _query_windows_gpus_fast() -> list:
    """
    Obtiene la lista de adaptadores físicos de video en Windows de forma instantánea (<1 ms)
    leyendo la clase de registro Display ({4d36e968-e325-11ce-bfc1-08002be10318}),
    filtrando pantallas virtuales (Parsec, RDP, Microsoft Basic, etc.).
    Usa PowerShell (con CREATE_NO_WINDOW) únicamente como respaldo si el registro no devuelve datos.
    """
    adapters = []
    if sys.platform != "win32":
        return adapters

    try:
        import winreg

        class_key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_key_path) as class_key:
            idx = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(class_key, idx)
                    idx += 1
                    if not subkey_name.isdigit():
                        continue
                    with winreg.OpenKey(class_key, subkey_name) as subkey:
                        desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                        if desc and isinstance(desc, str):
                            desc_clean = desc.strip()
                            desc_lower = desc_clean.lower()
                            if not any(vk in desc_lower for vk in _VIRTUAL_DISPLAY_KEYWORDS):
                                if desc_clean not in adapters:
                                    adapters.append(desc_clean)
                except OSError:
                    break
    except Exception:
        pass

    if adapters:
        return adapters

    try:
        import subprocess

        cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            stderr=subprocess.DEVNULL,
            timeout=3,
            creationflags=flags,
        ).decode(errors="ignore").strip()
        for line in out.splitlines():
            line_clean = line.strip()
            if line_clean and not any(vk in line_clean.lower() for vk in _VIRTUAL_DISPLAY_KEYWORDS):
                adapters.append(line_clean)
    except Exception:
        pass

    return adapters


def _store_cached_gpu_name(name: str) -> str:
    global _cached_gpu_name
    _cached_gpu_name = name
    cfg = _cfg_module()
    if cfg is not None:
        cfg._cached_gpu_name = name
    return name


def get_gpu_name() -> str:
    """Obtiene el nombre de la GPU dedicada o SoC activo (Apple Silicon, Snapdragon Adreno, Jetson, NVIDIA, AMD, etc.)."""
    global _cached_gpu_name
    cfg = _cfg_module()
    if cfg is not None and getattr(cfg, "_cached_gpu_name", _UNSET) is None:
        _cached_gpu_name = None

    if _cached_gpu_name is not None:
        return _cached_gpu_name

    # 1. En macOS (Darwin): Apple Silicon SoC o system_profiler
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            ).decode().strip()
            if out and "apple" in out.lower():
                return _store_cached_gpu_name(out)
        except Exception:
            pass

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
                        return _store_cached_gpu_name(chip_model)
        except Exception:
            pass

    # 2. En Windows: consultar primero el registro instantáneo antes de lanzar subprocesos externos
    if sys.platform == "win32":
        win_gpus = _query_windows_gpus_fast()
        for name in win_gpus:
            name_lower = name.lower()
            is_igpu = any(k in name_lower for k in ("intel", "uhd", "iris", "radeon(tm) graphics", "radeon graphics", "vega", "adreno"))
            is_dgpu = any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon rx", "radeon pro", "arc a")) or (
                any(k in name_lower for k in ("radeon", "amd")) and not is_igpu
            )
            if is_dgpu and not is_igpu:
                return _store_cached_gpu_name(name)
        for name in win_gpus:
            name_lower = name.lower()
            if any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd", "arc", "intel", "adreno", "qualcomm", "snapdragon")):
                return _store_cached_gpu_name(name)

    # 3. Probar nvidia-smi con timeout prudente (NVIDIA en Linux y Windows, incluyendo Jetson si está disponible)
    try:
        import subprocess

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=3,
            creationflags=flags,
        ).decode().strip()
        if out:
            return _store_cached_gpu_name(out.splitlines()[0].strip())
    except Exception:
        pass

    # 4. Probar rocm-smi en Linux si nvidia-smi no detectó tarjeta (AMD ROCm en Linux)
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
                    return _store_cached_gpu_name(line.split(":")[-1].strip())
        except Exception:
            pass

    # 5. Respaldo en Windows ARM64: consultar procesador Snapdragon si la GPU no retornó o para QNN
    if sys.platform == "win32" and is_arm_architecture():
        try:
            import platform

            proc = platform.processor()
            if proc and any(k in proc.lower() for k in ("snapdragon", "qualcomm")):
                return _store_cached_gpu_name(proc.strip())
        except Exception:
            pass
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                proc_name, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                if proc_name and any(kw in proc_name.lower() for kw in ("snapdragon", "qualcomm")):
                    return _store_cached_gpu_name(proc_name.strip())
        except Exception:
            pass

    elif sys.platform != "win32" and sys.platform != "darwin":
        # 6. En Linux x86_64 y ARM: lspci
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
                        return _store_cached_gpu_name(line.split(":")[-1].strip())
        except Exception:
            pass

        # 7. En Linux ARM64 (aarch64): SoCs como NVIDIA Jetson, Raspberry Pi, Rockchip
        if is_arm_architecture():
            try:
                dt_model = Path("/proc/device-tree/model")
                if dt_model.exists():
                    raw = dt_model.read_bytes().decode("utf-8", errors="ignore").replace("\x00", "").strip()
                    if raw:
                        return _store_cached_gpu_name(raw)
            except Exception:
                pass

            try:
                cpuinfo = Path("/proc/cpuinfo")
                if cpuinfo.exists():
                    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line_lower = line.lower()
                        if line_lower.startswith("model") or line_lower.startswith("hardware"):
                            val = line.split(":")[-1].strip()
                            if val and not val.isdigit():
                                return _store_cached_gpu_name(val)
            except Exception:
                pass

    return _store_cached_gpu_name("")


def is_integrated_gpu() -> bool:
    """
    Retorna True si la GPU detectada corresponde a arquitectura de memoria unificada/compartida
    (Apple Silicon UMA, Windows ARM64 Snapdragon/Adreno, Linux ARM64 SoCs, APU AMD Ryzen o Intel iGPU)
    que comparten la memoria RAM del sistema como VRAM.
    """
    if is_arm_architecture():
        return True

    if sys.platform == "darwin":
        name = _resolve_gpu_name().lower()
        if not name or any(k in name for k in ("intel", "iris", "uhd", "integrated")):
            return True
        if any(d in name for d in ("radeon pro", "discrete", "rx ")):
            return False
        return True

    name = _resolve_gpu_name().lower()
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
    filtrando adaptadores virtuales. Si solo hay una GPU física (dedicada o integrada), retorna 0.
    En Linux/macOS retorna 0 de inmediato al no utilizar DirectML.
    """
    global _best_dml_device_id

    if _is_force_cpu() or not _is_gpu_accel_enabled() or sys.platform != "win32":
        return 0

    if _best_dml_device_id is not None:
        return _best_dml_device_id

    with _dml_probe_lock:
        if _best_dml_device_id is not None:
            return _best_dml_device_id

        try:
            names = _query_windows_gpus_fast()
            if len(names) <= 1:
                _best_dml_device_id = 0
                return 0

            for idx, name in enumerate(names):
                name_lower = name.lower()
                is_igpu = any(k in name_lower for k in ("intel", "uhd", "iris", "radeon(tm) graphics", "radeon graphics", "vega", "adreno"))
                is_dgpu = any(k in name_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon rx", "radeon pro", "arc a")) or (
                    any(k in name_lower for k in ("radeon", "amd")) and not is_igpu
                )
                if is_dgpu and not is_igpu:
                    logger.info(f"GPU dedicada detectada: '{name}' (DirectML device_id={idx})")
                    _best_dml_device_id = idx
                    return _best_dml_device_id

            for idx, name in enumerate(names):
                name_lower = name.lower()
                if any(k in name_lower for k in ("radeon", "amd", "intel", "iris", "uhd", "adreno", "qualcomm")):
                    logger.info(f"Gráficos integrados detectados: '{name}' (DirectML device_id={idx})")
                    _best_dml_device_id = idx
                    return _best_dml_device_id
        except Exception as e:
            logger.warning(f"Error detectando adaptadores para DirectML: {e}")

        _best_dml_device_id = 0
        return _best_dml_device_id


def is_gpu_available() -> bool:
    """Verifica si hay una GPU o NPU compatible disponible (CUDA, ROCm, QNN, CoreML o DirectML)."""
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        return any(
            p in avail
            for p in [
                "CUDAExecutionProvider",
                "ROCMExecutionProvider",
                "QNNExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
            ]
        )
    except Exception:
        return False


def set_gpu_acceleration(enabled: bool) -> bool:
    """Activa o desactiva dinámicamente la aceleración por GPU."""
    global FORCE_CPU_ONLY
    FORCE_CPU_ONLY = not enabled
    cfg = _cfg_module()
    if cfg is not None:
        cfg.FORCE_CPU_ONLY = FORCE_CPU_ONLY
    return not FORCE_CPU_ONLY


def detect_device() -> str:
    """Retorna el nombre descriptivo del dispositivo activo para la interfaz (NVIDIA, AMD, Apple Silicon, Qualcomm QNN, DirectML ARM, Intel o CPU)."""
    import platform

    is_intel_mac = sys.platform == "darwin" and platform.machine() == "x86_64"
    is_arm = is_arm_architecture()

    if is_intel_mac:
        cpu_fallback_name = "Intel Mac (CPU)"
    elif is_arm and sys.platform.startswith("linux"):
        soc_name = _resolve_gpu_name()
        cpu_fallback_name = f"Linux ARM64 NEON ({soc_name})" if soc_name else "Linux ARM64 NEON (CPU)"
    else:
        cpu_fallback_name = "CPU"

    if _is_force_cpu() or not _is_gpu_accel_enabled():
        return cpu_fallback_name

    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        gpu_name = _resolve_gpu_name()

        if "CUDAExecutionProvider" in avail:
            if gpu_name:
                clean_name = gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
                return f"GPU: {clean_name}"
            return "NVIDIA CUDA (GPU)"
        elif "ROCMExecutionProvider" in avail:
            if gpu_name:
                clean_name = gpu_name.replace("AMD ", "").replace("Advanced Micro Devices, Inc. ", "")
                return f"GPU: {clean_name}"
            return "AMD ROCm (GPU)"
        elif "QNNExecutionProvider" in avail:
            if gpu_name and any(k in gpu_name.lower() for k in ("snapdragon", "qualcomm", "hexagon")):
                return f"Qualcomm QNN (NPU: {gpu_name})"
            return "Qualcomm QNN (NPU / Snapdragon)"
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
    if _is_force_cpu() or not _is_gpu_accel_enabled():
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        selected = []
        if "CUDAExecutionProvider" in avail:
            selected.append("CUDAExecutionProvider")
        if "ROCMExecutionProvider" in avail:
            selected.append(("ROCMExecutionProvider", {"device_id": 0}))
        if "QNNExecutionProvider" in avail:
            selected.append("QNNExecutionProvider")
        if "CoreMLExecutionProvider" in avail:
            selected.append("CoreMLExecutionProvider")
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

        if is_arm_architecture():
            try:
                sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass

        return sess_options
    except Exception:
        return None
