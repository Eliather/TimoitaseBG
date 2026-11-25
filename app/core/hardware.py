"""
TimoitaseBG - Detección de hardware, aceleración por GPU/NPU y opciones de sesión ONNX Runtime.
"""
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _resolve_windows_system_binary(relative_path: str, fallback: str) -> str:
    """Resuelve un binario del sistema Windows bajo %SystemRoot% para evitar CWD binary planting."""
    sys_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
    candidate = Path(sys_root) / relative_path
    if candidate.is_file():
        return str(candidate)
    return fallback


def _is_dedicated_gpu_name(name_lower: str) -> bool:
    """Determina si el nombre de adaptador corresponde a una GPU dedicada."""
    if re.search(r"\barc(?:\(tm\)|™)?\s+[ab]", name_lower):
        return True
    if any(
        k in name_lower
        for k in (
            "nvidia",
            "geforce",
            "rtx",
            "gtx",
            "quadro",
            "titan",
            "radeon rx",
            "radeon pro",
            "pro w",
            "discrete",
        )
    ):
        return True
    if re.search(r"\b(?:rx\s*\d|xt|xtx)\b", name_lower):
        return True
    return False


def _is_integrated_gpu_name(name_lower: str) -> bool:
    """Determina si el nombre de adaptador corresponde a gráficos integrados."""
    if _is_dedicated_gpu_name(name_lower):
        return False
    return any(
        k in name_lower
        for k in (
            "intel",
            "uhd",
            "iris",
            "hd graphics",
            "radeon(tm) graphics",
            "radeon graphics",
            "vega",
            "apu",
            "adreno",
        )
    )


def _enumerate_dxgi_adapters() -> List[Dict[str, Any]]:
    """
    Enumera adaptadores físicos de video en Windows usando DXGI (`dxgi.dll`:
    `CreateDXGIFactory1` + `IDXGIFactory1::EnumAdapters1` + `IDXGIAdapter1::GetDesc1`).
    Excluye adaptadores por software (`DXGI_ADAPTER_FLAG_SOFTWARE`) y pantallas virtuales.
    Cada entrada contiene el índice DXGI exacto que DirectML utiliza como `device_id`.
    """
    if sys.platform != "win32":
        return []

    adapters: List[Dict[str, Any]] = []
    factory_ptr = None
    try:
        import ctypes
        from ctypes import POINTER, Structure, byref, c_int32, c_long, c_size_t, c_ubyte, c_uint16, c_uint32, c_void_p, c_wchar

        class GUID(Structure):
            _fields_ = [
                ("Data1", c_uint32),
                ("Data2", c_uint16),
                ("Data3", c_uint16),
                ("Data4", c_ubyte * 8),
            ]

        class LUID(Structure):
            _fields_ = [
                ("LowPart", c_uint32),
                ("HighPart", c_int32),
            ]

        class DXGI_ADAPTER_DESC1(Structure):
            _fields_ = [
                ("Description", c_wchar * 128),
                ("VendorId", c_uint32),
                ("DeviceId", c_uint32),
                ("SubSysId", c_uint32),
                ("Revision", c_uint32),
                ("DedicatedVideoMemory", c_size_t),
                ("DedicatedSystemMemory", c_size_t),
                ("SharedSystemMemory", c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", c_uint32),
            ]

        DXGI_ADAPTER_FLAG_SOFTWARE = 2
        IID_IDXGIFactory1 = GUID(
            0x770AAE78,
            0xF26F,
            0x4DBA,
            (c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
        )

        dxgi = ctypes.WinDLL("dxgi.dll")
        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = [POINTER(GUID), POINTER(c_void_p)]
        create_factory.restype = c_long

        factory_void = c_void_p()
        hr = create_factory(byref(IID_IDXGIFactory1), byref(factory_void))
        if hr < 0 or not factory_void.value:
            return []
        factory_ptr = factory_void.value

        vtable_factory = ctypes.cast(ctypes.c_void_p.from_address(factory_ptr), POINTER(c_void_p))
        release_proto = ctypes.WINFUNCTYPE(c_uint32, c_void_p)
        enum_adapters1_proto = ctypes.WINFUNCTYPE(c_long, c_void_p, c_uint32, POINTER(c_void_p))
        get_desc1_proto = ctypes.WINFUNCTYPE(c_long, c_void_p, POINTER(DXGI_ADAPTER_DESC1))

        enum_adapters1 = enum_adapters1_proto(vtable_factory[12])
        factory_release = release_proto(vtable_factory[2])

        idx = 0
        while True:
            adapter_void = c_void_p()
            hr = enum_adapters1(factory_ptr, idx, byref(adapter_void))
            if hr < 0 or not adapter_void.value:
                break
            adapter_ptr = adapter_void.value
            try:
                vtable_adapter = ctypes.cast(ctypes.c_void_p.from_address(adapter_ptr), POINTER(c_void_p))
                get_desc1 = get_desc1_proto(vtable_adapter[10])
                desc = DXGI_ADAPTER_DESC1()
                if get_desc1(adapter_ptr, byref(desc)) >= 0:
                    flags = int(desc.Flags)
                    name = (desc.Description or "").strip()
                    name_lower = name.lower()
                    is_software = bool(flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    is_virtual = any(vk in name_lower for vk in _VIRTUAL_DISPLAY_KEYWORDS)
                    if name and not is_software and not is_virtual:
                        adapters.append(
                            {
                                "index": idx,
                                "name": name,
                                "dedicated_vram": int(desc.DedicatedVideoMemory),
                                "flags": flags,
                            }
                        )
            finally:
                try:
                    vtable_adapter = ctypes.cast(ctypes.c_void_p.from_address(adapter_ptr), POINTER(c_void_p))
                    adapter_release = release_proto(vtable_adapter[2])
                    adapter_release(adapter_ptr)
                except Exception:
                    pass
            idx += 1
    except Exception as e:
        logger.debug(f"Enumeración DXGI no disponible, usando respaldo: {e}")
    finally:
        if factory_ptr is not None:
            try:
                factory_release(factory_ptr)
            except Exception:
                pass

    return adapters


def _query_windows_gpus_fast() -> list:
    """
    Obtiene la lista de adaptadores físicos de video en Windows de forma instantánea (<1 ms)
    leyendo la clase de registro Display ({4d36e968-e325-11ce-bfc1-08002be10318}),
    filtrando pantallas virtuales (Parsec, RDP, Microsoft Basic, etc.).
    Usa PowerShell (con ruta absoluta y CREATE_NO_WINDOW) únicamente como respaldo si el registro no devuelve datos.
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
                except OSError:
                    break
                if not subkey_name.isdigit():
                    continue
                try:
                    with winreg.OpenKey(class_key, subkey_name) as subkey:
                        desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                        if desc and isinstance(desc, str):
                            desc_clean = desc.strip()
                            desc_lower = desc_clean.lower()
                            if not any(vk in desc_lower for vk in _VIRTUAL_DISPLAY_KEYWORDS):
                                if desc_clean not in adapters:
                                    adapters.append(desc_clean)
                except OSError:
                    continue
    except Exception:
        pass

    if adapters:
        return adapters

    try:
        import subprocess

        ps_exe = _resolve_windows_system_binary(
            r"System32\WindowsPowerShell\v1.0\powershell.exe",
            "powershell.exe",
        )
        cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            [ps_exe, "-NoProfile", "-Command", cmd],
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

    # 2. En Windows: consultar primero DXGI / registro instantáneo antes de lanzar subprocesos externos
    if sys.platform == "win32":
        dxgi_adapters = _enumerate_dxgi_adapters()
        win_gpus = [a["name"] for a in dxgi_adapters] if dxgi_adapters else _query_windows_gpus_fast()
        for name in win_gpus:
            name_lower = name.lower()
            is_igpu = _is_integrated_gpu_name(name_lower)
            is_dgpu = _is_dedicated_gpu_name(name_lower) or (
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
            ).decode(errors="ignore").strip()
            for line in out.splitlines():
                line_clean = line.strip()
                line_lower = line_clean.lower()
                if not line_clean or line_clean.startswith("=") or "vendor" in line_lower:
                    continue
                if any(k in line_lower for k in ("card series", "card model", "product name")):
                    val = line_clean.rsplit(":", 1)[-1].strip()
                    if val and val.lower() not in ("n/a", "unknown") and not val.lower().startswith("0x"):
                        return _store_cached_gpu_name(val)
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
            ).decode(errors="ignore").strip()
            lspci_candidates: List[str] = []
            for line in out.splitlines():
                line_lower = line.lower()
                if any(term in line_lower for term in ("vga", "3d controller", "display controller")):
                    if any(k in line_lower for k in ("nvidia", "geforce", "rtx", "gtx", "radeon", "amd", "intel", "arc", "adreno", "qualcomm")):
                        # Formato: "01:00.0 VGA compatible controller: Vendor Device [Name] (rev a1)"
                        parts = line.split(": ", 1)
                        desc = parts[1].strip() if len(parts) > 1 else line.split(":")[-1].strip()
                        if desc:
                            lspci_candidates.append(desc)
            for desc in lspci_candidates:
                desc_lower = desc.lower()
                if _is_dedicated_gpu_name(desc_lower) and not _is_integrated_gpu_name(desc_lower):
                    return _store_cached_gpu_name(desc)
            if lspci_candidates:
                return _store_cached_gpu_name(lspci_candidates[0])
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
    dedicated_keywords = (
        "rx ",
        "geforce",
        "rtx",
        "rtx a",
        "gtx",
        "quadro",
        "titan",
        "arc a",
        "arc b",
        "radeon pro",
        "discrete",
        "pro w",
    )
    has_dedicated = (
        any(d in name for d in dedicated_keywords)
        or bool(re.search(r"\b(?:xt|xtx)\b", name))
        or bool(re.search(r"\barc(?:\(tm\)|™)?\s+[ab]", name))
    )
    if any(k in name for k in integrated_keywords) and not has_dedicated:
        return True
    return False


def get_best_dml_device_id() -> int:
    """
    En portátiles o PCs con GPU dual (APU AMD/Intel integrada + GPU dedicada NVIDIA/AMD),
    DirectML suele asignar device_id=0 a la integrada y device_id=1 a la dedicada.
    Enumera primero los adaptadores reales de DXGI (`dxgi.dll`) para obtener el `device_id`
    exacto que usa DirectML, con respaldo por Registro/PowerShell si DXGI no está disponible.
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
            dxgi_adapters = _enumerate_dxgi_adapters()
            if dxgi_adapters:
                if len(dxgi_adapters) == 1:
                    _best_dml_device_id = int(dxgi_adapters[0]["index"])
                    return _best_dml_device_id

                dedicated_candidates = []
                for adapter in dxgi_adapters:
                    name_lower = adapter["name"].lower()
                    is_igpu = _is_integrated_gpu_name(name_lower)
                    is_dgpu = _is_dedicated_gpu_name(name_lower) or (
                        any(k in name_lower for k in ("radeon", "amd")) and not is_igpu
                    )
                    if is_dgpu and not is_igpu:
                        dedicated_candidates.append(adapter)

                if dedicated_candidates:
                    best = max(dedicated_candidates, key=lambda a: (a["dedicated_vram"], -a["index"]))
                    logger.info(
                        "GPU dedicada DXGI detectada: '%s' (VRAM=%d MB, DirectML device_id=%d)",
                        best["name"],
                        best["dedicated_vram"] // (1024 * 1024),
                        best["index"],
                    )
                    _best_dml_device_id = int(best["index"])
                    return _best_dml_device_id

                best = max(dxgi_adapters, key=lambda a: (a["dedicated_vram"], -a["index"]))
                logger.info(
                    "Adaptador DXGI seleccionado: '%s' (DirectML device_id=%d)",
                    best["name"],
                    best["index"],
                )
                _best_dml_device_id = int(best["index"])
                return _best_dml_device_id

            names = _query_windows_gpus_fast()
            if len(names) <= 1:
                _best_dml_device_id = 0
                return 0

            for idx, name in enumerate(names):
                name_lower = name.lower()
                is_igpu = _is_integrated_gpu_name(name_lower)
                is_dgpu = _is_dedicated_gpu_name(name_lower) or (
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


def get_optimized_session_options(use_dml: bool = False):
    """
    Retorna SessionOptions de ONNX Runtime optimizado para alto rendimiento
    sin saturar la CPU ni congelar la interfaz de usuario.
    Limita intra_op_num_threads para reservar núcleos al hilo principal y al sistema operativo.
    Cuando `use_dml=True`, desactiva `enable_mem_pattern` y fuerza `ORT_SEQUENTIAL` según
    requisitos oficiales de DirectML para evitar corrupción de memoria y deadlocks D3D12.
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

        if use_dml:
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if is_arm_architecture():
            try:
                sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass

        return sess_options
    except Exception:
        return None
