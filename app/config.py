"""
TimoitaseBG - Configuración global y rutas del sistema.
"""
import os
import sys
from typing import Optional, Tuple
from pathlib import Path


def resolve_app_dir() -> Path:
    """
    Resuelve el directorio de activos estáticos de la aplicación:
    - En modo empaquetado (PyInstaller / sys.frozen), apunta a sys._MEIPASS / 'app'
      para que estilos QSS e iconos vectoriales se carguen siempre correctamente.
    - En modo desarrollo, apunta a la carpeta del paquete 'app'.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            base_res = Path(meipass)
            if (base_res / "app").exists():
                return base_res / "app"
            elif (base_res / "ui").exists():
                return base_res
            return base_res / "app"
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / "app").exists():
            return exe_dir / "app"
        return exe_dir
    return Path(__file__).resolve().parent


def resolve_base_dir() -> Path:
    """
    Resuelve el directorio base o raíz del proyecto/ejecutable:
    - En modo empaquetado, el directorio que contiene el ejecutable (sys.executable).
    - En modo desarrollo, el directorio padre de 'app'.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_models_dir(base_dir: Optional[Path] = None, platform_name: Optional[str] = None) -> Path:
    """
    Resuelve el directorio de modelos ONNX garantizando que sea escribible:
    - Prioridad Portable (USB): Si junto al ejecutable (base_dir) existe una carpeta 'models'
      que ya contiene archivos .onnx o tiene permisos de escritura, se utiliza como destino
      prioritario para soportar instalaciones portables sin tocar el disco del sistema.
    - Modo Empaquetado Estándar (sys.frozen): En Windows, macOS y Linux la aplicación suele
      instalarse en rutas de solo lectura (C:\\Program Files, /Applications, /opt). Por tanto,
      se utiliza el directorio nativo de datos de usuario de cada sistema operativo:
        * Windows: %LOCALAPPDATA%\\TimoitaseBG\\models
        * macOS: ~/Library/Application Support/TimoitaseBG/models
        * Linux: $XDG_DATA_HOME/TimoitaseBG/models (o ~/.local/share/TimoitaseBG/models)
    - Modo Desarrollo: 'app/models' o la carpeta models local.
    """
    if platform_name is None:
        platform_name = sys.platform

    is_frozen = getattr(sys, "frozen", False)

    if base_dir is None:
        base_dir = Path(sys.executable).resolve().parent if is_frozen else Path(__file__).resolve().parent.parent

    if is_frozen:
        portable_models = base_dir / "models"
        if portable_models.exists() and portable_models.is_dir():
            has_onnx = any(portable_models.glob("*.onnx"))
            is_writable = False
            try:
                test_file = portable_models / ".timoitase_write_test"
                test_file.touch(exist_ok=True)
                test_file.unlink(missing_ok=True)
                is_writable = True
            except (OSError, PermissionError):
                is_writable = False

            if has_onnx or is_writable:
                try:
                    portable_models.mkdir(parents=True, exist_ok=True)
                    return portable_models
                except (OSError, PermissionError):
                    pass

        # Directorio de usuario nativo según el sistema operativo
        if platform_name == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA")
            base_user = Path(local_appdata) if local_appdata else (Path.home() / "AppData" / "Local")
            models_path = base_user / "TimoitaseBG" / "models"
        elif platform_name == "darwin":
            models_path = Path.home() / "Library" / "Application Support" / "TimoitaseBG" / "models"
        else:
            xdg_data = os.environ.get("XDG_DATA_HOME")
            base_user = Path(xdg_data) if xdg_data else (Path.home() / ".local" / "share")
            models_path = base_user / "TimoitaseBG" / "models"
    else:
        # Modo desarrollo local
        models_path = Path(__file__).resolve().parent / "models"

    try:
        models_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass

    return models_path


def update_u2net_env(models_dir: Path) -> None:
    """
    Sincroniza las variables de entorno de rembg (U2NET_HOME / U2NET_home)
    para que modelos como isnet-anime y u2net_human_seg guarden y carguen
    sus pesos exactamente en el directorio seguro de modelos.
    """
    models_dir_str = str(models_dir)
    os.environ["U2NET_HOME"] = models_dir_str
    os.environ["U2NET_home"] = models_dir_str


_SYSTEM_CA_BUNDLE_PATHS = (
    "/etc/ssl/certs/ca-certificates.crt",                  # Debian / Ubuntu / Gentoo / Arch
    "/etc/pki/tls/certs/ca-bundle.crt",                    # Fedora / RHEL / CentOS
    "/etc/ssl/ca-bundle.pem",                              # openSUSE / SLES
    "/etc/pki/tls/cacert.pem",                             # OpenELEC
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",   # CentOS / RHEL 7+
    "/etc/ssl/cert.pem",                                   # Alpine / macOS / FreeBSD
)


def resolve_ca_bundle_path() -> Optional[str]:
    """
    Resuelve la ruta de un archivo de autoridades de certificación (CA bundle) válido:
    1. Si SSL_CERT_FILE o REQUESTS_CA_BUNDLE ya apuntan a un archivo existente en disco, lo respeta.
    2. Intenta obtener el bundle empaquetado de `certifi.where()`.
    3. Recorre las rutas estándar de certificados CA del sistema en Linux y macOS.
    """
    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        existing = os.environ.get(env_var)
        if existing and Path(existing).is_file():
            return str(Path(existing).resolve())

    try:
        import certifi
        ca_path = certifi.where()
        if ca_path and Path(ca_path).is_file():
            return str(Path(ca_path).resolve())
    except Exception:
        pass

    for sys_ca in _SYSTEM_CA_BUNDLE_PATHS:
        if Path(sys_ca).is_file():
            return sys_ca

    return None


def configure_ssl_certificates() -> Optional[str]:
    """
    Garantiza que SSL_CERT_FILE y REQUESTS_CA_BUNDLE apunten a un archivo CA válido.
    Corrige en Linux (.AppImage / PyInstaller) el error:
    [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate (_ssl.c:1016)
    cuando OpenSSL busca en rutas de compilación inexistentes (ej. /opt/hostedtoolcache/...).
    """
    ca_bundle = resolve_ca_bundle_path()
    if ca_bundle:
        for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            curr = os.environ.get(env_var)
            if not curr or not Path(curr).is_file():
                os.environ[env_var] = ca_bundle
    return ca_bundle


# Definición e inicialización de rutas globales
APP_DIR: Path = resolve_app_dir()
BASE_DIR: Path = resolve_base_dir()
PROJECT_ROOT: Path = BASE_DIR

MODELS_DIR: Path = resolve_models_dir(BASE_DIR)
update_u2net_env(MODELS_DIR)
configure_ssl_certificates()

# ==============================================================================
# SEGURIDAD Y ACELERACIÓN POR HARDWARE (GPU) — Re-exportado desde app.core.hardware
# ==============================================================================
from app.core.hardware import (
    ENABLE_GPU_ACCELERATION,
    FORCE_CPU_ONLY,
    _cached_gpu_name,
    _probe_dml_device_worker,
    _query_windows_gpus_fast,
    detect_device,
    get_best_dml_device_id,
    get_gpu_name,
    get_inference_providers,
    get_optimized_session_options,
    is_arm_architecture,
    is_gpu_available,
    is_integrated_gpu,
    set_gpu_acceleration,
)

# Parámetros por defecto de la aplicación
APP_NAME = "TimoitaseBG"
APP_VERSION = "1.3.1"
APP_AUTHOR = "Eliather"
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
    ("isnet-anime", "Anime (IS-Net Anime)"),
    ("inspyrenet", "General (InSPyReNet Swin-B)"),
    ("birefnet-general", "Alta Definición (BiRefNet HD)"),
    ("u2net_human_seg", "Personas y Ropa (U2-Net)"),
]
DEFAULT_BG_MODEL = "isnet-anime"

# Opciones de escala para Restaurar
RESTORE_SCALES = [
    ("x2", "Escalar 2x"),
    ("x4", "Escalar 4x (Máxima resolución)"),
    ("enhance_only", "Solo nitidez (Sin cambiar tamaño)"),
]
