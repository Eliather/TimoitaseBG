"""
TimoitaseBG - Descargador y gestor de pesos de modelos locales autocontenidos.
"""
import os
import ssl
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from app.config import MODELS_DIR, configure_ssl_certificates, resolve_ca_bundle_path

logger = logging.getLogger("TimoitaseBG.ModelFetcher")

# URLs de descarga directa verificadas
MODEL_URLS = {
    "inspyrenet.onnx": "https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/resolve/main/onnx/model.onnx",
    "isnet-anime.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
    "birefnet-general.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
    "u2net_human_seg.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
    "RealESRGAN_x4plus_anime_6B.onnx": "https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx",
    "mobile_sam_image_encoder.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
    "sam_mask_decoder_single.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx",
    "yolo11n-seg.onnx": "https://huggingface.co/unileon-robotics/YOLO11-ONNX/resolve/main/yolo11n-seg.onnx",
}


def _create_ssl_context() -> ssl.SSLContext:
    """
    Construye un contexto SSL robusto para descargas HTTPS en Linux (.AppImage / PyInstaller),
    macOS y Windows:
    1. Sincroniza SSL_CERT_FILE / REQUESTS_CA_BUNDLE mediante configure_ssl_certificates().
    2. Carga explícitamente el bundle CA resuelto (certifi o certificados del sistema).
    3. Además carga los certificados por defecto del sistema operativo si están disponibles.
    """
    configure_ssl_certificates()
    ca_bundle = resolve_ca_bundle_path()
    try:
        if ca_bundle and Path(ca_bundle).is_file():
            ctx = ssl.create_default_context(cafile=ca_bundle)
        else:
            ctx = ssl.create_default_context()
        try:
            ctx.load_default_certs()
        except Exception:
            pass
        return ctx
    except Exception as e:
        logger.warning(f"No se pudo inicializar contexto SSL con cafile={ca_bundle}: {e}")
        return ssl.create_default_context()


def _is_ssl_cert_error(exc: BaseException) -> bool:
    """Detecta si una excepción corresponde a un fallo de verificación de certificado SSL."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return True
        if reason and "CERTIFICATE_VERIFY_FAILED" in str(reason):
            return True
    return "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _download_with_context(
    req: urllib.request.Request,
    temp_path: Path,
    ssl_ctx: ssl.SSLContext,
    filename: str = "",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Descarga el recurso desde req hacia temp_path usando el contexto SSL indicado."""
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp, open(temp_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 256 * 1024  # 256KB por bloque para progreso fluido

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                if progress_cb:
                    progress_cb(downloaded, total_size)
                if status_cb and filename:
                    pct = min(100, int(round((downloaded / total_size) * 100)))
                    dl_mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    status_cb(f"Descargando {filename}: {pct}% ({dl_mb:.1f} / {tot_mb:.1f} MB)")
            elif status_cb and filename:
                dl_mb = downloaded / (1024 * 1024)
                status_cb(f"Descargando {filename}: {dl_mb:.1f} MB...")


def ensure_model_file(
    filename: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Garantiza que el archivo de modelo esté descargado localmente en app/models.
    Si no existe, lo descarga mostrando progreso con soporte SSL multiplataforma.
    """
    int8_path = MODELS_DIR / f"{Path(filename).stem}_int8{Path(filename).suffix}"
    if int8_path.exists() and int8_path.stat().st_size > 1000000:
        return int8_path
        
    dest_path = MODELS_DIR / filename
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        return dest_path

    url = MODEL_URLS.get(filename)
    if not url:
        raise ValueError(f"URL no registrada para el modelo {filename}")

    if status_cb:
        status_cb(f"Descargando {filename} por primera vez...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    ssl_ctx = _create_ssl_context()
    try:
        _download_with_context(
            req, temp_path, ssl_ctx, filename=filename, progress_cb=progress_cb, status_cb=status_cb
        )
    except Exception as err:
        if _is_ssl_cert_error(err):
            logger.warning(
                f"Fallo de verificación SSL al descargar {filename} ({err}). "
                f"Reintentando con contexto SSL de respaldo..."
            )
            try:
                fallback_ctx = ssl._create_unverified_context()
                _download_with_context(
                    req, temp_path, fallback_ctx, filename=filename, progress_cb=progress_cb, status_cb=status_cb
                )
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        else:
            temp_path.unlink(missing_ok=True)
            raise

    # Renombrar al terminar
    if temp_path.exists():
        temp_path.replace(dest_path)

    if status_cb:
        status_cb(f"Modelo {filename} listo.")

    return dest_path
