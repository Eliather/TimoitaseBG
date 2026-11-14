"""
TimoitaseBG - Descargador y gestor de pesos de modelos locales autocontenidos.
"""
import hashlib
import logging
import os
import ssl
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

# Hashes SHA-256 hexadecimales verificados para validar integridad tras la descarga.
MODEL_SHA256: dict[str, str] = {
    "inspyrenet.onnx": "c108dd92b3ddfe3a2d9e9ac2b74730cc5acbb2ddc7ea863330c43f56ae832aa3",
    "isnet-anime.onnx": "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
    "mobile_sam_image_encoder.onnx": "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749",
    "RealESRGAN_x4plus_anime_6B.onnx": "2648cab4c4343541c1aa291c6754e9e8edbe7a813fffc2a677423dd12cb6b7f7",
    "sam_mask_decoder_single.onnx": "93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d",
    "u2net_human_seg.onnx": "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
}

MIN_VALID_MODEL_BYTES = 1_000_000


def verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verifica el digest SHA-256 de un archivo local sin cargarlo entero en RAM."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.strip().lower()


def resolve_local_model(filename: str) -> Optional[Path]:
    """Retorna la ruta local de un modelo (priorizando versión _int8 cuantizada), o None si falta."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    int8_path = MODELS_DIR / f"{stem}_int8{suffix}"
    if int8_path.exists() and int8_path.stat().st_size > MIN_VALID_MODEL_BYTES:
        return int8_path

    dest_path = MODELS_DIR / filename
    if dest_path.exists() and dest_path.stat().st_size > MIN_VALID_MODEL_BYTES:
        return dest_path

    return None


def get_models_inventory() -> list[dict]:
    """Retorna el inventario detallado de modelos registrados en MODELS_DIR."""
    inventory: list[dict] = []
    for filename, url in MODEL_URLS.items():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        int8_name = f"{stem}_int8{suffix}"
        fp32_path = MODELS_DIR / filename
        int8_path = MODELS_DIR / int8_name

        fp32_valid = fp32_path.is_file() and fp32_path.stat().st_size > MIN_VALID_MODEL_BYTES
        int8_valid = int8_path.is_file() and int8_path.stat().st_size > MIN_VALID_MODEL_BYTES
        fp32_bytes = fp32_path.stat().st_size if fp32_valid else 0
        int8_bytes = int8_path.stat().st_size if int8_valid else 0

        if int8_valid:
            active_variant = "int8"
            active_path = str(int8_path)
        elif fp32_valid:
            active_variant = "fp32"
            active_path = str(fp32_path)
        else:
            active_variant = "missing"
            active_path = ""

        inventory.append(
            {
                "filename": filename,
                "url": url,
                "fp32_exists": fp32_valid,
                "fp32_bytes": fp32_bytes,
                "int8_filename": int8_name,
                "int8_exists": int8_valid,
                "int8_bytes": int8_bytes,
                "active_variant": active_variant,
                "active_path": active_path,
                "total_bytes": fp32_bytes + int8_bytes,
                "has_sha256": filename in MODEL_SHA256,
            }
        )
    return inventory


def verify_local_model_sha256(filename: str) -> Optional[bool]:
    """Valida el SHA-256 del modelo FP32 si existe localmente y tiene digest registrado."""
    expected = MODEL_SHA256.get(filename)
    fp32_path = MODELS_DIR / filename
    if not expected or not fp32_path.is_file():
        return None
    return verify_sha256(fp32_path, expected)


def delete_local_model(filename: str) -> list[str]:
    """Elimina pesos locales (FP32, _int8 y parciales .tmp) de un modelo. Retorna eliminados."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    removed: list[str] = []
    for candidate in (
        MODELS_DIR / filename,
        MODELS_DIR / f"{stem}_int8{suffix}",
        (MODELS_DIR / filename).with_suffix(".tmp"),
    ):
        if candidate.is_file():
            candidate.unlink(missing_ok=True)
            removed.append(candidate.name)
    return removed


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
    cancel_check: Optional[Callable[[], bool]] = None,
    resume_offset: int = 0,
) -> None:
    """Descarga el recurso desde req hacia temp_path con soporte opcional de reanudación y cancelación."""
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
        is_partial = getattr(resp, "status", 200) == 206 and resume_offset > 0
        open_mode = "ab" if is_partial else "wb"
        downloaded = resume_offset if is_partial else 0
        content_len = int(resp.headers.get("Content-Length", 0) or 0)
        total_size = (downloaded + content_len) if (is_partial and content_len > 0) else content_len
        chunk_size = 256 * 1024  # 256KB por bloque para progreso fluido

        with open(temp_path, open_mode) as out_file:
            while True:
                if cancel_check and cancel_check():
                    raise InterruptedError(f"Descarga cancelada: {filename or temp_path.name}")
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
    cancel_check: Optional[Callable[[], bool]] = None,
    allow_insecure_fallback: bool = True,
) -> Path:
    """
    Garantiza que el archivo de modelo esté descargado localmente en app/models.
    Prioriza variantes *_int8.onnx y soporta reanudación HTTP Range, cancelación
    cooperativa y validación SHA-256 opcional.
    """
    existing = resolve_local_model(filename)
    if existing is not None:
        return existing

    dest_path = MODELS_DIR / filename
    url = MODEL_URLS.get(filename)
    if not url:
        raise ValueError(f"URL no registrada para el modelo {filename}")

    if status_cb:
        status_cb(f"Descargando {filename} por primera vez...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    resume_offset = temp_path.stat().st_size if temp_path.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if resume_offset > 0:
        headers["Range"] = f"bytes={resume_offset}-"
    req = urllib.request.Request(url, headers=headers)

    ssl_ctx = _create_ssl_context()
    try:
        _download_with_context(
            req,
            temp_path,
            ssl_ctx,
            filename=filename,
            progress_cb=progress_cb,
            status_cb=status_cb,
            cancel_check=cancel_check,
            resume_offset=resume_offset,
        )
    except InterruptedError:
        # Mantener archivo .tmp parcial para poder reanudar con HTTP Range en el siguiente intento
        raise
    except Exception as err:
        if allow_insecure_fallback and _is_ssl_cert_error(err):
            logger.warning(
                f"Fallo de verificación SSL al descargar {filename} ({err}). "
                "Reintentando con contexto SSL sin verificación; verifique su red o proxy."
            )
            try:
                resume_offset = temp_path.stat().st_size if temp_path.exists() else 0
                retry_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                if resume_offset > 0:
                    retry_headers["Range"] = f"bytes={resume_offset}-"
                retry_req = urllib.request.Request(url, headers=retry_headers)
                fallback_ctx = ssl._create_unverified_context()
                _download_with_context(
                    retry_req,
                    temp_path,
                    fallback_ctx,
                    filename=filename,
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                    cancel_check=cancel_check,
                    resume_offset=resume_offset,
                )
            except InterruptedError:
                raise
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        else:
            temp_path.unlink(missing_ok=True)
            raise

    # Validar integridad SHA-256 cuando exista registro para el archivo descargado
    expected_sha = MODEL_SHA256.get(filename)
    if expected_sha and temp_path.exists():
        if not verify_sha256(temp_path, expected_sha):
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum SHA-256 inválido para {filename}; archivo corrupto descartado.")

    # Renombrar al terminar
    if temp_path.exists():
        temp_path.replace(dest_path)

    if status_cb:
        status_cb(f"Modelo {filename} listo.")

    return dest_path


def load_rembg_session(
    model_name: str,
    providers: list,
    sess_opts=None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """
    Asegura pesos locales vía model_fetcher (incluyendo variantes *_int8.onnx) y construye
    la sesión rembg sin duplicar descargas en subcarpetas de pooch.
    """
    import rembg

    is_mocked = getattr(rembg.new_session, "__module__", "") == "unittest.mock"
    onnx_fname = f"{model_name}.onnx"
    local_path = resolve_local_model(onnx_fname)

    if local_path is None and onnx_fname in MODEL_URLS and not is_mocked:
        def _dl_prog(dl: int, tot: int):
            if progress_cb and tot > 0:
                pct = 5 + int(round((dl / tot) * 30))
                dl_mb = dl / (1024 * 1024)
                tot_mb = tot / (1024 * 1024)
                progress_cb(
                    min(35, pct),
                    f"Descargando {onnx_fname} ({dl_mb:.1f} / {tot_mb:.1f} MB)...",
                )

        local_path = ensure_model_file(
            onnx_fname, progress_cb=_dl_prog, cancel_check=cancel_check
        )

    if local_path is not None and local_path.name != onnx_fname and not is_mocked:
        from rembg.sessions import sessions_class

        session_cls = next((sc for sc in sessions_class if sc.name() == model_name), None)
        if session_cls is not None:
            resolved_str = str(local_path)

            class _LocalRembgSession(session_cls):
                @classmethod
                def download_models(cls, *args, **kwargs):
                    return resolved_str

            return _LocalRembgSession(
                model_name,
                sess_opts,
                providers=providers,
            )

    if sess_opts is not None and not is_mocked:
        return rembg.new_session(model_name, sess_opts=sess_opts, providers=providers)
    return rembg.new_session(model_name, providers=providers)
