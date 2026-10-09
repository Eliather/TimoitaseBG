"""
TimoitaseBG - Descargador y gestor de pesos de modelos locales autocontenidos.
"""
import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from app.config import MODELS_DIR

# URLs de descarga directa verificadas
MODEL_URLS = {
    "inspyrenet.onnx": "https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/resolve/main/onnx/model.onnx",
    "RealESRGAN_x4plus_anime_6B.onnx": "https://huggingface.co/deepghs/imgutils-models/resolve/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx",
    "mobile_sam_image_encoder.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
    "sam_mask_decoder_single.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx",
    "rmbg-2.0.onnx": "https://huggingface.co/yamura4/RMBG-2.0-ONNX/resolve/main/onnx/model_fp16.onnx",
    "yolo11n-seg.onnx": "https://huggingface.co/unileon-robotics/YOLO11-ONNX/resolve/main/yolo11n-seg.onnx",
}


def ensure_model_file(
    filename: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Garantiza que el archivo de modelo esté descargado localmente en app/models.
    Si no existe, lo descarga mostrando progreso.
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

    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB por bloque

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if progress_cb and total_size > 0:
                progress_cb(downloaded, total_size)

    # Renombrar al terminar
    if temp_path.exists():
        temp_path.replace(dest_path)

    if status_cb:
        status_cb(f"Modelo {filename} listo.")

    return dest_path
