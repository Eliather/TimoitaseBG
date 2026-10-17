"""
TimoitaseBG - Procesador de Video (Extracción y Exportación Frame-by-Frame).

Soporta lectura de la mayoría de formatos de video modernos (MP4, MOV, WebM, MKV,
AVI, M4V, WMV, FLV, MPEG, MPG, OGV, 3GP, TS) mediante OpenCV/FFmpeg y exportación
a formatos libres de regalías / gratuitos:
  - WebM (VP8/VP9, estándar abierto y libre de regalías)
  - MKV (Matroska con FFV1 sin pérdida o VP8/MJPG, 100% libre)
  - MOV / MP4 (MPEG-4 Part 2 `mp4v` integrado sin licencias de pago)
  - AVI (Motion JPEG `MJPG` universal)
  - WebP Animado / APNG / GIF (con transparencia alfa nativa de 8 bits / 1 bit)
  - Secuencia PNG transparente en archivo `.zip` (universal para editores de video)
"""
import io
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.core.bg_remover import get_bg_remover
from app.core.gif_processor import (
    save_animated_apng,
    save_animated_gif,
    save_animated_webp,
)

# Extensiones de video soportadas para lectura
VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".m4v",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
    ".ogv",
    ".3gp",
    ".ts",
)

# Extensiones de salida de video/secuencia libres soportadas para exportación
FREE_VIDEO_EXPORT_EXTENSIONS = (
    ".webm",
    ".mkv",
    ".mov",
    ".mp4",
    ".avi",
    ".webp",
    ".apng",
    ".gif",
    ".zip",
)

# Mapa de códecs FourCC libres/integrados por contenedor para cv2.VideoWriter
_CONTAINER_CODECS = {
    ".webm": ("VP80", "VP90"),
    ".mkv": ("FFV1", "VP80", "MJPG"),
    ".mov": ("mp4v", "MJPG"),
    ".mp4": ("mp4v", "MJPG"),
    ".m4v": ("mp4v", "MJPG"),
    ".avi": ("MJPG", "IYUV"),
    ".ogv": ("THEO", "VP80", "MJPG"),
}


def is_video_file(file_path: Path | str) -> bool:
    """
    Detecta si el archivo corresponde a un formato de video soportado.
    """
    p = Path(file_path)
    if not p.is_file():
        return False
    return p.suffix.lower() in VIDEO_EXTENSIONS


def is_sequence_or_video_file(file_path: Path | str) -> bool:
    """
    Detecta si el archivo es un video o una secuencia animada (GIF/WebP/APNG).
    """
    from app.core.gif_processor import is_animated_image

    p = Path(file_path)
    return is_video_file(p) or is_animated_image(p)


def extract_video_frames(
    video_path: Path | str,
    max_frames: int = 600,
) -> Tuple[List[Image.Image], List[int], float]:
    """
    Extrae los fotogramas de un archivo de video convirtiéndolos a imágenes PIL RGBA.

    Si el video supera `max_frames`, realiza un submuestreo uniforme (stride)
    para evitar desbordamientos de memoria RAM manteniendo la fluidez temporal.

    Retorna:
        (frames_rgba, durations_ms, effective_fps)
    """
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el archivo de video: {video_path}")

    try:
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if raw_fps <= 0.0 or raw_fps > 240.0 or np.isnan(raw_fps):
            raw_fps = 24.0

        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = 1
        if max_frames > 0 and total_est > max_frames:
            stride = max(1, int(np.ceil(total_est / float(max_frames))))

        effective_fps = max(1.0, raw_fps / float(stride))
        frame_duration_ms = max(15, int(round(1000.0 / effective_fps)))

        frames: List[Image.Image] = []
        raw_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if raw_idx % stride == 0:
                frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
                frames.append(Image.fromarray(frame_rgba, mode="RGBA"))
                if 0 < max_frames <= len(frames):
                    break

            raw_idx += 1

        if not frames:
            raise ValueError(f"No se pudieron decodificar fotogramas del video: {video_path}")

        durations = [frame_duration_ms] * len(frames)
        return frames, durations, effective_fps
    finally:
        cap.release()


def process_video_frames(
    frames: List[Image.Image],
    model_name: str = "isnet-anime",
    preserve_clothing: bool = False,
    temporal_smoothing: float = 0.22,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Image.Image]:
    """
    Elimina el fondo fotograma a fotograma en una secuencia de video con
    estabilización temporal anti-parpadeo de bordes.
    """
    remover = get_bg_remover()
    total = len(frames)
    processed_frames: List[Image.Image] = []
    prev_alpha: Optional[np.ndarray] = None

    for idx, frame in enumerate(frames):
        if cancel_check is not None and cancel_check():
            break

        if progress_callback is not None:
            progress_callback(idx + 1, total)

        out_rgba = remover.remove_background(
            frame,
            model_name=model_name,
            enable_clothing_protection=preserve_clothing,
        )

        if temporal_smoothing > 0.0:
            arr = np.array(out_rgba.convert("RGBA"), dtype=np.uint8)
            curr_alpha = arr[:, :, 3].astype(np.float32)
            if prev_alpha is not None and prev_alpha.shape == curr_alpha.shape:
                diff = np.abs(curr_alpha - prev_alpha)
                blend_mask = diff < 45.0
                w = float(temporal_smoothing)
                curr_alpha = np.where(
                    blend_mask,
                    (1.0 - w) * curr_alpha + w * prev_alpha,
                    curr_alpha,
                )
                arr[:, :, 3] = np.clip(np.round(curr_alpha), 0, 255).astype(np.uint8)
                out_rgba = Image.fromarray(arr, mode="RGBA")
            prev_alpha = curr_alpha

        processed_frames.append(out_rgba)

    return processed_frames


def _composite_rgba_to_bgr(
    frame_rgba: Image.Image,
    bg_rgb: Tuple[int, int, int],
) -> np.ndarray:
    """
    Compone una imagen PIL RGBA sobre un color RGB sólido y devuelve un arreglo
    NumPy `(H, W, 3)` en orden BGR (`uint8`) listo para `cv2.VideoWriter`.
    """
    rgba = np.asarray(frame_rgba.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    fg_rgb = rgba[:, :, :3]
    bg_arr = np.array(bg_rgb, dtype=np.float32).reshape((1, 1, 3))
    comp_rgb = np.clip(fg_rgb * alpha + bg_arr * (1.0 - alpha), 0, 255).astype(np.uint8)
    return cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)


def _try_save_transparent_ffmpeg(
    frames_rgba: List[Image.Image],
    output_path: Path,
    fps: float,
) -> bool:
    """
    Si `ffmpeg` está disponible en el PATH del sistema, intenta codificar video
    con canal alfa real (transparencia nativa) usando códecs 100% libres:
      - `.webm`: VP9 con `yuva420p`
      - `.mkv`: FFV1 sin pérdida con `bgra`
      - `.mov`: PNG en contenedor QuickTime (`rgba`)
    Retorna True si tuvo éxito, False para hacer fallback a OpenCV VideoWriter.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not frames_rgba:
        return False

    ext = output_path.suffix.lower()
    if ext not in (".webm", ".mkv", ".mov"):
        return False

    base_w, base_h = frames_rgba[0].size
    if ext == ".webm":
        codec_args = [
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-b:v", "2M",
            "-deadline", "realtime",
            "-cpu-used", "4",
            "-auto-alt-ref", "0",
        ]
    elif ext == ".mkv":
        codec_args = [
            "-c:v", "ffv1",
            "-level", "3",
            "-pix_fmt", "bgra",
        ]
    elif ext == ".mov":
        codec_args = [
            "-c:v", "png",
            "-pix_fmt", "rgba",
        ]
    else:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{base_w}x{base_h}",
        "-r", f"{max(1.0, float(fps)):.2f}",
        "-i", "-",
        "-an",
        *codec_args,
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdin is not None
        for frm in frames_rgba:
            rgba_img = frm.convert("RGBA")
            if rgba_img.size != (base_w, base_h):
                rgba_img = rgba_img.resize((base_w, base_h), Image.Resampling.LANCZOS)
            proc.stdin.write(rgba_img.tobytes())
        proc.stdin.close()
        ret_code = proc.wait(timeout=120)
        return ret_code == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _save_png_zip_sequence(
    frames_rgba: List[Image.Image],
    output_path: Path,
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
) -> None:
    """
    Exporta todos los fotogramas como una secuencia numerada de PNGs (`frame_00001.png`...)
    dentro de un archivo `.zip` manteniendo transparencia RGBA de 8 bits.
    """
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for idx, frm in enumerate(frames_rgba, start=1):
            rgba_img = frm.convert("RGBA")
            if bg_color_rgb is not None:
                bg = Image.new("RGBA", rgba_img.size, (*bg_color_rgb, 255))
                bg.alpha_composite(rgba_img)
                out_img = bg.convert("RGB")
            else:
                out_img = rgba_img

            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            zf.writestr(f"frame_{idx:05d}.png", buf.getvalue())


def save_video_sequence(
    frames_rgba: List[Image.Image],
    output_path: Path | str,
    fps: float = 24.0,
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
    chroma_fallback_rgb: Tuple[int, int, int] = (0, 255, 0),
    loop: int = 0,
) -> None:
    """
    Guarda una secuencia de fotogramas RGBA en formatos de video o animación libres.

    Formatos soportados:
      - `.webp`, `.apng`, `.gif`: Delega a `save_animated_sequence` con canal alfa nativo.
      - `.zip`: Secuencia numerada de PNGs transparentes (`frame_00001.png`, ...).
      - `.webm`, `.mkv`, `.mov`, `.mp4`, `.avi`, `.ogv`:
          * Si `bg_color_rgb is None` y `ffmpeg` está en el PATH para `.webm`/`.mkv`/`.mov`,
            exporta con transparencia alfa nativa (`yuva420p` / `bgra` / `rgba`).
          * Con `cv2.VideoWriter` (3 canales BGR), compone sobre `bg_color_rgb` si el usuario
            eligió un color de fondo, o sobre `chroma_fallback_rgb` (por defecto Verde Chroma
            `#00FF00` para edición rápida en programas de video).
    """
    if not frames_rgba:
        raise ValueError("No hay fotogramas para exportar el video.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    safe_fps = max(1.0, min(240.0, float(fps or 24.0)))
    frame_ms = max(15, int(round(1000.0 / safe_fps)))

    # 1. Formatos animados con soporte alfa nativo en Pillow
    if ext in (".webp", ".apng", ".gif"):
        durations = [frame_ms] * len(frames_rgba)
        if ext == ".webp":
            save_animated_webp(
                frames_rgba=frames_rgba,
                output_path=out_path,
                durations=durations,
                loop=loop,
                bg_color_rgb=bg_color_rgb,
            )
        elif ext == ".gif":
            save_animated_gif(
                frames_rgba=frames_rgba,
                output_path=out_path,
                durations=durations,
                loop=loop,
                bg_color_rgb=bg_color_rgb,
            )
        else:
            save_animated_apng(
                frames_rgba=frames_rgba,
                output_path=out_path,
                durations=durations,
                loop=loop,
                bg_color_rgb=bg_color_rgb,
            )
        return

    # 2. Secuencia de PNGs transparentes en ZIP
    if ext == ".zip":
        _save_png_zip_sequence(frames_rgba, out_path, bg_color_rgb=bg_color_rgb)
        return

    # 3. Si no hay color de fondo sólido, intentar codificación con alfa nativo vía FFmpeg CLI
    if bg_color_rgb is None and ext in (".webm", ".mkv", ".mov"):
        if _try_save_transparent_ffmpeg(frames_rgba, out_path, safe_fps):
            return

    # 4. Codificación directa vía OpenCV VideoWriter con códecs libres integrados
    codecs_to_try = _CONTAINER_CODECS.get(ext, ("mp4v", "VP80", "MJPG"))
    base_w, base_h = frames_rgba[0].size
    effective_bg = bg_color_rgb if bg_color_rgb is not None else chroma_fallback_rgb

    writer: Optional[cv2.VideoWriter] = None
    for fourcc_str in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        candidate = cv2.VideoWriter(str(out_path), fourcc, safe_fps, (base_w, base_h))
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None or not writer.isOpened():
        raise RuntimeError(
            f"No se encontró un códec libre compatible en OpenCV para exportar '{ext}'."
        )

    try:
        for frm in frames_rgba:
            if frm.size != (base_w, base_h):
                frm = frm.resize((base_w, base_h), Image.Resampling.LANCZOS)
            bgr_frame = _composite_rgba_to_bgr(frm, effective_bg)
            writer.write(bgr_frame)
    finally:
        writer.release()
