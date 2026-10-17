"""
TimoitaseBG - Motor de procesamiento y exportación de animaciones (.gif, .webp, .apng).
Incluye estabilización temporal de máscaras (anti-flicker), cuantización limpia de 255 colores + Alpha 1-bit para GIF,
y exportación de alta calidad sin pérdidas con canal alfa suave de 8 bits (0-255) para WebP y APNG.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageSequence


def is_animated_image(file_path: Path) -> bool:
    """Verifica si un archivo (.gif, .webp, .png/.apng) contiene múltiples fotogramas animados."""
    try:
        with Image.open(file_path) as im:
            return bool(getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1)
    except Exception:
        return False


def extract_gif_frames(gif_path: Path) -> Tuple[List[Image.Image], List[int], int]:
    """
    Extrae todos los fotogramas de una imagen animada (.gif, .webp, .png/.apng)
    componiéndolos en RGBA completo para evitar artefactos de frames parciales (delta frames).
    Retorna: (lista_frames_rgba, lista_duraciones_ms, loop_count)
    """
    frames: List[Image.Image] = []
    durations: List[int] = []

    with Image.open(gif_path) as im:
        is_gif = (getattr(im, "format", "") or "").upper() == "GIF"
        loop = int(im.info.get("loop", 0))
        default_duration = int(im.info.get("duration", 80)) or 80

        # Componer sobre lienzo base para respetar GIFs optimizados por deltas
        base_canvas = Image.new("RGBA", im.size, (0, 0, 0, 0))
        for frame in ImageSequence.Iterator(im):
            frame_rgba = frame.convert("RGBA")
            duration = int(frame.info.get("duration", default_duration)) or default_duration
            durations.append(max(20, duration))

            if not is_gif:
                composed = frame_rgba.copy()
            else:
                disposal = frame.info.get("disposal", 2)
                if disposal == 2:
                    # Restaurar a transparente en cada frame
                    composed = frame_rgba.copy()
                else:
                    composed = base_canvas.copy()
                    composed.alpha_composite(frame_rgba)
                    base_canvas = composed.copy()

            frames.append(composed)

    return frames, durations, loop



def rgba_to_gif_palette_frame(
    rgba_img: Image.Image,
    alpha_threshold: int = 128,
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Image.Image, bool]:
    """
    Convierte una imagen RGBA en un frame indexado (modo 'P') listo para GIF sin halos sucios:
    - Si bg_color_rgb está definido, compone el fondo sólido y usa 256 colores.
    - Si es transparente, reserva el índice 255 para la transparencia y cuantiza los colores en 255 tonos.
    Retorna: (frame_modo_P, usa_transparencia)
    """
    rgba = np.array(rgba_img.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]

    # Caso 1: El usuario eligió un color de fondo sólido en la barra lateral
    if bg_color_rgb is not None:
        from app.core.image_utils import apply_solid_background
        solid_img = apply_solid_background(rgba_img, bg_color_rgb).convert("RGB")
        return solid_img.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE), False

    # Caso 2: GIF con fondo transparente (1-bit alpha limpio sin borde negro)
    mask_transparent = alpha < alpha_threshold
    rgb_clean = rgba[:, :, :3].copy()
    # Limpiar píxeles transparentes para que no contaminen la paleta de colores
    rgb_clean[mask_transparent] = 0

    pil_rgb = Image.fromarray(rgb_clean, mode="RGB")
    # Cuantizar a 255 colores (dejando libre el índice 255 para transparencia)
    p_img = pil_rgb.quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)

    p_arr = np.array(p_img, dtype=np.uint8)
    p_arr[mask_transparent] = 255

    out_p = Image.fromarray(p_arr, mode="P")
    out_p.putpalette(p_img.getpalette())
    return out_p, True


def save_animated_gif(
    frames_rgba: List[Image.Image],
    output_path: Path,
    durations: List[int],
    loop: int = 0,
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
    alpha_threshold: int = 128,
):
    """
    Exporta una lista de fotogramas RGBA como un archivo .gif animado optimizado.
    """
    if not frames_rgba:
        raise ValueError("No hay fotogramas para exportar el GIF.")

    p_frames: List[Image.Image] = []
    has_trans = False

    for frame in frames_rgba:
        p_frame, is_trans = rgba_to_gif_palette_frame(
            frame,
            alpha_threshold=alpha_threshold,
            bg_color_rgb=bg_color_rgb,
        )
        p_frames.append(p_frame)
        has_trans = is_trans

    save_kwargs = {
        "save_all": True,
        "append_images": p_frames[1:],
        "duration": durations if len(durations) == len(p_frames) else 80,
        "loop": loop,
        "disposal": 2,  # Crítico: limpia el lienzo antes de dibujar el siguiente frame transparente
        "optimize": False,
    }
    if has_trans:
        save_kwargs["transparency"] = 255

    p_frames[0].save(str(output_path), format="GIF", **save_kwargs)


def save_animated_webp(
    frames_rgba: List[Image.Image],
    output_path: Path,
    durations: List[int],
    loop: int = 0,
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
    quality: int = 95,
    lossless: bool = False,
):
    """
    Exporta una lista de fotogramas RGBA como un archivo .webp animado de alta calidad.
    Preserva los 16.7 millones de colores y el canal alfa suave de 8 bits (0-255) sin bordes dentados.
    """
    if not frames_rgba:
        raise ValueError("No hay fotogramas para exportar el WebP animado.")

    from app.core.image_utils import apply_solid_background

    prepared_frames: List[Image.Image] = []
    for frame in frames_rgba:
        if bg_color_rgb is not None:
            prepared_frames.append(apply_solid_background(frame, bg_color_rgb).convert("RGB"))
        else:
            prepared_frames.append(frame.convert("RGBA"))

    save_kwargs = {
        "format": "WEBP",
        "save_all": True,
        "append_images": prepared_frames[1:],
        "duration": durations if len(durations) == len(prepared_frames) else 80,
        "loop": loop,
        "quality": quality,
        "lossless": lossless,
        "minimize_size": True,
        "allow_mixed": True,
    }
    prepared_frames[0].save(str(output_path), **save_kwargs)


def save_animated_apng(
    frames_rgba: List[Image.Image],
    output_path: Path,
    durations: List[int],
    loop: int = 0,
    bg_color_rgb: Optional[Tuple[int, int, int]] = None,
):
    """
    Exporta una lista de fotogramas RGBA como un archivo .png / .apng animado sin pérdida.
    """
    if not frames_rgba:
        raise ValueError("No hay fotogramas para exportar el APNG.")

    from app.core.image_utils import apply_solid_background

    prepared_frames: List[Image.Image] = []
    for frame in frames_rgba:
        if bg_color_rgb is not None:
            prepared_frames.append(apply_solid_background(frame, bg_color_rgb).convert("RGB"))
        else:
            prepared_frames.append(frame.convert("RGBA"))

    save_kwargs = {
        "format": "PNG",
        "save_all": True,
        "append_images": prepared_frames[1:],
        "duration": durations if len(durations) == len(prepared_frames) else 80,
        "loop": loop,
        "disposal": 1,
    }
    prepared_frames[0].save(str(output_path), **save_kwargs)

