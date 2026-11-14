"""
TimoitaseBG - Motor puro de efectos creativos, composición de fondos, crop y transformaciones.
Sin dependencias de PySide6.QtWidgets; opera íntegramente sobre imágenes PIL / NumPy / OpenCV.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

ASPECT_PRESETS = ("original", "512x512", "1:1", "16:9", "9:16")


def apply_sticker_outline(
    pil_img: Image.Image,
    width_px: int = 8,
    color_rgb: Tuple[int, int, int] = (255, 255, 255),
    expand_canvas: bool = True,
) -> Image.Image:
    """Agrega un borde tipo sticker alrededor de los píxeles opacos de una imagen RGBA."""
    rgba = pil_img.convert("RGBA")
    width_px = int(max(0, width_px))
    if width_px == 0:
        return rgba

    pad = width_px + 2 if expand_canvas else 0
    if pad > 0:
        padded = Image.new("RGBA", (rgba.width + pad * 2, rgba.height + pad * 2), (0, 0, 0, 0))
        padded.paste(rgba, (pad, pad))
        work = padded
    else:
        work = rgba

    arr = np.array(work, dtype=np.uint8)
    alpha = arr[:, :, 3]
    if int(alpha.max()) == 0:
        return work

    ksize = width_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=1)
    # Suavizado ligero en el borde exterior del sticker sin debilitar el interior
    softened = cv2.GaussianBlur(dilated, (3, 3), 0)
    outline_alpha = np.maximum(alpha, softened)

    outline_arr = np.zeros_like(arr)
    outline_arr[:, :, 0] = int(color_rgb[0])
    outline_arr[:, :, 1] = int(color_rgb[1])
    outline_arr[:, :, 2] = int(color_rgb[2])
    outline_arr[:, :, 3] = outline_alpha

    outline_img = Image.fromarray(outline_arr, mode="RGBA")
    return Image.alpha_composite(outline_img, work)


def apply_drop_shadow(
    pil_img: Image.Image,
    offset_x: int = 10,
    offset_y: int = 12,
    blur_radius: int = 16,
    opacity: float = 0.55,
    color_rgb: Tuple[int, int, int] = (0, 0, 0),
    expand_canvas: bool = True,
) -> Image.Image:
    """Proyecta una sombra paralela suave debajo de la silueta alfa del sujeto."""
    rgba = pil_img.convert("RGBA")
    blur_radius = int(max(0, blur_radius))
    opacity = float(max(0.0, min(1.0, opacity)))
    if opacity <= 0.0:
        return rgba

    pad_blur = blur_radius * 2 + 4
    if expand_canvas:
        pad_left = pad_blur + max(0, -int(offset_x))
        pad_right = pad_blur + max(0, int(offset_x))
        pad_top = pad_blur + max(0, -int(offset_y))
        pad_bottom = pad_blur + max(0, int(offset_y))
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0

    new_w = rgba.width + pad_left + pad_right
    new_h = rgba.height + pad_top + pad_bottom

    fg_canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    fg_canvas.paste(rgba, (pad_left, pad_top))

    fg_alpha = np.array(rgba, dtype=np.uint8)[:, :, 3]
    shadow_alpha = np.zeros((new_h, new_w), dtype=np.uint8)
    sx0 = pad_left + int(offset_x)
    sy0 = pad_top + int(offset_y)
    # Copiar con recorte seguro por si expand_canvas=False
    src_x0 = max(0, -sx0)
    src_y0 = max(0, -sy0)
    dst_x0 = max(0, sx0)
    dst_y0 = max(0, sy0)
    copy_w = min(rgba.width - src_x0, new_w - dst_x0)
    copy_h = min(rgba.height - src_y0, new_h - dst_y0)
    if copy_w > 0 and copy_h > 0:
        shadow_alpha[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = (
            fg_alpha[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
        )

    if blur_radius > 0:
        k = blur_radius * 2 + 1
        shadow_alpha = cv2.GaussianBlur(shadow_alpha, (k, k), 0)

    shadow_alpha = np.clip(shadow_alpha.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    shadow_arr = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    shadow_arr[:, :, 0] = int(color_rgb[0])
    shadow_arr[:, :, 1] = int(color_rgb[1])
    shadow_arr[:, :, 2] = int(color_rgb[2])
    shadow_arr[:, :, 3] = shadow_alpha

    shadow_img = Image.fromarray(shadow_arr, mode="RGBA")
    return Image.alpha_composite(shadow_img, fg_canvas)


def apply_aspect_preset(
    pil_img: Image.Image,
    preset: str = "original",
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """Ajusta el lienzo a una relación de aspecto o tamaño objetivo sin recortar al sujeto."""
    if not preset or preset == "original":
        return pil_img.copy()

    rgba = pil_img.convert("RGBA")
    fill_rgba = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), 255) if bg_color else (0, 0, 0, 0)

    if preset == "512x512":
        canvas = Image.new("RGBA", (512, 512), fill_rgba)
        contained = ImageOps.contain(rgba, (512, 512), method=Image.Resampling.LANCZOS)
        ox = (512 - contained.width) // 2
        oy = (512 - contained.height) // 2
        canvas.alpha_composite(contained, dest=(ox, oy))
        return canvas

    ratio_map = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "9:16": (9, 16),
    }
    if preset not in ratio_map:
        return rgba

    rw, rh = ratio_map[preset]
    w, h = rgba.size
    target_ratio = rw / rh
    current_ratio = w / max(1, h)
    if abs(current_ratio - target_ratio) < 1e-3:
        return rgba

    if current_ratio < target_ratio:
        new_h = h
        new_w = max(w, int(round(h * target_ratio)))
    else:
        new_w = w
        new_h = max(h, int(round(w / target_ratio)))

    canvas = Image.new("RGBA", (new_w, new_h), fill_rgba)
    canvas.alpha_composite(rgba, dest=((new_w - w) // 2, (new_h - h) // 2))
    return canvas


def compose_background_mode(
    fg_rgba: Image.Image,
    mode: str = "transparent",
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    original_rgb: Optional[Image.Image] = None,
    blur_radius: int = 20,
    bg_image: Optional[Image.Image] = None,
    aspect_preset: str = "original",
) -> Image.Image:
    """
    Compone la capa recortada sobre distintos modos de fondo:
      - "transparent": conserva transparencia RGBA
      - "solid": fondo de color sólido
      - "bokeh": fondo original desenfocado (retrato / bokeh)
      - "image": fotografía o textura personalizada ajustada por cover-fit
    """
    fg = fg_rgba.convert("RGBA")
    mode = (mode or "transparent").lower()

    if mode == "solid":
        base = Image.new("RGBA", fg.size, (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), 255))
        composed = Image.alpha_composite(base, fg)
    elif mode == "bokeh":
        src = original_rgb.convert("RGBA") if original_rgb is not None else fg.copy()
        if src.size != fg.size:
            src = src.resize(fg.size, Image.Resampling.LANCZOS)
        # Rellenar fondo base por si la referencia tuviera alfa parcial
        backdrop = Image.new("RGBA", fg.size, (18, 22, 30, 255))
        backdrop = Image.alpha_composite(backdrop, src)
        blurred = backdrop.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(1, int(blur_radius))))
        composed = Image.alpha_composite(blurred.convert("RGBA"), fg)
    elif mode == "image" and bg_image is not None:
        fitted_bg = ImageOps.fit(bg_image.convert("RGBA"), fg.size, method=Image.Resampling.LANCZOS)
        composed = Image.alpha_composite(fitted_bg, fg)
    else:
        composed = fg

    if aspect_preset and aspect_preset != "original":
        pad_color = bg_color if mode == "solid" else None
        composed = apply_aspect_preset(composed, aspect_preset, bg_color=pad_color)

    return composed


def rotate_flip_image(
    pil_img: Image.Image,
    rotate_deg: int = 0,
    flip_h: bool = False,
    flip_v: bool = False,
) -> Image.Image:
    """
    Rota la imagen en pasos de 90° en sentido horario (0, 90, 180, 270) y aplica volteo H/V.
    """
    out = pil_img.copy()
    deg = int(rotate_deg) % 360
    if deg == 90:
        out = out.transpose(Image.Transpose.ROTATE_270)
    elif deg == 180:
        out = out.transpose(Image.Transpose.ROTATE_180)
    elif deg == 270:
        out = out.transpose(Image.Transpose.ROTATE_90)

    if flip_h:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return out


def crop_image_box(
    pil_img: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Image.Image:
    """Recorta una imagen a las coordenadas especificadas con validación de límites."""
    w, h = pil_img.size
    x0 = max(0, min(w - 1, int(left)))
    y0 = max(0, min(h - 1, int(top)))
    x1 = max(x0 + 1, min(w, int(right)))
    y1 = max(y0 + 1, min(h, int(bottom)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("El rectángulo de recorte debe tener ancho y alto positivos.")
    return pil_img.crop((x0, y0, x1, y1))
