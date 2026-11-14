"""
TimoitaseBG - Funciones puras de preprocesamiento y postprocesamiento de máscaras alfa.
Sin dependencias de PySide6 ni de gestión de sesiones ONNX.
"""
import logging
from typing import Any, Tuple, Union

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

logger = logging.getLogger("TimoitaseBG.Postprocess")

# Fallback de compatibilidad para versiones de Pillow anteriores a 9.1
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


def letterbox_image(
    img: Image.Image,
    target_size: int = 1024,
    pad_color: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Redimensiona la imagen manteniendo su proporción original (aspect ratio) y añade
    padding simétrico para formar un lienzo cuadrado de target_size x target_size.
    Retorna:
    - canvas: imagen PIL de target_size x target_size
    - crop_box: tupla (pad_x, pad_y, new_w, new_h) de la región útil
    - orig_size: tupla (w, h) del tamaño nativo original
    """
    w, h = img.size
    scale = min(target_size / max(1, w), target_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.convert("RGB").resize((new_w, new_h), resample=_LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, (pad_x, pad_y, new_w, new_h), (w, h)


def unletterbox_mask(
    mask_1024: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """
    Recorta el padding de la máscara predicha a 1024x1024 y la escala de vuelta
    con interpolación bilineal suave a las dimensiones nativas originales.
    """
    pad_x, pad_y, new_w, new_h = crop_box
    orig_w, orig_h = orig_size

    crop = mask_1024[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    return cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _letterbox_topleft(
    img: Image.Image,
    target_size: int = 448,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Redimensiona la imagen manteniendo su proporción original (aspect ratio) y la coloca
    en la esquina superior izquierda (0, 0) de un lienzo cuadrado.
    """
    w, h = img.size
    scale = min(target_size / max(1, w), target_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.convert("RGB").resize((new_w, new_h), resample=_LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    canvas.paste(resized, (0, 0))

    return canvas, (new_w, new_h), (w, h)


def _unletterbox_topleft(
    mask_canvas: np.ndarray,
    resized_size: Tuple[int, int],
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """
    Recorta el alpha predicho desde la esquina superior izquierda [:new_h, :new_w]
    del lienzo y lo escala con interpolación bilineal suave a las dimensiones originales.
    """
    new_w, new_h = resized_size
    orig_w, orig_h = orig_size

    crop = mask_canvas[:new_h, :new_w]
    return cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _preprocess_inspyrenet(input_img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Preprocesamiento específico para InSPyReNet (Swin-B 1024x1024):
    1. Redimensiona DIRECTO a 1024x1024 con interpolación bilineal.
    2. Escala a [0, 1] float32 y aplica normalización ImageNet: (x - mean) / std.
    3. Transpone a formato NCHW [1, 3, 1024, 1024].
    """
    orig_w, orig_h = input_img.size
    resized = input_img.convert("RGB").resize((1024, 1024), resample=_BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    return tensor, (orig_w, orig_h)


def _postprocess_inspyrenet_mask(out: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
    """
    Postprocesamiento para InSPyReNet:
    Reescala directamente el alfa predicho (1024x1024) a (orig_w, orig_h)
    con interpolación bicúbica sin ningún recorte de padding.
    """
    raw_1024 = np.squeeze(out).astype(np.float32)
    if raw_1024.min() < 0.0 or raw_1024.max() > 1.0:
        raw_1024 = 1.0 / (1.0 + np.exp(-raw_1024))
    orig_w, orig_h = orig_size
    resized = cv2.resize(raw_1024, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0.0, 1.0)


def postprocess_inspyrenet(
    input_img: Image.Image,
    raw_mask: np.ndarray,
    enable_despill: bool = False,
    antialiasing_radius: int = 5,
    min_island_area: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocesamiento especializado de alta fidelidad para InSPyReNet:
    1. Preserva el mapa continuo de alfa de alta resolución.
    2. Suprime motas o islas de ruido flotantes en el fondo.
    3. Aplica recuperación topológica de huecos internos encerrados.
    4. Aplica anti-aliasing subpíxel bilateral.
    5. Preserva los colores originales de la imagen.
    """
    rgb_arr = np.array(input_img.convert("RGB"), dtype=np.uint8)
    alpha = np.clip(raw_mask.astype(np.float32), 0.0, 1.0)

    if min_island_area > 0:
        binary = (alpha > 0.05).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            remove_lut = stats[:, cv2.CC_STAT_AREA] < min_island_area
            remove_lut[0] = False
            alpha[remove_lut[labels]] = 0.0

    try:
        bin_solid = (alpha > 0.4).astype(np.uint8) * 255
        h_m, w_m = bin_solid.shape
        flood = bin_solid.copy()
        cv2.floodFill(flood, np.zeros((h_m + 2, w_m + 2), dtype=np.uint8), (0, 0), 255)
        enclosed_holes = cv2.bitwise_not(flood)
        if np.any(enclosed_holes > 0):
            alpha = np.maximum(alpha, (enclosed_holes > 0).astype(np.float32))
    except Exception as e:
        logger.debug(f"Hole recovery omitido: {e}")

    if antialiasing_radius > 0:
        alpha = cv2.bilateralFilter(
            alpha,
            d=antialiasing_radius,
            sigmaColor=0.15,
            sigmaSpace=1.5,
        )
        alpha = np.clip(alpha, 0.0, 1.0)

    cleaned_rgb = rgb_arr.copy()
    final_alpha = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    return cleaned_rgb, final_alpha


def _guided_filter_box(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Implementación de respaldo del Guided Filter mediante boxFilter si ximgproc no estuviera disponible."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    if len(guide.shape) == 3:
        I = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        I = guide.astype(np.float32) / 255.0

    p = src
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


def apply_guided_filter(
    guide_bgr: np.ndarray,
    prob_mask: np.ndarray,
    radius: int = 8,
    eps: float = 1e-6,
    band_dilation: int = 18,
) -> np.ndarray:
    """
    Refina las transiciones del mapa de probabilidad usando la imagen original como guía (Guided Filter)
    sobre la franja perimetral del sujeto.
    """
    try:
        guide_uint8 = np.ascontiguousarray(guide_bgr, dtype=np.uint8)
        src_f32 = np.ascontiguousarray(prob_mask, dtype=np.float32)
        h, w = src_f32.shape

        binary = (src_f32 >= 0.50).astype(np.uint8)
        ksize = 2 * band_dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(binary, kernel)
        eroded = cv2.erode(binary, kernel)
        band = (dilated > 0) & (eroded == 0)

        if not np.any(band):
            return src_f32

        ys, xs = np.where(band)
        margin = radius * 2 + 2
        ymin = max(0, int(ys.min()) - margin)
        ymax = min(h, int(ys.max()) + margin + 1)
        xmin = max(0, int(xs.min()) - margin)
        xmax = min(w, int(xs.max()) + margin + 1)

        crop_guide = guide_uint8[ymin:ymax, xmin:xmax]
        crop_src = src_f32[ymin:ymax, xmin:xmax]

        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            crop_refined = cv2.ximgproc.guidedFilter(
                guide=crop_guide,
                src=crop_src,
                radius=radius,
                eps=eps,
            )
        else:
            crop_refined = _guided_filter_box(crop_guide, crop_src, radius, eps)

        crop_band = band[ymin:ymax, xmin:xmax]
        out = src_f32.copy()
        crop_out = out[ymin:ymax, xmin:xmax]
        crop_out[crop_band] = np.clip(crop_refined[crop_band], 0.0, 1.0)
        return out
    except Exception as e:
        logger.exception(f"Error aplicando Guided Filter con ROI: {e}. Usando mapa crudo.")
        return src_f32


def apply_guided_filter_ambiguous(
    guide_bgr: np.ndarray,
    prob_mask: np.ndarray,
    radius: int = 8,
    eps: float = 1e-4,
    low_thresh: float = 0.15,
    high_thresh: float = 0.85,
    band_dilation: int = 10,
) -> np.ndarray:
    """
    Aplica Guided Filter exclusivamente sobre las franjas de baja confianza (píxeles ambiguos).
    """
    prob = np.clip(prob_mask, 0.0, 1.0).astype(np.float32)
    ambiguous = (prob >= low_thresh) & (prob <= high_thresh)
    if not np.any(ambiguous):
        return prob

    ksize = max(3, band_dilation * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    band = cv2.dilate(ambiguous.astype(np.uint8), kernel) > 0

    ys, xs = np.where(band)
    if len(ys) == 0:
        return prob

    margin = radius * 2 + 2
    ymin = max(0, int(ys.min()) - margin)
    ymax = min(prob.shape[0], int(ys.max()) + margin + 1)
    xmin = max(0, int(xs.min()) - margin)
    xmax = min(prob.shape[1], int(xs.max()) + margin + 1)

    guide_crop = guide_bgr[ymin:ymax, xmin:xmax]
    p_crop = prob[ymin:ymax, xmin:xmax]

    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            filtered_crop = cv2.ximgproc.guidedFilter(guide_crop, p_crop, radius=radius, eps=eps)
        else:
            filtered_crop = _guided_filter_box(guide_crop, p_crop, radius=radius, eps=eps)
    except Exception as e:
        logger.exception(f"Error en Guided Filter de franjas ambiguas: {e}")
        return prob

    refined = prob.copy()
    amb_crop = ambiguous[ymin:ymax, xmin:xmax]
    crop_ref = refined[ymin:ymax, xmin:xmax]
    crop_ref[amb_crop] = np.clip(filtered_crop[amb_crop], 0.0, 1.0)
    return refined


def postprocess_mask(
    raw_mask: np.ndarray,
    high_thresh: float = 0.50,
    low_thresh: float = 0.50,
    morph_ksize: int = 7,
    max_hole_area: int = 60,
) -> np.ndarray:
    """
    Genera el canal alfa continuo final preservando el anti-aliasing y gradiente sub-píxel real
    con hysteresis tipo Canny, cierre morfológico y relleno selectivo de huecos internos.
    """
    prob = np.clip(raw_mask, 0.0, 1.0).astype(np.float32)
    low = float(min(high_thresh, low_thresh))
    high = float(max(high_thresh, low_thresh))

    if high == low:
        binary_mask = (prob >= high).astype(np.uint8)
    else:
        candidate = (prob >= low).astype(np.uint8)
        strong = prob >= high
        if np.any(strong):
            num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)
            strong_labels = np.unique(labels[strong])
            strong_labels = strong_labels[strong_labels > 0]
            lookup = np.zeros(num_labels, dtype=bool)
            lookup[strong_labels] = True
            binary_mask = lookup[labels].astype(np.uint8)
        else:
            binary_mask = np.zeros_like(prob, dtype=np.uint8)

    if morph_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    else:
        closed = binary_mask

    filled = scipy.ndimage.binary_fill_holes(closed).astype(np.uint8)
    holes = (filled == 1) & (binary_mask == 0)

    if max_hole_area > 0 and np.any(holes):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
        if num_labels > 1:
            hole_lut = stats[:, cv2.CC_STAT_AREA] <= max_hole_area
            hole_lut[0] = False
            small_holes = hole_lut[labels]
        else:
            small_holes = np.zeros_like(holes, dtype=bool)
        internal_holes = small_holes & (prob < low)
    elif max_hole_area <= 0 and np.any(holes):
        internal_holes = holes
    else:
        internal_holes = np.zeros_like(holes, dtype=bool)

    alpha_continuous = np.where(internal_holes, 1.0, np.where(closed > 0, prob, 0.0))
    return np.clip(alpha_continuous * 255.0, 0, 255).astype(np.uint8)


def fuse_clothing_protection_core_only(
    main_alpha: np.ndarray,
    clothing_alpha: np.ndarray,
    inner_erosion_px: int = 12,
    cloth_confidence_thresh: int = 75,
) -> np.ndarray:
    """
    Fusiona la máscara de ropa (U2-Net 320x320) con la máscara principal
    restringiendo la inyección al núcleo interior para evitar perforaciones en prendas claras.
    """
    if main_alpha is None or clothing_alpha is None:
        return main_alpha

    h, w = main_alpha.shape[:2]
    if clothing_alpha.shape[:2] != (h, w):
        clothing_alpha = cv2.resize(clothing_alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    scale_factor = max(1.0, min(h, w) / 1024.0)
    eff_erosion = max(6, int(round(inner_erosion_px * scale_factor)))

    ksize = 2 * eff_erosion + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    eroded_clothing = cv2.erode(clothing_alpha, kernel)

    main_binary = (main_alpha > 30).astype(np.uint8)
    if np.any(main_binary):
        close_k = max(15, int(round(25 * scale_factor)) | 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        main_closed = cv2.morphologyEx(main_binary, cv2.MORPH_CLOSE, close_kernel)
        main_solid = scipy.ndimage.binary_fill_holes(main_closed).astype(np.uint8)
        safe_main_core = cv2.erode(main_solid, kernel)
    else:
        safe_main_core = np.zeros_like(main_alpha, dtype=np.uint8)

    valid_clothing = (eroded_clothing >= cloth_confidence_thresh) | (
        (clothing_alpha >= cloth_confidence_thresh) & (safe_main_core > 0)
    )

    if not np.any(valid_clothing):
        return main_alpha

    injection_alpha = np.where(valid_clothing, eroded_clothing, 0).astype(np.uint8)
    blur_k = max(5, int(round(7 * scale_factor)) | 1)
    injection_alpha = cv2.GaussianBlur(injection_alpha, (blur_k, blur_k), sigmaX=2.0)

    return np.maximum(main_alpha, injection_alpha)


def _predict_clothing_mask(
    input_img: Image.Image,
    human_sess: Any,
    max_side: int = 640,
) -> np.ndarray:
    """
    Ejecuta u2net_human_seg sobre una copia reducida (máx. max_side px).
    """
    import rembg

    w, h = input_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        work_img = input_img.resize((small_w, small_h), resample=_BILINEAR)
    else:
        work_img = input_img

    human_out = rembg.remove(work_img, session=human_sess)
    if isinstance(human_out, Image.Image):
        if human_out.mode == "RGBA":
            return np.array(human_out.getchannel("A"), dtype=np.uint8)
        return np.array(human_out.convert("L"), dtype=np.uint8)
    return np.asarray(human_out, dtype=np.uint8)


def generate_trimap(
    alpha_mask: Union[np.ndarray, Image.Image],
    erode_px: int = 15,
    dilate_px: int = 15,
) -> np.ndarray:
    """
    Genera un trimap estándar de 3 niveles {0, 128, 255} a partir de una máscara alfa.
    """
    if isinstance(alpha_mask, Image.Image):
        if alpha_mask.mode == "RGBA":
            alpha_arr = np.array(alpha_mask.split()[-1], dtype=np.uint8)
        else:
            alpha_arr = np.array(alpha_mask.convert("L"), dtype=np.uint8)
    else:
        alpha_arr = np.asarray(alpha_mask)
        if np.issubdtype(alpha_arr.dtype, np.floating):
            if alpha_arr.max() <= 1.0:
                alpha_arr = np.clip(np.round(alpha_arr * 255.0), 0, 255).astype(np.uint8)
            else:
                alpha_arr = np.clip(alpha_arr, 0, 255).astype(np.uint8)
        else:
            alpha_arr = alpha_arr.astype(np.uint8)

    binary = (alpha_arr >= 128).astype(np.uint8) * 255

    if erode_px > 0:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        fg_mask = cv2.erode(binary, k_erode)
    else:
        fg_mask = binary.copy()

    if dilate_px > 0:
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        dilated_mask = cv2.dilate(binary, k_dilate)
    else:
        dilated_mask = binary.copy()

    trimap = np.zeros_like(binary, dtype=np.uint8)
    trimap[dilated_mask > 0] = 128
    trimap[fg_mask > 0] = 255

    return trimap


def refine_alpha_edges(
    pil_img: Image.Image,
    choke_px: int = 0,
    feather_px: int = 0,
    despill_strength: float = 0.0,
) -> Image.Image:
    """
    Refina los bordes del canal alfa mediante Choke (erosión/dilatación),
    Feather (suavizado restringido al perímetro) y Despill (descontaminación cromática).
    """
    rgba = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]

    if choke_px != 0:
        r = abs(int(choke_px))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        alpha = cv2.erode(alpha, k) if choke_px > 0 else cv2.dilate(alpha, k)

    if feather_px > 0:
        f = int(feather_px)
        ksize = 2 * f + 1
        blurred = cv2.GaussianBlur(alpha.astype(np.float32), (ksize, ksize), sigmaX=max(0.5, f * 0.65))
        # Limitar el suavizado a la banda de transición alrededor del contorno
        band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        edge_band = cv2.dilate((alpha > 127).astype(np.uint8) * 255, band_k) - cv2.erode(
            (alpha > 127).astype(np.uint8) * 255, band_k
        )
        band_weight = cv2.GaussianBlur(edge_band.astype(np.float32) / 255.0, (5, 5), 0)
        alpha = np.clip(
            np.round(alpha.astype(np.float32) * (1.0 - band_weight) + blurred * band_weight), 0, 255
        ).astype(np.uint8)

    if despill_strength > 0.0:
        strength = float(max(0.0, min(1.0, despill_strength)))
        rgb_f = rgba[..., :3].astype(np.float32)
        fg_core = (alpha >= 230).astype(np.float32)[..., None]
        if np.any(fg_core > 0.0):
            # Propagar el color del núcleo opaco hacia el halo semitransparente perimetral
            k_bleed = (15, 15)
            weighted_rgb = cv2.GaussianBlur(rgb_f * fg_core, k_bleed, sigmaX=4.0)
            weight_sum = cv2.GaussianBlur(fg_core[:, :, 0], k_bleed, sigmaX=4.0)[..., None]
            clean_rgb = np.where(weight_sum > 1e-4, weighted_rgb / np.maximum(weight_sum, 1e-4), rgb_f)

            alpha_f = alpha.astype(np.float32) / 255.0
            halo_mask = ((alpha > 0) & (alpha < 235)).astype(np.float32) * (1.0 - alpha_f * 0.75)
            mix = (halo_mask * strength)[..., None]
            rgba[..., :3] = np.clip(np.round(rgb_f * (1.0 - mix) + clean_rgb * mix), 0, 255).astype(np.uint8)

    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")
