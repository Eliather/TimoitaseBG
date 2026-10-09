"""
TimoitaseBG - Módulo de eliminación de fondos con letterboxing, post-procesamiento morfológico y ensamble híbrido.
"""
import os
import logging
import threading
from typing import Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from app.config import get_inference_providers, get_optimized_session_options

logger = logging.getLogger("TimoitaseBG.BGRemover")

# Corrección bug 9: Fallback de compatibilidad para versiones de Pillow anteriores a 9.1
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

    # Corrección bug 9: Uso de _LANCZOS con fallback
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


def _preprocess_rmbg2(
    input_img: Image.Image,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Preprocesamiento para RMBG-2.0 (BiRefNet 1024x1024):
    1. Letterboxing simétrico a 1024x1024 preservando aspect ratio nativo.
    2. Conversión a float32 y normalización ImageNet: (x/255.0 - mean) / std.
    3. Formato NCHW [1, 3, 1024, 1024].
    """
    canvas, crop_box, orig_size = letterbox_image(input_img, target_size=1024)
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std
    tensor = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
    return tensor, crop_box, orig_size


def _postprocess_rmbg2_mask(
    out: Any,
    crop_box: Tuple[int, int, int, int],
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """
    Postprocesamiento para RMBG-2.0:
    Extrae la máscara de salida, aplica sigmoid si es necesario,
    recorta el padding del letterbox y redimensiona a (orig_w, orig_h).
    Retorna máscara uint8 [0, 255].
    """
    if isinstance(out, (list, tuple)):
        arr = out[0]
    else:
        arr = out

    if arr.ndim == 4:
        mask_1024 = arr[0, 0]
    elif arr.ndim == 3:
        mask_1024 = arr[0]
    else:
        mask_1024 = np.squeeze(arr)

    # Sigmoid si los valores son logits
    if np.any(mask_1024 < 0.0) or np.any(mask_1024 > 1.0):
        mask_1024 = 1.0 / (1.0 + np.exp(-np.clip(mask_1024, -20.0, 20.0)))

    mask_1024 = np.clip(mask_1024, 0.0, 1.0)
    unletterboxed = unletterbox_mask(mask_1024, crop_box, orig_size)
    return np.clip(np.round(unletterboxed * 255.0), 0, 255).astype(np.uint8)


def _letterbox_topleft(
    img: Image.Image,
    target_size: int = 448,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Redimensiona la imagen manteniendo su proporción original (aspect ratio) y la coloca
    en la esquina superior izquierda (0, 0) de un lienzo cuadrado (por defecto negro de 448x448),
    según la especificación oficial de withoutBG (v10.0.0).
    Retorna:
    - canvas: imagen PIL de target_size x target_size
    - resized_size: tupla (new_w, new_h) de las dimensiones de la imagen escalada en (0, 0)
    - orig_size: tupla (w, h) del tamaño nativo original
    """
    w, h = img.size
    scale = min(target_size / max(1, w), target_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.convert("RGB").resize((new_w, new_h), resample=_LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    # Pegado en la esquina superior izquierda (0, 0) sin centrar según spec oficial de withoutBG
    canvas.paste(resized, (0, 0))

    return canvas, (new_w, new_h), (w, h)


def _unletterbox_topleft(
    mask_canvas: np.ndarray,
    resized_size: Tuple[int, int],
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """
    Recorta el alpha predicho desde la esquina superior izquierda [:new_h, :new_w]
    del lienzo (descartando el relleno inferior/derecho) y lo escala con interpolación
    bilineal suave a las dimensiones nativas originales (orig_w, orig_h).
    """
    new_w, new_h = resized_size
    orig_w, orig_h = orig_size

    crop = mask_canvas[:new_h, :new_w]
    return cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def _preprocess_inspyrenet(input_img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Preprocesamiento específico para InSPyReNet (Swin-B 1024x1024):
    1. Redimensiona DIRECTO a 1024x1024 con interpolación bilineal (sin preservar aspect ratio y sin letterbox/padding).
    2. Escala a [0, 1] float32 y aplica normalización ImageNet: (x - mean) / std.
    3. Transpone a formato NCHW [1, 3, 1024, 1024].
    Retorna:
    - tensor: np.ndarray de forma (1, 3, 1024, 1024)
    - orig_size: tupla (orig_w, orig_h)
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
    1. Preserva el mapa continuo de alfa de alta resolución sin umbralización dura ni recorte morfológico.
    2. Suprime motas o islas de ruido flotantes en el fondo.
    3. Aplica recuperación topológica de huecos internos encerrados (evita perforaciones en ropa, mangas o piel).
    4. Aplica anti-aliasing subpíxel bilateral para eliminar dientes de sierra preservando bordes nítidos.
    5. Preserva el 100% de los colores originales de la imagen sin difuminar ni alterar píxeles del sujeto.

    Retorna:
    - cleaned_rgb: np.ndarray (H, W, 3) uint8 con los colores originales 100% respetados.
    - final_alpha: np.ndarray (H, W) uint8 con el canal alfa perfectamente suavizado y anti-aliasado.
    """
    rgb_arr = np.array(input_img.convert("RGB"), dtype=np.uint8)
    alpha = np.clip(raw_mask.astype(np.float32), 0.0, 1.0)

    # 1. Supresión de islas de ruido flotantes en el fondo
    if min_island_area > 0:
        binary = (alpha > 0.05).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_island_area:
                alpha[labels == lbl] = 0.0

    # 2. Recuperación topológica de huecos internos (evita perforaciones en ropa, mangas o piel que coincidan con el fondo)
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

    # 3. Anti-aliasing subpíxel bilateral (elimina dientes de sierra preservando bordes nítidos)
    if antialiasing_radius > 0:
        alpha = cv2.bilateralFilter(
            alpha,
            d=antialiasing_radius,
            sigmaColor=0.15,
            sigmaSpace=1.5,
        )
        alpha = np.clip(alpha, 0.0, 1.0)

    # 4. Fidelidad cromática absoluta: conservar colores originales sin difuminar
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
    Refina las transiciones del mapa de probabilidad usando la imagen original como guía (Guided Filter).
    Optimizado por Región de Interés (ROI / Franja perimetral):
    1. Genera una máscara binaria inicial (> 0.50) y dilata/erosiona el perímetro ~15-20px para definir
       la franja de transición donde el borde del sujeto interactúa con el fondo.
    2. Recorta el bounding box que contiene dicha franja con margen de seguridad.
    3. Aplica el Guided Filter exclusivamente sobre ese recorte.
    4. Reinserta los valores refinados únicamente en la franja del borde, preservando el interior sólido
       del sujeto y el fondo exterior intactos sin alterar texturas internas ni crear niebla en el fondo.
    """
    try:
        guide_uint8 = np.ascontiguousarray(guide_bgr, dtype=np.uint8)
        src_f32 = np.ascontiguousarray(prob_mask, dtype=np.float32)
        h, w = src_f32.shape

        # 1. Delimitar la franja perimetral (borde ± band_dilation px)
        binary = (src_f32 >= 0.50).astype(np.uint8)
        ksize = 2 * band_dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(binary, kernel)
        eroded = cv2.erode(binary, kernel)
        band = (dilated > 0) & (eroded == 0)

        # Si no hay borde discernible (imagen completamente vacía o llena), retornar mapa original
        # Corrección bug 3: Retornar src_f32 (float32 contiguo) en lugar de prob_mask para consistencia de tipo
        if not np.any(band):
            return src_f32

        # 2. Bounding box ajustado a la franja de transición con margen para el kernel del Guided Filter
        ys, xs = np.where(band)
        margin = radius * 2 + 2
        ymin = max(0, int(ys.min()) - margin)
        ymax = min(h, int(ys.max()) + margin + 1)
        xmin = max(0, int(xs.min()) - margin)
        xmax = min(w, int(xs.max()) + margin + 1)

        crop_guide = guide_uint8[ymin:ymax, xmin:xmax]
        crop_src = src_f32[ymin:ymax, xmin:xmax]

        # 3. Aplicar Guided Filter únicamente sobre la región de interés recortada
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            crop_refined = cv2.ximgproc.guidedFilter(
                guide=crop_guide,
                src=crop_src,
                radius=radius,
                eps=eps,
            )
        else:
            crop_refined = _guided_filter_box(crop_guide, crop_src, radius, eps)

        # 4. Reinsertar el resultado refinado exclusivamente dentro de los píxeles de la franja
        crop_band = band[ymin:ymax, xmin:xmax]
        out = src_f32.copy()
        crop_out = out[ymin:ymax, xmin:xmax]
        crop_out[crop_band] = np.clip(crop_refined[crop_band], 0.0, 1.0)
        # Corrección bug 6: Eliminada asignación no-op out[ymin:ymax, xmin:xmax] = crop_out (crop_out es una vista en memoria)
        return out
    except Exception as e:
        # Corrección bug 7: logger.exception para conservar stack trace completo en producción
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
    Aplica Guided Filter exclusivamente sobre las franjas de baja confianza (píxeles ambiguos)
    donde el modelo de matting (como withoutBG) presenta incertidumbre ([low_thresh, high_thresh]).
    Aprovecha la imagen original en alta resolución como guía para resolver huecos finos y transiciones delicadas.
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
        # Corrección bug 7: logger.exception para conservar stack trace completo en producción
        logger.exception(f"Error en Guided Filter de franjas ambiguas: {e}")
        return prob

    refined = prob.copy()
    amb_crop = ambiguous[ymin:ymax, xmin:xmax]
    crop_ref = refined[ymin:ymax, xmin:xmax]
    crop_ref[amb_crop] = np.clip(filtered_crop[amb_crop], 0.0, 1.0)
    # Corrección bug 6: Eliminada asignación no-op refined[ymin:ymax, xmin:xmax] = crop_ref (crop_ref es una vista en memoria)
    return refined


def postprocess_mask(
    raw_mask: np.ndarray,
    high_thresh: float = 0.50,
    low_thresh: float = 0.50,
    morph_ksize: int = 7,
    max_hole_area: int = 60,
) -> np.ndarray:
    """
    Genera el canal alfa continuo final preservando el anti-aliasing y gradiente sub-píxel real:
    1. Binarización interna con hysteresis tipo Canny (8-conectividad):
       - Píxeles >= high_thresh: foreground seguro.
       - Píxeles entre low_thresh y high_thresh: foreground únicamente si están conectados a foreground seguro.
       - Píxeles < low_thresh: background seguro.
    2. Cierre morfológico (cv2.morphologyEx MORPH_CLOSE) para conectar trazos finos (si morph_ksize > 0).
    3. Relleno de microhuecos internos (scipy.ndimage.binary_fill_holes):
       - max_hole_area > 0: filtra huecos por área máxima para evitar sellar oclusiones reales del fondo.
       - max_hole_area <= 0: desactiva el filtro de tamaño y rellena todos los huecos internos sin límite.
    4. El canal alfa de salida preserva el mapa continuo para regiones de foreground,
       inyecta 1.0 (255) en los huecos internos confirmados, y anula (0) el background seguro.
    """
    prob = np.clip(raw_mask, 0.0, 1.0).astype(np.float32)
    low = float(min(high_thresh, low_thresh))
    high = float(max(high_thresh, low_thresh))

    # 1. Corrección bug 1: Hysteresis real tipo Canny (8-conectividad)
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

    # 2. Cierre morfológico para conectar trazos finos (si morph_ksize > 0)
    if morph_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    else:
        closed = binary_mask

    # 3. Relleno de huecos internos
    filled = scipy.ndimage.binary_fill_holes(closed).astype(np.uint8)
    holes = (filled == 1) & (binary_mask == 0)

    if max_hole_area > 0 and np.any(holes):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
        small_holes = np.zeros_like(holes, dtype=bool)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
                small_holes |= (labels == i)
        internal_holes = small_holes & (prob < low)
    elif max_hole_area <= 0 and np.any(holes):
        # Corrección mejora 11: max_hole_area <= 0 desactiva el límite de área y rellena todos los huecos internos
        internal_holes = holes
    else:
        internal_holes = np.zeros_like(holes, dtype=bool)

    # 4. Inyectar opacidad completa (1.0) en huecos internos y preservar gradiente continuo en foreground
    # Background seguro y componentes desconectadas quedan en 0.0
    alpha_continuous = np.where(internal_holes, 1.0, np.where(closed > 0, prob, 0.0))

    # 5. Escalar mapa continuo a 0-255 uint8
    return np.clip(alpha_continuous * 255.0, 0, 255).astype(np.uint8)


def generate_trimap(
    alpha_mask: Union[np.ndarray, Image.Image],
    erode_px: int = 15,
    dilate_px: int = 15,
) -> np.ndarray:
    """
    Genera un trimap estándar de 3 niveles para matting a partir de una máscara alfa:
    - 255: Foreground seguro (máscara binaria fuertemente erosionada)
    - 0: Background seguro (exterior a la máscara binaria dilatada)
    - 128: Zona desconocida (franja de transición entre erosión y dilatación)

    Args:
        alpha_mask: Máscara 2D (uint8 o float) o PIL Image en escala de grises/RGBA.
        erode_px: Radio en píxeles para la erosión morfológica del foreground.
        dilate_px: Radio en píxeles para la dilatación morfológica del background.

    Returns:
        np.ndarray de forma (H, W) y tipo uint8 con valores exclusivamente en {0, 128, 255}.
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

    # 1. Binarizar alfa final en 128
    binary = (alpha_arr >= 128).astype(np.uint8) * 255

    # 2. Erosionar fuertemente la máscara binaria -> región "foreground seguro" (255)
    if erode_px > 0:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        fg_mask = cv2.erode(binary, k_erode)
    else:
        fg_mask = binary.copy()

    # 3. Dilatar la máscara binaria -> delimita "background seguro" (0 en el exterior)
    if dilate_px > 0:
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        dilated_mask = cv2.dilate(binary, k_dilate)
    else:
        dilated_mask = binary.copy()

    # 4. Construir trimap con zona desconocida (128) entre erosión y dilatación
    trimap = np.zeros_like(binary, dtype=np.uint8)
    trimap[dilated_mask > 0] = 128
    trimap[fg_mask > 0] = 255

    return trimap


def refine_with_vitmatte(
    image_rgb: Union[Image.Image, np.ndarray],
    trimap: Union[Image.Image, np.ndarray],
    session: Any = None,
) -> np.ndarray:
    """
    Refinamiento de matting de segunda etapa usando ViTMatte:
    1. Preprocesamiento según especificación oficial (VitMatteImageProcessor):
       - Normalización RGB: rescale / 255.0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] -> rango [-1.0, 1.0].
       - Normalización Trimap: rescale / 255.0 sin sustracción de mean/std -> rango [0.0, 1.0].
       - Concatenación de 4 canales [RGB, Trimap] en formato NCHW.
       - Acolchado a la derecha/abajo divisible por 32 (size_divisor=32).
    2. Inferencia ONNX produciendo mapa continuo de alfa.
    3. Fusión estricta:
       - 255 donde trimap original decía "foreground seguro" (255).
       - 0 donde decía "background seguro" (0).
       - Predicción de ViTMatte (0-255 uint8) EXCLUSIVAMENTE en la zona desconocida (trimap == 128).

    Returns:
        np.ndarray de forma (H, W) y tipo uint8 con el canal alfa refinado.
    """
    if isinstance(image_rgb, Image.Image):
        rgb_arr = np.array(image_rgb.convert("RGB"), dtype=np.uint8)
    else:
        rgb_arr = np.asarray(image_rgb, dtype=np.uint8)
        if rgb_arr.ndim == 2:
            rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2RGB)
        elif rgb_arr.shape[2] == 4:
            rgb_arr = rgb_arr[:, :, :3]

    if isinstance(trimap, Image.Image):
        trimap_arr = np.array(trimap.convert("L"), dtype=np.uint8)
    else:
        trimap_arr = np.asarray(trimap, dtype=np.uint8)

    h, w = rgb_arr.shape[:2]

    if session is None:
        remover = get_bg_remover()
        session = remover.get_vitmatte_session()

    # 1. Normalización de imagen RGB a [-1.0, 1.0]
    img_f32 = rgb_arr.astype(np.float32) / 255.0
    img_norm = (img_f32 - 0.5) / 0.5

    # 2. Rescale de Trimap a [0.0, 1.0] (sin normalización por mean/std)
    trimap_f32 = (trimap_arr.astype(np.float32) / 255.0)[:, :, None]

    # 3. Concatenar a 4 canales NCHW
    combined = np.concatenate([img_norm, trimap_f32], axis=2)
    combined = np.transpose(combined, (2, 0, 1))[None, ...]

    # 4. Acolchado a múltiplo de 32 hacia la derecha e inferior
    pad_h = 0 if h % 32 == 0 else 32 - (h % 32)
    pad_w = 0 if w % 32 == 0 else 32 - (w % 32)
    if pad_h > 0 or pad_w > 0:
        combined = np.pad(
            combined,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

    # 5. Inferencia ONNX
    input_name = session.get_inputs()[0].name
    pred_alphas = session.run(None, {input_name: combined})[0]
    raw_alpha = pred_alphas[0, 0, :h, :w]
    raw_alpha = np.clip(raw_alpha, 0.0, 1.0)

    # 6. Fusión estricta respetando el contrato de trimap
    vitmatte_alpha_u8 = np.clip(np.round(raw_alpha * 255.0), 0, 255).astype(np.uint8)
    refined_alpha = np.zeros((h, w), dtype=np.uint8)
    refined_alpha[trimap_arr == 255] = 255
    refined_alpha[trimap_arr == 0] = 0
    unknown_mask = (trimap_arr == 128)
    refined_alpha[unknown_mask] = vitmatte_alpha_u8[unknown_mask]

    return refined_alpha


class BackgroundRemover:
    """
    Motor de eliminación de fondo basado exclusivamente en InSPyReNet Swin-B (1024x1024)
    con postprocesamiento subpíxel bilateral y despill cromático estilo LocalBG.
    """

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_inspyrenet_session(self, force_cpu: bool = False):
        """Obtiene o inicializa una sesión ONNX para el modelo InSPyReNet (Swin-B 1024x1024)."""
        cache_key = "inspyrenet_cpu" if force_cpu else "inspyrenet"
        if cache_key not in self._sessions:
            with self._lock:
                if cache_key not in self._sessions:
                    from app.core.model_fetcher import ensure_model_file
                    model_path = ensure_model_file("inspyrenet.onnx")
                    import onnxruntime as ort
                    providers = ["CPUExecutionProvider"] if force_cpu else get_inference_providers()
                    sess_options = get_optimized_session_options() or ort.SessionOptions()
                    try:
                        self._sessions[cache_key] = ort.InferenceSession(str(model_path), sess_options, providers=providers)
                    except Exception as e:
                        logger.exception(f"No se pudo inicializar inspyrenet con GPU ({e}). Usando CPU...")
                        self._sessions[cache_key] = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
        return self._sessions[cache_key]

    def get_rmbg2_session(self, force_cpu: bool = False):
        """Obtiene o inicializa una sesión ONNX para el modelo RMBG-2.0 (BiRefNet 1024x1024)."""
        cache_key = "rmbg2_cpu" if force_cpu else "rmbg2"
        if cache_key not in self._sessions:
            with self._lock:
                if cache_key not in self._sessions:
                    from app.core.model_fetcher import ensure_model_file
                    model_path = ensure_model_file("rmbg-2.0.onnx")
                    import onnxruntime as ort
                    providers = ["CPUExecutionProvider"] if force_cpu else get_inference_providers()
                    sess_options = get_optimized_session_options() or ort.SessionOptions()
                    try:
                        self._sessions[cache_key] = ort.InferenceSession(str(model_path), sess_options, providers=providers)
                    except Exception as e:
                        logger.exception(f"No se pudo inicializar RMBG-2.0 con GPU ({e}). Usando CPU...")
                        self._sessions[cache_key] = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
        return self._sessions[cache_key]

    def get_session(self, model_name: str = "inspyrenet", force_cpu: bool = False):
        """Compatibilidad con get_session: redirige a la sesión correspondiente."""
        if model_name == "rmbg-2.0":
            return self.get_rmbg2_session(force_cpu=force_cpu)
        return self.get_inspyrenet_session(force_cpu=force_cpu)

    def _predict_rmbg2(self, session, input_img: Image.Image) -> np.ndarray:
        """
        Ejecuta inferencia con RMBG-2.0 (BiRefNet 1024x1024):
        1. Letterbox simétrico 1024x1024 y normalización ImageNet.
        2. Inferencia ONNX.
        3. Des-letterbox a tamaño nativo original con interpolación bilineal.
        """
        tensor, crop_box, orig_size = _preprocess_rmbg2(input_img)
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: tensor})[0]
        return _postprocess_rmbg2_mask(out, crop_box, orig_size)

    def _predict_inspyrenet(self, session, input_img: Image.Image) -> np.ndarray:
        """
        Ejecuta inferencia con InSPyReNet (Swin-B 1024x1024):
        1. Resize directo 1024x1024 bilineal y normalización ImageNet.
        2. Inferencia ONNX.
        3. Resize bicúbico del mapa continuo a dimensiones originales (orig_w, orig_h).
        """
        tensor, orig_size = _preprocess_inspyrenet(input_img)
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: tensor})[0]
        return _postprocess_inspyrenet_mask(out, orig_size)

    def _remove_background_cpu_local(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
    ) -> Image.Image:
        """
        Ejecución local en CPU con preservación 100% íntegra de colores originales.
        """
        if image.mode not in ("RGB", "RGBA"):
            input_img = image.convert("RGB")
        else:
            input_img = image.copy()

        if model_name == "rmbg-2.0":
            logger.info("[CPU Fallback/Local] Ejecutando RMBG-2.0 (BiRefNet 1024x1024)...")
            session = self.get_rmbg2_session(force_cpu=True)
            final_alpha = self._predict_rmbg2(session, input_img)
        elif model_name in ("isnet-anime", "birefnet-general", "u2net_human_seg"):
            import rembg
            logger.info(f"[CPU Fallback/Local] Ejecutando {model_name} en CPU...")
            sess = rembg.new_session(model_name, providers=["CPUExecutionProvider"])
            out = rembg.remove(input_img, session=sess)
            final_alpha = np.array(out.split()[-1], dtype=np.uint8)
        else:
            logger.info("[CPU Fallback/Local] Ejecutando InSPyReNet (Swin-B 1024x1024)...")
            session = self.get_inspyrenet_session(force_cpu=True)
            raw_mask = self._predict_inspyrenet(session, input_img)
            cleaned_rgb, final_alpha = postprocess_inspyrenet(input_img, raw_mask)
            input_img = Image.fromarray(cleaned_rgb, mode="RGB")

        # Fusión con Detector Semántico de Ropa y Sujetos (evita huecos en prendas blancas)
        if enable_clothing_protection and model_name != "u2net_human_seg":
            try:
                import rembg
                human_sess = rembg.new_session("u2net_human_seg", providers=["CPUExecutionProvider"])
                human_out = rembg.remove(input_img, session=human_sess)
                human_alpha = np.array(human_out.split()[-1], dtype=np.uint8)
                if np.any(human_alpha > 50):
                    final_alpha = np.maximum(final_alpha, human_alpha)
            except Exception as e:
                logger.warning(f"Protección de ropa u2net_human_seg omitida en CPU: {e}")

        if image.mode == "RGBA":
            orig_alpha = np.array(image.split()[-1], dtype=np.uint8)
            final_alpha = np.minimum(final_alpha, orig_alpha)

        output = input_img.convert("RGB").convert("RGBA")
        output.putalpha(Image.fromarray(final_alpha, mode="L"))
        return output

    def remove_background(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
    ) -> Image.Image:
        """
        Elimina el fondo de una imagen usando el modelo seleccionado (IS-Net Anime, U2-Net Human, BiRefNet o InSPyReNet).
        Si la GPU está activa, delega al proceso GPU Worker aislado con watchdog.
        Si la GPU está inactiva o no responde antes del timeout, ejecuta automáticamente en CPU.
        """
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()

        if gpu_mgr.is_gpu_enabled():
            return gpu_mgr.run_inference(
                command="remove_background",
                payload={
                    "image": image,
                    "model_name": model_name,
                    "ensemble_weight": ensemble_weight,
                    "refine_matting": refine_matting,
                    "enable_clothing_protection": enable_clothing_protection,
                },
                cpu_fallback_fn=lambda: self._remove_background_cpu_local(
                    image,
                    model_name=model_name,
                    ensemble_weight=ensemble_weight,
                    refine_matting=refine_matting,
                    enable_clothing_protection=enable_clothing_protection,
                ),
            )
        else:
            return self._remove_background_cpu_local(
                image,
                model_name=model_name,
                ensemble_weight=ensemble_weight,
                refine_matting=refine_matting,
                enable_clothing_protection=enable_clothing_protection,
            )


# Instancia singleton para reutilización de sesiones
_remover_instance: Optional[BackgroundRemover] = None


def get_bg_remover() -> BackgroundRemover:
    global _remover_instance
    if _remover_instance is None:
        _remover_instance = BackgroundRemover()
    return _remover_instance

