"""
TimoitaseBG - Lógica pura de selección estilo Photoshop (Varita Mágica).

Todas las máscaras de selección son arrays ``uint8`` de forma (H, W) con valores
0..255, donde 255 = totalmente seleccionado y valores intermedios = selección
parcial (bordes suavizados / anti-alias).

Comportamiento replicado de la Varita Mágica de Photoshop:
- Tolerancia (0-255): un píxel entra en la selección si CADA canal (R, G, B y Alfa)
  difiere del color del píxel clicado (semilla) como máximo en ``tolerance``.
  La comparación es siempre contra la semilla (rango fijo), no contra el vecino,
  para que la selección no "se arrastre" por degradados.
- Contiguo: solo se seleccionan píxeles conectados a la semilla (4-vecindad).
  Si se desactiva, se seleccionan todos los píxeles similares de la imagen.
- Suavizar (anti-alias): suaviza el borde escalonado de la selección.
- Modos: nueva, añadir (Shift), restar (Alt) e intersecar (Shift+Alt).
"""
from typing import Optional

import cv2
import numpy as np

SELECTION_MODES = ("new", "add", "subtract", "intersect")


def _to_rgba(img: np.ndarray) -> np.ndarray:
    """Normaliza la imagen a RGBA uint8."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    return img


def magic_wand_region(
    img: np.ndarray,
    x: int,
    y: int,
    tolerance: int = 32,
    contiguous: bool = True,
) -> Optional[np.ndarray]:
    """
    Calcula la región (bool H×W) que seleccionaría la varita al hacer clic en (x, y).
    Devuelve None si el punto está fuera de la imagen.
    """
    rgba = _to_rgba(img)
    h, w = rgba.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None

    tolerance = int(max(0, min(255, tolerance)))

    # Los píxeles totalmente transparentes no tienen un color "real": se unifican
    # a (0,0,0,0) para que el área transparente se seleccione como un bloque.
    work = rgba.astype(np.int16)
    work[rgba[:, :, 3] == 0] = 0

    seed = work[y, x]
    diff = np.abs(work - seed).max(axis=2)
    within = diff <= tolerance

    if not contiguous:
        return within

    # Relleno por inundación sobre la máscara binaria: conserva solo la región
    # conectada (4-vecindad) a la semilla.
    binary = within.astype(np.uint8) * 255
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(
        binary, ff_mask, (x, y), 0, 0, 0,
        flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
    )
    return ff_mask[1:-1, 1:-1] > 0


def region_to_mask(region: np.ndarray, anti_alias: bool = True) -> np.ndarray:
    """Convierte una región booleana en máscara uint8 (0-255), opcionalmente suavizada."""
    mask = region.astype(np.uint8) * 255
    if anti_alias:
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask


def combine_selection(current: Optional[np.ndarray], new: np.ndarray, mode: str) -> np.ndarray:
    """Combina la selección actual con una nueva según el modo de Photoshop."""
    if current is None or current.shape != new.shape or mode == "new":
        if mode in ("subtract", "intersect"):
            # Restar o intersecar sin selección previa no deja nada seleccionado.
            return np.zeros_like(new)
        return new.copy()
    if mode == "add":
        return np.maximum(current, new)
    if mode == "subtract":
        return np.minimum(current, 255 - new)
    if mode == "intersect":
        return np.minimum(current, new)
    raise ValueError(f"Modo de selección desconocido: {mode}")


def invert_selection(mask: np.ndarray) -> np.ndarray:
    return 255 - mask


def select_all(height: int, width: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def selection_pixel_count(mask: Optional[np.ndarray]) -> int:
    """Número de píxeles seleccionados (umbral al 50 %)."""
    if mask is None:
        return 0
    return int(np.count_nonzero(mask > 127))


def delete_selected_pixels(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Vuelve transparentes los píxeles seleccionados (como Supr en Photoshop sobre
    una capa). Respeta la selección parcial: alfa *= (1 - selección).
    """
    rgba = _to_rgba(img).copy()
    keep = (255 - mask.astype(np.uint16))
    rgba[:, :, 3] = ((rgba[:, :, 3].astype(np.uint16) * keep + 127) // 255).astype(np.uint8)
    return rgba
