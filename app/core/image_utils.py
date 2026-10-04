"""
TimoitaseBG - Utilidades de manipulación de imágenes y conversión Qt <-> PIL.
"""
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """
    Convierte un QImage a un array NumPy (H, W, 4) uint8 en memoria C-contigua,
    utilizando explícitamente bytesPerLine() para respetar strides y evitar desalineación de scanlines.
    """
    if qimg.isNull():
        raise ValueError("QImage está vacía o es nula")

    formatted = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    width = formatted.width()
    height = formatted.height()
    bytes_per_line = formatted.bytesPerLine()

    ptr = formatted.constBits()
    # Construir un array 2D mapeando la memoria con su stride real
    total_bytes = bytes_per_line * height
    buf = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes).reshape((height, bytes_per_line))

    # Extraer únicamente los 4 canales de cada píxel ignorando el padding de scanline
    rgba = buf[:, : width * 4].reshape((height, width, 4))
    return np.ascontiguousarray(rgba.copy())


def qimage_to_pil(qimg: QImage) -> Image.Image:
    """Convierte un objeto QImage a una imagen PIL.Image RGBA de forma segura."""
    arr = qimage_to_numpy(qimg)
    return Image.fromarray(arr, mode="RGBA")


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """
    Convierte un array NumPy (H, W, 4), (H, W, 3) o (H, W) a QImage de forma segura
    copiando la memoria para evitar punteros huérfanos.
    """
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]

    if arr.ndim == 3 and arr.shape[2] == 4:
        bytes_per_line = w * 4
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        return qimg.copy()
    elif arr.ndim == 3 and arr.shape[2] == 3:
        bytes_per_line = w * 3
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()
    elif arr.ndim == 2:
        bytes_per_line = w
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        return qimg.copy()
    else:
        raise ValueError(f"Formato de array NumPy no soportado: shape {arr.shape}")


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """Convierte una imagen PIL a QImage de forma segura usando NumPy contiguo."""
    if pil_img.mode == "RGBA":
        arr = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
        return numpy_to_qimage(arr)
    elif pil_img.mode == "RGB":
        arr = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))
        return numpy_to_qimage(arr)
    elif pil_img.mode in ("L", "1"):
        converted = pil_img.convert("RGBA")
        arr = np.ascontiguousarray(np.array(converted, dtype=np.uint8))
        return numpy_to_qimage(arr)
    else:
        converted = pil_img.convert("RGBA")
        arr = np.ascontiguousarray(np.array(converted, dtype=np.uint8))
        return numpy_to_qimage(arr)


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Convierte una imagen de PIL a QPixmap."""
    qimg = pil_to_qimage(pil_img)
    return QPixmap.fromImage(qimg)


def create_checkerboard_pattern(size: int = 16, c1: str = "#FFFFFF", c2: str = "#F0F0F0") -> QPixmap:
    """
    Crea un patrón de damero (ajedrez) en un QPixmap para renderizar el fondo de transparencias.
    """
    pattern_size = size * 2
    pixmap = QPixmap(pattern_size, pattern_size)
    painter = QPainter(pixmap)
    color1 = QColor(c1)
    color2 = QColor(c2)

    painter.fillRect(0, 0, size, size, color1)
    painter.fillRect(size, 0, size, size, color2)
    painter.fillRect(0, size, size, size, color2)
    painter.fillRect(size, size, size, size, color1)
    painter.end()

    return pixmap


def separate_alpha(pil_img: Image.Image) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    Separa una imagen en sus componentes RGB y canal Alfa (si existe).
    """
    if pil_img.mode == "RGBA":
        r, g, b, a = pil_img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        return rgb_img, a
    return pil_img.convert("RGB"), None


def combine_alpha(rgb_img: Image.Image, alpha: Optional[Image.Image]) -> Image.Image:
    """
    Recombina una imagen RGB con su canal Alfa.
    Garantiza que el canal alfa tenga exactamente las mismas dimensiones que el RGB
    mediante un resize final con Lanczos.
    """
    if alpha is None:
        return rgb_img

    rgb_clean = rgb_img.convert("RGB")
    if rgb_clean.size != alpha.size:
        alpha = alpha.resize(rgb_clean.size, resample=Image.Resampling.LANCZOS)

    r, g, b = rgb_clean.split()
    alpha_clean = alpha.convert("L") if alpha.mode != "L" else alpha
    return Image.merge("RGBA", (r, g, b, alpha_clean))


def autocrop_image(img: Image.Image, padding: int = 10) -> Optional[Image.Image]:
    """
    Recorta los márgenes transparentes alrededor del contenido visible de una imagen RGBA.
    Añade un margen de respiro (padding) sin salirse de las dimensiones originales.
    Retorna la imagen recortada, o None si no hay píxeles visibles o no es RGBA.
    """
    if img.mode != "RGBA":
        return None
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return None

    pad = max(0, padding)
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.width, bbox[2] + pad)
    bottom = min(img.height, bbox[3] + pad)

    return img.crop((left, top, right, bottom))


def apply_solid_background(img: Image.Image, color_rgb: Tuple[int, int, int]) -> Image.Image:
    """
    Compone una imagen RGBA sobre un fondo de color sólido usando Image.alpha_composite
    para preservar el antialiasing y bordes suaves (cabello, transparencias graduales).
    Retorna una nueva imagen en modo RGB sólido.
    """
    if img.mode != "RGBA":
        return img.copy()

    # Lienzo de fondo con el color RGB sólido y alfa 255
    bg = Image.new("RGBA", img.size, (color_rgb[0], color_rgb[1], color_rgb[2], 255))
    composed = Image.alpha_composite(bg, img)
    return composed.convert("RGB")

