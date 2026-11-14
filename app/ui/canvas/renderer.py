"""
TimoitaseBG - Funciones de renderizado, conversión RGBA y estilos para CanvasWidget.
Mantiene el código de pintado y QSS desacoplado de la gestión de eventos de entrada.
"""
from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap

from app.core.image_utils import pil_to_qimage
from app.i18n import tr


def ensure_rgba_qimage(pil_img: Image.Image) -> QImage:
    """Convierte un PIL.Image en QImage normalizado siempre a canal alfa RGBA8888."""
    qimg = pil_to_qimage(pil_img)
    if not qimg.hasAlphaChannel():
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    return qimg


def _axis_scrollbar_qss(
    orient: str,
    thickness_prop: str,
    min_handle_prop: str,
    zero_prop: str,
    track: str,
    handle: str,
    handle_hover: str,
) -> str:
    return f"""
        QScrollBar:{orient} {{
            border: none;
            background: {track};
            {thickness_prop}: 12px;
            margin: 0px;
            border-radius: 6px;
        }}
        QScrollBar::handle:{orient} {{
            background: {handle};
            {min_handle_prop}: 30px;
            border-radius: 6px;
        }}
        QScrollBar::handle:{orient}:hover {{
            background: {handle_hover};
        }}
        QScrollBar::add-line:{orient}, QScrollBar::sub-line:{orient},
        QScrollBar::add-page:{orient}, QScrollBar::sub-page:{orient} {{
            background: none;
            {zero_prop}: 0px;
        }}
    """


def build_scrollbar_stylesheet(is_dark: bool = True) -> str:
    """Construye la hoja QSS de las barras de desplazamiento flotantes del lienzo."""
    track = "rgba(15, 23, 42, 0.65)" if is_dark else "rgba(231, 229, 228, 0.4)"
    handle = "rgba(71, 85, 105, 0.8)" if is_dark else "rgba(120, 113, 108, 0.6)"
    handle_hover = "rgba(148, 163, 184, 0.95)" if is_dark else "rgba(120, 113, 108, 0.9)"
    horiz = _axis_scrollbar_qss("horizontal", "height", "min-width", "width", track, handle, handle_hover)
    vert = _axis_scrollbar_qss("vertical", "width", "min-height", "height", track, handle, handle_hover)
    return horiz + vert


def _fill_canvas_backdrop(
    painter: QPainter,
    img_rect: QRectF,
    checker_pixmap: QPixmap,
    bg_color_rgb: Optional[tuple[int, int, int]],
) -> None:
    """Dibuja el fondo sólido o el damero de transparencia dentro de img_rect."""
    if bg_color_rgb:
        painter.fillRect(img_rect, QColor(*bg_color_rgb))
    else:
        painter.drawTiledPixmap(img_rect, checker_pixmap)


def draw_comparison_divider(
    painter: QPainter,
    img_rect: QRectF,
    split_x: float,
    is_dark: bool = True,
) -> None:
    """Dibuja la línea divisoria vertical, la manija central con flechas y las etiquetas Antes/Después."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    painter.setPen(QPen(QColor(0, 0, 0, 70), 3.0))
    painter.drawLine(QPointF(split_x, img_rect.top()), QPointF(split_x, img_rect.bottom()))
    painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
    painter.drawLine(QPointF(split_x, img_rect.top()), QPointF(split_x, img_rect.bottom()))

    cy = img_rect.center().y()
    radius = 16.0
    handle_center = QPointF(split_x, cy)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 40))
    painter.drawEllipse(handle_center + QPointF(1, 1), radius + 1, radius + 1)

    painter.setBrush(QColor("#1E293B") if is_dark else QColor("#FFFFFF"))
    painter.setPen(QPen(QColor("#FF4F79"), 2.5))
    painter.drawEllipse(handle_center, radius, radius)

    painter.setPen(QColor("#FF4F79"))
    font = painter.font()
    font.setPixelSize(9)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(
        QRectF(split_x - radius, cy - radius, radius * 2, radius * 2),
        Qt.AlignmentFlag.AlignCenter,
        "◀ ▶",
    )

    badge_y = img_rect.top() + 12
    if badge_y + 24 < img_rect.bottom():
        font_b = painter.font()
        font_b.setPixelSize(8)
        font_b.setBold(True)
        painter.setFont(font_b)

        for rect, color, label_key in (
            (QRectF(img_rect.left() + 12, badge_y, 64, 22), QColor(15, 23, 42, 215), "compare_before"),
            (QRectF(img_rect.right() - 76, badge_y, 64, 22), QColor(244, 63, 94, 225), "compare_after"),
        ):
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tr(label_key))

    painter.restore()


def draw_canvas_scene(
    painter: QPainter,
    img_rect: QRectF,
    current_qimage: QImage,
    checker_pixmap: QPixmap,
    bg_color_rgb: Optional[tuple[int, int, int]],
    selection_qimage: Optional[QImage],
    comparison_mode: bool,
    before_qimage: Optional[QImage],
    comparison_split_ratio: float,
    is_dark: bool,
) -> None:
    """Dibuja sombra, fondo, imagen activa (o comparativa), overlay de selección y borde."""
    painter.save()
    for offset, alpha in [(8, 8), (5, 14), (3, 22), (1, 35)]:
        painter.fillRect(img_rect.adjusted(-offset, -offset, offset, offset), QColor(0, 0, 0, alpha))
    painter.restore()

    if comparison_mode and before_qimage:
        split_x = img_rect.left() + img_rect.width() * comparison_split_ratio

        left_rect = QRectF(img_rect.left(), img_rect.top(), max(0.0, split_x - img_rect.left()), img_rect.height())
        painter.save()
        painter.setClipRect(left_rect)
        painter.drawTiledPixmap(img_rect, checker_pixmap)
        painter.drawImage(img_rect, before_qimage)
        painter.restore()

        right_rect = QRectF(split_x, img_rect.top(), max(0.0, img_rect.right() - split_x), img_rect.height())
        painter.save()
        painter.setClipRect(right_rect)
        _fill_canvas_backdrop(painter, img_rect, checker_pixmap, bg_color_rgb)
        painter.drawImage(img_rect, current_qimage)
        painter.restore()

        draw_comparison_divider(painter, img_rect, split_x, is_dark=is_dark)
    else:
        painter.save()
        painter.setClipRect(img_rect)
        _fill_canvas_backdrop(painter, img_rect, checker_pixmap, bg_color_rgb)
        painter.drawImage(img_rect, current_qimage)
        if selection_qimage:
            painter.drawImage(img_rect, selection_qimage)
        painter.restore()

    painter.setPen(QPen(QColor(61, 61, 61), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(img_rect)


def draw_brush_cursor(
    painter: QPainter,
    mouse_cursor_pos: QPoint,
    brush_size: int,
    zoom_factor: float,
    eraser_mode: bool,
) -> None:
    """Dibuja el círculo del tamaño del pincel sobre el cursor del ratón."""
    visual_radius = (brush_size * zoom_factor) / 2.0
    r_draw = max(1.5, visual_radius)
    center = QPointF(mouse_cursor_pos.x(), mouse_cursor_pos.y())
    border_w = 1.0 if visual_radius < 3.0 else 1.5

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    if eraser_mode:
        painter.setBrush(QColor(15, 23, 42, 65))
        painter.setPen(QPen(QColor(255, 255, 255, 220), border_w + 1.0))
        painter.drawEllipse(center, r_draw, r_draw)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248, 240), border_w))
        painter.drawEllipse(center, r_draw, r_draw)
        dot_color = QColor(56, 189, 248, 245)
    else:
        painter.setBrush(QColor(255, 79, 121, 65))
        painter.setPen(QPen(QColor(255, 79, 121, 230), border_w))
        painter.drawEllipse(center, r_draw, r_draw)
        dot_color = QColor(255, 79, 121, 240)

    dot_r = 0.8 if visual_radius < 4.0 else 1.2
    painter.setBrush(dot_color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(center, dot_r, dot_r)
    painter.restore()
