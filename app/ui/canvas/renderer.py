"""
TimoitaseBG - Funciones de renderizado, conversión RGBA y estilos para CanvasWidget.
Mantiene el código de pintado y QSS desacoplado de la gestión de eventos de entrada.
"""
from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF, QRectF
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPen, QPixmap

from app.core.image_utils import pil_to_qimage
from app.i18n import tr


def ensure_rgba_qimage(pil_img: Image.Image) -> QImage:
    """Convierte un PIL.Image en QImage normalizado siempre a canal alfa RGBA8888."""
    qimg = pil_to_qimage(pil_img)
    if not qimg.hasAlphaChannel():
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    return qimg


def build_hidpi_dot_grid(spacing: int = 20, dpr: float = 1.0, is_dark: bool = True) -> QPixmap:
    """Genera un mosaico HiDPI de 20x20 con cuadrícula arquitectónica sutil de puntos."""
    safe_dpr = max(1.0, float(dpr or 1.0))
    tile_px = max(8, int(round(spacing * safe_dpr)))
    pixmap = QPixmap(tile_px, tile_px)
    pixmap.fill(Qt.GlobalColor.transparent)

    dot_color = QColor(255, 255, 255, 11) if is_dark else QColor(20, 23, 26, 16)
    center = tile_px / 2.0
    radius = max(1.0, 1.0 * safe_dpr)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(dot_color)
    painter.drawEllipse(QPointF(center, center), radius, radius)
    painter.end()

    pixmap.setDevicePixelRatio(safe_dpr)
    return pixmap


def build_hidpi_checkerboard(size: int = 16, dpr: float = 1.0, is_dark: bool = True) -> QPixmap:
    """Genera un patrón damero consciente de devicePixelRatioF() para pantallas HiDPI."""
    safe_dpr = max(1.0, float(dpr or 1.0))
    cell_px = max(4, int(round(size * safe_dpr)))
    pattern_px = cell_px * 2
    pixmap = QPixmap(pattern_px, pattern_px)
    c1 = QColor("#1D2127") if is_dark else QColor("#FFFFFF")
    c2 = QColor("#15181C") if is_dark else QColor("#ECEFF2")

    painter = QPainter(pixmap)
    painter.fillRect(0, 0, cell_px, cell_px, c1)
    painter.fillRect(cell_px, 0, cell_px, cell_px, c2)
    painter.fillRect(0, cell_px, cell_px, cell_px, c2)
    painter.fillRect(cell_px, cell_px, cell_px, cell_px, c1)
    painter.end()

    pixmap.setDevicePixelRatio(safe_dpr)
    return pixmap


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
    track = "rgba(14, 16, 18, 0.55)" if is_dark else "rgba(236, 239, 242, 0.65)"
    handle = "rgba(155, 163, 174, 0.45)" if is_dark else "rgba(85, 94, 107, 0.35)"
    handle_hover = "rgba(155, 163, 174, 0.75)" if is_dark else "rgba(85, 94, 107, 0.65)"
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
    """Dibuja la línea divisoria vertical, tirador circular de 28px con flechas y etiquetas Antes/Después."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Línea vertical nítida de 2px blanca con sombra sutil
    painter.setPen(QPen(QColor(0, 0, 0, 85), 4.0))
    painter.drawLine(QPointF(split_x, img_rect.top()), QPointF(split_x, img_rect.bottom()))
    painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
    painter.drawLine(QPointF(split_x, img_rect.top()), QPointF(split_x, img_rect.bottom()))

    # Tirador circular central de 28px (radio 14px)
    cy = img_rect.center().y()
    radius = 14.0
    handle_center = QPointF(split_x, cy)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 32))
    painter.drawEllipse(handle_center + QPointF(0, 2), radius + 3.0, radius + 3.0)
    painter.setBrush(QColor(0, 0, 0, 75))
    painter.drawEllipse(handle_center + QPointF(0, 1.5), radius + 1.0, radius + 1.0)

    accent = QColor("#2E7CF6") if is_dark else QColor("#1D63DC")
    painter.setBrush(QColor(21, 24, 28, 238) if is_dark else QColor(255, 255, 255, 245))
    painter.setPen(QPen(QColor("#FFFFFF") if is_dark else accent, 1.8))
    painter.drawEllipse(handle_center, radius, radius)

    painter.setPen(QColor("#FFFFFF") if is_dark else accent)
    font = painter.font()
    font.setPixelSize(10)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(
        QRectF(split_x - radius, cy - radius, radius * 2, radius * 2),
        Qt.AlignmentFlag.AlignCenter,
        "◀ ▶",
    )

    # Etiquetas flotantes discretas Antes / Después en esquinas superiores
    badge_y = img_rect.top() + 12
    if badge_y + 26 < img_rect.bottom() and img_rect.width() >= 140:
        font_b = painter.font()
        font_b.setPixelSize(12)
        font_b.setWeight(font_b.Weight.DemiBold)
        painter.setFont(font_b)
        fm = QFontMetricsF(font_b)

        before_txt = tr("compare_before")
        after_txt = tr("compare_after")
        before_w = max(62.0, fm.horizontalAdvance(before_txt) + 20.0)
        after_w = max(68.0, fm.horizontalAdvance(after_txt) + 20.0)

        badge_bg = QColor(14, 16, 18, 185)
        badge_border = QPen(QColor(255, 255, 255, 36), 1.0)

        for rect, label_txt in (
            (QRectF(img_rect.left() + 12, badge_y, before_w, 24), before_txt),
            (QRectF(img_rect.right() - after_w - 12, badge_y, after_w, 24), after_txt),
        ):
            painter.setBrush(badge_bg)
            painter.setPen(badge_border)
            painter.drawRoundedRect(rect, 7, 7)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label_txt)

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
    cached_pixmap: Optional[QPixmap] = None,
    cached_before_pixmap: Optional[QPixmap] = None,
) -> None:
    """Dibuja sombra ambiental multicapa, fondo, imagen activa (o comparativa), overlay y hairline."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    if is_dark:
        shadow_layers = (
            (18.0, 6.0, 7, 10.0),
            (13.0, 4.0, 11, 8.0),
            (9.0, 3.0, 16, 6.0),
            (5.0, 2.0, 22, 4.0),
            (2.0, 1.0, 32, 2.0),
        )
        shadow_rgb = (0, 0, 0)
    else:
        shadow_layers = (
            (16.0, 6.0, 4, 10.0),
            (11.0, 4.0, 7, 8.0),
            (7.0, 3.0, 10, 6.0),
            (4.0, 2.0, 14, 4.0),
            (1.5, 1.0, 18, 2.0),
        )
        shadow_rgb = (15, 20, 28)

    for spread, y_off, alpha, radius in shadow_layers:
        shadow_rect = img_rect.adjusted(-spread, -spread + y_off, spread, spread + y_off)
        painter.setBrush(QColor(shadow_rgb[0], shadow_rgb[1], shadow_rgb[2], alpha))
        painter.drawRoundedRect(shadow_rect, radius, radius)
    painter.restore()

    def _draw_layer(qimg: QImage, pix: Optional[QPixmap]) -> None:
        if pix is not None and not pix.isNull():
            painter.drawPixmap(img_rect, pix, QRectF(pix.rect()))
        else:
            painter.drawImage(img_rect, qimg)

    if comparison_mode and before_qimage:
        split_x = img_rect.left() + img_rect.width() * comparison_split_ratio

        left_rect = QRectF(img_rect.left(), img_rect.top(), max(0.0, split_x - img_rect.left()), img_rect.height())
        painter.save()
        painter.setClipRect(left_rect)
        painter.drawTiledPixmap(img_rect, checker_pixmap)
        _draw_layer(before_qimage, cached_before_pixmap)
        painter.restore()

        right_rect = QRectF(split_x, img_rect.top(), max(0.0, img_rect.right() - split_x), img_rect.height())
        painter.save()
        painter.setClipRect(right_rect)
        _fill_canvas_backdrop(painter, img_rect, checker_pixmap, bg_color_rgb)
        _draw_layer(current_qimage, cached_pixmap)
        painter.restore()

        draw_comparison_divider(painter, img_rect, split_x, is_dark=is_dark)
    else:
        painter.save()
        painter.setClipRect(img_rect)
        _fill_canvas_backdrop(painter, img_rect, checker_pixmap, bg_color_rgb)
        _draw_layer(current_qimage, cached_pixmap)
        if selection_qimage:
            painter.drawImage(img_rect, selection_qimage)
        painter.restore()

    border_color = QColor(255, 255, 255, 22) if is_dark else QColor(20, 23, 26, 26)
    painter.setPen(QPen(border_color, 1.0))
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
        painter.setBrush(QColor(14, 16, 18, 60))
        painter.setPen(QPen(QColor(255, 255, 255, 220), border_w + 1.0))
        painter.drawEllipse(center, r_draw, r_draw)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(224, 79, 95, 240), border_w))
        painter.drawEllipse(center, r_draw, r_draw)
        dot_color = QColor(224, 79, 95, 245)
    else:
        painter.setBrush(QColor(46, 124, 246, 55))
        painter.setPen(QPen(QColor(46, 124, 246, 235), border_w))
        painter.drawEllipse(center, r_draw, r_draw)
        dot_color = QColor(46, 124, 246, 245)

    dot_r = 0.8 if visual_radius < 4.0 else 1.2
    painter.setBrush(dot_color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(center, dot_r, dot_r)
    painter.restore()
