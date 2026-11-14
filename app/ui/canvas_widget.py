"""
TimoitaseBG - Canvas interactivo para visualización, zoom, paneo y trazado de máscara de pincel.
"""
import logging
from typing import Optional
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, Signal, QTimer
from PySide6.QtWidgets import QWidget, QScrollBar
from PySide6.QtGui import (
    QPainter,
    QImage,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
)

from app.i18n import tr
from app.core.image_utils import qimage_to_pil, create_checkerboard_pattern
from app.core import selection as sel_ops
from app.config import DEFAULT_BRUSH_SIZE, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE, CHECKER_SIZE
from app.ui.canvas import (
    BrushEngine,
    CropTool,
    SelectionTool,
    build_scrollbar_stylesheet,
    draw_brush_cursor,
    draw_canvas_scene,
    draw_comparison_divider,
    ensure_rgba_qimage,
)

logger = logging.getLogger("TimoitaseBG.CanvasWidget")


def _tool_prop(tool_attr: str, field: str, read_only: bool = False) -> property:
    """Genera una propiedad delegada hacia un subsistema del lienzo."""
    def _getter(self):
        return getattr(getattr(self, tool_attr), field)

    if read_only:
        return property(_getter)

    def _setter(self, value) -> None:
        setattr(getattr(self, tool_attr), field, value)

    return property(_getter, _setter)


class CanvasWidget(QWidget):
    """
    Widget central que muestra la imagen, soporta zoom/paneo interactivo y
    permite dibujar máscaras para el pincel mágico con vista previa en tiempo real.
    """

    maskChanged = Signal(bool)          # Emite True si hay trazos dibujados, False si está limpia
    zoomChanged = Signal(float)         # Emite el porcentaje de zoom actual
    fileDropped = Signal(str)           # Emite la ruta de archivo si el usuario arrastra una imagen
    imageDropped = Signal(object)       # Emite un PIL.Image cuando se arrastra una imagen en memoria
    browseRequested = Signal()          # Emite cuando el usuario hace clic en la zona vacía para explorar
    imageEditedDirectly = Signal(str)   # Descripción de la edición para el historial
    selectionChanged = Signal(int, int) # (píxeles seleccionados, píxeles totales)
    cropConfirmed = Signal(tuple)       # (left, top, right, bottom)

    wand_mode = _tool_prop("_selection_tool", "wand_mode")
    wand_ai_mode = _tool_prop("_selection_tool", "wand_ai_mode")
    wand_tolerance = _tool_prop("_selection_tool", "wand_tolerance")
    wand_selection_mode = _tool_prop("_selection_tool", "wand_selection_mode")
    wand_anti_alias = _tool_prop("_selection_tool", "wand_anti_alias")
    wand_contiguous = _tool_prop("_selection_tool", "wand_contiguous")
    active_selection = _tool_prop("_selection_tool", "active_mask")
    selection_qimage = _tool_prop("_selection_tool", "overlay_qimage", read_only=True)
    selection_path = _tool_prop("_selection_tool", "contour_path", read_only=True)
    _pending_wand_click = _tool_prop("_selection_tool", "pending_wand_click")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        # Estados de visualización
        self.current_qimage: Optional[QImage] = None
        self.original_qimage: Optional[QImage] = None
        self.bg_color_rgb: Optional[tuple[int, int, int]] = None

        # Zoom y paneo
        self.zoom_factor: float = 1.0
        self.pan_offset: QPointF = QPointF(0, 0)
        self.is_panning: bool = False
        self.last_pan_pos: QPoint = QPoint()
        self.space_pressed: bool = False

        # Scrollbars flotantes
        sb_style = self._scrollbar_style()
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        self.v_scrollbar.setStyleSheet(sb_style)
        self.v_scrollbar.valueChanged.connect(self._on_v_scrollbar_changed)
        self.v_scrollbar.hide()

        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self.h_scrollbar.setStyleSheet(sb_style)
        self.h_scrollbar.valueChanged.connect(self._on_h_scrollbar_changed)
        self.h_scrollbar.hide()
        self._updating_scrollbars = False

        # Subsistemas de pincel, selección y recorte
        self.brush_mode: bool = False
        self.brush_size: int = DEFAULT_BRUSH_SIZE
        self.brush_smoothness: int = 0
        self.brush_opacity: float = 1.0
        self.brush_hardness: float = 1.0
        self.is_drawing_mask: bool = False
        self.last_draw_point: Optional[QPointF] = None
        self.mouse_cursor_pos: QPoint = QPoint(-1000, -1000)
        self.eraser_mode: bool = False
        self._brush_engine = BrushEngine()
        self._selection_tool = SelectionTool()
        self._crop_tool = CropTool()

        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(40)  # 25 FPS reales
        self._ants_timer.timeout.connect(self._advance_marching_ants)

        self.checker_pixmap = create_checkerboard_pattern(size=CHECKER_SIZE)

        # Modo Comparación (Antes / Después con cortina deslizante)
        self.comparison_mode: bool = False
        self.before_qimage: Optional[QImage] = None
        self.comparison_split_ratio: float = 0.5
        self.is_dragging_split: bool = False

        # Drop Zone widget estilizado para el lienzo vacío
        from app.ui.drop_zone import DropZoneWidget
        self.drop_zone = DropZoneWidget(self)
        self.drop_zone.fileDropped.connect(self.fileDropped.emit)
        self.drop_zone.imageDropped.connect(self.imageDropped.emit)
        self.drop_zone.clicked.connect(self.browseRequested.emit)
        self.drop_zone.show()

    # -------------------------------------------------------------
    # Recorte, Comparación e Imagen Activa
    # -------------------------------------------------------------
    def set_crop_mode(self, enabled: bool, aspect_ratio: Optional[float] = None):
        """Activa o desactiva la herramienta visual de recorte manual (Crop)."""
        if enabled and self.current_qimage is not None:
            self._crop_tool.activate(
                self.current_qimage.width(),
                self.current_qimage.height(),
                aspect_ratio=aspect_ratio,
            )
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._crop_tool.deactivate()
            self.setCursor(self._tool_cursor())
        self.update()

    def set_crop_aspect_ratio(self, aspect_ratio: Optional[float]):
        if self.current_qimage is not None:
            self._crop_tool.set_aspect_ratio(
                aspect_ratio, self.current_qimage.width(), self.current_qimage.height()
            )
            self.update()

    def get_crop_box(self) -> Optional[tuple]:
        if self.current_qimage is None:
            return None
        return self._crop_tool.get_crop_box(
            self.current_qimage.width(), self.current_qimage.height()
        )

    def set_comparison_mode(self, enabled: bool, before_image: Optional[Image.Image] = None):
        """Activa o desactiva la vista comparativa antes/después con cortina."""
        self.comparison_mode = enabled
        if enabled and before_image is not None:
            self.before_qimage = pil_to_qimage(before_image)
            self.comparison_split_ratio = 0.5
        else:
            self.before_qimage = None
            self.comparison_mode = False
        self.is_dragging_split = False
        self.update()

    def set_original_image(self, pil_img: Optional[Image.Image]):
        """Actualiza la imagen original para su uso con el pincel restaurador."""
        self.original_qimage = ensure_rgba_qimage(pil_img) if pil_img else None

    def set_background_color(self, color_rgb: Optional[tuple[int, int, int]]):
        """Define un color de fondo sólido para renderizar detrás de la imagen transparente."""
        self.bg_color_rgb = color_rgb
        self.update()

    def set_image(self, pil_img: Optional[Image.Image], reset_zoom: bool = False):
        """Establece la imagen activa y ajusta o preserva la máscara."""
        if pil_img is None:
            self.current_qimage = None
            self.clear_selection()
            if hasattr(self, "drop_zone"):
                self.drop_zone.show()
                self._center_drop_zone()
            self.update()
            self.maskChanged.emit(False)
            return

        if hasattr(self, "drop_zone"):
            self.drop_zone.hide()

        prev_w = self.current_qimage.width() if self.current_qimage else 0
        prev_h = self.current_qimage.height() if self.current_qimage else 0

        self.current_qimage = ensure_rgba_qimage(pil_img)
        img_w = self.current_qimage.width()
        img_h = self.current_qimage.height()

        if reset_zoom or (img_w, img_h) != (prev_w, prev_h):
            self.clear_selection()

        if reset_zoom or prev_w == 0:
            self.fit_to_view()
        else:
            self._clamp_pan_offset()
            self.update()

    # -------------------------------------------------------------
    # Configuración de Pincel y Varita
    # -------------------------------------------------------------
    def _tool_cursor(self) -> Qt.CursorShape:
        if not self.brush_mode:
            return Qt.CursorShape.ArrowCursor
        if self.wand_mode:
            return Qt.CursorShape.CrossCursor
        return Qt.CursorShape.BlankCursor

    def set_brush_mode(self, enabled: bool):
        self.brush_mode = enabled
        self.setCursor(self._tool_cursor())
        self.update()

    def set_brush_size(self, size: int):
        self.brush_size = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, size))
        self.update()

    def set_brush_smoothness(self, smoothness: int):
        self.brush_smoothness = max(0, min(100, smoothness))

    def set_brush_opacity(self, opacity: float):
        self.brush_opacity = opacity
        self.update()

    def set_brush_hardness(self, hardness: float):
        self.brush_hardness = hardness
        self.update()

    def set_eraser_mode(self, enabled: bool):
        self.eraser_mode = enabled

    def set_wand_mode(self, enabled: bool):
        self.wand_mode = enabled
        self.setCursor(self._tool_cursor())
        self.update()

    def set_wand_ai_mode(self, enabled: bool):
        self.wand_ai_mode = enabled
        self.update()

    def set_wand_tolerance(self, tolerance: int):
        self.wand_tolerance = int(max(0, min(255, tolerance)))

    def set_wand_selection_mode(self, mode: str):
        if mode in sel_ops.SELECTION_MODES:
            self.wand_selection_mode = mode

    def set_wand_anti_alias(self, enabled: bool):
        self.wand_anti_alias = enabled

    def set_wand_contiguous(self, enabled: bool):
        self.wand_contiguous = enabled

    # -------------------------------------------------------------
    # Selección (delegado en SelectionTool)
    # -------------------------------------------------------------
    def has_selection(self) -> bool:
        return self._selection_tool.has_selection()

    def _set_selection(self, mask: Optional[np.ndarray]):
        img_w = self.current_qimage.width() if self.current_qimage else 0
        img_h = self.current_qimage.height() if self.current_qimage else 0
        sel_px, total = self._selection_tool.set_mask(mask, img_w, img_h)

        if self._selection_tool.has_selection():
            if not self._ants_timer.isActive():
                self._ants_timer.start()
        else:
            self._ants_timer.stop()

        self.selectionChanged.emit(sel_px, total)
        self.update()

    def _advance_marching_ants(self):
        br = self._selection_tool.advance_ants()
        if br is not None:
            tl = self.image_to_widget_coords(br.topLeft())
            br_pt = self.image_to_widget_coords(br.bottomRight())
            ants_rect = QRectF(tl, br_pt).normalized().toAlignedRect().adjusted(-4, -4, 4, 4)
            self.update(ants_rect)
        else:
            self.update()

    def clear_selection(self):
        if self._selection_tool.has_selection() or self._selection_tool.contour_path is not None:
            self._set_selection(None)

    def select_all(self):
        if self.current_qimage:
            self._set_selection(
                sel_ops.select_all(self.current_qimage.height(), self.current_qimage.width())
            )

    def invert_selection(self):
        if self._selection_tool.active_mask is not None:
            self._set_selection(sel_ops.invert_selection(self._selection_tool.active_mask))

    def delete_selection(self) -> bool:
        """Vuelve transparentes los píxeles seleccionados preservando la selección activa."""
        try:
            new_qimg, should_clear = self._selection_tool.delete_selected_pixels(self.current_qimage)
            if should_clear:
                self.clear_selection()
                return False
            if new_qimg is None:
                return False
            self.current_qimage = new_qimg
            self.imageEditedDirectly.emit(tr("history_delete_selection"))
            self.update()
            return True
        except Exception as e:
            logger.warning("Error borrando selección: %s", e)
            return False

    # -------------------------------------------------------------
    # Paneo, Zoom y Coordenadas
    # -------------------------------------------------------------
    def get_min_zoom(self) -> float:
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            return 0.1
        fit_scale = min(
            self.width() / max(1, self.current_qimage.width()),
            self.height() / max(1, self.current_qimage.height()),
        )
        return max(0.08, min(fit_scale * 0.5, 1.0))

    @staticmethod
    def _clamp_axis_offset(view_dim: float, img_dim: float, current_offset: float) -> float:
        if img_dim <= view_dim:
            return (view_dim - img_dim) / 2.0
        padding = min(80.0, view_dim * 0.2)
        return max(view_dim - img_dim - padding, min(padding, current_offset))

    def _clamp_pan_offset(self):
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            return
        vw = float(self.width())
        vh = float(self.height())
        iw = float(self.current_qimage.width()) * self.zoom_factor
        ih = float(self.current_qimage.height()) * self.zoom_factor
        self.pan_offset = QPointF(
            self._clamp_axis_offset(vw, iw, self.pan_offset.x()),
            self._clamp_axis_offset(vh, ih, self.pan_offset.y()),
        )
        self._update_scrollbars()

    def fit_to_view(self):
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            return
        margin = 32
        avail_w = max(10, self.width() - margin * 2)
        avail_h = max(10, self.height() - margin * 2)
        img_w = self.current_qimage.width()
        img_h = self.current_qimage.height()
        self.zoom_factor = min(avail_w / img_w, avail_h / img_h, 1.0)
        self.pan_offset = QPointF(
            (self.width() - img_w * self.zoom_factor) / 2.0,
            (self.height() - img_h * self.zoom_factor) / 2.0,
        )
        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def reset_zoom_100(self):
        if not self.current_qimage:
            return
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(
            (self.width() - self.current_qimage.width()) / 2.0,
            (self.height() - self.current_qimage.height()) / 2.0,
        )
        self._clamp_pan_offset()
        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def set_zoom_delta(self, factor_multiplier: float, center_point: Optional[QPoint] = None):
        old_zoom = self.zoom_factor
        new_zoom = max(self.get_min_zoom(), min(30.0, self.zoom_factor * factor_multiplier))
        if abs(new_zoom - old_zoom) < 1e-4:
            return
        if center_point is None:
            center_point = QPoint(self.width() // 2, self.height() // 2)

        wx, wy = center_point.x(), center_point.y()
        ix = (wx - self.pan_offset.x()) / old_zoom
        iy = (wy - self.pan_offset.y()) / old_zoom
        self.zoom_factor = new_zoom
        self.pan_offset = QPointF(wx - ix * new_zoom, wy - iy * new_zoom)
        self._clamp_pan_offset()
        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def widget_to_image_coords(self, pt: QPointF) -> QPointF:
        if self.zoom_factor <= 0:
            return QPointF(0, 0)
        return QPointF(
            (pt.x() - self.pan_offset.x()) / self.zoom_factor,
            (pt.y() - self.pan_offset.y()) / self.zoom_factor,
        )

    def image_to_widget_coords(self, pt: QPointF) -> QPointF:
        return QPointF(
            pt.x() * self.zoom_factor + self.pan_offset.x(),
            pt.y() * self.zoom_factor + self.pan_offset.y(),
        )

    def get_image_rect_in_widget(self) -> QRectF:
        if not self.current_qimage:
            return QRectF()
        return QRectF(
            self.pan_offset.x(),
            self.pan_offset.y(),
            self.current_qimage.width() * self.zoom_factor,
            self.current_qimage.height() * self.zoom_factor,
        )

    # -------------------------------------------------------------
    # Eventos de Entrada
    # -------------------------------------------------------------
    def _update_comparison_split(self, pos_x: float, img_rect: QRectF) -> None:
        ratio = (pos_x - img_rect.left()) / max(1.0, img_rect.width())
        self.comparison_split_ratio = max(0.01, min(0.99, ratio))

    def _brush_cursor_rect(self, center: QPoint) -> QRect:
        r = int((self.brush_size * self.zoom_factor) / 2.0) + 6
        return QRect(center.x() - r, center.y() - r, r * 2, r * 2)

    def wheelEvent(self, event: QWheelEvent):
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.15 if angle > 0 else (1.0 / 1.15)
            self.set_zoom_delta(factor, event.position().toPoint())
            event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        self.mouse_cursor_pos = pos.toPoint()

        if self.comparison_mode and self.current_qimage and event.button() == Qt.MouseButton.LeftButton:
            img_rect = self.get_image_rect_in_widget()
            split_x = img_rect.left() + img_rect.width() * self.comparison_split_ratio
            if abs(pos.x() - split_x) <= 24 or img_rect.contains(pos):
                self.is_dragging_split = True
                self._update_comparison_split(pos.x(), img_rect)
                self.setCursor(Qt.CursorShape.SplitHCursor)
                self.update()
                event.accept()
                return

        if event.button() == Qt.MouseButton.MiddleButton or self.space_pressed or (
            event.button() == Qt.MouseButton.RightButton and not self.brush_mode
        ):
            self.is_panning = True
            self.last_pan_pos = pos.toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self._crop_tool.active and self.current_qimage and event.button() == Qt.MouseButton.LeftButton:
            img_pt = self.widget_to_image_coords(pos)
            if self._crop_tool.handle_mouse_press(
                img_pt, pos, self.image_to_widget_coords,
                self.current_qimage.width(), self.current_qimage.height(),
            ):
                self.update()
                event.accept()
                return

        if self.brush_mode and self.wand_mode and self.wand_ai_mode and self.current_qimage:
            img_pt = self.widget_to_image_coords(pos)
            mods = event.modifiers()
            if event.button() == Qt.MouseButton.RightButton or (
                event.button() == Qt.MouseButton.LeftButton and bool(mods & Qt.KeyboardModifier.AltModifier)
            ):
                self._apply_sam_refinement_point(img_pt, label=0, reset_points=False)
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton and bool(mods & Qt.KeyboardModifier.ShiftModifier):
                self._apply_sam_refinement_point(img_pt, label=1, reset_points=not self._selection_tool.sam_points)
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self.brush_mode:
            if self.wand_mode and self.current_qimage:
                self._apply_magic_wand(
                    self.widget_to_image_coords(pos), self._effective_selection_mode(event.modifiers())
                )
                event.accept()
                return
            if self.current_qimage:
                self.is_drawing_mask = True
                img_pt = self.widget_to_image_coords(pos)
                self.last_draw_point = img_pt
                self._draw_brush_stroke(img_pt, img_pt)
                self.update()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        prev_cursor_pos = self.mouse_cursor_pos
        self.mouse_cursor_pos = pos.toPoint()

        if self.is_panning:
            delta = pos.toPoint() - self.last_pan_pos
            self.pan_offset += QPointF(delta.x(), delta.y())
            self._clamp_pan_offset()
            self.last_pan_pos = pos.toPoint()
            self.update()
            event.accept()
            return

        if self._crop_tool.active and self.current_qimage:
            if self._crop_tool.handle_mouse_move(
                self.widget_to_image_coords(pos),
                self.current_qimage.width(),
                self.current_qimage.height(),
            ):
                self.update()
                event.accept()
                return

        if self.comparison_mode and self.current_qimage:
            img_rect = self.get_image_rect_in_widget()
            split_x = img_rect.left() + img_rect.width() * self.comparison_split_ratio
            if self.is_dragging_split:
                self._update_comparison_split(pos.x(), img_rect)
                self.update()
                event.accept()
                return
            if abs(pos.x() - split_x) <= 14 and img_rect.top() <= pos.y() <= img_rect.bottom():
                self.setCursor(Qt.CursorShape.SplitHCursor)
            elif not self.brush_mode:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.is_drawing_mask and self.brush_mode and self.current_qimage:
            img_pt = self.widget_to_image_coords(pos)
            prev_pt = self.last_draw_point
            if prev_pt:
                if self.brush_smoothness > 0:
                    factor = 1.0 - (self.brush_smoothness / 100.0) * 0.98
                    current_pt = prev_pt + (img_pt - prev_pt) * factor
                else:
                    current_pt = img_pt
                self._draw_brush_stroke(prev_pt, current_pt)
                self.last_draw_point = current_pt
                w_p1 = self.image_to_widget_coords(prev_pt).toPoint()
                w_p2 = self.image_to_widget_coords(current_pt).toPoint()
                dirty_rect = (
                    self._brush_cursor_rect(w_p1)
                    .united(self._brush_cursor_rect(w_p2))
                    .united(self._brush_cursor_rect(prev_cursor_pos))
                    .united(self._brush_cursor_rect(self.mouse_cursor_pos))
                )
                self.update(dirty_rect)
            else:
                self.last_draw_point = img_pt
                self.update(self._brush_cursor_rect(self.mouse_cursor_pos))
            event.accept()
            return

        if self.brush_mode and not self.wand_mode:
            self.update(
                self._brush_cursor_rect(prev_cursor_pos).united(
                    self._brush_cursor_rect(self.mouse_cursor_pos)
                )
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.comparison_mode and self.is_dragging_split:
            self.is_dragging_split = False
            self.update()
            event.accept()
            return

        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or self.is_panning:
            self.is_panning = False
            self.setCursor(self._tool_cursor())
            self.update()

        if self._crop_tool.active and event.button() == Qt.MouseButton.LeftButton:
            if self._crop_tool.handle_mouse_release():
                self.update()
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing_mask:
            if self.brush_smoothness > 0 and self.last_draw_point:
                self._draw_brush_stroke(
                    self.last_draw_point, self.widget_to_image_coords(event.position())
                )
            self.is_drawing_mask = False
            self.last_draw_point = None
            self.imageEditedDirectly.emit("Edición Manual")
            self.update()

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_pressed = True
            if not self.is_panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_pressed = False
            if not self.is_panning:
                self.setCursor(self._tool_cursor())
        super().keyReleaseEvent(event)

    def leaveEvent(self, event):
        self.mouse_cursor_pos = QPoint(-1000, -1000)
        self.update()
        super().leaveEvent(event)

    @staticmethod
    def _accept_media_drag(event) -> None:
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()

    def dragEnterEvent(self, event):
        self._accept_media_drag(event)

    def dragMoveEvent(self, event):
        self._accept_media_drag(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    self.fileDropped.emit(path)
                    event.acceptProposedAction()
                    return
        if mime.hasImage():
            img_data = mime.imageData()
            qimg = img_data if isinstance(img_data, QImage) else QImage(img_data)
            if not qimg.isNull():
                self.imageDropped.emit(qimage_to_pil(qimg))
                event.acceptProposedAction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_pan_offset()
        self._center_drop_zone()
        sb_thick, margin = 12, 4
        self.v_scrollbar.setGeometry(
            self.width() - sb_thick - margin, margin, sb_thick, self.height() - (margin * 2) - sb_thick
        )
        self.h_scrollbar.setGeometry(
            margin, self.height() - sb_thick - margin, self.width() - (margin * 2) - sb_thick, sb_thick
        )
        self._update_scrollbars()
        self.update()

    def _center_drop_zone(self):
        if hasattr(self, "drop_zone"):
            self.drop_zone.move(
                int((self.width() - self.drop_zone.width()) / 2),
                int((self.height() - self.drop_zone.height()) / 2),
            )

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if (event.button() == Qt.MouseButton.LeftButton and not self.brush_mode) or (
            event.button() == Qt.MouseButton.MiddleButton
        ):
            self.fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # -------------------------------------------------------------
    # Trazado y Varita
    # -------------------------------------------------------------
    def _draw_brush_stroke(self, start_pt: QPointF, end_pt: QPointF):
        self._brush_engine.draw_stroke(
            target_qimage=self.current_qimage,
            original_qimage=self.original_qimage,
            start_pt=start_pt,
            end_pt=end_pt,
            brush_size=self.brush_size,
            brush_opacity=self.brush_opacity,
            brush_hardness=self.brush_hardness,
            eraser_mode=self.eraser_mode,
            active_selection=getattr(self, "active_selection", None),
        )

    def _effective_selection_mode(self, modifiers) -> str:
        return self._selection_tool.effective_mode(modifiers)

    def _sam_async_trigger(self):
        main_win = self.window()
        if main_win is not self and hasattr(main_win, "_ensure_sam_embedding_async"):
            return main_win._ensure_sam_embedding_async
        return None

    def _apply_magic_wand(self, pt: QPointF, mode: str = "new"):
        if not self.current_qimage:
            return
        try:
            combined = self._selection_tool.compute_wand_mask(
                self.current_qimage, pt, mode=mode, async_sam_trigger=self._sam_async_trigger()
            )
            if combined is not None:
                self._set_selection(combined)
        except Exception as e:
            logger.warning("Error en Varita: %s", e)

    def _apply_sam_refinement_point(self, pt: QPointF, label: int, reset_points: bool = False):
        if not self.current_qimage:
            return
        mask = self._selection_tool.add_sam_refinement_point(
            self.current_qimage,
            pt,
            label=label,
            reset_points=reset_points,
            async_sam_trigger=self._sam_async_trigger(),
        )
        if mask is not None:
            self._set_selection(mask)

    def _draw_marching_ants(self, painter: QPainter, img_rect: QRectF):
        self._selection_tool.draw_marching_ants(painter, img_rect, self.zoom_factor)

    # -------------------------------------------------------------
    # Pintado y Barras de Desplazamiento
    # -------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if not self.current_qimage:
            if hasattr(self, "drop_zone"):
                if not self.drop_zone.isVisible():
                    self.drop_zone.show()
                self._center_drop_zone()
            return

        img_rect = self.get_image_rect_in_widget()
        draw_canvas_scene(
            painter=painter,
            img_rect=img_rect,
            current_qimage=self.current_qimage,
            checker_pixmap=self.checker_pixmap,
            bg_color_rgb=self.bg_color_rgb,
            selection_qimage=self.selection_qimage,
            comparison_mode=self.comparison_mode,
            before_qimage=self.before_qimage,
            comparison_split_ratio=self.comparison_split_ratio,
            is_dark=getattr(self, "_is_dark_theme", True),
        )

        if not self.comparison_mode:
            self._draw_marching_ants(painter, img_rect)
            self._selection_tool.draw_sam_points(painter, img_rect, self.zoom_factor)
            self._crop_tool.draw_overlay(painter, img_rect, self.image_to_widget_coords)

        if (
            self.brush_mode
            and not self.wand_mode
            and self.rect().contains(self.mouse_cursor_pos)
            and not self.is_panning
        ):
            self._draw_brush_cursor(painter)

    def _draw_brush_cursor(self, painter: QPainter):
        draw_brush_cursor(
            painter, self.mouse_cursor_pos, self.brush_size, self.zoom_factor, self.eraser_mode
        )

    def _draw_comparison_divider(self, painter: QPainter, img_rect: QRectF, split_x: float):
        draw_comparison_divider(
            painter, img_rect, split_x, is_dark=getattr(self, "_is_dark_theme", True)
        )

    def _scrollbar_style(self, is_dark: bool = True) -> str:
        return build_scrollbar_stylesheet(is_dark)

    def retranslate_ui(self):
        if hasattr(self, "drop_zone"):
            self.drop_zone.retranslate_ui()

    def update_theme(self, is_dark: bool):
        self._is_dark_theme = is_dark
        if hasattr(self, "drop_zone"):
            self.drop_zone.update_theme(is_dark)
        style = self._scrollbar_style(is_dark)
        for sb_attr in ("v_scrollbar", "h_scrollbar"):
            if hasattr(self, sb_attr):
                getattr(self, sb_attr).setStyleSheet(style)
        self.update()

    @staticmethod
    def _sync_axis_scrollbar(scrollbar: QScrollBar, view_dim: float, img_dim: float, pan_coord: float) -> None:
        if img_dim <= view_dim:
            scrollbar.hide()
            return
        scrollbar.show()
        padding = min(80.0, view_dim * 0.2)
        min_off = view_dim - img_dim - padding
        scrollbar.setRange(0, int(padding - min_off))
        scrollbar.setPageStep(int(view_dim))
        scrollbar.setValue(int(padding - pan_coord))

    def _update_scrollbars(self):
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            self.v_scrollbar.hide()
            self.h_scrollbar.hide()
            return
        self._updating_scrollbars = True
        self._sync_axis_scrollbar(
            self.h_scrollbar,
            float(self.width()),
            float(self.current_qimage.width()) * self.zoom_factor,
            self.pan_offset.x(),
        )
        self._sync_axis_scrollbar(
            self.v_scrollbar,
            float(self.height()),
            float(self.current_qimage.height()) * self.zoom_factor,
            self.pan_offset.y(),
        )
        self._updating_scrollbars = False

    def _apply_scrollbar_axis(self, is_horizontal: bool, value: int) -> None:
        if self._updating_scrollbars:
            return
        dim = float(self.width() if is_horizontal else self.height())
        target = min(80.0, dim * 0.2) - value
        if is_horizontal:
            self.pan_offset.setX(target)
        else:
            self.pan_offset.setY(target)
        self.update()

    def _on_h_scrollbar_changed(self, value):
        self._apply_scrollbar_axis(True, value)

    def _on_v_scrollbar_changed(self, value):
        self._apply_scrollbar_axis(False, value)
