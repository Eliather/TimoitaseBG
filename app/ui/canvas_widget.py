"""
TimoitaseBG - Canvas interactivo para visualización, zoom, paneo y trazado de máscara de pincel.
"""
from typing import Optional
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, Signal, QTimer
from PySide6.QtWidgets import QWidget, QScrollBar
from app.i18n import tr
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QImage,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
)
from app.core.image_utils import (
    pil_to_qimage,
    qimage_to_pil,
    create_checkerboard_pattern,
)
from app.core import selection as sel_ops
from app.config import DEFAULT_BRUSH_SIZE, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE, CHECKER_SIZE
from app.ui.canvas import BrushEngine, SelectionTool


class CanvasWidget(QWidget):
    """
    Widget central que muestra la imagen, soporta zoom/paneo interactivo y
    permite dibujar máscaras para el pincel mágico con vista previa en tiempo real.
    """

    maskChanged = Signal(bool)  # Emite True si hay trazos dibujados, False si está limpia
    zoomChanged = Signal(float) # Emite el porcentaje de zoom actual
    fileDropped = Signal(str)   # Emite la ruta de archivo si el usuario arrastra una imagen
    imageDropped = Signal(object)  # Emite un PIL.Image cuando se arrastra una imagen en memoria
    browseRequested = Signal()  # Emite cuando el usuario hace clic en la zona vacía para explorar
    imageEditedDirectly = Signal(str)  # Descripción de la edición para el historial
    selectionChanged = Signal(int, int)  # (píxeles seleccionados, píxeles totales)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        # Estados de visualización
        self.current_qimage: Optional[QImage] = None
        self.original_qimage: Optional[QImage] = None
        
        # Color de fondo en capa separada
        self.bg_color_rgb: Optional[tuple[int, int, int]] = None

        # Zoom y paneo
        self.zoom_factor: float = 1.0
        self.pan_offset: QPointF = QPointF(0, 0)
        self.is_panning: bool = False
        self.last_pan_pos: QPoint = QPoint()
        self.space_pressed: bool = False
        
        # Scrollbars flotantes
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        self.v_scrollbar.setStyleSheet(self._scrollbar_style())
        self.v_scrollbar.valueChanged.connect(self._on_v_scrollbar_changed)
        self.v_scrollbar.hide()
        
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self.h_scrollbar.setStyleSheet(self._scrollbar_style())
        self.h_scrollbar.valueChanged.connect(self._on_h_scrollbar_changed)
        self.h_scrollbar.hide()
        self._updating_scrollbars = False

        # Pincel mágico
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

        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(40)  # 25 FPS reales
        self._ants_timer.timeout.connect(self._advance_marching_ants)

        # Patrón de transparencia
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

    @property
    def wand_mode(self) -> bool:
        return self._selection_tool.wand_mode

    @wand_mode.setter
    def wand_mode(self, value: bool) -> None:
        self._selection_tool.wand_mode = value

    @property
    def wand_ai_mode(self) -> bool:
        return self._selection_tool.wand_ai_mode

    @wand_ai_mode.setter
    def wand_ai_mode(self, value: bool) -> None:
        self._selection_tool.wand_ai_mode = value

    @property
    def wand_tolerance(self) -> int:
        return self._selection_tool.wand_tolerance

    @wand_tolerance.setter
    def wand_tolerance(self, value: int) -> None:
        self._selection_tool.wand_tolerance = value

    @property
    def wand_selection_mode(self) -> str:
        return self._selection_tool.wand_selection_mode

    @wand_selection_mode.setter
    def wand_selection_mode(self, value: str) -> None:
        self._selection_tool.wand_selection_mode = value

    @property
    def wand_anti_alias(self) -> bool:
        return self._selection_tool.wand_anti_alias

    @wand_anti_alias.setter
    def wand_anti_alias(self, value: bool) -> None:
        self._selection_tool.wand_anti_alias = value

    @property
    def wand_contiguous(self) -> bool:
        return self._selection_tool.wand_contiguous

    @wand_contiguous.setter
    def wand_contiguous(self, value: bool) -> None:
        self._selection_tool.wand_contiguous = value

    @property
    def active_selection(self) -> Optional[np.ndarray]:
        return self._selection_tool.active_mask

    @active_selection.setter
    def active_selection(self, value: Optional[np.ndarray]) -> None:
        self._selection_tool.active_mask = value

    @property
    def selection_qimage(self) -> Optional[QImage]:
        return self._selection_tool.overlay_qimage

    @property
    def selection_path(self):
        return self._selection_tool.contour_path

    @property
    def _pending_wand_click(self) -> Optional[tuple]:
        return self._selection_tool.pending_wand_click

    @_pending_wand_click.setter
    def _pending_wand_click(self, value: Optional[tuple]) -> None:
        self._selection_tool.pending_wand_click = value

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
        if pil_img:
            qimg = pil_to_qimage(pil_img)
            if not qimg.hasAlphaChannel():
                qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
            self.original_qimage = qimg
        else:
            self.original_qimage = None

    def set_background_color(self, color_rgb: Optional[tuple[int, int, int]]):
        """Define un color de fondo sólido para renderizar detrás de la imagen transparente."""
        self.bg_color_rgb = color_rgb
        self.update()

    # -------------------------------------------------------------
    # Gestión de Imagen y Máscara
    # -------------------------------------------------------------
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

        # Normalizar siempre a formato con canal alfa para que el borrador manual funcione en imágenes RGB
        qimg = pil_to_qimage(pil_img)
        if not qimg.hasAlphaChannel():
            qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
        self.current_qimage = qimg
        img_w = self.current_qimage.width()
        img_h = self.current_qimage.height()

        # Una selección solo es válida para imágenes del mismo tamaño
        if reset_zoom or (img_w, img_h) != (prev_w, prev_h):
            self.clear_selection()

        if reset_zoom or prev_w == 0:
            self.fit_to_view()
        else:
            self._clamp_pan_offset()
            self.update()



    def _tool_cursor(self) -> Qt.CursorShape:
        """Cursor adecuado para la herramienta activa."""
        if not self.brush_mode:
            return Qt.CursorShape.ArrowCursor
        if self.wand_mode:
            return Qt.CursorShape.CrossCursor
        return Qt.CursorShape.BlankCursor

    def set_brush_mode(self, enabled: bool):
        """Activa o desactiva el modo de dibujo de máscara."""
        self.brush_mode = enabled
        self.setCursor(self._tool_cursor())
        self.update()

    def set_brush_size(self, size: int):
        """Ajusta el radio del pincel en píxeles de imagen nativa."""
        self.brush_size = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, size))
        self.update()

    def set_brush_smoothness(self, smoothness: int):
        """Ajusta el nivel de suavizado del pincel (0 a 100)."""
        self.brush_smoothness = max(0, min(100, smoothness))

    def set_brush_opacity(self, opacity: float):
        """Ajusta la opacidad global del pincel."""
        self.brush_opacity = opacity
        self.update()

    def set_brush_hardness(self, hardness: float):
        """Ajusta la dureza del pincel (0.0 a 1.0)."""
        self.brush_hardness = hardness
        self.update()

    def set_eraser_mode(self, enabled: bool):
        """Activa o desactiva el modo borrador (borrar trazos de la máscara)."""
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
        """Establece la máscara de selección y actualiza temporizador y señales."""
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
        """Deseleccionar (Ctrl+D)."""
        if self._selection_tool.has_selection() or self._selection_tool.contour_path is not None:
            self._set_selection(None)

    def select_all(self):
        """Seleccionar todo (Ctrl+A)."""
        if self.current_qimage:
            self._set_selection(
                sel_ops.select_all(self.current_qimage.height(), self.current_qimage.width())
            )

    def invert_selection(self):
        """Invertir selección (Ctrl+Shift+I)."""
        if self._selection_tool.active_mask is not None:
            self._set_selection(sel_ops.invert_selection(self._selection_tool.active_mask))

    def delete_selection(self) -> bool:
        """
        Borra (vuelve transparentes) los píxeles seleccionados, como Supr en Photoshop.
        La selección se mantiene activa tras borrar. Devuelve True si hubo cambios.
        """
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
            print(f"Error borrando selección: {e}")
            return False

    # -------------------------------------------------------------
    # Paneo, Zoom y Transformaciones
    # -------------------------------------------------------------
    def get_min_zoom(self) -> float:
        """
        Calcula el factor de zoom mínimo permitido para que la imagen nunca
        se vuelva un punto minúsculo imperceptible ni se pierda de vista.
        """
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            return 0.1
        img_w = self.current_qimage.width()
        img_h = self.current_qimage.height()
        fit_scale = min(self.width() / max(1, img_w), self.height() / max(1, img_h))
        # Permitir alejar como máximo hasta un 50% de la vista completa, con suelo mínimo de 0.08
        return max(0.08, min(fit_scale * 0.5, 1.0))

    def _clamp_pan_offset(self):
        """
        Garantiza que la imagen nunca se salga del área visible del canvas ni se pierda.
        - Si la dimensión de la imagen cabe en el canvas, se mantiene centrada en ese eje.
        - Si la dimensión es mayor que el canvas, permite paneo libre pero limita
          los bordes a un margen de seguridad para que la imagen siempre esté visible.
        """
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            return

        vw = float(self.width())
        vh = float(self.height())
        iw = float(self.current_qimage.width()) * self.zoom_factor
        ih = float(self.current_qimage.height()) * self.zoom_factor

        # Margen de seguridad cómodo para trabajar bordes con el pincel
        padding_x = min(80.0, vw * 0.2)
        padding_y = min(80.0, vh * 0.2)

        # Eje Horizontal (X)
        if iw <= vw:
            target_x = (vw - iw) / 2.0
        else:
            min_x = vw - iw - padding_x
            max_x = padding_x
            target_x = max(min_x, min(max_x, self.pan_offset.x()))

        # Eje Vertical (Y)
        if ih <= vh:
            target_y = (vh - ih) / 2.0
        else:
            min_y = vh - ih - padding_y
            max_y = padding_y
            target_y = max(min_y, min(max_y, self.pan_offset.y()))

        self.pan_offset = QPointF(target_x, target_y)
        self._update_scrollbars()

    def fit_to_view(self):
        """Ajusta la imagen para que encaje perfectamente dentro del widget."""
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            return

        margin = 32
        avail_w = max(10, self.width() - margin * 2)
        avail_h = max(10, self.height() - margin * 2)

        img_w = self.current_qimage.width()
        img_h = self.current_qimage.height()

        scale_w = avail_w / img_w
        scale_h = avail_h / img_h
        self.zoom_factor = min(scale_w, scale_h, 1.0)

        # Centrar
        rendered_w = img_w * self.zoom_factor
        rendered_h = img_h * self.zoom_factor
        self.pan_offset = QPointF(
            (self.width() - rendered_w) / 2.0,
            (self.height() - rendered_h) / 2.0,
        )
        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def reset_zoom_100(self):
        """Ajusta el zoom al 100% (escala 1:1) centrado."""
        if not self.current_qimage:
            return
        self.zoom_factor = 1.0
        rendered_w = self.current_qimage.width()
        rendered_h = self.current_qimage.height()
        self.pan_offset = QPointF(
            (self.width() - rendered_w) / 2.0,
            (self.height() - rendered_h) / 2.0,
        )
        self._clamp_pan_offset()
        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def set_zoom_delta(self, factor_multiplier: float, center_point: Optional[QPoint] = None):
        """Aplica un cambio de zoom centrado sobre una coordenada del canvas dentro de límites seguros."""
        old_zoom = self.zoom_factor
        min_zoom = self.get_min_zoom()
        new_zoom = max(min_zoom, min(30.0, self.zoom_factor * factor_multiplier))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        if center_point is None:
            center_point = QPoint(self.width() // 2, self.height() // 2)

        # Mantener el punto bajo el cursor en la misma posición visual
        widget_x = center_point.x()
        widget_y = center_point.y()

        img_x = (widget_x - self.pan_offset.x()) / old_zoom
        img_y = (widget_y - self.pan_offset.y()) / old_zoom

        self.zoom_factor = new_zoom
        self.pan_offset = QPointF(
            widget_x - img_x * new_zoom,
            widget_y - img_y * new_zoom,
        )
        self._clamp_pan_offset()

        self.zoomChanged.emit(self.zoom_factor)
        self.update()

    def widget_to_image_coords(self, pt: QPointF) -> QPointF:
        """Convierte una coordenada del widget a coordenadas nativas de la imagen."""
        if self.zoom_factor <= 0:
            return QPointF(0, 0)
        ix = (pt.x() - self.pan_offset.x()) / self.zoom_factor
        iy = (pt.y() - self.pan_offset.y()) / self.zoom_factor
        return QPointF(ix, iy)

    def image_to_widget_coords(self, pt: QPointF) -> QPointF:
        """Convierte coordenadas nativas de la imagen a coordenadas del widget."""
        wx = pt.x() * self.zoom_factor + self.pan_offset.x()
        wy = pt.y() * self.zoom_factor + self.pan_offset.y()
        return QPointF(wx, wy)

    def get_image_rect_in_widget(self) -> QRectF:
        """Retorna el rectángulo que ocupa la imagen en el widget."""
        if not self.current_qimage:
            return QRectF()
        return QRectF(
            self.pan_offset.x(),
            self.pan_offset.y(),
            self.current_qimage.width() * self.zoom_factor,
            self.current_qimage.height() * self.zoom_factor,
        )

    # -------------------------------------------------------------
    # Eventos de Entrada (Mouse, Rueda, Teclas, Drag & Drop)
    # -------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        """Zoom fluido con la rueda del ratón."""
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.15 if angle > 0 else (1.0 / 1.15)
            self.set_zoom_delta(factor, event.position().toPoint())
            event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        self.mouse_cursor_pos = pos.toPoint()

        # Manejo de la cortina en modo comparación con botón izquierdo
        if self.comparison_mode and self.current_qimage and event.button() == Qt.MouseButton.LeftButton:
            img_rect = self.get_image_rect_in_widget()
            split_x = img_rect.left() + img_rect.width() * self.comparison_split_ratio
            if abs(pos.x() - split_x) <= 24 or img_rect.contains(pos):
                self.is_dragging_split = True
                ratio = (pos.x() - img_rect.left()) / max(1.0, img_rect.width())
                self.comparison_split_ratio = max(0.01, min(0.99, ratio))
                self.setCursor(Qt.CursorShape.SplitHCursor)
                self.update()
                event.accept()
                return

        # Paneo con botón central, botón derecho o si la barra espaciadora está pulsada
        if event.button() == Qt.MouseButton.MiddleButton or self.space_pressed or (
            event.button() == Qt.MouseButton.RightButton and not self.brush_mode
        ):
            self.is_panning = True
            self.last_pan_pos = pos.toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.brush_mode:
            if self.wand_mode and self.current_qimage:
                img_pt = self.widget_to_image_coords(pos)
                self._apply_magic_wand(img_pt, self._effective_selection_mode(event.modifiers()))
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

    def _brush_cursor_rect(self, center: QPoint) -> QRect:
        """Retorna el QRect en coordenadas del widget que abarca el círculo del cursor del pincel."""
        r = int((self.brush_size * self.zoom_factor) / 2.0) + 6
        return QRect(center.x() - r, center.y() - r, r * 2, r * 2)

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

        if self.comparison_mode and self.current_qimage:
            img_rect = self.get_image_rect_in_widget()
            split_x = img_rect.left() + img_rect.width() * self.comparison_split_ratio
            if self.is_dragging_split:
                ratio = (pos.x() - img_rect.left()) / max(1.0, img_rect.width())
                self.comparison_split_ratio = max(0.01, min(0.99, ratio))
                self.update()
                event.accept()
                return
            elif abs(pos.x() - split_x) <= 14 and img_rect.top() <= pos.y() <= img_rect.bottom():
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
                dirty_stroke_rect = (
                    self._brush_cursor_rect(w_p1)
                    .united(self._brush_cursor_rect(w_p2))
                    .united(self._brush_cursor_rect(prev_cursor_pos))
                    .united(self._brush_cursor_rect(self.mouse_cursor_pos))
                )
                self.update(dirty_stroke_rect)
            else:
                self.last_draw_point = img_pt
                self.update(self._brush_cursor_rect(self.mouse_cursor_pos))
            event.accept()
            return

        if self.brush_mode and not self.wand_mode:
            dirty_rect = self._brush_cursor_rect(prev_cursor_pos).united(
                self._brush_cursor_rect(self.mouse_cursor_pos)
            )
            self.update(dirty_rect)

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

        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing_mask:
            if self.brush_smoothness > 0 and self.last_draw_point:
                img_pt = self.widget_to_image_coords(event.position())
                self._draw_brush_stroke(self.last_draw_point, img_pt)
                
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

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()

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
        sb_thick = 12
        margin = 4
        self.v_scrollbar.setGeometry(self.width() - sb_thick - margin, margin, sb_thick, self.height() - (margin * 2) - sb_thick)
        self.h_scrollbar.setGeometry(margin, self.height() - sb_thick - margin, self.width() - (margin * 2) - sb_thick, sb_thick)
        self._update_scrollbars()
        self.update()

    def _center_drop_zone(self):
        if hasattr(self, "drop_zone"):
            dw = self.drop_zone.width()
            dh = self.drop_zone.height()
            self.drop_zone.move(
                int((self.width() - dw) / 2),
                int((self.height() - dh) / 2),
            )

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Doble clic para re-encajar la imagen en la vista (fit_to_view)."""
        if (event.button() == Qt.MouseButton.LeftButton and not self.brush_mode) or (
            event.button() == Qt.MouseButton.MiddleButton
        ):
            self.fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # -------------------------------------------------------------
    # Trazo de Pincel sobre la Máscara
    # -------------------------------------------------------------
    def _draw_brush_stroke(self, start_pt: QPointF, end_pt: QPointF):
        """Dibuja una línea continua o punto con extremos redondeados, aplicando opacidad y dureza."""
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
        """Modo de selección combinando Shift/Alt como Photoshop."""
        return self._selection_tool.effective_mode(modifiers)

    def _apply_magic_wand(self, pt: QPointF, mode: str = "new"):
        """Selecciona píxeles (por IA de MobileSAM o por tolerancia de color)."""
        if not self.current_qimage:
            return
        try:
            main_win = self.window()
            sam_trigger = (
                main_win._ensure_sam_embedding_async
                if main_win is not self and hasattr(main_win, "_ensure_sam_embedding_async")
                else None
            )
            combined = self._selection_tool.compute_wand_mask(
                self.current_qimage, pt, mode=mode, async_sam_trigger=sam_trigger
            )
            if combined is not None:
                self._set_selection(combined)
        except Exception as e:
            print(f"Error en Varita: {e}")

    def _draw_marching_ants(self, painter: QPainter, img_rect: QRectF):
        """Dibuja el contorno animado de la selección (hormigas marchantes)."""
        self._selection_tool.draw_marching_ants(painter, img_rect, self.zoom_factor)

    # -------------------------------------------------------------
    # Pintado Principal
    # -------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # 1. Fondo general de la ventana (se deja que el QSS lo maneje)

        # Si no hay imagen cargada, la drop_zone se encarga de mostrar la tarjeta interactiva
        if not self.current_qimage:
            if hasattr(self, "drop_zone"):
                if not self.drop_zone.isVisible():
                    self.drop_zone.show()
                self._center_drop_zone()
            return

        img_rect = self.get_image_rect_in_widget()

        # 2. Sombra suave (drop shadow) alrededor de la imagen para que destaque sobre el fondo
        painter.save()
        for offset, alpha in [(8, 8), (5, 14), (3, 22), (1, 35)]:
            shadow_rect = img_rect.adjusted(-offset, -offset, offset, offset)
            painter.fillRect(shadow_rect, QColor(0, 0, 0, alpha))
        painter.restore()

        # Renderizado de imagen: Modo Comparación (Antes/Después) o Modo Normal
        if self.comparison_mode and self.before_qimage:
            split_x = img_rect.left() + img_rect.width() * self.comparison_split_ratio

            # Lado Izquierdo: "Antes" (Imagen Original)
            left_rect = QRectF(img_rect.left(), img_rect.top(), max(0.0, split_x - img_rect.left()), img_rect.height())
            painter.save()
            painter.setClipRect(left_rect)
            painter.drawTiledPixmap(img_rect, self.checker_pixmap)
            painter.drawImage(img_rect, self.before_qimage)
            painter.restore()

            # Lado Derecho: "Después" (Imagen Actual)
            right_rect = QRectF(split_x, img_rect.top(), max(0.0, img_rect.right() - split_x), img_rect.height())
            painter.save()
            painter.setClipRect(right_rect)
            if self.bg_color_rgb:
                painter.fillRect(img_rect, QColor(*self.bg_color_rgb))
            else:
                painter.drawTiledPixmap(img_rect, self.checker_pixmap)
            painter.drawImage(img_rect, self.current_qimage)
            painter.restore()

            # Línea divisoria y manija central
            self._draw_comparison_divider(painter, img_rect, split_x)
        else:
            # Fondo de damero sutil o color sólido dentro del área de la imagen para transparencias
            painter.save()
            painter.setClipRect(img_rect)
            if self.bg_color_rgb:
                painter.fillRect(img_rect, QColor(*self.bg_color_rgb))
            else:
                painter.drawTiledPixmap(img_rect, self.checker_pixmap)

            # Dibujar la imagen escalada
            painter.drawImage(img_rect, self.current_qimage)
            
            # Dibujar la selección (overlay azul) si existe
            if self.selection_qimage:
                painter.drawImage(img_rect, self.selection_qimage)
                
            painter.restore()

        # 5. (Eliminado overlay de máscara)

        # 6. Borde sutil alrededor de la imagen
        pen_border = QPen(QColor(61, 61, 61), 1) # #3D3D3D
        painter.setPen(pen_border)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(img_rect)

        # 6b. Hormigas marchantes de la selección activa
        if not self.comparison_mode:
            self._draw_marching_ants(painter, img_rect)

        # 7. Indicador del cursor del pincel (la varita usa cursor en cruz)
        if (
            self.brush_mode
            and not self.wand_mode
            and self.rect().contains(self.mouse_cursor_pos)
            and not self.is_panning
        ):
            self._draw_brush_cursor(painter)

    def _draw_brush_cursor(self, painter: QPainter):
        """Dibuja el círculo del tamaño del pincel sobre el cursor del ratón con precisión en esquinas."""
        visual_radius = (self.brush_size * self.zoom_factor) / 2.0
        cx = self.mouse_cursor_pos.x()
        cy = self.mouse_cursor_pos.y()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        if self.eraser_mode:
            # Cursor con doble halo de alto contraste para que sea visible tanto en fondos oscuros como claros
            painter.setBrush(QColor(15, 23, 42, 65))
            border_w = 1.0 if visual_radius < 3.0 else 1.5
            painter.setPen(QPen(QColor(255, 255, 255, 220), border_w + 1.0))
            painter.drawEllipse(QPointF(cx, cy), max(1.5, visual_radius), max(1.5, visual_radius))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(56, 189, 248, 240), border_w))
            painter.drawEllipse(QPointF(cx, cy), max(1.5, visual_radius), max(1.5, visual_radius))
        else:
            # Círculo semitransparente interior con tinte sakura
            painter.setBrush(QColor(255, 79, 121, 65))
            # Borde rosa sakura (#FF4F79)
            border_w = 1.0 if visual_radius < 3.0 else 1.5
            painter.setPen(QPen(QColor(255, 79, 121, 230), border_w))
            painter.drawEllipse(QPointF(cx, cy), max(1.5, visual_radius), max(1.5, visual_radius))

        # Punto central de precisión (micro-mira para esquinas)
        dot_r = 0.8 if visual_radius < 4.0 else 1.2
        if self.eraser_mode:
            painter.setBrush(QColor(56, 189, 248, 245))
        else:
            painter.setBrush(QColor(255, 79, 121, 240))
            
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), dot_r, dot_r)
        painter.restore()

    def _draw_comparison_divider(self, painter: QPainter, img_rect: QRectF, split_x: float):
        """Dibuja la línea divisoria vertical, la manija central con flechas y las etiquetas Antes/Después."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Línea divisoria vertical con sombra tenue y línea blanca
        painter.setPen(QPen(QColor(0, 0, 0, 70), 3.0))
        painter.drawLine(QPointF(split_x, img_rect.top()), QPointF(split_x, img_rect.bottom()))
        painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
        painter.drawLine(QPointF(split_x, img_rect.top()), QPointF(split_x, img_rect.bottom()))

        # 2. Manija central circular
        cy = img_rect.center().y()
        radius = 16.0
        handle_center = QPointF(split_x, cy)

        # Sombra de la manija
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 40))
        painter.drawEllipse(handle_center + QPointF(1, 1), radius + 1, radius + 1)

        # Círculo oscuro/claro con borde rosa sakura
        is_dark = getattr(self, "_is_dark_theme", True)
        painter.setBrush(QColor("#1E293B") if is_dark else QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#FF4F79"), 2.5))
        painter.drawEllipse(handle_center, radius, radius)

        # Flechas ◀ ▶ dentro de la manija
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

        # 3. Badges "ANTES" y "DESPUÉS" internacionalizados
        badge_y = img_rect.top() + 12
        if badge_y + 24 < img_rect.bottom():
            # Badge ANTES (Izquierda)
            badge_antes_rect = QRectF(img_rect.left() + 12, badge_y, 64, 22)
            painter.setBrush(QColor(15, 23, 42, 215))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_antes_rect, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            font_b = painter.font()
            font_b.setPixelSize(8)
            font_b.setBold(True)
            painter.setFont(font_b)
            painter.drawText(badge_antes_rect, Qt.AlignmentFlag.AlignCenter, tr("compare_before"))

            # Badge DESPUÉS (Derecha)
            badge_desp_rect = QRectF(img_rect.right() - 76, badge_y, 64, 22)
            painter.setBrush(QColor(244, 63, 94, 225))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_desp_rect, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge_desp_rect, Qt.AlignmentFlag.AlignCenter, tr("compare_after"))

        painter.restore()

    # -------------------------------------------------------------
    # Scrollbars (UI y Lógica)
    # -------------------------------------------------------------
    def _scrollbar_style(self, is_dark: bool = True):
        track = "rgba(15, 23, 42, 0.65)" if is_dark else "rgba(231, 229, 228, 0.4)"
        handle = "rgba(71, 85, 105, 0.8)" if is_dark else "rgba(120, 113, 108, 0.6)"
        handle_hover = "rgba(148, 163, 184, 0.95)" if is_dark else "rgba(120, 113, 108, 0.9)"
        return f"""
            QScrollBar:horizontal {{
                border: none;
                background: {track};
                height: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background: {handle};
                min-width: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {handle_hover};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: none;
                width: 0px;
            }}
            
            QScrollBar:vertical {{
                border: none;
                background: {track};
                width: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {handle};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {handle_hover};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
                height: 0px;
            }}
        """

    def retranslate_ui(self):
        """Actualiza los textos del lienzo al cambiar idioma."""
        if hasattr(self, "drop_zone"):
            self.drop_zone.retranslate_ui()

    def update_theme(self, is_dark: bool):
        """Actualiza el aspecto de los componentes del lienzo con el tema."""
        self._is_dark_theme = is_dark
        if hasattr(self, "drop_zone"):
            self.drop_zone.update_theme(is_dark)
        if hasattr(self, "v_scrollbar"):
            self.v_scrollbar.setStyleSheet(self._scrollbar_style(is_dark))
        if hasattr(self, "h_scrollbar"):
            self.h_scrollbar.setStyleSheet(self._scrollbar_style(is_dark))
        self.update()

    def _update_scrollbars(self):
        if not self.current_qimage or self.width() <= 0 or self.height() <= 0:
            self.v_scrollbar.hide()
            self.h_scrollbar.hide()
            return

        vw = float(self.width())
        vh = float(self.height())
        iw = float(self.current_qimage.width()) * self.zoom_factor
        ih = float(self.current_qimage.height()) * self.zoom_factor

        padding_x = min(80.0, vw * 0.2)
        padding_y = min(80.0, vh * 0.2)

        self._updating_scrollbars = True

        if iw <= vw:
            self.h_scrollbar.hide()
        else:
            self.h_scrollbar.show()
            min_x = vw - iw - padding_x
            max_x = padding_x
            range_x = int(max_x - min_x)
            self.h_scrollbar.setRange(0, range_x)
            self.h_scrollbar.setPageStep(int(vw))
            val_x = int(max_x - self.pan_offset.x())
            self.h_scrollbar.setValue(val_x)

        if ih <= vh:
            self.v_scrollbar.hide()
        else:
            self.v_scrollbar.show()
            min_y = vh - ih - padding_y
            max_y = padding_y
            range_y = int(max_y - min_y)
            self.v_scrollbar.setRange(0, range_y)
            self.v_scrollbar.setPageStep(int(vh))
            val_y = int(max_y - self.pan_offset.y())
            self.v_scrollbar.setValue(val_y)

        self._updating_scrollbars = False

    def _on_h_scrollbar_changed(self, value):
        if self._updating_scrollbars: return
        vw = float(self.width())
        padding_x = min(80.0, vw * 0.2)
        target_x = padding_x - value
        self.pan_offset.setX(target_x)
        self.update()

    def _on_v_scrollbar_changed(self, value):
        if self._updating_scrollbars: return
        vh = float(self.height())
        padding_y = min(80.0, vh * 0.2)
        target_y = padding_y - value
        self.pan_offset.setY(target_y)
        self.update()
