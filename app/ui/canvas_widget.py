"""
TimoitaseBG - Canvas interactivo para visualización, zoom, paneo y trazado de máscara de pincel.
"""
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, Signal, QTimer
from PySide6.QtWidgets import QWidget, QScrollBar
from app.i18n import tr
from PySide6.QtGui import (
    QPainter,
    QPainterPath,
    QColor,
    QPen,
    QBrush,
    QImage,
    QPixmap,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
    QCursor,
)
from app.core.image_utils import (
    pil_to_qimage,
    qimage_to_pil,
    create_checkerboard_pattern,
)
from app.core import selection as sel_ops
from app.config import DEFAULT_BRUSH_SIZE, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE, CHECKER_SIZE
from app.ui.icons import render_icon


class CanvasWidget(QWidget):
    """
    Widget central que muestra la imagen, soporta zoom/paneo interactivo y
    permite dibujar máscaras para el pincel mágico con vista previa en tiempo real.
    """

    maskChanged = Signal(bool)  # Emite True si hay trazos dibujados, False si está limpia
    zoomChanged = Signal(float) # Emite el porcentaje de zoom actual
    fileDropped = Signal(str)   # Emite la ruta de archivo si el usuario arrastra una imagen
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
        self.wand_mode: bool = False
        self.wand_ai_mode: bool = False         # True: MobileSAM por clic, False: Color por tolerancia
        self.wand_tolerance: int = 32           # 0-255, igual que Photoshop
        self.wand_selection_mode: str = "new"   # new | add | subtract | intersect
        self.wand_anti_alias: bool = True
        self.wand_contiguous: bool = True
        
        # Selección (máscara uint8 0-255) + representación visual
        self.active_selection: Optional[np.ndarray] = None
        self.selection_qimage: Optional[QImage] = None
        self.selection_path: Optional[QPainterPath] = None
        self._ants_offset: float = 0.0
        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(140)
        self._ants_timer.timeout.connect(self._advance_marching_ants)

        # Patrón de transparencia
        self.checker_pixmap = create_checkerboard_pattern(size=CHECKER_SIZE)

        # Modo Comparación (Antes / Después con cortina deslizante)
        self.comparison_mode: bool = False
        self.before_qimage: Optional[QImage] = None
        self.comparison_split_ratio: float = 0.5
        self.is_dragging_split: bool = False

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
            self.original_qimage = pil_to_qimage(pil_img)
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
            self.update()
            self.maskChanged.emit(False)
            return

        prev_w = self.current_qimage.width() if self.current_qimage else 0
        prev_h = self.current_qimage.height() if self.current_qimage else 0

        self.current_qimage = pil_to_qimage(pil_img)
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
    # Selección (estilo Photoshop)
    # -------------------------------------------------------------
    def has_selection(self) -> bool:
        return self.active_selection is not None

    def _set_selection(self, mask: Optional[np.ndarray]):
        """Establece la máscara de selección y regenera overlay, contorno y señales."""
        if mask is not None and sel_ops.selection_pixel_count(mask) == 0:
            mask = None

        self.active_selection = mask
        total = 0
        if self.current_qimage:
            total = self.current_qimage.width() * self.current_qimage.height()

        if mask is None:
            self.selection_qimage = None
            self.selection_path = None
            self._ants_timer.stop()
            self.selectionChanged.emit(0, total)
            self.update()
            return

        h, w = mask.shape
        # Tinte azul muy sutil para ver las islas pequeñas; el borde lo marcan las "hormigas"
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[..., 1] = 120
        overlay[..., 2] = 255
        overlay[..., 3] = (mask.astype(np.uint16) * 45 // 255).astype(np.uint8)
        self.selection_qimage = pil_to_qimage(Image.fromarray(overlay, "RGBA"))

        # Contorno vectorial en coordenadas de imagen para las hormigas marchantes
        binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        path = QPainterPath()
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            if len(pts) == 0:
                continue
            path.moveTo(float(pts[0][0]) + 0.5, float(pts[0][1]) + 0.5)
            for px, py in pts[1:]:
                path.lineTo(float(px) + 0.5, float(py) + 0.5)
            path.closeSubpath()
        self.selection_path = path

        if not self._ants_timer.isActive():
            self._ants_timer.start()
        self.selectionChanged.emit(sel_ops.selection_pixel_count(mask), total)
        self.update()

    def _advance_marching_ants(self):
        self._ants_offset = (self._ants_offset + 1.0) % 8.0
        self.update()

    def clear_selection(self):
        """Deseleccionar (Ctrl+D)."""
        if self.active_selection is not None or self.selection_path is not None:
            self._set_selection(None)

    def select_all(self):
        """Seleccionar todo (Ctrl+A)."""
        if self.current_qimage:
            self._set_selection(
                sel_ops.select_all(self.current_qimage.height(), self.current_qimage.width())
            )

    def invert_selection(self):
        """Invertir selección (Ctrl+Shift+I)."""
        if self.active_selection is not None:
            self._set_selection(sel_ops.invert_selection(self.active_selection))

    def delete_selection(self) -> bool:
        """
        Borra (vuelve transparentes) los píxeles seleccionados, como Supr en Photoshop.
        La selección se mantiene activa tras borrar. Devuelve True si hubo cambios.
        """
        if self.active_selection is None or not self.current_qimage:
            return False
        try:
            img_np = np.array(qimage_to_pil(self.current_qimage).convert("RGBA"))
            if self.active_selection.shape != img_np.shape[:2]:
                self.clear_selection()
                return False
            new_np = sel_ops.delete_selected_pixels(img_np, self.active_selection)
            if np.array_equal(new_np[..., 3], img_np[..., 3]):
                return False  # Ya era transparente: no ensuciar el historial
            self.current_qimage = pil_to_qimage(Image.fromarray(new_np, "RGBA"))
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

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
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
            if self.last_draw_point:
                if self.brush_smoothness > 0:
                    factor = 1.0 - (self.brush_smoothness / 100.0) * 0.98
                    current_pt = self.last_draw_point + (img_pt - self.last_draw_point) * factor
                else:
                    current_pt = img_pt
                
                self._draw_brush_stroke(self.last_draw_point, current_pt)
                
                self.last_draw_point = current_pt
            else:
                self.last_draw_point = img_pt
            self.update()
            event.accept()
            return

        if self.brush_mode:
            self.update()

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
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.fileDropped.emit(path)
                event.acceptProposedAction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_pan_offset()
        self.update()

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
        if not self.current_qimage:
            return

        radius = max(1.0, self.brush_size / 2.0)
        pad = radius + 2
        min_x = int(min(start_pt.x(), end_pt.x()) - pad)
        min_y = int(min(start_pt.y(), end_pt.y()) - pad)
        max_x = int(max(start_pt.x(), end_pt.x()) + pad)
        max_y = int(max(start_pt.y(), end_pt.y()) + pad)
        
        w, h = max_x - min_x, max_y - min_y
        if w <= 0 or h <= 0: return

        # Si es duro (100%) y NO hay selección activa, dibujamos directo para máxima velocidad
        if self.brush_hardness >= 0.99 and self.active_selection is None:
            painter_img = QPainter(self.current_qimage)
            painter_img.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter_img.setOpacity(self.brush_opacity)

            if self.eraser_mode:
                painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
                base_color = QColor(0, 0, 0, 255)
                pen = QPen(base_color, radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter_img.setPen(pen)
                
                if (start_pt - end_pt).manhattanLength() < 0.5:
                    painter_img.drawPoint(start_pt)
                else:
                    painter_img.drawLine(start_pt, end_pt)
            else:
                if not self.original_qimage: 
                    painter_img.end()
                    return
                painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                original_brush = QBrush(self.original_qimage)
                pen = QPen(original_brush, radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter_img.setPen(pen)
                
                if (start_pt - end_pt).manhattanLength() < 0.5:
                    painter_img.drawPoint(start_pt)
                else:
                    painter_img.drawLine(start_pt, end_pt)
                    
            painter_img.end()
            return

        # Para pincel suave o cuando hay selección activa
        if self.brush_hardness < 0.25:
            steps = 6
        elif self.brush_hardness < 0.50:
            steps = 4
        elif self.brush_hardness < 0.80:
            steps = 3
        else:
            steps = 2
        
        # Máscara suave en un QImage temporal
        temp_img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        temp_img.fill(Qt.GlobalColor.transparent)
        
        temp_p = QPainter(temp_img)
        temp_p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        t_start = QPointF(start_pt.x() - min_x, start_pt.y() - min_y)
        t_end = QPointF(end_pt.x() - min_x, end_pt.y() - min_y)
        
        if self.brush_hardness >= 0.99:
            # Trazo duro dentro de temp_img
            temp_p.setOpacity(1.0)
            pen = QPen(QColor(0, 0, 0, 255), radius * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            temp_p.setPen(pen)
            if (start_pt - end_pt).manhattanLength() < 0.5:
                temp_p.drawPoint(t_start)
            else:
                temp_p.drawLine(t_start, t_end)
        else:
            # Trazo suave
            step_alpha = 1.0 - (0.05)**(1.0/steps)
            temp_p.setOpacity(step_alpha)
            temp_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            for i in range(steps):
                r = radius * (1.0 - (i / steps) * (1.0 - self.brush_hardness))
                pen = QPen(QColor(0, 0, 0, 255), r * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                temp_p.setPen(pen)
                if (start_pt - end_pt).manhattanLength() < 0.5:
                    temp_p.drawPoint(t_start)
                else:
                    temp_p.drawLine(t_start, t_end)
        temp_p.end()
        
        # Si hay selección, recortar (clip) la máscara temporal (temp_img)
        if getattr(self, 'active_selection', None) is not None:
            # Extraer ROI de active_selection
            sel_h, sel_w = self.active_selection.shape
            
            # Limites del ROI seguros
            roi_y1 = max(0, min(sel_h, min_y))
            roi_y2 = max(0, min(sel_h, max_y))
            roi_x1 = max(0, min(sel_w, min_x))
            roi_x2 = max(0, min(sel_w, max_x))
            
            # Si el ROI está fuera de los límites de la imagen, cancelamos
            if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
                return
                
            sel_patch = self.active_selection[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Ajustar la posición en temp_img si se recortó
            t_off_y = roi_y1 - min_y
            t_off_x = roi_x1 - min_x
            
            # Convertir temp_img a numpy
            try:
                temp_np = np.array(qimage_to_pil(temp_img).convert("RGBA"))
                patch_h, patch_w = sel_patch.shape
                # Multiplicar el alfa del trazo por la selección (0-255, respeta bordes suaves)
                region = temp_np[t_off_y:t_off_y + patch_h, t_off_x:t_off_x + patch_w, 3].astype(np.uint16)
                temp_np[t_off_y:t_off_y + patch_h, t_off_x:t_off_x + patch_w, 3] = (
                    region * sel_patch.astype(np.uint16) // 255
                ).astype(np.uint8)
                temp_img = pil_to_qimage(Image.fromarray(temp_np, "RGBA"))
            except Exception as e:
                print(f"Error clipping brush to selection: {e}")
        
        # Componer sobre la imagen real
        painter_img = QPainter(self.current_qimage)
        painter_img.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter_img.setOpacity(self.brush_opacity)
        
        if self.eraser_mode:
            painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            painter_img.drawImage(min_x, min_y, temp_img)
        else:
            if not self.original_qimage: 
                painter_img.end()
                return
            painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            orig_rect = QRect(int(min_x), int(min_y), w, h)
            patch = self.original_qimage.copy(orig_rect)
            
            patch_p = QPainter(patch)
            patch_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            patch_p.drawImage(0, 0, temp_img)
            patch_p.end()
            
            painter_img.drawImage(min_x, min_y, patch)
            
        painter_img.end()

    def _effective_selection_mode(self, modifiers) -> str:
        """
        Modo de selección según los modificadores, igual que Photoshop:
        Shift = añadir, Alt = restar, Shift+Alt = intersecar; si no, el modo de la barra.
        """
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        if shift and alt:
            return "intersect"
        if shift:
            return "add"
        if alt:
            return "subtract"
        return self.wand_selection_mode

    def _apply_magic_wand(self, pt: QPointF, mode: str = "new"):
        """Selecciona píxeles (por IA de MobileSAM o por tolerancia de color)."""
        if not self.current_qimage:
            return

        x, y = int(pt.x()), int(pt.y())
        if x < 0 or y < 0 or x >= self.current_qimage.width() or y >= self.current_qimage.height():
            return

        try:
            if getattr(self, "wand_ai_mode", False):
                # Varita Inteligente por IA (MobileSAM)
                from app.core.sam_manager import get_sam_manager
                sam_mgr = get_sam_manager()
                pil_img = qimage_to_pil(self.current_qimage)
                if not sam_mgr.has_embedding():
                    sam_mgr.prepare_image(pil_img)
                new_mask_bool = sam_mgr.predict_mask_at_point(
                    x, y, orig_img_size=(pil_img.width, pil_img.height)
                )
                if new_mask_bool is None:
                    return
                new_mask = (new_mask_bool.astype(np.uint8) * 255)
            else:
                img_np = np.array(qimage_to_pil(self.current_qimage).convert("RGBA"))
                region = sel_ops.magic_wand_region(
                    img_np, x, y,
                    tolerance=self.wand_tolerance,
                    contiguous=self.wand_contiguous,
                )
                if region is None:
                    return
                new_mask = sel_ops.region_to_mask(region, anti_alias=self.wand_anti_alias)

            current = self.active_selection
            if current is not None and current.shape != new_mask.shape:
                current = None
            self._set_selection(sel_ops.combine_selection(current, new_mask, mode))
        except Exception as e:
            print(f"Error en Varita: {e}")

    def _draw_marching_ants(self, painter: QPainter, img_rect: QRectF):
        """Dibuja el contorno animado de la selección (hormigas marchantes de Photoshop)."""
        if self.selection_path is None or self.selection_path.isEmpty():
            return
        painter.save()
        painter.setClipRect(img_rect.adjusted(-1, -1, 1, 1))
        painter.translate(img_rect.topLeft())
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        white_pen = QPen(QColor(255, 255, 255), 1)
        white_pen.setCosmetic(True)
        painter.setPen(white_pen)
        painter.drawPath(self.selection_path)

        black_pen = QPen(QColor(0, 0, 0), 1)
        black_pen.setCosmetic(True)
        black_pen.setDashPattern([4, 4])
        black_pen.setDashOffset(self._ants_offset)
        painter.setPen(black_pen)
        painter.drawPath(self.selection_path)
        painter.restore()

    # -------------------------------------------------------------
    # Pintado Principal
    # -------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # 1. Fondo general de la ventana (se deja que el QSS lo maneje)

        # Si no hay imagen cargada, mostrar pantalla de bienvenida / dropzone
        if not self.current_qimage:
            self._draw_empty_state(painter)
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

    def _draw_empty_state(self, painter: QPainter):
        """Muestra una tarjeta moderna indicando arrastrar y soltar o abrir imagen."""
        cx = self.width() / 2
        cy = self.height() / 2

        box_w = min(440, max(280, self.width() - 80))
        box_h = min(230, max(180, self.height() - 80))
        box_rect = QRectF(cx - box_w / 2, cy - box_h / 2, box_w, box_h)

        radius = 16.0

        # Sombra suave redondeada (concentric rounded rects, sin esquinas cuadradas residuales)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        shadow_layers = [
            (8, 4),
            (6, 7),
            (4, 10),
            (2, 14),
            (1, 20),
        ]
        for off, a in shadow_layers:
            s_rect = box_rect.adjusted(-off, -off + 2, off, off + 2)
            painter.setBrush(QColor(0, 0, 0, a))
            painter.drawRoundedRect(s_rect, radius + off, radius + off)
        painter.restore()

        is_dark = True
        main_win = self.window()
        if hasattr(main_win, 'is_dark_theme'):
            is_dark = main_win.is_dark_theme
            
        bg_color = QColor(45, 45, 45) if is_dark else QColor(255, 255, 255)
        border_color = QColor(61, 61, 61) if is_dark else QColor(226, 232, 240)
        dash_color = QColor(85, 85, 85) if is_dark else QColor(203, 213, 225)
        text_color = QColor("#FFFFFF") if is_dark else QColor("#1E293B")
        subtext_color = QColor("#AAAAAA") if is_dark else QColor("#64748B")
        subtext2_color = QColor("#777777") if is_dark else QColor("#94A3B8")

        # Fondo de la tarjeta oscura con esquinas redondeadas limpias y borde perimetral sutil
        painter.save()
        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(bg_color)
        painter.drawRoundedRect(box_rect, radius, radius)
        painter.restore()

        # Borde punteado interior concéntrico
        inner_margin = 7.0
        inner_radius = radius - inner_margin
        pen = QPen(dash_color, 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            box_rect.adjusted(inner_margin, inner_margin, -inner_margin, -inner_margin),
            inner_radius,
            inner_radius,
        )

        # Icono decorativo / badge
        icon_rect = QRectF(cx - 24, box_rect.y() + 28, 48, 48)
        painter.setBrush(QColor("#FFE4EC"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(icon_rect, 24, 24)

        # Icono SVG vectorial (Lucide image)
        render_icon(painter, "image", icon_rect.adjusted(12, 12, -12, -12), color="#FF4F79")

        # Texto principal
        painter.setPen(text_color)
        font = painter.font()
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QRectF(box_rect.x(), box_rect.y() + 86, box_rect.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            tr("canvas_drop_title"),
        )

        # Subtexto de ayuda
        font.setPixelSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(subtext_color)
        painter.drawText(
            QRectF(box_rect.x(), box_rect.y() + 118, box_rect.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            tr("canvas_drop_subtitle"),
        )

        painter.setFont(font)
        painter.setPen(subtext2_color)
        painter.drawText(
            QRectF(box_rect.x(), box_rect.y() + 155, box_rect.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            tr("canvas_drop_formats"),
        )



    def _draw_brush_cursor(self, painter: QPainter):
        """Dibuja el círculo del tamaño del pincel sobre el cursor del ratón con precisión en esquinas."""
        visual_radius = (self.brush_size * self.zoom_factor) / 2.0
        cx = self.mouse_cursor_pos.x()
        cy = self.mouse_cursor_pos.y()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        if self.eraser_mode:
            # Cursor oscuro para borrador
            painter.setBrush(QColor(0, 0, 0, 65))
            border_w = 1.0 if visual_radius < 3.0 else 1.5
            painter.setPen(QPen(QColor(0, 0, 0, 230), border_w))
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
            painter.setBrush(QColor(0, 0, 0, 240))
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

        # Círculo blanco con borde rosa sakura
        painter.setBrush(QColor("#FFFFFF"))
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

        # 3. Badges "ANTES" y "DESPUÉS"
        badge_y = img_rect.top() + 12
        if badge_y + 24 < img_rect.bottom():
            # Badge ANTES (Izquierda)
            badge_antes_rect = QRectF(img_rect.left() + 12, badge_y, 60, 22)
            painter.setBrush(QColor(28, 25, 23, 200))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_antes_rect, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            font_b = painter.font()
            font_b.setPixelSize(8)
            font_b.setBold(True)
            painter.setFont(font_b)
            painter.drawText(badge_antes_rect, Qt.AlignmentFlag.AlignCenter, "ANTES")

            # Badge DESPUÉS (Derecha)
            badge_desp_rect = QRectF(img_rect.right() - 76, badge_y, 64, 22)
            painter.setBrush(QColor(255, 79, 121, 220))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_desp_rect, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge_desp_rect, Qt.AlignmentFlag.AlignCenter, "DESPUÉS")

        painter.restore()

    # -------------------------------------------------------------
    # Scrollbars (UI y Lógica)
    # -------------------------------------------------------------
    def _scrollbar_style(self):
        return """
            QScrollBar:horizontal {
                border: none;
                background: rgba(231, 229, 228, 0.4);
                height: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(120, 113, 108, 0.6);
                min-width: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(120, 113, 108, 0.9);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                width: 0px;
            }
            
            QScrollBar:vertical {
                border: none;
                background: rgba(231, 229, 228, 0.4);
                width: 12px;
                margin: 0px 0px 0px 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(120, 113, 108, 0.6);
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(120, 113, 108, 0.9);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
                height: 0px;
            }
        """

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sb_thick = 12
        margin = 4
        self.v_scrollbar.setGeometry(self.width() - sb_thick - margin, margin, sb_thick, self.height() - (margin * 2) - sb_thick)
        self.h_scrollbar.setGeometry(margin, self.height() - sb_thick - margin, self.width() - (margin * 2) - sb_thick, sb_thick)
        self._update_scrollbars()

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
