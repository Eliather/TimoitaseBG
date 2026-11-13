"""
Motor de trazado de pincel y borrador con soporte de dureza, opacidad y recorte por selección.
"""
from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRect, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QImage

from app.core.image_utils import qimage_to_numpy, numpy_to_qimage


class BrushEngine:
    """Ejecuta trazos de pincel/borrador reutilizando un buffer temporal entre eventos."""

    def __init__(self):
        self._scratch_buf: Optional[QImage] = None

    def _get_scratch_buffer(self, w: int, h: int) -> QImage:
        """Retorna un buffer ARGB32_Premultiplied de al menos (w, h), evitando reasignaciones por cuadro."""
        if (
            self._scratch_buf is None
            or self._scratch_buf.width() < w
            or self._scratch_buf.height() < h
        ):
            alloc_w = max(w, 128)
            alloc_h = max(h, 128)
            self._scratch_buf = QImage(alloc_w, alloc_h, QImage.Format.Format_ARGB32_Premultiplied)
        return self._scratch_buf

    def clear_cache(self):
        self._scratch_buf = None

    def draw_stroke(
        self,
        target_qimage: QImage,
        original_qimage: Optional[QImage],
        start_pt: QPointF,
        end_pt: QPointF,
        brush_size: int,
        brush_opacity: float,
        brush_hardness: float,
        eraser_mode: bool,
        active_selection: Optional[np.ndarray] = None,
    ) -> bool:
        """Dibuja una línea continua o punto con extremos redondeados sobre target_qimage."""
        if target_qimage is None or target_qimage.isNull():
            return False

        radius = max(1.0, brush_size / 2.0)
        pad = radius + 2
        min_x = int(min(start_pt.x(), end_pt.x()) - pad)
        min_y = int(min(start_pt.y(), end_pt.y()) - pad)
        max_x = int(max(start_pt.x(), end_pt.x()) + pad)
        max_y = int(max(start_pt.y(), end_pt.y()) + pad)

        w, h = max_x - min_x, max_y - min_y
        if w <= 0 or h <= 0:
            return False

        # Ruta rápida: pincel duro (100%) sin selección activa
        if brush_hardness >= 0.99 and active_selection is None:
            return self._draw_hard_direct(
                target_qimage, original_qimage, start_pt, end_pt, radius, brush_opacity, eraser_mode
            )

        # Ruta suavizada o recortada por selección
        mask_img = self._render_stroke_mask(start_pt, end_pt, min_x, min_y, w, h, radius, brush_hardness)
        mask_rect = QRect(0, 0, w, h)

        if active_selection is not None:
            clipped = self._clip_mask_to_selection(mask_img, active_selection, min_x, min_y, max_x, max_y, w, h)
            if clipped is None:
                return False
            mask_img = clipped
            mask_rect = QRect(0, 0, w, h)

        painter_img = QPainter(target_qimage)
        painter_img.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter_img.setOpacity(brush_opacity)

        if eraser_mode:
            painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            painter_img.drawImage(QRect(min_x, min_y, w, h), mask_img, mask_rect)
        else:
            if not original_qimage:
                painter_img.end()
                return False
            painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            patch = original_qimage.copy(QRect(int(min_x), int(min_y), w, h))
            if not patch.hasAlphaChannel():
                patch = patch.convertToFormat(QImage.Format.Format_RGBA8888)
            patch_p = QPainter(patch)
            patch_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            patch_p.drawImage(QRect(0, 0, w, h), mask_img, mask_rect)
            patch_p.end()
            painter_img.drawImage(min_x, min_y, patch)

        painter_img.end()
        return True

    def _draw_hard_direct(
        self,
        target_qimage: QImage,
        original_qimage: Optional[QImage],
        start_pt: QPointF,
        end_pt: QPointF,
        radius: float,
        brush_opacity: float,
        eraser_mode: bool,
    ) -> bool:
        painter_img = QPainter(target_qimage)
        painter_img.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter_img.setOpacity(brush_opacity)

        if eraser_mode:
            painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            pen = QPen(
                QColor(0, 0, 0, 255),
                radius * 2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        else:
            if not original_qimage:
                painter_img.end()
                return False
            painter_img.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            pen = QPen(
                QBrush(original_qimage),
                radius * 2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )

        painter_img.setPen(pen)
        if (start_pt - end_pt).manhattanLength() < 0.5:
            painter_img.drawPoint(start_pt)
        else:
            painter_img.drawLine(start_pt, end_pt)

        painter_img.end()
        return True

    def _render_stroke_mask(
        self,
        start_pt: QPointF,
        end_pt: QPointF,
        min_x: int,
        min_y: int,
        w: int,
        h: int,
        radius: float,
        brush_hardness: float,
    ) -> QImage:
        temp_img = self._get_scratch_buffer(w, h)
        temp_p = QPainter(temp_img)
        temp_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        temp_p.fillRect(0, 0, w, h, Qt.GlobalColor.transparent)
        temp_p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        temp_p.setClipRect(0, 0, w, h)

        t_start = QPointF(start_pt.x() - min_x, start_pt.y() - min_y)
        t_end = QPointF(end_pt.x() - min_x, end_pt.y() - min_y)
        is_dot = (start_pt - end_pt).manhattanLength() < 0.5

        if brush_hardness >= 0.99:
            temp_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            temp_p.setOpacity(1.0)
            pen = QPen(
                QColor(0, 0, 0, 255),
                radius * 2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            temp_p.setPen(pen)
            if is_dot:
                temp_p.drawPoint(t_start)
            else:
                temp_p.drawLine(t_start, t_end)
        else:
            if brush_hardness < 0.25:
                steps = 6
            elif brush_hardness < 0.50:
                steps = 4
            elif brush_hardness < 0.80:
                steps = 3
            else:
                steps = 2

            step_alpha = 1.0 - (0.05) ** (1.0 / steps)
            temp_p.setOpacity(step_alpha)
            temp_p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            for i in range(steps):
                r = radius * (1.0 - (i / steps) * (1.0 - brush_hardness))
                pen = QPen(
                    QColor(0, 0, 0, 255),
                    r * 2,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                temp_p.setPen(pen)
                if is_dot:
                    temp_p.drawPoint(t_start)
                else:
                    temp_p.drawLine(t_start, t_end)

        temp_p.end()
        return temp_img

    def _clip_mask_to_selection(
        self,
        temp_img: QImage,
        active_selection: np.ndarray,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        w: int,
        h: int,
    ) -> Optional[QImage]:
        sel_h, sel_w = active_selection.shape
        roi_y1 = max(0, min(sel_h, min_y))
        roi_y2 = max(0, min(sel_h, max_y))
        roi_x1 = max(0, min(sel_w, min_x))
        roi_x2 = max(0, min(sel_w, max_x))

        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            return None

        sel_patch = active_selection[roi_y1:roi_y2, roi_x1:roi_x2]
        t_off_y = roi_y1 - min_y
        t_off_x = roi_x1 - min_x

        try:
            temp_np = qimage_to_numpy(temp_img.copy(0, 0, w, h))
            patch_h, patch_w = sel_patch.shape
            clipped_alpha = np.zeros((h, w), dtype=np.uint8)
            region = temp_np[t_off_y : t_off_y + patch_h, t_off_x : t_off_x + patch_w, 3].astype(np.uint16)
            clipped_alpha[t_off_y : t_off_y + patch_h, t_off_x : t_off_x + patch_w] = (
                region * sel_patch.astype(np.uint16) // 255
            ).astype(np.uint8)
            temp_np[:, :, 3] = clipped_alpha
            return numpy_to_qimage(temp_np)
        except Exception as e:
            print(f"Error clipping brush to selection: {e}")
            return temp_img
