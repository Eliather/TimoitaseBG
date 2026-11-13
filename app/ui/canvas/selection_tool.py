"""
Gestor modular de selección, varita mágica y hormigas marchantes para el Canvas.
"""
from typing import Callable, Optional
import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from app.core import selection as sel_ops
from app.core.image_utils import numpy_to_qimage, pil_to_qimage, qimage_to_numpy, qimage_to_pil


class SelectionTool:
    """Administra la máscara de selección activa, contorno vectorial y operaciones de varita."""

    def __init__(self):
        self.active_mask: Optional[np.ndarray] = None
        self.overlay_qimage: Optional[QImage] = None
        self.contour_path: Optional[QPainterPath] = None
        self.ants_offset: float = 0.0
        self.pending_wand_click: Optional[tuple] = None

        self.wand_mode: bool = False
        self.wand_ai_mode: bool = False
        self.wand_tolerance: int = 32
        self.wand_selection_mode: str = "new"
        self.wand_anti_alias: bool = True
        self.wand_contiguous: bool = True

    def has_selection(self) -> bool:
        return self.active_mask is not None

    def set_mask(self, mask: Optional[np.ndarray], img_w: int, img_h: int) -> tuple[int, int]:
        """Actualiza máscara, overlay azul y contorno vectorial. Retorna (seleccionados, total)."""
        if mask is not None and sel_ops.selection_pixel_count(mask) == 0:
            mask = None

        self.active_mask = mask
        total = img_w * img_h if img_w > 0 and img_h > 0 else 0

        if mask is None:
            self.overlay_qimage = None
            self.contour_path = None
            return 0, total

        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[..., 1] = 120
        overlay[..., 2] = 255
        overlay[..., 3] = (mask.astype(np.uint16) * 45 // 255).astype(np.uint8)
        self.overlay_qimage = numpy_to_qimage(overlay)

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
        self.contour_path = path

        return sel_ops.selection_pixel_count(mask), total

    def advance_ants(self) -> Optional[QRectF]:
        """Avanza el desplazamiento del patrón de hormigas y retorna el bounding rect en coords imagen."""
        self.ants_offset = (self.ants_offset + 1.0) % 8.0
        if self.contour_path is not None and not self.contour_path.isEmpty():
            return self.contour_path.boundingRect()
        return None

    def effective_mode(self, modifiers) -> str:
        """Determina el modo efectivo de selección combinando Shift/Alt como Photoshop."""
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        if shift and alt:
            return "intersect"
        if shift:
            return "add"
        if alt:
            return "subtract"
        return self.wand_selection_mode

    def compute_wand_mask(
        self,
        qimage: Optional[QImage],
        pt: QPointF,
        mode: str = "new",
        async_sam_trigger: Optional[Callable[[], None]] = None,
    ) -> Optional[np.ndarray]:
        """Calcula la nueva máscara combinada para el clic de varita o encola MobileSAM."""
        if qimage is None or qimage.isNull():
            return None

        x, y = int(pt.x()), int(pt.y())
        w, h = qimage.width(), qimage.height()
        if x < 0 or y < 0 or x >= w or y >= h:
            return None

        if self.wand_ai_mode:
            from app.core.sam_manager import get_sam_manager
            sam_mgr = get_sam_manager()
            if not sam_mgr.has_embedding():
                if async_sam_trigger is not None:
                    self.pending_wand_click = (pt, mode)
                    async_sam_trigger()
                    return None
                sam_mgr.prepare_image(qimage_to_pil(qimage))
            new_mask_bool = sam_mgr.predict_mask_at_point(x, y, orig_img_size=(w, h))
            if new_mask_bool is None:
                return None
            new_mask = new_mask_bool.astype(np.uint8) * 255
        else:
            img_np = qimage_to_numpy(qimage)
            region = sel_ops.magic_wand_region(
                img_np, x, y, tolerance=self.wand_tolerance, contiguous=self.wand_contiguous
            )
            if region is None:
                return None
            new_mask = sel_ops.region_to_mask(region, anti_alias=self.wand_anti_alias)

        current = self.active_mask
        if current is not None and current.shape != new_mask.shape:
            current = None
        return sel_ops.combine_selection(current, new_mask, mode)

    def delete_selected_pixels(self, qimage: Optional[QImage]) -> tuple[Optional[QImage], bool]:
        """
        Vuelve transparentes los píxeles dentro de la selección activa.
        Retorna (nuevo_qimage, requiere_limpiar_seleccion).
        """
        if self.active_mask is None or qimage is None or qimage.isNull():
            return None, False

        img_np = np.array(qimage_to_pil(qimage).convert("RGBA"))
        if self.active_mask.shape != img_np.shape[:2]:
            return None, True

        new_np = sel_ops.delete_selected_pixels(img_np, self.active_mask)
        if np.array_equal(new_np[..., 3], img_np[..., 3]):
            return None, False

        return pil_to_qimage(Image.fromarray(new_np, "RGBA")), False

    def draw_overlay(self, painter: QPainter, img_rect: QRectF) -> None:
        if self.overlay_qimage is not None:
            painter.drawImage(img_rect, self.overlay_qimage)

    def draw_marching_ants(self, painter: QPainter, img_rect: QRectF, zoom_factor: float) -> None:
        """Dibuja el contorno animado de la selección."""
        if self.contour_path is None or self.contour_path.isEmpty():
            return
        painter.save()
        painter.setClipRect(img_rect.adjusted(-1, -1, 1, 1))
        painter.translate(img_rect.topLeft())
        painter.scale(zoom_factor, zoom_factor)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        white_pen = QPen(QColor(255, 255, 255), 1)
        white_pen.setCosmetic(True)
        painter.setPen(white_pen)
        painter.drawPath(self.contour_path)

        black_pen = QPen(QColor(0, 0, 0), 1)
        black_pen.setCosmetic(True)
        black_pen.setDashPattern([4, 4])
        black_pen.setDashOffset(self.ants_offset)
        painter.setPen(black_pen)
        painter.drawPath(self.contour_path)
        painter.restore()
