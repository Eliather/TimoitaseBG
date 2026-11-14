"""
Subsistema modular de recorte manual interactivo (Crop) con manijas y relación de aspecto.
"""
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen


class CropTool:
    """Gestiona el rectángulo de recorte en coordenadas nativas de imagen."""

    def __init__(self):
        self.active: bool = False
        self.aspect_ratio: Optional[float] = None
        self.rect: Optional[QRectF] = None
        self._dragging_handle: Optional[str] = None
        self._drag_start_img: QPointF = QPointF()
        self._drag_start_rect: Optional[QRectF] = None

    def activate(
        self,
        img_w: int,
        img_h: int,
        aspect_ratio: Optional[float] = None,
    ) -> None:
        self.active = True
        self.aspect_ratio = aspect_ratio
        if img_w <= 0 or img_h <= 0:
            self.rect = None
            return
        self.rect = self._fit_centered_rect(img_w, img_h, aspect_ratio)

    def deactivate(self) -> None:
        self.active = False
        self.rect = None
        self._dragging_handle = None
        self._drag_start_rect = None

    def set_aspect_ratio(
        self, aspect_ratio: Optional[float], img_w: int, img_h: int
    ) -> None:
        self.aspect_ratio = aspect_ratio
        if not self.active or img_w <= 0 or img_h <= 0:
            return
        if self.rect is None:
            self.rect = self._fit_centered_rect(img_w, img_h, aspect_ratio)
            return
        if aspect_ratio is None:
            return
        cx = self.rect.center().x()
        cy = self.rect.center().y()
        w = max(8.0, self.rect.width())
        h = w / aspect_ratio
        if h > img_h:
            h = float(img_h)
            w = h * aspect_ratio
        if w > img_w:
            w = float(img_w)
            h = w / aspect_ratio
        x0 = max(0.0, min(img_w - w, cx - w / 2.0))
        y0 = max(0.0, min(img_h - h, cy - h / 2.0))
        self.rect = QRectF(x0, y0, w, h)

    @staticmethod
    def _fit_centered_rect(
        img_w: int, img_h: int, aspect_ratio: Optional[float]
    ) -> QRectF:
        margin_x = img_w * 0.08
        margin_y = img_h * 0.08
        avail_w = max(8.0, img_w - margin_x * 2.0)
        avail_h = max(8.0, img_h - margin_y * 2.0)
        if aspect_ratio and aspect_ratio > 0:
            if avail_w / avail_h > aspect_ratio:
                h = avail_h
                w = h * aspect_ratio
            else:
                w = avail_w
                h = w / aspect_ratio
        else:
            w, h = avail_w, avail_h
        x0 = (img_w - w) / 2.0
        y0 = (img_h - h) / 2.0
        return QRectF(x0, y0, w, h)

    def _hit_test(
        self,
        widget_pt: QPointF,
        image_to_widget_fn: Callable[[QPointF], QPointF],
    ) -> Optional[str]:
        if self.rect is None:
            return None
        tl = image_to_widget_fn(self.rect.topLeft())
        tr = image_to_widget_fn(self.rect.topRight())
        bl = image_to_widget_fn(self.rect.bottomLeft())
        br = image_to_widget_fn(self.rect.bottomRight())
        tol = 12.0
        for name, corner in (("tl", tl), ("tr", tr), ("bl", bl), ("br", br)):
            if abs(widget_pt.x() - corner.x()) <= tol and abs(widget_pt.y() - corner.y()) <= tol:
                return name
        widget_rect = QRectF(tl, br).normalized()
        if widget_rect.contains(widget_pt):
            return "move"
        return "new"

    def handle_mouse_press(
        self,
        img_pt: QPointF,
        widget_pt: QPointF,
        image_to_widget_fn: Callable[[QPointF], QPointF],
        img_w: int,
        img_h: int,
    ) -> bool:
        if not self.active or img_w <= 0 or img_h <= 0:
            return False
        hit = self._hit_test(widget_pt, image_to_widget_fn)
        self._dragging_handle = hit
        self._drag_start_img = QPointF(
            max(0.0, min(float(img_w), img_pt.x())),
            max(0.0, min(float(img_h), img_pt.y())),
        )
        if hit == "new":
            self.rect = QRectF(self._drag_start_img, self._drag_start_img)
        self._drag_start_rect = QRectF(self.rect) if self.rect is not None else None
        return True

    def handle_mouse_move(
        self,
        img_pt: QPointF,
        img_w: int,
        img_h: int,
    ) -> bool:
        if not self.active or not self._dragging_handle or self._drag_start_rect is None:
            return False

        cx = max(0.0, min(float(img_w), img_pt.x()))
        cy = max(0.0, min(float(img_h), img_pt.y()))
        r0 = self._drag_start_rect
        min_sz = 8.0

        if self._dragging_handle == "move":
            dx = cx - self._drag_start_img.x()
            dy = cy - self._drag_start_img.y()
            nx = max(0.0, min(float(img_w) - r0.width(), r0.left() + dx))
            ny = max(0.0, min(float(img_h) - r0.height(), r0.top() + dy))
            self.rect = QRectF(nx, ny, r0.width(), r0.height())
            return True

        if self._dragging_handle == "new":
            x0, y0 = self._drag_start_img.x(), self._drag_start_img.y()
            dx = cx - x0
            dy = cy - y0
            if self.aspect_ratio and self.aspect_ratio > 0:
                w = max(min_sz, abs(dx))
                h = w / self.aspect_ratio
                sx = 1.0 if dx >= 0 else -1.0
                sy = 1.0 if dy >= 0 else -1.0
                x1 = max(0.0, min(float(img_w), x0 + sx * w))
                y1 = max(0.0, min(float(img_h), y0 + sy * h))
                self.rect = QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized()
            else:
                self.rect = QRectF(QPointF(x0, y0), QPointF(cx, cy)).normalized()
            return True

        left, top, right, bottom = r0.left(), r0.top(), r0.right(), r0.bottom()
        if self._dragging_handle == "tl":
            left = min(cx, right - min_sz)
            top = min(cy, bottom - min_sz)
        elif self._dragging_handle == "tr":
            right = max(cx, left + min_sz)
            top = min(cy, bottom - min_sz)
        elif self._dragging_handle == "bl":
            left = min(cx, right - min_sz)
            bottom = max(cy, top + min_sz)
        elif self._dragging_handle == "br":
            right = max(cx, left + min_sz)
            bottom = max(cy, top + min_sz)

        if self.aspect_ratio and self.aspect_ratio > 0:
            w = max(min_sz, right - left)
            h = w / self.aspect_ratio
            if self._dragging_handle in ("tl", "tr"):
                top = max(0.0, bottom - h)
                h = bottom - top
                w = h * self.aspect_ratio
                if self._dragging_handle == "tl":
                    left = max(0.0, right - w)
                else:
                    right = min(float(img_w), left + w)
            else:
                bottom = min(float(img_h), top + h)
                h = bottom - top
                w = h * self.aspect_ratio
                if self._dragging_handle == "bl":
                    left = max(0.0, right - w)
                else:
                    right = min(float(img_w), left + w)

        self.rect = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        return True

    def handle_mouse_release(self) -> bool:
        if not self._dragging_handle:
            return False
        self._dragging_handle = None
        self._drag_start_rect = None
        return True

    def get_crop_box(self, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
        if not self.active or self.rect is None:
            return None
        r = self.rect.normalized()
        x0 = max(0, min(img_w - 1, int(round(r.left()))))
        y0 = max(0, min(img_h - 1, int(round(r.top()))))
        x1 = max(x0 + 1, min(img_w, int(round(r.right()))))
        y1 = max(y0 + 1, min(img_h, int(round(r.bottom()))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return (x0, y0, x1, y1)

    def draw_overlay(
        self,
        painter: QPainter,
        image_rect_in_widget: QRectF,
        image_to_widget_fn: Callable[[QPointF], QPointF],
    ) -> None:
        if not self.active or self.rect is None:
            return
        tl = image_to_widget_fn(self.rect.topLeft())
        br = image_to_widget_fn(self.rect.bottomRight())
        crop_wrect = QRectF(tl, br).normalized()

        painter.save()
        # Oscurecer el exterior del área de recorte
        outer_path = QPainterPath()
        outer_path.addRect(image_rect_in_widget)
        inner_path = QPainterPath()
        inner_path.addRect(crop_wrect)
        scrim_path = outer_path.subtracted(inner_path)
        painter.fillPath(scrim_path, QColor(0, 0, 0, 140))

        # Guías regla de los tercios
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1.0, Qt.PenStyle.DashLine))
        w = crop_wrect.width()
        h = crop_wrect.height()
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            x = crop_wrect.left() + w * frac
            y = crop_wrect.top() + h * frac
            painter.drawLine(QPointF(x, crop_wrect.top()), QPointF(x, crop_wrect.bottom()))
            painter.drawLine(QPointF(crop_wrect.left(), y), QPointF(crop_wrect.right(), y))

        # Borde y manijas de las 4 esquinas
        painter.setPen(QPen(QColor("#38BDF8"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(crop_wrect)

        painter.setBrush(QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#0F172A"), 1.5))
        hs = 8.0
        for corner in (
            crop_wrect.topLeft(),
            crop_wrect.topRight(),
            crop_wrect.bottomLeft(),
            crop_wrect.bottomRight(),
        ):
            painter.drawRect(QRectF(corner.x() - hs / 2, corner.y() - hs / 2, hs, hs))
        painter.restore()
