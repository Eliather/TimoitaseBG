"""
Controles reutilizables del sidebar: contenedor autoajustable, toggle GPU/CPU y cabeceras.
"""
from typing import List, Optional, Tuple
from PySide6.QtCore import Qt, Signal, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QBrush
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QAbstractButton,
)

from app.ui.icons import get_pixmap


def create_section_header(title: str, tooltip: str) -> Tuple[QHBoxLayout, QLabel, QLabel]:
    """Crea una cabecera compacta con título y un ícono 'ⓘ' con tooltip explicativo."""
    header = QHBoxLayout()
    header.setSpacing(6)
    lbl = QLabel(title)
    lbl.setObjectName("fieldLabel")
    info = QLabel("ⓘ")
    info.setObjectName("toolInfoIcon")
    info.setToolTip(tooltip)
    info.setCursor(Qt.CursorShape.PointingHandCursor)
    header.addWidget(lbl)
    header.addWidget(info)
    header.addStretch()
    return header, lbl, info


class AutoResizingStackedWidget(QWidget):
    """
    Contenedor dinámico de herramientas que adapta su tamaño exclusivamente
    a la pestaña activa, ocultando las inactivas para que QScrollArea mida con
    total fidelidad la altura y oculte la scrollbar cuando no sea necesaria.
    """
    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._pages: List[QWidget] = []
        self._current_index: int = 0

    def addWidget(self, widget: QWidget) -> int:
        idx = len(self._pages)
        self._pages.append(widget)
        self._layout.addWidget(widget)
        if idx > 0:
            widget.hide()
        return idx

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int) -> Optional[QWidget]:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def currentWidget(self) -> Optional[QWidget]:
        return self.widget(self._current_index)

    def currentIndex(self) -> int:
        return self._current_index

    def setCurrentIndex(self, index: int):
        if 0 <= index < len(self._pages):
            self._current_index = index
            for i, page in enumerate(self._pages):
                if i == index:
                    page.show()
                else:
                    page.hide()
            self.updateGeometry()
            parent = self.parentWidget()
            if parent:
                parent.updateGeometry()
                parent.adjustSize()
            self.currentChanged.emit(index)


class ToggleSwitch(QAbstractButton):
    """
    Interruptor deslizante tipo toggle moderno (estilo iOS / macOS / Fluent) con soporte de tema oscuro/claro.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(36, 20)
        self._is_dark: bool = True
        self._thumb_pos = 1.0 if self.isChecked() else 0.0
        self._anim = QPropertyAnimation(self, b"thumb_pos", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.toggled.connect(self._on_toggled)

    def set_dark_theme(self, is_dark: bool):
        self._is_dark = is_dark
        self.update()

    def _get_thumb_pos(self) -> float:
        return self._thumb_pos

    def _set_thumb_pos(self, pos: float):
        self._thumb_pos = pos
        self.update()

    thumb_pos = Property(float, _get_thumb_pos, _set_thumb_pos)

    def _on_toggled(self, checked: bool):
        self._anim.stop()
        self._anim.setStartValue(self._thumb_pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        self._thumb_pos = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        radius = h / 2.0
        thumb_diameter = h - 4.0

        t = self._thumb_pos
        if not self.isEnabled():
            track_color = QColor("#1E293B") if self._is_dark else QColor("#E2E8F0")
        else:
            r0, g0, b0 = (51, 65, 85) if self._is_dark else (203, 213, 225)
            r = int(r0 * (1 - t) + 16 * t)
            g = int(g0 * (1 - t) + 185 * t)
            b = int(b0 * (1 - t) + 129 * t)
            track_color = QColor(r, g, b)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(track_color))
        p.drawRoundedRect(0, 0, w, h, radius, radius)

        x_min = 2.0
        x_max = w - thumb_diameter - 2.0
        thumb_x = x_min + t * (x_max - x_min)
        thumb_y = 2.0

        p.setBrush(QBrush(QColor(0, 0, 0, 45 if self._is_dark else 30)))
        p.drawEllipse(int(thumb_x), int(thumb_y + 1), int(thumb_diameter), int(thumb_diameter))

        if not self.isEnabled():
            thumb_fill = QColor("#64748B") if self._is_dark else QColor("#94A3B8")
        else:
            thumb_fill = QColor("#F8FAFC") if self._is_dark else QColor("#FFFFFF")
        p.setBrush(QBrush(thumb_fill))
        p.drawEllipse(int(thumb_x), int(thumb_y), int(thumb_diameter), int(thumb_diameter))
        p.end()


class GpuToggle(QWidget):
    """
    Control de un solo toggle para conmutar entre Aceleración GPU y CPU.
    Integra una pastilla interactiva compacta con etiqueta y un interruptor deslizante.
    """
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gpuToggleWidget")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)
        self._is_dark: bool = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(6)

        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(14, 14)
        self.lbl_icon.setStyleSheet("background: transparent; border: none;")

        self.lbl_text = QLabel("GPU")
        self.lbl_text.setObjectName("gpuToggleLabel")
        self.lbl_text.setCursor(Qt.CursorShape.PointingHandCursor)

        self.switch = ToggleSwitch(self)
        self.switch.toggled.connect(self._on_switch_toggled)

        layout.addWidget(self.lbl_icon)
        layout.addWidget(self.lbl_text)
        layout.addWidget(self.switch)

        self._is_gpu = True
        self._is_blocked = False
        self._update_presentation()

    def set_dark_theme(self, is_dark: bool):
        self._is_dark = is_dark
        self.switch.set_dark_theme(is_dark)
        self._update_presentation()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled() and not self._is_blocked:
            self.switch.toggle()
        super().mousePressEvent(event)

    def _on_switch_toggled(self, checked: bool):
        self._is_gpu = checked
        self._update_presentation()
        self.toggled.emit(checked)

    def is_gpu_active(self) -> bool:
        return self._is_gpu and self.switch.isChecked()

    def set_state(self, is_gpu: bool, is_blocked: bool = False):
        self._is_blocked = is_blocked
        self._is_gpu = is_gpu
        self.switch.blockSignals(True)
        if is_blocked:
            self.switch.setChecked(False)
            self.switch.setEnabled(False)
            self.setEnabled(False)
        else:
            self.switch.setEnabled(True)
            self.setEnabled(True)
            self.switch.setChecked(is_gpu)
        self.switch.blockSignals(False)
        self._update_presentation()

    def _apply_pill_style(self, fg: str, bg: str, bd: str, weight: int):
        self.lbl_text.setStyleSheet(
            f"color: {fg}; font-weight: {weight}; font-size: 11px; border: none; background: transparent;"
        )
        self.setStyleSheet(
            f"#gpuToggleWidget {{ background-color: {bg}; border: 1px solid {bd}; border-radius: 14px; padding: 0 4px; }}"
        )

    def _update_presentation(self):
        if self._is_blocked:
            fg = "#F87171" if self._is_dark else "#DC2626"
            bg = "#450A0A" if self._is_dark else "#FEF2F2"
            bd = "#DC2626" if self._is_dark else "#FECACA"
            self.lbl_icon.setPixmap(get_pixmap("alert-triangle", size=14, color=fg))
            self.lbl_text.setText("CPU (Seguro)")
            self._apply_pill_style(fg, bg, bd, 700)
            self.setToolTip("GPU bloqueada por fallo previo. La app funciona en CPU de forma segura.")
        elif not self.switch.isEnabled():
            fg = "#94A3B8"
            bg = "#0F172A" if self._is_dark else "#F1F5F9"
            bd = "#334155" if self._is_dark else "#E2E8F0"
            self.lbl_icon.setPixmap(get_pixmap("cpu", size=14, color=fg))
            self.lbl_text.setText("CPU")
            self._apply_pill_style(fg, bg, bd, 600)
            self.setToolTip("GPU no disponible en este equipo. Modo CPU activo.")
        elif self.switch.isChecked():
            fg = "#34D399" if self._is_dark else "#059669"
            bg = "#064E3B" if self._is_dark else "#ECFDF5"
            bd = "#059669" if self._is_dark else "#A7F3D0"
            self.lbl_icon.setPixmap(get_pixmap("zap", size=14, color=fg))
            self.lbl_text.setText("⚡ GPU Activa")
            self._apply_pill_style(fg, bg, bd, 700)
            try:
                from app.config import get_gpu_name, detect_device
                detected = detect_device()
                name = get_gpu_name() or (detected if "GPU" in detected else "GPU")
                clean = (
                    name.replace("NVIDIA ", "")
                    .replace("AMD ", "")
                    .replace(" Laptop GPU", "")
                    .replace("Advanced Micro Devices, Inc. ", "")
                )
                self.setToolTip(f"Aceleración GPU activa ({clean}). Clic para pasar a CPU.")
            except Exception:
                self.setToolTip("Aceleración GPU activa. Clic para pasar a CPU.")
        else:
            fg = "#CBD5E1" if self._is_dark else "#64748B"
            bg = "#0F172A" if self._is_dark else "#F1F5F9"
            bd = "#334155" if self._is_dark else "#CBD5E1"
            self.lbl_icon.setPixmap(get_pixmap("cpu", size=14, color=fg))
            self.lbl_text.setText("CPU Activo")
            self._apply_pill_style(fg, bg, bd, 600)
            self.setToolTip("Modo CPU activo (0 MB VRAM). Clic para activar GPU.")


class _GpuToggleAdapter:
    """Adaptador de compatibilidad para código o pruebas previas."""

    def __init__(self, toggle: GpuToggle, is_gpu_target: bool):
        self._toggle = toggle
        self._is_gpu_target = is_gpu_target

    def click(self):
        if self._is_gpu_target:
            if not self._toggle.switch.isChecked():
                self._toggle.switch.setChecked(True)
        else:
            if self._toggle.switch.isChecked():
                self._toggle.switch.setChecked(False)

    def isChecked(self) -> bool:
        if self._is_gpu_target:
            return self._toggle.switch.isChecked()
        return not self._toggle.switch.isChecked()

    def isEnabled(self) -> bool:
        return self._toggle.switch.isEnabled()

    def setChecked(self, checked: bool):
        if self._is_gpu_target:
            self._toggle.switch.setChecked(checked)
        else:
            self._toggle.switch.setChecked(not checked)

    def setToolTip(self, text: str):
        pass
