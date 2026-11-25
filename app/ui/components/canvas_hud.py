"""
Controles flotantes sobre el lienzo (CanvasHudBar y ToastBanner) para TimoitaseBG.
"""
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from app.i18n import tr
from app.ui.icons import get_icon


class CanvasHudBar(QFrame):
    """Cápsula flotante en la parte inferior central del lienzo con zoom y comparación."""

    zoomOutRequested = Signal()
    zoomInRequested = Signal()
    fitRequested = Signal()
    zoom100Requested = Signal()
    compareToggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CanvasHudBar")
        self.setFixedHeight(38)
        self._is_dark: bool = True
        self._setup_ui()
        self.apply_theme(is_dark=True)

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        self.btn_zoom_out = self._make_hud_icon_btn("tooltip_zoom_out", self.zoomOutRequested.emit)
        layout.addWidget(self.btn_zoom_out)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setObjectName("HudZoomLabel")
        self.lbl_zoom.setFixedWidth(46)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_zoom)

        self.btn_zoom_in = self._make_hud_icon_btn("tooltip_zoom_in", self.zoomInRequested.emit)
        layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton(tr("btn_zoom_fit"))
        self.btn_fit.setObjectName("HudBtn")
        self.btn_fit.setFixedHeight(28)
        self.btn_fit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fit.setToolTip(tr("tooltip_zoom_fit"))
        self.btn_fit.clicked.connect(self.fitRequested.emit)
        layout.addWidget(self.btn_fit)

        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setObjectName("HudBtn")
        self.btn_zoom_100.setFixedHeight(28)
        self.btn_zoom_100.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_100.setToolTip(tr("tooltip_zoom_100"))
        self.btn_zoom_100.clicked.connect(self.zoom100Requested.emit)
        layout.addWidget(self.btn_zoom_100)

        self.separator = QFrame(self)
        self.separator.setObjectName("HudSeparator")
        self.separator.setFixedSize(1, 18)
        self.separator.setStyleSheet("background-color: rgba(155, 163, 174, 0.25); border: none;")
        layout.addWidget(self.separator)

        self.btn_compare = QPushButton(tr("btn_compare"))
        self.btn_compare.setObjectName("HudCompareBtn")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setFixedHeight(28)
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.setToolTip(tr("tooltip_compare"))
        self.btn_compare.clicked.connect(self.compareToggled.emit)
        self.btn_compare.toggled.connect(self._on_compare_toggled)
        layout.addWidget(self.btn_compare)

        self._update_compare_min_width()
        self.adjustSize()

    def _update_compare_min_width(self) -> None:
        """Reserva ancho suficiente para el texto en negrita (:checked) sin recortes."""
        bold_font = self.btn_compare.font()
        bold_font.setWeight(bold_font.Weight.DemiBold)
        text_w = self.btn_compare.fontMetrics().horizontalAdvance(self.btn_compare.text())
        from PySide6.QtGui import QFontMetrics
        bold_w = QFontMetrics(bold_font).horizontalAdvance(self.btn_compare.text())
        # Icono (15) + espacio (8) + padding horizontal (24) + holgura (8)
        self.btn_compare.setMinimumWidth(max(106, max(text_w, bold_w) + 56))

    def _on_compare_toggled(self, _checked: bool) -> None:
        self._update_compare_min_width()
        self.adjustSize()

    def _make_hud_icon_btn(self, tip_key: str, slot) -> QPushButton:
        btn = QPushButton("")
        btn.setObjectName("HudBtn")
        btn.setFixedSize(28, 28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tr(tip_key))
        btn.clicked.connect(slot)
        return btn

    def set_zoom(self, zoom: float) -> None:
        self.lbl_zoom.setText(f"{int(zoom * 100)}%")

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        fg_color = "#F2F4F7" if is_dark else "#14171A"
        accent_color = "#2E7CF6" if is_dark else "#1D63DC"
        dis_color = "#5C6470" if is_dark else "#9BA3AE"

        for btn, icon_name in ((self.btn_zoom_out, "zoom-out"), (self.btn_zoom_in, "zoom-in")):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=15))
            btn.setIconSize(QSize(15, 15))
            btn.setText("")

        for btn, icon_name in ((self.btn_fit, "fit"), (self.btn_compare, "compare")):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=15))
            btn.setIconSize(QSize(15, 15))

        sep_color = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(20, 23, 26, 0.12)"
        self.separator.setStyleSheet(f"background-color: {sep_color}; border: none;")

    def retranslate_ui(self) -> None:
        self.btn_zoom_out.setToolTip(tr("tooltip_zoom_out"))
        self.btn_zoom_in.setToolTip(tr("tooltip_zoom_in"))
        self.btn_fit.setText(tr("btn_zoom_fit"))
        self.btn_fit.setToolTip(tr("tooltip_zoom_fit"))
        self.btn_zoom_100.setToolTip(tr("tooltip_zoom_100"))
        self.btn_compare.setText(tr("btn_compare"))
        self.btn_compare.setToolTip(tr("tooltip_compare"))
        self._update_compare_min_width()
        self.adjustSize()


class ToastBanner(QFrame):
    """Píldora flotante temporal sobre la parte superior central del lienzo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ToastBanner")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QLabel("")
        self._label.setObjectName("ToastBannerLabel")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

        self.hide()

    def show_message(self, text: str, duration_ms: int = 2600) -> None:
        clean = (text or "").strip()
        if not clean:
            return
        self._label.setText(clean)
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        self._timer.start(max(400, int(duration_ms)))

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        max_w = max(160, parent.width() - 40)
        if self.width() > max_w:
            self.setFixedWidth(max_w)
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self.adjustSize()
        x = max(12, (parent.width() - self.width()) // 2)
        y = 16
        self.move(x, y)
