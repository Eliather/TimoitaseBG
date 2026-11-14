"""
Barra superior arrastrable (TopBar) y diálogo sin marco (FramelessPopup) para TimoitaseBG.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDialog,
)

from app.config import APP_NAME, APP_VERSION
from app.i18n import tr, get_available_languages, get_current_language
from app.ui.icons import get_icon


class FramelessDragMixin:
    """Mixin para arrastrar barras de título o diálogos sin bordes con botón izquierdo."""

    def _drag_target(self):
        return getattr(self, "parent_window", self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = True
            self._start_pos = event.globalPosition().toPoint() - self._drag_target().pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if getattr(self, "_is_tracking", False):
            self._drag_target().move(event.globalPosition().toPoint() - self._start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = False
            event.accept()


class CustomTopBar(FramelessDragMixin, QWidget):
    """Barra de título personalizada con arrastre y doble clic para maximizar."""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self._is_tracking = False
        self._start_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_window._toggle_maximize()
            event.accept()


class FramelessPopup(FramelessDragMixin, QDialog):
    """Popup modal sin marco para notificaciones de guardado o advertencias."""

    def __init__(self, parent, title: str, message: str):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        self.resize(380, 160)

        self.root_widget = QWidget(self)
        self.root_widget.setObjectName("framelessPopupRoot")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.root_widget)

        layout = QVBoxLayout(self.root_widget)
        layout.setContentsMargins(15, 12, 15, 15)

        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("popupTitle")
        top_layout.addWidget(title_lbl)
        top_layout.addStretch()

        self.btn_close = QPushButton("")
        self.btn_close.setObjectName("macCloseBtn")
        self.btn_close.setFixedSize(14, 14)
        self.btn_close.clicked.connect(self.close)
        top_layout.addWidget(self.btn_close)

        layout.addWidget(top_bar)
        layout.addSpacing(10)

        msg_lbl = QLabel(message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg_lbl)
        layout.addStretch()

        self.btn_ok = QPushButton(tr("btn_ok"))
        self.btn_ok.setObjectName("actionButton")
        self.btn_ok.setFixedSize(100, 32)
        self.btn_ok.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


class TopBar(CustomTopBar):
    """Barra superior completa con controles de archivo, historial, zoom, tema e idioma."""

    themeToggled = Signal()
    languageChanged = Signal(int)
    modelsRequested = Signal()
    openRequested = Signal()
    saveRequested = Signal()
    undoRequested = Signal()
    redoRequested = Signal()
    resetRequested = Signal()
    zoomOutRequested = Signal()
    zoomInRequested = Signal()
    fitRequested = Signal()
    zoom100Requested = Signal()
    compareToggled = Signal(bool)
    minimizeRequested = Signal()
    maximizeRequested = Signal()
    closeRequested = Signal()

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.setObjectName("TopBar")
        self.setFixedHeight(58)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        title_label = QLabel(APP_NAME)
        title_label.setObjectName("AppTitle")
        layout.addWidget(title_label)

        self.version_badge = QLabel(f"v{APP_VERSION}")
        self.version_badge.setObjectName("appVersionBadge")
        self.version_badge.setToolTip(f"{APP_NAME} v{APP_VERSION}")
        layout.addWidget(self.version_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addSpacing(6)

        self.btn_theme = self._make_icon_btn("compactIconButton", 34, 34, "tooltip_theme", self.themeToggled.emit)
        layout.addWidget(self.btn_theme)

        self.btn_models = self._make_icon_btn("compactIconButton", 34, 34, "tooltip_models", self.modelsRequested.emit)
        layout.addWidget(self.btn_models)

        self.combo_lang = QComboBox()
        self.combo_lang.setObjectName("langSelector")
        self.combo_lang.setToolTip(tr("tooltip_lang"))
        self.combo_lang.setFixedHeight(34)
        self.combo_lang.setFixedWidth(110)
        for code, name in get_available_languages():
            self.combo_lang.addItem(name, code)
        idx = self.combo_lang.findData(get_current_language())
        if idx >= 0:
            self.combo_lang.setCurrentIndex(idx)
        self.combo_lang.currentIndexChanged.connect(self.languageChanged.emit)
        layout.addWidget(self.combo_lang)
        layout.addSpacing(6)

        self.btn_open = self._make_text_btn("PrimaryBtn", 34, "btn_open", "tooltip_open", self.openRequested.emit)
        self.btn_save = self._make_text_btn("PrimaryBtn", 34, "btn_save", "tooltip_save", self.saveRequested.emit)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_save)
        layout.addSpacing(4)

        self.btn_undo = self._make_icon_btn("compactIconButton", 34, 34, "tooltip_undo", self.undoRequested.emit)
        self.btn_redo = self._make_icon_btn("compactIconButton", 34, 34, "tooltip_redo", self.redoRequested.emit)
        self.btn_reset_orig = self._make_icon_btn("compactIconButton", 34, 34, "tooltip_reset", self.resetRequested.emit)
        layout.addWidget(self.btn_undo)
        layout.addWidget(self.btn_redo)
        layout.addWidget(self.btn_reset_orig)
        layout.addStretch()

        self.btn_zoom_out = self._make_icon_btn("zoomButton", 32, 32, "tooltip_zoom_out", self.zoomOutRequested.emit)
        layout.addWidget(self.btn_zoom_out)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setObjectName("zoomLabel")
        self.lbl_zoom.setFixedWidth(52)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_zoom)

        self.btn_zoom_in = self._make_icon_btn("zoomButton", 32, 32, "tooltip_zoom_in", self.zoomInRequested.emit)
        layout.addWidget(self.btn_zoom_in)

        self.btn_fit = self._make_text_btn("zoomButton", 32, "btn_zoom_fit", "tooltip_zoom_fit", self.fitRequested.emit)
        layout.addWidget(self.btn_fit)

        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setObjectName("zoomButton")
        self.btn_zoom_100.setFixedHeight(32)
        self.btn_zoom_100.setToolTip(tr("tooltip_zoom_100"))
        self.btn_zoom_100.clicked.connect(self.zoom100Requested.emit)
        layout.addWidget(self.btn_zoom_100)
        layout.addSpacing(6)

        self.btn_compare = self._make_text_btn(
            "PrimaryBtn", 34, "btn_compare", "tooltip_compare", checkable=True
        )
        self.btn_compare.clicked.connect(self.compareToggled.emit)
        layout.addWidget(self.btn_compare)
        layout.addSpacing(14)

        self.btn_min = self._make_window_dot("macMinBtn", "Minimizar", self.minimizeRequested.emit)
        self.btn_max = self._make_window_dot("macMaxBtn", "Maximizar/Restaurar", self.maximizeRequested.emit)
        self.btn_close = self._make_window_dot("macCloseBtn", "Cerrar", self.closeRequested.emit)
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

    def _make_icon_btn(self, obj_name: str, w: int, h: int, tip_key: str, slot) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName(obj_name)
        btn.setFixedSize(w, h)
        btn.setToolTip(tr(tip_key))
        btn.clicked.connect(slot)
        return btn

    def _make_text_btn(
        self, obj_name: str, h: int, text_key: str, tip_key: str, slot=None, checkable: bool = False
    ) -> QPushButton:
        btn = QPushButton(tr(text_key))
        btn.setObjectName(obj_name)
        btn.setFixedHeight(h)
        btn.setCheckable(checkable)
        btn.setToolTip(tr(tip_key))
        if slot is not None:
            btn.clicked.connect(slot)
        return btn

    def _make_window_dot(self, obj_name: str, tip: str, slot) -> QPushButton:
        btn = QPushButton("")
        btn.setObjectName(obj_name)
        btn.setFixedSize(14, 14)
        btn.setToolTip(tip)
        btn.clicked.connect(slot)
        return btn

    def apply_theme(self, is_dark: bool):
        fg_color = "#EAEAEA" if is_dark else "#292524"
        accent_color = "#FF4F79"
        dis_color = "#666666" if is_dark else "#A8A29E"

        self.btn_theme.setIcon(get_icon("sun" if is_dark else "moon", color=fg_color, active_color=accent_color, size=18))
        self.btn_theme.setIconSize(QSize(18, 18))
        self.btn_theme.setText("")

        self.btn_models.setIcon(get_icon("hard-drive", color=fg_color, active_color=accent_color, size=18))
        self.btn_models.setIconSize(QSize(18, 18))
        self.btn_models.setText("")

        for btn, icon_name in ((self.btn_open, "folder-open"), (self.btn_save, "save")):
            btn.setIcon(get_icon(icon_name, color="#FFFFFF", active_color="#FFFFFF", disabled_color="#FECDD3", size=16))
            btn.setIconSize(QSize(16, 16))

        for btn, icon_name in ((self.btn_undo, "undo"), (self.btn_redo, "redo"), (self.btn_reset_orig, "reset")):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=18))
            btn.setIconSize(QSize(18, 18))
            btn.setText("")

        for btn, icon_name in ((self.btn_zoom_out, "zoom-out"), (self.btn_zoom_in, "zoom-in")):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            btn.setIconSize(QSize(16, 16))
            btn.setText("")

        for btn, icon_name in ((self.btn_fit, "fit"), (self.btn_compare, "compare")):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            btn.setIconSize(QSize(16, 16))

        if is_dark:
            self.version_badge.setStyleSheet("""
                background-color: #334155; color: #CBD5E1; border: 1px solid #475569;
                padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 600;
            """)
        else:
            self.version_badge.setStyleSheet("""
                background-color: #F1F5F9; color: #64748B; border: 1px solid #E2E8F0;
                padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 600;
            """)

    def retranslate_ui(self):
        self.btn_open.setText(tr("btn_open"))
        self.btn_open.setToolTip(tr("tooltip_open"))
        self.btn_save.setText(tr("btn_save"))
        self.btn_save.setToolTip(tr("tooltip_save"))
        self.btn_compare.setText(tr("btn_compare"))
        self.btn_compare.setToolTip(tr("tooltip_compare"))
        self.btn_fit.setText(tr("btn_zoom_fit"))
        self.btn_fit.setToolTip(tr("tooltip_zoom_fit"))
        self.btn_theme.setToolTip(tr("tooltip_theme"))
        self.btn_models.setToolTip(tr("tooltip_models"))
        self.combo_lang.setToolTip(tr("tooltip_lang"))
        self.btn_undo.setToolTip(tr("tooltip_undo"))
        self.btn_redo.setToolTip(tr("tooltip_redo"))
        self.btn_reset_orig.setToolTip(tr("tooltip_reset"))
        self.btn_zoom_out.setToolTip(tr("tooltip_zoom_out"))
        self.btn_zoom_in.setToolTip(tr("tooltip_zoom_in"))
        self.btn_zoom_100.setToolTip(tr("tooltip_zoom_100"))
