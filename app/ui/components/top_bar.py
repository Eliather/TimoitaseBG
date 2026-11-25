"""
Barra superior arrastrable (TopBar) y diálogo sin marco (FramelessPopup) para TimoitaseBG.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDialog,
    QFrame,
    QMenu,
)

from app.config import APP_NAME, APP_VERSION
from app.i18n import tr, get_available_languages, get_current_language
from app.ui.icons import LUCIDE_SVG_PATHS, get_icon


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

        self.btn_close = QPushButton("×")
        self.btn_close.setObjectName("WinCloseBtn")
        self.btn_close.setFixedSize(26, 24)
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
    """Barra superior minimalista con archivo, historial, tema, menú de ajustes y controles de ventana."""

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
        self.setFixedHeight(48)
        self._lang_actions: dict[str, QAction] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 10, 0)
        layout.setSpacing(8)

        title_label = QLabel(APP_NAME)
        title_label.setObjectName("AppTitle")
        layout.addWidget(title_label)

        self.version_badge = QLabel(f"v{APP_VERSION}")
        self.version_badge.setObjectName("appVersionBadge")
        self.version_badge.setToolTip(f"{APP_NAME} v{APP_VERSION}")
        layout.addWidget(self.version_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addSpacing(10)

        self.btn_open = self._make_text_btn("secondaryActionButton", 32, "btn_open", "tooltip_open", self.openRequested.emit)
        self.btn_save = self._make_text_btn("PrimaryBtn", 32, "btn_save", "tooltip_save", self.saveRequested.emit)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_save)

        self._history_sep = QFrame(self)
        self._history_sep.setFixedSize(1, 20)
        self._history_sep.setStyleSheet("background-color: rgba(155, 163, 174, 0.22); border: none;")
        layout.addSpacing(2)
        layout.addWidget(self._history_sep)
        layout.addSpacing(2)

        self.btn_undo = self._make_icon_btn("compactIconButton", 32, 32, "tooltip_undo", self.undoRequested.emit)
        self.btn_redo = self._make_icon_btn("compactIconButton", 32, 32, "tooltip_redo", self.redoRequested.emit)
        self.btn_reset_orig = self._make_icon_btn("compactIconButton", 32, 32, "tooltip_reset", self.resetRequested.emit)
        self.btn_reset = self.btn_reset_orig
        layout.addWidget(self.btn_undo)
        layout.addWidget(self.btn_redo)
        layout.addWidget(self.btn_reset_orig)

        layout.addStretch()

        self.btn_theme = self._make_icon_btn("compactIconButton", 32, 32, "tooltip_theme", self.themeToggled.emit)
        layout.addWidget(self.btn_theme)

        # Botón de Ajustes (⚙) con menú limpio (Idioma + Gestor de Modelos IA)
        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setObjectName("SettingsBtn")
        self.btn_settings.setFixedSize(32, 32)
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_settings.setToolTip(tr("tooltip_lang"))
        self._setup_settings_menu()
        self.btn_settings.clicked.connect(self._open_settings_menu)
        layout.addWidget(self.btn_settings)

        # Controles ocultos para compatibilidad total con tests y código previo
        self.btn_models = QPushButton(self)
        self.btn_models.setObjectName("compactIconButton")
        self.btn_models.setToolTip(tr("tooltip_models"))
        self.btn_models.clicked.connect(self.modelsRequested.emit)
        self.btn_models.hide()

        self.combo_lang = QComboBox(self)
        self.combo_lang.setObjectName("langSelector")
        self.combo_lang.setToolTip(tr("tooltip_lang"))
        for code, name in get_available_languages():
            self.combo_lang.addItem(name, code)
        idx = self.combo_lang.findData(get_current_language())
        if idx >= 0:
            self.combo_lang.setCurrentIndex(idx)
        self.combo_lang.currentIndexChanged.connect(self._on_combo_lang_changed)
        self.combo_lang.hide()

        layout.addSpacing(8)

        # Controles de ventana monocromos en orden estándar Windows: minimizar, maximizar, cerrar
        self.btn_min = self._make_window_btn("WinMinBtn", "—", "Minimizar", self.minimizeRequested.emit)
        self.btn_max = self._make_window_btn("WinMaxBtn", "□", "Maximizar/Restaurar", self.maximizeRequested.emit)
        self.btn_close = self._make_window_btn("WinCloseBtn", "×", "Cerrar", self.closeRequested.emit)
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

    def _setup_settings_menu(self) -> None:
        self.settings_menu = QMenu(self)
        self.settings_menu.setObjectName("SettingsMenu")

        self.lang_menu = self.settings_menu.addMenu(tr("tooltip_lang"))
        self._lang_action_group = QActionGroup(self)
        self._lang_action_group.setExclusive(True)
        current_lang = get_current_language()

        for idx, (code, name) in enumerate(get_available_languages()):
            act = QAction(name, self)
            act.setCheckable(True)
            act.setChecked(code == current_lang)
            act.triggered.connect(lambda _checked=False, lang_idx=idx: self._select_language_index(lang_idx))
            self._lang_action_group.addAction(act)
            self.lang_menu.addAction(act)
            self._lang_actions[code] = act

        self.settings_menu.addSeparator()

        self.action_models = QAction(tr("models_dialog_title"), self)
        self.action_models.setToolTip(tr("tooltip_models"))
        self.action_models.triggered.connect(self.modelsRequested.emit)
        self.settings_menu.addAction(self.action_models)

    def _open_settings_menu(self) -> None:
        pos = self.btn_settings.mapToGlobal(self.btn_settings.rect().bottomRight())
        menu_hint = self.settings_menu.sizeHint()
        pos.setX(pos.x() - menu_hint.width())
        pos.setY(pos.y() + 4)
        self.settings_menu.exec(pos)

    def _select_language_index(self, index: int) -> None:
        if 0 <= index < self.combo_lang.count():
            if self.combo_lang.currentIndex() != index:
                self.combo_lang.setCurrentIndex(index)
            else:
                self.languageChanged.emit(index)

    def _on_combo_lang_changed(self, index: int) -> None:
        code = self.combo_lang.itemData(index)
        if code and code in self._lang_actions:
            self._lang_actions[code].setChecked(True)
        self.languageChanged.emit(index)

    def _make_icon_btn(self, obj_name: str, w: int, h: int, tip_key: str, slot) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName(obj_name)
        btn.setFixedSize(w, h)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
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
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tr(tip_key))
        if slot is not None:
            btn.clicked.connect(slot)
        return btn

    def _make_window_btn(self, obj_name: str, glyph: str, tip: str, slot) -> QPushButton:
        btn = QPushButton(glyph)
        btn.setObjectName(obj_name)
        btn.setFixedSize(32, 28)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tip)
        btn.clicked.connect(slot)
        return btn

    def apply_theme(self, is_dark: bool):
        fg_color = "#F2F4F7" if is_dark else "#14171A"
        accent_color = "#2E7CF6" if is_dark else "#1D63DC"
        dis_color = "#5C6470" if is_dark else "#9BA3AE"

        self.btn_theme.setIcon(get_icon("sun" if is_dark else "moon", color=fg_color, active_color=accent_color, size=17))
        self.btn_theme.setIconSize(QSize(17, 17))
        self.btn_theme.setText("")

        if "settings" in LUCIDE_SVG_PATHS:
            self.btn_settings.setIcon(get_icon("settings", color=fg_color, active_color=accent_color, size=17))
            self.btn_settings.setIconSize(QSize(17, 17))
            self.btn_settings.setText("")
        else:
            self.btn_settings.setText("⚙")

        self.btn_models.setIcon(get_icon("hard-drive", color=fg_color, active_color=accent_color, size=16))
        self.btn_models.setIconSize(QSize(16, 16))
        self.action_models.setIcon(get_icon("hard-drive", color=fg_color, active_color=accent_color, size=15))

        self.btn_open.setIcon(get_icon("folder-open", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=15))
        self.btn_open.setIconSize(QSize(15, 15))

        self.btn_save.setIcon(get_icon("save", color="#FFFFFF", active_color="#FFFFFF", disabled_color=dis_color, size=15))
        self.btn_save.setIconSize(QSize(15, 15))

        for btn, icon_name in ((self.btn_undo, "undo"), (self.btn_redo, "redo"), (self.btn_reset_orig, "reset")):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=17))
            btn.setIconSize(QSize(17, 17))
            btn.setText("")

        for btn, icon_name, fallback in (
            (self.btn_min, "win-min", "—"),
            (self.btn_max, "win-max", "□"),
            (self.btn_close, "win-close", "×"),
        ):
            if icon_name in LUCIDE_SVG_PATHS:
                btn.setIcon(get_icon(icon_name, color=fg_color, active_color="#FFFFFF" if icon_name == "win-close" else fg_color, size=14))
                btn.setIconSize(QSize(14, 14))
                btn.setText("")
            else:
                btn.setText(fallback)

        sep_color = "rgba(255, 255, 255, 0.10)" if is_dark else "rgba(20, 23, 26, 0.12)"
        self._history_sep.setStyleSheet(f"background-color: {sep_color}; border: none;")
        self.version_badge.setStyleSheet("")

    def retranslate_ui(self):
        self.btn_open.setText(tr("btn_open"))
        self.btn_open.setToolTip(tr("tooltip_open"))
        self.btn_save.setText(tr("btn_save"))
        self.btn_save.setToolTip(tr("tooltip_save"))
        self.btn_theme.setToolTip(tr("tooltip_theme"))
        self.btn_settings.setToolTip(tr("tooltip_lang"))
        self.lang_menu.setTitle(tr("tooltip_lang"))
        self.action_models.setText(tr("models_dialog_title"))
        self.action_models.setToolTip(tr("tooltip_models"))
        self.btn_models.setToolTip(tr("tooltip_models"))
        self.combo_lang.setToolTip(tr("tooltip_lang"))
        self.btn_undo.setToolTip(tr("tooltip_undo"))
        self.btn_redo.setToolTip(tr("tooltip_redo"))
        self.btn_reset_orig.setToolTip(tr("tooltip_reset"))
