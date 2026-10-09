"""
TimoitaseBG - Ventana principal de la aplicación.
"""
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QIcon, QKeySequence, QShortcut, QAction, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QSplitter,
    QFrame,
    QDialog,
)

from app.i18n import tr, set_language, get_available_languages, get_current_language

from app.config import (
    APP_NAME,
    DEFAULT_WINDOW_TITLE,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    detect_device,
    APP_DIR,
)
import numpy as np
import logging

logger = logging.getLogger("TimoitaseBG.MainWindow")
from app.core.image_state import ImageState
from app.core.image_utils import qimage_to_pil, autocrop_image, apply_solid_background
from app.core.bg_remover import get_bg_remover
from app.core.restorer import get_image_restorer
from app.core.workers import WorkerThread
from app.ui.canvas_widget import CanvasWidget
from app.ui.toolbar import ToolSidebar
from app.ui.icons import get_icon
from app.ui.shortcuts_dialog import ShortcutsDialog

class CustomTopBar(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self._is_tracking = False
        self._start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = True
            self._start_pos = event.globalPosition().toPoint() - self.parent_window.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_tracking:
            self.parent_window.move(event.globalPosition().toPoint() - self._start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = False
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_window._toggle_maximize()
            event.accept()

class FramelessPopup(QDialog):
    def __init__(self, parent, title, message):
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = True
            self._start_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if hasattr(self, '_is_tracking') and self._is_tracking:
            self.move(event.globalPosition().toPoint() - self._start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = False
            event.accept()

class MainWindow(QMainWindow):
    """
    Ventana principal de TimoitaseBG.
    Coordina el estado de la imagen, el canvas interactivo y el panel de herramientas.
    """

    gpuStatusNotified = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(1150, 750)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Estado global de la imagen e historial
        self.image_state = ImageState()
        self.current_filepath: Optional[Path] = None

        # Dispositivo detectado
        self.device = detect_device()

        # Configurar UI
        self._init_ui()
        self._setup_shortcuts()
        self._update_action_states()

        # Conectar notificaciones asíncronas de GPU al hilo de la UI
        self.gpuStatusNotified.connect(self._handle_gpu_status)
        from app.core.gpu_manager import get_gpu_manager
        get_gpu_manager().set_status_callback(lambda msg: self.gpuStatusNotified.emit(msg))
        
        # Cargar estilos nativos (Modo Oscuro)
        self._load_stylesheet()

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _set_status(self, key: str, **kwargs):
        """Asigna texto a la barra de estado y guarda la clave para re-traducción en vivo."""
        self._current_status_i18n = (key, kwargs)
        if hasattr(self, "lbl_status"):
            self.lbl_status.setText(tr(key, **kwargs))

    def _set_status_raw(self, text: str):
        """Asigna texto directo que no tiene clave de traducción."""
        self._current_status_i18n = None
        if hasattr(self, "lbl_status"):
            self.lbl_status.setText(text)

    def _on_lang_changed(self, index):
        code = self.combo_lang.itemData(index)
        if code and code != get_current_language():
            set_language(code)
            self.retranslate_ui()

    def retranslate_ui(self):
        """Actualiza dinámicamente los textos de toda la interfaz al cambiar de idioma."""
        # Barra de estado inferior
        if hasattr(self, "lbl_status"):
            if getattr(self, "_current_status_i18n", None):
                k, args = self._current_status_i18n
                self.lbl_status.setText(tr(k, **args))
            else:
                from app.i18n import TRANSLATIONS
                ready_texts = {trans.get("status_ready", "") for trans in TRANSLATIONS.values()}
                ready_texts.update({"Listo", "Ready", "Готово", "就绪", ""})
                if not getattr(self, "image_state", None) or not self.image_state.has_image or self.lbl_status.text() in ready_texts:
                    self._set_status("status_ready")

        # Botones de la barra de estado inferior
        if hasattr(self, "btn_shortcuts"):
            self.btn_shortcuts.setText(tr("btn_shortcuts"))
            self.btn_shortcuts.setToolTip(tr("tooltip_shortcuts"))
        if hasattr(self, "btn_kofi"):
            self.btn_kofi.setText(tr("btn_kofi"))
            self.btn_kofi.setToolTip(tr("tooltip_kofi"))

        # Botones de la barra superior
        if hasattr(self, "btn_open"):
            self.btn_open.setText(tr("btn_open"))
            self.btn_open.setToolTip(tr("tooltip_open"))
        if hasattr(self, "btn_save"):
            self.btn_save.setText(tr("btn_save"))
            self.btn_save.setToolTip(tr("tooltip_save"))
        if hasattr(self, "btn_compare"):
            self.btn_compare.setText(tr("btn_compare"))
            self.btn_compare.setToolTip(tr("tooltip_compare"))
        if hasattr(self, "btn_fit"):
            self.btn_fit.setText(tr("btn_zoom_fit"))
            self.btn_fit.setToolTip(tr("tooltip_zoom_fit"))

        # Tooltips de barra superior
        if hasattr(self, "btn_theme"):
            self.btn_theme.setToolTip(tr("tooltip_theme"))
        if hasattr(self, "combo_lang"):
            self.combo_lang.setToolTip(tr("tooltip_lang"))
        if hasattr(self, "btn_undo"):
            self.btn_undo.setToolTip(tr("tooltip_undo"))
        if hasattr(self, "btn_redo"):
            self.btn_redo.setToolTip(tr("tooltip_redo"))
        if hasattr(self, "btn_reset_orig"):
            self.btn_reset_orig.setToolTip(tr("tooltip_reset"))
        if hasattr(self, "btn_zoom_out"):
            self.btn_zoom_out.setToolTip(tr("tooltip_zoom_out"))
        if hasattr(self, "btn_zoom_in"):
            self.btn_zoom_in.setToolTip(tr("tooltip_zoom_in"))
        if hasattr(self, "btn_zoom_100"):
            self.btn_zoom_100.setToolTip(tr("tooltip_zoom_100"))

        # Barra lateral
        if hasattr(self, "sidebar") and hasattr(self.sidebar, "retranslate_ui"):
            self.sidebar.retranslate_ui()

        # Canvas (texto central cuando no hay imagen)
        if hasattr(self, "canvas"):
            self.canvas.update()

    def _load_stylesheet(self):
        self.is_dark_theme = getattr(self, "is_dark_theme", False)
        style_file = "style.qss" if self.is_dark_theme else "style_light.qss"
        style_path = Path(APP_DIR) / "ui" / style_file
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                content = f.read()
                assets_path = (Path(APP_DIR) / "ui" / "assets").as_posix()
                content = content.replace("{ASSETS_PATH}", assets_path)
                self.setStyleSheet(content)
        self._update_theme_icons()

    def _toggle_theme(self):
        self.is_dark_theme = not getattr(self, "is_dark_theme", False)
        self._load_stylesheet()

    def _update_theme_icons(self):
        """Aplica iconos vectoriales nítidos y adaptados al contraste del tema activo."""
        fg_color = "#EAEAEA" if self.is_dark_theme else "#292524"
        accent_color = "#FF4F79"
        dis_color = "#666666" if self.is_dark_theme else "#A8A29E"

        theme_icon = "sun" if self.is_dark_theme else "moon"
        if hasattr(self, "btn_theme"):
            self.btn_theme.setIcon(get_icon(theme_icon, color=fg_color, active_color=accent_color, size=18))
            self.btn_theme.setIconSize(QSize(18, 18))
            self.btn_theme.setText("")

        if hasattr(self, "btn_open"):
            self.btn_open.setIcon(get_icon("folder-open", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_open.setIconSize(QSize(16, 16))

        if hasattr(self, "btn_save"):
            self.btn_save.setIcon(get_icon("save", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_save.setIconSize(QSize(16, 16))

        if hasattr(self, "btn_undo"):
            self.btn_undo.setIcon(get_icon("undo", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=18))
            self.btn_undo.setIconSize(QSize(18, 18))
            self.btn_undo.setText("")

        if hasattr(self, "btn_redo"):
            self.btn_redo.setIcon(get_icon("redo", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=18))
            self.btn_redo.setIconSize(QSize(18, 18))
            self.btn_redo.setText("")

        if hasattr(self, "btn_reset_orig"):
            self.btn_reset_orig.setIcon(get_icon("reset", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=18))
            self.btn_reset_orig.setIconSize(QSize(18, 18))
            self.btn_reset_orig.setText("")

        if hasattr(self, "btn_zoom_out"):
            self.btn_zoom_out.setIcon(get_icon("zoom-out", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_zoom_out.setIconSize(QSize(16, 16))
            self.btn_zoom_out.setText("")

        if hasattr(self, "btn_zoom_in"):
            self.btn_zoom_in.setIcon(get_icon("zoom-in", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_zoom_in.setIconSize(QSize(16, 16))
            self.btn_zoom_in.setText("")

        if hasattr(self, "btn_fit"):
            self.btn_fit.setIcon(get_icon("fit", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_fit.setIconSize(QSize(16, 16))

        if hasattr(self, "btn_compare"):
            self.btn_compare.setIcon(get_icon("compare", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_compare.setIconSize(QSize(16, 16))

        if hasattr(self, "btn_kofi"):
            self.btn_kofi.setIcon(get_icon("coffee", color="#FFFFFF", size=16))
            self.btn_kofi.setIconSize(QSize(16, 16))

        if hasattr(self, "btn_shortcuts"):
            self.btn_shortcuts.setIcon(get_icon("keyboard", color=fg_color, active_color=accent_color, size=15))
            self.btn_shortcuts.setIconSize(QSize(15, 15))

        if hasattr(self, "sidebar") and hasattr(self.sidebar, "update_theme_icons"):
            self.sidebar.update_theme_icons(self.is_dark_theme)
                
    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Instanciar Canvas central y Sidebar primero
        self.canvas = CanvasWidget()
        self.canvas.maskChanged.connect(self._on_mask_changed)
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        self.canvas.fileDropped.connect(self.open_image_file)
        self.canvas.imageEditedDirectly.connect(self._on_image_edited_directly)

        self.sidebar = ToolSidebar()
        self.sidebar.setObjectName("toolSidebar")
        self.sidebar.brushModeToggled.connect(self._on_brush_mode_toggled)
        self.sidebar.eraserModeToggled.connect(self.canvas.set_eraser_mode)
        self.sidebar.wandModeToggled.connect(self._on_wand_mode_toggled)
        self.sidebar.wandAIModeToggled.connect(self._on_wand_ai_mode_toggled)
        self.canvas.set_wand_ai_mode(self.sidebar.is_wand_ai_mode())
        self.sidebar.wandToleranceChanged.connect(self.canvas.set_wand_tolerance)
        self.sidebar.wandSelectionModeChanged.connect(self.canvas.set_wand_selection_mode)
        self.sidebar.wandAntiAliasChanged.connect(self.canvas.set_wand_anti_alias)
        self.sidebar.wandContiguousChanged.connect(self.canvas.set_wand_contiguous)
        self.sidebar.deleteSelectionRequested.connect(self.perform_delete_selection)
        self.sidebar.invertSelectionRequested.connect(self.canvas.invert_selection)
        self.sidebar.selectAllRequested.connect(self.canvas.select_all)
        self.sidebar.clearSelectionRequested.connect(self.canvas.clear_selection)
        self.canvas.selectionChanged.connect(self.sidebar.set_selection_info)
        self.sidebar.brushSizeChanged.connect(self.canvas.set_brush_size)
        self.sidebar.brushSmoothnessChanged.connect(self.canvas.set_brush_smoothness)
        self.sidebar.brushOpacityChanged.connect(self.canvas.set_brush_opacity)
        self.sidebar.brushHardnessChanged.connect(self.canvas.set_brush_hardness)
        self.sidebar.removeBgRequested.connect(self.start_remove_bg)
        self.sidebar.restoreRequested.connect(self.start_restore)
        self.sidebar.autoCropRequested.connect(self.perform_autocrop)
        self.sidebar.applyBgColorRequested.connect(self.perform_apply_bg_color)
        self.sidebar.restoreTransparencyRequested.connect(self.perform_restore_transparency)
        self.sidebar.deviceModeChanged.connect(self._on_device_mode_changed)
        self.sidebar.batchRequested.connect(self.start_batch_processing)
        self.sidebar.detectSubjectsRequested.connect(self.start_detect_subjects)
        self.sidebar.isolateSubjectRequested.connect(self.perform_isolate_subject)
        self.sidebar.selectSubjectRequested.connect(self.perform_select_subject)

        # 2. Barra Superior (Top Bar)
        top_bar = self._create_top_bar()
        root_layout.addWidget(top_bar)

        # 3. Área central: Canvas + Sidebar
        main_content = QWidget()
        content_layout = QHBoxLayout(main_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        content_layout.addWidget(self.canvas, stretch=1)
        content_layout.addWidget(self.sidebar)

        root_layout.addWidget(main_content, stretch=1)

        # 4. Barra Inferior de Estado
        self.status_bar = self._create_status_bar()
        root_layout.addWidget(self.status_bar)

    # -------------------------------------------------------------
    # Barra Superior
    # -------------------------------------------------------------
    def _create_top_bar(self) -> QWidget:
        top_bar = CustomTopBar(self)
        top_bar.setObjectName("topBar")
        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # Título
        from app.config import DEFAULT_WINDOW_TITLE
        title_label = QLabel(DEFAULT_WINDOW_TITLE)
        title_label.setObjectName("appNameLabel")
        layout.addWidget(title_label)

        layout.addSpacing(12)

        self.btn_theme = QPushButton()
        self.btn_theme.setObjectName("compactIconButton")
        self.btn_theme.setFixedSize(30, 30)
        self.btn_theme.setToolTip(tr("tooltip_theme"))
        self.btn_theme.clicked.connect(self._toggle_theme)
        layout.addWidget(self.btn_theme)

        layout.addSpacing(12)

        self.combo_lang = QComboBox()
        self.combo_lang.setObjectName("langSelector")
        self.combo_lang.setToolTip(tr("tooltip_lang"))
        self.combo_lang.setFixedHeight(30)
        for code, name in get_available_languages():
            self.combo_lang.addItem(name, code)
        
        current = get_current_language()
        idx = self.combo_lang.findData(current)
        if idx >= 0:
            self.combo_lang.setCurrentIndex(idx)
        
        self.combo_lang.currentIndexChanged.connect(self._on_lang_changed)
        layout.addWidget(self.combo_lang)

        layout.addSpacing(12)

        # Botón Abrir Imagen
        self.btn_open = QPushButton(tr("btn_open"))
        self.btn_open.setObjectName("actionButton")
        self.btn_open.setFixedHeight(32)
        self.btn_open.setToolTip(tr("tooltip_open"))
        self.btn_open.clicked.connect(self.prompt_open_image)
        layout.addWidget(self.btn_open)

        # Botón Guardar Como
        self.btn_save = QPushButton(tr("btn_save"))
        self.btn_save.setObjectName("actionButton")
        self.btn_save.setFixedHeight(32)
        self.btn_save.setToolTip(tr("tooltip_save"))
        self.btn_save.clicked.connect(self.prompt_save_image)
        layout.addWidget(self.btn_save)

        layout.addWidget(self._create_separator())

        # Botones compactos de historial con tooltip
        self.btn_undo = QPushButton()
        self.btn_undo.setObjectName("compactIconButton")
        self.btn_undo.setFixedSize(34, 34)
        self.btn_undo.setToolTip(tr("tooltip_undo"))
        self.btn_undo.clicked.connect(self.perform_undo)
        layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton()
        self.btn_redo.setObjectName("compactIconButton")
        self.btn_redo.setFixedSize(34, 34)
        self.btn_redo.setToolTip(tr("tooltip_redo"))
        self.btn_redo.clicked.connect(self.perform_redo)
        layout.addWidget(self.btn_redo)

        self.btn_reset_orig = QPushButton()
        self.btn_reset_orig.setObjectName("compactIconButton")
        self.btn_reset_orig.setFixedSize(34, 34)
        self.btn_reset_orig.setToolTip(tr("tooltip_reset"))
        self.btn_reset_orig.clicked.connect(self.perform_reset_original)
        layout.addWidget(self.btn_reset_orig)

        layout.addStretch()

        # Controles de Zoom
        self.btn_zoom_out = QPushButton()
        self.btn_zoom_out.setObjectName("zoomButton")
        self.btn_zoom_out.setFixedSize(30, 30)
        self.btn_zoom_out.setToolTip(tr("tooltip_zoom_out"))
        self.btn_zoom_out.clicked.connect(lambda: self.canvas.set_zoom_delta(1.0 / 1.2))
        layout.addWidget(self.btn_zoom_out)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setObjectName("zoomLabel")
        self.lbl_zoom.setFixedWidth(50)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_zoom)

        self.btn_zoom_in = QPushButton()
        self.btn_zoom_in.setObjectName("zoomButton")
        self.btn_zoom_in.setFixedSize(30, 30)
        self.btn_zoom_in.setToolTip(tr("tooltip_zoom_in"))
        self.btn_zoom_in.clicked.connect(lambda: self.canvas.set_zoom_delta(1.2))
        layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton(tr("btn_zoom_fit"))
        self.btn_fit.setObjectName("zoomButton")
        self.btn_fit.setFixedHeight(30)
        self.btn_fit.setToolTip(tr("tooltip_zoom_fit"))
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)
        layout.addWidget(self.btn_fit)

        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setObjectName("zoomButton")
        self.btn_zoom_100.setFixedHeight(30)
        self.btn_zoom_100.setToolTip(tr("tooltip_zoom_100"))
        self.btn_zoom_100.clicked.connect(self.canvas.reset_zoom_100)
        layout.addWidget(self.btn_zoom_100)

        layout.addWidget(self._create_separator())

        # Botón Modo Comparar (Antes / Después con cortina)
        self.btn_compare = QPushButton(tr("btn_compare"))
        self.btn_compare.setObjectName("actionButton")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setFixedHeight(32)
        self.btn_compare.setToolTip(tr("tooltip_compare"))
        self.btn_compare.clicked.connect(self._on_toggle_compare)
        layout.addWidget(self.btn_compare)

        layout.addSpacing(15)

        # Botones de ventana (estilo macOS)
        self.btn_min = QPushButton("")
        self.btn_min.setObjectName("macMinBtn")
        self.btn_min.setFixedSize(14, 14)
        self.btn_min.setToolTip("Minimizar")
        self.btn_min.clicked.connect(self.showMinimized)

        self.btn_max = QPushButton("")
        self.btn_max.setObjectName("macMaxBtn")
        self.btn_max.setFixedSize(14, 14)
        self.btn_max.setToolTip("Maximizar/Restaurar")
        self.btn_max.clicked.connect(self._toggle_maximize)

        self.btn_close = QPushButton("")
        self.btn_close.setObjectName("macCloseBtn")
        self.btn_close.setFixedSize(14, 14)
        self.btn_close.setToolTip("Cerrar")
        self.btn_close.clicked.connect(self.close)

        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

        return top_bar
    # -------------------------------------------------------------
    # Barra Inferior de Estado
    # -------------------------------------------------------------
    def _create_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("statusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)

        # Botón de Atajos Rápidos (a la izquierda)
        self.btn_shortcuts = QPushButton(tr("btn_shortcuts"))
        self.btn_shortcuts.setObjectName("statusShortcutsButton")
        self.btn_shortcuts.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_shortcuts.setToolTip(tr("tooltip_shortcuts"))
        self.btn_shortcuts.clicked.connect(self.show_shortcuts_modal)
        layout.addWidget(self.btn_shortcuts)

        layout.addWidget(self._create_separator())

        # Estado del documento / imagen
        self.lbl_status = QLabel(tr("status_ready"))
        self.lbl_status.setObjectName("statusText")
        self._current_status_i18n = ("status_ready", {})
        layout.addWidget(self.lbl_status)

        self.lbl_img_info = QLabel("")
        self.lbl_img_info.setObjectName("imageInfoText")
        layout.addWidget(self.lbl_img_info)

        layout.addStretch()

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setRange(0, 0) # Indeterminado por defecto
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Botón de Donaciones Ko-fi
        self.btn_kofi = QPushButton(tr("btn_kofi"))
        self.btn_kofi.setObjectName("kofiButton")
        self.btn_kofi.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_kofi.setToolTip(tr("tooltip_kofi"))
        import webbrowser
        self.btn_kofi.clicked.connect(lambda: webbrowser.open("https://ko-fi.com/eliather"))
        layout.addWidget(self.btn_kofi)

        # Añadir grip de redimensionamiento en la esquina inferior derecha
        from PySide6.QtWidgets import QSizeGrip
        self.size_grip = QSizeGrip(self)
        layout.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        return bar

    def _create_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setStyleSheet("background-color: #E7E5E4; width: 1px; max-height: 20px; border: none;")
        return sep

    def show_shortcuts_modal(self):
        """Abre el diálogo modal con la lista interactiva de atajos de teclado y gestos."""
        dialog = ShortcutsDialog(self, is_dark_theme=getattr(self, "is_dark_theme", False))
        dialog.exec()

    # -------------------------------------------------------------
    # Atajos de Teclado
    # -------------------------------------------------------------
    def _setup_shortcuts(self):
        # Ayuda y atajos
        QShortcut(QKeySequence("F1"), self, self.show_shortcuts_modal)
        QShortcut(QKeySequence("Ctrl+/"), self, self.show_shortcuts_modal)

        # Abrir / Guardar / Pegar
        QShortcut(QKeySequence("Ctrl+O"), self, self.prompt_open_image)
        QShortcut(QKeySequence("Ctrl+S"), self, self.prompt_save_image)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self, self.paste_from_clipboard)

        # Deshacer / Rehacer
        QShortcut(QKeySequence("Ctrl+Z"), self, self.perform_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.perform_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.perform_redo)

        # Ajuste de vista
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas.fit_to_view)
        QShortcut(QKeySequence("Ctrl+1"), self, self.canvas.reset_zoom_100)

        # Selección (mismos atajos que Photoshop)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.perform_delete_selection)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, self.perform_delete_selection)
        QShortcut(QKeySequence("Ctrl+D"), self, self.canvas.clear_selection)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.canvas.clear_selection)
        QShortcut(QKeySequence("Ctrl+A"), self, self.canvas.select_all)
        QShortcut(QKeySequence("Ctrl+Shift+I"), self, self.canvas.invert_selection)

    def perform_delete_selection(self):
        """Borra los píxeles seleccionados con la varita (Supr / botón 'Borrar selección')."""
        if not self.canvas.has_selection():
            return
        if self.canvas.delete_selection():
            self._set_status("status_selection_deleted")
        else:
            self._set_status("status_selection_nothing")

    # -------------------------------------------------------------
    # Lógica de Apertura, Pegado y Guardado
    # -------------------------------------------------------------
    def prompt_open_image(self):
        """Abre un diálogo para seleccionar archivo de imagen."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Imágenes (*.png *.jpg *.jpeg *.webp *.bmp);;Todos los archivos (*.*)",
        )
        if path:
            self.open_image_file(path)

    def load_new_pil_image(self, pil_img: Image.Image, source_name: str, filepath: Optional[Path] = None):
        """
        Carga una nueva imagen PIL en el estado inmutable, reinicia el historial y actualiza el canvas y la UI.
        Reutilizado tanto para 'Abrir Imagen' como para 'Pegar desde el portapapeles (Ctrl+V)'.
        """
        if pil_img.width <= 0 or pil_img.height <= 0:
            return

        self.current_filepath = filepath
        loaded = self.image_state.load_new_image(pil_img)

        # Invalidar embedding SAM previo
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.btn_wand.isChecked():
            self._ensure_sam_embedding_async()
        if hasattr(self, "sidebar"):
            self.sidebar.set_detected_subjects([])

        self.btn_compare.setChecked(False)
        self.canvas.set_comparison_mode(False)
        self.canvas.set_image(loaded, reset_zoom=True)
        self.canvas.set_original_image(self.image_state.current_brush_reference)

        self._set_status("status_loaded", name=source_name)
        self._update_image_info_label(loaded)
        self._update_action_states()

    def open_image_file(self, file_path: str):
        """Carga una imagen desde archivo en el estado y canvas."""
        try:
            path = Path(file_path)
            if not path.exists():
                return

            pil_img = Image.open(path)
            self.load_new_pil_image(pil_img, source_name=path.name, filepath=path)
        except Exception as e:
            QMessageBox.critical(self, "Error al abrir imagen", f"No se pudo cargar el archivo:\n{e}")

    def paste_from_clipboard(self):
        """
        Pega una imagen desde el portapapeles con Ctrl+V.
        Detecta estrictamente contenido de imagen:
        - Imagen directa en el portapapeles (bitmap / QImage de navegador, capturador, editor).
        - Archivo de imagen copiado desde el Explorador de Windows (.png, .jpg, .jpeg, .webp, .bmp).
        Si el portapapeles contiene texto plano, URL de texto o archivos no imagen, no realiza ninguna acción.
        """
        clipboard = QGuiApplication.clipboard()
        mime_data = clipboard.mimeData()

        # 1. Detección estricta de mapa de bits (QImage) en portapapeles (ej. Copiar imagen en navegador)
        if mime_data.hasImage():
            qimg = clipboard.image()
            if not qimg.isNull() and qimg.width() > 0 and qimg.height() > 0:
                try:
                    pil_img = qimage_to_pil(qimg)
                    self.load_new_pil_image(pil_img, source_name="Imagen desde portapapeles", filepath=None)
                    return
                except Exception as e:
                    logger.warning(f"Error al procesar imagen del portapapeles: {e}")

        # 2. Detección de archivo copiado desde el explorador de Windows (ej. clic derecho -> Copiar en .png)
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                local_path = Path(urls[0].toLocalFile())
                valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                if local_path.is_file() and local_path.suffix.lower() in valid_extensions:
                    self.open_image_file(str(local_path))
                    return

        # 3. Si no hay imagen (texto plano, URL de texto, archivo de texto, etc.), feedback discreto en barra de estado
        self._set_status("status_no_clipboard")

    def prompt_save_image(self):
        """Guarda la imagen procesada en disco."""
        if not self.image_state.has_image:
            FramelessPopup(self, tr("popup_no_image_title"), tr("popup_no_image_msg")).exec()
            return

        default_name = "anime_processed.png"
        if self.current_filepath:
            default_name = f"{self.current_filepath.stem}_edit.png"

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Guardar Imagen Como",
            default_name,
            "PNG transparente (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)",
        )
        if not path:
            return

        try:
            img = self.image_state.current_image
            if not img:
                return

            if self.canvas.bg_color_rgb is not None:
                img = apply_solid_background(img, self.canvas.bg_color_rgb)

            ext = Path(path).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                # Si se guarda en JPG, convertir a RGB rellenando fondo blanco
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    bg.save(path, quality=95)
                else:
                    img.convert("RGB").save(path, quality=95)
            elif ext == ".webp":
                img.save(path, quality=95)
            else:
                # PNG por defecto
                img.save(path)

            self._set_status("status_saved", name=Path(path).name)
            FramelessPopup(self, tr("popup_save_success_title"), tr("popup_save_success_msg", path=path)).exec()
        except Exception as e:
            FramelessPopup(self, tr("popup_error_save_title"), tr("popup_error_save_msg", error=str(e))).exec()

    # -------------------------------------------------------------
    # Acciones de Historial (Undo / Redo / Reset)
    # -------------------------------------------------------------
    def perform_undo(self):
        img = self.image_state.undo()
        if img:
            from app.core.sam_manager import get_sam_manager
            get_sam_manager().clear_embedding()
            self.canvas.set_image(img, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self._set_status("status_undo", action=self.image_state.get_status_summary())
            self._update_image_info_label(img)
            self._update_action_states()
            if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.btn_wand.isChecked():
                self._ensure_sam_embedding_async()

    def perform_redo(self):
        img = self.image_state.redo()
        if img:
            from app.core.sam_manager import get_sam_manager
            get_sam_manager().clear_embedding()
            self.canvas.set_image(img, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self._set_status("status_redo", action=self.image_state.get_status_summary())
            self._update_image_info_label(img)
            self._update_action_states()
            if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.btn_wand.isChecked():
                self._ensure_sam_embedding_async()

    def perform_reset_original(self):
        img = self.image_state.reset_to_original()
        if img:
            from app.core.sam_manager import get_sam_manager
            get_sam_manager().clear_embedding()
            self.canvas.set_image(img, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self._set_status("status_reset")
            self._update_image_info_label(img)
            self._update_action_states()
            if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.btn_wand.isChecked():
                self._ensure_sam_embedding_async()

    def push_new_image_state(self, image: Image.Image, description: str, brush_reference: Optional[Image.Image] = None):
        """Registra un nuevo estado procesado en el historial y canvas."""
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        current = self.image_state.push_state(image, description, brush_reference=brush_reference)
        self.canvas.set_image(current, reset_zoom=False)
        self.canvas.set_original_image(self.image_state.current_brush_reference)
        self._set_status_raw(f"Aplicado: {description}")
        self._update_image_info_label(current)
        self._update_action_states()
        if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.btn_wand.isChecked():
            self._ensure_sam_embedding_async()

    # -------------------------------------------------------------
    # Actualizaciones de UI
    # -------------------------------------------------------------
    def _update_action_states(self):
        has_img = self.image_state.has_image
        self.btn_save.setEnabled(has_img)
        self.btn_undo.setEnabled(self.image_state.can_undo())
        self.btn_redo.setEnabled(self.image_state.can_redo())
        self.btn_reset_orig.setEnabled(self.image_state.can_reset() and self.image_state.can_undo())

        self.btn_zoom_in.setEnabled(has_img)
        self.btn_zoom_out.setEnabled(has_img)
        self.btn_fit.setEnabled(has_img)
        self.btn_zoom_100.setEnabled(has_img)

        # Modo comparación antes/después
        can_compare = has_img and self.image_state._current_index > 0
        self.btn_compare.setEnabled(can_compare)
        if not can_compare and self.canvas.comparison_mode:
            self.btn_compare.setChecked(False)
            self.canvas.set_comparison_mode(False)
        elif self.canvas.comparison_mode:
            self.canvas.set_comparison_mode(True, self.image_state.original_image)

        # Sidebar
        self.sidebar.btn_remove_bg.setEnabled(has_img)
        self.sidebar.btn_restore.setEnabled(has_img)
        self.sidebar.set_transparency_available(
            has_transparency=self.image_state.has_transparency,
            can_restore=self.image_state.get_last_transparent_state() is not None,
        )

    def _update_image_info_label(self, img: Image.Image):
        self.lbl_img_info.setText(f"{img.width} × {img.height} px • {img.mode}")

    def _on_mask_changed(self, has_mask: bool):
        self.sidebar.set_has_mask(has_mask)

    def _on_image_edited_directly(self, description: str = "Edición Manual"):
        """Captura la edición directa de píxeles (Pincel / Borrador / Borrar selección) y la guarda en el historial."""
        if self.canvas.current_qimage:
            pil_img = qimage_to_pil(self.canvas.current_qimage)
            self.push_new_image_state(pil_img, description or "Edición Manual")

    def _on_zoom_changed(self, zoom: float):
        percent = int(zoom * 100)
        self.lbl_zoom.setText(f"{percent}%")

    def set_processing(self, is_processing: bool, message: str = ""):
        """Activa o desactiva el estado de carga y barra de progreso."""
        self.progress_bar.setVisible(is_processing)
        self.sidebar.set_busy(is_processing)
        self.btn_open.setEnabled(not is_processing)
        self.btn_save.setEnabled(not is_processing and self.image_state.has_image)
        self.btn_undo.setEnabled(not is_processing and self.image_state.can_undo())
        self.btn_redo.setEnabled(not is_processing and self.image_state.can_redo())
        self.btn_reset_orig.setEnabled(not is_processing and self.image_state.can_reset())
        self.btn_compare.setEnabled(not is_processing and self.image_state.has_image and self.image_state._current_index > 0)

        if is_processing:
            self._set_status_raw(f"⏳ {message}...")
        else:
            if message:
                self._set_status_raw(message)
            else:
                self._set_status("status_ready")

    # -------------------------------------------------------------
    # Inferencia Asíncrona de Modelos (Workers)
    # -------------------------------------------------------------
    def start_remove_bg(self, model_id: str, enable_clothing_protection: Optional[bool] = None):
        if not self.image_state.has_image:
            return
        if enable_clothing_protection is None:
            enable_clothing_protection = getattr(self.sidebar, "is_clothing_protection_enabled", lambda: True)()
        current_img = self.image_state.current_image
        self.set_processing(True, f"Quitando fondo con {model_id}")
        self.active_worker = WorkerThread(
            get_bg_remover().remove_background,
            current_img,
            model_name=model_id,
            enable_clothing_protection=enable_clothing_protection,
            description=f"Quitar fondo ({model_id})",
        )
        self.active_worker.finishedResult.connect(self._on_worker_finished)
        self.active_worker.failed.connect(self._on_worker_failed)
        self.active_worker.start()

    def start_restore(self, scale_mode: str, use_tiling: bool):
        if not self.image_state.has_image:
            return
        current_img = self.image_state.current_image
        desc_map = {"x2": "Escalado 2x", "x4": "Escalado 4x", "enhance_only": "Mejora de nitidez"}
        desc = desc_map.get(scale_mode, f"Escalado {scale_mode}")
        self.set_processing(True, desc)
        self.active_worker = WorkerThread(
            get_image_restorer().restore_image,
            current_img,
            scale_mode=scale_mode,
            use_tiling=use_tiling,
            description=desc,
        )
        self.active_worker.finishedResult.connect(self._on_worker_finished)
        self.active_worker.failed.connect(self._on_worker_failed)
        self.active_worker.start()

    def start_batch_processing(self, input_dir: str, output_dir: str, model_id: str, enable_clothing_protection: bool):
        from app.core.workers import BatchWorkerThread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100) # Se actualizará luego
        self._set_status("status_batch_preparing")
        self.sidebar.btn_run_batch.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.batch_worker = BatchWorkerThread(input_dir, output_dir, model_id, enable_clothing_protection)
        self.batch_worker.progressText.connect(self._set_status_raw)
        
        def update_progress(current, total):
            if total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(current)
                
        self.batch_worker.progress.connect(update_progress)
        
        def on_batch_finished(success_count, total):
            self.progress_bar.setVisible(False)
            self.sidebar._update_batch_btn_state()
            self.btn_open.setEnabled(True)
            self.btn_save.setEnabled(self.image_state.has_image)
            QMessageBox.information(self, "Lote Completado", f"Se procesaron {success_count} de {total} imágenes con éxito.")
            self._set_status("status_batch_finished")
            
        def on_batch_failed(err_msg):
            self.progress_bar.setVisible(False)
            self.sidebar._update_batch_btn_state()
            self.btn_open.setEnabled(True)
            self.btn_save.setEnabled(self.image_state.has_image)
            QMessageBox.critical(self, "Error en lote", f"El procesamiento se detuvo por un error:\n{err_msg}")
            self._set_status("status_batch_error")

        self.batch_worker.finishedResult.connect(on_batch_finished)
        self.batch_worker.failed.connect(on_batch_failed)
        self.batch_worker.start()

    def _on_brush_mode_toggled(self, enabled: bool):
        self.canvas.set_brush_mode(enabled)

    def _on_wand_mode_toggled(self, enabled: bool):
        self.canvas.set_wand_mode(enabled)
        if enabled and self.sidebar.is_wand_ai_mode():
            self._ensure_sam_embedding_async()

    def _on_wand_ai_mode_toggled(self, enabled: bool):
        self.canvas.set_wand_ai_mode(enabled)
        if enabled and self.sidebar.btn_wand.isChecked():
            self._ensure_sam_embedding_async()

    def _ensure_sam_embedding_async(self):
        """Precomputa el embedding de MobileSAM en segundo plano si aún no está listo."""
        if not self.image_state.has_image:
            return
        from app.core.sam_manager import get_sam_manager
        sam_mgr = get_sam_manager()
        if sam_mgr.has_embedding():
            self.sidebar.set_wand_ai_status(tr("wand_ai_ready"))
            return

        self.sidebar.set_wand_ai_status(tr("wand_ai_computing"))
        from app.core.workers import SamEmbeddingWorker
        pil_img = self.image_state.current_image
        self.sam_worker = SamEmbeddingWorker(pil_img)
        self.sam_worker.finishedResult.connect(self._on_sam_embedding_ready)
        self.sam_worker.failed.connect(self._on_sam_embedding_failed)
        self.sam_worker.start()

    def _on_sam_embedding_ready(self, _):
        self.sidebar.set_wand_ai_status(tr("wand_ai_ready"))
        self._set_status("wand_ai_ready")

    def _on_sam_embedding_failed(self, err_msg: str):
        self.sidebar.set_wand_ai_status("Error")
        self._set_status_raw(f"SAM: {err_msg}")

    def update_device_badge(self):
        """Actualiza el badge de la barra de estado y el selector del sidebar según el estado actual."""
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()

        if gpu_mgr.has_orphaned_worker():
            self.update_device_badge_blocked()
            if hasattr(self, "sidebar"):
                self.sidebar.set_device_mode(is_gpu=False, is_blocked=True)
            return

        from app.config import detect_device
        self.device = detect_device()
        is_gpu = "GPU" in self.device.upper()
        device_text = f"{self.device}"
        self.lbl_device.setText(device_text)
        self.lbl_device.setObjectName("deviceBadgeGpu" if is_gpu else "deviceBadgeCpu")
        self.lbl_device.style().unpolish(self.lbl_device)
        self.lbl_device.style().polish(self.lbl_device)

        if hasattr(self, "sidebar"):
            self.sidebar.set_device_mode(is_gpu=is_gpu, is_blocked=False)

    def _toggle_device_mode(self):
        """Alterna el modo de aceleración entre GPU y CPU al hacer clic en el badge."""
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()
        new_mode = not gpu_mgr.is_gpu_enabled()
        self._on_device_mode_changed(new_mode)

    def _on_device_mode_changed(self, is_gpu: bool):
        """Aplica el cambio de modo de dispositivo solicitado por el usuario."""
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()
        success = gpu_mgr.set_gpu_enabled(is_gpu)

        self.update_device_badge()

        if is_gpu and success:
            from app.config import get_gpu_name
            gpu_name = get_gpu_name() or "DirectML"
            clean_name = gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
            self._set_status_raw(f"Aceleración por GPU activada ({clean_name}).")
        elif not is_gpu:
            self._set_status_raw("Modo CPU activado (GPU liberada de VRAM).")
        else:
            self._set_status_raw("No se pudo activar la GPU. Se mantiene modo CPU.")

    def update_device_badge_blocked(self):
        """Actualiza el badge de la barra de estado indicando que la GPU está bloqueada y se fuerza CPU."""
        self.lbl_device.setText("CPU (GPU bloqueada — reinicia la app)")
        self.lbl_device.setObjectName("deviceBadgeBlocked")
        self.lbl_device.style().unpolish(self.lbl_device)
        self.lbl_device.style().polish(self.lbl_device)
        if hasattr(self, "sidebar"):
            self.sidebar.set_device_mode(is_gpu=False, is_blocked=True)

    @Slot(str)
    def _handle_gpu_status(self, message: str):
        """Slot ejecutado en el hilo de la UI al recibir notificaciones del GPUWorkerManager."""
        from app.core.gpu_manager import get_gpu_manager
        if get_gpu_manager().has_orphaned_worker() or "bloqueada" in message or "reiniciar" in message or "no pudo cerrarse" in message:
            self.update_device_badge_blocked()
            self._set_status_raw(message)
        elif "GPU no respondió" in message or "GPU con error" in message:
            self._set_status_raw(message)

    def _on_worker_finished(self, result_image: Image.Image, description: str):
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()
        last_status = gpu_mgr.get_last_status_message()

        if gpu_mgr.has_orphaned_worker():
            self.update_device_badge_blocked()

        if "no pudo cerrarse" in last_status or "reiniciar" in last_status or gpu_mgr.has_orphaned_worker():
            self.set_processing(False, "GPU no respondió y el proceso no pudo cerrarse — se recomienda reiniciar la aplicación")
            gpu_mgr._last_status_message = ""
        elif "GPU no respondió" in last_status:
            self.set_processing(False, "GPU no respondió, se procesó con CPU")
            gpu_mgr._last_status_message = ""
        elif "GPU con error" in last_status:
            self.set_processing(False, "GPU con error, se procesó con CPU")
            gpu_mgr._last_status_message = ""
        else:
            self.set_processing(False, f"{description} completado.")
            
        brush_ref = None
        if "Escalado" in description or "Mejora de nitidez" in description:
            curr_brush = self.image_state.current_brush_reference
            if curr_brush:
                if not self.image_state.has_background_been_removed:
                    brush_ref = result_image.copy()
                else:
                    brush_ref = curr_brush.resize(result_image.size, resample=Image.Resampling.LANCZOS)
                
        self.push_new_image_state(result_image, description, brush_reference=brush_ref)

    def _on_inpaint_finished(self, result_image: Image.Image, description: str):
        self.canvas.reset_mask(emit_signal=True)
        self._on_worker_finished(result_image, description)

    def _on_worker_failed(self, error_message: str):
        self.set_processing(False, "Error en el procesamiento.")
        QMessageBox.critical(
            self,
            "Error al procesar",
            f"Ocurrió un error durante la inferencia:\n{error_message}",
        )

    # -------------------------------------------------------------
    # Herramientas de Edición y Visualización
    # -------------------------------------------------------------
    def _on_toggle_compare(self):
        """Activa o desactiva la vista comparativa antes/después con cortina."""
        if not self.image_state.has_image:
            self.btn_compare.setChecked(False)
            self._set_status("status_open_first")
            return

        # Si aún no se ha aplicado ninguna herramienta (índice 0) o no hay diferencias
        if self.image_state._current_index == 0:
            self.btn_compare.setChecked(False)
            self.canvas.set_comparison_mode(False)
            self._set_status("status_no_changes_compare")
            return

        is_active = self.btn_compare.isChecked()
        if is_active:
            orig = self.image_state.original_image
            self.canvas.set_comparison_mode(True, orig)
            self._set_status("status_compare_active")
        else:
            self.canvas.set_comparison_mode(False)
            self._set_status("status_compare_inactive")

    def perform_autocrop(self):
        """Recorta los márgenes transparentes alrededor del sujeto con 10px de respiro."""
        if not self.image_state.has_image:
            return
        if not self.image_state.has_transparency:
            self._set_status("status_crop_first_remove")
            return

        current_img = self.image_state.current_image
        alpha = current_img.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            self._set_status("status_crop_none")
            return

        pad = 10
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(current_img.width, bbox[2] + pad)
        bottom = min(current_img.height, bbox[3] + pad)
        
        crop_box = (left, top, right, bottom)
        cropped = current_img.crop(crop_box)

        if cropped.size == current_img.size:
            self._set_status("status_crop_already")
            return

        brush_ref = self.image_state.current_brush_reference
        if brush_ref:
            brush_ref = brush_ref.crop(crop_box)

        self.push_new_image_state(cropped, f"Recorte al contenido ({cropped.width}x{cropped.height})", brush_reference=brush_ref)

    def perform_apply_bg_color(self, color_rgb: Tuple[int, int, int]):
        """Compone la imagen actual con transparencia sobre un fondo de color sólido en la capa del canvas."""
        if not self.image_state.has_image:
            return
        
        self.canvas.set_background_color(color_rgb)
        hex_str = f"#{color_rgb[0]:02X}{color_rgb[1]:02X}{color_rgb[2]:02X}"
        self._set_status_raw(f"Capa de fondo visualizada: {hex_str}")

    def perform_restore_transparency(self):
        """Oculta la capa de color sólido restaurando el fondo transparente original."""
        if not self.image_state.has_image:
            return
            
        self.canvas.set_background_color(None)
        self._set_status_raw("Capa de fondo transparente activada")

    def start_detect_subjects(self):
        """Ejecuta detección y segmentación multiobjeto con YOLO11 en segundo plano."""
        if not self.image_state.has_image:
            self._set_status("status_open_first")
            return

        self.sidebar.set_detecting_subjects_state(True)
        self._set_status("detect_status_computing")
        from app.core.workers import YoloDetectionWorker
        self.yolo_worker = YoloDetectionWorker(self.image_state.current_image)
        self.yolo_worker.finishedResult.connect(self._on_yolo_finished)
        self.yolo_worker.failed.connect(self._on_yolo_failed)
        self.yolo_worker.start()

    def _on_yolo_finished(self, subjects: list):
        self.sidebar.set_detecting_subjects_state(False)
        self.sidebar.set_detected_subjects(subjects)
        if subjects:
            self._set_status_raw(f"YOLO11: {len(subjects)} sujeto(s) detectado(s). Selecciona un chip para aislar.")
        else:
            self._set_status("detect_none_found")

    def _on_yolo_failed(self, err_msg: str):
        self.sidebar.set_detecting_subjects_state(False)
        self._set_status_raw(f"Error YOLO: {err_msg}")

    def perform_isolate_subject(self, subject):
        """Aísla el sujeto detectado, dejando transparente todo el resto."""
        if not self.image_state.has_image or subject is None:
            return
        curr = self.image_state.current_image.convert("RGBA")
        curr_arr = np.array(curr)
        mask_bool = subject.mask  # (H, W) bool
        if mask_bool.shape != curr_arr.shape[:2]:
            return

        # Multiplicar canal alfa por la máscara del sujeto
        curr_arr[..., 3] = np.where(mask_bool, curr_arr[..., 3], 0)
        isolated_img = Image.fromarray(curr_arr, "RGBA")
        self.push_new_image_state(isolated_img, f"Aislar {subject.class_name.capitalize()}")

    def perform_select_subject(self, subject):
        """Carga la máscara del sujeto detectado en la selección de hormigas marchantes."""
        if not self.image_state.has_image or subject is None:
            return
        mask_u8 = (subject.mask.astype(np.uint8) * 255)
        self.canvas._set_selection(mask_u8)
        self._set_status_raw(f"Seleccionado: {subject.class_name.capitalize()}")
