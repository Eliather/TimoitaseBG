"""
TimoitaseBG - Ventana principal de la aplicación.
"""
from typing import Optional, Tuple
from pathlib import Path
from PIL import Image

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QIcon, QKeySequence, QShortcut, QAction, QGuiApplication
from PySide6.QtWidgets import (
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
        
        self.btn_ok = QPushButton("Aceptar")
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

    def _load_stylesheet(self):
        self.is_dark_theme = getattr(self, "is_dark_theme", False)
        style_file = "style.qss" if self.is_dark_theme else "style_light.qss"
        style_path = Path(APP_DIR) / "ui" / style_file
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
                
    def _toggle_theme(self):
        self.is_dark_theme = not getattr(self, "is_dark_theme", False)
        if self.is_dark_theme:
            self.btn_theme.setText("☀️")
        else:
            self.btn_theme.setText("🌙")
        self._load_stylesheet()
                
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

        # Título y Badge
        from app.config import DEFAULT_WINDOW_TITLE
        title_label = QLabel(DEFAULT_WINDOW_TITLE)
        title_label.setObjectName("appNameLabel")
        layout.addWidget(title_label)

        badge_label = QLabel("LOCAL AI")
        badge_label.setObjectName("appBadgeLabel")
        layout.addWidget(badge_label)

        layout.addSpacing(12)

        self.btn_theme = QPushButton("🌙")
        self.btn_theme.setObjectName("compactIconButton")
        self.btn_theme.setFixedSize(30, 30)
        self.btn_theme.setToolTip("Cambiar entre tema Claro/Oscuro")
        self.btn_theme.clicked.connect(self._toggle_theme)
        layout.addWidget(self.btn_theme)

        layout.addSpacing(12)

        # Botón Abrir Imagen
        self.btn_open = QPushButton("Abrir")
        self.btn_open.setObjectName("actionButton")
        self.btn_open.setFixedHeight(32)
        self.btn_open.setToolTip("Abrir imagen desde el explorador (Ctrl+O) o pegar con Ctrl+V")
        self.btn_open.clicked.connect(self.prompt_open_image)
        layout.addWidget(self.btn_open)

        # Botón Guardar Como
        self.btn_save = QPushButton("Guardar")
        self.btn_save.setObjectName("actionButton")
        self.btn_save.setFixedHeight(32)
        self.btn_save.setToolTip("Guardar imagen procesada en disco (Ctrl+S)")
        self.btn_save.clicked.connect(self.prompt_save_image)
        layout.addWidget(self.btn_save)

        layout.addWidget(self._create_separator())

        # Botones compactos de historial (↶, ↷, ↺) con tooltip
        self.btn_undo = QPushButton("↶")
        self.btn_undo.setObjectName("compactIconButton")
        self.btn_undo.setFixedSize(34, 34)
        self.btn_undo.setToolTip("Deshacer última acción (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.perform_undo)
        layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↷")
        self.btn_redo.setObjectName("compactIconButton")
        self.btn_redo.setFixedSize(34, 34)
        self.btn_redo.setToolTip("Rehacer acción (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.perform_redo)
        layout.addWidget(self.btn_redo)

        self.btn_reset_orig = QPushButton("↺")
        self.btn_reset_orig.setObjectName("compactIconButton")
        self.btn_reset_orig.setFixedSize(34, 34)
        self.btn_reset_orig.setToolTip("Recuperar Original (Restaura la imagen inicial)")
        self.btn_reset_orig.clicked.connect(self.perform_reset_original)
        layout.addWidget(self.btn_reset_orig)

        layout.addStretch()

        # Controles de Zoom
        self.btn_zoom_out = QPushButton("－")
        self.btn_zoom_out.setObjectName("zoomButton")
        self.btn_zoom_out.setFixedSize(30, 30)
        self.btn_zoom_out.setToolTip("Reducir zoom")
        self.btn_zoom_out.clicked.connect(lambda: self.canvas.set_zoom_delta(1.0 / 1.2))
        layout.addWidget(self.btn_zoom_out)

        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setObjectName("zoomLabel")
        self.lbl_zoom.setFixedWidth(50)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_zoom)

        self.btn_zoom_in = QPushButton("＋")
        self.btn_zoom_in.setObjectName("zoomButton")
        self.btn_zoom_in.setFixedSize(30, 30)
        self.btn_zoom_in.setToolTip("Aumentar zoom")
        self.btn_zoom_in.clicked.connect(lambda: self.canvas.set_zoom_delta(1.2))
        layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton("Ajustar")
        self.btn_fit.setObjectName("zoomButton")
        self.btn_fit.setFixedHeight(30)
        self.btn_fit.setToolTip("Ajustar imagen completa a la ventana")
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)
        layout.addWidget(self.btn_fit)

        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setObjectName("zoomButton")
        self.btn_zoom_100.setFixedHeight(30)
        self.btn_zoom_100.setToolTip("Zoom al 100%")
        self.btn_zoom_100.clicked.connect(self.canvas.reset_zoom_100)
        layout.addWidget(self.btn_zoom_100)

        layout.addWidget(self._create_separator())

        # Botón Modo Comparar (Antes / Después con cortina)
        self.btn_compare = QPushButton("Comparar")
        self.btn_compare.setObjectName("actionButton")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setFixedHeight(32)
        self.btn_compare.setToolTip("Comparar antes/después con cortina interactiva")
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
        layout.setSpacing(14)

        # Estado del documento / imagen
        self.lbl_status = QLabel("Listo. Arrastra una imagen o pulsa 'Abrir Imagen'.")
        self.lbl_status.setObjectName("statusText")
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
        self.btn_kofi = QPushButton("☕ Apoyar a Eliather")
        self.btn_kofi.setObjectName("kofiButton")
        self.btn_kofi.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_kofi.setToolTip("Cómprame un café en Ko-fi para apoyar el desarrollo")
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

    # -------------------------------------------------------------
    # Atajos de Teclado
    # -------------------------------------------------------------
    def _setup_shortcuts(self):
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
        self.btn_compare.setChecked(False)
        self.canvas.set_comparison_mode(False)
        self.canvas.set_image(loaded, reset_zoom=True)
        self.canvas.set_original_image(self.image_state.current_brush_reference)

        self.lbl_status.setText(f"Cargada: {source_name}")
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
        self.lbl_status.setText("El portapapeles no contiene una imagen.")

    def prompt_save_image(self):
        """Guarda la imagen procesada en disco."""
        if not self.image_state.has_image:
            FramelessPopup(self, "Sin imagen", "No hay ninguna imagen activa para guardar.").exec()
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

            self.lbl_status.setText(f"Imagen guardada en: {Path(path).name}")
            FramelessPopup(self, "Guardado con éxito", f"La imagen se ha guardado correctamente en:\n{path}").exec()
        except Exception as e:
            FramelessPopup(self, "Error al guardar", f"No se pudo guardar la imagen:\n{e}").exec()

    # -------------------------------------------------------------
    # Acciones de Historial (Undo / Redo / Reset)
    # -------------------------------------------------------------
    def perform_undo(self):
        img = self.image_state.undo()
        if img:
            self.canvas.set_image(img, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self.lbl_status.setText(f"Deshecho: {self.image_state.get_status_summary()}")
            self._update_image_info_label(img)
            self._update_action_states()

    def perform_redo(self):
        img = self.image_state.redo()
        if img:
            self.canvas.set_image(img, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self.lbl_status.setText(f"Rehecho: {self.image_state.get_status_summary()}")
            self._update_image_info_label(img)
            self._update_action_states()

    def perform_reset_original(self):
        img = self.image_state.reset_to_original()
        if img:
            self.canvas.set_image(img, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self.lbl_status.setText("Imagen restaurada al estado original.")
            self._update_image_info_label(img)
            self._update_action_states()

    def push_new_image_state(self, image: Image.Image, description: str, brush_reference: Optional[Image.Image] = None):
        """Registra un nuevo estado procesado en el historial y canvas."""
        current = self.image_state.push_state(image, description, brush_reference=brush_reference)
        self.canvas.set_image(current, reset_zoom=False)
        self.canvas.set_original_image(self.image_state.current_brush_reference)
        self.lbl_status.setText(f"Aplicado: {description}")
        self._update_image_info_label(current)
        self._update_action_states()

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
        self.sidebar.btn_toggle_brush.setEnabled(has_img)
        self.sidebar.set_transparency_available(
            has_transparency=self.image_state.has_transparency,
            can_restore=self.image_state.get_last_transparent_state() is not None,
        )

    def _update_image_info_label(self, img: Image.Image):
        self.lbl_img_info.setText(f"{img.width} × {img.height} px • {img.mode}")

    def _on_mask_changed(self, has_mask: bool):
        self.sidebar.set_has_mask(has_mask)

    def _on_image_edited_directly(self):
        """Captura la edición directa de píxeles (ej. Pincel / Borrador) y lo guarda en el historial."""
        if self.canvas.current_qimage:
            pil_img = qimage_to_pil(self.canvas.current_qimage)
            self.push_new_image_state(pil_img, "Edición Manual")

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
            self.lbl_status.setText(f"⏳ {message}...")
        else:
            self.lbl_status.setText(message or "Listo")

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
        self.lbl_status.setText("Preparando lote...")
        self.sidebar.btn_run_batch.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.batch_worker = BatchWorkerThread(input_dir, output_dir, model_id, enable_clothing_protection)
        self.batch_worker.progressText.connect(self.lbl_status.setText)
        
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
            self.lbl_status.setText("Procesamiento por lote finalizado.")
            
        def on_batch_failed(err_msg):
            self.progress_bar.setVisible(False)
            self.sidebar._update_batch_btn_state()
            self.btn_open.setEnabled(True)
            self.btn_save.setEnabled(self.image_state.has_image)
            QMessageBox.critical(self, "Error en lote", f"El procesamiento se detuvo por un error:\n{err_msg}")
            self.lbl_status.setText("Error en lote.")

        self.batch_worker.finishedResult.connect(on_batch_finished)
        self.batch_worker.failed.connect(on_batch_failed)
        self.batch_worker.start()

    def _on_brush_mode_toggled(self, enabled: bool):
        self.canvas.set_brush_mode(enabled)

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
        gpu_icon = "⚡ " if is_gpu else "💻 "
        device_text = f"{gpu_icon}{self.device}"
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
            self.lbl_status.setText(f"⚡ Aceleración por GPU activada ({clean_name}).")
        elif not is_gpu:
            self.lbl_status.setText("💻 Modo CPU activado (GPU liberada de VRAM).")
        else:
            self.lbl_status.setText("⚠️ No se pudo activar la GPU. Se mantiene modo CPU.")

    def update_device_badge_blocked(self):
        """Actualiza el badge de la barra de estado indicando que la GPU está bloqueada y se fuerza CPU."""
        self.lbl_device.setText("⚠️ CPU (GPU bloqueada — reinicia la app)")
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
            self.lbl_status.setText("⚠️ " + message)
        elif "GPU no respondió" in message or "GPU con error" in message:
            self.lbl_status.setText("⚠️ " + message)

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
            self.lbl_status.setText("Abre una imagen primero.")
            return

        # Si aún no se ha aplicado ninguna herramienta (índice 0) o no hay diferencias
        if self.image_state._current_index == 0:
            self.btn_compare.setChecked(False)
            self.canvas.set_comparison_mode(False)
            self.lbl_status.setText("No hay cambios que comparar")
            return

        is_active = self.btn_compare.isChecked()
        if is_active:
            orig = self.image_state.original_image
            self.canvas.set_comparison_mode(True, orig)
            self.lbl_status.setText("Modo comparación activo: arrastra la barra divisoria para comparar con el original.")
        else:
            self.canvas.set_comparison_mode(False)
            self.lbl_status.setText("Modo comparación desactivado.")

    def perform_autocrop(self):
        """Recorta los márgenes transparentes alrededor del sujeto con 10px de respiro."""
        if not self.image_state.has_image:
            return
        if not self.image_state.has_transparency:
            self.lbl_status.setText("Aplica primero Quitar Fondo para recortar al contenido.")
            return

        current_img = self.image_state.current_image
        alpha = current_img.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            self.lbl_status.setText("No se detectó contenido para recortar.")
            return

        pad = 10
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(current_img.width, bbox[2] + pad)
        bottom = min(current_img.height, bbox[3] + pad)
        
        crop_box = (left, top, right, bottom)
        cropped = current_img.crop(crop_box)

        if cropped.size == current_img.size:
            self.lbl_status.setText("La imagen ya está ajustada al contenido.")
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
        self.lbl_status.setText(f"Capa de fondo visualizada: {hex_str}")

    def perform_restore_transparency(self):
        """Oculta la capa de color sólido restaurando el fondo transparente original."""
        if not self.image_state.has_image:
            return
            
        self.canvas.set_background_color(None)
        self.lbl_status.setText("Capa de fondo transparente activada")
