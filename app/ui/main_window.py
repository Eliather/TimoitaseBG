"""
TimoitaseBG - Ventana principal de la aplicación.
"""
import io
import logging
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QMimeData, QByteArray
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QFrame,
)

from app.i18n import tr, set_language, get_current_language
from app.config import (
    DEFAULT_WINDOW_TITLE,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    detect_device,
    APP_DIR,
)
from app.controllers.ai_coordinator import AICoordinator
from app.controllers.document_controller import DocumentController
from app.core.image_state import ImageState, SequenceSession
from app.core.image_utils import pil_to_qimage, qimage_to_pil
from app.core.workers import WorkerThread
from app.ui.canvas_widget import CanvasWidget
from app.ui.components import (
    FramelessPopup,
    TopBar,
    SequenceBar,
    StatusBar,
    register_shortcuts,
)
from app.ui.toolbar import ToolSidebar
from app.ui.shortcuts_dialog import ShortcutsDialog

logger = logging.getLogger("TimoitaseBG.MainWindow")


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
        self.resize(max(1200, WINDOW_MIN_WIDTH), max(780, WINDOW_MIN_HEIGHT))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Controladores desacoplados de documento/secuencias e inferencia IA
        self.doc = DocumentController()
        self.ai = AICoordinator()

        self.gif_timer = QTimer(self)
        self.gif_timer.timeout.connect(self._on_gif_timer_tick)

        # Dispositivo detectado
        self.device = detect_device()

        # Configurar UI
        self._init_ui()
        self._setup_shortcuts()
        self._update_action_states()
        self.update_device_badge()

        # Conectar notificaciones asíncronas de GPU al hilo de la UI
        self.gpuStatusNotified.connect(self._handle_gpu_status)
        from app.core.gpu_manager import get_gpu_manager
        get_gpu_manager().set_status_callback(lambda msg: self.gpuStatusNotified.emit(msg))

        # Cargar estilos nativos (Modo Oscuro)
        self._load_stylesheet()

    # Propiedades de compatibilidad hacia los controladores
    @property
    def image_state(self) -> ImageState:
        return self.doc.image_state

    @image_state.setter
    def image_state(self, state: ImageState) -> None:
        self.doc.image_state = state

    @property
    def sequence_session(self) -> Optional[SequenceSession]:
        return self.doc.sequence_session

    @sequence_session.setter
    def sequence_session(self, session: Optional[SequenceSession]) -> None:
        self.doc.sequence_session = session

    @property
    def current_filepath(self) -> Optional[Path]:
        return self.doc.current_filepath

    @current_filepath.setter
    def current_filepath(self, path: Optional[Path]) -> None:
        self.doc.current_filepath = path

    @property
    def active_worker(self):
        return self.ai.active_worker

    @active_worker.setter
    def active_worker(self, w):
        self.ai.active_worker = w

    @property
    def batch_worker(self):
        return self.ai.batch_worker

    @batch_worker.setter
    def batch_worker(self, w):
        self.ai.batch_worker = w

    @property
    def sam_worker(self):
        return self.ai.sam_worker

    @sam_worker.setter
    def sam_worker(self, w):
        self.ai.sam_worker = w

    @property
    def yolo_worker(self):
        return self.ai.yolo_worker

    @yolo_worker.setter
    def yolo_worker(self, w):
        self.ai.yolo_worker = w

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _set_status(self, key: str, **kwargs):
        """Asigna texto a la barra de estado y guarda la clave para re-traducción en vivo."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_status(key, **kwargs)
        elif hasattr(self, "lbl_status"):
            self.lbl_status.setText(tr(key, **kwargs))

    def _set_status_raw(self, text: str):
        """Asigna texto directo cuando proviene de un mensaje dinámico externo."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_status_raw(text)
        elif hasattr(self, "lbl_status"):
            self.lbl_status.setText(text)

    def _on_lang_changed(self, index):
        code = self.combo_lang.itemData(index)
        if code and code != get_current_language():
            set_language(code)
            self.retranslate_ui()

    def retranslate_ui(self):
        """Actualiza dinámicamente los textos de toda la interfaz al cambiar de idioma."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.retranslate_ui()
        if hasattr(self, "top_bar"):
            self.top_bar.retranslate_ui()
        if hasattr(self, "sidebar") and hasattr(self, "sidebar") and hasattr(self.sidebar, "retranslate_ui"):
            self.sidebar.retranslate_ui()
        if hasattr(self, "gif_bar") and hasattr(self.gif_bar, "retranslate_ui"):
            self.gif_bar.retranslate_ui()
        if hasattr(self, "canvas"):
            if hasattr(self.canvas, "retranslate_ui"):
                self.canvas.retranslate_ui()
            self.canvas.update()
        self.update_device_badge()

    def _load_stylesheet(self):
        self.is_dark_theme = getattr(self, "is_dark_theme", True)
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
        self.is_dark_theme = not getattr(self, "is_dark_theme", True)
        self._load_stylesheet()

    def _update_theme_icons(self):
        """Aplica iconos vectoriales nítidos y adaptados al contraste del tema activo."""
        if hasattr(self, "top_bar"):
            self.top_bar.apply_theme(self.is_dark_theme)
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.apply_theme(self.is_dark_theme)
        if hasattr(self, "sidebar") and hasattr(self.sidebar, "update_theme_icons"):
            self.sidebar.update_theme_icons(self.is_dark_theme)
        if hasattr(self, "canvas") and hasattr(self.canvas, "update_theme"):
            self.canvas.update_theme(self.is_dark_theme)
        if hasattr(self, "gif_bar") and hasattr(self.gif_bar, "apply_theme"):
            is_playing = hasattr(self, "gif_timer") and self.gif_timer.isActive()
            self.gif_bar.apply_theme(self.is_dark_theme, is_playing=is_playing)
                
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
        self.canvas.imageDropped.connect(lambda img: self.load_new_pil_image(img, "Imagen arrastrada"))
        self.canvas.browseRequested.connect(self.prompt_open_image)
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
        self.sidebar.inpaintRequested.connect(self.perform_inpaint_selection)
        self.sidebar.edgeRefineRequested.connect(self.perform_edge_refinement)
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
        self.sidebar.applyEffectsRequested.connect(self.perform_apply_effects)
        self.sidebar.transformRequested.connect(self.perform_transform)
        self.sidebar.cropModeToggled.connect(self.on_crop_mode_toggled)
        self.sidebar.cropApplyRequested.connect(self.perform_manual_crop)
        self.sidebar.deviceModeChanged.connect(self._on_device_mode_changed)
        self.sidebar.batchRequested.connect(self.start_batch_processing)
        self.sidebar.detectSubjectsRequested.connect(self.start_detect_subjects)
        self.sidebar.isolateSubjectRequested.connect(self.perform_isolate_subject)
        self.sidebar.selectSubjectRequested.connect(self.perform_select_subject)

        # 2. Barra Superior (Top Bar)
        top_bar = self._create_top_bar()
        root_layout.addWidget(top_bar)

        # 3. Área central: Canvas (con HUD de paginación GIF) + Sidebar
        main_content = QWidget()
        content_layout = QHBoxLayout(main_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        canvas_container = QWidget()
        canvas_vbox = QVBoxLayout(canvas_container)
        canvas_vbox.setContentsMargins(0, 0, 0, 0)
        canvas_vbox.setSpacing(0)
        canvas_vbox.addWidget(self.canvas, stretch=1)

        self.gif_bar = self._create_gif_pagination_bar()
        self.gif_bar.setVisible(False)
        canvas_vbox.addWidget(self.gif_bar)

        content_layout.addWidget(canvas_container, stretch=1)
        content_layout.addWidget(self.sidebar)

        root_layout.addWidget(main_content, stretch=1)

        # 4. Barra Inferior de Estado
        self.status_bar = self._create_status_bar()
        root_layout.addWidget(self.status_bar)

    # -------------------------------------------------------------
    # Barra Superior
    # -------------------------------------------------------------
    def _create_top_bar(self) -> QWidget:
        top_bar = TopBar(self)
        self.top_bar = top_bar
        top_bar.themeToggled.connect(self._toggle_theme)
        top_bar.modelsRequested.connect(self.open_model_manager)
        top_bar.languageChanged.connect(self._on_lang_changed)
        top_bar.openRequested.connect(self.prompt_open_image)
        top_bar.saveRequested.connect(self.prompt_save_image)
        top_bar.undoRequested.connect(self.perform_undo)
        top_bar.redoRequested.connect(self.perform_redo)
        top_bar.resetRequested.connect(self.perform_reset_original)
        top_bar.zoomOutRequested.connect(lambda: self.canvas.set_zoom_delta(1.0 / 1.2))
        top_bar.zoomInRequested.connect(lambda: self.canvas.set_zoom_delta(1.2))
        top_bar.fitRequested.connect(self.canvas.fit_to_view)
        top_bar.zoom100Requested.connect(self.canvas.reset_zoom_100)
        top_bar.compareToggled.connect(self._on_toggle_compare)
        top_bar.minimizeRequested.connect(self.showMinimized)
        top_bar.maximizeRequested.connect(self._toggle_maximize)
        top_bar.closeRequested.connect(self.close)

        # Alias retrocompatibles con controles de la barra superior
        self.version_badge = top_bar.version_badge
        self.btn_theme = top_bar.btn_theme
        self.combo_lang = top_bar.combo_lang
        self.btn_open = top_bar.btn_open
        self.btn_save = top_bar.btn_save
        self.btn_undo = top_bar.btn_undo
        self.btn_redo = top_bar.btn_redo
        self.btn_reset_orig = top_bar.btn_reset_orig
        self.btn_zoom_out = top_bar.btn_zoom_out
        self.lbl_zoom = top_bar.lbl_zoom
        self.btn_zoom_in = top_bar.btn_zoom_in
        self.btn_fit = top_bar.btn_fit
        self.btn_zoom_100 = top_bar.btn_zoom_100
        self.btn_compare = top_bar.btn_compare
        self.btn_min = top_bar.btn_min
        self.btn_max = top_bar.btn_max
        self.btn_close = top_bar.btn_close
        return top_bar
    # -------------------------------------------------------------
    # Barra Inferior de Estado
    # -------------------------------------------------------------
    def _create_status_bar(self) -> QWidget:
        bar = StatusBar(size_grip_target=self, parent=self)
        self.status_bar_widget = bar
        bar.shortcutsRequested.connect(self.show_shortcuts_modal)
        bar.cancelRequested.connect(self.cancel_active_processing)
        bar.deviceToggleRequested.connect(self._toggle_device_mode)
        bar.kofiRequested.connect(lambda: webbrowser.open("https://ko-fi.com/eliather"))

        self.btn_shortcuts = bar.btn_shortcuts
        self.lbl_status = bar.lbl_status
        self.lbl_img_info = bar.lbl_img_info
        self.progress_bar = bar.progress_bar
        self.btn_cancel = bar.btn_cancel
        self.lbl_device = bar.lbl_device
        self.btn_kofi = bar.btn_kofi
        self.size_grip = bar.size_grip
        return bar

    def show_shortcuts_modal(self):
        """Abre el diálogo modal con la lista interactiva de atajos de teclado y gestos."""
        dialog = ShortcutsDialog(self, is_dark_theme=getattr(self, "is_dark_theme", True))
        dialog.exec()

    def open_model_manager(self):
        """Abre el gestor visual de modelos ONNX locales (descargas, INT8 y SHA-256)."""
        from app.ui.dialogs import ModelManagerDialog

        dialog = ModelManagerDialog(self, is_dark_theme=getattr(self, "is_dark_theme", True))
        dialog.modelsChanged.connect(self.ai.unload_sessions)
        dialog.exec()

    def _setup_shortcuts(self):
        self._shortcuts = register_shortcuts(self)

    def _create_gif_pagination_bar(self) -> QWidget:
        """Crea la barra de navegación y reproducción por fotogramas para secuencias GIF/Video."""
        bar = SequenceBar(self)
        bar.apply_theme(getattr(self, "is_dark_theme", True), is_playing=False)
        bar.firstRequested.connect(lambda: self._navigate_gif_frame(0))
        bar.prevRequested.connect(lambda: self._navigate_gif_frame(self.sequence_session.current_index - 1 if self.sequence_session else 0))
        bar.playToggled.connect(self._toggle_gif_playback)
        bar.nextRequested.connect(lambda: self._navigate_gif_frame(self.sequence_session.current_index + 1 if self.sequence_session else 0))
        bar.lastRequested.connect(lambda: self._navigate_gif_frame(self.sequence_session.total_frames - 1 if self.sequence_session else 0))
        bar.trimRangeChanged.connect(self._on_sequence_trim_changed)
        bar.fpsChanged.connect(self._on_sequence_fps_changed)
        bar.trackSequenceRequested.connect(self.start_track_sequence)

        self.btn_gif_first = bar.btn_gif_first
        self.btn_gif_prev = bar.btn_gif_prev
        self.btn_gif_play = bar.btn_gif_play
        self.btn_gif_next = bar.btn_gif_next
        self.btn_gif_last = bar.btn_gif_last
        self.lbl_gif_pagination = bar.lbl_gif_pagination
        return bar

    def _navigate_gif_frame(self, new_index: int):
        """Navega a un fotograma específico y conmuta el ImageState activo."""
        if not self.sequence_session or self.sequence_session.total_frames <= 0:
            return
        idx, total, cur_dur, curr_img = self.doc.select_sequence_frame(new_index)
        is_playing = self.gif_timer.isActive()

        if curr_img:
            self.canvas.set_image(curr_img, reset_zoom=False)
            if not is_playing:
                self.canvas.set_original_image(self.image_state.current_brush_reference)
                self._update_image_info_label(curr_img)
        if not is_playing:
            self._update_action_states()

        self.gif_bar.update_frame_info(idx, total, cur_dur)

    def _toggle_gif_playback(self):
        """Alterna entre reproducción en bucle y pausa para el GIF animado."""
        if not self.sequence_session or self.sequence_session.total_frames <= 1:
            return
        if self.gif_timer.isActive():
            self.gif_timer.stop()
            self.gif_bar.set_playing(False)
            curr_img = self.image_state.current_image
            if curr_img:
                self.canvas.set_original_image(self.image_state.current_brush_reference)
                self._update_image_info_label(curr_img)
            self._update_action_states()
        else:
            dur = self.sequence_session.durations[self.sequence_session.current_index] if self.sequence_session.durations else 80
            self.gif_timer.start(dur)
            self.gif_bar.set_playing(True)

    def _on_gif_timer_tick(self):
        """Avanza al siguiente fotograma respetando recorte temporal y FPS objetivo."""
        if not self.sequence_session or self.sequence_session.total_frames <= 0:
            self.gif_timer.stop()
            return
        s = self.sequence_session.trim_start
        e = self.sequence_session.trim_end
        nxt = self.sequence_session.current_index + 1
        if nxt > e or nxt < s:
            nxt = s
        self._navigate_gif_frame(nxt)
        if abs(self.sequence_session.target_fps - self.sequence_session.fps) > 1e-2:
            dur = max(15, int(round(1000.0 / max(1.0, self.sequence_session.target_fps))))
        else:
            dur = self.sequence_session.durations[nxt] if self.sequence_session.durations else 80
        self.gif_timer.setInterval(dur)

    def _on_sequence_trim_changed(self, start: int, end: int):
        if not self.sequence_session:
            return
        s, e = self.sequence_session.set_trim_range(start, end)
        if self.sequence_session.current_index < s or self.sequence_session.current_index > e:
            self._navigate_gif_frame(s)

    def _on_sequence_fps_changed(self, fps: float):
        if not self.sequence_session:
            return
        self.sequence_session.set_target_fps(fps)

    def start_track_sequence(self):
        """Rastrea automáticamente el sujeto activo a lo largo de los fotogramas de la secuencia."""
        if not self.doc.has_sequence or self.ai.is_active_running():
            return
        if self.gif_timer.isActive():
            self.gif_timer.stop()
            self.gif_bar.set_playing(False)

        frames = self.sequence_session.get_all_current_frames()
        seed_idx = self.sequence_session.current_index
        seed_subject = self.sidebar.panel_bg.get_selected_subject() if hasattr(self.sidebar, "panel_bg") else None
        seed_mask = self.canvas.active_selection if self.canvas.has_selection() else None

        self.set_processing(True, tr("seq_tracking_status"))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.ai.start_track_sequence(
            frames=frames,
            seed_frame_idx=seed_idx,
            seed_subject=seed_subject,
            seed_mask=seed_mask,
            trim_start=self.sequence_session.trim_start,
            trim_end=self.sequence_session.trim_end,
            on_progress=lambda cur, tot: self.progress_bar.setValue(cur),
            on_progress_text=self._set_status_raw,
            on_finished=self._on_gif_worker_finished,
            on_failed=self._on_worker_failed,
            on_cancelled=self._on_worker_cancelled,
        )

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
        """Abre un diálogo para seleccionar archivo de imagen, animación o video."""
        video_globs = "*.mp4 *.mov *.webm *.mkv *.avi *.m4v *.wmv *.flv *.mpeg *.mpg *.ogv *.3gp *.ts"
        image_globs = "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.apng"
        filters = (
            f"Imágenes y Videos ({image_globs} {video_globs});;"
            f"Imágenes y Animaciones ({image_globs});;"
            f"Videos ({video_globs});;"
            "Todos los archivos (*.*)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen o Video",
            "",
            filters,
        )
        if path:
            self.open_image_file(path)

    def _reset_workspace_ui(self, loaded_img: Optional[Image.Image] = None, reset_zoom: bool = True):
        """Limpia estado transitorio de IA, selección, comparación y actualiza canvas tras abrir un nuevo archivo."""
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        if hasattr(self, "sidebar"):
            self.sidebar.set_detected_subjects([])
            if self.sidebar.is_wand_ai_mode() and self.sidebar.is_wand_tool_active():
                self._ensure_sam_embedding_async()
        self.btn_compare.setChecked(False)
        self.canvas.set_comparison_mode(False)
        self.canvas.clear_selection()
        if loaded_img is not None:
            self.canvas.set_image(loaded_img, reset_zoom=reset_zoom)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self._update_image_info_label(loaded_img)
        self._update_action_states()

    def load_new_pil_image(self, pil_img: Image.Image, source_name: str, filepath: Optional[Path] = None):
        """
        Carga una nueva imagen PIL en el estado inmutable, reinicia el historial y actualiza el canvas y la UI.
        Reutilizado tanto para 'Abrir Imagen' como para 'Pegar desde el portapapeles (Ctrl+V)'.
        """
        loaded = self.doc.load_pil_image(pil_img, filepath=filepath, clear_seq=False)
        if loaded is None:
            return
        self._reset_workspace_ui(loaded, reset_zoom=True)
        self._set_status("status_loaded", name=source_name)

    def open_image_file(self, file_path: str):
        """Carga una imagen estática, animación (.gif, .webp, .apng) o video (.mp4, .mov, .webm, .mkv, etc.) en el estado y canvas."""
        try:
            def _progress_status(msg: str):
                self._set_status_raw(msg)
                QGuiApplication.processEvents()

            info = self.doc.open_media(file_path, status_cb=_progress_status)
            kind = info["kind"]
            if kind == "missing":
                return

            if self.gif_timer.isActive():
                self.gif_timer.stop()

            path = info["path"]
            if kind == "video":
                if hasattr(self, "gif_bar"):
                    self.gif_bar.configure_sequence(self.sequence_session.total_frames, self.sequence_session.fps)
                    self.gif_bar.setVisible(info["frames_count"] > 1)
                self._navigate_gif_frame(0)
                self._reset_workspace_ui(info["image"], reset_zoom=True)
                self._set_status_raw(
                    f"🎬 Video {info['fmt_name']} cargado: {path.name} ({info['frames_count']} fotogramas @ {info['fps']:.1f} FPS)"
                )
                return

            if kind == "animation":
                if hasattr(self, "gif_bar"):
                    self.gif_bar.configure_sequence(self.sequence_session.total_frames, self.sequence_session.fps)
                    self.gif_bar.setVisible(True)
                self._navigate_gif_frame(0)
                self._reset_workspace_ui(info["image"], reset_zoom=True)
                self._set_status_raw(f"{info['fmt_name']} animado cargado: {path.name} ({info['frames_count']} fotogramas)")
                return

            if hasattr(self, "gif_bar"):
                self.gif_bar.setVisible(False)
            self._reset_workspace_ui(info["image"], reset_zoom=True)
            self._set_status("status_loaded", name=path.name)
        except Exception as e:
            QMessageBox.critical(self, "Error al abrir archivo", f"No se pudo cargar el archivo:\n{e}")

    def copy_to_clipboard(self):
        """
        Copia la imagen actual al portapapeles con canal alfa transparente preservado (Ctrl+C).
        """
        built = self.doc.build_clipboard_png(bg_color_rgb=self.canvas.bg_color_rgb)
        if built is None:
            self._set_status("status_open_first")
            return
        png_bytes, img = built

        mime_data = QMimeData()
        byte_array = QByteArray(png_bytes)
        mime_data.setData("image/png", byte_array)
        mime_data.setData("application/x-qt-image", byte_array)
        mime_data.setImageData(pil_to_qimage(img))

        clipboard = QGuiApplication.clipboard()
        clipboard.setMimeData(mime_data)

        self._set_status("status_copied_clipboard")

    def paste_from_clipboard(self):
        """
        Pega una imagen o video desde el portapapeles con Ctrl+V.
        Detecta estrictamente contenido multimedia:
        - Stream PNG directo en el portapapeles (image/png) preservando canal alfa completo.
        - Imagen directa en el portapapeles (bitmap / QImage de navegador, capturador, editor).
        - Archivo de imagen o video copiado desde el Explorador de Windows.
        Si el portapapeles contiene texto plano, URL de texto o archivos no soportados, no realiza ninguna acción.
        """
        clipboard = QGuiApplication.clipboard()
        mime_data = clipboard.mimeData()

        # 1. Detección de stream PNG en memoria (máxima fidelidad de alfa 8-bit)
        if mime_data.hasFormat("image/png"):
            try:
                png_bytes = bytes(mime_data.data("image/png"))
                if png_bytes:
                    pil_img = Image.open(io.BytesIO(png_bytes))
                    if self.gif_timer.isActive():
                        self.gif_timer.stop()
                    self.sequence_session = None
                    if hasattr(self, "gif_bar"):
                        self.gif_bar.setVisible(False)
                    self.load_new_pil_image(pil_img, source_name="Imagen desde portapapeles", filepath=None)
                    return
            except Exception as e:
                logger.warning(f"Error al leer stream image/png del portapapeles: {e}")

        # 2. Detección de mapa de bits (QImage) en portapapeles (ej. Copiar imagen en navegador)
        if mime_data.hasImage():
            qimg = clipboard.image()
            if not qimg.isNull() and qimg.width() > 0 and qimg.height() > 0:
                try:
                    if self.gif_timer.isActive():
                        self.gif_timer.stop()
                    self.sequence_session = None
                    if hasattr(self, "gif_bar"):
                        self.gif_bar.setVisible(False)
                    pil_img = qimage_to_pil(qimg)
                    self.load_new_pil_image(pil_img, source_name="Imagen desde portapapeles", filepath=None)
                    return
                except Exception as e:
                    logger.warning(f"Error al procesar imagen del portapapeles: {e}")

        # 3. Detección de archivo copiado desde el explorador de Windows (ej. clic derecho -> Copiar en imagen o video)
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                from app.core.video_processor import VIDEO_EXTENSIONS
                local_path = Path(urls[0].toLocalFile())
                valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".apng", *VIDEO_EXTENSIONS}
                if local_path.is_file() and local_path.suffix.lower() in valid_extensions:
                    self.open_image_file(str(local_path))
                    return

        # 4. Si no hay imagen (texto plano, URL de texto, archivo de texto, etc.), feedback discreto en barra de estado
        self._set_status("status_no_clipboard")

    def prompt_save_image(self):
        """Guarda la imagen procesada, secuencia animada o video en formatos libres/gratuitos en disco."""
        if not self.image_state.has_image:
            FramelessPopup(self, tr("popup_no_image_title"), tr("popup_no_image_msg")).exec()
            return

        is_animated_seq = bool(self.sequence_session and self.sequence_session.total_frames > 1)
        if is_animated_seq:
            from app.core.video_processor import VIDEO_EXTENSIONS

            orig_ext = self.current_filepath.suffix.lower() if self.current_filepath else ".webm"
            is_vid = bool(getattr(self.sequence_session, "is_video", False) or orig_ext in VIDEO_EXTENSIONS)

            if orig_ext in (".webm", ".mkv", ".mov", ".mp4", ".avi", ".webp", ".gif", ".apng"):
                default_ext = orig_ext
            elif orig_ext == ".png":
                default_ext = ".apng"
            elif is_vid:
                default_ext = ".webm"
            else:
                default_ext = ".webp"

            stem = self.current_filepath.stem if self.current_filepath else "video_processed"
            default_name = f"{stem}_edit{default_ext}"

            filter_map = {
                ".webm": "Video WebM Libre - VP8/VP9 (*.webm)",
                ".mkv": "Video Matroska Libre - FFV1/VP8 (*.mkv)",
                ".mov": "Video QuickTime MOV - MPEG-4 Libre (*.mov)",
                ".mp4": "Video MP4 - MPEG-4 Libre (*.mp4)",
                ".avi": "Video AVI - Motion JPEG Libre (*.avi)",
                ".webp": "WebP Animado HD - Alfa 8 bits (*.webp)",
                ".gif": "GIF Animado Transparente (*.gif)",
                ".apng": "APNG Animado - Alfa 8 bits (*.apng *.png)",
                ".zip": "Secuencia PNG Transparente en ZIP (*.zip)",
                ".png": "PNG Fotograma Actual (*.png)",
            }
            filters = ";;".join(filter_map.values())
            initial_filter = filter_map.get(default_ext, filter_map[".webm" if is_vid else ".webp"])

            dialog_title = "Guardar Video / Animación Como" if is_vid else "Guardar Animación Como"
            path, selected_filter = QFileDialog.getSaveFileName(
                self, dialog_title, default_name, filters, initial_filter
            )
        else:
            default_ext = ".png"
            stem = self.current_filepath.stem if self.current_filepath else "anime_processed"
            default_name = f"{stem}_edit{default_ext}"
            filters = "PNG transparente (*.png);;GIF (*.gif);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)"
            path, selected_filter = QFileDialog.getSaveFileName(
                self, "Guardar Imagen Como", default_name, filters
            )

        if not path:
            return

        try:
            if is_animated_seq:
                self.set_processing(True, tr("status_saving_sequence"))
                self.progress_bar.setRange(0, 0)
                if hasattr(self, "btn_cancel"):
                    self.btn_cancel.setVisible(False)

                def _on_seq_saved(saved_path):
                    self.progress_bar.setRange(0, 100)
                    self.set_processing(False)
                    if saved_path is not None:
                        self._set_status("status_saved", name=saved_path.name)
                        FramelessPopup(
                            self,
                            tr("popup_save_success_title"),
                            tr("popup_save_success_msg", path=str(saved_path)),
                        ).exec()

                def _on_seq_save_failed(err_msg):
                    self.progress_bar.setRange(0, 100)
                    self.set_processing(False)
                    FramelessPopup(
                        self,
                        tr("popup_error_save_title"),
                        tr("popup_error_save_msg", error=err_msg),
                    ).exec()

                self.ai.start_save_sequence(
                    save_fn=lambda: self.doc.save_to_path(
                        path,
                        selected_filter=selected_filter,
                        bg_color_rgb=self.canvas.bg_color_rgb,
                    ),
                    on_saved=_on_seq_saved,
                    on_failed=_on_seq_save_failed,
                )
                return

            saved_path = self.doc.save_to_path(
                path,
                selected_filter=selected_filter,
                bg_color_rgb=self.canvas.bg_color_rgb,
            )
            if saved_path is None:
                return
            self._set_status("status_saved", name=saved_path.name)
            FramelessPopup(self, tr("popup_save_success_title"), tr("popup_save_success_msg", path=str(saved_path))).exec()
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
            if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.is_wand_tool_active():
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
            if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.is_wand_tool_active():
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
            if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.is_wand_tool_active():
                self._ensure_sam_embedding_async()

    def push_new_image_state(self, image: Image.Image, description: str, brush_reference: Optional[Image.Image] = None):
        """Registra un nuevo estado procesado en el historial y canvas."""
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        current = self.image_state.push_state(image, description, brush_reference=brush_reference)
        self.canvas.set_image(current, reset_zoom=False)
        self.canvas.set_original_image(self.image_state.current_brush_reference)
        self._set_status("status_applied", desc=description)
        self._update_image_info_label(current)
        self._update_action_states()
        if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.is_wand_tool_active():
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

    @Slot(int, str)
    def _on_worker_progress(self, percent: int, message: str):
        """Actualiza la barra de progreso real (0% - 100%) y el mensaje de estado en vivo."""
        pct = max(0, min(100, int(percent)))
        if self.progress_bar.maximum() != 100 or self.progress_bar.minimum() != 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(pct)
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        if message:
            self._set_status("status_processing", pct=pct, message=message)

    def cancel_active_processing(self):
        """Solicita cancelación cooperativa del worker o descarga en curso."""
        if self.ai.cancel_active():
            self._set_status("status_cancelled")

    def _on_worker_cancelled(self):
        self.progress_bar.setRange(0, 100)
        self.set_processing(False)
        self.sidebar._update_batch_btn_state()
        self.sidebar.set_detecting_subjects_state(False)
        self._set_status("status_cancelled")

    def set_processing(self, is_processing: bool, message: str = ""):
        """Activa o desactiva el estado de carga y barra de progreso."""
        if is_processing:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_busy(is_processing)
        else:
            self.progress_bar.setVisible(is_processing)
        self.sidebar.set_busy(is_processing)
        self.btn_open.setEnabled(not is_processing)
        self.btn_save.setEnabled(not is_processing and self.image_state.has_image)
        self.btn_undo.setEnabled(not is_processing and self.image_state.can_undo())
        self.btn_redo.setEnabled(not is_processing and self.image_state.can_redo())
        self.btn_reset_orig.setEnabled(not is_processing and self.image_state.can_reset())
        self.btn_compare.setEnabled(not is_processing and self.image_state.has_image and self.image_state._current_index > 0)

        if is_processing:
            self._set_status("status_starting", message=message)
        else:
            if message:
                self._set_status_raw(message)
            else:
                self._set_status("status_ready")

    # -------------------------------------------------------------
    # Inferencia Asíncrona de Modelos (Workers)
    # -------------------------------------------------------------
    def start_remove_bg(self, model_id: str, enable_clothing_protection: Optional[bool] = None):
        if not self.image_state.has_image or self.ai.is_active_running():
            return
        if enable_clothing_protection is None:
            enable_clothing_protection = getattr(self.sidebar, "is_clothing_protection_enabled", lambda: True)()

        # Si el archivo actual es un GIF animado o Video multicapa, procesar toda la secuencia
        if self.doc.has_sequence:
            if self.gif_timer.isActive():
                self.gif_timer.stop()
                self.gif_bar.set_playing(False)

            frames = self.sequence_session.get_all_current_frames()
            media_lbl = "Video" if getattr(self.sequence_session, "is_video", False) else "GIF"
            self.set_processing(True, f"Procesando {media_lbl} ({len(frames)} frames) con {model_id}")
            self.progress_bar.setRange(0, len(frames))
            self.progress_bar.setValue(0)
            self.ai.start_remove_bg_sequence(
                frames=frames,
                model_id=model_id,
                enable_clothing_protection=enable_clothing_protection,
                media_label=media_lbl,
                on_progress=lambda cur, tot: self.progress_bar.setValue(cur),
                on_progress_text=self._set_status_raw,
                on_finished=self._on_gif_worker_finished,
                on_failed=self._on_worker_failed,
                on_cancelled=self._on_worker_cancelled,
            )
            return

        self.set_processing(True, f"Quitando fondo con {model_id}")
        self.ai.start_remove_bg_single(
            image=self.image_state.current_image,
            model_id=model_id,
            enable_clothing_protection=enable_clothing_protection,
            worker_cls=WorkerThread,
            on_progress=self._on_worker_progress,
            on_finished=self._on_worker_finished,
            on_failed=self._on_worker_failed,
            on_cancelled=self._on_worker_cancelled,
        )

    def _on_gif_worker_finished(self, processed_frames: list, description: str):
        self.progress_bar.setRange(0, 100)
        self.set_processing(False)
        self._set_status("status_completed", desc=description)
        if processed_frames:
            target_idx = self.doc.apply_sequence_results(processed_frames, description)
            self._navigate_gif_frame(target_idx)

    def start_restore(self, scale_mode: str, use_tiling: bool):
        if not self.image_state.has_image or self.ai.is_active_running():
            return
        desc_map = {"x2": "Escalado 2x", "x4": "Escalado 4x", "enhance_only": "Mejora de nitidez"}
        desc = desc_map.get(scale_mode, f"Escalado {scale_mode}")

        if self.doc.has_sequence:
            if self.gif_timer.isActive():
                self.gif_timer.stop()
                self.gif_bar.set_playing(False)
            frames = self.sequence_session.get_all_current_frames()
            media_lbl = "Video" if getattr(self.sequence_session, "is_video", False) else "GIF"
            self.set_processing(True, f"{desc} ({len(frames)} frames)")
            self.progress_bar.setRange(0, len(frames))
            self.progress_bar.setValue(0)
            self.ai.start_restore_sequence(
                frames=frames,
                scale_mode=scale_mode,
                use_tiling=use_tiling,
                description=desc,
                media_label=media_lbl,
                on_progress=lambda cur, tot: self.progress_bar.setValue(cur),
                on_progress_text=self._set_status_raw,
                on_finished=self._on_gif_worker_finished,
                on_failed=self._on_worker_failed,
                on_cancelled=self._on_worker_cancelled,
            )
            return

        self.set_processing(True, desc)
        self.ai.start_restore(
            image=self.image_state.current_image,
            scale_mode=scale_mode,
            use_tiling=use_tiling,
            description=desc,
            worker_cls=WorkerThread,
            on_progress=self._on_worker_progress,
            on_finished=self._on_worker_finished,
            on_failed=self._on_worker_failed,
            on_cancelled=self._on_worker_cancelled,
        )

    def start_batch_processing(self, input_dir: str, output_dir: str, model_id: str, enable_clothing_protection: bool):
        if self.ai.is_batch_running():
            logger.info("Procesamiento en lote ya en curso, ignorando solicitud duplicada.")
            return
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_busy(True)
        else:
            self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self._set_status("status_batch_preparing")
        self.sidebar.btn_run_batch.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_save.setEnabled(False)

        def update_progress(current, total):
            if total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(current)

        def on_batch_finished(success_count, total, errors=None):
            if hasattr(self, "status_bar_widget"):
                self.status_bar_widget.set_busy(False)
            else:
                self.progress_bar.setVisible(False)
            self.sidebar._update_batch_btn_state()
            self.btn_open.setEnabled(True)
            self.btn_save.setEnabled(self.image_state.has_image)
            errors = errors or []
            if errors:
                preview = "\n".join(f"• {fname}: {reason}" for fname, reason in errors[:10])
                if len(errors) > 10:
                    preview += f"\n… (+{len(errors) - 10})"
                QMessageBox.warning(
                    self,
                    tr("popup_batch_done_title"),
                    tr(
                        "popup_batch_done_errors_msg",
                        success=success_count,
                        total=total,
                        failed_count=len(errors),
                        details=preview,
                    ),
                )
            else:
                QMessageBox.information(
                    self,
                    tr("popup_batch_done_title"),
                    tr("popup_batch_done_msg", success=success_count, total=total),
                )
            self._set_status("status_batch_finished")

        def on_batch_failed(err_msg):
            if hasattr(self, "status_bar_widget"):
                self.status_bar_widget.set_busy(False)
            else:
                self.progress_bar.setVisible(False)
            self.sidebar._update_batch_btn_state()
            self.btn_open.setEnabled(True)
            self.btn_save.setEnabled(self.image_state.has_image)
            QMessageBox.critical(
                self,
                tr("popup_batch_error_title"),
                tr("popup_batch_error_msg", error=err_msg),
            )
            self._set_status("status_batch_error")

        pipeline_cfg = (
            self.sidebar.panel_batch.get_pipeline_config()
            if hasattr(self.sidebar, "panel_batch")
            and hasattr(self.sidebar.panel_batch, "get_pipeline_config")
            else None
        )
        self.ai.start_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            model_id=model_id,
            enable_clothing_protection=enable_clothing_protection,
            on_progress=update_progress,
            on_progress_text=self._set_status_raw,
            on_finished=on_batch_finished,
            on_failed=on_batch_failed,
            on_finished_with_errors=on_batch_finished,
            on_cancelled=self._on_worker_cancelled,
            pipeline_config=pipeline_cfg,
        )

    def _on_brush_mode_toggled(self, enabled: bool):
        self.canvas.set_brush_mode(enabled)

    def _on_wand_mode_toggled(self, enabled: bool):
        self.canvas.set_wand_mode(enabled)
        if enabled and self.sidebar.is_wand_ai_mode():
            self._ensure_sam_embedding_async()

    def _on_wand_ai_mode_toggled(self, enabled: bool):
        self.canvas.set_wand_ai_mode(enabled)
        if enabled and self.sidebar.is_wand_tool_active():
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

        if self.ai.is_sam_running():
            return

        self.sidebar.set_wand_ai_status(tr("wand_ai_computing"))
        self.ai.start_sam_embedding(
            image=self.image_state.current_image,
            on_progress=self._on_worker_progress,
            on_ready=self._on_sam_embedding_ready,
            on_failed=self._on_sam_embedding_failed,
        )

    def _on_sam_embedding_ready(self, _):
        self.progress_bar.setVisible(False)
        self.sidebar.set_wand_ai_status(tr("wand_ai_ready"))
        self._set_status("wand_ai_ready")
        pending = getattr(self.canvas, "_pending_wand_click", None)
        if pending is not None:
            self.canvas._pending_wand_click = None
            pt, mode = pending
            self.canvas._apply_magic_wand(pt, mode=mode)

    def _on_sam_embedding_failed(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.sidebar.set_wand_ai_status("Error")
        self._set_status("status_sam_error", error=err_msg)

    def update_device_badge(self):
        """Actualiza el badge de la barra de estado y el selector del sidebar según el estado actual."""
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()

        if gpu_mgr.has_orphaned_worker():
            self.update_device_badge_blocked()
            return

        self.device = detect_device()
        is_gpu = "GPU" in self.device.upper()
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_device_badge(self.device, "gpu" if is_gpu else "cpu")
        elif hasattr(self, "lbl_device"):
            self.lbl_device.setText(self.device)
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
        """Aplica el cambio de modo de dispositivo solicitado por el usuario y libera VRAM residual."""
        _, status_code = self.ai.switch_device_mode(is_gpu)
        self.update_device_badge()
        if status_code.startswith("gpu_enabled:"):
            self._set_status("status_gpu_enabled", name=status_code.split(":", 1)[1])
        elif status_code == "cpu_enabled":
            self._set_status("status_cpu_enabled")
        elif status_code == "gpu_failed":
            self._set_status("status_gpu_failed")
        else:
            self._set_status_raw(status_code)

    def update_device_badge_blocked(self):
        """Actualiza el badge de la barra de estado indicando que la GPU está bloqueada y se fuerza CPU."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_device_badge(tr("status_gpu_blocked_badge"), "blocked")
        elif hasattr(self, "lbl_device"):
            self.lbl_device.setText(tr("status_gpu_blocked_badge"))
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
            self.set_processing(False)
            self._set_status("status_gpu_orphaned_warn")
            gpu_mgr._last_status_message = ""
        elif "GPU no respondió" in last_status:
            self.set_processing(False)
            self._set_status("status_gpu_timeout_cpu")
            gpu_mgr._last_status_message = ""
        elif "GPU con error" in last_status:
            self.set_processing(False)
            self._set_status("status_gpu_error_cpu")
            gpu_mgr._last_status_message = ""
        else:
            self.set_processing(False)
            self._set_status("status_completed", desc=description)
            
        brush_ref = None
        if "Escalado" in description or "Mejora de nitidez" in description:
            curr_brush = self.image_state.current_brush_reference
            if curr_brush:
                if not self.image_state.has_background_been_removed:
                    brush_ref = result_image.copy()
                else:
                    brush_ref = curr_brush.resize(result_image.size, resample=Image.Resampling.LANCZOS)
                
        self.push_new_image_state(result_image, description, brush_reference=brush_ref)

    def _on_worker_failed(self, error_message: str):
        self.set_processing(False)
        self._set_status("status_error_process")
        QMessageBox.critical(
            self,
            tr("popup_error_process_title"),
            tr("popup_error_process_msg", error=error_message),
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
        """Recorta los márgenes transparentes alrededor del sujeto (imagen o secuencia completa)."""
        if self.doc.has_sequence:
            err_key, res = self.doc.apply_autocrop()
            if err_key:
                self._set_status(err_key)
                return
            if res is None:
                return
            cropped, _brush_ref, desc = res
            self._navigate_gif_frame(self.sequence_session.current_index)
            self._set_status("status_applied", desc=desc)
            self._update_image_info_label(cropped)
            self._update_action_states()
            return

        err_key, res = self.doc.compute_autocrop()
        if err_key:
            self._set_status(err_key)
            return
        if res is None:
            return
        cropped, brush_ref, desc = res
        self.push_new_image_state(cropped, desc, brush_reference=brush_ref)

    def perform_apply_bg_color(self, color_rgb: Tuple[int, int, int]):
        """Compone la imagen actual con transparencia sobre un fondo de color sólido en la capa del canvas."""
        if not self.image_state.has_image:
            return
        
        self.canvas.set_background_color(color_rgb)
        hex_str = f"#{color_rgb[0]:02X}{color_rgb[1]:02X}{color_rgb[2]:02X}"
        self._set_status("status_bg_color", hex=hex_str)

    def perform_edge_refinement(self, choke_px: int, feather_px: int, despill_strength: float):
        """Refina los bordes alfa (Choke / Feather / Despill) sobre imagen o secuencia."""
        if not self.image_state.has_image:
            return
        if choke_px == 0 and feather_px == 0 and despill_strength <= 0.0:
            return
        out = self.doc.apply_edge_refinement(choke_px, feather_px, despill_strength)
        if out is None:
            return
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        if self.doc.has_sequence:
            self._navigate_gif_frame(self.sequence_session.current_index)
        else:
            self.canvas.set_image(out, reset_zoom=False)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
        self._set_status("status_edge_refined")
        self._update_image_info_label(out)
        self._update_action_states()

    def perform_inpaint_selection(self):
        """Rellena el área seleccionada mediante inpainting OpenCV."""
        if not self.image_state.has_image or not self.canvas.has_selection():
            self._set_status("status_selection_nothing")
            return
        out = self.doc.apply_inpaint_selection(self.canvas.active_selection)
        if out is None:
            return
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        self.canvas.set_image(out, reset_zoom=False)
        self.canvas.set_original_image(self.image_state.current_brush_reference)
        self._set_status("status_inpainted")
        self._update_image_info_label(out)
        self._update_action_states()

    def _sync_after_document_edit(self, out_img, status_key: str, reset_zoom: bool = False):
        """Sincroniza canvas, caché SAM e indicadores tras una transformación de documento."""
        from app.core.sam_manager import get_sam_manager
        get_sam_manager().clear_embedding()
        if self.doc.has_sequence:
            self._navigate_gif_frame(self.sequence_session.current_index)
        else:
            self.canvas.set_image(out_img, reset_zoom=reset_zoom)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
        self._set_status(status_key)
        self._update_image_info_label(out_img)
        self._update_action_states()

    def perform_apply_effects(self, cfg: dict):
        """Aplica efectos creativos (Sticker, Sombra, Bokeh/Imagen BG, Relación de Aspecto)."""
        if not self.image_state.has_image:
            return
        bg_img = None
        bg_path = str(cfg.get("bg_image_path", "") or "").strip()
        if bg_path:
            from pathlib import Path
            from PIL import Image
            p = Path(bg_path)
            if p.is_file():
                try:
                    with Image.open(p) as im:
                        bg_img = im.convert("RGBA")
                except Exception:
                    bg_img = None
        out = self.doc.apply_effects(
            sticker_width=int(cfg.get("sticker_width", 0)),
            sticker_color=tuple(cfg.get("sticker_color", (255, 255, 255))),
            shadow_opacity=float(cfg.get("shadow_opacity", 0.0)),
            shadow_blur=int(cfg.get("shadow_blur", 16)),
            shadow_offset_x=int(cfg.get("shadow_offset_x", 10)),
            shadow_offset_y=int(cfg.get("shadow_offset_y", 12)),
            bg_mode=str(cfg.get("bg_mode", "transparent")),
            bg_color=tuple(cfg.get("bg_color", (255, 255, 255))),
            bokeh_blur=int(cfg.get("bokeh_blur", 20)),
            bg_image=bg_img,
            aspect_preset=str(cfg.get("aspect_preset", "original")),
        )
        if out is not None:
            self._sync_after_document_edit(out, "status_effects_applied", reset_zoom=False)

    def perform_transform(self, rotate_deg: int, flip_h: bool, flip_v: bool):
        """Aplica rotación horaria de 90° o volteo horizontal/vertical."""
        if not self.image_state.has_image:
            return
        out = self.doc.apply_transform(rotate_deg=rotate_deg, flip_h=flip_h, flip_v=flip_v)
        if out is not None:
            self._sync_after_document_edit(out, "status_transformed", reset_zoom=True)

    def on_crop_mode_toggled(self, enabled: bool, aspect_ratio: object):
        """Activa o desactiva la superposición interactiva de recorte en el lienzo."""
        if enabled and not self.image_state.has_image:
            self.sidebar.panel_effects.reset_crop_toggle()
            return
        ratio = float(aspect_ratio) if aspect_ratio is not None else None
        self.canvas.set_crop_mode(bool(enabled), aspect_ratio=ratio)

    def perform_manual_crop(self):
        """Ejecuta el recorte manual definido sobre el lienzo."""
        if not self.image_state.has_image:
            return
        box = self.canvas.get_crop_box()
        if not box:
            return
        out = self.doc.apply_manual_crop(*box)
        self.canvas.set_crop_mode(False)
        self.sidebar.panel_effects.reset_crop_toggle()
        if out is not None:
            self._sync_after_document_edit(out, "status_cropped_manual", reset_zoom=True)

    def perform_restore_transparency(self):
        """Oculta la capa de color sólido restaurando el fondo transparente original."""
        if not self.image_state.has_image:
            return
            
        self.canvas.set_background_color(None)
        self._set_status("status_bg_transparent")

    def start_detect_subjects(self):
        """Ejecuta detección y segmentación multiobjeto con YOLO11 en segundo plano."""
        if not self.image_state.has_image:
            self._set_status("status_open_first")
            return
        if self.ai.is_yolo_running():
            logger.info("Detección YOLO ya en curso, ignorando solicitud duplicada.")
            return

        self.sidebar.set_detecting_subjects_state(True)
        self._set_status("detect_status_computing")
        self.ai.start_yolo_detection(
            image=self.image_state.current_image,
            on_progress=self._on_worker_progress,
            on_finished=self._on_yolo_finished,
            on_failed=self._on_yolo_failed,
        )

    def _on_yolo_finished(self, subjects: list):
        self.progress_bar.setVisible(False)
        self.sidebar.set_detecting_subjects_state(False)
        self.sidebar.set_detected_subjects(subjects)
        if subjects:
            self._set_status("status_yolo_detected", count=len(subjects))
            # Precalcular embedding de MobileSAM en segundo plano para aislamiento HD instantáneo
            self._ensure_sam_embedding_async()
        else:
            self._set_status("detect_none_found")

    def _on_yolo_failed(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.sidebar.set_detecting_subjects_state(False)
        self._set_status("status_yolo_error", error=err_msg)

    def _refine_yolo_subject_mask(self, subject) -> Optional[np.ndarray]:
        """Delega el refinamiento HD de la máscara YOLO11-seg al módulo de dominio."""
        if not self.image_state.has_image or subject is None:
            return None
        from app.core.yolo_detector import refine_yolo_subject_mask
        return refine_yolo_subject_mask(self.image_state.current_image, subject)

    def perform_isolate_subject(self, subject):
        """Aísla el sujeto detectado con refinamiento HD sobre la imagen o toda la secuencia."""
        if self.doc.has_sequence:
            isolated_curr = self.doc.apply_isolate_subject_sequence(subject)
            if isolated_curr is None:
                return
            self._navigate_gif_frame(self.sequence_session.current_index)
            self._set_status("status_applied", desc=f"Aislar {subject.class_name.capitalize()}")
            self._update_image_info_label(isolated_curr)
            self._update_action_states()
            return

        isolated_img = self.doc.build_isolated_subject(subject)
        if isolated_img is None:
            return
        self.push_new_image_state(isolated_img, f"Aislar {subject.class_name.capitalize()}")

    def perform_select_subject(self, subject):
        """Carga la máscara refinada del sujeto detectado en la selección de hormigas marchantes."""
        if not self.image_state.has_image or subject is None:
            return

        high_res_mask = self._refine_yolo_subject_mask(subject)
        if high_res_mask is None:
            high_res_mask = (subject.mask.astype(np.uint8) * 255)

        self.canvas._set_selection(high_res_mask)
        self._set_status("status_subject_selected", name=subject.class_name.capitalize())

    def closeEvent(self, event):
        """Asegura que todos los hilos en segundo plano finalicen limpiamente antes de cerrar la ventana."""
        if hasattr(self, "gif_timer") and self.gif_timer.isActive():
            self.gif_timer.stop()
        self.ai.shutdown_workers(timeout_ms=1000)
        try:
            super().closeEvent(event)
        except Exception:
            if hasattr(event, "accept"):
                event.accept()
