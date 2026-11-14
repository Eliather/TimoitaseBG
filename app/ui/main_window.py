"""
TimoitaseBG - Ventana principal de la aplicación.
"""
import logging
import webbrowser
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from app.config import (
    APP_DIR,
    DEFAULT_WINDOW_TITLE,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    detect_device,
)
from app.controllers.ai_coordinator import AICoordinator
from app.controllers.document_controller import DocumentController
from app.core.image_state import ImageState, SequenceSession
from app.core.image_utils import qimage_to_pil
from app.core.workers import WorkerThread
from app.i18n import get_current_language, set_language, tr
from app.ui import media_actions
from app.ui.canvas_widget import CanvasWidget
from app.ui.components import SequenceBar, StatusBar, TopBar, register_shortcuts
from app.ui.shortcuts_dialog import ShortcutsDialog
from app.ui.toolbar import ToolSidebar

logger = logging.getLogger("TimoitaseBG.MainWindow")


def _bridge_prop(target_attr: str, field: str) -> property:
    """Crea una propiedad delegada hacia un controlador interno (`doc` o `ai`)."""
    return property(
        lambda self: getattr(getattr(self, target_attr), field),
        lambda self, value: setattr(getattr(self, target_attr), field, value),
    )


class MainWindow(QMainWindow):
    """
    Ventana principal de TimoitaseBG.
    Coordina el estado de la imagen, el canvas interactivo y el panel de herramientas.
    """

    gpuStatusNotified = Signal(str)

    # Propiedades de compatibilidad hacia los controladores
    image_state: ImageState = _bridge_prop("doc", "image_state")  # type: ignore[assignment]
    sequence_session: Optional[SequenceSession] = _bridge_prop("doc", "sequence_session")  # type: ignore[assignment]
    current_filepath: Optional[Path] = _bridge_prop("doc", "current_filepath")  # type: ignore[assignment]
    active_worker = _bridge_prop("ai", "active_worker")
    batch_worker = _bridge_prop("ai", "batch_worker")
    sam_worker = _bridge_prop("ai", "sam_worker")
    yolo_worker = _bridge_prop("ai", "yolo_worker")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(max(1200, WINDOW_MIN_WIDTH), max(780, WINDOW_MIN_HEIGHT))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self.doc = DocumentController()
        self.ai = AICoordinator()

        self.gif_timer = QTimer(self)
        self.gif_timer.timeout.connect(self._on_gif_timer_tick)

        self.device = detect_device()

        self._init_ui()
        self._setup_shortcuts()
        self._update_action_states()
        self.update_device_badge()

        self.gpuStatusNotified.connect(self._handle_gpu_status)
        self._gpu().set_status_callback(lambda msg: self.gpuStatusNotified.emit(msg))

        self._load_stylesheet()

    @staticmethod
    def _sam():
        from app.core.sam_manager import get_sam_manager
        return get_sam_manager()

    @staticmethod
    def _gpu():
        from app.core.gpu_manager import get_gpu_manager
        return get_gpu_manager()

    def _bind_widget_attrs(self, source: Any, names: tuple[str, ...]) -> None:
        """Expone controles hijos como atributos retrocompatibles en MainWindow."""
        for name in names:
            setattr(self, name, getattr(source, name))

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _set_status(self, key: str, **kwargs: Any) -> None:
        """Asigna texto traducido a la barra de estado."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_status(key, **kwargs)
        elif hasattr(self, "lbl_status"):
            self.lbl_status.setText(tr(key, **kwargs))

    def _set_status_raw(self, text: str) -> None:
        """Asigna texto directo en barra de estado para mensajes externos."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_status_raw(text)
        elif hasattr(self, "lbl_status"):
            self.lbl_status.setText(text)

    def _set_status_bar_busy(self, busy: bool) -> None:
        """Sincroniza el indicador ocupado/progreso de la barra de estado."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_busy(busy)
        else:
            self.progress_bar.setVisible(busy)

    def _on_lang_changed(self, index: int) -> None:
        code = self.combo_lang.itemData(index)
        if code and code != get_current_language():
            set_language(code)
            self.retranslate_ui()

    def retranslate_ui(self) -> None:
        """Actualiza dinámicamente los textos de toda la interfaz al cambiar de idioma."""
        for widget_name in ("status_bar_widget", "top_bar", "sidebar", "gif_bar", "canvas"):
            widget = getattr(self, widget_name, None)
            if widget is not None and hasattr(widget, "retranslate_ui"):
                widget.retranslate_ui()
        if hasattr(self, "canvas"):
            self.canvas.update()
        self.update_device_badge()

    def _load_stylesheet(self) -> None:
        self.is_dark_theme = getattr(self, "is_dark_theme", True)
        style_file = "style.qss" if self.is_dark_theme else "style_light.qss"
        style_path = Path(APP_DIR) / "ui" / style_file
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                content = f.read()
            assets_path = (Path(APP_DIR) / "ui" / "assets").as_posix()
            self.setStyleSheet(content.replace("{ASSETS_PATH}", assets_path))
        self._update_theme_icons()

    def _toggle_theme(self) -> None:
        self.is_dark_theme = not getattr(self, "is_dark_theme", True)
        self._load_stylesheet()

    def _update_theme_icons(self) -> None:
        """Aplica iconos vectoriales adaptados al contraste del tema activo."""
        for widget_name, method_name in (
            ("top_bar", "apply_theme"),
            ("status_bar_widget", "apply_theme"),
            ("sidebar", "update_theme_icons"),
            ("canvas", "update_theme"),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None and hasattr(widget, method_name):
                getattr(widget, method_name)(self.is_dark_theme)
        if hasattr(self, "gif_bar") and hasattr(self.gif_bar, "apply_theme"):
            is_playing = hasattr(self, "gif_timer") and self.gif_timer.isActive()
            self.gif_bar.apply_theme(self.is_dark_theme, is_playing=is_playing)

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._init_canvas_and_sidebar()
        root_layout.addWidget(self._create_top_bar())

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

        self.status_bar = self._create_status_bar()
        root_layout.addWidget(self.status_bar)

    def _init_canvas_and_sidebar(self) -> None:
        """Instancia y conecta señales entre CanvasWidget, ToolSidebar y MainWindow."""
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

        self._bind_widget_attrs(
            top_bar,
            (
                "version_badge", "btn_theme", "combo_lang", "btn_open", "btn_save",
                "btn_undo", "btn_redo", "btn_reset_orig", "btn_zoom_out", "lbl_zoom",
                "btn_zoom_in", "btn_fit", "btn_zoom_100", "btn_compare",
                "btn_min", "btn_max", "btn_close",
            ),
        )
        return top_bar

    def _create_status_bar(self) -> QWidget:
        bar = StatusBar(size_grip_target=self, parent=self)
        self.status_bar_widget = bar
        bar.shortcutsRequested.connect(self.show_shortcuts_modal)
        bar.cancelRequested.connect(self.cancel_active_processing)
        bar.deviceToggleRequested.connect(self._toggle_device_mode)
        bar.kofiRequested.connect(lambda: webbrowser.open("https://ko-fi.com/eliather"))

        self._bind_widget_attrs(
            bar,
            (
                "btn_shortcuts", "lbl_status", "lbl_img_info", "progress_bar",
                "btn_cancel", "lbl_device", "btn_kofi", "size_grip",
            ),
        )
        return bar

    def show_shortcuts_modal(self) -> None:
        """Abre el diálogo modal con la lista interactiva de atajos de teclado y gestos."""
        ShortcutsDialog(self, is_dark_theme=getattr(self, "is_dark_theme", True)).exec()

    def open_model_manager(self) -> None:
        """Abre el gestor visual de modelos ONNX locales."""
        from app.ui.dialogs import ModelManagerDialog

        dialog = ModelManagerDialog(self, is_dark_theme=getattr(self, "is_dark_theme", True))
        dialog.modelsChanged.connect(self.ai.unload_sessions)
        dialog.exec()

    def _setup_shortcuts(self) -> None:
        self._shortcuts = register_shortcuts(self)

    def _create_gif_pagination_bar(self) -> QWidget:
        """Crea la barra de navegación y reproducción por fotogramas para GIF/Video."""
        bar = SequenceBar(self)
        bar.apply_theme(getattr(self, "is_dark_theme", True), is_playing=False)
        bar.firstRequested.connect(lambda: self._navigate_gif_frame(0))
        bar.prevRequested.connect(
            lambda: self._navigate_gif_frame(self.sequence_session.current_index - 1 if self.sequence_session else 0)
        )
        bar.playToggled.connect(self._toggle_gif_playback)
        bar.nextRequested.connect(
            lambda: self._navigate_gif_frame(self.sequence_session.current_index + 1 if self.sequence_session else 0)
        )
        bar.lastRequested.connect(
            lambda: self._navigate_gif_frame(self.sequence_session.total_frames - 1 if self.sequence_session else 0)
        )
        bar.trimRangeChanged.connect(self._on_sequence_trim_changed)
        bar.fpsChanged.connect(self._on_sequence_fps_changed)
        bar.trackSequenceRequested.connect(self.start_track_sequence)

        self._bind_widget_attrs(
            bar,
            ("btn_gif_first", "btn_gif_prev", "btn_gif_play", "btn_gif_next", "btn_gif_last", "lbl_gif_pagination"),
        )
        return bar

    def _navigate_gif_frame(self, new_index: int) -> None:
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

    def _toggle_gif_playback(self) -> None:
        """Alterna entre reproducción en bucle y pausa para la secuencia."""
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

    def _on_gif_timer_tick(self) -> None:
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

    def _on_sequence_trim_changed(self, start: int, end: int) -> None:
        if not self.sequence_session:
            return
        s, e = self.sequence_session.set_trim_range(start, end)
        if self.sequence_session.current_index < s or self.sequence_session.current_index > e:
            self._navigate_gif_frame(s)

    def _on_sequence_fps_changed(self, fps: float) -> None:
        if self.sequence_session:
            self.sequence_session.set_target_fps(fps)

    def _prepare_sequence_run(self, status_msg: str, progress_max: Optional[int] = None) -> tuple[list, str]:
        """Pausa reproducción, prepara progreso y devuelve frames + etiqueta de medio."""
        if self.gif_timer.isActive():
            self.gif_timer.stop()
            self.gif_bar.set_playing(False)
        frames = self.sequence_session.get_all_current_frames()
        media_lbl = "Video" if getattr(self.sequence_session, "is_video", False) else "GIF"
        self.set_processing(True, status_msg)
        self.progress_bar.setRange(0, progress_max if progress_max is not None else len(frames))
        self.progress_bar.setValue(0)
        return frames, media_lbl

    def start_track_sequence(self) -> None:
        """Rastrea automáticamente el sujeto activo a lo largo de la secuencia."""
        if not self.doc.has_sequence or self.ai.is_active_running():
            return
        seed_idx = self.sequence_session.current_index
        seed_subject = self.sidebar.panel_bg.get_selected_subject() if hasattr(self.sidebar, "panel_bg") else None
        seed_mask = self.canvas.active_selection if self.canvas.has_selection() else None
        frames, _ = self._prepare_sequence_run(tr("seq_tracking_status"), progress_max=100)
        self.ai.start_track_sequence(
            frames=frames,
            seed_frame_idx=seed_idx,
            seed_subject=seed_subject,
            seed_mask=seed_mask,
            trim_start=self.sequence_session.trim_start,
            trim_end=self.sequence_session.trim_end,
            on_progress=lambda cur, _tot: self.progress_bar.setValue(cur),
            on_progress_text=self._set_status_raw,
            on_finished=self._on_gif_worker_finished,
            on_failed=self._on_worker_failed,
            on_cancelled=self._on_worker_cancelled,
        )

    def perform_delete_selection(self) -> None:
        """Borra los píxeles seleccionados con la varita."""
        if not self.canvas.has_selection():
            return
        key = "status_selection_deleted" if self.canvas.delete_selection() else "status_selection_nothing"
        self._set_status(key)

    # -------------------------------------------------------------
    # Lógica de Apertura, Pegado y Guardado (delegada a media_actions)
    # -------------------------------------------------------------
    def prompt_open_image(self) -> None:
        media_actions.prompt_open_image(self)

    def _reset_workspace_ui(self, loaded_img: Optional[Image.Image] = None, reset_zoom: bool = True) -> None:
        """Limpia estado transitorio de IA, selección, comparación y actualiza canvas."""
        self._sam().clear_embedding()
        if hasattr(self, "sidebar"):
            self.sidebar.set_detected_subjects([])
            self._refresh_sam_if_wand_ai()
        self.btn_compare.setChecked(False)
        self.canvas.set_comparison_mode(False)
        self.canvas.clear_selection()
        if loaded_img is not None:
            self.canvas.set_image(loaded_img, reset_zoom=reset_zoom)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
            self._update_image_info_label(loaded_img)
        self._update_action_states()

    def load_new_pil_image(self, pil_img: Image.Image, source_name: str, filepath: Optional[Path] = None) -> None:
        """Carga una nueva imagen PIL en el estado inmutable y actualiza el workspace."""
        loaded = self.doc.load_pil_image(pil_img, filepath=filepath, clear_seq=False)
        if loaded is None:
            return
        self._reset_workspace_ui(loaded, reset_zoom=True)
        self._set_status("status_loaded", name=source_name)

    def open_image_file(self, file_path: str) -> None:
        media_actions.open_image_file(self, file_path)

    def copy_to_clipboard(self) -> None:
        media_actions.copy_to_clipboard(self)

    def paste_from_clipboard(self) -> None:
        media_actions.paste_from_clipboard(self)

    def prompt_save_image(self) -> None:
        media_actions.prompt_save_image(self)

    # -------------------------------------------------------------
    # Acciones de Historial y Sincronización Común
    # -------------------------------------------------------------
    def _refresh_sam_if_wand_ai(self) -> None:
        if hasattr(self, "sidebar") and self.sidebar.is_wand_ai_mode() and self.sidebar.is_wand_tool_active():
            self._ensure_sam_embedding_async()

    def _sync_after_document_edit(
        self,
        out_img: Image.Image,
        status_key: str,
        *,
        reset_zoom: bool = False,
        refresh_sam: bool = False,
        use_sequence_nav: bool = True,
        **status_kwargs: Any,
    ) -> None:
        """Sincroniza canvas, caché SAM e indicadores tras editar imagen o historial."""
        self._sam().clear_embedding()
        if use_sequence_nav and self.doc.has_sequence:
            self._navigate_gif_frame(self.sequence_session.current_index)
        else:
            self.canvas.set_image(out_img, reset_zoom=reset_zoom)
            self.canvas.set_original_image(self.image_state.current_brush_reference)
        self._set_status(status_key, **status_kwargs)
        self._update_image_info_label(out_img)
        self._update_action_states()
        if refresh_sam:
            self._refresh_sam_if_wand_ai()

    def _apply_history_navigation(
        self,
        img: Optional[Image.Image],
        status_key: str,
        *,
        with_action_summary: bool = False,
    ) -> None:
        """Aplica el resultado de Undo, Redo o Reset sobre canvas e indicadores."""
        if not img:
            return
        kwargs = {"action": self.image_state.get_status_summary()} if with_action_summary else {}
        self._sync_after_document_edit(
            img,
            status_key,
            reset_zoom=False,
            refresh_sam=True,
            use_sequence_nav=False,
            **kwargs,
        )

    def perform_undo(self) -> None:
        self._apply_history_navigation(self.image_state.undo(), "status_undo", with_action_summary=True)

    def perform_redo(self) -> None:
        self._apply_history_navigation(self.image_state.redo(), "status_redo", with_action_summary=True)

    def perform_reset_original(self) -> None:
        self._apply_history_navigation(self.image_state.reset_to_original(), "status_reset")

    def push_new_image_state(
        self,
        image: Image.Image,
        description: str,
        brush_reference: Optional[Image.Image] = None,
    ) -> None:
        """Registra un nuevo estado procesado en el historial y canvas."""
        current = self.image_state.push_state(image, description, brush_reference=brush_reference)
        self._sync_after_document_edit(
            current,
            "status_applied",
            reset_zoom=False,
            refresh_sam=True,
            use_sequence_nav=False,
            desc=description,
        )

    # -------------------------------------------------------------
    # Actualizaciones de UI
    # -------------------------------------------------------------
    def _update_action_states(self) -> None:
        has_img = self.image_state.has_image
        self.btn_save.setEnabled(has_img)
        self.btn_undo.setEnabled(self.image_state.can_undo())
        self.btn_redo.setEnabled(self.image_state.can_redo())
        self.btn_reset_orig.setEnabled(self.image_state.can_reset() and self.image_state.can_undo())

        for btn in (self.btn_zoom_in, self.btn_zoom_out, self.btn_fit, self.btn_zoom_100):
            btn.setEnabled(has_img)

        can_compare = has_img and self.image_state._current_index > 0
        self.btn_compare.setEnabled(can_compare)
        if not can_compare and self.canvas.comparison_mode:
            self.btn_compare.setChecked(False)
            self.canvas.set_comparison_mode(False)
        elif self.canvas.comparison_mode:
            self.canvas.set_comparison_mode(True, self.image_state.original_image)

        self.sidebar.btn_remove_bg.setEnabled(has_img)
        self.sidebar.btn_restore.setEnabled(has_img)
        self.sidebar.set_transparency_available(
            has_transparency=self.image_state.has_transparency,
            can_restore=self.image_state.get_last_transparent_state() is not None,
        )

    def _update_image_info_label(self, img: Image.Image) -> None:
        self.lbl_img_info.setText(f"{img.width} × {img.height} px • {img.mode}")

    def _on_mask_changed(self, has_mask: bool) -> None:
        self.sidebar.set_has_mask(has_mask)

    def _on_image_edited_directly(self, description: str = "Edición Manual") -> None:
        """Captura la edición directa de píxeles y la guarda en el historial."""
        if self.canvas.current_qimage:
            pil_img = qimage_to_pil(self.canvas.current_qimage)
            self.push_new_image_state(pil_img, description or "Edición Manual")

    def _on_zoom_changed(self, zoom: float) -> None:
        self.lbl_zoom.setText(f"{int(zoom * 100)}%")

    @Slot(int, str)
    def _on_worker_progress(self, percent: int, message: str) -> None:
        """Actualiza la barra de progreso y el mensaje de estado en vivo."""
        pct = max(0, min(100, int(percent)))
        if self.progress_bar.maximum() != 100 or self.progress_bar.minimum() != 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(pct)
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        if message:
            self._set_status("status_processing", pct=pct, message=message)

    def cancel_active_processing(self) -> None:
        """Solicita cancelación cooperativa del worker o descarga en curso."""
        if self.ai.cancel_active():
            self._set_status("status_cancelled")

    def _on_worker_cancelled(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.set_processing(False)
        self.sidebar._update_batch_btn_state()
        self.sidebar.set_detecting_subjects_state(False)
        self._set_status("status_cancelled")

    def set_processing(self, is_processing: bool, message: str = "") -> None:
        """Activa o desactiva el estado de carga y controles dependientes."""
        if is_processing:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        self._set_status_bar_busy(is_processing)
        self.sidebar.set_busy(is_processing)
        self.btn_open.setEnabled(not is_processing)
        self.btn_save.setEnabled(not is_processing and self.image_state.has_image)
        self.btn_undo.setEnabled(not is_processing and self.image_state.can_undo())
        self.btn_redo.setEnabled(not is_processing and self.image_state.can_redo())
        self.btn_reset_orig.setEnabled(not is_processing and self.image_state.can_reset())
        self.btn_compare.setEnabled(
            not is_processing and self.image_state.has_image and self.image_state._current_index > 0
        )

        if is_processing:
            self._set_status("status_starting", message=message)
        elif message:
            self._set_status_raw(message)
        else:
            self._set_status("status_ready")

    # -------------------------------------------------------------
    # Inferencia Asíncrona de Modelos (Workers)
    # -------------------------------------------------------------
    def start_remove_bg(self, model_id: str, enable_clothing_protection: Optional[bool] = None) -> None:
        if not self.image_state.has_image or self.ai.is_active_running():
            return
        if enable_clothing_protection is None:
            enable_clothing_protection = getattr(self.sidebar, "is_clothing_protection_enabled", lambda: True)()

        if self.doc.has_sequence:
            total = len(self.sequence_session.get_all_current_frames())
            media_lbl = "Video" if getattr(self.sequence_session, "is_video", False) else "GIF"
            frames, media_lbl = self._prepare_sequence_run(f"Procesando {media_lbl} ({total} frames) con {model_id}")
            self.ai.start_remove_bg_sequence(
                frames=frames,
                model_id=model_id,
                enable_clothing_protection=enable_clothing_protection,
                media_label=media_lbl,
                on_progress=lambda cur, _tot: self.progress_bar.setValue(cur),
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

    def _on_gif_worker_finished(self, processed_frames: list, description: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.set_processing(False)
        self._set_status("status_completed", desc=description)
        if processed_frames:
            target_idx = self.doc.apply_sequence_results(processed_frames, description)
            self._navigate_gif_frame(target_idx)

    def start_restore(self, scale_mode: str, use_tiling: bool) -> None:
        if not self.image_state.has_image or self.ai.is_active_running():
            return
        desc_map = {"x2": "Escalado 2x", "x4": "Escalado 4x", "enhance_only": "Mejora de nitidez"}
        desc = desc_map.get(scale_mode, f"Escalado {scale_mode}")

        if self.doc.has_sequence:
            total = len(self.sequence_session.get_all_current_frames())
            frames, media_lbl = self._prepare_sequence_run(f"{desc} ({total} frames)")
            self.ai.start_restore_sequence(
                frames=frames,
                scale_mode=scale_mode,
                use_tiling=use_tiling,
                description=desc,
                media_label=media_lbl,
                on_progress=lambda cur, _tot: self.progress_bar.setValue(cur),
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

    def _set_batch_busy(self, busy: bool) -> None:
        """Alterna estado de controles globales durante procesamiento en lote."""
        self._set_status_bar_busy(busy)
        if busy:
            self.progress_bar.setRange(0, 100)
            self.sidebar.btn_run_batch.setEnabled(False)
        else:
            self.sidebar._update_batch_btn_state()
        self.btn_open.setEnabled(not busy)
        self.btn_save.setEnabled(not busy and self.image_state.has_image)

    def start_batch_processing(
        self,
        input_dir: str,
        output_dir: str,
        model_id: str,
        enable_clothing_protection: bool,
    ) -> None:
        if self.ai.is_batch_running():
            logger.info("Procesamiento en lote ya en curso, ignorando solicitud duplicada.")
            return
        self._set_batch_busy(True)
        self._set_status("status_batch_preparing")

        def update_progress(current: int, total: int) -> None:
            if total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(current)

        def on_batch_finished(success_count: int, total: int, errors=None) -> None:
            self._set_batch_busy(False)
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

        def on_batch_failed(err_msg: str) -> None:
            self._set_batch_busy(False)
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

    def _on_brush_mode_toggled(self, enabled: bool) -> None:
        self.canvas.set_brush_mode(enabled)

    def _on_wand_mode_toggled(self, enabled: bool) -> None:
        self.canvas.set_wand_mode(enabled)
        if enabled and self.sidebar.is_wand_ai_mode():
            self._ensure_sam_embedding_async()

    def _on_wand_ai_mode_toggled(self, enabled: bool) -> None:
        self.canvas.set_wand_ai_mode(enabled)
        if enabled and self.sidebar.is_wand_tool_active():
            self._ensure_sam_embedding_async()

    def _ensure_sam_embedding_async(self) -> None:
        """Precomputa el embedding de MobileSAM en segundo plano si aún no está listo."""
        if not self.image_state.has_image:
            return
        if self._sam().has_embedding():
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

    def _on_sam_embedding_ready(self, _res: Any) -> None:
        self.progress_bar.setVisible(False)
        self.sidebar.set_wand_ai_status(tr("wand_ai_ready"))
        self._set_status("wand_ai_ready")
        pending = getattr(self.canvas, "_pending_wand_click", None)
        if pending is not None:
            self.canvas._pending_wand_click = None
            pt, mode = pending
            self.canvas._apply_magic_wand(pt, mode=mode)

    def _on_sam_embedding_failed(self, err_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.sidebar.set_wand_ai_status("Error")
        self._set_status("status_sam_error", error=err_msg)

    def _apply_device_badge(
        self,
        text: str,
        mode: str,
        object_name: str,
        *,
        is_gpu: bool,
        is_blocked: bool,
    ) -> None:
        """Actualiza el badge de dispositivo en barra inferior y sidebar."""
        if hasattr(self, "status_bar_widget"):
            self.status_bar_widget.set_device_badge(text, mode)
        elif hasattr(self, "lbl_device"):
            self.lbl_device.setText(text)
            self.lbl_device.setObjectName(object_name)
            self.lbl_device.style().unpolish(self.lbl_device)
            self.lbl_device.style().polish(self.lbl_device)
        if hasattr(self, "sidebar"):
            self.sidebar.set_device_mode(is_gpu=is_gpu, is_blocked=is_blocked)

    def update_device_badge(self) -> None:
        """Actualiza el badge de barra de estado y sidebar según el dispositivo activo."""
        if self._gpu().has_orphaned_worker():
            self.update_device_badge_blocked()
            return
        self.device = detect_device()
        is_gpu = "GPU" in self.device.upper()
        self._apply_device_badge(
            self.device,
            "gpu" if is_gpu else "cpu",
            "deviceBadgeGpu" if is_gpu else "deviceBadgeCpu",
            is_gpu=is_gpu,
            is_blocked=False,
        )

    def _toggle_device_mode(self) -> None:
        """Alterna entre GPU y CPU al hacer clic en el badge."""
        self._on_device_mode_changed(not self._gpu().is_gpu_enabled())

    def _on_device_mode_changed(self, is_gpu: bool) -> None:
        """Aplica el cambio de modo de dispositivo y actualiza estado."""
        _, status_code = self.ai.switch_device_mode(is_gpu)
        self.update_device_badge()
        if status_code.startswith("gpu_enabled:"):
            self._set_status("status_gpu_enabled", name=status_code.split(":", 1)[1])
        elif status_code in ("cpu_enabled", "gpu_failed"):
            self._set_status(f"status_{status_code}")
        else:
            self._set_status_raw(status_code)

    def update_device_badge_blocked(self) -> None:
        """Marca en UI que la GPU quedó bloqueada y se fuerza CPU."""
        self._apply_device_badge(
            tr("status_gpu_blocked_badge"),
            "blocked",
            "deviceBadgeBlocked",
            is_gpu=False,
            is_blocked=True,
        )

    @Slot(str)
    def _handle_gpu_status(self, message: str) -> None:
        """Slot UI para notificaciones asíncronas del GPUWorkerManager."""
        if self._gpu().has_orphaned_worker() or any(
            k in message for k in ("bloqueada", "reiniciar", "no pudo cerrarse")
        ):
            self.update_device_badge_blocked()
            self._set_status_raw(message)
        elif "GPU no respondió" in message or "GPU con error" in message:
            self._set_status_raw(message)

    def _on_worker_finished(self, result_image: Image.Image, description: str) -> None:
        gpu_mgr = self._gpu()
        last_status = gpu_mgr.get_last_status_message()

        if gpu_mgr.has_orphaned_worker():
            self.update_device_badge_blocked()

        self.set_processing(False)
        if "no pudo cerrarse" in last_status or "reiniciar" in last_status or gpu_mgr.has_orphaned_worker():
            self._set_status("status_gpu_orphaned_warn")
            gpu_mgr._last_status_message = ""
        elif "GPU no respondió" in last_status:
            self._set_status("status_gpu_timeout_cpu")
            gpu_mgr._last_status_message = ""
        elif "GPU con error" in last_status:
            self._set_status("status_gpu_error_cpu")
            gpu_mgr._last_status_message = ""
        else:
            self._set_status("status_completed", desc=description)

        brush_ref = None
        if "Escalado" in description or "Mejora de nitidez" in description:
            curr_brush = self.image_state.current_brush_reference
            if curr_brush:
                brush_ref = (
                    result_image.copy()
                    if not self.image_state.has_background_been_removed
                    else curr_brush.resize(result_image.size, resample=Image.Resampling.LANCZOS)
                )

        self.push_new_image_state(result_image, description, brush_reference=brush_ref)

    def _on_worker_failed(self, error_message: str) -> None:
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
    def _on_toggle_compare(self) -> None:
        """Activa o desactiva la vista comparativa antes/después con cortina."""
        if not self.image_state.has_image:
            self.btn_compare.setChecked(False)
            self._set_status("status_open_first")
            return
        if self.image_state._current_index == 0:
            self.btn_compare.setChecked(False)
            self.canvas.set_comparison_mode(False)
            self._set_status("status_no_changes_compare")
            return

        if self.btn_compare.isChecked():
            self.canvas.set_comparison_mode(True, self.image_state.original_image)
            self._set_status("status_compare_active")
        else:
            self.canvas.set_comparison_mode(False)
            self._set_status("status_compare_inactive")

    def perform_autocrop(self) -> None:
        """Recorta márgenes transparentes alrededor del sujeto (imagen o secuencia)."""
        is_seq = self.doc.has_sequence
        err_key, res = self.doc.apply_autocrop() if is_seq else self.doc.compute_autocrop()
        if err_key:
            self._set_status(err_key)
            return
        if res is None:
            return
        cropped, brush_ref, desc = res
        if is_seq:
            self._sync_after_document_edit(cropped, "status_applied", desc=desc)
        else:
            self.push_new_image_state(cropped, desc, brush_reference=brush_ref)

    def perform_apply_bg_color(self, color_rgb: Tuple[int, int, int]) -> None:
        """Compone la imagen sobre un fondo sólido en la capa del canvas."""
        if not self.image_state.has_image:
            return
        self.canvas.set_background_color(color_rgb)
        self._set_status("status_bg_color", hex=f"#{color_rgb[0]:02X}{color_rgb[1]:02X}{color_rgb[2]:02X}")

    def perform_edge_refinement(self, choke_px: int, feather_px: int, despill_strength: float) -> None:
        """Refina los bordes alfa sobre imagen o secuencia."""
        if not self.image_state.has_image:
            return
        if choke_px == 0 and feather_px == 0 and despill_strength <= 0.0:
            return
        out = self.doc.apply_edge_refinement(choke_px, feather_px, despill_strength)
        if out is not None:
            self._sync_after_document_edit(out, "status_edge_refined")

    def perform_inpaint_selection(self) -> None:
        """Rellena el área seleccionada mediante inpainting OpenCV."""
        if not self.image_state.has_image or not self.canvas.has_selection():
            self._set_status("status_selection_nothing")
            return
        out = self.doc.apply_inpaint_selection(self.canvas.active_selection)
        if out is not None:
            self._sync_after_document_edit(out, "status_inpainted", use_sequence_nav=False)

    def perform_apply_effects(self, cfg: dict) -> None:
        """Aplica efectos creativos (Sticker, Sombra, Bokeh/Imagen BG, Relación de Aspecto)."""
        if not self.image_state.has_image:
            return
        bg_img = None
        bg_path = str(cfg.get("bg_image_path", "") or "").strip()
        if bg_path:
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

    def perform_transform(self, rotate_deg: int, flip_h: bool, flip_v: bool) -> None:
        """Aplica rotación horaria de 90° o volteo horizontal/vertical."""
        if not self.image_state.has_image:
            return
        out = self.doc.apply_transform(rotate_deg=rotate_deg, flip_h=flip_h, flip_v=flip_v)
        if out is not None:
            self._sync_after_document_edit(out, "status_transformed", reset_zoom=True)

    def on_crop_mode_toggled(self, enabled: bool, aspect_ratio: object) -> None:
        """Activa o desactiva la superposición interactiva de recorte en el lienzo."""
        if enabled and not self.image_state.has_image:
            self.sidebar.panel_effects.reset_crop_toggle()
            return
        ratio = float(aspect_ratio) if aspect_ratio is not None else None
        self.canvas.set_crop_mode(bool(enabled), aspect_ratio=ratio)

    def perform_manual_crop(self) -> None:
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

    def perform_restore_transparency(self) -> None:
        """Restaura el fondo transparente ocultando la capa de color sólido."""
        if not self.image_state.has_image:
            return
        self.canvas.set_background_color(None)
        self._set_status("status_bg_transparent")

    def start_detect_subjects(self) -> None:
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

    def _on_yolo_finished(self, subjects: list) -> None:
        self.progress_bar.setVisible(False)
        self.sidebar.set_detecting_subjects_state(False)
        self.sidebar.set_detected_subjects(subjects)
        if subjects:
            self._set_status("status_yolo_detected", count=len(subjects))
            self._ensure_sam_embedding_async()
        else:
            self._set_status("detect_none_found")

    def _on_yolo_failed(self, err_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.sidebar.set_detecting_subjects_state(False)
        self._set_status("status_yolo_error", error=err_msg)

    def _refine_yolo_subject_mask(self, subject: Any) -> Optional[np.ndarray]:
        """Delega el refinamiento HD de la máscara YOLO11-seg al módulo de dominio."""
        if not self.image_state.has_image or subject is None:
            return None
        from app.core.yolo_detector import refine_yolo_subject_mask
        return refine_yolo_subject_mask(self.image_state.current_image, subject)

    def perform_isolate_subject(self, subject: Any) -> None:
        """Aísla el sujeto detectado sobre la imagen o toda la secuencia."""
        desc = f"Aislar {subject.class_name.capitalize()}"
        if self.doc.has_sequence:
            isolated_curr = self.doc.apply_isolate_subject_sequence(subject)
            if isolated_curr is not None:
                self._sync_after_document_edit(isolated_curr, "status_applied", desc=desc)
            return

        isolated_img = self.doc.build_isolated_subject(subject)
        if isolated_img is not None:
            self.push_new_image_state(isolated_img, desc)

    def perform_select_subject(self, subject: Any) -> None:
        """Carga la máscara refinada del sujeto detectado en la selección activa."""
        if not self.image_state.has_image or subject is None:
            return
        high_res_mask = self._refine_yolo_subject_mask(subject)
        if high_res_mask is None:
            high_res_mask = subject.mask.astype(np.uint8) * 255
        self.canvas._set_selection(high_res_mask)
        self._set_status("status_subject_selected", name=subject.class_name.capitalize())

    def closeEvent(self, event: Any) -> None:
        """Finaliza hilos activos de manera segura al cerrar la ventana."""
        if hasattr(self, "gif_timer") and self.gif_timer.isActive():
            self.gif_timer.stop()
        self.ai.shutdown_workers(timeout_ms=1000)
        try:
            super().closeEvent(event)
        except Exception:
            if hasattr(event, "accept"):
                event.accept()
