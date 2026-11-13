"""
TimoitaseBG - Contenedor principal del panel lateral derecho (Sidebar).
Orquesta la cabecera fija, el rail de pestañas y los subpaneles modulares en app/ui/panels/.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QButtonGroup,
    QFrame,
    QScrollArea,
    QSizePolicy,
)

from app.i18n import tr
from app.ui.icons import get_icon
from app.ui.widgets.toggles import (
    AutoResizingStackedWidget,
    ToggleSwitch,
    GpuToggle,
    _GpuToggleAdapter,
)
from app.ui.panels import BackgroundPanel, RestorePanel, BrushPanel, BatchPanel

__all__ = [
    "AutoResizingStackedWidget",
    "ToggleSwitch",
    "GpuToggle",
    "_GpuToggleAdapter",
    "ToolSidebar",
]


class ToolSidebar(QWidget):
    """
    Panel lateral derecho con selector de herramientas estilo rail:
    1. Quitar Fondo
    2. Restaurar
    3. Pincel Mágico
    4. Lotes (Batch)
    """

    removeBgRequested = Signal(str, bool)
    restoreRequested = Signal(str, bool)
    autoCropRequested = Signal()
    applyBgColorRequested = Signal(tuple)
    restoreTransparencyRequested = Signal()
    brushModeToggled = Signal(bool)
    eraserModeToggled = Signal(bool)
    brushSizeChanged = Signal(int)
    brushSmoothnessChanged = Signal(int)
    brushOpacityChanged = Signal(float)
    brushHardnessChanged = Signal(float)
    wandModeToggled = Signal(bool)
    wandAIModeToggled = Signal(bool)
    wandToleranceChanged = Signal(int)
    wandSelectionModeChanged = Signal(str)
    wandAntiAliasChanged = Signal(bool)
    wandContiguousChanged = Signal(bool)
    deleteSelectionRequested = Signal()
    invertSelectionRequested = Signal()
    selectAllRequested = Signal()
    clearSelectionRequested = Signal()
    clearMaskRequested = Signal()
    inpaintRequested = Signal()
    deviceModeChanged = Signal(bool)
    batchRequested = Signal(str, str, str, bool)
    detectSubjectsRequested = Signal()
    isolateSubjectRequested = Signal(object)
    selectSubjectRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self.setFixedWidth(380)
        self.is_dark_theme: bool = True
        self._setup_ui()
        self._init_accel_state()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Cabecera fija superior
        self.sidebar_header = QWidget()
        self.sidebar_header.setObjectName("sidebar_header")
        self.sidebar_header.setStyleSheet("background: transparent;")
        self.sidebar_header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        header_layout = QVBoxLayout(self.sidebar_header)
        header_layout.setContentsMargins(16, 14, 16, 6)
        header_layout.setSpacing(4)

        header_top_row = QHBoxLayout()
        header_top_row.setContentsMargins(0, 0, 0, 0)
        header_top_row.setSpacing(8)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self._title_label = QLabel(tr("sidebar_title"))
        self._title_label.setObjectName("SectionTitle")
        self._title_label.setStyleSheet("background: transparent;")
        self._subtitle_label = QLabel(tr("sidebar_subtitle"))
        self._subtitle_label.setObjectName("SubTitle")
        self._subtitle_label.setStyleSheet("background: transparent;")
        text_layout.addWidget(self._title_label)
        text_layout.addWidget(self._subtitle_label)
        header_top_row.addLayout(text_layout)
        header_top_row.addStretch()

        self.gpu_toggle = GpuToggle(self)
        self.gpu_toggle.toggled.connect(self.deviceModeChanged.emit)
        header_top_row.addWidget(self.gpu_toggle, alignment=Qt.AlignmentFlag.AlignVCenter)

        header_layout.addLayout(header_top_row)
        main_layout.addWidget(self.sidebar_header)

        # Rail de pestañas fijo
        self.tool_tabs_row = QWidget()
        self.tool_tabs_row.setObjectName("tool_tabs_row")
        self.tool_tabs_row.setStyleSheet("background: transparent;")
        self.tool_tabs_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tabs_layout = QHBoxLayout(self.tool_tabs_row)
        tabs_layout.setContentsMargins(16, 2, 16, 8)
        tabs_layout.setSpacing(6)

        self.rail_group = QButtonGroup(self)
        self.rail_group.setExclusive(True)

        self.btn_rail_bg = self._make_rail_tab("sidebar_tab_bg", 0, checked=True)
        self.btn_rail_restore = self._make_rail_tab("sidebar_tab_restore", 1)
        self.btn_rail_brush = self._make_rail_tab("sidebar_tab_brush", 2)
        self.btn_rail_batch = self._make_rail_tab("sidebar_tab_batch", 3)

        for btn in (self.btn_rail_bg, self.btn_rail_restore, self.btn_rail_brush, self.btn_rail_batch):
            tabs_layout.addWidget(btn)

        main_layout.addWidget(self.tool_tabs_row)

        # Área scrolleable de paneles
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("toolScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scroll_content")
        self.scroll_content.setMaximumWidth(380)
        scroll_layout = QVBoxLayout(self.scroll_content)
        scroll_layout.setContentsMargins(16, 16, 20, 16)
        scroll_layout.setSpacing(12)

        self.stack = AutoResizingStackedWidget()
        self.stack.setObjectName("toolsStack")

        self.panel_bg = BackgroundPanel()
        self.panel_restore = RestorePanel()
        self.panel_brush = BrushPanel()
        self.panel_batch = BatchPanel()
        self._panels = (self.panel_bg, self.panel_restore, self.panel_brush, self.panel_batch)

        for panel in self._panels:
            self.stack.addWidget(panel)

        scroll_layout.addWidget(self.stack)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, stretch=1)

        self._connect_panel_signals()
        self.rail_group.idClicked.connect(self._on_rail_clicked)
        self.update_theme_icons(is_dark=True)

    def _make_rail_tab(self, label_key: str, idx: int, checked: bool = False) -> QPushButton:
        btn = QPushButton(tr(label_key))
        btn.setObjectName("ToolTab")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setFixedHeight(44)
        self.rail_group.addButton(btn, idx)
        return btn

    def _connect_panel_signals(self):
        self.panel_bg.removeBgRequested.connect(self.removeBgRequested.emit)
        self.panel_bg.autoCropRequested.connect(self.autoCropRequested.emit)
        self.panel_bg.applyBgColorRequested.connect(self.applyBgColorRequested.emit)
        self.panel_bg.restoreTransparencyRequested.connect(self.restoreTransparencyRequested.emit)
        self.panel_bg.detectSubjectsRequested.connect(self.detectSubjectsRequested.emit)
        self.panel_bg.isolateSubjectRequested.connect(self.isolateSubjectRequested.emit)
        self.panel_bg.selectSubjectRequested.connect(self.selectSubjectRequested.emit)

        self.panel_restore.restoreRequested.connect(self.restoreRequested.emit)

        self.panel_brush.eraserModeToggled.connect(self.eraserModeToggled.emit)
        self.panel_brush.wandModeToggled.connect(self.wandModeToggled.emit)
        self.panel_brush.wandAIModeToggled.connect(self.wandAIModeToggled.emit)
        self.panel_brush.brushSizeChanged.connect(self.brushSizeChanged.emit)
        self.panel_brush.brushSmoothnessChanged.connect(self.brushSmoothnessChanged.emit)
        self.panel_brush.brushOpacityChanged.connect(self.brushOpacityChanged.emit)
        self.panel_brush.brushHardnessChanged.connect(self.brushHardnessChanged.emit)
        self.panel_brush.wandToleranceChanged.connect(self.wandToleranceChanged.emit)
        self.panel_brush.wandSelectionModeChanged.connect(self.wandSelectionModeChanged.emit)
        self.panel_brush.wandAntiAliasChanged.connect(self.wandAntiAliasChanged.emit)
        self.panel_brush.wandContiguousChanged.connect(self.wandContiguousChanged.emit)
        self.panel_brush.deleteSelectionRequested.connect(self.deleteSelectionRequested.emit)
        self.panel_brush.invertSelectionRequested.connect(self.invertSelectionRequested.emit)
        self.panel_brush.selectAllRequested.connect(self.selectAllRequested.emit)
        self.panel_brush.clearSelectionRequested.connect(self.clearSelectionRequested.emit)
        self.panel_brush.geometryChanged.connect(self._refresh_stack_geometry)

        self.panel_batch.batchRequested.connect(self.batchRequested.emit)

    def _on_rail_clicked(self, index: int):
        self.stack.setCurrentIndex(index)
        self.scroll_area.verticalScrollBar().setValue(0)
        self.brushModeToggled.emit(index == 2)

    def _refresh_stack_geometry(self):
        self.panel_brush.updateGeometry()
        self.panel_brush.adjustSize()
        self.stack.updateGeometry()
        self.scroll_content.adjustSize()

    # --- Tema e Idioma uniformes ---
    def update_theme_icons(self, is_dark: bool = True):
        """Aplica y refresca iconos vectoriales y temas en cabecera, rail y subpaneles."""
        self.is_dark_theme = is_dark
        self.gpu_toggle.set_dark_theme(is_dark)

        accent_color = "#FF4F79"
        sub_color = "#999999" if is_dark else "#78716C"

        for btn, icon_name in (
            (self.btn_rail_bg, "sparkles"),
            (self.btn_rail_restore, "maximize-2"),
            (self.btn_rail_brush, "brush"),
            (self.btn_rail_batch, "folder"),
        ):
            btn.setIcon(get_icon(icon_name, color=sub_color, active_color=accent_color, size=18))
            btn.setIconSize(QSize(18, 18))

        for panel in self._panels:
            panel.apply_theme(is_dark)

    def retranslate_ui(self):
        """Actualiza etiquetas, combos y tooltips de la barra lateral y sus subpaneles."""
        self._title_label.setText(tr("sidebar_title"))
        self._subtitle_label.setText(tr("sidebar_subtitle"))
        self.btn_rail_bg.setText(tr("sidebar_tab_bg"))
        self.btn_rail_restore.setText(tr("sidebar_tab_restore"))
        self.btn_rail_brush.setText(tr("sidebar_tab_brush"))
        self.btn_rail_batch.setText(tr("sidebar_tab_batch"))
        for panel in self._panels:
            panel.retranslate_ui()

    # --- Estado de aceleración ---
    @property
    def btn_accel_gpu(self):
        return _GpuToggleAdapter(self.gpu_toggle, is_gpu_target=True)

    @property
    def btn_accel_cpu(self):
        return _GpuToggleAdapter(self.gpu_toggle, is_gpu_target=False)

    def _init_accel_state(self):
        try:
            from app.core.gpu_manager import get_gpu_manager
            gpu_mgr = get_gpu_manager()
            self.set_device_mode(
                is_gpu=gpu_mgr.is_gpu_enabled(),
                is_blocked=gpu_mgr.has_orphaned_worker(),
            )
        except Exception:
            pass

    def set_device_mode(self, is_gpu: bool, is_blocked: bool = False):
        self.gpu_toggle.set_state(is_gpu, is_blocked)

    # --- API pública delegada a los subpaneles (compatibilidad con MainWindow y tests) ---
    def is_clothing_protection_enabled(self) -> bool:
        return self.panel_bg.is_clothing_protection_enabled()

    def set_has_mask(self, has_mask: bool):
        pass

    def set_transparency_available(self, has_transparency: bool, can_restore: bool):
        self.panel_bg.set_transparency_available(has_transparency, can_restore)

    def set_busy(self, busy: bool):
        self.panel_bg.set_busy(busy)
        self.panel_restore.set_busy(busy)

    def set_detected_subjects(self, subjects: list):
        self.panel_bg.set_detected_subjects(subjects)

    def set_detecting_subjects_state(self, is_detecting: bool):
        self.panel_bg.set_detecting_subjects_state(is_detecting)

    def set_selection_info(self, selected_px: int, total_px: int):
        self.panel_brush.set_selection_info(selected_px, total_px)

    def set_wand_ai_status(self, text: str):
        self.panel_brush.set_wand_ai_status(text)

    def is_wand_ai_mode(self) -> bool:
        return self.panel_brush.is_wand_ai_mode()

    def is_wand_tool_active(self) -> bool:
        return self.panel_brush.is_wand_tool_active()

    def set_brush_size(self, size: int):
        self.panel_brush.set_brush_size(size)

    def _update_batch_btn_state(self):
        self.panel_batch._update_batch_btn_state()

    # Alias de atributos internos para tests y referencias directas
    @property
    def combo_bg_model(self):
        return self.panel_bg.combo_bg_model

    @property
    def chk_clothing_protection(self):
        return self.panel_bg.chk_clothing_protection

    @property
    def btn_remove_bg(self):
        return self.panel_bg.btn_remove_bg

    @property
    def btn_autocrop(self):
        return self.panel_bg.btn_autocrop

    @property
    def chips_layout(self):
        return self.panel_bg.chips_layout

    @property
    def subject_actions_widget(self):
        return self.panel_bg.subject_actions_widget

    @property
    def btn_isolate_subject(self):
        return self.panel_bg.btn_isolate_subject

    @property
    def btn_select_subject(self):
        return self.panel_bg.btn_select_subject

    @property
    def btn_restore(self):
        return self.panel_restore.btn_restore

    @property
    def combo_restore_scale(self):
        return self.panel_restore.combo_restore_scale

    @property
    def chk_tiling(self):
        return self.panel_restore.chk_tiling

    @property
    def btn_tool_paint(self):
        return self.panel_brush.btn_tool_paint

    @property
    def btn_tool_erase(self):
        return self.panel_brush.btn_tool_erase

    @property
    def btn_tool_wand(self):
        return self.panel_brush.btn_tool_wand

    @property
    def btn_wand(self):
        return self.panel_brush.btn_wand

    @property
    def btn_wand_type_color(self):
        return self.panel_brush.btn_wand_type_color

    @property
    def btn_wand_type_ai(self):
        return self.panel_brush.btn_wand_type_ai

    @property
    def wand_color_widget(self):
        return self.panel_brush.wand_color_widget

    @property
    def wand_ai_widget(self):
        return self.panel_brush.wand_ai_widget

    @property
    def wand_mode_buttons(self):
        return self.panel_brush.wand_mode_buttons

    @property
    def btn_delete_selection(self):
        return self.panel_brush.btn_delete_selection

    @property
    def btn_invert_selection(self):
        return self.panel_brush.btn_invert_selection

    @property
    def btn_select_all(self):
        return self.panel_brush.btn_select_all

    @property
    def btn_clear_selection(self):
        return self.panel_brush.btn_clear_selection

    @property
    def combo_batch_model(self):
        return self.panel_batch.combo_batch_model

    @property
    def btn_run_batch(self):
        return self.panel_batch.btn_run_batch

    @property
    def btn_batch_input(self):
        return self.panel_batch.btn_batch_input

    @property
    def btn_batch_output(self):
        return self.panel_batch.btn_batch_output
