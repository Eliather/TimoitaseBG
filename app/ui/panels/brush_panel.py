"""
Panel de edición manual (Pincel, Borrador, Varita por Color e IA MobileSAM) para el sidebar.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QCheckBox,
    QButtonGroup,
    QGroupBox,
)

from app.config import DEFAULT_BRUSH_SIZE, MIN_BRUSH_SIZE, MAX_BRUSH_SIZE
from app.i18n import tr
from app.ui.icons import get_icon
from app.ui.widgets.toggles import create_section_header


class BrushPanel(QWidget):
    """Panel modular para pincel, borrador y selección con varita clásica o MobileSAM."""

    eraserModeToggled = Signal(bool)
    wandModeToggled = Signal(bool)
    wandAIModeToggled = Signal(bool)
    brushSizeChanged = Signal(int)
    brushSmoothnessChanged = Signal(int)
    brushOpacityChanged = Signal(float)
    brushHardnessChanged = Signal(float)
    wandToleranceChanged = Signal(int)
    wandSelectionModeChanged = Signal(str)
    wandAntiAliasChanged = Signal(bool)
    wandContiguousChanged = Signal(bool)
    deleteSelectionRequested = Signal()
    inpaintSelectionRequested = Signal()
    invertSelectionRequested = Signal()
    selectAllRequested = Signal()
    clearSelectionRequested = Signal()
    edgeRefineRequested = Signal(int, int, float)
    geometryChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_selection: bool = False
        self._active_brush_tool: int = 0
        self._selected_px: int = 0
        self._total_px: int = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_brush, self._lbl_brush_hdr, self._info_brush_hdr = create_section_header(
            tr("header_brush"),
            tr("brush_desc"),
        )
        layout.addLayout(header_brush)
        layout.addLayout(self._build_tool_mode_row())
        layout.addWidget(self._build_brush_options_box())
        layout.addWidget(self._build_wand_options_box())
        layout.addWidget(self._build_selection_box())
        layout.addWidget(self._build_edge_refine_box())

        self.set_selection_info(0, 0)
        self._apply_brush_tool_visibility()
        layout.addStretch()

    def _build_tool_mode_row(self) -> QHBoxLayout:
        """Crea la fila exclusiva de modos (Pintar / Borrar / Varita)."""
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)

        self.brush_mode_group = QButtonGroup(self)
        self.brush_mode_group.setExclusive(True)

        self.btn_tool_paint = self._make_mode_button("btn_paint", checked=True)
        self.btn_tool_erase = self._make_mode_button("btn_erase")
        self.btn_tool_wand = self._make_mode_button("btn_wand", tip_key="tip_wand")

        for idx, btn in enumerate((self.btn_tool_paint, self.btn_tool_erase, self.btn_tool_wand)):
            self.brush_mode_group.addButton(btn, idx)
            mode_row.addWidget(btn)
        self.brush_mode_group.idClicked.connect(self._on_brush_tool_clicked)
        return mode_row

    def _build_brush_options_box(self) -> QGroupBox:
        """Construye los controles de tamaño, suavizado, opacidad y dureza del pincel."""
        self.brush_options_box = QGroupBox(tr("brush_settings"))
        self.brush_options_box.setObjectName("brushOptionsBox")
        b_layout = QVBoxLayout(self.brush_options_box)
        b_layout.setContentsMargins(12, 14, 12, 10)
        b_layout.setSpacing(10)

        self.slider_brush_size, self.lbl_brush_size, self._lbl_size_name = self._add_slider_row(
            b_layout,
            tr("lbl_size_thick"),
            50,
            MIN_BRUSH_SIZE,
            MAX_BRUSH_SIZE,
            DEFAULT_BRUSH_SIZE,
            f"{DEFAULT_BRUSH_SIZE} px",
            self._on_slider_size_changed,
            label_width=42,
            value_obj_name="brushSizeLabel",
        )
        self.slider_brush_smooth, self.lbl_brush_smooth, self._lbl_smooth_name = self._add_slider_row(
            b_layout,
            tr("lbl_smoothness"),
            50,
            0,
            100,
            0,
            "0 %",
            self._on_slider_smooth_changed,
            label_width=42,
            value_obj_name="brushSizeLabel",
        )
        self.slider_brush_opacity, self.lbl_brush_opacity, self._lbl_op_name = self._add_slider_row(
            b_layout,
            tr("lbl_opacity"),
            60,
            1,
            100,
            100,
            "100 %",
            self._on_slider_opacity_changed,
            label_width=40,
        )
        self.slider_brush_hardness, self.lbl_brush_hardness, self._lbl_hd_name = self._add_slider_row(
            b_layout,
            tr("lbl_hardness"),
            60,
            0,
            100,
            100,
            "100 %",
            self._on_slider_hardness_changed,
            label_width=40,
        )

        presets_row = QHBoxLayout()
        presets_row.setSpacing(6)
        for sz in (2, 10, 25, 50, 100):
            btn = QPushButton(f"{sz}px")
            btn.setObjectName("presetButton")
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda _, s=sz: self.set_brush_size(s))
            presets_row.addWidget(btn)
        b_layout.addLayout(presets_row)
        return self.brush_options_box

    def _build_wand_options_box(self) -> QGroupBox:
        """Construye los ajustes de Varita Mágica por Color y Varita IA (MobileSAM)."""
        self.wand_options_box = QGroupBox(tr("wand_settings"))
        self.wand_options_box.setObjectName("wandOptionsBox")
        w_layout = QVBoxLayout(self.wand_options_box)
        w_layout.setContentsMargins(12, 14, 12, 10)
        w_layout.setSpacing(10)

        wand_type_row = QHBoxLayout()
        wand_type_row.setSpacing(6)
        self.wand_type_group = QButtonGroup(self)
        self.wand_type_group.setExclusive(True)

        self.btn_wand_type_color = self._make_wand_type_button("wand_type_color", "tip_wand_color", checked=True)
        self.btn_wand_type_ai = self._make_wand_type_button("wand_type_ai", "tip_wand_ai")
        self.wand_type_group.addButton(self.btn_wand_type_color, 0)
        self.wand_type_group.addButton(self.btn_wand_type_ai, 1)
        wand_type_row.addWidget(self.btn_wand_type_color)
        wand_type_row.addWidget(self.btn_wand_type_ai)
        w_layout.addLayout(wand_type_row)

        self.btn_wand_type_color.clicked.connect(lambda: self._on_wand_type_clicked(False))
        self.btn_wand_type_ai.clicked.connect(lambda: self._on_wand_type_clicked(True))

        self._lbl_sel_mode = QLabel(tr("lbl_wand_mode"))
        self._lbl_sel_mode.setObjectName("fieldLabel")
        w_layout.addWidget(self._lbl_sel_mode)

        sel_mode_row = QHBoxLayout()
        sel_mode_row.setSpacing(4)
        self.wand_mode_group = QButtonGroup(self)
        self.wand_mode_group.setExclusive(True)
        self._wand_mode_ids = {}
        self.wand_mode_buttons = []
        self._wand_mode_tips = []
        for idx, (mode_id, icon_name, tip_key) in enumerate([
            ("new", "sel-new", "tip_wand_new"),
            ("add", "sel-add", "tip_wand_add"),
            ("subtract", "sel-subtract", "tip_wand_subtract"),
            ("intersect", "sel-intersect", "tip_wand_intersect"),
        ]):
            btn = QPushButton()
            btn.setObjectName("selectionModeButton")
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setToolTip(tr(tip_key))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if mode_id == "new":
                btn.setChecked(True)
            self.wand_mode_group.addButton(btn, idx)
            self._wand_mode_ids[idx] = mode_id
            self.wand_mode_buttons.append((btn, icon_name))
            self._wand_mode_tips.append((btn, tip_key))
            sel_mode_row.addWidget(btn)
        self.wand_mode_group.idClicked.connect(
            lambda i: self.wandSelectionModeChanged.emit(self._wand_mode_ids[i])
        )
        w_layout.addLayout(sel_mode_row)

        self.wand_color_widget = QWidget()
        cw_layout = QVBoxLayout(self.wand_color_widget)
        cw_layout.setContentsMargins(0, 0, 0, 0)
        cw_layout.setSpacing(8)

        self.slider_wand_tolerance, self.lbl_wand_tolerance, self._lbl_tol = self._add_slider_row(
            cw_layout,
            tr("lbl_tolerance"),
            72,
            0,
            255,
            32,
            "32",
            self._on_slider_wand_tolerance_changed,
            label_width=34,
            value_obj_name="brushSizeLabel",
            tooltip=tr("tip_tolerance"),
        )

        checks_row = QHBoxLayout()
        self.chk_wand_anti_alias = QCheckBox(tr("chk_anti_alias"))
        self.chk_wand_anti_alias.setChecked(True)
        self.chk_wand_anti_alias.setToolTip(tr("tip_anti_alias"))
        self.chk_wand_anti_alias.toggled.connect(self.wandAntiAliasChanged.emit)

        self.chk_wand_contiguous = QCheckBox(tr("chk_contiguous"))
        self.chk_wand_contiguous.setChecked(True)
        self.chk_wand_contiguous.setToolTip(tr("tip_contiguous"))
        self.chk_wand_contiguous.toggled.connect(self.wandContiguousChanged.emit)

        checks_row.addWidget(self.chk_wand_anti_alias)
        checks_row.addWidget(self.chk_wand_contiguous)
        checks_row.addStretch()
        cw_layout.addLayout(checks_row)
        w_layout.addWidget(self.wand_color_widget)

        self.wand_ai_widget = QWidget()
        ai_layout = QVBoxLayout(self.wand_ai_widget)
        ai_layout.setContentsMargins(0, 0, 0, 0)
        ai_layout.setSpacing(6)
        self.lbl_wand_ai_desc = QLabel(tr("wand_ai_desc"))
        self.lbl_wand_ai_desc.setObjectName("shortcutDesc")
        self.lbl_wand_ai_desc.setWordWrap(True)
        ai_layout.addWidget(self.lbl_wand_ai_desc)
        self.lbl_wand_ai_status = QLabel(tr("wand_ai_ready"))
        self.lbl_wand_ai_status.setObjectName("fieldLabel")
        ai_layout.addWidget(self.lbl_wand_ai_status)
        self.wand_ai_widget.setVisible(False)
        w_layout.addWidget(self.wand_ai_widget)

        self._lbl_wand_hint = QLabel(tr("wand_hint"))
        self._lbl_wand_hint.setObjectName("shortcutDesc")
        self._lbl_wand_hint.setWordWrap(True)
        w_layout.addWidget(self._lbl_wand_hint)
        return self.wand_options_box

    def _build_selection_box(self) -> QGroupBox:
        """Construye acciones directas sobre la selección (Borrar / Rellenar / Invertir)."""
        self.selection_box = QGroupBox(tr("selection_box"))
        self.selection_box.setObjectName("selectionBox")
        s_layout = QVBoxLayout(self.selection_box)
        s_layout.setContentsMargins(12, 14, 12, 10)
        s_layout.setSpacing(8)

        self.lbl_selection_info = QLabel(tr("selection_none"))
        self.lbl_selection_info.setObjectName("shortcutDesc")
        self.lbl_selection_info.setWordWrap(True)
        s_layout.addWidget(self.lbl_selection_info)

        self.btn_delete_selection = QPushButton(tr("btn_delete_selection"))
        self.btn_delete_selection.setObjectName("dangerActionButton")
        self.btn_delete_selection.setFixedHeight(38)
        self.btn_delete_selection.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_selection.setToolTip(tr("tip_delete_selection"))
        self.btn_delete_selection.clicked.connect(self.deleteSelectionRequested.emit)
        s_layout.addWidget(self.btn_delete_selection)

        self.btn_inpaint_selection = QPushButton(tr("btn_inpaint_selection"))
        self.btn_inpaint_selection.setObjectName("PrimaryBtn")
        self.btn_inpaint_selection.setFixedHeight(34)
        self.btn_inpaint_selection.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_inpaint_selection.setToolTip(tr("tip_inpaint_selection"))
        self.btn_inpaint_selection.clicked.connect(self.inpaintSelectionRequested.emit)
        s_layout.addWidget(self.btn_inpaint_selection)

        sel_actions_row = QHBoxLayout()
        sel_actions_row.setSpacing(6)
        self.btn_invert_selection = self._make_sel_action("btn_invert_selection", "tip_invert_selection", self.invertSelectionRequested.emit)
        self.btn_select_all = self._make_sel_action("btn_select_all", "tip_select_all", self.selectAllRequested.emit)
        self.btn_clear_selection = self._make_sel_action("btn_clear_selection", "tip_clear_selection", self.clearSelectionRequested.emit)
        sel_actions_row.addWidget(self.btn_invert_selection)
        sel_actions_row.addWidget(self.btn_select_all)
        sel_actions_row.addWidget(self.btn_clear_selection)
        s_layout.addLayout(sel_actions_row)
        return self.selection_box

    def _build_edge_refine_box(self) -> QGroupBox:
        """Construye controles de refinamiento alfa (Choke / Feather / Despill)."""
        self.edge_refine_box = QGroupBox(tr("edge_refine_box"))
        self.edge_refine_box.setObjectName("edgeRefineBox")
        er_layout = QVBoxLayout(self.edge_refine_box)
        er_layout.setContentsMargins(12, 14, 12, 10)
        er_layout.setSpacing(8)

        self.slider_choke, self.lbl_choke, self._lbl_choke_name = self._add_slider_row(
            er_layout,
            tr("lbl_edge_choke"),
            72,
            -8,
            8,
            0,
            "0 px",
            lambda v: self.lbl_choke.setText(f"{v:+d} px" if v != 0 else "0 px"),
            label_width=44,
        )
        self.slider_feather, self.lbl_feather, self._lbl_feather_name = self._add_slider_row(
            er_layout,
            tr("lbl_edge_feather"),
            72,
            0,
            15,
            0,
            "0 px",
            lambda v: self.lbl_feather.setText(f"{v} px"),
            label_width=44,
        )
        self.slider_despill, self.lbl_despill, self._lbl_despill_name = self._add_slider_row(
            er_layout,
            tr("lbl_edge_despill"),
            72,
            0,
            100,
            0,
            "0 %",
            lambda v: self.lbl_despill.setText(f"{v} %"),
            label_width=44,
        )
        self.btn_apply_edge_refine = QPushButton(tr("btn_apply_edge_refine"))
        self.btn_apply_edge_refine.setObjectName("ActionBtn")
        self.btn_apply_edge_refine.setFixedHeight(34)
        self.btn_apply_edge_refine.clicked.connect(
            lambda: self.edgeRefineRequested.emit(
                int(self.slider_choke.value()),
                int(self.slider_feather.value()),
                self.slider_despill.value() / 100.0,
            )
        )
        er_layout.addWidget(self.btn_apply_edge_refine)
        return self.edge_refine_box

    def _make_wand_type_button(self, text_key: str, tip_key: str, checked: bool = False) -> QPushButton:
        btn = QPushButton(tr(text_key))
        btn.setObjectName("presetButton")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setFixedHeight(30)
        btn.setToolTip(tr(tip_key))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _make_mode_button(self, text_key: str, checked: bool = False, tip_key: str = "") -> QPushButton:
        btn = QPushButton(tr(text_key))
        btn.setObjectName("brushToolModeButton")
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setFixedHeight(32)
        if tip_key:
            btn.setToolTip(tr(tip_key))
        return btn

    def _make_sel_action(self, text_key: str, tip_key: str, slot) -> QPushButton:
        btn = QPushButton(tr(text_key))
        btn.setObjectName("presetButton")
        btn.setFixedHeight(30)
        btn.setToolTip(tr(tip_key))
        btn.clicked.connect(slot)
        return btn

    def _add_slider_row(
        self,
        parent_layout,
        title: str,
        title_width: int,
        min_val: int,
        max_val: int,
        default_val: int,
        initial_text: str,
        callback,
        label_width: int = 42,
        value_obj_name: str = "",
        tooltip: str = "",
    ):
        row = QHBoxLayout()
        name_lbl = QLabel(title)
        name_lbl.setFixedWidth(title_width)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        if tooltip:
            name_lbl.setToolTip(tooltip)
            slider.setToolTip(tooltip)
        slider.valueChanged.connect(callback)

        val_lbl = QLabel(initial_text)
        if value_obj_name:
            val_lbl.setObjectName(value_obj_name)
        val_lbl.setFixedWidth(label_width)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.addWidget(name_lbl)
        row.addWidget(slider)
        row.addWidget(val_lbl)
        parent_layout.addLayout(row)
        return slider, val_lbl, name_lbl

    def apply_theme(self, is_dark: bool):
        fg_color = "#EAEAEA" if is_dark else "#292524"
        accent_color = "#FF4F79"
        dis_color = "#666666" if is_dark else "#A8A29E"

        for btn, icon_name in (
            (self.btn_tool_paint, "brush"),
            (self.btn_tool_erase, "eraser"),
            (self.btn_tool_wand, "wand"),
        ):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, size=16))
            btn.setIconSize(QSize(16, 16))

        for btn, icon_name in self.wand_mode_buttons:
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            btn.setIconSize(QSize(16, 16))

        self.btn_delete_selection.setIcon(get_icon("trash", color="#FFFFFF", size=16))
        self.btn_delete_selection.setIconSize(QSize(16, 16))
        self.btn_inpaint_selection.setIcon(get_icon("sparkles", color="#FFFFFF", size=16))
        self.btn_inpaint_selection.setIconSize(QSize(16, 16))
        self.btn_apply_edge_refine.setIcon(get_icon("sliders", color="#FFFFFF", size=15))
        self.btn_apply_edge_refine.setIconSize(QSize(15, 15))
        for btn, icon_name in (
            (self.btn_invert_selection, "invert"),
            (self.btn_select_all, "select-all"),
            (self.btn_clear_selection, "deselect"),
        ):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=14))
            btn.setIconSize(QSize(14, 14))

    def retranslate_ui(self):
        self._lbl_brush_hdr.setText(tr("header_brush"))
        self._info_brush_hdr.setToolTip(tr("brush_desc"))
        self.btn_tool_paint.setText(tr("btn_paint"))
        self.btn_tool_erase.setText(tr("btn_erase"))
        self.btn_tool_wand.setText(tr("btn_wand"))
        self.btn_tool_wand.setToolTip(tr("tip_wand"))

        self.brush_options_box.setTitle(tr("brush_settings"))
        self._lbl_size_name.setText(tr("lbl_size_thick"))
        self._lbl_smooth_name.setText(tr("lbl_smoothness"))
        self._lbl_op_name.setText(tr("lbl_opacity"))
        self._lbl_hd_name.setText(tr("lbl_hardness"))

        self.wand_options_box.setTitle(tr("wand_settings"))
        self.btn_wand_type_color.setText(tr("wand_type_color"))
        self.btn_wand_type_color.setToolTip(tr("tip_wand_color"))
        self.btn_wand_type_ai.setText(tr("wand_type_ai"))
        self.btn_wand_type_ai.setToolTip(tr("tip_wand_ai"))
        self._lbl_sel_mode.setText(tr("lbl_wand_mode"))
        for btn, tip_key in self._wand_mode_tips:
            btn.setToolTip(tr(tip_key))
        self._lbl_tol.setText(tr("lbl_tolerance"))
        self._lbl_tol.setToolTip(tr("tip_tolerance"))
        self.slider_wand_tolerance.setToolTip(tr("tip_tolerance"))
        self.chk_wand_anti_alias.setText(tr("chk_anti_alias"))
        self.chk_wand_anti_alias.setToolTip(tr("tip_anti_alias"))
        self.chk_wand_contiguous.setText(tr("chk_contiguous"))
        self.chk_wand_contiguous.setToolTip(tr("tip_contiguous"))
        self.lbl_wand_ai_desc.setText(tr("wand_ai_desc"))
        self._lbl_wand_hint.setText(tr("wand_hint"))
        if self.lbl_wand_ai_status.text() in [
            "IA lista para seleccionar", "AI ready to select", "ИИ готов к выделению", "AI 已就绪"
        ]:
            self.lbl_wand_ai_status.setText(tr("wand_ai_ready"))

        self.selection_box.setTitle(tr("selection_box"))
        self.set_selection_info(self._selected_px, self._total_px)
        self.btn_delete_selection.setText(tr("btn_delete_selection"))
        self.btn_delete_selection.setToolTip(tr("tip_delete_selection"))
        self.btn_inpaint_selection.setText(tr("btn_inpaint_selection"))
        self.btn_inpaint_selection.setToolTip(tr("tip_inpaint_selection"))
        self.edge_refine_box.setTitle(tr("edge_refine_box"))
        self._lbl_choke_name.setText(tr("lbl_edge_choke"))
        self._lbl_feather_name.setText(tr("lbl_edge_feather"))
        self._lbl_despill_name.setText(tr("lbl_edge_despill"))
        self.btn_apply_edge_refine.setText(tr("btn_apply_edge_refine"))
        self.btn_invert_selection.setText(tr("btn_invert_selection"))
        self.btn_invert_selection.setToolTip(tr("tip_invert_selection"))
        self.btn_select_all.setText(tr("btn_select_all"))
        self.btn_select_all.setToolTip(tr("tip_select_all"))
        self.btn_clear_selection.setText(tr("btn_clear_selection"))
        self.btn_clear_selection.setToolTip(tr("tip_clear_selection"))

    def _on_brush_tool_clicked(self, index: int):
        self.eraserModeToggled.emit(index == 1)
        self.wandModeToggled.emit(index == 2)
        self._active_brush_tool = index
        self._apply_brush_tool_visibility()

    def _apply_brush_tool_visibility(self):
        is_wand = self._active_brush_tool == 2
        self.brush_options_box.setVisible(not is_wand)
        self.wand_options_box.setVisible(is_wand)
        self.selection_box.setVisible(is_wand or self._has_selection)
        self.updateGeometry()
        self.adjustSize()
        self.geometryChanged.emit()

    def set_selection_info(self, selected_px: int, total_px: int):
        self._selected_px = selected_px
        self._total_px = total_px
        has_sel = selected_px > 0
        self._has_selection = has_sel
        if has_sel:
            pct = (selected_px / total_px * 100.0) if total_px else 0.0
            self.lbl_selection_info.setText(
                tr("selection_info", px=f"{selected_px:,}".replace(",", "."), pct=f"{pct:.1f}")
            )
        else:
            self.lbl_selection_info.setText(tr("selection_none"))
        self.btn_delete_selection.setEnabled(has_sel)
        self.btn_inpaint_selection.setEnabled(has_sel)
        self.btn_invert_selection.setEnabled(has_sel)
        self.btn_clear_selection.setEnabled(has_sel)
        self._apply_brush_tool_visibility()

    def _on_wand_type_clicked(self, is_ai: bool):
        self.wand_color_widget.setVisible(not is_ai)
        self.wand_ai_widget.setVisible(is_ai)
        self.updateGeometry()
        self.adjustSize()
        self.geometryChanged.emit()
        self.wandAIModeToggled.emit(is_ai)

    def set_wand_ai_status(self, text: str):
        self.lbl_wand_ai_status.setText(text)

    def is_wand_ai_mode(self) -> bool:
        return self.btn_wand_type_ai.isChecked()

    @property
    def btn_wand(self):
        return self.btn_tool_wand

    def is_wand_tool_active(self) -> bool:
        return self.btn_tool_wand.isChecked()

    def _on_slider_wand_tolerance_changed(self, value: int):
        self.lbl_wand_tolerance.setText(str(value))
        self.wandToleranceChanged.emit(value)

    def _on_slider_size_changed(self, value: int):
        self.lbl_brush_size.setText(f"{value} px")
        self.brushSizeChanged.emit(value)

    def _on_slider_smooth_changed(self, value: int):
        self.lbl_brush_smooth.setText(f"{value} %")
        self.brushSmoothnessChanged.emit(value)

    def _on_slider_opacity_changed(self, value: int):
        self.lbl_brush_opacity.setText(f"{value} %")
        self.brushOpacityChanged.emit(value / 100.0)

    def _on_slider_hardness_changed(self, value: int):
        self.lbl_brush_hardness.setText(f"{value} %")
        self.brushHardnessChanged.emit(value / 100.0)

    def set_brush_size(self, size: int):
        self.slider_brush_size.setValue(size)
