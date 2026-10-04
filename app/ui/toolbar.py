"""
TimoitaseBG - Panel lateral de herramientas (Sidebar) inspirado en el diseño de remove.bg y Canva.
"""
from typing import Optional, List, Tuple
from PySide6.QtCore import Qt, Signal, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QBrush, QPen
from app.i18n import tr
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QStackedWidget,
    QButtonGroup,
    QGroupBox,
    QFrame,
    QColorDialog,
    QScrollArea,
    QSizePolicy,
    QAbstractButton,
)
from app.config import (
    BG_REMOVER_MODELS,
    RESTORE_SCALES,
    DEFAULT_BRUSH_SIZE,
    MIN_BRUSH_SIZE,
    MAX_BRUSH_SIZE,
)


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
    Interruptor deslizante tipo toggle moderno (estilo iOS / macOS / Fluent).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(36, 20)
        self._thumb_pos = 1.0 if self.isChecked() else 0.0
        self._anim = QPropertyAnimation(self, b"thumb_pos", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.toggled.connect(self._on_toggled)

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
        # Interpolar suavemente el fondo:
        # 0.0 (inactivo/CPU) -> #CBD5E1 (slate suave)
        # 1.0 (activo/GPU) -> #10B981 (verde esmeralda vibrante)
        if not self.isEnabled():
            track_color = QColor("#E2E8F0")
        else:
            r = int(203 * (1 - t) + 16 * t)
            g = int(213 * (1 - t) + 185 * t)
            b = int(225 * (1 - t) + 129 * t)
            track_color = QColor(r, g, b)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(track_color))
        p.drawRoundedRect(0, 0, w, h, radius, radius)

        # Posición horizontal del círculo deslizante
        x_min = 2.0
        x_max = w - thumb_diameter - 2.0
        thumb_x = x_min + t * (x_max - x_min)
        thumb_y = 2.0

        # Sombra sutil del thumb
        p.setBrush(QBrush(QColor(0, 0, 0, 30)))
        p.drawEllipse(int(thumb_x), int(thumb_y + 1), int(thumb_diameter), int(thumb_diameter))

        # Thumb blanco
        thumb_fill = QColor("#FFFFFF") if self.isEnabled() else QColor("#94A3B8")
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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(6)

        self.lbl_text = QLabel("⚡ GPU")
        self.lbl_text.setObjectName("gpuToggleLabel")
        self.lbl_text.setCursor(Qt.CursorShape.PointingHandCursor)

        self.switch = ToggleSwitch(self)
        self.switch.toggled.connect(self._on_switch_toggled)

        layout.addWidget(self.lbl_text)
        layout.addWidget(self.switch)

        self._is_gpu = True
        self._is_blocked = False

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

    def _update_presentation(self):
        if self._is_blocked:
            self.lbl_text.setText("⚠️ CPU")
            self.lbl_text.setStyleSheet("color: #DC2626; font-weight: 700; font-size: 11px; border: none; background: transparent;")
            self.setToolTip("GPU bloqueada por fallo previo. La app funciona en CPU de forma segura.")
        elif not self.switch.isEnabled():
            self.lbl_text.setText("💻 CPU")
            self.lbl_text.setStyleSheet("color: #94A3B8; font-weight: 600; font-size: 11px; border: none; background: transparent;")
            self.setToolTip("GPU no disponible en este equipo. Modo CPU activo.")
        elif self.switch.isChecked():
            self.lbl_text.setText("⚡ GPU")
            self.lbl_text.setStyleSheet("color: #059669; font-weight: 700; font-size: 11px; border: none; background: transparent;")
            try:
                from app.config import get_gpu_name
                name = get_gpu_name() or "DirectML"
                clean = name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
                self.setToolTip(f"Aceleración GPU activa ({clean}). Clic para pasar a CPU.")
            except Exception:
                self.setToolTip("Aceleración GPU activa. Clic para pasar a CPU.")
        else:
            self.lbl_text.setText("💻 CPU")
            self.lbl_text.setStyleSheet("color: #64748B; font-weight: 600; font-size: 11px; border: none; background: transparent;")
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


class ToolSidebar(QWidget):
    """
    Panel lateral derecho con selector de herramientas estilo rail (Canva/remove.bg):
    1. ⬚ Quitar Fondo
    2. ✦ Restaurar
    3. 🖌 Pincel Mágico
    """

    # Señales para comunicar con la ventana principal
    removeBgRequested = Signal(str, bool)      # model_name, enable_clothing_protection
    restoreRequested = Signal(str, bool)       # scale_mode, use_tiling
    autoCropRequested = Signal()
    applyBgColorRequested = Signal(tuple)      # (r, g, b)
    restoreTransparencyRequested = Signal()
    brushModeToggled = Signal(bool)            # enabled
    eraserModeToggled = Signal(bool)           # is_eraser
    brushSizeChanged = Signal(int)             # size
    brushSmoothnessChanged = Signal(int)       # smoothness percentage (0-100)
    brushOpacityChanged = Signal(float)
    brushHardnessChanged = Signal(float)
    clearMaskRequested = Signal()
    inpaintRequested = Signal()
    deviceModeChanged = Signal(bool)           # is_gpu
    batchRequested = Signal(str, str, str, bool) # input_dir, output_dir, model_id, clothing_prot

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(380)
        self._selected_is_none: bool = False
        self._selected_color_rgb: Tuple[int, int, int] = (255, 255, 255)
        self.has_transparency: bool = False
        self.can_restore_transparency: bool = False
        self._shortcuts_expanded: bool = False
        self._setup_ui()
        self._init_accel_state()

    def _setup_ui(self):
        # Layout principal de ToolSidebar: sin spacing extra, margins 0
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------------------------------------------------
        # [BLOQUE FIJO SUPERIOR] QWidget "sidebar_header"
        # ---------------------------------------------------------
        self.sidebar_header = QWidget()
        self.sidebar_header.setObjectName("sidebar_header")
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

        title_label = QLabel(tr("sidebar_title"))
        title_label.setObjectName("sidebarHeader")
        subtitle_label = QLabel(tr("sidebar_subtitle"))
        subtitle_label.setObjectName("sidebarSubheader")
        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        header_top_row.addLayout(text_layout)
        header_top_row.addStretch()

        # Toggle interactivo único GPU / CPU (moderno con animación fluida)
        self.gpu_toggle = GpuToggle(self)
        self.gpu_toggle.toggled.connect(lambda is_gpu: self.deviceModeChanged.emit(is_gpu))
        header_top_row.addWidget(self.gpu_toggle, alignment=Qt.AlignmentFlag.AlignVCenter)

        header_layout.addLayout(header_top_row)
        main_layout.addWidget(self.sidebar_header)

        # ---------------------------------------------------------
        # [BLOQUE FIJO] QWidget "tool_tabs_row"
        # ---------------------------------------------------------
        self.tool_tabs_row = QWidget()
        self.tool_tabs_row.setObjectName("tool_tabs_row")
        self.tool_tabs_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tabs_layout = QHBoxLayout(self.tool_tabs_row)
        tabs_layout.setContentsMargins(16, 2, 16, 8)
        tabs_layout.setSpacing(8)

        self.rail_group = QButtonGroup(self)
        self.rail_group.setExclusive(True)

        self.btn_rail_bg = QPushButton(tr("sidebar_tab_bg"))
        self.btn_rail_bg.setObjectName("toolRailButton")
        self.btn_rail_bg.setCheckable(True)
        self.btn_rail_bg.setChecked(True)
        self.btn_rail_bg.setFixedHeight(58)
        self.rail_group.addButton(self.btn_rail_bg, 0)
        tabs_layout.addWidget(self.btn_rail_bg)

        self.btn_rail_restore = QPushButton(tr("sidebar_tab_restore"))
        self.btn_rail_restore.setObjectName("toolRailButton")
        self.btn_rail_restore.setCheckable(True)
        self.btn_rail_restore.setFixedHeight(58)
        self.rail_group.addButton(self.btn_rail_restore, 1)
        tabs_layout.addWidget(self.btn_rail_restore)

        self.btn_rail_brush = QPushButton(tr("sidebar_tab_brush"))
        self.btn_rail_brush.setObjectName("toolRailButton")
        self.btn_rail_brush.setCheckable(True)
        self.btn_rail_brush.setFixedHeight(58)
        self.rail_group.addButton(self.btn_rail_brush, 2)
        tabs_layout.addWidget(self.btn_rail_brush)

        self.btn_rail_batch = QPushButton(tr("sidebar_tab_batch"))
        self.btn_rail_batch.setObjectName("toolRailButton")
        self.btn_rail_batch.setCheckable(True)
        self.btn_rail_batch.setFixedHeight(58)
        self.rail_group.addButton(self.btn_rail_batch, 3)
        tabs_layout.addWidget(self.btn_rail_batch)

        main_layout.addWidget(self.tool_tabs_row)

        # ---------------------------------------------------------
        # [ÁREA SCROLLEABLE - toma todo el espacio vertical restante]
        # ---------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("toolScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Widget interno con ancho máximo explícito y margen derecho de 20px reservado para la scrollbar
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scroll_content")
        self.scroll_content.setMaximumWidth(380)
        scroll_layout = QVBoxLayout(self.scroll_content)
        scroll_layout.setContentsMargins(16, 16, 20, 16)
        scroll_layout.setSpacing(12)

        # QStackedWidget dinámico con las secciones de la pestaña activa
        self.stack = AutoResizingStackedWidget()
        self.stack.setObjectName("toolsStack")

        # 1. Panel Quitar Fondo
        self.panel_bg = QWidget()
        self._setup_bg_panel()
        self.stack.addWidget(self.panel_bg)

        # 2. Panel Restaurar
        self.panel_restore = QWidget()
        self._setup_restore_panel()
        self.stack.addWidget(self.panel_restore)

        # 3. Panel Pincel Mágico
        self.panel_brush = QWidget()
        self._setup_brush_panel()
        self.stack.addWidget(self.panel_brush)

        # 4. Panel Lotes (Batch)
        self.panel_batch = QWidget()
        self._setup_batch_panel()
        self.stack.addWidget(self.panel_batch)

        scroll_layout.addWidget(self.stack)
        self.scroll_area.setWidget(self.scroll_content)

        main_layout.addWidget(self.scroll_area, stretch=1)

        # Conectar cambio de herramienta en el rail
        self.rail_group.idClicked.connect(self._on_rail_clicked)

        # ---------------------------------------------------------
        # [BLOQUE FIJO INFERIOR] QWidget "quick_shortcuts_box"
        # ---------------------------------------------------------
        self.quick_shortcuts_box = QWidget()
        self.quick_shortcuts_box.setObjectName("quick_shortcuts_box")
        self.quick_shortcuts_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        shortcuts_box_layout = QVBoxLayout(self.quick_shortcuts_box)
        shortcuts_box_layout.setContentsMargins(16, 4, 16, 12)
        shortcuts_box_layout.setSpacing(0)

        info_frame = QFrame()
        info_frame.setObjectName("infoCard")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 6, 10, 6)
        info_layout.setSpacing(4)

        # Header clickeable para colapsar/expandir atajos
        self.btn_toggle_shortcuts = QPushButton(tr("sidebar_shortcuts_col"))
        self.btn_toggle_shortcuts.setObjectName("infoCardToggle")
        self.btn_toggle_shortcuts.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_shortcuts.clicked.connect(self._toggle_shortcuts)
        info_layout.addWidget(self.btn_toggle_shortcuts)

        # Contenedor colapsable de atajos
        self.shortcuts_content = QWidget()
        self.shortcuts_content.setObjectName("shortcutsContent")
        content_layout = QVBoxLayout(self.shortcuts_content)
        content_layout.setContentsMargins(0, 4, 0, 2)
        content_layout.setSpacing(4)

        shortcuts = [
            (tr("shortcut_zoom"), tr("shortcut_zoom_desc")),
            (tr("shortcut_pan"), tr("shortcut_pan_desc")),
            (tr("shortcut_undo"), tr("shortcut_undo_desc")),
        ]
        for key, desc in shortcuts:
            row = QHBoxLayout()
            k_lbl = QLabel(key)
            k_lbl.setObjectName("shortcutBadge")
            d_lbl = QLabel(desc)
            d_lbl.setObjectName("shortcutDesc")
            row.addWidget(k_lbl)
            row.addStretch()
            row.addWidget(d_lbl)
            content_layout.addLayout(row)

        info_layout.addWidget(self.shortcuts_content)
        shortcuts_box_layout.addWidget(info_frame)
        main_layout.addWidget(self.quick_shortcuts_box, stretch=0)

        # Aplicar visibilidad inicial (colapsado)
        self._update_shortcuts_ui()

    def _create_section_header(self, title: str, tooltip: str) -> QHBoxLayout:
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
        return header

    def _toggle_shortcuts(self):
        self._shortcuts_expanded = not self._shortcuts_expanded
        self._update_shortcuts_ui()

    def _update_shortcuts_ui(self):
        if self._shortcuts_expanded:
            self.btn_toggle_shortcuts.setText(tr("sidebar_shortcuts_exp"))
            self.shortcuts_content.show()
        else:
            self.btn_toggle_shortcuts.setText(tr("sidebar_shortcuts_col"))
            self.shortcuts_content.hide()

    @property
    def btn_accel_gpu(self):
        """Adaptador de compatibilidad retroactiva."""
        return _GpuToggleAdapter(self.gpu_toggle, is_gpu_target=True)

    @property
    def btn_accel_cpu(self):
        """Adaptador de compatibilidad retroactiva."""
        return _GpuToggleAdapter(self.gpu_toggle, is_gpu_target=False)

    def _init_accel_state(self):
        """Inicializa el estado visual del toggle de aceleración."""
        try:
            from app.core.gpu_manager import get_gpu_manager
            gpu_mgr = get_gpu_manager()
            self.set_device_mode(
                is_gpu=gpu_mgr.is_gpu_enabled(),
                is_blocked=gpu_mgr.has_orphaned_worker(),
            )
        except Exception:
            pass

    def _on_accel_clicked(self, is_gpu: bool):
        self.deviceModeChanged.emit(is_gpu)

    def set_device_mode(self, is_gpu: bool, is_blocked: bool = False):
        """Sincroniza el estado visual del toggle GPU / CPU."""
        if hasattr(self, "gpu_toggle"):
            self.gpu_toggle.set_state(is_gpu, is_blocked)

    # -------------------------------------------------------------
    # 1. Panel Quitar Fondo
    # -------------------------------------------------------------
    def _setup_bg_panel(self):
        layout = QVBoxLayout(self.panel_bg)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_model = self._create_section_header(
            tr("header_model"),
            tr("model_desc")
        )
        layout.addLayout(header_model)

        self.combo_bg_model = QComboBox()
        self.combo_bg_model.setFixedHeight(38)
        for model_id, model_label in BG_REMOVER_MODELS:
            self.combo_bg_model.addItem(tr(f"model_{model_id}"), model_id)
        self.combo_bg_model.setCurrentIndex(0)
        layout.addWidget(self.combo_bg_model)

        self.chk_clothing_protection = QCheckBox(tr("chk_clothing"))
        self.chk_clothing_protection.setChecked(True)
        self.chk_clothing_protection.setToolTip(tr("chk_clothing_tooltip"))
        layout.addWidget(self.chk_clothing_protection)

        self.btn_remove_bg = QPushButton(tr("btn_remove_now"))
        self.btn_remove_bg.setObjectName("primaryActionButton")
        self.btn_remove_bg.setFixedHeight(40)
        self.btn_remove_bg.clicked.connect(self._on_remove_bg_clicked)
        layout.addWidget(self.btn_remove_bg)

        # ---------------------------------------------------------
        # Sección: Recorte Automático al Contenido (Auto-crop)
        # ---------------------------------------------------------
        sep1 = QFrame()
        sep1.setObjectName("panelSeparator")
        sep1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep1)

        lbl_crop = QLabel(tr("header_crop"))
        lbl_crop.setObjectName("fieldLabel")
        layout.addWidget(lbl_crop)

        self.btn_autocrop = QPushButton(tr("btn_autocrop_now"))
        self.btn_autocrop.setObjectName("secondaryActionButton")
        self.btn_autocrop.setFixedHeight(36)
        self.btn_autocrop.setToolTip(tr("btn_autocrop_tooltip"))
        self.btn_autocrop.setEnabled(False)
        self.btn_autocrop.clicked.connect(self.autoCropRequested.emit)
        layout.addWidget(self.btn_autocrop)

        # ---------------------------------------------------------
        # Sección: Reemplazar Fondo por Color Sólido
        # ---------------------------------------------------------
        sep2 = QFrame()
        sep2.setObjectName("panelSeparator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        lbl_bg_section = QLabel(tr("header_bg"))
        lbl_bg_section.setObjectName("fieldLabel")
        layout.addWidget(lbl_bg_section)

        # Fila de muestras de color (Swatches)
        swatches_row = QHBoxLayout()
        swatches_row.setSpacing(6)

        self.swatch_group = QButtonGroup(self)
        self.swatch_group.setExclusive(True)

        # 1. Swatch Ninguno / Transparente
        self.btn_swatch_none = QPushButton("∅")
        self.btn_swatch_none.setObjectName("swatchNoneButton")
        self.btn_swatch_none.setCheckable(True)
        self.btn_swatch_none.setFixedSize(36, 32)
        self.btn_swatch_none.setToolTip(tr("swatch_transparent"))
        self.btn_swatch_none.setEnabled(False)
        self.swatch_group.addButton(self.btn_swatch_none)
        swatches_row.addWidget(self.btn_swatch_none)

        # 2. Swatch Blanco
        self.btn_swatch_white = QPushButton()
        self.btn_swatch_white.setObjectName("swatchWhiteButton")
        self.btn_swatch_white.setCheckable(True)
        self.btn_swatch_white.setChecked(True)
        self.btn_swatch_white.setFixedSize(36, 32)
        self.btn_swatch_white.setToolTip(tr("swatch_white"))
        self.btn_swatch_white.setEnabled(False)
        self.swatch_group.addButton(self.btn_swatch_white)
        swatches_row.addWidget(self.btn_swatch_white)

        # 3. Swatch Negro
        self.btn_swatch_black = QPushButton()
        self.btn_swatch_black.setObjectName("swatchBlackButton")
        self.btn_swatch_black.setCheckable(True)
        self.btn_swatch_black.setFixedSize(36, 32)
        self.btn_swatch_black.setToolTip(tr("swatch_black"))
        self.btn_swatch_black.setEnabled(False)
        self.swatch_group.addButton(self.btn_swatch_black)
        swatches_row.addWidget(self.btn_swatch_black)

        # 4. Swatch Gris Suave
        self.btn_swatch_gray = QPushButton()
        self.btn_swatch_gray.setObjectName("swatchGrayButton")
        self.btn_swatch_gray.setCheckable(True)
        self.btn_swatch_gray.setFixedSize(36, 32)
        self.btn_swatch_gray.setToolTip(tr("swatch_gray"))
        self.btn_swatch_gray.setEnabled(False)
        self.swatch_group.addButton(self.btn_swatch_gray)
        swatches_row.addWidget(self.btn_swatch_gray)

        # 5. Swatch Personalizado
        self.btn_swatch_custom = QPushButton("🎨")
        self.btn_swatch_custom.setObjectName("swatchCustomButton")
        self.btn_swatch_custom.setCheckable(True)
        self.btn_swatch_custom.setFixedSize(36, 32)
        self.btn_swatch_custom.setToolTip(tr("swatch_custom"))
        self.btn_swatch_custom.setEnabled(False)
        self.swatch_group.addButton(self.btn_swatch_custom)
        swatches_row.addWidget(self.btn_swatch_custom)

        swatches_row.addStretch()
        layout.addLayout(swatches_row)

        # Fila de preview del color actual
        color_action_row = QHBoxLayout()
        color_action_row.setSpacing(8)

        self.color_preview_box = QFrame()
        self.color_preview_box.setObjectName("swatchPreview")
        self.color_preview_box.setFixedSize(22, 22)
        color_action_row.addWidget(self.color_preview_box)

        self.lbl_color_hex = QLabel(tr("color_white"))
        self.lbl_color_hex.setObjectName("colorHexLabel")
        color_action_row.addWidget(self.lbl_color_hex)

        color_action_row.addStretch()
        layout.addLayout(color_action_row)

        # Conectar eventos de swatches
        self.btn_swatch_none.clicked.connect(self._on_swatch_none_clicked)
        self.btn_swatch_white.clicked.connect(lambda: self._on_color_swatch_clicked((255, 255, 255), tr("color_white")))
        self.btn_swatch_black.clicked.connect(lambda: self._on_color_swatch_clicked((0, 0, 0), tr("color_black")))
        self.btn_swatch_gray.clicked.connect(lambda: self._on_color_swatch_clicked((226, 232, 240), tr("color_gray")))
        self.btn_swatch_custom.clicked.connect(self._on_swatch_custom_clicked)

        # Inicializar estado visual del color
        self._update_color_preview((255, 255, 255), tr("color_white"))

        layout.addStretch()

    # -------------------------------------------------------------
    # 2. Panel Restaurar
    # -------------------------------------------------------------
    def _setup_restore_panel(self):
        layout = QVBoxLayout(self.panel_restore)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_scale = self._create_section_header(
            tr("header_scale"),
            tr("scale_desc")
        )
        layout.addLayout(header_scale)

        self.combo_restore_scale = QComboBox()
        self.combo_restore_scale.setFixedHeight(38)
        for scale_id, scale_label in RESTORE_SCALES:
            self.combo_restore_scale.addItem(tr(f"scale_{scale_id}"), scale_id)
        self.combo_restore_scale.setCurrentIndex(0)
        layout.addWidget(self.combo_restore_scale)

        self.chk_tiling = QCheckBox(tr("chk_tiling"))
        self.chk_tiling.setChecked(True)
        self.chk_tiling.setToolTip(tr("chk_tiling_tooltip"))
        layout.addWidget(self.chk_tiling)

        self.btn_restore = QPushButton(tr("btn_scale_now"))
        self.btn_restore.setObjectName("primaryActionButton")
        self.btn_restore.setFixedHeight(40)
        self.btn_restore.clicked.connect(self._on_restore_clicked)
        layout.addWidget(self.btn_restore)

        layout.addStretch()

    # -------------------------------------------------------------
    # 3. Panel Pincel Mágico
    # -------------------------------------------------------------
    def _setup_brush_panel(self):
        layout = QVBoxLayout(self.panel_brush)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_brush = self._create_section_header(
            tr("header_brush"),
            tr("brush_desc")
        )
        layout.addLayout(header_brush)

        # Botón de activación del modo pincel
        self.btn_toggle_brush = QPushButton(tr("btn_brush_mode"))
        self.btn_toggle_brush.setObjectName("brushToggleButton")
        self.btn_toggle_brush.setCheckable(True)
        self.btn_toggle_brush.setChecked(False)
        self.btn_toggle_brush.setFixedHeight(38)
        self.btn_toggle_brush.clicked.connect(self._on_brush_toggled)
        layout.addWidget(self.btn_toggle_brush)

        # Sub-herramienta: Pintar o Borrar (Goma)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        
        self.brush_mode_group = QButtonGroup(self)
        self.brush_mode_group.setExclusive(True)
        
        self.btn_tool_paint = QPushButton(tr("btn_paint"))
        self.btn_tool_paint.setObjectName("brushToolModeButton")
        self.btn_tool_paint.setCheckable(True)
        self.btn_tool_paint.setChecked(True)
        self.btn_tool_paint.setFixedHeight(32)
        
        self.btn_tool_erase = QPushButton(tr("btn_erase"))
        self.btn_tool_erase.setObjectName("brushToolModeButton")
        self.btn_tool_erase.setCheckable(True)
        self.btn_tool_erase.setFixedHeight(32)
        
        self.brush_mode_group.addButton(self.btn_tool_paint, 0)
        self.brush_mode_group.addButton(self.btn_tool_erase, 1)
        
        self.brush_mode_group.idClicked.connect(self._on_brush_tool_clicked)
        
        mode_row.addWidget(self.btn_tool_paint)
        mode_row.addWidget(self.btn_tool_erase)
        layout.addLayout(mode_row)

        # Control de tamaño del pincel
        brush_ctrl_box = QGroupBox(tr("brush_settings"))
        b_layout = QVBoxLayout(brush_ctrl_box)
        b_layout.setContentsMargins(12, 14, 12, 10)
        b_layout.setSpacing(10)

        # Fila de tamaño
        slider_row = QHBoxLayout()
        size_lbl = QLabel(tr("lbl_size_thick"))
        size_lbl.setFixedWidth(50)
        self.slider_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.slider_brush_size.setValue(DEFAULT_BRUSH_SIZE)
        self.slider_brush_size.valueChanged.connect(self._on_slider_size_changed)

        self.lbl_brush_size = QLabel(f"{DEFAULT_BRUSH_SIZE} px")
        self.lbl_brush_size.setObjectName("brushSizeLabel")
        self.lbl_brush_size.setFixedWidth(42)
        self.lbl_brush_size.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        slider_row.addWidget(size_lbl)
        slider_row.addWidget(self.slider_brush_size)
        slider_row.addWidget(self.lbl_brush_size)
        b_layout.addLayout(slider_row)

        # Fila de suavizado (Smoothness)
        smooth_row = QHBoxLayout()
        smooth_lbl = QLabel(tr("lbl_smoothness"))
        smooth_lbl.setFixedWidth(50)
        self.slider_brush_smooth = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_smooth.setRange(0, 100)
        self.slider_brush_smooth.setValue(0)
        self.slider_brush_smooth.valueChanged.connect(self._on_slider_smooth_changed)
        
        self.lbl_brush_smooth = QLabel("0 %")
        self.lbl_brush_smooth.setObjectName("brushSizeLabel")
        self.lbl_brush_smooth.setFixedWidth(42)
        self.lbl_brush_smooth.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        smooth_row.addWidget(smooth_lbl)
        smooth_row.addWidget(self.slider_brush_smooth)
        smooth_row.addWidget(self.lbl_brush_smooth)
        b_layout.addLayout(smooth_row)
        
        # Opacidad
        op_row = QHBoxLayout()
        lbl_op = QLabel(tr("lbl_opacity"))
        lbl_op.setFixedWidth(60)
        op_row.addWidget(lbl_op)
        self.slider_brush_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_opacity.setRange(1, 100)
        self.slider_brush_opacity.setValue(100)
        self.slider_brush_opacity.valueChanged.connect(self._on_slider_opacity_changed)
        self.lbl_brush_opacity = QLabel("100 %")
        self.lbl_brush_opacity.setFixedWidth(40)
        self.lbl_brush_opacity.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        op_row.addWidget(self.slider_brush_opacity)
        op_row.addWidget(self.lbl_brush_opacity)
        b_layout.addLayout(op_row)
        
        # Dureza
        hd_row = QHBoxLayout()
        lbl_hd = QLabel(tr("lbl_hardness"))
        lbl_hd.setFixedWidth(60)
        hd_row.addWidget(lbl_hd)
        self.slider_brush_hardness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_hardness.setRange(0, 100)
        self.slider_brush_hardness.setValue(100)
        self.slider_brush_hardness.valueChanged.connect(self._on_slider_hardness_changed)
        self.lbl_brush_hardness = QLabel("100 %")
        self.lbl_brush_hardness.setFixedWidth(40)
        self.lbl_brush_hardness.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hd_row.addWidget(self.slider_brush_hardness)
        hd_row.addWidget(self.lbl_brush_hardness)
        b_layout.addLayout(hd_row)

        # Botones rápidos de tamaño
        presets_row = QHBoxLayout()
        presets_row.setSpacing(6)
        for sz in [2, 10, 25, 50, 100]:
            btn = QPushButton(f"{sz}px")
            btn.setObjectName("presetButton")
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda _, s=sz: self.set_brush_size(s))
            presets_row.addWidget(btn)
        b_layout.addLayout(presets_row)
        layout.addWidget(brush_ctrl_box)


        layout.addStretch()

    # -------------------------------------------------------------
    # 4. Panel Lotes (Batch)
    # -------------------------------------------------------------
    def _setup_batch_panel(self):
        layout = QVBoxLayout(self.panel_batch)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_batch = self._create_section_header(
            tr("header_batch"),
            tr("batch_desc")
        )
        layout.addLayout(header_batch)

        # Seleccionar carpeta de entrada
        self.btn_batch_input = QPushButton(tr("btn_batch_input"))
        self.btn_batch_input.setFixedHeight(36)
        self.btn_batch_input.clicked.connect(self._on_batch_input_clicked)
        layout.addWidget(self.btn_batch_input)
        
        self.lbl_batch_input = QLabel(tr("batch_no_folder"))
        self.lbl_batch_input.setObjectName("shortcutDesc")
        self.lbl_batch_input.setWordWrap(True)
        layout.addWidget(self.lbl_batch_input)

        # Seleccionar carpeta de salida
        self.btn_batch_output = QPushButton(tr("btn_batch_output"))
        self.btn_batch_output.setFixedHeight(36)
        self.btn_batch_output.clicked.connect(self._on_batch_output_clicked)
        layout.addWidget(self.btn_batch_output)

        self.lbl_batch_output = QLabel(tr("batch_no_folder"))
        self.lbl_batch_output.setObjectName("shortcutDesc")
        self.lbl_batch_output.setWordWrap(True)
        layout.addWidget(self.lbl_batch_output)

        sep1 = QFrame()
        sep1.setObjectName("panelSeparator")
        sep1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep1)

        # Opciones de modelo
        lbl_model = QLabel(tr("lbl_batch_model"))
        lbl_model.setObjectName("fieldLabel")
        layout.addWidget(lbl_model)

        self.combo_batch_model = QComboBox()
        self.combo_batch_model.setFixedHeight(38)
        for model_id, model_label in BG_REMOVER_MODELS:
            self.combo_batch_model.addItem(tr(f"model_{model_id}"), model_id)
        layout.addWidget(self.combo_batch_model)

        self.chk_batch_clothing = QCheckBox(tr("chk_batch_clothing"))
        self.chk_batch_clothing.setChecked(True)
        layout.addWidget(self.chk_batch_clothing)

        # Botón Iniciar
        self.btn_run_batch = QPushButton(tr("btn_run_batch"))
        self.btn_run_batch.setObjectName("primaryActionButton")
        self.btn_run_batch.setFixedHeight(40)
        self.btn_run_batch.setEnabled(False)
        self.btn_run_batch.clicked.connect(self._on_run_batch_clicked)
        layout.addWidget(self.btn_run_batch)

        layout.addStretch()

        self._batch_input_dir = ""
        self._batch_output_dir = ""

    # -------------------------------------------------------------
    # Handlers Internos
    # -------------------------------------------------------------
    def _on_rail_clicked(self, index: int):
        self.stack.setCurrentIndex(index)
        self.scroll_area.verticalScrollBar().setValue(0)
        is_brush = (index == 2)
        self.btn_toggle_brush.setChecked(is_brush)
        self.brushModeToggled.emit(is_brush)

    def _on_brush_toggled(self, checked: bool):
        if checked:
            self.btn_rail_brush.setChecked(True)
            self.stack.setCurrentIndex(2)
            self.scroll_area.verticalScrollBar().setValue(0)
        self.brushModeToggled.emit(checked)

    def _on_brush_tool_clicked(self, index: int):
        self.eraserModeToggled.emit(index == 1)

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

    def is_clothing_protection_enabled(self) -> bool:
        """Indica si el usuario tiene activa la protección de prendas y ropa blanca."""
        return self.chk_clothing_protection.isChecked() if hasattr(self, "chk_clothing_protection") else True

    def _on_remove_bg_clicked(self):
        model_id = self.combo_bg_model.currentData()
        enable_protection = self.is_clothing_protection_enabled()
        self.removeBgRequested.emit(model_id, enable_protection)

    def _on_restore_clicked(self):
        scale_mode = self.combo_restore_scale.currentData()
        use_tiling = self.chk_tiling.isChecked()
        self.restoreRequested.emit(scale_mode, use_tiling)

    def _on_batch_input_clicked(self):
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes de origen")
        if dir_path:
            self._batch_input_dir = dir_path
            self.lbl_batch_input.setText(tr("batch_input_path", path=dir_path))
            self._update_batch_btn_state()

    def _on_batch_output_clicked(self):
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de destino")
        if dir_path:
            self._batch_output_dir = dir_path
            self.lbl_batch_output.setText(tr("batch_output_path", path=dir_path))
            self._update_batch_btn_state()

    def _update_batch_btn_state(self):
        self.btn_run_batch.setEnabled(bool(self._batch_input_dir) and bool(self._batch_output_dir))

    def _on_run_batch_clicked(self):
        model_id = self.combo_batch_model.currentData()
        enable_protection = self.chk_batch_clothing.isChecked()
        self.batchRequested.emit(self._batch_input_dir, self._batch_output_dir, model_id, enable_protection)

    def set_has_mask(self, has_mask: bool):
        """Actualiza la disponibilidad de los botones de borrado según si hay máscara."""
        pass

    def _on_swatch_none_clicked(self):
        self._selected_is_none = True
        self.color_preview_box.setStyleSheet("background-color: transparent; border: 1.5px solid #CBD5E1; border-radius: 4px;")
        self.lbl_color_hex.setText("Transparente")
        self.restoreTransparencyRequested.emit()

    def _on_color_swatch_clicked(self, color_rgb: Tuple[int, int, int], label: str):
        self._selected_is_none = False
        self._selected_color_rgb = color_rgb
        self._update_color_preview(color_rgb, label)
        self.applyBgColorRequested.emit(color_rgb)

    def _on_swatch_custom_clicked(self):
        initial = QColor(*self._selected_color_rgb)
        color = QColorDialog.getColor(initial, self, "Seleccionar Color de Fondo", options=QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            hex_str = f"#{color.red():02X}{color.green():02X}{color.blue():02X}"
            self._selected_is_none = False
            self._selected_color_rgb = rgb
            self.btn_swatch_custom.setChecked(True)
            self._update_color_preview(rgb, f"Personalizado ({hex_str})")
            self.applyBgColorRequested.emit(rgb)
        else:
            if self._selected_is_none:
                self.btn_swatch_none.setChecked(True)
            elif self._selected_color_rgb == (255, 255, 255):
                self.btn_swatch_white.setChecked(True)
            elif self._selected_color_rgb == (0, 0, 0):
                self.btn_swatch_black.setChecked(True)
            elif self._selected_color_rgb == (226, 232, 240):
                self.btn_swatch_gray.setChecked(True)

    def _update_color_preview(self, color_rgb: Tuple[int, int, int], label: str):
        if self._selected_is_none:
            self.color_preview_box.setStyleSheet(
                "background-image: url(:/patterns/checker.png); border: 1px solid #D6D3D1; border-radius: 4px;"
            )
        else:
            r, g, b = color_rgb
            self.color_preview_box.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #D6D3D1; border-radius: 4px;"
            )
        self.lbl_color_hex.setText(label)

    def set_transparency_available(self, has_transparency: bool, can_restore: bool):
        """
        Actualiza los controles de recorte y fondo según la disponibilidad de transparencia
        en la imagen actual o en el historial.
        """
        self.has_transparency = has_transparency
        self.can_restore_transparency = can_restore

        # 1. Recorte al contenido
        self.btn_autocrop.setEnabled(has_transparency)
        if has_transparency:
            self.btn_autocrop.setToolTip("Elimina los márgenes transparentes sobrantes con 10px de respiro.")
        else:
            self.btn_autocrop.setToolTip(tr("btn_autocrop_tooltip"))

        # 2. Selector de fondo
        self.btn_swatch_none.setEnabled(has_transparency)
        self.btn_swatch_white.setEnabled(has_transparency)
        self.btn_swatch_black.setEnabled(has_transparency)
        self.btn_swatch_gray.setEnabled(has_transparency)
        self.btn_swatch_custom.setEnabled(has_transparency)

        if not has_transparency:
            self.btn_swatch_none.setChecked(True)
            self._selected_is_none = True

    def set_busy(self, busy: bool):
        """Deshabilita los botones de acción durante el procesamiento."""
        self.btn_remove_bg.setEnabled(not busy)
        self.btn_restore.setEnabled(not busy)
        if busy:
            self.btn_autocrop.setEnabled(False)
        else:
            self.set_transparency_available(self.has_transparency, self.can_restore_transparency)
