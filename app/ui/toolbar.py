"""
TimoitaseBG - Panel lateral de herramientas (Sidebar) inspirado en el diseño de remove.bg y Canva.
"""
from typing import Optional, List, Tuple
from PySide6.QtCore import Qt, Signal, Property, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QColor, QPainter, QBrush, QPen
from app.i18n import tr
from app.ui.icons import get_icon, get_pixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
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

        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(14, 14)
        self.lbl_icon.setStyleSheet("background: transparent; border: none;")

        self.lbl_text = QLabel("GPU")
        self.lbl_text.setObjectName("gpuToggleLabel")
        self.lbl_text.setCursor(Qt.CursorShape.PointingHandCursor)

        self.switch = ToggleSwitch(self)
        self.switch.toggled.connect(self._on_switch_toggled)

        layout.addWidget(self.lbl_icon)
        layout.addWidget(self.lbl_text)
        layout.addWidget(self.switch)

        self._is_gpu = True
        self._is_blocked = False
        self._update_presentation()

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
            self.lbl_icon.setPixmap(get_pixmap("alert-triangle", size=14, color="#DC2626"))
            self.lbl_text.setText("CPU")
            self.lbl_text.setStyleSheet("color: #DC2626; font-weight: 700; font-size: 11px; border: none; background: transparent;")
            self.setToolTip("GPU bloqueada por fallo previo. La app funciona en CPU de forma segura.")
        elif not self.switch.isEnabled():
            self.lbl_icon.setPixmap(get_pixmap("cpu", size=14, color="#94A3B8"))
            self.lbl_text.setText("CPU")
            self.lbl_text.setStyleSheet("color: #94A3B8; font-weight: 600; font-size: 11px; border: none; background: transparent;")
            self.setToolTip("GPU no disponible en este equipo. Modo CPU activo.")
        elif self.switch.isChecked():
            self.lbl_icon.setPixmap(get_pixmap("zap", size=14, color="#059669"))
            self.lbl_text.setText("GPU")
            self.lbl_text.setStyleSheet("color: #059669; font-weight: 700; font-size: 11px; border: none; background: transparent;")
            try:
                from app.config import get_gpu_name
                name = get_gpu_name() or "DirectML"
                clean = name.replace("NVIDIA ", "").replace(" Laptop GPU", "")
                self.setToolTip(f"Aceleración GPU activa ({clean}). Clic para pasar a CPU.")
            except Exception:
                self.setToolTip("Aceleración GPU activa. Clic para pasar a CPU.")
        else:
            self.lbl_icon.setPixmap(get_pixmap("cpu", size=14, color="#64748B"))
            self.lbl_text.setText("CPU")
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
    1. Quitar Fondo
    2. Restaurar
    3. Pincel Mágico
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
    wandModeToggled = Signal(bool)             # is_wand
    wandAIModeToggled = Signal(bool)           # is_ai_wand (MobileSAM)
    wandToleranceChanged = Signal(int)         # tolerance (0-255, como Photoshop)
    wandSelectionModeChanged = Signal(str)     # new | add | subtract | intersect
    wandAntiAliasChanged = Signal(bool)
    wandContiguousChanged = Signal(bool)
    deleteSelectionRequested = Signal()
    invertSelectionRequested = Signal()
    selectAllRequested = Signal()
    clearSelectionRequested = Signal()
    clearMaskRequested = Signal()
    inpaintRequested = Signal()
    deviceModeChanged = Signal(bool)           # is_gpu
    batchRequested = Signal(str, str, str, bool) # input_dir, output_dir, model_id, clothing_prot
    detectSubjectsRequested = Signal()
    isolateSubjectRequested = Signal(object)   # DetectedSubject
    selectSubjectRequested = Signal(object)    # DetectedSubject

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(380)
        self._selected_is_none: bool = False
        self._selected_color_rgb: Tuple[int, int, int] = (255, 255, 255)
        self.has_transparency: bool = False
        self.can_restore_transparency: bool = False
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

        title_label = QLabel(tr("sidebar_title"))
        title_label.setObjectName("sidebarHeader")
        title_label.setStyleSheet("background: transparent;")
        subtitle_label = QLabel(tr("sidebar_subtitle"))
        subtitle_label.setObjectName("sidebarSubheader")
        subtitle_label.setStyleSheet("background: transparent;")
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
        self.tool_tabs_row.setStyleSheet("background: transparent;")
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

        self.update_theme_icons(is_dark=True)

    def update_theme_icons(self, is_dark: bool = True):
        """Aplica y refresca iconos vectoriales Open Source adaptados al tema."""
        fg_color = "#EAEAEA" if is_dark else "#292524"
        accent_color = "#FF4F79"
        sub_color = "#999999" if is_dark else "#78716C"
        dis_color = "#666666" if is_dark else "#A8A29E"

        # Pestañas del Rail
        if hasattr(self, "btn_rail_bg"):
            self.btn_rail_bg.setIcon(get_icon("sparkles", color=sub_color, active_color=accent_color, size=18))
            self.btn_rail_bg.setIconSize(QSize(18, 18))
        if hasattr(self, "btn_rail_restore"):
            self.btn_rail_restore.setIcon(get_icon("maximize-2", color=sub_color, active_color=accent_color, size=18))
            self.btn_rail_restore.setIconSize(QSize(18, 18))
        if hasattr(self, "btn_rail_brush"):
            self.btn_rail_brush.setIcon(get_icon("brush", color=sub_color, active_color=accent_color, size=18))
            self.btn_rail_brush.setIconSize(QSize(18, 18))
        if hasattr(self, "btn_rail_batch"):
            self.btn_rail_batch.setIcon(get_icon("folder", color=sub_color, active_color=accent_color, size=18))
            self.btn_rail_batch.setIconSize(QSize(18, 18))

        # Panel Quitar Fondo
        if hasattr(self, "btn_remove_bg"):
            self.btn_remove_bg.setIcon(get_icon("sparkles", color="#FFFFFF", size=16))
            self.btn_remove_bg.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_autocrop"):
            self.btn_autocrop.setIcon(get_icon("crop", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
            self.btn_autocrop.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_swatch_custom"):
            self.btn_swatch_custom.setIcon(get_icon("palette", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=18))
            self.btn_swatch_custom.setIconSize(QSize(18, 18))

        # Panel Restaurar
        if hasattr(self, "btn_restore"):
            self.btn_restore.setIcon(get_icon("maximize-2", color="#FFFFFF", size=16))
            self.btn_restore.setIconSize(QSize(16, 16))

        # Subherramientas de pincel
        if hasattr(self, "btn_tool_paint"):
            self.btn_tool_paint.setIcon(get_icon("brush", color=fg_color, active_color=accent_color, size=16))
            self.btn_tool_paint.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_tool_erase"):
            self.btn_tool_erase.setIcon(get_icon("eraser", color=fg_color, active_color=accent_color, size=16))
            self.btn_tool_erase.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_tool_wand"):
            self.btn_tool_wand.setIcon(get_icon("wand", color=fg_color, active_color=accent_color, size=16))
            self.btn_tool_wand.setIconSize(QSize(16, 16))

        # Modos de selección de varita
        if hasattr(self, "wand_mode_buttons"):
            for btn, icon_name in self.wand_mode_buttons:
                btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
                btn.setIconSize(QSize(16, 16))

        # Acciones de selección
        if hasattr(self, "btn_delete_selection"):
            self.btn_delete_selection.setIcon(get_icon("trash", color="#FFFFFF", size=16))
            self.btn_delete_selection.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_invert_selection"):
            self.btn_invert_selection.setIcon(get_icon("invert", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=14))
            self.btn_invert_selection.setIconSize(QSize(14, 14))
        if hasattr(self, "btn_select_all"):
            self.btn_select_all.setIcon(get_icon("select-all", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=14))
            self.btn_select_all.setIconSize(QSize(14, 14))
        if hasattr(self, "btn_clear_selection"):
            self.btn_clear_selection.setIcon(get_icon("deselect", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=14))
            self.btn_clear_selection.setIconSize(QSize(14, 14))

        # Panel Lotes (Batch)
        if hasattr(self, "btn_batch_input"):
            self.btn_batch_input.setIcon(get_icon("folder-open", color=fg_color, active_color=accent_color, size=16))
            self.btn_batch_input.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_batch_output"):
            self.btn_batch_output.setIcon(get_icon("folder", color=fg_color, active_color=accent_color, size=16))
            self.btn_batch_output.setIconSize(QSize(16, 16))
        if hasattr(self, "btn_run_batch"):
            self.btn_run_batch.setIcon(get_icon("rocket", color="#FFFFFF", size=16))
            self.btn_run_batch.setIconSize(QSize(16, 16))

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
        # Sección: Detección Multiobjeto con IA (YOLO11-seg)
        # ---------------------------------------------------------
        self.box_subjects = QGroupBox(tr("header_detected_subjects"))
        self.box_subjects.setObjectName("detectedSubjectsBox")
        box_subj_layout = QVBoxLayout(self.box_subjects)
        box_subj_layout.setContentsMargins(10, 12, 10, 10)
        box_subj_layout.setSpacing(8)

        self.btn_detect_subjects = QPushButton(tr("btn_detect_subjects"))
        self.btn_detect_subjects.setObjectName("secondaryActionButton")
        self.btn_detect_subjects.setFixedHeight(34)
        self.btn_detect_subjects.clicked.connect(self.detectSubjectsRequested.emit)
        box_subj_layout.addWidget(self.btn_detect_subjects)

        self.lbl_detect_status = QLabel(tr("detect_status_idle"))
        self.lbl_detect_status.setObjectName("shortcutDesc")
        self.lbl_detect_status.setWordWrap(True)
        box_subj_layout.addWidget(self.lbl_detect_status)

        # Contenedor de chips de sujetos
        self.chips_container = QWidget()
        self.chips_layout = QVBoxLayout(self.chips_container)
        self.chips_layout.setContentsMargins(0, 0, 0, 0)
        self.chips_layout.setSpacing(6)
        box_subj_layout.addWidget(self.chips_container)

        # Botones de acción del sujeto seleccionado
        self.subject_actions_widget = QWidget()
        subj_act_layout = QHBoxLayout(self.subject_actions_widget)
        subj_act_layout.setContentsMargins(0, 0, 0, 0)
        subj_act_layout.setSpacing(6)

        self.btn_isolate_subject = QPushButton(tr("btn_isolate_subject"))
        self.btn_isolate_subject.setObjectName("secondaryActionButton")
        self.btn_isolate_subject.setFixedHeight(30)
        self.btn_isolate_subject.clicked.connect(self._on_isolate_subject_clicked)
        subj_act_layout.addWidget(self.btn_isolate_subject)

        self.btn_select_subject = QPushButton(tr("btn_select_subject"))
        self.btn_select_subject.setObjectName("secondaryActionButton")
        self.btn_select_subject.setFixedHeight(30)
        self.btn_select_subject.clicked.connect(self._on_select_subject_clicked)
        subj_act_layout.addWidget(self.btn_select_subject)

        self.subject_actions_widget.setVisible(False)
        box_subj_layout.addWidget(self.subject_actions_widget)

        layout.addWidget(self.box_subjects)

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
        self.btn_swatch_custom = QPushButton()
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
        
        self.btn_tool_wand = QPushButton(tr("btn_wand"))
        self.btn_tool_wand.setObjectName("brushToolModeButton")
        self.btn_tool_wand.setCheckable(True)
        self.btn_tool_wand.setFixedHeight(32)
        self.btn_tool_wand.setToolTip(tr("tip_wand"))
        
        self.brush_mode_group.addButton(self.btn_tool_paint, 0)
        self.brush_mode_group.addButton(self.btn_tool_erase, 1)
        self.brush_mode_group.addButton(self.btn_tool_wand, 2)
        
        self.brush_mode_group.idClicked.connect(self._on_brush_tool_clicked)
        
        mode_row.addWidget(self.btn_tool_paint)
        mode_row.addWidget(self.btn_tool_erase)
        mode_row.addWidget(self.btn_tool_wand)
        layout.addLayout(mode_row)

        # ---------- Opciones del Pincel / Borrador (se ocultan con la Varita) ----------
        brush_ctrl_box = QGroupBox(tr("brush_settings"))
        brush_ctrl_box.setObjectName("brushOptionsBox")
        self.brush_options_box = brush_ctrl_box
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

        # ---------- Opciones de la Varita Mágica (estilo barra de opciones de Photoshop) ----------
        self.wand_options_box = QGroupBox(tr("wand_settings"))
        self.wand_options_box.setObjectName("wandOptionsBox")
        w_layout = QVBoxLayout(self.wand_options_box)
        w_layout.setContentsMargins(12, 14, 12, 10)
        w_layout.setSpacing(10)

        # Selector de Tipo de Varita: Por Color vs Por IA (MobileSAM)
        wand_type_row = QHBoxLayout()
        wand_type_row.setSpacing(6)
        self.wand_type_group = QButtonGroup(self)
        self.wand_type_group.setExclusive(True)

        self.btn_wand_type_color = QPushButton(tr("wand_type_color"))
        self.btn_wand_type_color.setObjectName("presetButton")
        self.btn_wand_type_color.setCheckable(True)
        self.btn_wand_type_color.setChecked(True)
        self.btn_wand_type_color.setFixedHeight(30)
        self.btn_wand_type_color.setToolTip(tr("tip_wand_color"))
        self.btn_wand_type_color.setCursor(Qt.CursorShape.PointingHandCursor)
        self.wand_type_group.addButton(self.btn_wand_type_color, 0)
        wand_type_row.addWidget(self.btn_wand_type_color)

        self.btn_wand_type_ai = QPushButton(tr("wand_type_ai"))
        self.btn_wand_type_ai.setObjectName("presetButton")
        self.btn_wand_type_ai.setCheckable(True)
        self.btn_wand_type_ai.setFixedHeight(30)
        self.btn_wand_type_ai.setToolTip(tr("tip_wand_ai"))
        self.btn_wand_type_ai.setCursor(Qt.CursorShape.PointingHandCursor)
        self.wand_type_group.addButton(self.btn_wand_type_ai, 1)
        wand_type_row.addWidget(self.btn_wand_type_ai)
        w_layout.addLayout(wand_type_row)

        self.btn_wand_type_color.clicked.connect(lambda: self._on_wand_type_clicked(False))
        self.btn_wand_type_ai.clicked.connect(lambda: self._on_wand_type_clicked(True))

        # Modo de selección: Nueva / Añadir / Restar / Intersecar
        lbl_sel_mode = QLabel(tr("lbl_wand_mode"))
        lbl_sel_mode.setObjectName("fieldLabel")
        w_layout.addWidget(lbl_sel_mode)

        sel_mode_row = QHBoxLayout()
        sel_mode_row.setSpacing(4)
        self.wand_mode_group = QButtonGroup(self)
        self.wand_mode_group.setExclusive(True)
        self._wand_mode_ids = {}
        self.wand_mode_buttons = []
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
            sel_mode_row.addWidget(btn)
        self.wand_mode_group.idClicked.connect(
            lambda i: self.wandSelectionModeChanged.emit(self._wand_mode_ids[i])
        )
        w_layout.addLayout(sel_mode_row)

        # Contenedor de opciones específicas para Varita por Color
        self.wand_color_widget = QWidget()
        cw_layout = QVBoxLayout(self.wand_color_widget)
        cw_layout.setContentsMargins(0, 0, 0, 0)
        cw_layout.setSpacing(8)

        # Tolerancia (0-255, por defecto 32 como en Photoshop)
        tol_row = QHBoxLayout()
        lbl_tol = QLabel(tr("lbl_tolerance"))
        lbl_tol.setFixedWidth(72)
        lbl_tol.setToolTip(tr("tip_tolerance"))
        tol_row.addWidget(lbl_tol)
        self.slider_wand_tolerance = QSlider(Qt.Orientation.Horizontal)
        self.slider_wand_tolerance.setRange(0, 255)
        self.slider_wand_tolerance.setValue(32)
        self.slider_wand_tolerance.setToolTip(tr("tip_tolerance"))
        self.slider_wand_tolerance.valueChanged.connect(self._on_slider_wand_tolerance_changed)
        self.lbl_wand_tolerance = QLabel("32")
        self.lbl_wand_tolerance.setObjectName("brushSizeLabel")
        self.lbl_wand_tolerance.setFixedWidth(34)
        self.lbl_wand_tolerance.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tol_row.addWidget(self.slider_wand_tolerance)
        tol_row.addWidget(self.lbl_wand_tolerance)
        cw_layout.addLayout(tol_row)

        # Suavizar (anti-alias) y Contiguo
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

        # Contenedor de opciones específicas para Varita por IA (MobileSAM)
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

        lbl_wand_hint = QLabel(tr("wand_hint"))
        lbl_wand_hint.setObjectName("shortcutDesc")
        lbl_wand_hint.setWordWrap(True)
        w_layout.addWidget(lbl_wand_hint)
        layout.addWidget(self.wand_options_box)

        # ---------- Acciones sobre la selección ----------
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

        sel_actions_row = QHBoxLayout()
        sel_actions_row.setSpacing(6)
        self.btn_invert_selection = QPushButton(tr("btn_invert_selection"))
        self.btn_invert_selection.setObjectName("presetButton")
        self.btn_invert_selection.setFixedHeight(30)
        self.btn_invert_selection.setToolTip(tr("tip_invert_selection"))
        self.btn_invert_selection.clicked.connect(self.invertSelectionRequested.emit)

        self.btn_select_all = QPushButton(tr("btn_select_all"))
        self.btn_select_all.setObjectName("presetButton")
        self.btn_select_all.setFixedHeight(30)
        self.btn_select_all.setToolTip(tr("tip_select_all"))
        self.btn_select_all.clicked.connect(self.selectAllRequested.emit)

        self.btn_clear_selection = QPushButton(tr("btn_clear_selection"))
        self.btn_clear_selection.setObjectName("presetButton")
        self.btn_clear_selection.setFixedHeight(30)
        self.btn_clear_selection.setToolTip(tr("tip_clear_selection"))
        self.btn_clear_selection.clicked.connect(self.clearSelectionRequested.emit)

        sel_actions_row.addWidget(self.btn_invert_selection)
        sel_actions_row.addWidget(self.btn_select_all)
        sel_actions_row.addWidget(self.btn_clear_selection)
        s_layout.addLayout(sel_actions_row)
        layout.addWidget(self.selection_box)

        self._has_selection = False
        self._active_brush_tool = 0
        self.set_selection_info(0, 0)
        self._apply_brush_tool_visibility()

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
        self.brushModeToggled.emit(is_brush)

    def _on_brush_tool_clicked(self, index: int):
        self.eraserModeToggled.emit(index == 1)
        self.wandModeToggled.emit(index == 2)
        self._active_brush_tool = index
        self._apply_brush_tool_visibility()

    def _apply_brush_tool_visibility(self):
        """
        Muestra solo las opciones de la herramienta activa (como la barra de opciones
        de Photoshop): Pincel/Borrador -> ajustes del pincel; Varita -> opciones de
        selección. El bloque de Selección aparece con la Varita o si hay selección activa.
        """
        is_wand = self._active_brush_tool == 2
        self.brush_options_box.setVisible(not is_wand)
        self.wand_options_box.setVisible(is_wand)
        self.selection_box.setVisible(is_wand or self._has_selection)
        self._refresh_stack_geometry()

    def _refresh_stack_geometry(self):
        """Recalcula la altura del panel tras mostrar/ocultar grupos para no dejar huecos."""
        self.panel_brush.updateGeometry()
        self.panel_brush.adjustSize()
        if hasattr(self, "stack"):
            self.stack.updateGeometry()
            if hasattr(self, "scroll_content"):
                self.scroll_content.adjustSize()

    def set_selection_info(self, selected_px: int, total_px: int):
        """Actualiza el estado del bloque de selección (contador y botones)."""
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
        self.btn_invert_selection.setEnabled(has_sel)
        self.btn_clear_selection.setEnabled(has_sel)
        self._apply_brush_tool_visibility()

    def _on_wand_type_clicked(self, is_ai: bool):
        self.wand_color_widget.setVisible(not is_ai)
        self.wand_ai_widget.setVisible(is_ai)
        self._refresh_stack_geometry()
        self.wandAIModeToggled.emit(is_ai)

    def set_wand_ai_status(self, text: str):
        if hasattr(self, "lbl_wand_ai_status"):
            self.lbl_wand_ai_status.setText(text)

    def is_wand_ai_mode(self) -> bool:
        return getattr(self, "btn_wand_type_ai", None) is not None and self.btn_wand_type_ai.isChecked()

    @property
    def btn_wand(self):
        """Compatibilidad con referencias a btn_wand."""
        return getattr(self, "btn_tool_wand", None)

    def is_wand_tool_active(self) -> bool:
        """Indica si la herramienta de varita está seleccionada en el sidebar."""
        return getattr(self, "btn_tool_wand", None) is not None and self.btn_tool_wand.isChecked()

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

    def retranslate_ui(self):
        """Actualiza las etiquetas y botones de la barra lateral al cambiar idioma."""
        if hasattr(self, "btn_rail_bg"):
            self.btn_rail_bg.setText(tr("sidebar_tab_bg"))
        if hasattr(self, "btn_rail_restore"):
            self.btn_rail_restore.setText(tr("sidebar_tab_restore"))
        if hasattr(self, "btn_rail_brush"):
            self.btn_rail_brush.setText(tr("sidebar_tab_brush"))
        if hasattr(self, "btn_rail_batch"):
            self.btn_rail_batch.setText(tr("sidebar_tab_batch"))
        if hasattr(self, "btn_remove_bg"):
            self.btn_remove_bg.setText(tr("btn_remove_now"))
        if hasattr(self, "btn_autocrop"):
            self.btn_autocrop.setText(tr("btn_autocrop_now"))
            self.btn_autocrop.setToolTip(tr("btn_autocrop_tooltip"))
        if hasattr(self, "btn_restore"):
            self.btn_restore.setText(tr("btn_scale_now"))
        if hasattr(self, "btn_tool_paint"):
            self.btn_tool_paint.setText(tr("btn_paint"))
        if hasattr(self, "btn_tool_erase"):
            self.btn_tool_erase.setText(tr("btn_erase"))
        if hasattr(self, "btn_tool_wand"):
            self.btn_tool_wand.setText(tr("btn_wand"))
        if hasattr(self, "btn_delete_selection"):
            self.btn_delete_selection.setText(tr("btn_delete_selection"))
            self.btn_delete_selection.setToolTip(tr("tip_delete_selection"))
        if hasattr(self, "btn_invert_selection"):
            self.btn_invert_selection.setText(tr("btn_invert_selection"))
            self.btn_invert_selection.setToolTip(tr("tip_invert_selection"))
        if hasattr(self, "btn_select_all"):
            self.btn_select_all.setText(tr("btn_select_all"))
            self.btn_select_all.setToolTip(tr("tip_select_all"))
        if hasattr(self, "btn_clear_selection"):
            self.btn_clear_selection.setText(tr("btn_clear_selection"))
            self.btn_clear_selection.setToolTip(tr("tip_clear_selection"))
        if hasattr(self, "btn_run_batch"):
            self.btn_run_batch.setText(tr("btn_run_batch"))
        if hasattr(self, "chk_tiling"):
            self.chk_tiling.setText(tr("chk_tiling"))
            self.chk_tiling.setToolTip(tr("chk_tiling_tooltip"))
        if hasattr(self, "chk_clothing_protection"):
            self.chk_clothing_protection.setText(tr("chk_clothing"))
            self.chk_clothing_protection.setToolTip(tr("chk_clothing_tooltip"))
        if hasattr(self, "chk_batch_clothing"):
            self.chk_batch_clothing.setText(tr("chk_batch_clothing"))
        if hasattr(self, "selection_box"):
            self.selection_box.setTitle(tr("selection_box"))
        if hasattr(self, "wand_options_box"):
            self.wand_options_box.setTitle(tr("wand_settings"))
        if hasattr(self, "btn_wand_type_color"):
            self.btn_wand_type_color.setText(tr("wand_type_color"))
            self.btn_wand_type_color.setToolTip(tr("tip_wand_color"))
        if hasattr(self, "btn_wand_type_ai"):
            self.btn_wand_type_ai.setText(tr("wand_type_ai"))
            self.btn_wand_type_ai.setToolTip(tr("tip_wand_ai"))
        if hasattr(self, "lbl_wand_ai_desc"):
            self.lbl_wand_ai_desc.setText(tr("wand_ai_desc"))
        if hasattr(self, "lbl_wand_ai_status") and self.lbl_wand_ai_status.text() in [
            "IA lista para seleccionar", "AI ready to select", "ИИ готов к выделению", "AI 已就绪"
        ]:
            self.lbl_wand_ai_status.setText(tr("wand_ai_ready"))

        if hasattr(self, "combo_bg_model"):
            curr_idx = self.combo_bg_model.currentIndex()
            for i in range(self.combo_bg_model.count()):
                m_id = self.combo_bg_model.itemData(i)
                self.combo_bg_model.setItemText(i, tr(f"model_{m_id}"))
            self.combo_bg_model.setCurrentIndex(curr_idx)

        if hasattr(self, "combo_batch_model"):
            curr_idx = self.combo_batch_model.currentIndex()
            for i in range(self.combo_batch_model.count()):
                m_id = self.combo_batch_model.itemData(i)
                self.combo_batch_model.setItemText(i, tr(f"model_{m_id}"))
            self.combo_batch_model.setCurrentIndex(curr_idx)

        if hasattr(self, "box_subjects"):
            self.box_subjects.setTitle(tr("header_detected_subjects"))
        if hasattr(self, "btn_detect_subjects"):
            self.btn_detect_subjects.setText(tr("btn_detect_subjects"))
        if hasattr(self, "btn_isolate_subject"):
            self.btn_isolate_subject.setText(tr("btn_isolate_subject"))
        if hasattr(self, "btn_select_subject"):
            self.btn_select_subject.setText(tr("btn_select_subject"))

    def set_detected_subjects(self, subjects: list):
        """Puebla los chips interactivos para los sujetos detectados por YOLO11."""
        self._current_detected_subjects = subjects
        self._active_detected_subject = None
        self.subject_actions_widget.setVisible(False)

        # Limpiar chips previos
        while self.chips_layout.count() > 0:
            item = self.chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not subjects:
            self.lbl_detect_status.setText(tr("detect_none_found"))
            return

        self.lbl_detect_status.setText(f"{len(subjects)} objetos encontrados:")
        self.subject_chip_group = QButtonGroup(self)
        self.subject_chip_group.setExclusive(True)

        for i, s in enumerate(subjects):
            btn = QPushButton(f"{s.emoji}  {s.class_name.capitalize()} ({int(s.confidence * 100)}%)")
            btn.setObjectName("subjectChipButton")
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid #44403C;
                    border-left: 4px solid {s.color_hex};
                    border-radius: 6px;
                    padding: 4px 10px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: 500;
                }}
                QPushButton:checked {{
                    background-color: {s.color_hex}33;
                    border-color: {s.color_hex};
                    font-weight: 700;
                }}
            """)
            btn.clicked.connect(lambda checked, subj=s: self._on_subject_chip_clicked(subj))
            self.subject_chip_group.addButton(btn, i)
            self.chips_layout.addWidget(btn)

    def set_detecting_subjects_state(self, is_detecting: bool):
        self.btn_detect_subjects.setEnabled(not is_detecting)
        if is_detecting:
            self.lbl_detect_status.setText(tr("detect_status_computing"))
        else:
            self.lbl_detect_status.setText(tr("detect_status_idle"))

    def _on_subject_chip_clicked(self, subject):
        self._active_detected_subject = subject
        self.subject_actions_widget.setVisible(True)

    def _on_isolate_subject_clicked(self):
        if getattr(self, "_active_detected_subject", None):
            self.isolateSubjectRequested.emit(self._active_detected_subject)

    def _on_select_subject_clicked(self):
        if getattr(self, "_active_detected_subject", None):
            self.selectSubjectRequested.emit(self._active_detected_subject)

