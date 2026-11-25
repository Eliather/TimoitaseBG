"""
Panel de eliminación de fondo, detección YOLO11, recorte automático y muestras de color.
"""
from typing import Optional, Tuple
from PySide6.QtCore import Qt, Signal, QSize, QRectF, QSettings
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QButtonGroup,
    QGroupBox,
    QFrame,
    QColorDialog,
)

from app.config import APP_AUTHOR, APP_NAME, BG_REMOVER_MODELS
from app.i18n import tr
from app.ui.icons import get_icon, render_icon
from app.ui.widgets.toggles import create_section_header

_LAST_CUSTOM_COLOR_KEY = "bg/last_custom_color"
_DEFAULT_LAST_CUSTOM_COLOR = "#2E7CF6"


class _ColorSwatchButton(QPushButton):
    """Botón de muestra de color con renderizado vectorial antialiasado HiDPI."""

    def __init__(
        self,
        mode: str = "solid",
        color: Optional[QColor] = None,
        is_dark: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("", parent)
        self._mode: str = mode
        self._color: QColor = QColor(color) if color is not None else QColor("#FFFFFF")
        self._is_dark: bool = is_dark
        self._hovered: bool = False
        self.setCheckable(True)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    def set_dark_theme(self, is_dark: bool) -> None:
        if self._is_dark != is_dark:
            self._is_dark = is_dark
            self.update()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = float(self.width())
        h = float(self.height())
        radius = 8.0
        enabled = self.isEnabled()
        checked = self.isChecked()

        if checked:
            fill_rect = QRectF(3.5, 3.5, w - 7.0, h - 7.0)
            fill_radius = 5.5
        else:
            fill_rect = QRectF(1.5, 1.5, w - 3.0, h - 3.0)
            fill_radius = radius - 0.5
        clip_path = QPainterPath()
        clip_path.addRoundedRect(fill_rect, fill_radius, fill_radius)

        painter.save()
        painter.setClipPath(clip_path)

        if self._mode == "checkerboard":
            if self._is_dark:
                c_light = QColor("#4C5564") if enabled else QColor("#3D4450")
                c_dark = QColor("#222730") if enabled else QColor("#1D2129")
            else:
                c_light = QColor("#FFFFFF") if enabled else QColor("#F8FAFC")
                c_dark = QColor("#D5DCE6") if enabled else QColor("#E2E8F0")

            painter.fillRect(fill_rect, c_light)
            cells = 4
            cell_w = fill_rect.width() / cells
            cell_h = fill_rect.height() / cells
            for row in range(cells):
                for col in range(cells):
                    if (row + col) % 2 == 1:
                        cell_rect = QRectF(
                            fill_rect.left() + col * cell_w,
                            fill_rect.top() + row * cell_h,
                            cell_w + 0.25,
                            cell_h + 0.25,
                        )
                        painter.fillRect(cell_rect, c_dark)

        elif self._mode == "solid":
            painter.fillPath(clip_path, self._color)

        elif self._mode == "palette":
            if self._is_dark:
                if not enabled:
                    bg = QColor("#1A1E24")
                elif self._hovered:
                    bg = QColor("#272D36")
                else:
                    bg = QColor("#1D2127")
            else:
                if not enabled:
                    bg = QColor("#F2F4F7")
                elif self._hovered:
                    bg = QColor("#E2E6EC")
                else:
                    bg = QColor("#ECEFF2")

            painter.fillPath(clip_path, bg)

            if checked:
                icon_color = "#2E7CF6" if self._is_dark else "#1D63DC"
            elif not enabled:
                icon_color = "#6E7785" if self._is_dark else "#9BA3AE"
            else:
                icon_color = "#F2F4F7" if self._is_dark else "#14171A"

            icon_size = 18.0
            icon_rect = QRectF(
                (w - icon_size) / 2.0,
                (h - icon_size) / 2.0,
                icon_size,
                icon_size,
            )
            render_icon(painter, "palette", icon_rect, color=icon_color)

        painter.restore()

        if checked:
            accent = QColor("#2E7CF6") if self._is_dark else QColor("#1D63DC")
            if not enabled:
                accent.setAlphaF(0.65)
            ring_pen = QPen(accent, 2.0)
            ring_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0), radius, radius)
        else:
            if self._is_dark:
                if enabled and self._hovered:
                    border = QColor(255, 255, 255, 128)
                elif enabled:
                    border = QColor(255, 255, 255, 68)
                else:
                    border = QColor(255, 255, 255, 44)
            else:
                if enabled and self._hovered:
                    border = QColor("#94A3B8")
                elif enabled:
                    border = QColor("#CBD5E1")
                else:
                    border = QColor(203, 213, 225, 166)

            border_pen = QPen(border, 1.0)
            border_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(fill_rect, fill_radius, fill_radius)

        painter.end()


class BackgroundPanel(QWidget):
    """Panel modular para segmentación de fondo, detección YOLO11 y fondo sólido."""

    removeBgRequested = Signal(str, bool)
    autoCropRequested = Signal()
    applyBgColorRequested = Signal(tuple)
    restoreTransparencyRequested = Signal()
    detectSubjectsRequested = Signal()
    isolateSubjectRequested = Signal(object)
    selectSubjectRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_dark_theme: bool = True
        self._selected_is_none: bool = False
        self._selected_color_rgb: Tuple[int, int, int] = (255, 255, 255)
        self._last_custom_color: QColor = self._load_last_custom_color()
        self._has_transparency: bool = False
        self._can_restore_transparency: bool = False
        self._current_detected_subjects: list = []
        self._active_detected_subject = None
        self._chip_buttons: list = []
        self._swatch_buttons: Tuple[_ColorSwatchButton, ...] = ()
        self._setup_ui()

    @staticmethod
    def _load_last_custom_color() -> QColor:
        try:
            settings = QSettings(APP_AUTHOR, APP_NAME)
            raw = settings.value(_LAST_CUSTOM_COLOR_KEY, _DEFAULT_LAST_CUSTOM_COLOR)
            if isinstance(raw, str):
                candidate = QColor(raw.strip())
                if candidate.isValid():
                    return candidate
        except Exception:
            pass
        return QColor(_DEFAULT_LAST_CUSTOM_COLOR)

    @staticmethod
    def _save_last_custom_color(color: QColor) -> None:
        if not color.isValid():
            return
        try:
            settings = QSettings(APP_AUTHOR, APP_NAME)
            settings.setValue(_LAST_CUSTOM_COLOR_KEY, color.name().upper())
            settings.sync()
        except Exception:
            pass

    def _format_last_color_tooltip(self) -> str:
        hex_str = self._last_custom_color.name().upper()
        translated = tr("swatch_last_color", hex=hex_str)
        if translated == "swatch_last_color":
            return f"Último color guardado ({hex_str})"
        return translated

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # --- TARJETA 1: Segmentación ---
        self.seg_card = QFrame()
        self.seg_card.setObjectName("Card")
        seg_layout = QVBoxLayout(self.seg_card)
        seg_layout.setContentsMargins(14, 14, 14, 14)
        seg_layout.setSpacing(10)

        header_model, self._lbl_model_hdr, self._info_model_hdr = create_section_header(
            tr("header_model"),
            tr("model_desc"),
        )
        seg_layout.addLayout(header_model)

        self.combo_bg_model = QComboBox()
        self.combo_bg_model.setFixedHeight(38)
        for model_id, _model_label in BG_REMOVER_MODELS:
            self.combo_bg_model.addItem(tr(f"model_{model_id}"), model_id)
        self.combo_bg_model.setCurrentIndex(0)
        seg_layout.addWidget(self.combo_bg_model)

        self.chk_clothing_protection = QCheckBox(tr("chk_batch_clothing"))
        self.chk_clothing_protection.setChecked(True)
        self.chk_clothing_protection.setToolTip(tr("chk_clothing_tooltip"))
        seg_layout.addWidget(self.chk_clothing_protection)

        seg_layout.addSpacing(4)

        self.btn_remove_bg = QPushButton(tr("btn_remove_now"))
        self.btn_remove_bg.setObjectName("ActionBtn")
        self.btn_remove_bg.setFixedHeight(40)
        self.btn_remove_bg.clicked.connect(self._on_remove_bg_clicked)
        seg_layout.addWidget(self.btn_remove_bg)

        layout.addWidget(self.seg_card)

        # --- TARJETA 2: Detección YOLO11 ---
        self.box_subjects = QGroupBox(tr("header_detected_subjects"))
        self.box_subjects.setObjectName("Card")
        box_subj_layout = QVBoxLayout(self.box_subjects)
        box_subj_layout.setContentsMargins(14, 14, 14, 14)
        box_subj_layout.setSpacing(8)

        self.btn_detect_subjects = QPushButton(tr("btn_detect_subjects").lstrip("🔍 ").strip())
        self.btn_detect_subjects.setObjectName("secondaryActionButton")
        self.btn_detect_subjects.setFixedHeight(34)
        self.btn_detect_subjects.clicked.connect(self.detectSubjectsRequested.emit)
        box_subj_layout.addWidget(self.btn_detect_subjects)

        self.lbl_detect_status = QLabel(tr("detect_status_idle"))
        self.lbl_detect_status.setObjectName("SubTitle")
        self.lbl_detect_status.setWordWrap(True)
        box_subj_layout.addWidget(self.lbl_detect_status)

        self.chips_container = QWidget()
        self.chips_layout = QVBoxLayout(self.chips_container)
        self.chips_layout.setContentsMargins(0, 0, 0, 0)
        self.chips_layout.setSpacing(6)
        box_subj_layout.addWidget(self.chips_container)

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

        # --- SECCIÓN: Ajuste de Lienzo y Fondo ---
        self._lbl_crop = QLabel(tr("header_crop"))
        self._lbl_crop.setObjectName("FieldLabel")
        layout.addWidget(self._lbl_crop)

        self.btn_autocrop = QPushButton(tr("btn_autocrop_now"))
        self.btn_autocrop.setObjectName("secondaryActionButton")
        self.btn_autocrop.setFixedHeight(36)
        self.btn_autocrop.setToolTip(tr("btn_autocrop_tooltip"))
        self.btn_autocrop.setEnabled(False)
        self.btn_autocrop.clicked.connect(self.autoCropRequested.emit)
        layout.addWidget(self.btn_autocrop)

        self._lbl_bg_section = QLabel(tr("header_bg"))
        self._lbl_bg_section.setObjectName("FieldLabel")
        layout.addWidget(self._lbl_bg_section)

        swatches_row = QHBoxLayout()
        swatches_row.setSpacing(8)

        self.swatch_group = QButtonGroup(self)
        self.swatch_group.setExclusive(True)

        self.btn_swatch_none = self._make_swatch_button(
            "swatchNoneButton",
            tr("swatch_transparent"),
            mode="checkerboard",
        )
        self.btn_swatch_white = self._make_swatch_button(
            "swatchWhiteButton",
            tr("swatch_white"),
            mode="solid",
            color=QColor("#FFFFFF"),
            checked=True,
        )
        self.btn_swatch_black = self._make_swatch_button(
            "swatchBlackButton",
            tr("swatch_black"),
            mode="solid",
            color=QColor("#0B0D10"),
        )
        self.btn_swatch_gray = self._make_swatch_button(
            "swatchGrayButton",
            tr("swatch_gray"),
            mode="solid",
            color=QColor("#7C8594"),
        )
        self.btn_swatch_last_color = self._make_swatch_button(
            "swatchLastColorButton",
            self._format_last_color_tooltip(),
            mode="solid",
            color=self._last_custom_color,
        )
        self.btn_swatch_custom = self._make_swatch_button(
            "swatchCustomButton",
            tr("swatch_custom"),
            mode="palette",
        )

        self._swatch_buttons = (
            self.btn_swatch_none,
            self.btn_swatch_white,
            self.btn_swatch_black,
            self.btn_swatch_gray,
            self.btn_swatch_last_color,
            self.btn_swatch_custom,
        )
        for btn in self._swatch_buttons:
            self.swatch_group.addButton(btn)
            swatches_row.addWidget(btn)

        swatches_row.addStretch()
        layout.addLayout(swatches_row)

        self.custom_color_row = QWidget()
        color_action_row = QHBoxLayout(self.custom_color_row)
        color_action_row.setContentsMargins(0, 0, 0, 0)
        color_action_row.setSpacing(8)

        self.color_preview_box = QFrame()
        self.color_preview_box.setObjectName("swatchPreview")
        self.color_preview_box.setFixedSize(22, 22)
        color_action_row.addWidget(self.color_preview_box)

        self.lbl_color_hex = QLabel(tr("color_white"))
        self.lbl_color_hex.setObjectName("colorHexLabel")
        color_action_row.addWidget(self.lbl_color_hex)

        color_action_row.addStretch()
        self.custom_color_row.setVisible(False)
        layout.addWidget(self.custom_color_row)

        self.btn_swatch_none.clicked.connect(self._on_swatch_none_clicked)
        self.btn_swatch_white.clicked.connect(lambda: self._on_color_swatch_clicked((255, 255, 255), tr("color_white")))
        self.btn_swatch_black.clicked.connect(lambda: self._on_color_swatch_clicked((0, 0, 0), tr("color_black")))
        self.btn_swatch_gray.clicked.connect(lambda: self._on_color_swatch_clicked((226, 232, 240), tr("color_gray")))
        self.btn_swatch_last_color.clicked.connect(self._on_swatch_last_color_clicked)
        self.btn_swatch_custom.clicked.connect(self._on_swatch_custom_clicked)

        self._update_color_preview((255, 255, 255), tr("color_white"))
        layout.addStretch()

    def _make_swatch_button(
        self,
        obj_name: str,
        tooltip: str,
        mode: str = "solid",
        color: Optional[QColor] = None,
        checked: bool = False,
    ) -> _ColorSwatchButton:
        btn = _ColorSwatchButton(mode=mode, color=color, is_dark=self._is_dark_theme)
        btn.setObjectName(obj_name)
        btn.setChecked(checked)
        btn.setToolTip(tooltip)
        btn.setEnabled(False)
        return btn

    def apply_theme(self, is_dark: bool):
        self._is_dark_theme = is_dark
        fg_color = "#F2F4F7" if is_dark else "#14171A"
        accent_color = "#2E7CF6" if is_dark else "#1D63DC"
        dis_color = "#5C6470" if is_dark else "#9BA3AE"

        self.btn_remove_bg.setIcon(get_icon("sparkles", color="#FFFFFF", size=16))
        self.btn_remove_bg.setIconSize(QSize(16, 16))
        self.btn_detect_subjects.setIcon(get_icon("zoom-in", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=15))
        self.btn_detect_subjects.setIconSize(QSize(15, 15))
        self.btn_autocrop.setIcon(get_icon("crop", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
        self.btn_autocrop.setIconSize(QSize(16, 16))

        for btn in self._swatch_buttons:
            btn.set_dark_theme(is_dark)

        self._update_color_preview(self._selected_color_rgb, self.lbl_color_hex.text())
        for btn, subject in self._chip_buttons:
            self._style_subject_chip(btn, subject, is_dark)

    def retranslate_ui(self):
        self._lbl_model_hdr.setText(tr("header_model"))
        self._info_model_hdr.setToolTip(tr("model_desc"))
        self.btn_remove_bg.setText(tr("btn_remove_now"))
        self._lbl_crop.setText(tr("header_crop"))
        self.btn_autocrop.setText(tr("btn_autocrop_now"))
        self.btn_autocrop.setToolTip(tr("btn_autocrop_tooltip"))
        self._lbl_bg_section.setText(tr("header_bg"))
        self.chk_clothing_protection.setText(tr("chk_batch_clothing"))
        self.chk_clothing_protection.setToolTip(tr("chk_clothing_tooltip"))

        self.btn_swatch_none.setToolTip(tr("swatch_transparent"))
        self.btn_swatch_white.setToolTip(tr("swatch_white"))
        self.btn_swatch_black.setToolTip(tr("swatch_black"))
        self.btn_swatch_gray.setToolTip(tr("swatch_gray"))
        self.btn_swatch_last_color.setToolTip(self._format_last_color_tooltip())
        self.btn_swatch_custom.setToolTip(tr("swatch_custom"))

        curr_idx = self.combo_bg_model.currentIndex()
        for i in range(self.combo_bg_model.count()):
            m_id = self.combo_bg_model.itemData(i)
            self.combo_bg_model.setItemText(i, tr(f"model_{m_id}"))
        self.combo_bg_model.setCurrentIndex(curr_idx)

        self.box_subjects.setTitle(tr("header_detected_subjects"))
        self.btn_detect_subjects.setText(tr("btn_detect_subjects").lstrip("🔍 ").strip())
        self.btn_isolate_subject.setText(tr("btn_isolate_subject"))
        self.btn_select_subject.setText(tr("btn_select_subject"))
        if not self._current_detected_subjects and self.btn_detect_subjects.isEnabled():
            self.lbl_detect_status.setText(tr("detect_status_idle"))

    def is_clothing_protection_enabled(self) -> bool:
        return self.chk_clothing_protection.isChecked()

    def _on_remove_bg_clicked(self):
        model_id = self.combo_bg_model.currentData()
        self.removeBgRequested.emit(model_id, self.is_clothing_protection_enabled())

    def _on_swatch_none_clicked(self):
        self._selected_is_none = True
        self.custom_color_row.setVisible(False)
        bd = "#475569" if self._is_dark_theme else "#CBD5E1"
        self.color_preview_box.setStyleSheet(
            f"background-color: transparent; border: 1.5px solid {bd}; border-radius: 4px;"
        )
        self.lbl_color_hex.setText("Transparente")
        self.restoreTransparencyRequested.emit()

    def _on_color_swatch_clicked(self, color_rgb: Tuple[int, int, int], label: str):
        self._selected_is_none = False
        self._selected_color_rgb = color_rgb
        self.custom_color_row.setVisible(False)
        self._update_color_preview(color_rgb, label)
        self.applyBgColorRequested.emit(color_rgb)

    def _on_swatch_last_color_clicked(self):
        c = self._last_custom_color
        rgb = (c.red(), c.green(), c.blue())
        hex_str = c.name().upper()
        self._selected_is_none = False
        self._selected_color_rgb = rgb
        self.btn_swatch_last_color.setChecked(True)
        self.custom_color_row.setVisible(True)
        self._update_color_preview(rgb, f"Personalizado ({hex_str})")
        self.applyBgColorRequested.emit(rgb)

    def _on_swatch_custom_clicked(self):
        initial = QColor(*self._selected_color_rgb)
        color = QColorDialog.getColor(
            initial,
            self,
            "Seleccionar Color de Fondo",
            options=QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            hex_str = f"#{color.red():02X}{color.green():02X}{color.blue():02X}"
            self._selected_is_none = False
            self._selected_color_rgb = rgb
            self._last_custom_color = QColor(color)
            self.btn_swatch_last_color.set_color(self._last_custom_color)
            self.btn_swatch_last_color.setToolTip(self._format_last_color_tooltip())
            self._save_last_custom_color(self._last_custom_color)
            self.btn_swatch_last_color.setChecked(True)
            self.custom_color_row.setVisible(True)
            self._update_color_preview(rgb, f"Personalizado ({hex_str})")
            self.applyBgColorRequested.emit(rgb)
        else:
            last_rgb = (
                self._last_custom_color.red(),
                self._last_custom_color.green(),
                self._last_custom_color.blue(),
            )
            if self._selected_is_none:
                self.btn_swatch_none.setChecked(True)
            elif self._selected_color_rgb == (255, 255, 255):
                self.btn_swatch_white.setChecked(True)
            elif self._selected_color_rgb == (0, 0, 0):
                self.btn_swatch_black.setChecked(True)
            elif self._selected_color_rgb == (226, 232, 240):
                self.btn_swatch_gray.setChecked(True)
            elif self._selected_color_rgb == last_rgb:
                self.btn_swatch_last_color.setChecked(True)

    def _update_color_preview(self, color_rgb: Tuple[int, int, int], label: str):
        bd = "#475569" if self._is_dark_theme else "#CBD5E1"
        if self._selected_is_none:
            self.color_preview_box.setStyleSheet(
                f"background-color: transparent; border: 1.5px solid {bd}; border-radius: 4px;"
            )
        else:
            r, g, b = color_rgb
            self.color_preview_box.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid {bd}; border-radius: 4px;"
            )
        if label:
            self.lbl_color_hex.setText(label)

    def set_transparency_available(self, has_transparency: bool, can_restore: bool):
        self._has_transparency = has_transparency
        self._can_restore_transparency = can_restore

        self.btn_autocrop.setEnabled(has_transparency)
        if has_transparency:
            self.btn_autocrop.setToolTip("Elimina los márgenes transparentes sobrantes con 10px de respiro.")
        else:
            self.btn_autocrop.setToolTip(tr("btn_autocrop_tooltip"))

        for btn in self._swatch_buttons:
            btn.setEnabled(has_transparency)

        if not has_transparency:
            self.btn_swatch_none.setChecked(True)
            self._selected_is_none = True
            self.custom_color_row.setVisible(False)

    def set_busy(self, busy: bool):
        self.btn_remove_bg.setEnabled(not busy)
        if busy:
            self.btn_autocrop.setEnabled(False)
        else:
            self.set_transparency_available(self._has_transparency, self._can_restore_transparency)

    def _style_subject_chip(self, btn: QPushButton, subject, is_dark: bool):
        chip_bg = "#1D2127" if is_dark else "#ECEFF2"
        chip_hover = "#252A32" if is_dark else "#E2E6EC"
        chip_bd = "rgba(255, 255, 255, 0.08)" if is_dark else "#DCE1E8"
        chip_fg = "#F2F4F7" if is_dark else "#14171A"
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {chip_bg};
                color: {chip_fg};
                border: 1px solid {chip_bd};
                border-left: 3px solid {subject.color_hex};
                border-radius: 7px;
                padding: 4px 10px;
                text-align: left;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {chip_hover};
            }}
            QPushButton:checked {{
                background-color: {subject.color_hex}2E;
                border-color: {subject.color_hex};
                color: {chip_fg};
                font-weight: 600;
            }}
        """)

    def set_detected_subjects(self, subjects: list):
        """Puebla los chips interactivos para los sujetos detectados por YOLO11."""
        self._current_detected_subjects = subjects
        self._active_detected_subject = None
        self._chip_buttons.clear()
        self.subject_actions_widget.setVisible(False)

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
            self._style_subject_chip(btn, s, self._is_dark_theme)
            btn.clicked.connect(lambda checked, subj=s: self._on_subject_chip_clicked(subj))
            self.subject_chip_group.addButton(btn, i)
            self.chips_layout.addWidget(btn)
            self._chip_buttons.append((btn, s))

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
        if self._active_detected_subject:
            self.isolateSubjectRequested.emit(self._active_detected_subject)

    def _on_select_subject_clicked(self):
        if self._active_detected_subject:
            self.selectSubjectRequested.emit(self._active_detected_subject)

    def get_selected_subject(self):
        """Retorna el sujeto YOLO seleccionado actualmente en el panel, o None."""
        return self._active_detected_subject
