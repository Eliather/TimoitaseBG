"""
Panel de eliminación de fondo, detección YOLO11, recorte automático y muestras de color.
"""
from typing import Tuple
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor
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

from app.config import BG_REMOVER_MODELS
from app.i18n import tr
from app.ui.icons import get_icon
from app.ui.widgets.toggles import create_section_header


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
        self._has_transparency: bool = False
        self._can_restore_transparency: bool = False
        self._current_detected_subjects: list = []
        self._active_detected_subject = None
        self._chip_buttons: list = []
        self._setup_ui()

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

        self.chk_clothing_protection = QCheckBox(tr("chk_clothing"))
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

        self.btn_detect_subjects = QPushButton(tr("btn_detect_subjects"))
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

        self.btn_swatch_none = self._make_swatch_button("swatchNoneButton", "swatch_transparent", text="∅")
        self.btn_swatch_white = self._make_swatch_button("swatchWhiteButton", "swatch_white", checked=True)
        self.btn_swatch_black = self._make_swatch_button("swatchBlackButton", "swatch_black")
        self.btn_swatch_gray = self._make_swatch_button("swatchGrayButton", "swatch_gray")
        self.btn_swatch_custom = self._make_swatch_button("swatchCustomButton", "swatch_custom")

        for btn in (
            self.btn_swatch_none,
            self.btn_swatch_white,
            self.btn_swatch_black,
            self.btn_swatch_gray,
            self.btn_swatch_custom,
        ):
            self.swatch_group.addButton(btn)
            swatches_row.addWidget(btn)

        swatches_row.addStretch()
        layout.addLayout(swatches_row)

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

        self.btn_swatch_none.clicked.connect(self._on_swatch_none_clicked)
        self.btn_swatch_white.clicked.connect(lambda: self._on_color_swatch_clicked((255, 255, 255), tr("color_white")))
        self.btn_swatch_black.clicked.connect(lambda: self._on_color_swatch_clicked((0, 0, 0), tr("color_black")))
        self.btn_swatch_gray.clicked.connect(lambda: self._on_color_swatch_clicked((226, 232, 240), tr("color_gray")))
        self.btn_swatch_custom.clicked.connect(self._on_swatch_custom_clicked)

        self._update_color_preview((255, 255, 255), tr("color_white"))
        layout.addStretch()

    def _make_swatch_button(self, obj_name: str, tip_key: str, text: str = "", checked: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName(obj_name)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setFixedSize(36, 36)
        btn.setToolTip(tr(tip_key))
        btn.setEnabled(False)
        return btn

    def apply_theme(self, is_dark: bool):
        self._is_dark_theme = is_dark
        fg_color = "#EAEAEA" if is_dark else "#292524"
        accent_color = "#FF4F79"
        dis_color = "#666666" if is_dark else "#A8A29E"

        self.btn_remove_bg.setIcon(get_icon("sparkles", color="#FFFFFF", size=16))
        self.btn_remove_bg.setIconSize(QSize(16, 16))
        self.btn_autocrop.setIcon(get_icon("crop", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=16))
        self.btn_autocrop.setIconSize(QSize(16, 16))
        self.btn_swatch_custom.setText("")
        self.btn_swatch_custom.setIcon(get_icon("palette", color=fg_color, active_color=accent_color, disabled_color=dis_color, size=18))
        self.btn_swatch_custom.setIconSize(QSize(18, 18))

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
        self.chk_clothing_protection.setText(tr("chk_clothing"))
        self.chk_clothing_protection.setToolTip(tr("chk_clothing_tooltip"))

        self.btn_swatch_none.setToolTip(tr("swatch_transparent"))
        self.btn_swatch_white.setToolTip(tr("swatch_white"))
        self.btn_swatch_black.setToolTip(tr("swatch_black"))
        self.btn_swatch_gray.setToolTip(tr("swatch_gray"))
        self.btn_swatch_custom.setToolTip(tr("swatch_custom"))

        curr_idx = self.combo_bg_model.currentIndex()
        for i in range(self.combo_bg_model.count()):
            m_id = self.combo_bg_model.itemData(i)
            self.combo_bg_model.setItemText(i, tr(f"model_{m_id}"))
        self.combo_bg_model.setCurrentIndex(curr_idx)

        self.box_subjects.setTitle(tr("header_detected_subjects"))
        self.btn_detect_subjects.setText(tr("btn_detect_subjects"))
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
        bd = "#475569" if self._is_dark_theme else "#CBD5E1"
        self.color_preview_box.setStyleSheet(
            f"background-color: transparent; border: 1.5px solid {bd}; border-radius: 4px;"
        )
        self.lbl_color_hex.setText("Transparente")
        self.restoreTransparencyRequested.emit()

    def _on_color_swatch_clicked(self, color_rgb: Tuple[int, int, int], label: str):
        self._selected_is_none = False
        self._selected_color_rgb = color_rgb
        self._update_color_preview(color_rgb, label)
        self.applyBgColorRequested.emit(color_rgb)

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

        for btn in (
            self.btn_swatch_none,
            self.btn_swatch_white,
            self.btn_swatch_black,
            self.btn_swatch_gray,
            self.btn_swatch_custom,
        ):
            btn.setEnabled(has_transparency)

        if not has_transparency:
            self.btn_swatch_none.setChecked(True)
            self._selected_is_none = True

    def set_busy(self, busy: bool):
        self.btn_remove_bg.setEnabled(not busy)
        if busy:
            self.btn_autocrop.setEnabled(False)
        else:
            self.set_transparency_available(self._has_transparency, self._can_restore_transparency)

    def _style_subject_chip(self, btn: QPushButton, subject, is_dark: bool):
        chip_bg = "#1E293B" if is_dark else "#F8FAFC"
        chip_hover = "#334155" if is_dark else "#F1F5F9"
        chip_bd = "#334155" if is_dark else "#E2E8F0"
        chip_fg = "#F8FAFC" if is_dark else "#1E293B"
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {chip_bg};
                color: {chip_fg};
                border: 1px solid {chip_bd};
                border-left: 4px solid {subject.color_hex};
                border-radius: 6px;
                padding: 4px 10px;
                text-align: left;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {chip_hover};
            }}
            QPushButton:checked {{
                background-color: {subject.color_hex}33;
                border-color: {subject.color_hex};
                color: {chip_fg};
                font-weight: 700;
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
