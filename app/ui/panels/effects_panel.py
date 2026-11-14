"""
Panel modular de Estudio Creativo (Efectos Sticker/Sombra, Bokeh/Imagen BG, Crop manual y Transformaciones).
"""
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.i18n import tr
from app.ui.icons import get_icon
from app.ui.widgets.toggles import create_section_header


class EffectsPanel(QWidget):
    """Subpanel del ToolSidebar para efectos creativos, fondos y recorte."""

    applyEffectsRequested = Signal(dict)
    transformRequested = Signal(int, bool, bool)
    cropModeToggled = Signal(bool, object)
    cropApplyRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_dark_theme: bool = True
        self._sticker_color: Tuple[int, int, int] = (255, 255, 255)
        self._solid_color: Tuple[int, int, int] = (255, 255, 255)
        self._bg_image_path: str = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # --- TARJETA 1: Sticker & Sombra ---
        self.fx_card = QFrame()
        self.fx_card.setObjectName("Card")
        fx_layout = QVBoxLayout(self.fx_card)
        fx_layout.setContentsMargins(14, 14, 14, 14)
        fx_layout.setSpacing(8)

        hdr_fx, self._lbl_fx_hdr, self._info_fx_hdr = create_section_header(
            tr("header_effects_studio"),
            tr("effects_studio_desc"),
        )
        fx_layout.addLayout(hdr_fx)

        sticker_row = QHBoxLayout()
        self.lbl_sticker_width = QLabel(f"{tr('lbl_sticker_outline')}: 0 px")
        self.btn_sticker_color = QPushButton()
        self.btn_sticker_color.setFixedSize(26, 22)
        self.btn_sticker_color.setToolTip(tr("tip_sticker_color"))
        self.btn_sticker_color.clicked.connect(self._pick_sticker_color)
        sticker_row.addWidget(self.lbl_sticker_width, 1)
        sticker_row.addWidget(self.btn_sticker_color)
        fx_layout.addLayout(sticker_row)

        self.slider_sticker_width = QSlider(Qt.Orientation.Horizontal)
        self.slider_sticker_width.setRange(0, 40)
        self.slider_sticker_width.setValue(0)
        self.slider_sticker_width.valueChanged.connect(self._update_slider_labels)
        fx_layout.addWidget(self.slider_sticker_width)

        self.lbl_shadow_opacity = QLabel(f"{tr('lbl_drop_shadow')}: 0%")
        fx_layout.addWidget(self.lbl_shadow_opacity)
        self.slider_shadow_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_shadow_opacity.setRange(0, 100)
        self.slider_shadow_opacity.setValue(0)
        self.slider_shadow_opacity.valueChanged.connect(self._update_slider_labels)
        fx_layout.addWidget(self.slider_shadow_opacity)

        self.lbl_shadow_blur = QLabel(f"{tr('lbl_shadow_blur')}: 16 px")
        fx_layout.addWidget(self.lbl_shadow_blur)
        self.slider_shadow_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_shadow_blur.setRange(0, 40)
        self.slider_shadow_blur.setValue(16)
        self.slider_shadow_blur.valueChanged.connect(self._update_slider_labels)
        fx_layout.addWidget(self.slider_shadow_blur)

        layout.addWidget(self.fx_card)

        # --- TARJETA 2: Composición de Fondo & Aspecto ---
        self.bg_card = QFrame()
        self.bg_card.setObjectName("Card")
        bg_layout = QVBoxLayout(self.bg_card)
        bg_layout.setContentsMargins(14, 14, 14, 14)
        bg_layout.setSpacing(8)

        self.lbl_bg_mode = QLabel(tr("lbl_bg_composition"))
        bg_layout.addWidget(self.lbl_bg_mode)
        self.combo_bg_mode = QComboBox()
        self.combo_bg_mode.setFixedHeight(34)
        self._populate_bg_modes()
        self.combo_bg_mode.currentIndexChanged.connect(self._on_bg_mode_changed)
        bg_layout.addWidget(self.combo_bg_mode)

        self.lbl_bokeh_blur = QLabel(f"{tr('lbl_bokeh_blur')}: 20 px")
        bg_layout.addWidget(self.lbl_bokeh_blur)
        self.slider_bokeh_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_bokeh_blur.setRange(2, 50)
        self.slider_bokeh_blur.setValue(20)
        self.slider_bokeh_blur.valueChanged.connect(self._update_slider_labels)
        bg_layout.addWidget(self.slider_bokeh_blur)

        img_row = QHBoxLayout()
        self.btn_pick_bg_image = QPushButton(tr("btn_pick_bg_image"))
        self.btn_pick_bg_image.setObjectName("secondaryActionButton")
        self.btn_pick_bg_image.setFixedHeight(32)
        self.btn_pick_bg_image.clicked.connect(self._pick_bg_image)
        self.btn_pick_solid_color = QPushButton(tr("btn_pick_solid_color"))
        self.btn_pick_solid_color.setObjectName("secondaryActionButton")
        self.btn_pick_solid_color.setFixedHeight(32)
        self.btn_pick_solid_color.clicked.connect(self._pick_solid_color)
        img_row.addWidget(self.btn_pick_bg_image)
        img_row.addWidget(self.btn_pick_solid_color)
        bg_layout.addLayout(img_row)

        self.lbl_bg_image_name = QLabel(tr("lbl_no_bg_image"))
        self.lbl_bg_image_name.setObjectName("MutedText")
        self.lbl_bg_image_name.setWordWrap(True)
        bg_layout.addWidget(self.lbl_bg_image_name)

        self.lbl_aspect = QLabel(tr("lbl_aspect_preset"))
        bg_layout.addWidget(self.lbl_aspect)
        self.combo_aspect = QComboBox()
        self.combo_aspect.setFixedHeight(34)
        self._populate_aspect_presets()
        self.combo_aspect.currentIndexChanged.connect(self._on_aspect_changed)
        bg_layout.addWidget(self.combo_aspect)

        self.btn_apply_effects = QPushButton(tr("btn_apply_effects"))
        self.btn_apply_effects.setObjectName("ActionBtn")
        self.btn_apply_effects.setFixedHeight(38)
        self.btn_apply_effects.clicked.connect(self._emit_apply_effects)
        bg_layout.addWidget(self.btn_apply_effects)

        layout.addWidget(self.bg_card)

        # --- TARJETA 3: Transformar & Recortar (Crop) ---
        self.geom_card = QFrame()
        self.geom_card.setObjectName("Card")
        geom_layout = QVBoxLayout(self.geom_card)
        geom_layout.setContentsMargins(14, 14, 14, 14)
        geom_layout.setSpacing(8)

        self.lbl_transform_hdr = QLabel(tr("header_transform_crop"))
        geom_layout.addWidget(self.lbl_transform_hdr)

        tr_row = QHBoxLayout()
        self.btn_rotate_90 = QPushButton(tr("btn_rotate_90"))
        self.btn_rotate_90.setObjectName("secondaryActionButton")
        self.btn_rotate_90.setFixedHeight(32)
        self.btn_rotate_90.clicked.connect(lambda: self.transformRequested.emit(90, False, False))

        self.btn_flip_h = QPushButton(tr("btn_flip_h"))
        self.btn_flip_h.setObjectName("secondaryActionButton")
        self.btn_flip_h.setFixedHeight(32)
        self.btn_flip_h.clicked.connect(lambda: self.transformRequested.emit(0, True, False))

        self.btn_flip_v = QPushButton(tr("btn_flip_v"))
        self.btn_flip_v.setObjectName("secondaryActionButton")
        self.btn_flip_v.setFixedHeight(32)
        self.btn_flip_v.clicked.connect(lambda: self.transformRequested.emit(0, False, True))

        tr_row.addWidget(self.btn_rotate_90)
        tr_row.addWidget(self.btn_flip_h)
        tr_row.addWidget(self.btn_flip_v)
        geom_layout.addLayout(tr_row)

        crop_row = QHBoxLayout()
        self.btn_toggle_crop = QPushButton(tr("btn_toggle_crop"))
        self.btn_toggle_crop.setObjectName("secondaryActionButton")
        self.btn_toggle_crop.setCheckable(True)
        self.btn_toggle_crop.setFixedHeight(34)
        self.btn_toggle_crop.toggled.connect(self._on_crop_toggled)

        self.btn_apply_crop = QPushButton(tr("btn_apply_crop"))
        self.btn_apply_crop.setObjectName("ActionBtn")
        self.btn_apply_crop.setFixedHeight(34)
        self.btn_apply_crop.setEnabled(False)
        self.btn_apply_crop.clicked.connect(self.cropApplyRequested.emit)

        crop_row.addWidget(self.btn_toggle_crop, 1)
        crop_row.addWidget(self.btn_apply_crop, 1)
        geom_layout.addLayout(crop_row)

        layout.addWidget(self.geom_card)
        layout.addStretch()

        self._refresh_color_buttons()
        self._on_bg_mode_changed()

    def _populate_bg_modes(self):
        cur = self.combo_bg_mode.currentData()
        self.combo_bg_mode.blockSignals(True)
        self.combo_bg_mode.clear()
        self.combo_bg_mode.addItem(tr("bg_mode_transparent"), "transparent")
        self.combo_bg_mode.addItem(tr("bg_mode_solid"), "solid")
        self.combo_bg_mode.addItem(tr("bg_mode_bokeh"), "bokeh")
        self.combo_bg_mode.addItem(tr("bg_mode_image"), "image")
        idx = self.combo_bg_mode.findData(cur)
        self.combo_bg_mode.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_bg_mode.blockSignals(False)

    def _populate_aspect_presets(self):
        cur = self.combo_aspect.currentData()
        self.combo_aspect.blockSignals(True)
        self.combo_aspect.clear()
        self.combo_aspect.addItem(tr("aspect_original"), "original")
        self.combo_aspect.addItem("512×512 (Sticker)", "512x512")
        self.combo_aspect.addItem("1:1", "1:1")
        self.combo_aspect.addItem("16:9", "16:9")
        self.combo_aspect.addItem("9:16", "9:16")
        idx = self.combo_aspect.findData(cur)
        self.combo_aspect.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_aspect.blockSignals(False)

    def _refresh_color_buttons(self):
        r, g, b = self._sticker_color
        self.btn_sticker_color.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #64748B; border-radius: 4px;"
        )

    def _pick_sticker_color(self):
        r, g, b = self._sticker_color
        color = QColorDialog.getColor(QColor(r, g, b), self, tr("tip_sticker_color"))
        if color.isValid():
            self._sticker_color = (color.red(), color.green(), color.blue())
            self._refresh_color_buttons()

    def _pick_solid_color(self):
        r, g, b = self._solid_color
        color = QColorDialog.getColor(QColor(r, g, b), self, tr("btn_pick_solid_color"))
        if color.isValid():
            self._solid_color = (color.red(), color.green(), color.blue())

    def _pick_bg_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("btn_pick_bg_image"),
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self._bg_image_path = path
            self.lbl_bg_image_name.setText(Path(path).name)
            idx = self.combo_bg_mode.findData("image")
            if idx >= 0:
                self.combo_bg_mode.setCurrentIndex(idx)

    def _update_slider_labels(self):
        self.lbl_sticker_width.setText(
            f"{tr('lbl_sticker_outline')}: {self.slider_sticker_width.value()} px"
        )
        self.lbl_shadow_opacity.setText(
            f"{tr('lbl_drop_shadow')}: {self.slider_shadow_opacity.value()}%"
        )
        self.lbl_shadow_blur.setText(
            f"{tr('lbl_shadow_blur')}: {self.slider_shadow_blur.value()} px"
        )
        self.lbl_bokeh_blur.setText(
            f"{tr('lbl_bokeh_blur')}: {self.slider_bokeh_blur.value()} px"
        )

    def _on_bg_mode_changed(self):
        mode = self.combo_bg_mode.currentData() or "transparent"
        is_bokeh = mode == "bokeh"
        is_image = mode == "image"
        is_solid = mode == "solid"
        self.lbl_bokeh_blur.setVisible(is_bokeh)
        self.slider_bokeh_blur.setVisible(is_bokeh)
        self.btn_pick_bg_image.setVisible(is_image or mode == "transparent")
        self.lbl_bg_image_name.setVisible(is_image or bool(self._bg_image_path))
        self.btn_pick_solid_color.setVisible(is_solid)

    def _aspect_ratio_value(self) -> Optional[float]:
        preset = self.combo_aspect.currentData()
        if preset in ("1:1", "512x512"):
            return 1.0
        if preset == "16:9":
            return 16.0 / 9.0
        if preset == "9:16":
            return 9.0 / 16.0
        return None

    def _on_aspect_changed(self):
        if self.btn_toggle_crop.isChecked():
            self.cropModeToggled.emit(True, self._aspect_ratio_value())

    def _on_crop_toggled(self, checked: bool):
        self.btn_apply_crop.setEnabled(checked)
        self.cropModeToggled.emit(checked, self._aspect_ratio_value())

    def reset_crop_toggle(self):
        if self.btn_toggle_crop.isChecked():
            self.btn_toggle_crop.blockSignals(True)
            self.btn_toggle_crop.setChecked(False)
            self.btn_toggle_crop.blockSignals(False)
        self.btn_apply_crop.setEnabled(False)

    def _emit_apply_effects(self):
        payload = {
            "sticker_width": self.slider_sticker_width.value(),
            "sticker_color": self._sticker_color,
            "shadow_opacity": self.slider_shadow_opacity.value() / 100.0,
            "shadow_blur": self.slider_shadow_blur.value(),
            "shadow_offset_x": 10,
            "shadow_offset_y": 12,
            "bg_mode": self.combo_bg_mode.currentData() or "transparent",
            "bg_color": self._solid_color,
            "bokeh_blur": self.slider_bokeh_blur.value(),
            "bg_image_path": self._bg_image_path,
            "aspect_preset": self.combo_aspect.currentData() or "original",
        }
        self.applyEffectsRequested.emit(payload)

    def apply_theme(self, is_dark: bool):
        self._is_dark_theme = is_dark
        icon_color = "#CBD5E1" if is_dark else "#475569"
        self.btn_rotate_90.setIcon(get_icon("rotate-cw", color=icon_color, size=15))

    def retranslate_ui(self):
        self._lbl_fx_hdr.setText(tr("header_effects_studio"))
        self._info_fx_hdr.setToolTip(tr("effects_studio_desc"))
        self.btn_sticker_color.setToolTip(tr("tip_sticker_color"))
        self.lbl_bg_mode.setText(tr("lbl_bg_composition"))
        self.btn_pick_bg_image.setText(tr("btn_pick_bg_image"))
        self.btn_pick_solid_color.setText(tr("btn_pick_solid_color"))
        if not self._bg_image_path:
            self.lbl_bg_image_name.setText(tr("lbl_no_bg_image"))
        self.lbl_aspect.setText(tr("lbl_aspect_preset"))
        self.btn_apply_effects.setText(tr("btn_apply_effects"))
        self.lbl_transform_hdr.setText(tr("header_transform_crop"))
        self.btn_rotate_90.setText(tr("btn_rotate_90"))
        self.btn_flip_h.setText(tr("btn_flip_h"))
        self.btn_flip_v.setText(tr("btn_flip_v"))
        self.btn_toggle_crop.setText(tr("btn_toggle_crop"))
        self.btn_apply_crop.setText(tr("btn_apply_crop"))
        self._populate_bg_modes()
        self._populate_aspect_presets()
        self._update_slider_labels()
