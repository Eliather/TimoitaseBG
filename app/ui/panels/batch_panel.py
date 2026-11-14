"""
Panel de procesamiento masivo por lotes (Batch) con pipelines encadenados y presets.
"""
from PySide6.QtCore import QSettings, QSize, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import BG_REMOVER_MODELS
from app.core.batch_pipeline import BatchPipelineConfig
from app.i18n import tr
from app.ui.icons import get_icon
from app.ui.widgets.toggles import create_section_header


class BatchPanel(QWidget):
    """Panel modular para configurar y ejecutar el procesamiento por lotes."""

    batchRequested = Signal(str, str, str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._batch_input_dir: str = ""
        self._batch_output_dir: str = ""
        self._setup_ui()
        self._load_saved_pipeline()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        batch_card = QFrame()
        batch_card.setObjectName("Card")
        b_layout = QVBoxLayout(batch_card)
        b_layout.setContentsMargins(14, 14, 14, 14)
        b_layout.setSpacing(8)

        header_batch, self._lbl_batch_hdr, self._info_batch_hdr = create_section_header(
            tr("header_batch"),
            tr("batch_desc"),
        )
        b_layout.addLayout(header_batch)

        self.btn_batch_input = QPushButton(tr("btn_batch_input"))
        self.btn_batch_input.setFixedHeight(36)
        self.btn_batch_input.clicked.connect(self._on_batch_input_clicked)
        b_layout.addWidget(self.btn_batch_input)

        self.lbl_batch_input = QLabel(tr("batch_no_folder"))
        self.lbl_batch_input.setObjectName("SubTitle")
        self.lbl_batch_input.setWordWrap(True)
        b_layout.addWidget(self.lbl_batch_input)

        self.btn_batch_output = QPushButton(tr("btn_batch_output"))
        self.btn_batch_output.setFixedHeight(36)
        self.btn_batch_output.clicked.connect(self._on_batch_output_clicked)
        b_layout.addWidget(self.btn_batch_output)

        self.lbl_batch_output = QLabel(tr("batch_no_folder"))
        self.lbl_batch_output.setObjectName("SubTitle")
        self.lbl_batch_output.setWordWrap(True)
        b_layout.addWidget(self.lbl_batch_output)

        self._lbl_model = QLabel(tr("lbl_batch_model"))
        self._lbl_model.setObjectName("FieldLabel")
        b_layout.addWidget(self._lbl_model)

        self.combo_batch_model = QComboBox()
        self.combo_batch_model.setFixedHeight(36)
        for model_id, _model_label in BG_REMOVER_MODELS:
            self.combo_batch_model.addItem(tr(f"model_{model_id}"), model_id)
        b_layout.addWidget(self.combo_batch_model)

        self.chk_batch_clothing = QCheckBox(tr("chk_batch_clothing"))
        self.chk_batch_clothing.setChecked(True)
        b_layout.addWidget(self.chk_batch_clothing)

        # --- Tarjeta de Pipeline Encadenado & Presets ---
        self._lbl_preset = QLabel(tr("lbl_batch_preset"))
        self._lbl_preset.setObjectName("FieldLabel")
        b_layout.addWidget(self._lbl_preset)

        self.combo_batch_preset = QComboBox()
        self.combo_batch_preset.setFixedHeight(34)
        self._populate_presets()
        self.combo_batch_preset.currentIndexChanged.connect(self._on_preset_changed)
        b_layout.addWidget(self.combo_batch_preset)

        self._lbl_upscale = QLabel(tr("lbl_batch_upscale"))
        self._lbl_upscale.setObjectName("FieldLabel")
        b_layout.addWidget(self._lbl_upscale)

        self.combo_batch_upscale = QComboBox()
        self.combo_batch_upscale.setFixedHeight(34)
        self._populate_upscale()
        self.combo_batch_upscale.currentIndexChanged.connect(self._save_pipeline_prefs)
        b_layout.addWidget(self.combo_batch_upscale)

        self.chk_batch_autocrop = QCheckBox(tr("chk_batch_autocrop"))
        self.chk_batch_autocrop.toggled.connect(self._save_pipeline_prefs)
        b_layout.addWidget(self.chk_batch_autocrop)

        self.chk_batch_sticker = QCheckBox(tr("chk_batch_sticker"))
        self.chk_batch_sticker.toggled.connect(self._save_pipeline_prefs)
        b_layout.addWidget(self.chk_batch_sticker)

        self._lbl_batch_aspect = QLabel(tr("lbl_aspect_preset"))
        self._lbl_batch_aspect.setObjectName("FieldLabel")
        b_layout.addWidget(self._lbl_batch_aspect)

        self.combo_batch_aspect = QComboBox()
        self.combo_batch_aspect.setFixedHeight(34)
        self._populate_aspect()
        self.combo_batch_aspect.currentIndexChanged.connect(self._save_pipeline_prefs)
        b_layout.addWidget(self.combo_batch_aspect)

        b_layout.addSpacing(4)

        self.btn_run_batch = QPushButton(tr("btn_run_batch"))
        self.btn_run_batch.setObjectName("ActionBtn")
        self.btn_run_batch.setFixedHeight(40)
        self.btn_run_batch.setEnabled(False)
        self.btn_run_batch.clicked.connect(self._on_run_batch_clicked)
        b_layout.addWidget(self.btn_run_batch)

        layout.addWidget(batch_card)
        layout.addStretch()

    def _populate_presets(self):
        cur = self.combo_batch_preset.currentData()
        self.combo_batch_preset.blockSignals(True)
        self.combo_batch_preset.clear()
        self.combo_batch_preset.addItem(tr("batch_preset_custom"), "custom")
        self.combo_batch_preset.addItem(tr("batch_preset_ecommerce"), "ecommerce")
        self.combo_batch_preset.addItem(tr("batch_preset_sticker"), "sticker")
        self.combo_batch_preset.addItem(tr("batch_preset_social"), "social")
        idx = self.combo_batch_preset.findData(cur)
        self.combo_batch_preset.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_batch_preset.blockSignals(False)

    def _populate_upscale(self):
        cur = self.combo_batch_upscale.currentData()
        self.combo_batch_upscale.blockSignals(True)
        self.combo_batch_upscale.clear()
        self.combo_batch_upscale.addItem(tr("batch_upscale_none"), "none")
        self.combo_batch_upscale.addItem("Real-ESRGAN 2x", "x2")
        self.combo_batch_upscale.addItem("Real-ESRGAN 4x", "x4")
        self.combo_batch_upscale.addItem("Enhance Only", "enhance_only")
        idx = self.combo_batch_upscale.findData(cur)
        self.combo_batch_upscale.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_batch_upscale.blockSignals(False)

    def _populate_aspect(self):
        cur = self.combo_batch_aspect.currentData()
        self.combo_batch_aspect.blockSignals(True)
        self.combo_batch_aspect.clear()
        self.combo_batch_aspect.addItem(tr("aspect_original"), "original")
        self.combo_batch_aspect.addItem("512×512 (Sticker)", "512x512")
        self.combo_batch_aspect.addItem("1:1", "1:1")
        self.combo_batch_aspect.addItem("16:9", "16:9")
        self.combo_batch_aspect.addItem("9:16", "9:16")
        idx = self.combo_batch_aspect.findData(cur)
        self.combo_batch_aspect.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_batch_aspect.blockSignals(False)

    def _on_preset_changed(self):
        preset = self.combo_batch_preset.currentData()
        if preset == "ecommerce":
            self.chk_batch_autocrop.setChecked(True)
            self.chk_batch_sticker.setChecked(False)
            self.combo_batch_aspect.setCurrentIndex(self.combo_batch_aspect.findData("1:1"))
        elif preset == "sticker":
            self.chk_batch_autocrop.setChecked(True)
            self.chk_batch_sticker.setChecked(True)
            self.combo_batch_aspect.setCurrentIndex(self.combo_batch_aspect.findData("512x512"))
        elif preset == "social":
            self.chk_batch_autocrop.setChecked(False)
            self.chk_batch_sticker.setChecked(False)
            self.combo_batch_aspect.setCurrentIndex(self.combo_batch_aspect.findData("9:16"))
        self._save_pipeline_prefs()

    def _save_pipeline_prefs(self):
        try:
            s = QSettings("Eliather", "TimoitaseBG")
            s.setValue("batch/upscale", self.combo_batch_upscale.currentData() or "none")
            s.setValue("batch/autocrop", self.chk_batch_autocrop.isChecked())
            s.setValue("batch/sticker", self.chk_batch_sticker.isChecked())
            s.setValue("batch/aspect", self.combo_batch_aspect.currentData() or "original")
        except Exception:
            pass

    def _load_saved_pipeline(self):
        try:
            s = QSettings("Eliather", "TimoitaseBG")
            up = s.value("batch/upscale", "none")
            ac = s.value("batch/autocrop", False, type=bool)
            st = s.value("batch/sticker", False, type=bool)
            asp = s.value("batch/aspect", "original")
            idx_u = self.combo_batch_upscale.findData(up)
            if idx_u >= 0:
                self.combo_batch_upscale.setCurrentIndex(idx_u)
            self.chk_batch_autocrop.setChecked(bool(ac))
            self.chk_batch_sticker.setChecked(bool(st))
            idx_a = self.combo_batch_aspect.findData(asp)
            if idx_a >= 0:
                self.combo_batch_aspect.setCurrentIndex(idx_a)
        except Exception:
            pass

    def get_pipeline_config(self) -> BatchPipelineConfig:
        """Devuelve la configuración encadenada actual del panel Batch."""
        preset = self.combo_batch_preset.currentData()
        bg_color = (255, 255, 255) if preset == "ecommerce" else None
        return BatchPipelineConfig(
            remove_bg=True,
            upscale_mode=self.combo_batch_upscale.currentData() or "none",
            autocrop=self.chk_batch_autocrop.isChecked(),
            sticker_width=10 if self.chk_batch_sticker.isChecked() else 0,
            aspect_preset=self.combo_batch_aspect.currentData() or "original",
            bg_color=bg_color,
        )

    def _on_batch_input_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes de origen")
        if dir_path:
            self._batch_input_dir = dir_path
            self.lbl_batch_input.setText(tr("batch_input_path", path=dir_path))
            self._update_batch_btn_state()

    def _on_batch_output_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de destino")
        if dir_path:
            self._batch_output_dir = dir_path
            self.lbl_batch_output.setText(tr("batch_output_path", path=dir_path))
            self._update_batch_btn_state()

    def _update_batch_btn_state(self):
        self.btn_run_batch.setEnabled(bool(self._batch_input_dir) and bool(self._batch_output_dir))

    def _on_run_batch_clicked(self):
        model_id = self.combo_batch_model.currentData()
        self.batchRequested.emit(
            self._batch_input_dir,
            self._batch_output_dir,
            model_id,
            self.chk_batch_clothing.isChecked(),
        )

    def apply_theme(self, is_dark: bool):
        fg_color = "#EAEAEA" if is_dark else "#292524"
        accent_color = "#FF4F79"

        self.btn_batch_input.setIcon(get_icon("folder-open", color=fg_color, active_color=accent_color, size=16))
        self.btn_batch_input.setIconSize(QSize(16, 16))
        self.btn_batch_output.setIcon(get_icon("folder", color=fg_color, active_color=accent_color, size=16))
        self.btn_batch_output.setIconSize(QSize(16, 16))
        self.btn_run_batch.setIcon(get_icon("rocket", color="#FFFFFF", size=16))
        self.btn_run_batch.setIconSize(QSize(16, 16))

    def retranslate_ui(self):
        self._lbl_batch_hdr.setText(tr("header_batch"))
        self._info_batch_hdr.setToolTip(tr("batch_desc"))
        self.btn_batch_input.setText(tr("btn_batch_input"))
        self.btn_batch_output.setText(tr("btn_batch_output"))
        self.lbl_batch_input.setText(
            tr("batch_input_path", path=self._batch_input_dir) if self._batch_input_dir else tr("batch_no_folder")
        )
        self.lbl_batch_output.setText(
            tr("batch_output_path", path=self._batch_output_dir) if self._batch_output_dir else tr("batch_no_folder")
        )
        self._lbl_model.setText(tr("lbl_batch_model"))
        self.chk_batch_clothing.setText(tr("chk_batch_clothing"))
        self._lbl_preset.setText(tr("lbl_batch_preset"))
        self._lbl_upscale.setText(tr("lbl_batch_upscale"))
        self.chk_batch_autocrop.setText(tr("chk_batch_autocrop"))
        self.chk_batch_sticker.setText(tr("chk_batch_sticker"))
        self._lbl_batch_aspect.setText(tr("lbl_aspect_preset"))
        self.btn_run_batch.setText(tr("btn_run_batch"))
        self._populate_presets()
        self._populate_upscale()
        self._populate_aspect()

        curr_idx = self.combo_batch_model.currentIndex()
        for i in range(self.combo_batch_model.count()):
            m_id = self.combo_batch_model.itemData(i)
            self.combo_batch_model.setItemText(i, tr(f"model_{m_id}"))
        self.combo_batch_model.setCurrentIndex(curr_idx)
