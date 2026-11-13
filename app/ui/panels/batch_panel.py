"""
Panel de procesamiento masivo por lotes (Batch) para el sidebar de TimoitaseBG.
"""
from PySide6.QtCore import Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFrame,
    QFileDialog,
)

from app.config import BG_REMOVER_MODELS
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

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        batch_card = QFrame()
        batch_card.setObjectName("Card")
        b_layout = QVBoxLayout(batch_card)
        b_layout.setContentsMargins(14, 14, 14, 14)
        b_layout.setSpacing(10)

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
        self.combo_batch_model.setFixedHeight(38)
        for model_id, _model_label in BG_REMOVER_MODELS:
            self.combo_batch_model.addItem(tr(f"model_{model_id}"), model_id)
        b_layout.addWidget(self.combo_batch_model)

        self.chk_batch_clothing = QCheckBox(tr("chk_batch_clothing"))
        self.chk_batch_clothing.setChecked(True)
        b_layout.addWidget(self.chk_batch_clothing)

        b_layout.addSpacing(4)

        self.btn_run_batch = QPushButton(tr("btn_run_batch"))
        self.btn_run_batch.setObjectName("ActionBtn")
        self.btn_run_batch.setFixedHeight(40)
        self.btn_run_batch.setEnabled(False)
        self.btn_run_batch.clicked.connect(self._on_run_batch_clicked)
        b_layout.addWidget(self.btn_run_batch)

        layout.addWidget(batch_card)
        layout.addStretch()

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
        self.btn_run_batch.setText(tr("btn_run_batch"))

        curr_idx = self.combo_batch_model.currentIndex()
        for i in range(self.combo_batch_model.count()):
            m_id = self.combo_batch_model.itemData(i)
            self.combo_batch_model.setItemText(i, tr(f"model_{m_id}"))
        self.combo_batch_model.setCurrentIndex(curr_idx)
