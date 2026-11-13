"""
Panel de mejora y escalado de resolución (Real-ESRGAN) para el sidebar de TimoitaseBG.
"""
from PySide6.QtCore import Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFrame,
)

from app.config import RESTORE_SCALES
from app.i18n import tr
from app.ui.icons import get_icon
from app.ui.widgets.toggles import create_section_header


class RestorePanel(QWidget):
    """Panel modular para restaurar y reescalar imágenes."""

    restoreRequested = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        restore_card = QFrame()
        restore_card.setObjectName("Card")
        res_layout = QVBoxLayout(restore_card)
        res_layout.setContentsMargins(14, 14, 14, 14)
        res_layout.setSpacing(10)

        header_scale, self._lbl_scale_hdr, self._info_scale_hdr = create_section_header(
            tr("header_scale"),
            tr("scale_desc"),
        )
        res_layout.addLayout(header_scale)

        self.combo_restore_scale = QComboBox()
        self.combo_restore_scale.setFixedHeight(38)
        for scale_id, _scale_label in RESTORE_SCALES:
            self.combo_restore_scale.addItem(tr(f"scale_{scale_id}"), scale_id)
        self.combo_restore_scale.setCurrentIndex(0)
        res_layout.addWidget(self.combo_restore_scale)

        self.chk_tiling = QCheckBox(tr("chk_tiling"))
        self.chk_tiling.setChecked(True)
        self.chk_tiling.setToolTip(tr("chk_tiling_tooltip"))
        res_layout.addWidget(self.chk_tiling)

        res_layout.addSpacing(4)

        self.btn_restore = QPushButton(tr("btn_scale_now"))
        self.btn_restore.setObjectName("ActionBtn")
        self.btn_restore.setFixedHeight(40)
        self.btn_restore.clicked.connect(self._on_restore_clicked)
        res_layout.addWidget(self.btn_restore)

        layout.addWidget(restore_card)
        layout.addStretch()

    def _on_restore_clicked(self):
        scale_mode = self.combo_restore_scale.currentData()
        use_tiling = self.chk_tiling.isChecked()
        self.restoreRequested.emit(scale_mode, use_tiling)

    def set_busy(self, busy: bool):
        self.btn_restore.setEnabled(not busy)

    def apply_theme(self, is_dark: bool):
        self.btn_restore.setIcon(get_icon("maximize-2", color="#FFFFFF", size=16))
        self.btn_restore.setIconSize(QSize(16, 16))

    def retranslate_ui(self):
        self._lbl_scale_hdr.setText(tr("header_scale"))
        self._info_scale_hdr.setToolTip(tr("scale_desc"))
        self.btn_restore.setText(tr("btn_scale_now"))
        self.chk_tiling.setText(tr("chk_tiling"))
        self.chk_tiling.setToolTip(tr("chk_tiling_tooltip"))
        curr_idx = self.combo_restore_scale.currentIndex()
        for i in range(self.combo_restore_scale.count()):
            s_id = self.combo_restore_scale.itemData(i)
            self.combo_restore_scale.setItemText(i, tr(f"scale_{s_id}"))
        self.combo_restore_scale.setCurrentIndex(curr_idx)
