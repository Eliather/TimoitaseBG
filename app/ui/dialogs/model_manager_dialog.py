"""
Diálogo modal para gestionar descargas, cuantización INT8 e integridad SHA-256 de modelos ONNX.
"""
from typing import Optional
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
    QProgressBar,
)

from app.config import MODELS_DIR
from app.core.model_fetcher import delete_local_model, get_models_inventory
from app.core.workers import ModelActionWorker
from app.i18n import tr
from app.ui.icons import get_icon


class ModelManagerDialog(QDialog):
    """Gestor visual de modelos locales ONNX con soporte INT8, SHA-256 y cancelación."""

    modelsChanged = Signal()

    def __init__(self, parent=None, is_dark_theme: bool = True):
        super().__init__(parent)
        self.is_dark_theme = is_dark_theme
        self._worker: Optional[ModelActionWorker] = None
        self._modified = False

        self.setWindowTitle(tr("models_dialog_title"))
        self.setModal(True)
        self.resize(700, 520)
        self.setMinimumSize(620, 440)

        self._setup_ui()
        self._reload_rows()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        self.lbl_title = QLabel(tr("models_dialog_title"))
        self.lbl_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.lbl_dir = QLabel(tr("models_dir_info", path=str(MODELS_DIR)))
        self.lbl_dir.setObjectName("SubTitle")
        self.lbl_dir.setWordWrap(True)
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_dir)
        header.addLayout(title_box, stretch=1)

        self.lbl_total_size = QLabel("")
        self.lbl_total_size.setObjectName("SubTitle")
        self.lbl_total_size.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.lbl_total_size)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        self._list_layout.setSpacing(8)
        scroll.setWidget(self._list_container)
        root.addWidget(scroll, stretch=1)

        # Barra de progreso para acciones activas
        prog_row = QHBoxLayout()
        prog_row.setSpacing(8)
        self.lbl_status = QLabel(tr("models_status_idle"))
        self.lbl_status.setObjectName("SubTitle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setVisible(False)
        self.btn_cancel = QPushButton(tr("btn_cancel"))
        self.btn_cancel.setObjectName("statusCancelButton")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_worker)

        prog_row.addWidget(self.lbl_status, stretch=1)
        prog_row.addWidget(self.progress_bar)
        prog_row.addWidget(self.btn_cancel)
        root.addLayout(prog_row)

        footer = QHBoxLayout()
        footer.addStretch()
        self.btn_close = QPushButton(tr("btn_ok"))
        self.btn_close.setObjectName("PrimaryBtn")
        self.btn_close.setFixedHeight(34)
        self.btn_close.clicked.connect(self.accept)
        footer.addWidget(self.btn_close)
        root.addLayout(footer)

    def _clear_rows(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _reload_rows(self):
        self._clear_rows()
        inventory = get_models_inventory()
        total_mb = sum(item["total_bytes"] for item in inventory) / (1024 * 1024)
        self.lbl_total_size.setText(tr("models_total_size", mb=f"{total_mb:.1f}"))

        busy = self._worker is not None and self._worker.isRunning()
        fg = "#EAEAEA" if self.is_dark_theme else "#292524"
        accent = "#FF4F79"

        for entry in inventory:
            fname = entry["filename"]
            card = QFrame()
            card.setObjectName("Card")
            row = QHBoxLayout(card)
            row.setContentsMargins(12, 10, 12, 10)
            row.setSpacing(10)

            info_col = QVBoxLayout()
            info_col.setSpacing(2)
            name_lbl = QLabel(fname)
            name_lbl.setStyleSheet("font-weight: 600;")

            variant = entry["active_variant"]
            mb = entry["total_bytes"] / (1024 * 1024)
            if variant == "int8":
                status_str = tr("models_state_int8", mb=f"{mb:.1f}")
            elif variant == "fp32":
                status_str = tr("models_state_fp32", mb=f"{mb:.1f}")
            else:
                status_str = tr("models_state_missing")
            if entry["has_sha256"]:
                status_str += f" • {tr('models_sha_registered')}"

            sub_lbl = QLabel(status_str)
            sub_lbl.setObjectName("SubTitle")
            info_col.addWidget(name_lbl)
            info_col.addWidget(sub_lbl)
            row.addLayout(info_col, stretch=1)

            btn_dl = QPushButton(tr("models_btn_download"))
            btn_dl.setFixedHeight(30)
            btn_dl.setIcon(get_icon("download", color=fg, active_color=accent, size=14))
            btn_dl.setIconSize(QSize(14, 14))
            btn_dl.setEnabled(not busy and not entry["fp32_exists"])
            btn_dl.clicked.connect(lambda _c=False, f=fname: self._start_action("download", f))
            row.addWidget(btn_dl)

            btn_int8 = QPushButton("INT8")
            btn_int8.setFixedHeight(30)
            btn_int8.setToolTip(tr("models_tip_quantize"))
            btn_int8.setEnabled(not busy and entry["fp32_exists"] and not entry["int8_exists"])
            btn_int8.clicked.connect(lambda _c=False, f=fname: self._start_action("quantize", f))
            row.addWidget(btn_int8)

            btn_verify = QPushButton("SHA-256")
            btn_verify.setFixedHeight(30)
            btn_verify.setToolTip(tr("models_tip_verify"))
            btn_verify.setEnabled(not busy and entry["fp32_exists"] and entry["has_sha256"])
            btn_verify.clicked.connect(lambda _c=False, f=fname: self._start_action("verify", f))
            row.addWidget(btn_verify)

            btn_del = QPushButton()
            btn_del.setFixedSize(30, 30)
            btn_del.setToolTip(tr("models_tip_delete"))
            btn_del.setIcon(get_icon("trash", color=fg, active_color="#EF4444", size=14))
            btn_del.setIconSize(QSize(14, 14))
            btn_del.setEnabled(not busy and (entry["fp32_exists"] or entry["int8_exists"]))
            btn_del.clicked.connect(lambda _c=False, f=fname: self._delete_model(f))
            row.addWidget(btn_del)

            self._list_layout.addWidget(card)

        self._list_layout.addStretch()

    def _start_action(self, action: str, filename: str):
        if self._worker is not None and self._worker.isRunning():
            return
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_cancel.setVisible(action == "download")
        self.lbl_status.setText(tr("models_action_running", action=action, name=filename))

        worker = ModelActionWorker(action, filename)
        worker.progressPercent.connect(self.progress_bar.setValue)
        worker.progressText.connect(self.lbl_status.setText)
        worker.finishedAction.connect(self._on_action_finished)
        worker.cancelled.connect(self._on_action_cancelled)
        worker.failed.connect(self._on_action_failed)
        self._worker = worker
        self._reload_rows()
        worker.start()

    def _cancel_worker(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self.btn_cancel.setEnabled(False)

    def _on_action_finished(self, filename: str, status: str):
        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        if status == "sha_ok":
            self.lbl_status.setText(tr("models_sha_ok", name=filename))
        elif status == "sha_fail":
            self.lbl_status.setText(tr("models_sha_fail", name=filename))
        elif status == "quantized":
            self._modified = True
            self.lbl_status.setText(tr("models_quantized_ok", name=filename))
        else:
            self._modified = True
            self.lbl_status.setText(tr("models_downloaded_ok", name=filename))
        self._worker = None
        self._reload_rows()
        if self._modified:
            self.modelsChanged.emit()

    def _on_action_cancelled(self):
        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText(tr("status_cancelled"))
        self._worker = None
        self._reload_rows()

    def _on_action_failed(self, err: str):
        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText(tr("models_action_failed", error=err))
        self._worker = None
        self._reload_rows()

    def _delete_model(self, filename: str):
        removed = delete_local_model(filename)
        if removed:
            self._modified = True
            self.lbl_status.setText(tr("models_deleted_ok", name=filename))
            self._reload_rows()
            self.modelsChanged.emit()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        super().closeEvent(event)
