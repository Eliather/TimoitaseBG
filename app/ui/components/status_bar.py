"""
Barra inferior de estado, progreso determinista, cancelación y selector de hardware.
"""
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizeGrip,
)

from app.i18n import tr
from app.ui.icons import get_icon


class StatusBar(QFrame):
    """Barra de estado modular con soporte de cancelación cooperativa e i18n en vivo."""

    shortcutsRequested = Signal()
    cancelRequested = Signal()
    deviceToggleRequested = Signal()
    kofiRequested = Signal()

    def __init__(self, size_grip_target=None, parent=None):
        super().__init__(parent)
        self.setObjectName("BottomBar")
        self.setFixedHeight(44)
        self._current_status_i18n = ("status_ready", {})

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # Atajos rápidos
        self.btn_shortcuts = QPushButton(tr("btn_shortcuts"))
        self.btn_shortcuts.setObjectName("statusShortcutsButton")
        self.btn_shortcuts.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_shortcuts.setToolTip(tr("tooltip_shortcuts"))
        self.btn_shortcuts.clicked.connect(self.shortcutsRequested.emit)
        layout.addWidget(self.btn_shortcuts, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Estado e información de imagen
        self.lbl_status = QLabel(tr("status_ready"))
        self.lbl_status.setObjectName("statusText")
        layout.addWidget(self.lbl_status, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.lbl_img_info = QLabel("")
        self.lbl_img_info.setObjectName("imageInfoText")
        layout.addWidget(self.lbl_img_info, alignment=Qt.AlignmentFlag.AlignVCenter)

        layout.addStretch()

        # Barra de progreso + botón cancelar tarea
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.btn_cancel = QPushButton(tr("btn_cancel"))
        self.btn_cancel.setObjectName("statusCancelButton")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setToolTip(tr("tooltip_cancel"))
        self.btn_cancel.clicked.connect(self.cancelRequested.emit)
        self.btn_cancel.setVisible(False)
        layout.addWidget(self.btn_cancel, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Badge clickable de dispositivo GPU/CPU
        self.lbl_device = QPushButton("CPU")
        self.lbl_device.setObjectName("deviceBadgeCpu")
        self.lbl_device.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_device.setToolTip(tr("tooltip_device"))
        self.lbl_device.clicked.connect(self.deviceToggleRequested.emit)
        layout.addWidget(self.lbl_device, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Botón Ko-fi
        self.btn_kofi = QPushButton(tr("btn_kofi"))
        self.btn_kofi.setObjectName("PrimaryBtn")
        self.btn_kofi.setStyleSheet("padding: 5px 14px; border-radius: 14px; font-size: 12px;")
        self.btn_kofi.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_kofi.setToolTip(tr("tooltip_kofi"))
        self.btn_kofi.clicked.connect(self.kofiRequested.emit)
        layout.addWidget(self.btn_kofi, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.size_grip = QSizeGrip(size_grip_target or self)
        self.size_grip.setStyleSheet("background: transparent;")
        layout.addWidget(
            self.size_grip,
            0,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
        )

    def set_status(self, key: str, **kwargs) -> None:
        self._current_status_i18n = (key, kwargs)
        self.lbl_status.setText(tr(key, **kwargs))

    def set_status_raw(self, text: str) -> None:
        self._current_status_i18n = None
        self.lbl_status.setText(text)

    def set_busy(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        self.btn_cancel.setEnabled(busy)

    def set_device_badge(self, text: str, state: str) -> None:
        """Actualiza el badge de hardware ('gpu', 'cpu' o 'blocked')."""
        name_map = {
            "gpu": "deviceBadgeGpu",
            "cpu": "deviceBadgeCpu",
            "blocked": "deviceBadgeBlocked",
        }
        self.lbl_device.setText(text)
        self.lbl_device.setObjectName(name_map.get(state, "deviceBadgeCpu"))
        self.lbl_device.style().unpolish(self.lbl_device)
        self.lbl_device.style().polish(self.lbl_device)

    def apply_theme(self, is_dark: bool) -> None:
        fg_color = "#EAEAEA" if is_dark else "#292524"
        accent_color = "#FF4F79"
        self.btn_kofi.setIcon(get_icon("coffee", color="#FFFFFF", size=16))
        self.btn_kofi.setIconSize(QSize(16, 16))
        self.btn_shortcuts.setIcon(
            get_icon("keyboard", color=fg_color, active_color=accent_color, size=15)
        )
        self.btn_shortcuts.setIconSize(QSize(15, 15))

    def retranslate_ui(self) -> None:
        if self._current_status_i18n:
            k, args = self._current_status_i18n
            self.lbl_status.setText(tr(k, **args))
        self.btn_shortcuts.setText(tr("btn_shortcuts"))
        self.btn_shortcuts.setToolTip(tr("tooltip_shortcuts"))
        self.btn_cancel.setText(tr("btn_cancel"))
        self.btn_cancel.setToolTip(tr("tooltip_cancel"))
        self.lbl_device.setToolTip(tr("tooltip_device"))
        self.btn_kofi.setText(tr("btn_kofi"))
        self.btn_kofi.setToolTip(tr("tooltip_kofi"))
