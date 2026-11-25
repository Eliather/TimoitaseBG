"""
TimoitaseBG - Widget de zona de arrastre (DropZone) para el estado vacío del canvas.
"""
import logging
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGraphicsDropShadowEffect,
)
from app.i18n import tr
from app.ui.icons import get_icon, get_pixmap

logger = logging.getLogger("TimoitaseBG.DropZone")


class DropZoneWidget(QFrame):
    """Área central limpia y arquitectónica para arrastrar imágenes o abrir archivos."""

    clicked = Signal()
    fileDropped = Signal(str)
    imageDropped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("dragging", False)
        self.setFixedSize(460, 290)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(28)
        self.shadow.setColor(QColor(0, 0, 0, 75))
        self.shadow.setOffset(0, 8)
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(32, 28, 32, 28)

        self.icon_lbl = QLabel()
        self.icon_lbl.setObjectName("DropZoneIcon")
        self.icon_lbl.setFixedSize(52, 52)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_lbl = QLabel(tr("canvas_drop_title"))
        self.title_lbl.setObjectName("DropZoneTitle")
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl.setWordWrap(True)

        self.subtitle_lbl = QLabel(tr("canvas_drop_subtitle"))
        self.subtitle_lbl.setObjectName("DropZoneSubtitle")
        self.subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_lbl.setWordWrap(True)

        self.btn_open_canvas = QPushButton(tr("btn_open"))
        self.btn_open_canvas.setObjectName("PrimaryBtn")
        self.btn_open_canvas.setFixedHeight(38)
        self.btn_open_canvas.setMinimumWidth(148)
        self.btn_open_canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_canvas.clicked.connect(self.clicked.emit)

        self.formats_lbl = QLabel(tr("canvas_drop_formats"))
        self.formats_lbl.setObjectName("DropZoneFormats")
        self.formats_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formats_lbl.setWordWrap(True)

        layout.addWidget(self.icon_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(4)
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.subtitle_lbl)
        layout.addSpacing(8)
        layout.addWidget(self.btn_open_canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(4)
        layout.addWidget(self.formats_lbl)

        self.update_theme(is_dark=True)

    def retranslate_ui(self):
        """Actualiza los textos al cambiar de idioma."""
        self.title_lbl.setText(tr("canvas_drop_title"))
        self.subtitle_lbl.setText(tr("canvas_drop_subtitle"))
        self.btn_open_canvas.setText(tr("btn_open"))
        self.formats_lbl.setText(tr("canvas_drop_formats"))

    def update_theme(self, is_dark: bool):
        """Ajusta la sombra e iconos vectoriales para el tema activo."""
        accent = "#2E7CF6" if is_dark else "#1D63DC"
        self.icon_lbl.setPixmap(get_pixmap("image", size=24, color=accent))
        self.btn_open_canvas.setIcon(get_icon("folder-open", color="#FFFFFF", size=16))
        self.btn_open_canvas.setIconSize(QSize(16, 16))
        if is_dark:
            self.shadow.setColor(QColor(0, 0, 0, 80))
        else:
            self.shadow.setColor(QColor(20, 23, 26, 18))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def _set_dragging(self, active: bool) -> None:
        if bool(self.property("dragging")) != active:
            self.setProperty("dragging", active)
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

    @staticmethod
    def _accept_media_drag(event) -> bool:
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def dragEnterEvent(self, event):
        if self._accept_media_drag(event):
            self._set_dragging(True)

    def dragMoveEvent(self, event):
        self._accept_media_drag(event)

    def dragLeaveEvent(self, event):
        self._set_dragging(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._set_dragging(False)
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls:
                url = urls[0]
                if url.isLocalFile() and not url.host():
                    file_path = url.toLocalFile()
                    if file_path and not str(file_path).startswith(("\\\\", "//")):
                        self.fileDropped.emit(file_path)
                        event.acceptProposedAction()
                        return
                logger.warning("URL remota o ruta UNC rechazada en DropZone.")
                event.ignore()
                return
        if mime.hasImage():
            from PySide6.QtGui import QImage
            from app.core.image_utils import qimage_to_pil

            img_data = mime.imageData()
            qimg = img_data if isinstance(img_data, QImage) else QImage(img_data)
            if not qimg.isNull():
                self.imageDropped.emit(qimage_to_pil(qimg))
                event.acceptProposedAction()
                return
        event.ignore()
