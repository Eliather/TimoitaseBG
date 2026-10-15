"""
TimoitaseBG - Widget de zona de arrastre (DropZone) para el estado vacío del canvas.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QLabel,
    QGraphicsDropShadowEffect,
)
from app.i18n import tr


class DropZoneWidget(QFrame):
    """Área central estilizada para arrastrar y soltar imágenes o hacer clic para abrir."""
    clicked = Signal()
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setFixedSize(460, 270)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Sombra suave
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(25)
        self.shadow.setColor(QColor(15, 23, 42, 18))
        self.shadow.setOffset(0, 8)
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Icono circular
        self.icon_lbl = QLabel("🖼️")
        self.icon_lbl.setObjectName("DropZoneIcon")
        self.icon_lbl.setFixedSize(54, 54)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_lbl = QLabel(tr("canvas_drop_title"))
        self.title_lbl.setObjectName("DropZoneTitle")
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.subtitle_lbl = QLabel(tr("canvas_drop_subtitle"))
        self.subtitle_lbl.setObjectName("DropZoneSubtitle")
        self.subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.formats_lbl = QLabel(tr("canvas_drop_formats"))
        self.formats_lbl.setObjectName("DropZoneFormats")
        self.formats_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.subtitle_lbl)
        layout.addSpacing(4)
        layout.addWidget(self.formats_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

    def retranslate_ui(self):
        """Actualiza los textos al cambiar de idioma."""
        self.title_lbl.setText(tr("canvas_drop_title"))
        self.subtitle_lbl.setText(tr("canvas_drop_subtitle"))
        self.formats_lbl.setText(tr("canvas_drop_formats"))

    def update_theme(self, is_dark: bool):
        """Ajusta la sombra y estilos para el tema activo."""
        if is_dark:
            self.shadow.setColor(QColor(0, 0, 0, 90))
        else:
            self.shadow.setColor(QColor(15, 23, 42, 18))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.fileDropped.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()
