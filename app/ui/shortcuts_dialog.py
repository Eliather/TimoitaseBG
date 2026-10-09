"""
TimoitaseBG - Diálogo Modal de Atajos de Teclado y Gestos
Diseño moderno sin marco con soporte para temas claro y oscuro.
"""
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGridLayout
)
from app.i18n import tr
from app.ui.icons import get_icon


class ShortcutsDialog(QDialog):
    """Modal moderno para visualizar todos los atajos de teclado y gestos."""

    def __init__(self, parent=None, is_dark_theme: bool = True):
        super().__init__(parent)
        self.is_dark_theme = is_dark_theme
        self._is_tracking = False
        self._start_pos = QPoint()

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        self.resize(640, 460)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.root_widget = QWidget(self)
        self.root_widget.setObjectName("shortcutsDialogRoot")
        main_layout.addWidget(self.root_widget)

        root_layout = QVBoxLayout(self.root_widget)
        root_layout.setContentsMargins(22, 18, 22, 18)
        root_layout.setSpacing(14)

        # 1. Cabecera (Draggable)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # Icono de teclado
        icon_color = "#FF4F79"
        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_icon("keyboard", color=icon_color, size=18).pixmap(18, 18))
        header_layout.addWidget(lbl_icon)

        title_lbl = QLabel(tr("shortcuts_modal_title"))
        title_lbl.setObjectName("shortcutsDialogTitle")
        header_layout.addWidget(title_lbl)

        header_layout.addStretch()

        self.btn_close = QPushButton("")
        self.btn_close.setObjectName("macCloseBtn")
        self.btn_close.setFixedSize(14, 14)
        self.btn_close.setToolTip("Cerrar")
        self.btn_close.clicked.connect(self.accept)
        header_layout.addWidget(self.btn_close)

        root_layout.addWidget(header)

        # Separador superior
        sep_top = QFrame()
        sep_top.setObjectName("panelSeparator")
        sep_top.setFrameShape(QFrame.Shape.HLine)
        root_layout.addWidget(sep_top)

        # 2. Contenido en 2 columnas para visualización inmediata sin scroll
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(28)
        columns_layout.setContentsMargins(0, 4, 0, 4)

        col1_layout = QVBoxLayout()
        col1_layout.setSpacing(14)

        col2_layout = QVBoxLayout()
        col2_layout.setSpacing(14)

        categories_col1 = [
            (
                tr("shortcuts_cat_nav"),
                [
                    (tr("shortcut_zoom"), tr("shortcut_zoom_desc")),
                    (tr("shortcut_pan"), tr("shortcut_pan_desc")),
                    (tr("shortcut_fit"), tr("shortcut_fit_desc")),
                    (tr("shortcut_100"), tr("shortcut_100_desc")),
                ]
            ),
            (
                tr("shortcuts_cat_selection"),
                [
                    (tr("shortcut_del"), tr("shortcut_del_desc")),
                    (tr("shortcut_deselect"), tr("shortcut_deselect_desc")),
                    (tr("shortcut_select_all"), tr("shortcut_select_all_desc")),
                    (tr("shortcut_invert"), tr("shortcut_invert_desc")),
                ]
            ),
        ]

        categories_col2 = [
            (
                tr("shortcuts_cat_edit"),
                [
                    (tr("shortcut_open"), tr("shortcut_open_desc")),
                    (tr("shortcut_save"), tr("shortcut_save_desc")),
                    (tr("shortcut_paste"), tr("shortcut_paste_desc")),
                    (tr("shortcut_undo"), tr("shortcut_undo_desc")),
                    (tr("shortcut_redo"), tr("shortcut_redo_desc")),
                ]
            ),
        ]

        def _build_category(cat_title, items):
            cat_box = QWidget()
            cat_layout = QVBoxLayout(cat_box)
            cat_layout.setContentsMargins(0, 0, 0, 0)
            cat_layout.setSpacing(6)

            cat_lbl = QLabel(cat_title)
            cat_lbl.setObjectName("shortcutsCategoryTitle")
            cat_layout.addWidget(cat_lbl)

            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(5)
            grid.setContentsMargins(2, 2, 2, 2)

            for row_idx, (badge_text, desc_text) in enumerate(items):
                b_lbl = QLabel(badge_text)
                b_lbl.setObjectName("shortcutBadge")
                b_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

                d_lbl = QLabel(desc_text)
                d_lbl.setObjectName("shortcutDesc")

                grid.addWidget(b_lbl, row_idx, 0, Qt.AlignmentFlag.AlignLeft)
                grid.addWidget(d_lbl, row_idx, 1, Qt.AlignmentFlag.AlignLeft)

            grid.setColumnStretch(1, 1)
            cat_layout.addLayout(grid)
            return cat_box

        for cat_title, items in categories_col1:
            col1_layout.addWidget(_build_category(cat_title, items))
        col1_layout.addStretch()

        for cat_title, items in categories_col2:
            col2_layout.addWidget(_build_category(cat_title, items))
        col2_layout.addStretch()

        columns_layout.addLayout(col1_layout, stretch=1)

        # Divisor vertical entre columnas
        v_sep = QFrame()
        v_sep.setObjectName("panelSeparator")
        v_sep.setFrameShape(QFrame.Shape.VLine)
        columns_layout.addWidget(v_sep)

        columns_layout.addLayout(col2_layout, stretch=1)

        root_layout.addLayout(columns_layout, stretch=1)

        # 3. Pie con botón cerrar
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_ok = QPushButton(tr("btn_close_shortcuts"))
        self.btn_ok.setObjectName("primaryActionButton")
        self.btn_ok.setFixedSize(110, 32)
        self.btn_ok.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_ok)
        root_layout.addLayout(btn_layout)

    # Soporte para arrastrar la ventana
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = True
            self._start_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_tracking:
            self.move(event.globalPosition().toPoint() - self._start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_tracking = False
            event.accept()
