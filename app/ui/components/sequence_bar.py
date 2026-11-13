"""
Barra inferior de paginación y reproducción para animaciones GIF/APNG/WebP y videos.
"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from app.ui.icons import get_icon


class SequenceBar(QFrame):
    """HUD de navegación entre fotogramas para secuencias animadas y videos."""

    firstRequested = Signal()
    prevRequested = Signal()
    playToggled = Signal()
    nextRequested = Signal()
    lastRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gifPaginationBar")
        self.setFixedHeight(46)
        self._setup_ui()
        self.apply_theme(is_dark=True, is_playing=False)

    def _setup_ui(self):
        bar_layout = QHBoxLayout(self)
        bar_layout.setContentsMargins(16, 4, 16, 4)
        bar_layout.setSpacing(8)

        self.btn_gif_first = self._make_nav_btn("Primer fotograma (Home)", self.firstRequested.emit)
        self.btn_gif_prev = self._make_nav_btn("Fotograma anterior (←)", self.prevRequested.emit)

        self.btn_gif_play = QPushButton("Play")
        self.btn_gif_play.setObjectName("gifPlayBtn")
        self.btn_gif_play.setFixedHeight(32)
        self.btn_gif_play.setMinimumWidth(82)
        self.btn_gif_play.setToolTip("Reproducir / Pausar animación (Espacio)")
        self.btn_gif_play.clicked.connect(self.playToggled.emit)

        self.btn_gif_next = self._make_nav_btn("Fotograma siguiente (→)", self.nextRequested.emit)
        self.btn_gif_last = self._make_nav_btn("Último fotograma (End)", self.lastRequested.emit)

        self.lbl_gif_pagination = QLabel("Fotograma 1 / 1")
        self.lbl_gif_pagination.setObjectName("gifPaginationLabel")
        self.lbl_gif_pagination.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bar_layout.addStretch()
        bar_layout.addWidget(self.btn_gif_first)
        bar_layout.addWidget(self.btn_gif_prev)
        bar_layout.addWidget(self.btn_gif_play)
        bar_layout.addWidget(self.btn_gif_next)
        bar_layout.addWidget(self.btn_gif_last)
        bar_layout.addSpacing(12)
        bar_layout.addWidget(self.lbl_gif_pagination)
        bar_layout.addStretch()

    def _make_nav_btn(self, tooltip: str, slot) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("gifNavBtn")
        btn.setFixedSize(32, 32)
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        return btn

    def apply_theme(self, is_dark: bool, is_playing: bool = False):
        fg_color = "#F8FAFC" if is_dark else "#0F172A"
        accent_color = "#38BDF8" if is_dark else "#0284C7"
        dis_color = "#475569" if is_dark else "#CBD5E1"

        for btn, icon_name, sz in (
            (self.btn_gif_first, "skip-back", 15),
            (self.btn_gif_prev, "chevron-left", 16),
            (self.btn_gif_next, "chevron-right", 16),
            (self.btn_gif_last, "skip-forward", 15),
        ):
            btn.setIcon(get_icon(icon_name, color=fg_color, active_color=accent_color, disabled_color=dis_color, size=sz))
            btn.setIconSize(QSize(sz, sz))

        self.set_playing(is_playing)

    def set_playing(self, is_playing: bool):
        self.btn_gif_play.setText("Pausa" if is_playing else "Play")
        self.btn_gif_play.setIcon(
            get_icon("pause" if is_playing else "play", color="#FFFFFF", active_color="#FFFFFF", size=14)
        )
        self.btn_gif_play.setIconSize(QSize(14, 14))

    def update_frame_info(self, idx: int, total: int, duration_ms: int):
        self.lbl_gif_pagination.setText(f"Fotograma {idx + 1} / {total} ({duration_ms} ms)")
        self.btn_gif_first.setEnabled(idx > 0)
        self.btn_gif_prev.setEnabled(idx > 0)
        self.btn_gif_next.setEnabled(idx < total - 1)
        self.btn_gif_last.setEnabled(idx < total - 1)
