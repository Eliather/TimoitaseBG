"""
Barra inferior de paginación, recorte temporal (Trim), FPS objetivo y rastreo para animaciones y videos.
"""
from typing import Optional

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from app.i18n import tr
from app.ui.icons import get_icon


class SequenceBar(QFrame):
    """HUD de navegación entre fotogramas para secuencias animadas y videos."""

    firstRequested = Signal()
    prevRequested = Signal()
    playToggled = Signal()
    nextRequested = Signal()
    lastRequested = Signal()
    trimRangeChanged = Signal(int, int)
    fpsChanged = Signal(float)
    trackSequenceRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gifPaginationBar")
        self.setFixedHeight(50)
        self._total_frames: int = 1
        self._setup_ui()
        self.apply_theme(is_dark=True, is_playing=False)

    def _setup_ui(self):
        bar_layout = QHBoxLayout(self)
        bar_layout.setContentsMargins(14, 4, 14, 4)
        bar_layout.setSpacing(6)

        self.btn_gif_first = self._make_nav_btn("Primer fotograma (Home)", self.firstRequested.emit)
        self.btn_gif_prev = self._make_nav_btn("Fotograma anterior (←)", self.prevRequested.emit)

        self.btn_gif_play = QPushButton("Play")
        self.btn_gif_play.setObjectName("gifPlayBtn")
        self.btn_gif_play.setFixedHeight(32)
        self.btn_gif_play.setMinimumWidth(78)
        self.btn_gif_play.setToolTip("Reproducir / Pausar animación (Espacio)")
        self.btn_gif_play.clicked.connect(self.playToggled.emit)

        self.btn_gif_next = self._make_nav_btn("Fotograma siguiente (→)", self.nextRequested.emit)
        self.btn_gif_last = self._make_nav_btn("Último fotograma (End)", self.lastRequested.emit)

        self.lbl_gif_pagination = QLabel("Fotograma 1 / 1")
        self.lbl_gif_pagination.setObjectName("gifPaginationLabel")
        self.lbl_gif_pagination.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Controles de rango temporal (Trim 1-based para UI)
        self.lbl_trim = QLabel(tr("seq_lbl_trim"))
        self.spin_trim_start = QSpinBox()
        self.spin_trim_start.setRange(1, 1)
        self.spin_trim_start.setFixedWidth(56)
        self.spin_trim_start.setToolTip(tr("seq_tip_trim_start"))
        self.spin_trim_start.valueChanged.connect(self._on_trim_changed)

        self.lbl_trim_sep = QLabel("–")
        self.spin_trim_end = QSpinBox()
        self.spin_trim_end.setRange(1, 1)
        self.spin_trim_end.setFixedWidth(56)
        self.spin_trim_end.setToolTip(tr("seq_tip_trim_end"))
        self.spin_trim_end.valueChanged.connect(self._on_trim_changed)

        # Selector de FPS objetivo de exportación
        self.lbl_fps = QLabel("FPS:")
        self.spin_target_fps = QDoubleSpinBox()
        self.spin_target_fps.setRange(1.0, 120.0)
        self.spin_target_fps.setDecimals(1)
        self.spin_target_fps.setSingleStep(1.0)
        self.spin_target_fps.setValue(24.0)
        self.spin_target_fps.setFixedWidth(68)
        self.spin_target_fps.setToolTip(tr("seq_tip_fps"))
        self.spin_target_fps.valueChanged.connect(lambda v: self.fpsChanged.emit(float(v)))

        # Botón de rastreo inteligente de sujeto en secuencia
        self.btn_track_sequence = QPushButton(tr("seq_btn_track"))
        self.btn_track_sequence.setObjectName("secondaryActionButton")
        self.btn_track_sequence.setFixedHeight(32)
        self.btn_track_sequence.setToolTip(tr("seq_tip_track"))
        self.btn_track_sequence.clicked.connect(self.trackSequenceRequested.emit)

        bar_layout.addStretch()
        bar_layout.addWidget(self.btn_gif_first)
        bar_layout.addWidget(self.btn_gif_prev)
        bar_layout.addWidget(self.btn_gif_play)
        bar_layout.addWidget(self.btn_gif_next)
        bar_layout.addWidget(self.btn_gif_last)
        bar_layout.addSpacing(8)
        bar_layout.addWidget(self.lbl_gif_pagination)
        bar_layout.addSpacing(12)
        bar_layout.addWidget(self.lbl_trim)
        bar_layout.addWidget(self.spin_trim_start)
        bar_layout.addWidget(self.lbl_trim_sep)
        bar_layout.addWidget(self.spin_trim_end)
        bar_layout.addSpacing(8)
        bar_layout.addWidget(self.lbl_fps)
        bar_layout.addWidget(self.spin_target_fps)
        bar_layout.addSpacing(8)
        bar_layout.addWidget(self.btn_track_sequence)
        bar_layout.addStretch()

    def _make_nav_btn(self, tooltip: str, slot) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("gifNavBtn")
        btn.setFixedSize(32, 32)
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        return btn

    def _on_trim_changed(self):
        s_1 = self.spin_trim_start.value()
        e_1 = self.spin_trim_end.value()
        if s_1 > e_1:
            if self.sender() is self.spin_trim_start:
                self.spin_trim_end.blockSignals(True)
                self.spin_trim_end.setValue(s_1)
                self.spin_trim_end.blockSignals(False)
                e_1 = s_1
            else:
                self.spin_trim_start.blockSignals(True)
                self.spin_trim_start.setValue(e_1)
                self.spin_trim_start.blockSignals(False)
                s_1 = e_1
        self.trimRangeChanged.emit(max(0, s_1 - 1), max(0, e_1 - 1))

    def configure_sequence(
        self,
        total_frames: int,
        fps: float,
        trim_start: int = 0,
        trim_end: Optional[int] = None,
    ):
        total = max(1, int(total_frames))
        self._total_frames = total
        end_0 = (total - 1) if trim_end is None else max(0, min(total - 1, int(trim_end)))
        start_0 = max(0, min(end_0, int(trim_start)))

        for sp in (self.spin_trim_start, self.spin_trim_end, self.spin_target_fps):
            sp.blockSignals(True)
        self.spin_trim_start.setRange(1, total)
        self.spin_trim_end.setRange(1, total)
        self.spin_trim_start.setValue(start_0 + 1)
        self.spin_trim_end.setValue(end_0 + 1)
        self.spin_target_fps.setValue(max(1.0, min(120.0, float(fps or 24.0))))
        for sp in (self.spin_trim_start, self.spin_trim_end, self.spin_target_fps):
            sp.blockSignals(False)

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

        self.btn_track_sequence.setIcon(get_icon("film", color=fg_color, active_color=accent_color, size=14))
        self.btn_track_sequence.setIconSize(QSize(14, 14))
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

    def retranslate_ui(self):
        self.lbl_trim.setText(tr("seq_lbl_trim"))
        self.spin_trim_start.setToolTip(tr("seq_tip_trim_start"))
        self.spin_trim_end.setToolTip(tr("seq_tip_trim_end"))
        self.spin_target_fps.setToolTip(tr("seq_tip_fps"))
        self.btn_track_sequence.setText(tr("seq_btn_track"))
        self.btn_track_sequence.setToolTip(tr("seq_tip_track"))
