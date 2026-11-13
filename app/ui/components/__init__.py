"""
Componentes modulares de barra superior, diálogos sin marco y controles de secuencia.
"""
from app.ui.components.top_bar import CustomTopBar, FramelessPopup, TopBar
from app.ui.components.sequence_bar import SequenceBar
from app.ui.components.status_bar import StatusBar
from app.ui.components.shortcuts_setup import register_shortcuts

__all__ = [
    "CustomTopBar",
    "FramelessPopup",
    "TopBar",
    "SequenceBar",
    "StatusBar",
    "register_shortcuts",
]
