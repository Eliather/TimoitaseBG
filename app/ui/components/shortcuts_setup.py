"""
Registro centralizado de atajos de teclado de MainWindow.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut


def register_shortcuts(window) -> list:
    """Registra todos los atajos globales sobre window y retorna la lista de QShortcut."""
    shortcuts = [
        # Ayuda y atajos
        QShortcut(QKeySequence("F1"), window, window.show_shortcuts_modal),
        QShortcut(QKeySequence("Ctrl+/"), window, window.show_shortcuts_modal),
        # Abrir / Guardar / Copiar / Pegar
        QShortcut(QKeySequence("Ctrl+O"), window, window.prompt_open_image),
        QShortcut(QKeySequence("Ctrl+S"), window, window.prompt_save_image),
        QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), window, window.copy_to_clipboard),
        QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), window, window.paste_from_clipboard),
        # Deshacer / Rehacer
        QShortcut(QKeySequence("Ctrl+Z"), window, window.perform_undo),
        QShortcut(QKeySequence("Ctrl+Y"), window, window.perform_redo),
        QShortcut(QKeySequence("Ctrl+Shift+Z"), window, window.perform_redo),
        # Ajuste de vista
        QShortcut(QKeySequence("Ctrl+0"), window, window.canvas.fit_to_view),
        QShortcut(QKeySequence("Ctrl+1"), window, window.canvas.reset_zoom_100),
        # Selección (mismos atajos que Photoshop)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), window, window.perform_delete_selection),
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), window, window.perform_delete_selection),
        QShortcut(QKeySequence("Ctrl+D"), window, window.canvas.clear_selection),
        QShortcut(QKeySequence(Qt.Key.Key_Escape), window, window.canvas.clear_selection),
        QShortcut(QKeySequence("Ctrl+A"), window, window.canvas.select_all),
        QShortcut(QKeySequence("Ctrl+Shift+I"), window, window.canvas.invert_selection),
        # Navegación y reproducción en secuencias GIF/Video
        QShortcut(QKeySequence(Qt.Key.Key_Space), window, window._toggle_gif_playback),
        QShortcut(
            QKeySequence(Qt.Key.Key_Left),
            window,
            lambda: window._navigate_gif_frame(
                window.sequence_session.current_index - 1 if window.sequence_session else 0
            ),
        ),
        QShortcut(
            QKeySequence(Qt.Key.Key_Right),
            window,
            lambda: window._navigate_gif_frame(
                window.sequence_session.current_index + 1 if window.sequence_session else 0
            ),
        ),
        QShortcut(QKeySequence(Qt.Key.Key_Home), window, lambda: window._navigate_gif_frame(0)),
        QShortcut(
            QKeySequence(Qt.Key.Key_End),
            window,
            lambda: window._navigate_gif_frame(
                window.sequence_session.total_frames - 1 if window.sequence_session else 0
            ),
        ),
    ]
    return shortcuts
