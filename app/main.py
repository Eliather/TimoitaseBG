"""
TimoitaseBG - Punto de entrada de la aplicación.
"""
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from app.config import APP_NAME, APP_DIR
from app.ui.main_window import MainWindow


def load_stylesheet() -> str:
    """Carga la hoja de estilos QSS desde el archivo correspondiente."""
    qss_path = APP_DIR / "ui" / "styles.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as f:
            content = f.read()
            assets_path = (APP_DIR / "ui" / "assets").as_posix()
            return content.replace("{ASSETS_PATH}", assets_path)
    return ""


def main():
    import multiprocessing
    multiprocessing.freeze_support()

    # Inicializar la aplicación Qt
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)

    # Cargar y aplicar tema QSS
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    # Crear y mostrar la ventana principal
    window = MainWindow()
    window.show()

    # Ejecutar bucle de eventos
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
