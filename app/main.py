"""
TimoitaseBG - Punto de entrada de la aplicación.
"""
import sys
import multiprocessing
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path (soporte tanto para desarrollo como ejecutables)
if getattr(sys, "frozen", False):
    bundle_dir = getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def load_stylesheet() -> str:
    """Carga la hoja de estilos QSS desde el archivo correspondiente."""
    from app.config import APP_DIR
    for qss_name in ("styles.qss", "style.qss", "style_light.qss"):
        qss_path = APP_DIR / "ui" / qss_name
        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                content = f.read()
                assets_path = (APP_DIR / "ui" / "assets").as_posix()
                return content.replace("{ASSETS_PATH}", assets_path)
    return ""


def main():
    """Inicializa el entorno gráfico y ejecuta la ventana principal."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from app.config import APP_NAME, APP_VERSION, APP_AUTHOR
    from app.ui.main_window import MainWindow

    # Inicializar la aplicación Qt
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_AUTHOR)

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
    # 1. Blindaje absoluto de multiprocessing en modo congelado (PyInstaller):
    # DEBE ser la primera instrucción absoluta antes de set_start_method("spawn")
    # y antes de importar o inicializar QApplication. De este modo, los subprocesos
    # de GPUWorkerManager ejecutan su función objetivo y terminan con sys.exit()
    # sin inicializar la interfaz gráfica ni abrir una segunda ventana.
    multiprocessing.freeze_support()

    # 2. Forzar 'spawn' en Linux y macOS para evitar deadlocks entre ONNX Runtime, Qt y hilos nativos
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 3. Iniciar la aplicación
    main()

