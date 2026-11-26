"""
Configuración global de pytest para ciclo de vida determinista de Qt6/PySide6 en modo headless.

Mantiene viva una única instancia de QApplication durante toda la sesión para evitar que
Shiboken destruya o recree el singleton C++ entre módulos, y drena widgets, portapapeles,
cachés globales de QPixmap/QSvgRenderer y eventos DeferredDelete antes de Py_FinalizeEx.
"""
import atexit
import gc
import sys
from typing import Optional

import pytest
from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

_GLOBAL_QAPP: Optional[QApplication] = None


def get_or_create_qapp() -> QApplication:
    """Devuelve el QApplication único de sesión reteniendo una referencia global fuerte."""
    global _GLOBAL_QAPP
    inst = QApplication.instance()
    if inst is None:
        inst = QApplication([])
    _GLOBAL_QAPP = inst  # type: ignore[assignment]
    return _GLOBAL_QAPP


def _cleanup_qt_state(final_teardown: bool = False) -> None:
    """Limpia portapapeles, widgets de nivel superior, callbacks y eventos diferidos de Qt."""
    global _GLOBAL_QAPP
    app = QApplication.instance()
    if app is None:
        return

    # Garantizar que el wrapper Python jamás quede huérfano si algún módulo lo instanció directo
    _GLOBAL_QAPP = app  # type: ignore[assignment]

    # 1. Limpiar portapapeles (libera QMimeData / QImage retenidos en el plugin offscreen)
    try:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.clear()
    except Exception:
        pass

    # 2. Cerrar, desinstalar filtros, liberar cachés QPixmap y programar deleteLater en topLevelWidgets
    try:
        from app.ui.canvas_widget import CanvasWidget
    except Exception:
        CanvasWidget = None  # type: ignore[assignment]

    for widget in list(app.topLevelWidgets()):
        try:
            for timer in widget.findChildren(QTimer):
                if timer.isActive():
                    timer.stop()
        except RuntimeError:
            continue

        container = getattr(widget, "canvas_container", None)
        if container is not None:
            try:
                container.removeEventFilter(widget)
            except RuntimeError:
                pass

        if CanvasWidget is not None:
            try:
                canvases = [widget] if isinstance(widget, CanvasWidget) else []
                canvases.extend(widget.findChildren(CanvasWidget))
                for cv in canvases:
                    cv.release_qt_resources()
            except RuntimeError:
                pass

        try:
            widget.hide()
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass

    # 3. Limpiar callback global de estado en GPUWorkerManager
    if "app.core.gpu_manager" in sys.modules:
        try:
            from app.core.gpu_manager import get_gpu_manager

            gpu_mgr = get_gpu_manager()
            gpu_mgr.set_status_callback(None)
            if final_teardown:
                gpu_mgr.shutdown(blocking=True)
        except Exception:
            pass

    # 4. En el cierre final de sesión, vaciar cachés globales de QPixmap / QSvgRenderer y QSS
    if final_teardown:
        if "app.ui.icons" in sys.modules:
            try:
                from app.ui.icons import clear_icon_cache

                clear_icon_cache()
            except Exception:
                pass
        try:
            app.setStyleSheet("")
        except Exception:
            pass

    # 5. Drenar DeferredDelete y recolectar basura mientras QApplication sigue viva
    try:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        gc.collect()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
    except Exception:
        pass


atexit.register(_cleanup_qt_state, True)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return get_or_create_qapp()


@pytest.fixture(autouse=True)
def _qt_test_teardown():
    yield
    _cleanup_qt_state(final_teardown=False)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _cleanup_qt_state(final_teardown=True)
