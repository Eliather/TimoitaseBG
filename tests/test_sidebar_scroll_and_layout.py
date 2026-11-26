"""
Tests de layout del sidebar, scroll, cambio de tema global, DropZone limpio y limpieza de secuencia.
"""
from PIL import Image

from app.core.image_state import SequenceSession
from app.ui.main_window import MainWindow
from app.ui.toolbar import ToolSidebar


def test_sidebar_scroll_and_tab_switching(qapp):
    """ToolSidebar debe cambiar entre todas las pestañas y mantener el área de scroll operativa."""
    sidebar = ToolSidebar()
    sidebar.resize(380, 620)
    sidebar.show()

    assert sidebar.scroll_area.widget() is sidebar.scroll_content
    assert sidebar.stack.count() == 5

    for idx, btn in enumerate((
        sidebar.btn_rail_bg,
        sidebar.btn_rail_restore,
        sidebar.btn_rail_brush,
        sidebar.btn_rail_batch,
        sidebar.btn_rail_effects,
    )):
        btn.click()
        assert sidebar.stack.currentIndex() == idx
        assert sidebar.stack.currentWidget().isVisible()

    # Verificar temas claro/oscuro sin excepciones
    sidebar.update_theme_icons(is_dark=False)
    sidebar.update_theme_icons(is_dark=True)
    sidebar.close()


def test_main_window_theme_applies_to_qapplication_and_load_pil_clears_sequence(qapp):
    """Cambiar tema aplica QSS a QApplication.instance() y load_new_pil_image limpia secuencias."""
    win = MainWindow()
    win.show()

    # Verificar DropZone limpio con botón primario de abrir
    assert hasattr(win.canvas.drop_zone, "btn_open_canvas")
    assert win.canvas.drop_zone.btn_open_canvas.objectName() == "PrimaryBtn"

    # Toggle tema claro -> oscuro debe actualizar QApplication stylesheet
    win._toggle_theme()
    assert win.is_dark_theme is False
    assert "#F6F7F8" in qapp.styleSheet()

    win._toggle_theme()
    assert win.is_dark_theme is True
    assert "#0E1012" in qapp.styleSheet()

    # Cargar una secuencia y luego una imagen PIL estática debe detener gif_timer y limpiar secuencia
    f1 = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    f2 = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
    win.doc.sequence_session = SequenceSession([f1, f2], [100, 100])
    win.gif_timer.start(100)
    assert win.gif_timer.isActive() is True

    static_img = Image.new("RGB", (48, 48), (10, 20, 30))
    win.load_new_pil_image(static_img, "test_static")
    assert win.gif_timer.isActive() is False
    assert win.sequence_session is None
    assert win.image_state.has_image is True

    win.close()
