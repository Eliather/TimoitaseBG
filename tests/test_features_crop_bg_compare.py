"""
Tests para Modo Comparación, Caché QPixmap/HiDPI del Canvas, SelectionTool, BrushEngine y seguridad UNC.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
from PySide6.QtGui import QDropEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from app.ui import media_actions
from app.ui.canvas_widget import CanvasWidget
from app.ui.canvas.renderer import build_hidpi_checkerboard
from app.ui.drop_zone import DropZoneWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_set_comparison_mode_with_pil_image_and_paint(qapp):
    """set_comparison_mode(True, pil_img) no debe lanzar NameError y debe renderizar la cortina."""
    canvas = CanvasWidget()
    canvas.resize(400, 300)

    before_pil = Image.new("RGB", (120, 80), (40, 80, 160))
    after_pil = Image.new("RGBA", (120, 80), (200, 50, 90, 200))

    canvas.set_image(after_pil, reset_zoom=True)
    canvas.set_comparison_mode(True, before_pil)

    assert canvas.comparison_mode is True
    assert canvas.before_qimage is not None
    assert canvas.before_qimage.width() == 120
    assert canvas.before_qimage.height() == 80
    assert canvas.before_qimage.hasAlphaChannel()

    # Renderizar realmente a un QPixmap para verificar que draw_canvas_scene + divider funcionan
    target = QPixmap(400, 300)
    canvas.render(target)
    assert not target.isNull()
    assert canvas._cached_before_pixmap is not None

    canvas.set_comparison_mode(False)
    assert canvas.comparison_mode is False
    assert canvas.before_qimage is None


def test_canvas_pixmap_cache_reused_during_marching_ants_and_invalidated_on_edit(qapp):
    """El QPixmap cacheado se reutiliza entre repintados de marching ants y se invalida al editar."""
    canvas = CanvasWidget()
    canvas.resize(200, 200)
    img = Image.new("RGBA", (400, 400), (80, 120, 200, 255))
    canvas.set_image(img, reset_zoom=True)

    target = QPixmap(200, 200)
    canvas.render(target)
    first_pix = canvas._cached_pixmap
    assert first_pix is not None

    # Simular tick de marching ants sin cambiar imagen ni zoom
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    canvas._set_selection(mask)
    canvas._advance_marching_ants()
    canvas.render(target)
    assert canvas._cached_pixmap is first_pix

    # Al borrar selección, la caché debe invalidarse y regenerarse
    assert canvas.delete_selection() is True
    assert canvas._cached_pixmap is None
    canvas.render(target)
    assert canvas._cached_pixmap is not None
    assert canvas._cached_pixmap is not first_pix
    canvas.clear_selection()


def test_hidpi_checkerboard_sets_device_pixel_ratio(qapp):
    """build_hidpi_checkerboard escala celdas físicas y aplica devicePixelRatio."""
    pix = build_hidpi_checkerboard(size=16, dpr=1.5, is_dark=True)
    assert pix.devicePixelRatio() == pytest.approx(1.5)
    assert pix.width() == 48
    assert pix.height() == 48


def test_selection_delete_and_brush_clip_with_selection(qapp):
    """SelectionTool.delete_selected_pixels y BrushEngine con selección funcionan sin errores."""
    canvas = CanvasWidget()
    img = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
    canvas.set_image(img)
    canvas.set_original_image(img)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    canvas._set_selection(mask)

    assert canvas._selection_tool.contour_path is not None
    assert not canvas._selection_tool.contour_path.isEmpty()

    assert canvas.delete_selection() is True
    assert canvas.current_qimage.pixelColor(20, 20).alpha() == 0
    assert canvas.current_qimage.pixelColor(2, 2).alpha() == 255

    # Pincel restaurador recortado por la selección activa
    ok = canvas._brush_engine.draw_stroke(
        target_qimage=canvas.current_qimage,
        original_qimage=canvas.original_qimage,
        start_pt=QPointF(20, 20),
        end_pt=QPointF(22, 22),
        brush_size=16,
        brush_opacity=1.0,
        brush_hardness=0.5,
        eraser_mode=False,
        active_selection=mask,
    )
    assert ok is True
    assert canvas.current_qimage.pixelColor(20, 20).alpha() > 0


def test_unc_paths_rejected_in_drop_zone_canvas_and_clipboard(qapp):
    """Rutas UNC (\\\\server\\share) y URLs con host deben rechazarse en DropZone, Canvas y Portapapeles."""
    unc_url = QUrl.fromLocalFile(r"\\evil-server\share\leak.png")
    mime = QMimeData()
    mime.setUrls([unc_url])

    ev = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    dz = DropZoneWidget()
    dz_files = []
    dz.fileDropped.connect(dz_files.append)
    dz.dropEvent(ev)
    assert dz_files == []

    canvas = CanvasWidget()
    cv_files = []
    canvas.fileDropped.connect(cv_files.append)
    canvas.dropEvent(ev)
    assert cv_files == []

    fake_win = MagicMock()
    fake_clip = MagicMock()
    fake_clip.mimeData.return_value = mime
    with patch("app.ui.media_actions.QGuiApplication.clipboard", return_value=fake_clip):
        media_actions.paste_from_clipboard(fake_win)
    fake_win.open_image_file.assert_not_called()


def test_bg_panel_last_custom_color_swatch_and_qsettings_persistence(qapp, tmp_path):
    """El swatch de último color guardado funciona, persiste en QSettings y renderiza en ambos temas."""
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QColor
    from app.ui.panels.bg_panel import BackgroundPanel, _LAST_CUSTOM_COLOR_KEY

    ini_path = str(tmp_path / "test_settings.ini")

    def _make_settings(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    with patch("app.ui.panels.bg_panel.QSettings", side_effect=_make_settings):
        panel = BackgroundPanel()
        assert panel.btn_swatch_last_color.objectName() == "swatchLastColorButton"
        assert panel.btn_swatch_last_color.isCheckable() is True
        assert panel.btn_swatch_last_color.color().name().upper() == "#2E7CF6"
        assert "#2E7CF6" in panel.btn_swatch_last_color.toolTip()

        # Orden exacto de los 6 swatches
        assert panel._swatch_buttons == (
            panel.btn_swatch_none,
            panel.btn_swatch_white,
            panel.btn_swatch_black,
            panel.btn_swatch_gray,
            panel.btn_swatch_last_color,
            panel.btn_swatch_custom,
        )

        # Habilitar transparencia habilita también btn_swatch_last_color
        panel.set_transparency_available(True, True)
        assert panel.btn_swatch_last_color.isEnabled() is True

        emitted = []
        panel.applyBgColorRequested.connect(emitted.append)

        # Clic directo en el último color guardado aplica su RGB
        panel.btn_swatch_last_color.click()
        assert emitted == [(46, 124, 246)]
        assert panel.btn_swatch_last_color.isChecked() is True

        # Elegir un color nuevo desde la paleta actualiza swatch, tooltip y QSettings
        with patch(
            "app.ui.panels.bg_panel.QColorDialog.getColor",
            return_value=QColor("#E11D48"),
        ):
            panel.btn_swatch_custom.click()

        assert emitted[-1] == (225, 29, 72)
        assert panel.btn_swatch_last_color.color().name().upper() == "#E11D48"
        assert "#E11D48" in panel.btn_swatch_last_color.toolTip()
        assert panel.btn_swatch_last_color.isChecked() is True
        assert _make_settings().value(_LAST_CUSTOM_COLOR_KEY) == "#E11D48"

        # Una nueva instancia debe restaurar el color persistido en QSettings
        panel2 = BackgroundPanel()
        assert panel2.btn_swatch_last_color.color().name().upper() == "#E11D48"
        assert "#E11D48" in panel2.btn_swatch_last_color.toolTip()

        # Renderizado antialiasado en tema oscuro y claro sin errores
        for is_dark in (True, False):
            panel2.apply_theme(is_dark)
            for btn in panel2._swatch_buttons:
                pix = QPixmap(btn.size())
                btn.render(pix)
                assert not pix.isNull()
