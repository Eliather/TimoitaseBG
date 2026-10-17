"""
Tests unitarios para:
1. Copiar al Portapapeles (Ctrl+C) con soporte universal de transparencia (PNG stream + QImage).
2. Exportación e Importación de .WEBP Animado y .APNG con canal alfa de 8 bits continuo.
3. Carga de animaciones multiformato en MainWindow activando SequenceSession.
"""
import io
from pathlib import Path
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QMimeData

from app.core.gif_processor import (
    is_animated_image,
    extract_gif_frames,
    save_animated_gif,
    save_animated_webp,
    save_animated_apng,
)
from app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_is_animated_image(tmp_path: Path):
    """Verifica detección precisa de archivos animados (.gif, .webp, .png/.apng) vs estáticos."""
    f1 = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    f2 = Image.new("RGBA", (32, 32), (0, 255, 0, 255))

    # 1. Imagen estática PNG
    static_png = tmp_path / "static.png"
    f1.save(static_png, format="PNG")
    assert is_animated_image(static_png) is False

    # 2. Imagen estática JPG
    static_jpg = tmp_path / "static.jpg"
    f1.convert("RGB").save(static_jpg, format="JPEG")
    assert is_animated_image(static_jpg) is False

    # 3. GIF animado (2 fotogramas)
    anim_gif = tmp_path / "anim.gif"
    save_animated_gif([f1, f2], anim_gif, durations=[100, 100], loop=0)
    assert is_animated_image(anim_gif) is True

    # 4. WebP animado (2 fotogramas)
    anim_webp = tmp_path / "anim.webp"
    save_animated_webp([f1, f2], anim_webp, durations=[100, 100], loop=0)
    assert is_animated_image(anim_webp) is True

    # 5. APNG animado (2 fotogramas)
    anim_apng = tmp_path / "anim.png"
    save_animated_apng([f1, f2], anim_apng, durations=[100, 100], loop=0)
    assert is_animated_image(anim_apng) is True

    # 6. Archivo inexistente o inválido
    assert is_animated_image(tmp_path / "non_existent.webp") is False


def test_save_animated_webp_preserves_8bit_alpha(tmp_path: Path):
    """
    Verifica que save_animated_webp exporta un WebP animado válido preservando
    los 16.7M de colores y el canal alfa suave continuo de 8 bits (sin cuantizar a 1-bit como GIF).
    """
    f1 = Image.new("RGBA", (40, 40), (255, 50, 50, 128))  # Semitransparente 128
    f2 = Image.new("RGBA", (40, 40), (50, 255, 50, 64))   # Semitransparente 64

    output_webp = tmp_path / "output.webp"
    save_animated_webp([f1, f2], output_webp, durations=[120, 150], loop=0, quality=95)

    assert output_webp.exists()

    with Image.open(output_webp) as im:
        assert getattr(im, "is_animated", False) is True
        assert getattr(im, "n_frames", 1) == 2

        # Frame 0
        im.seek(0)
        p0 = im.convert("RGBA").getpixel((10, 10))
        assert p0[3] == 128  # Alfa suave preservado exactamente

        # Frame 1
        im.seek(1)
        p1 = im.convert("RGBA").getpixel((10, 10))
        assert p1[3] == 64   # Alfa suave preservado exactamente


def test_save_animated_apng_preserves_8bit_alpha(tmp_path: Path):
    """
    Verifica que save_animated_apng exporta un APNG animado válido preservando
    el canal alfa de 8 bits sin pérdida.
    """
    f1 = Image.new("RGBA", (40, 40), (200, 100, 50, 180))
    f2 = Image.new("RGBA", (40, 40), (50, 100, 200, 90))

    output_apng = tmp_path / "output.apng"
    save_animated_apng([f1, f2], output_apng, durations=[100, 120], loop=0)

    assert output_apng.exists()

    with Image.open(output_apng) as im:
        assert getattr(im, "is_animated", False) is True
        assert getattr(im, "n_frames", 1) == 2

        im.seek(0)
        p0 = im.convert("RGBA").getpixel((10, 10))
        assert p0[3] == 180

        im.seek(1)
        p1 = im.convert("RGBA").getpixel((10, 10))
        assert p1[3] == 90


def test_extract_gif_frames_all_formats(tmp_path: Path):
    """Verifica que extract_gif_frames extrae correctamente frames de GIF, WebP y APNG."""
    f1 = Image.new("RGBA", (30, 30), (255, 0, 0, 200))
    f2 = Image.new("RGBA", (30, 30), (0, 255, 0, 100))

    for fmt, ext, save_fn in [
        ("GIF", ".gif", save_animated_gif),
        ("WEBP", ".webp", save_animated_webp),
        ("APNG", ".png", save_animated_apng),
    ]:
        anim_path = tmp_path / f"test_{fmt.lower()}{ext}"
        save_fn([f1, f2], anim_path, durations=[110, 130], loop=0)

        frames, durs, loop = extract_gif_frames(anim_path)
        assert len(frames) == 2, f"Fallo al extraer frames de {fmt}"
        assert len(durs) == 2, f"Fallo al extraer duraciones de {fmt}"
        assert loop == 0


def test_copy_to_clipboard_transparency(qapp):
    """
    Verifica que copy_to_clipboard() inyecta en el portapapeles:
    - Stream binario 'image/png' con canal alfa intacto
    - MIME 'application/x-qt-image'
    - QImage con soporte alfa
    """
    window = MainWindow()
    test_img = Image.new("RGBA", (64, 64), (255, 128, 0, 175))
    window.load_new_pil_image(test_img, source_name="test_clipboard.png")

    window.copy_to_clipboard()

    clipboard = QGuiApplication.clipboard()
    mime = clipboard.mimeData()

    assert mime.hasImage() is True
    assert mime.hasFormat("image/png") is True
    assert mime.hasFormat("application/x-qt-image") is True

    # Comprobar que los bytes recuperados de 'image/png' conservan el alfa real (175)
    png_bytes = bytes(mime.data("image/png"))
    assert len(png_bytes) > 0

    with Image.open(io.BytesIO(png_bytes)) as restored:
        assert restored.format == "PNG"
        assert restored.mode == "RGBA"
        p = restored.getpixel((10, 10))
        assert p[0] == 255
        assert p[1] == 128
        assert p[2] == 0
        assert p[3] == 175

    window.close()


def test_copy_to_clipboard_with_solid_background(qapp):
    """Verifica que si hay un color de fondo sólido en el canvas, se compone antes de copiar."""
    window = MainWindow()
    test_img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    window.load_new_pil_image(test_img, source_name="test_bg.png")

    # Activar fondo blanco sólido en el lienzo
    window.canvas.set_background_color((255, 255, 255))
    window.copy_to_clipboard()

    mime = QGuiApplication.clipboard().mimeData()
    png_bytes = bytes(mime.data("image/png"))

    with Image.open(io.BytesIO(png_bytes)) as restored:
        p = restored.convert("RGBA").getpixel((10, 10))
        # Rojo semitransparente sobre blanco debe dar un tono rojizo sin alfa transparente
        assert p[3] == 255
        assert p[0] > 200

    window.close()


def test_open_image_file_activates_sequence_session(qapp, tmp_path: Path):
    """
    Verifica que open_image_file() activa SequenceSession y la barra de navegación
    tanto para .gif como para .webp animados, y mantiene SequenceSession en None para estáticos.
    """
    window = MainWindow()

    f1 = Image.new("RGBA", (30, 30), (255, 0, 0, 255))
    f2 = Image.new("RGBA", (30, 30), (0, 0, 255, 255))

    # 1. Cargar WebP animado
    webp_path = tmp_path / "session_anim.webp"
    save_animated_webp([f1, f2], webp_path, durations=[90, 90], loop=0)

    window.open_image_file(str(webp_path))
    assert window.sequence_session is not None
    assert window.sequence_session.total_frames == 2
    assert window.gif_bar.isHidden() is False

    # 2. Cargar GIF animado
    gif_path = tmp_path / "session_anim.gif"
    save_animated_gif([f1, f2], gif_path, durations=[90, 90], loop=0)

    window.open_image_file(str(gif_path))
    assert window.sequence_session is not None
    assert window.sequence_session.total_frames == 2
    assert window.gif_bar.isHidden() is False

    # 3. Cargar imagen estática PNG -> desactiva sesión animada
    static_path = tmp_path / "session_static.png"
    f1.save(static_path, format="PNG")

    window.open_image_file(str(static_path))
    assert window.sequence_session is None
    assert window.gif_bar.isHidden() is True

    window.close()
