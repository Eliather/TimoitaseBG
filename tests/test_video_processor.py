"""
Tests unitarios para el soporte de video multifomato (app/core/video_processor.py).
Verifica lectura de MP4, MOV, WebM, MKV, AVI, extracción de fotogramas RGBA,
eliminación de fondo frame-by-frame y exportación a formatos libres de regalías.
"""
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.core.image_state import SequenceSession
from app.core.video_processor import (
    VIDEO_EXTENSIONS,
    extract_video_frames,
    is_video_file,
    process_video_frames,
    save_video_sequence,
)


def _make_synthetic_frames(count: int = 5, size: tuple[int, int] = (96, 72)) -> list[Image.Image]:
    """Genera una lista de fotogramas RGBA sintéticos con un círculo en movimiento."""
    frames: list[Image.Image] = []
    w, h = size
    for i in range(count):
        img = Image.new("RGBA", (w, h), (20, 30, 40, 255))
        draw = ImageDraw.Draw(img)
        cx = 16 + i * 12
        cy = h // 2
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(255, 80, 40, 255))
        frames.append(img)
    return frames


def _make_transparent_frames(count: int = 4, size: tuple[int, int] = (80, 60)) -> list[Image.Image]:
    """Genera fotogramas RGBA con fondo transparente (alpha=0) y sujeto opaco."""
    frames: list[Image.Image] = []
    w, h = size
    for i in range(count):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx = 15 + i * 10
        cy = h // 2
        draw.rectangle((cx - 8, cy - 8, cx + 8, cy + 8), fill=(50, 200, 255, 255))
        frames.append(img)
    return frames


def test_is_video_file_extensions(tmp_path: Path):
    """Verifica que todos los formatos de video actuales sean reconocidos como video."""
    for ext in VIDEO_EXTENSIONS:
        vid_file = tmp_path / f"sample{ext}"
        vid_file.write_bytes(b"dummy")
        assert is_video_file(vid_file) is True

    png_file = tmp_path / "image.png"
    png_file.write_bytes(b"dummy")
    assert is_video_file(png_file) is False
    assert is_video_file(tmp_path / "missing.mp4") is False


@pytest.mark.parametrize("ext", [".webm", ".mkv", ".mov", ".mp4", ".avi"])
def test_video_roundtrip_extract_and_save_free_containers(tmp_path: Path, ext: str):
    """
    Crea un video en cada contenedor libre (.webm, .mkv, .mov, .mp4, .avi) y verifica
    que extract_video_frames extraiga todos los fotogramas como RGBA con FPS válido.
    """
    src_frames = _make_synthetic_frames(count=5, size=(96, 72))
    video_path = tmp_path / f"test_clip{ext}"

    save_video_sequence(src_frames, video_path, fps=24.0)
    assert video_path.exists()
    assert video_path.stat().st_size > 0

    extracted_frames, durations, fps = extract_video_frames(video_path)
    assert len(extracted_frames) == 5
    assert len(durations) == 5
    assert fps > 0
    for frm in extracted_frames:
        assert frm.mode == "RGBA"
        assert frm.size == (96, 72)


def test_extract_video_frames_max_frames_stride(tmp_path: Path):
    """Verifica que max_frames realice submuestreo uniforme para proteger la RAM."""
    src_frames = _make_synthetic_frames(count=10, size=(64, 48))
    video_path = tmp_path / "long_clip.mp4"
    save_video_sequence(src_frames, video_path, fps=30.0)

    extracted, durations, eff_fps = extract_video_frames(video_path, max_frames=4)
    assert 1 <= len(extracted) <= 4
    assert len(durations) == len(extracted)
    assert eff_fps > 0


@pytest.mark.parametrize("ext", [".webp", ".apng", ".gif", ".zip"])
def test_save_video_sequence_to_transparent_formats(tmp_path: Path, ext: str):
    """
    Verifica que una secuencia de video con fondo eliminado pueda exportarse a formatos
    libres con transparencia nativa (.webp, .apng, .gif y secuencia PNG en .zip).
    """
    transparent_frames = _make_transparent_frames(count=4, size=(80, 60))
    out_path = tmp_path / f"exported_transparent{ext}"

    save_video_sequence(transparent_frames, out_path, fps=20.0, bg_color_rgb=None)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    if ext == ".zip":
        with zipfile.ZipFile(out_path, "r") as zf:
            names = sorted(zf.namelist())
            assert len(names) == 4
            assert names[0] == "frame_00001.png"
            with Image.open(zf.open(names[0])) as first_png:
                assert first_png.mode == "RGBA"
                assert first_png.getpixel((0, 0))[3] == 0


def test_process_video_frames_frame_by_frame():
    """Verifica la eliminación de fondo frame por frame con estabilización temporal."""
    raw_frames = _make_synthetic_frames(count=4, size=(64, 64))
    transparent_frames = _make_transparent_frames(count=4, size=(64, 64))

    call_idx = {"i": 0}

    def fake_remove_bg(img, model_name="isnet-anime", enable_clothing_protection=False):
        idx = call_idx["i"]
        call_idx["i"] += 1
        return transparent_frames[min(idx, len(transparent_frames) - 1)].copy()

    progress_steps = []

    with patch("app.core.video_processor.get_bg_remover") as mock_get_remover:
        mock_get_remover.return_value.remove_background.side_effect = fake_remove_bg
        result = process_video_frames(
            raw_frames,
            model_name="isnet-anime",
            preserve_clothing=True,
            temporal_smoothing=0.22,
            progress_callback=lambda cur, tot: progress_steps.append((cur, tot)),
        )

    assert len(result) == 4
    assert progress_steps == [(1, 4), (2, 4), (3, 4), (4, 4)]
    for frm in result:
        assert frm.mode == "RGBA"
        # Esquina superior izquierda debe ser transparente
        assert frm.getpixel((0, 0))[3] == 0


def test_sequence_session_video_metadata():
    """Verifica que SequenceSession conserve fps e is_video correctamente."""
    frames = _make_synthetic_frames(count=3, size=(64, 64))
    session = SequenceSession(frames, durations=[33, 33, 33], loop=0, fps=30.0, is_video=True)
    assert session.is_video is True
    assert session.fps == 30.0
    assert session.total_frames == 3
