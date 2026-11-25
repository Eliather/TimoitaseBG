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
    libres con transparencia nativa (.webp, .apng, .gif y secuencia PNG en .zip),
    incluyendo la ruta paralelizada con >4 fotogramas.
    """
    transparent_frames = _make_transparent_frames(count=6, size=(80, 60))
    out_path = tmp_path / f"exported_transparent{ext}"

    save_video_sequence(transparent_frames, out_path, fps=20.0, bg_color_rgb=None)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    if ext == ".zip":
        with zipfile.ZipFile(out_path, "r") as zf:
            names = sorted(zf.namelist())
            assert len(names) == 6
            assert names[0] == "frame_00001.png"
            assert names[-1] == "frame_00006.png"
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


def test_extract_video_frames_ram_budget(tmp_path: Path):
    """Verifica que extract_video_frames respete max_ram_bytes (submuestreo y rechazo)."""
    src_frames = _make_synthetic_frames(count=8, size=(64, 48))
    video_path = tmp_path / "ram_clip.mp4"
    save_video_sequence(src_frames, video_path, fps=24.0)

    bytes_per_frame = 64 * 48 * 4
    # Permitir como máximo 2 fotogramas en presupuesto RAM
    extracted, durations, _ = extract_video_frames(video_path, max_frames=600, max_ram_bytes=bytes_per_frame * 2)
    assert 1 <= len(extracted) <= 2
    assert len(durations) == len(extracted)

    # Rechazar si un solo fotograma excede el presupuesto
    with pytest.raises(ValueError, match="presupuesto"):
        extract_video_frames(video_path, max_frames=600, max_ram_bytes=bytes_per_frame - 1)


def test_find_ffmpeg_safe_excludes_cwd_on_windows(tmp_path: Path, monkeypatch):
    """Verifica que _find_ffmpeg_safe excluya el CWD del PATH en Windows."""
    from app.core import video_processor as vp

    monkeypatch.setattr(vp.sys, "platform", "win32")
    monkeypatch.chdir(tmp_path)
    other_dir = tmp_path / "bin"
    other_dir.mkdir()
    monkeypatch.setenv("PATH", f".;{tmp_path};{other_dir}")

    captured_path = {}

    def fake_which(name, path=None):
        captured_path["path"] = path
        return None

    monkeypatch.setattr(vp.shutil, "which", fake_which)
    vp._find_ffmpeg_safe()
    parts = (captured_path.get("path") or "").split(vp.os.pathsep)
    assert str(other_dir) in parts
    assert "." not in parts
    assert str(tmp_path) not in parts


def test_try_save_transparent_ffmpeg_kills_on_error_and_resolves_path(tmp_path: Path, monkeypatch):
    """Verifica que _try_save_transparent_ffmpeg pase ruta resuelta y mate el proceso en finally ante fallo."""
    from unittest.mock import MagicMock
    from app.core import video_processor as vp

    frames = _make_transparent_frames(count=2, size=(32, 32))
    out_path = tmp_path / "out.webm"

    fake_proc = MagicMock()
    fake_proc.stdin = MagicMock()
    fake_proc.stdin.closed = False
    fake_proc.stdin.write.side_effect = BrokenPipeError("broken pipe")
    fake_proc.poll.return_value = None

    captured_cmd = {}

    def fake_popen(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return fake_proc

    monkeypatch.setattr(vp, "_find_ffmpeg_safe", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(vp.subprocess, "Popen", fake_popen)

    ok = vp._try_save_transparent_ffmpeg(frames, out_path, fps=24.0)
    assert ok is False
    assert captured_cmd["cmd"][-1] == str(out_path.resolve())
    fake_proc.kill.assert_called_once()
    fake_proc.wait.assert_called()
