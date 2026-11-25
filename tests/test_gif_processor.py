"""
Tests unitarios para app/core/gif_processor.py (GIF, WebP y APNG animados,
límites de fotogramas y presupuesto de memoria RGBA).
"""
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.core.gif_processor import (
    extract_gif_frames,
    is_animated_image,
    save_animated_apng,
    save_animated_gif,
    save_animated_webp,
)


def _make_rgba_frames(count: int = 6, size: tuple[int, int] = (48, 48)) -> list[Image.Image]:
    frames: list[Image.Image] = []
    w, h = size
    for i in range(count):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        x = 6 + i * 4
        draw.rectangle((x, 12, x + 12, 32), fill=(60, 180, 240, 255))
        frames.append(img)
    return frames


def test_gif_roundtrip_and_max_frames(tmp_path: Path):
    frames = _make_rgba_frames(count=8, size=(48, 48))
    gif_path = tmp_path / "anim.gif"
    save_animated_gif(frames, gif_path, durations=[60] * len(frames), loop=0)

    assert is_animated_image(gif_path) is True
    extracted, durations, loop = extract_gif_frames(gif_path)
    assert len(extracted) == 8
    assert len(durations) == 8
    assert loop == 0
    assert all(f.mode == "RGBA" for f in extracted)

    subsampled, sub_durations, _ = extract_gif_frames(gif_path, max_frames=3)
    assert 1 <= len(subsampled) <= 3
    assert len(sub_durations) == len(subsampled)


def test_gif_ram_budget_enforcement(tmp_path: Path):
    frames = _make_rgba_frames(count=6, size=(48, 48))
    gif_path = tmp_path / "budget.gif"
    save_animated_gif(frames, gif_path, durations=[50] * len(frames))

    bytes_per_frame = 48 * 48 * 4
    limited, durations, _ = extract_gif_frames(gif_path, max_frames=600, max_ram_bytes=bytes_per_frame * 2)
    assert 1 <= len(limited) <= 2
    assert len(durations) == len(limited)

    with pytest.raises(ValueError, match="presupuesto"):
        extract_gif_frames(gif_path, max_frames=600, max_ram_bytes=bytes_per_frame - 1)


def test_webp_and_apng_roundtrip_rgba(tmp_path: Path):
    frames = _make_rgba_frames(count=4, size=(40, 40))

    webp_path = tmp_path / "anim.webp"
    save_animated_webp(frames, webp_path, durations=[70] * 4)
    assert is_animated_image(webp_path) is True
    w_frames, w_durs, _ = extract_gif_frames(webp_path)
    assert len(w_frames) == 4
    assert len(w_durs) == 4
    assert all(f.mode == "RGBA" for f in w_frames)

    apng_path = tmp_path / "anim.png"
    save_animated_apng(frames, apng_path, durations=[70] * 4)
    assert is_animated_image(apng_path) is True
    a_frames, a_durs, _ = extract_gif_frames(apng_path)
    assert len(a_frames) == 4
    assert len(a_durs) == 4
    assert all(f.mode == "RGBA" for f in a_frames)
