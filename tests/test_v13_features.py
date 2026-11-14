"""
Pruebas unitarias e integración headless para las características de TimoitaseBG v1.3.0.
Cubre Fases 1-5: Gestor de Modelos/SHA256, Multi-Punto SAM, Refinamiento/Inpainting,
Estudio Creativo (Effects/Crop/Transform), Tracking Temporal/Trim/FPS, Pipeline Batch y CLI.
"""
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.config import APP_VERSION, PROJECT_ROOT
from app.core import model_fetcher as mf
from app.core.batch_pipeline import BatchPipelineConfig, process_image_pipeline
from app.core.effects import (
    apply_drop_shadow,
    apply_sticker_outline,
    compose_background_mode,
    crop_image_box,
    rotate_flip_image,
)
from app.core.image_state import SequenceSession
from app.core.postprocess import refine_alpha_edges
from app.core.sam_manager import SAMManager
from app.core.selection import inpaint_selected_region
from app.core.sequence_tracker import track_subject_across_sequence
from app.core.yolo_detector import DetectedSubject


def test_app_version_is_130():
    assert APP_VERSION == "1.3.0"


def test_locales_100_percent_key_parity():
    import re
    locales_dir = PROJECT_ROOT / "app" / "locales"
    expected_files = ("es.json", "en.json", "ru.json", "zh.json")
    loaded = {}
    for fname in expected_files:
        p = locales_dir / fname
        assert p.is_file(), f"Falta archivo de idioma: {p}"
        loaded[fname] = json.loads(p.read_text(encoding="utf-8"))

    base_keys = set(loaded["es.json"].keys())
    for fname, data in loaded.items():
        assert set(data.keys()) == base_keys, f"Claves desincronizadas en {fname}"

    used_keys = set()
    pattern = re.compile(r"""tr\(\s*["']([a-zA-Z0-9_]+)["']""")
    for py_file in (PROJECT_ROOT / "app").rglob("*.py"):
        used_keys.update(pattern.findall(py_file.read_text(encoding="utf-8")))
    missing = sorted(used_keys - base_keys)
    assert not missing, f"Claves tr() faltantes en locales: {missing}"


def test_model_inventory_verify_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "MODELS_DIR", tmp_path)
    sample_name = "inspyrenet.onnx"
    payload = b"onnx-dummy-weights-" + (b"0" * mf.MIN_VALID_MODEL_BYTES)
    sample_path = tmp_path / sample_name
    sample_path.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(mf.MODEL_SHA256, sample_name, digest)

    assert mf.verify_local_model_sha256(sample_name) is True
    inv = mf.get_models_inventory()
    entry = next(x for x in inv if x["filename"] == sample_name)
    assert entry["fp32_exists"] is True
    assert entry["has_sha256"] is True
    assert entry["active_variant"] == "fp32"

    removed = mf.delete_local_model(sample_name)
    assert sample_name in removed
    assert not sample_path.exists()
    assert mf.verify_local_model_sha256(sample_name) is None


def test_sam_predict_mask_from_multipoint():
    mgr = SAMManager()
    mgr.cached_embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)
    mgr.orig_size = (100, 80)
    mgr.scaled_size = (1024, 819)
    mgr.scale_factor = 1024.0 / 100.0

    mock_dec = MagicMock()
    mock_dec.get_inputs.return_value = [
        MagicMock(name="image_embeddings"),
        MagicMock(name="point_coords"),
        MagicMock(name="point_labels"),
        MagicMock(name="mask_input"),
        MagicMock(name="has_mask_input"),
        MagicMock(name="orig_im_size"),
    ]
    for inp, n in zip(
        mock_dec.get_inputs.return_value,
        ["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"],
    ):
        inp.name = n
    mock_dec.run.return_value = [np.ones((1, 1, 80, 100), dtype=np.float32) * 2.0]
    mgr.decoder_session = mock_dec

    mask = mgr.predict_mask_from_points([(25, 20), (80, 70)], [1, 0], orig_img_size=(100, 80))
    assert mask is not None
    assert mask.shape == (80, 100)
    assert mask.dtype == bool
    feeds = mock_dec.run.call_args[0][1]
    assert feeds["point_coords"].shape == (1, 2, 2)
    assert list(feeds["point_labels"][0]) == [1.0, 0.0]


def test_refine_alpha_edges_and_inpaint_selection():
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    arr = np.array(img, dtype=np.uint8)
    arr[10:30, 10:30] = [50, 220, 60, 255]
    img = Image.fromarray(arr, "RGBA")

    shrunk = refine_alpha_edges(img, choke_px=2, feather_px=2, despill_strength=0.6)
    shrunk_arr = np.array(shrunk, dtype=np.uint8)
    # Borde exterior del cuadrado debe haberse reducido o suavizado
    assert int(shrunk_arr[10, 10, 3]) < 255
    assert int(shrunk_arr[20, 20, 3]) == 255

    # Inpainting sobre una máscara central
    sel_mask = np.zeros((40, 40), dtype=np.uint8)
    sel_mask[16:24, 16:24] = 255
    test_rgba = np.full((40, 40, 4), 255, dtype=np.uint8)
    test_rgba[:, :, :3] = [200, 100, 50]
    test_rgba[16:24, 16:24, :3] = [0, 0, 0]
    inpainted = inpaint_selected_region(test_rgba, sel_mask, radius=3)
    assert inpainted.shape == (40, 40, 4)
    # La región negra seleccionada debe acercarse al color circundante
    assert int(inpainted[20, 20, 0]) > 120


def test_creative_effects_crop_and_transform():
    subject = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    arr = np.array(subject, dtype=np.uint8)
    arr[8:24, 8:24] = [220, 40, 80, 255]
    subject = Image.fromarray(arr, "RGBA")

    sticker = apply_sticker_outline(subject, width_px=4, color_rgb=(255, 255, 255), expand_canvas=True)
    assert sticker.width > subject.width
    # Con pad = 6, el sujeto empieza en y=14, x=[14..30); el borde de 4px cubre y=[10..14)
    assert np.array(sticker)[11, 22, 3] > 0

    shadowed = apply_drop_shadow(subject, offset_x=6, offset_y=6, blur_radius=4, opacity=0.6, expand_canvas=True)
    assert shadowed.width > subject.width

    orig_bg = Image.new("RGB", (32, 32), (20, 90, 200))
    bokeh = compose_background_mode(subject, mode="bokeh", original_rgb=orig_bg, blur_radius=6, aspect_preset="512x512")
    assert bokeh.size == (512, 512)

    custom_bg = Image.new("RGBA", (64, 64), (10, 200, 30, 255))
    img_bg = compose_background_mode(subject, mode="image", bg_image=custom_bg, aspect_preset="16:9")
    assert img_bg.width >= img_bg.height

    rot = rotate_flip_image(Image.new("RGBA", (30, 10), (1, 2, 3, 255)), rotate_deg=90, flip_h=True)
    assert rot.size == (10, 30)

    cropped = crop_image_box(subject, 8, 8, 24, 24)
    assert cropped.size == (16, 16)


def test_sequence_tracker_and_trim_fps():
    base = np.full((40, 40, 4), [10, 20, 30, 255], dtype=np.uint8)
    base[10:26, 6:22] = [240, 240, 240, 255]
    frames = [Image.fromarray(base.copy(), "RGBA") for _ in range(4)]

    def _make_subj(frame_i: int):
        mask = np.zeros((40, 40), dtype=bool)
        x0 = 6 + frame_i * 2
        mask[10:26, x0:x0 + 12] = True
        return DetectedSubject(
            subject_id=1,
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=(x0, 10, x0 + 12, 26),
            mask=mask,
            color_hex="#3B82F6",
            emoji="",
        )

    fake_yolo = MagicMock()
    fake_yolo.detect.side_effect = lambda img, conf_threshold=0.25: [_make_subj(0)]

    out = track_subject_across_sequence(
        frames=frames,
        seed_frame_idx=1,
        seed_subject=_make_subj(1),
        trim_start=1,
        trim_end=2,
        yolo_detector=fake_yolo,
        use_sam_refinement=False,
    )
    assert len(out) == 4
    # Frame 0 fuera del rango trim conserva alfa original (255 en esquina)
    assert np.array(out[0])[0, 0, 3] == 255
    # Frames 1 y 2 dentro del rango quedan transparentes en esquina y opacos en sujeto
    assert np.array(out[1])[0, 0, 3] < 20
    assert np.array(out[2])[0, 0, 3] < 20
    assert np.array(out[2])[18, 12, 3] > 200

    sess = SequenceSession(frames, durations=[100] * 4, fps=10.0, is_video=True)
    sess.set_trim_range(1, 2)
    sess.set_target_fps(20.0)
    exp_frames, exp_durs, exp_fps = sess.get_export_frames_and_durations()
    assert len(exp_frames) == 2
    assert exp_fps == 20.0
    assert exp_durs == [50, 50]


def test_batch_pipeline_and_headless_cli(tmp_path):
    dummy_img = Image.new("RGBA", (60, 40), (200, 100, 50, 255))
    arr = np.array(dummy_img, dtype=np.uint8)
    arr[0:10, :] = [0, 0, 0, 0]
    in_img = Image.fromarray(arr, "RGBA")

    fake_remover = MagicMock()
    fake_remover.remove_background.return_value = in_img

    cfg = BatchPipelineConfig(
        remove_bg=True,
        autocrop=True,
        sticker_width=4,
        aspect_preset="512x512",
    )
    out_pipe = process_image_pipeline(in_img, pipeline_config=cfg, remover=fake_remover)
    assert out_pipe.size == (512, 512)

    # Verificar que app.cli no importa PySide6.QtWidgets en un subproceso limpio
    code = (
        "import sys, app.cli; "
        "assert 'PySide6.QtWidgets' not in sys.modules; "
        "assert app.cli.main(['models', '--list']) == 0"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # Ejecutar CLI 'remove' con remover mockeado
    src_file = tmp_path / "in.png"
    dst_file = tmp_path / "out.png"
    in_img.save(src_file)
    with patch("app.core.bg_remover.get_bg_remover", return_value=fake_remover):
        from app.cli import main as cli_main
        rc = cli_main(["remove", str(src_file), str(dst_file), "--autocrop", "--aspect", "512x512"])
    assert rc == 0
    assert dst_file.is_file()
    with Image.open(dst_file) as saved:
        assert saved.size == (512, 512)
