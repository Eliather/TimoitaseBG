"""Tests unitarios para la integración de BiRefNet (Dichotomous Image Segmentation - MIT)."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.config import BG_REMOVER_MODELS
from app.core.bg_remover import (
    BackgroundRemover,
    get_bg_remover,
)
from app.ui.toolbar import ToolSidebar


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_birefnet_in_config():
    model_ids = [m[0] for m in BG_REMOVER_MODELS]
    assert "birefnet-general" in model_ids
    # RMBG-2.0 debe estar completamente ausente
    assert "rmbg-2.0" not in model_ids


def test_toolbar_model_combos_contain_birefnet(qapp):
    sidebar = ToolSidebar()
    
    # Comprobar combo de Quitar Fondo individual
    bg_ids = [sidebar.combo_bg_model.itemData(i) for i in range(sidebar.combo_bg_model.count())]
    assert "birefnet-general" in bg_ids
    assert "rmbg-2.0" not in bg_ids

    # Comprobar combo de Lote
    batch_ids = [sidebar.combo_batch_model.itemData(i) for i in range(sidebar.combo_batch_model.count())]
    assert "birefnet-general" in batch_ids
    assert "rmbg-2.0" not in batch_ids

    sidebar.close()


def test_birefnet_inference_mocked():
    remover = BackgroundRemover()
    test_img = Image.new("RGB", (300, 200), (255, 100, 50))

    # Mock de rembg para simular inferencia exitosa de birefnet-general
    mock_out = test_img.convert("RGBA")
    mock_alpha = np.full((200, 300), 255, dtype=np.uint8)
    mock_alpha[:50, :50] = 0
    mock_out.putalpha(Image.fromarray(mock_alpha, mode="L"))

    with patch("rembg.remove", return_value=mock_out) as mock_rembg_remove, \
         patch("rembg.new_session", return_value=MagicMock()):
        result = remover._remove_background_cpu_local(
            test_img,
            model_name="birefnet-general",
            enable_clothing_protection=False,
        )
        assert result.mode == "RGBA"
        assert result.size == (300, 200)
        # Verificar que los canales RGB originales se conservan íntegramente
        rgb_orig = np.array(test_img)
        rgb_res = np.array(result.convert("RGB"))
        np.testing.assert_array_equal(rgb_orig, rgb_res)
        mock_rembg_remove.assert_called_once()


def test_vectorized_postprocess_inspyrenet_and_mask():
    """Verifica que la supresión vectorizada de islas y relleno de huecos funcione exactamente."""
    from app.core.bg_remover import postprocess_inspyrenet, postprocess_mask

    img = Image.new("RGB", (120, 120), (200, 150, 100))
    raw = np.zeros((120, 120), dtype=np.float32)
    # Sujeto principal grande (60x60 = 3600 px)
    raw[30:90, 30:90] = 1.0
    # Mota pequeña flotante (3x3 = 9 px < min_island_area=30)
    raw[5:8, 5:8] = 0.9

    _, alpha_insp = postprocess_inspyrenet(img, raw, antialiasing_radius=0, min_island_area=30)
    assert np.all(alpha_insp[5:8, 5:8] == 0)
    assert np.all(alpha_insp[40:80, 40:80] == 255)

    # Hueco pequeño interno (2x2 = 4 px <= max_hole_area=60)
    raw_with_hole = raw.copy()
    raw_with_hole[5:8, 5:8] = 0.0
    raw_with_hole[55:57, 55:57] = 0.0
    alpha_mask = postprocess_mask(raw_with_hole, morph_ksize=0, max_hole_area=60)
    assert np.all(alpha_mask[55:57, 55:57] == 255)


def test_predict_clothing_mask_downscales_large_images():
    """Verifica que _predict_clothing_mask reduzca imágenes >640px antes de invocar rembg.remove."""
    from app.core.bg_remover import _predict_clothing_mask

    large_img = Image.new("RGB", (1600, 1200), (255, 255, 255))
    captured_sizes = []

    def fake_remove(im, session=None):
        captured_sizes.append(im.size)
        out = im.convert("RGBA")
        out.putalpha(Image.new("L", im.size, 200))
        return out

    with patch("rembg.remove", side_effect=fake_remove):
        mask = _predict_clothing_mask(large_img, human_sess=MagicMock(), max_side=640)

    assert len(captured_sizes) == 1
    assert max(captured_sizes[0]) == 640
    assert mask.shape == (captured_sizes[0][1], captured_sizes[0][0])
    assert np.all(mask == 200)


def test_restorer_single_alpha_resize_in_x2_and_enhance_only():
    """Verifica que ImageRestorer escale al tamaño destino y preserve RGBA en x2 y enhance_only."""
    from app.core.restorer import ImageRestorer

    restorer = ImageRestorer()
    rgba_in = Image.new("RGBA", (32, 24), (100, 150, 200, 180))

    fake_sess = MagicMock()
    mock_in = MagicMock()
    mock_in.name = "input"
    mock_out = MagicMock()
    mock_out.name = "output"
    fake_sess.get_inputs.return_value = [mock_in]
    fake_sess.get_outputs.return_value = [mock_out]
    fake_sess.run.return_value = [np.full((1, 3, 24 * 4, 32 * 4), 0.5, dtype=np.float32)]

    with patch.object(restorer, "_get_session", return_value=fake_sess):
        out_x2 = restorer._restore_image_impl(rgba_in, scale_mode="x2", use_tiling=False)
        assert out_x2.mode == "RGBA"
        assert out_x2.size == (64, 48)
        assert out_x2.getpixel((10, 10))[3] == 180

        out_1x = restorer._restore_image_impl(rgba_in, scale_mode="enhance_only", use_tiling=False)
        assert out_1x.mode == "RGBA"
        assert out_1x.size == (32, 24)
        assert out_1x.getpixel((10, 10))[3] == 180

