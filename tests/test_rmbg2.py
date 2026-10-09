"""Tests unitarios para la integración de RMBG-2.0 (Bria AI SOTA)."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.config import BG_REMOVER_MODELS
from app.core.bg_remover import (
    _preprocess_rmbg2,
    _postprocess_rmbg2_mask,
    BackgroundRemover,
    get_bg_remover,
)
from app.ui.toolbar import ToolSidebar


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_rmbg2_in_config():
    model_ids = [m[0] for m in BG_REMOVER_MODELS]
    assert "rmbg-2.0" in model_ids


def test_rmbg2_preprocessing():
    img = Image.new("RGB", (640, 480), (100, 150, 200))
    tensor, crop_box, orig_size = _preprocess_rmbg2(img)

    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert orig_size == (640, 480)
    pad_x, pad_y, new_w, new_h = crop_box
    assert new_w == 1024
    assert new_h == int(round(480 * (1024 / 640)))  # 768
    assert pad_x == 0
    assert pad_y == (1024 - 768) // 2


def test_rmbg2_postprocessing():
    crop_box = (0, 128, 1024, 768)
    orig_size = (640, 480)

    # Simular logits donde el centro del sujeto es positivo y el fondo negativo
    logits = np.full((1, 1, 1024, 1024), -5.0, dtype=np.float32)
    logits[0, 0, 200:600, 200:800] = 5.0

    mask = _postprocess_rmbg2_mask(logits, crop_box, orig_size)
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert mask.max() > 200
    assert mask.min() < 50


def test_rmbg2_inference_mocked():
    remover = BackgroundRemover()
    test_img = Image.new("RGB", (300, 200), (255, 100, 50))

    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "pixel_values"
    mock_sess.get_inputs.return_value = [mock_input]

    # Salida simulada de 1024x1024
    dummy_alphas = np.full((1, 1, 1024, 1024), 0.9, dtype=np.float32)
    mock_sess.run.return_value = [dummy_alphas]

    with patch.object(remover, "get_rmbg2_session", return_value=mock_sess):
        result = remover._remove_background_cpu_local(
            test_img,
            model_name="rmbg-2.0",
            enable_clothing_protection=False,
        )
        assert result.mode == "RGBA"
        assert result.size == (300, 200)
        # Verificar que los canales RGB originales se conservan
        rgb_orig = np.array(test_img)
        rgb_res = np.array(result.convert("RGB"))
        np.testing.assert_array_equal(rgb_orig, rgb_res)


def test_toolbar_model_combos_contain_rmbg2(qapp):
    sidebar = ToolSidebar()
    
    # Comprobar combo de Quitar Fondo individual
    bg_ids = [sidebar.combo_bg_model.itemData(i) for i in range(sidebar.combo_bg_model.count())]
    assert "rmbg-2.0" in bg_ids

    # Comprobar combo de Lote
    batch_ids = [sidebar.combo_batch_model.itemData(i) for i in range(sidebar.combo_batch_model.count())]
    assert "rmbg-2.0" in batch_ids

    sidebar.close()
