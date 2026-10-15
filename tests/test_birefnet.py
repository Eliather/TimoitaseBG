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
