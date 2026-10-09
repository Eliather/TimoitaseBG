"""Tests unitarios para la integración de MobileSAM (Varita Mágica por IA)."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication

from app.core.sam_manager import SAMManager, get_sam_manager
from app.ui.toolbar import ToolSidebar
from app.ui.canvas_widget import CanvasWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_sam_manager_singleton():
    mgr1 = get_sam_manager()
    mgr2 = get_sam_manager()
    assert mgr1 is mgr2


def test_sam_preprocess_scaling():
    mgr = SAMManager()
    img = Image.new("RGB", (800, 400), (255, 0, 0))
    tensor, scale, new_w, new_h = mgr._preprocess_image(img)
    
    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32
    # El lado más largo (800) se escala a 1024: factor = 1024/800 = 1.28
    assert scale == pytest.approx(1024 / 800, 1e-4)
    assert new_w == 1024
    assert new_h == int(round(400 * (1024 / 800)))  # 512


def test_sam_mocked_prediction():
    mgr = SAMManager()
    mgr.clear_embedding()
    assert not mgr.has_embedding()

    orig_w, orig_h = 200, 100
    img = Image.new("RGB", (orig_w, orig_h), (120, 200, 50))

    # Mock encoder session
    mock_encoder = MagicMock()
    enc_inp = MagicMock()
    enc_inp.name = "image"
    mock_encoder.get_inputs.return_value = [enc_inp]
    mock_encoder.run.return_value = [np.zeros((1, 256, 64, 64), dtype=np.float32)]

    # Mock decoder session
    mock_decoder = MagicMock()
    inp_names = ["image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input", "orig_im_size"]
    mock_inputs = []
    for n in inp_names:
        m = MagicMock()
        m.name = n
        mock_inputs.append(m)
    mock_decoder.get_inputs.return_value = mock_inputs

    dummy_logits = np.full((1, 1, 256, 256), -5.0, dtype=np.float32)
    dummy_logits[0, 0, 50:150, 50:150] = 5.0
    mock_decoder.run.return_value = [dummy_logits, np.array([[0.95]], dtype=np.float32)]

    mgr.encoder_session = mock_encoder
    mgr.decoder_session = mock_decoder

    # 1. Preparar embedding
    mgr.prepare_image(img)
    assert mgr.has_embedding()
    mock_encoder.run.assert_called_once()

    # 2. Predecir máscara en el punto (100, 50)
    mask = mgr.predict_mask_at_point(100, 50, orig_img_size=(orig_w, orig_h))
    assert mask is not None
    assert mask.shape == (orig_h, orig_w)
    assert mask.dtype == bool
    mock_decoder.run.assert_called_once()

    # 3. Limpiar embedding
    mgr.clear_embedding()
    assert not mgr.has_embedding()


def test_wand_ai_toolbar_toggle(qapp):
    sidebar = ToolSidebar()
    sidebar.show()

    # Ir a panel de Varita
    sidebar.btn_rail_brush.click()
    sidebar.btn_tool_wand.click()

    ai_toggled = []
    sidebar.wandAIModeToggled.connect(ai_toggled.append)

    # Inicialmente por defecto es Color
    assert not sidebar.is_wand_ai_mode()
    assert sidebar.wand_color_widget.isVisible()
    assert not sidebar.wand_ai_widget.isVisible()

    # Alternar a modo IA
    sidebar.btn_wand_type_ai.click()
    assert sidebar.is_wand_ai_mode()
    assert ai_toggled == [True]
    assert not sidebar.wand_color_widget.isVisible()
    assert sidebar.wand_ai_widget.isVisible()

    # Alternar de vuelta a Color
    sidebar.btn_wand_type_color.click()
    assert not sidebar.is_wand_ai_mode()
    assert ai_toggled == [True, False]
    assert sidebar.wand_color_widget.isVisible()
    assert not sidebar.wand_ai_widget.isVisible()

    sidebar.close()


def test_canvas_wand_ai_interaction(qapp):
    canvas = CanvasWidget()
    test_img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    canvas.set_image(test_img)
    canvas.set_wand_mode(True)
    canvas.set_wand_ai_mode(True)

    # Mock SAMManager
    fake_mask = np.zeros((100, 100), dtype=bool)
    fake_mask[20:80, 20:80] = True

    mock_sam = MagicMock()
    mock_sam.has_embedding.return_value = True
    mock_sam.predict_mask_at_point.return_value = fake_mask

    with patch("app.core.sam_manager.get_sam_manager", return_value=mock_sam):
        canvas._apply_magic_wand(QPointF(50, 50), mode="new")
        assert canvas.has_selection()
        sel = canvas.active_selection
        assert sel is not None
        assert sel.shape == (100, 100)
        assert sel[50, 50] == 255
        assert sel[5, 5] == 0
