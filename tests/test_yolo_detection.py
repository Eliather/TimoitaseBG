"""Tests unitarios para la integración de YOLO11-seg (Detección y aislamiento multiobjeto)."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.core.yolo_detector import (
    YOLODetector,
    DetectedSubject,
    get_yolo_detector,
    COCO_CLASSES,
)
from app.ui.toolbar import ToolSidebar


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_yolo_detector_singleton():
    d1 = get_yolo_detector()
    d2 = get_yolo_detector()
    assert d1 is d2


def test_yolo_preprocessing():
    detector = YOLODetector()
    img = Image.new("RGB", (400, 200), (50, 100, 150))
    tensor, crop_box, orig_size, scale = detector._preprocess(img, target_size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert orig_size == (400, 200)
    pad_x, pad_y, new_w, new_h = crop_box
    # 400x200 escalado a max 640 -> scale = 640/400 = 1.6 -> 640x320
    assert new_w == 640
    assert new_h == 320
    assert pad_x == 0
    assert pad_y == (640 - 320) // 2


def test_yolo_mock_inference():
    detector = YOLODetector()
    img = Image.new("RGB", (100, 100), (200, 200, 200))

    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_sess.get_inputs.return_value = [mock_input]

    # Salida 0: (1, 116, 8400)
    dummy_out0 = np.zeros((1, 116, 8400), dtype=np.float32)
    # Box 0: centro (320, 320), tamaño (200, 200)
    dummy_out0[0, 0, 0] = 320.0
    dummy_out0[0, 1, 0] = 320.0
    dummy_out0[0, 2, 0] = 200.0
    dummy_out0[0, 3, 0] = 200.0
    # Clase 0 (person) = 0.95
    dummy_out0[0, 4, 0] = 0.95
    # Coeficiente de máscara
    dummy_out0[0, 84, 0] = 1.0

    # Salida 1: Protos (1, 32, 160, 160)
    dummy_out1 = np.ones((1, 32, 160, 160), dtype=np.float32)

    mock_sess.run.return_value = [dummy_out0, dummy_out1]
    detector._session = mock_sess

    subjects = detector.detect_subjects(img, conf_threshold=0.35)
    assert len(subjects) == 1
    s = subjects[0]
    assert s.class_name == "person"
    assert s.emoji == "👤"
    assert s.confidence == pytest.approx(0.95, 1e-3)
    assert s.mask.shape == (100, 100)
    assert s.mask.dtype == bool


def test_toolbar_chips_integration(qapp):
    sidebar = ToolSidebar()
    sidebar.show()

    dummy_mask = np.ones((50, 50), dtype=bool)
    dummy_subject = DetectedSubject(
        subject_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(5, 5, 45, 45),
        mask=dummy_mask,
        color_hex="#3B82F6",
        emoji="👤",
    )

    sidebar.set_detected_subjects([dummy_subject])
    assert sidebar.chips_layout.count() == 1
    chip_btn = sidebar.chips_layout.itemAt(0).widget()
    assert "Person" in chip_btn.text()
    assert "92%" in chip_btn.text()
    assert not sidebar.subject_actions_widget.isVisible()

    # Clic en el chip
    chip_btn.click()
    assert sidebar.subject_actions_widget.isVisible()

    # Emitir aislamiento
    isolated_list = []
    sidebar.isolateSubjectRequested.connect(isolated_list.append)
    sidebar.btn_isolate_subject.click()
    assert len(isolated_list) == 1
    assert isolated_list[0].class_name == "person"

    # Emitir selección
    selected_list = []
    sidebar.selectSubjectRequested.connect(selected_list.append)
    sidebar.btn_select_subject.click()
    assert len(selected_list) == 1
    assert selected_list[0].class_name == "person"

    sidebar.close()
