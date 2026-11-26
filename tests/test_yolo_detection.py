"""Tests unitarios para la integración de YOLO11-seg (Detección y aislamiento multiobjeto)."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from app.core.yolo_detector import (
    YOLODetector,
    DetectedSubject,
    get_yolo_detector,
    COCO_CLASSES,
)
from app.ui.toolbar import ToolSidebar


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


def test_yolo_rectangular_480x640_and_120x160_protos():
    """Verifica soporte dinámico de entrada rectangular 480x640 y prototipos 120x160."""
    detector = YOLODetector()
    img = Image.new("RGB", (320, 240), (120, 140, 160))

    mock_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 480, 640]
    mock_sess.get_inputs.return_value = [mock_input]

    # Salida 0: (1, 116, 6300) con 1 detección centrada en (320, 240), tamaño (200, 160)
    dummy_out0 = np.zeros((1, 116, 6300), dtype=np.float32)
    dummy_out0[0, 0, 0] = 320.0
    dummy_out0[0, 1, 0] = 240.0
    dummy_out0[0, 2, 0] = 200.0
    dummy_out0[0, 3, 0] = 160.0
    dummy_out0[0, 4 + 15, 0] = 0.91  # cat
    dummy_out0[0, 84, 0] = 1.0

    # Salida 1: Protos dinámicos (1, 32, 120, 160)
    dummy_out1 = np.ones((1, 32, 120, 160), dtype=np.float32)

    captured_tensors = []

    def fake_run(_outputs, feed):
        captured_tensors.append(feed["images"])
        return [dummy_out0, dummy_out1]

    mock_sess.run.side_effect = fake_run
    detector._session = mock_sess

    subjects = detector.detect_subjects(img, conf_threshold=0.35)
    assert len(captured_tensors) == 1
    assert captured_tensors[0].shape == (1, 3, 480, 640)
    assert len(subjects) == 1
    s = subjects[0]
    assert s.class_name == "cat"
    assert s.mask.shape == (240, 320)
    assert s.mask.dtype == bool
    assert bool(s.mask[120, 160]) is True


def test_yolo_cpu_fallback_on_gpu_run_error(tmp_path):
    """Si session.run() falla en GPU, YOLODetector reintenta automáticamente en CPU."""
    detector = YOLODetector()
    fake_model = tmp_path / "yolo11n-seg.onnx"
    fake_model.write_bytes(b"0" * 16)
    detector._model_path = fake_model

    gpu_sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 640, 640]
    gpu_sess.get_inputs.return_value = [mock_input]
    gpu_sess.get_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
    gpu_sess.run.side_effect = RuntimeError("DirectML device removed")

    cpu_sess = MagicMock()
    cpu_sess.get_inputs.return_value = [mock_input]
    cpu_sess.get_providers.return_value = ["CPUExecutionProvider"]
    dummy_out0 = np.zeros((1, 116, 8400), dtype=np.float32)
    dummy_out1 = np.zeros((1, 32, 160, 160), dtype=np.float32)
    cpu_sess.run.return_value = [dummy_out0, dummy_out1]

    detector._session = gpu_sess
    img = Image.new("RGB", (64, 64), (10, 20, 30))

    with patch("onnxruntime.InferenceSession", return_value=cpu_sess) as mock_ort:
        res = detector.detect_subjects(img)

    assert res == []
    assert detector._session is cpu_sess
    mock_ort.assert_called_once()
    assert mock_ort.call_args.kwargs["providers"] == ["CPUExecutionProvider"]


def test_sequence_tracker_uses_detect_subjects():
    """sequence_tracker debe invocar detector.detect_subjects (no detector.detect)."""
    from app.core.sequence_tracker import track_subject_across_sequence

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect_subjects(self, pil_img, conf_threshold=0.25):
            self.calls += 1
            mask = np.zeros((pil_img.height, pil_img.width), dtype=bool)
            mask[10:50, 10:50] = True
            return [
                DetectedSubject(
                    subject_id=1,
                    class_id=0,
                    class_name="person",
                    confidence=0.9,
                    bbox=(10, 10, 50, 50),
                    mask=mask,
                    color_hex="#3B82F6",
                    emoji="👤",
                )
            ]

    frames = [
        Image.new("RGB", (64, 64), (100, 120, 140)),
        Image.new("RGB", (64, 64), (105, 125, 145)),
    ]
    fake_det = FakeDetector()
    out = track_subject_across_sequence(
        frames,
        seed_frame_idx=0,
        yolo_detector=fake_det,
        use_sam_refinement=False,
    )
    assert fake_det.calls == 2
    assert len(out) == 2
    assert all(im.mode == "RGBA" and im.size == (64, 64) for im in out)
