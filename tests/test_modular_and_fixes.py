"""
Tests de integración y regresión para las correcciones de Bloques 1–5.
"""
from unittest.mock import MagicMock, patch
import hashlib
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QImage
from PySide6.QtWidgets import QApplication

from app.controllers.document_controller import DocumentController
from app.core.image_state import SequenceSession
from app.core.restorer import ImageRestorer
from app.core.sam_manager import SAMManager
from app.core.workers import BatchWorkerThread
from app.core.yolo_detector import DetectedSubject
from app.ui.canvas_widget import CanvasWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_rgb_canvas_manual_eraser_produces_transparency(qapp):
    """El borrador manual sobre una imagen RGB debe volver los píxeles transparentes, no negros."""
    canvas = CanvasWidget()
    rgb_img = Image.new("RGB", (40, 40), (200, 100, 50))
    canvas.set_image(rgb_img)

    assert canvas.current_qimage is not None
    assert canvas.current_qimage.hasAlphaChannel()

    canvas.set_eraser_mode(True)
    canvas.set_brush_size(12)
    ok = canvas._brush_engine.draw_stroke(
        target_qimage=canvas.current_qimage,
        original_qimage=None,
        start_pt=QPointF(20, 20),
        end_pt=QPointF(20, 20),
        brush_size=12,
        brush_opacity=1.0,
        brush_hardness=1.0,
        eraser_mode=True,
    )
    assert ok is True
    px_color = canvas.current_qimage.pixelColor(20, 20)
    assert px_color.alpha() == 0
    # Zonas no tocadas siguen opacas con su color original
    assert canvas.current_qimage.pixelColor(2, 2).alpha() == 255


def test_canvas_accepts_dropped_in_memory_image(qapp):
    """CanvasWidget debe emitir imageDropped cuando mimeData contiene una imagen en memoria."""
    canvas = CanvasWidget()
    received = []
    canvas.imageDropped.connect(received.append)

    mime = QMimeData()
    qimg = QImage(16, 16, QImage.Format.Format_RGBA8888)
    qimg.fill(Qt.GlobalColor.red)
    mime.setImageData(qimg)

    ev = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.dropEvent(ev)
    assert len(received) == 1
    assert received[0].size == (16, 16)


def test_document_controller_autocrop_and_isolate_propagate_across_sequence():
    """apply_autocrop y apply_isolate_subject_sequence deben transformar todos los fotogramas."""
    f1 = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    f1.paste((255, 0, 0, 255), (20, 20, 30, 30))
    f2 = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    f2.paste((0, 255, 0, 255), (25, 25, 35, 35))

    doc = DocumentController()
    doc.sequence_session = SequenceSession([f1, f2], [100, 100])
    doc.image_state = doc.sequence_session.active_state

    err_key, res = doc.apply_autocrop()
    assert err_key is None
    assert res is not None
    # Union bbox = (20, 20, 35, 35) + pad=10 -> (10, 10, 45, 45) -> 35x35
    assert doc.sequence_session.states[0].current_image.size == (35, 35)
    assert doc.sequence_session.states[1].current_image.size == (35, 35)

    mask = np.zeros((35, 35), dtype=bool)
    mask[10:20, 10:20] = True
    subj = DetectedSubject(
        subject_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(10, 10, 20, 20),
        mask=mask,
        color_hex="#3B82F6",
        emoji="👤",
    )
    isolated = doc.apply_isolate_subject_sequence(subj)
    assert isolated is not None
    for st in doc.sequence_session.states:
        arr = np.array(st.current_image)
        assert arr[1, 1, 3] == 0


def test_sam_manager_recognizes_int8_models(tmp_path):
    """SAMManager.is_available() debe detectar variantes *_int8.onnx en MODELS_DIR."""
    enc = tmp_path / "mobile_sam_image_encoder_int8.onnx"
    dec = tmp_path / "sam_mask_decoder_single_int8.onnx"
    enc.write_bytes(b"0" * 1_100_000)
    dec.write_bytes(b"0" * 1_100_000)

    mgr = SAMManager()
    with patch("app.core.model_fetcher.MODELS_DIR", tmp_path):
        assert mgr.is_available() is True


def test_model_fetcher_range_resume_cancel_and_sha256(tmp_path):
    """ensure_model_file soporta reanudación HTTP 206, cancel_check y validación SHA-256."""
    import app.core.model_fetcher as mf

    payload = b"ABCD" * 300_000
    prefix = payload[:400_000]
    suffix = payload[400_000:]
    expected_sha = hashlib.sha256(payload).hexdigest()

    temp_file = tmp_path / "fake_model.tmp"
    temp_file.write_bytes(prefix)

    captured_headers = {}

    class PartialResp:
        status = 206
        headers = {"Content-Length": str(len(suffix))}

        def __init__(self):
            self._off = 0

        def read(self, n):
            chunk = suffix[self._off : self._off + n]
            self._off += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, context=None, timeout=60):
        captured_headers.update(dict(req.header_items()))
        return PartialResp()

    with patch.object(mf, "MODELS_DIR", tmp_path), \
         patch.dict(mf.MODEL_URLS, {"fake_model.onnx": "https://example.com/fake.onnx"}), \
         patch.dict(mf.MODEL_SHA256, {"fake_model.onnx": expected_sha}), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = mf.ensure_model_file("fake_model.onnx")

    assert captured_headers.get("Range") == f"bytes={len(prefix)}-"
    assert out.exists()
    assert out.read_bytes() == payload

    # Cancelación cooperativa antes de completar
    tmp_path_cancel = tmp_path / "cancel_case"
    tmp_path_cancel.mkdir()
    with patch.object(mf, "MODELS_DIR", tmp_path_cancel), \
         patch.dict(mf.MODEL_URLS, {"fake_model.onnx": "https://example.com/fake.onnx"}), \
         patch("urllib.request.urlopen", return_value=PartialResp()):
        with pytest.raises(InterruptedError):
            mf.ensure_model_file("fake_model.onnx", cancel_check=lambda: True)


def test_restorer_allows_gpu_above_1080p_when_tiling_enabled():
    """Con use_tiling=True una imagen 4K puede usar GPU; sin tiling hace fallback a CPU."""
    restorer = ImageRestorer()
    large_img = Image.new("RGB", (3840, 2160), (10, 20, 30))

    mock_gpu = MagicMock()
    mock_gpu.is_gpu_enabled.return_value = True
    mock_gpu.run_inference.return_value = large_img

    with patch("app.core.gpu_manager.get_gpu_manager", return_value=mock_gpu), \
         patch.object(restorer, "_restore_image_impl", return_value=large_img) as mock_cpu:
        restorer.restore_image(large_img, scale_mode="x2", use_tiling=True)
        mock_gpu.run_inference.assert_called_once()
        mock_cpu.assert_not_called()

        mock_gpu.run_inference.reset_mock()
        restorer.restore_image(large_img, scale_mode="x2", use_tiling=False)
        mock_gpu.run_inference.assert_not_called()
        mock_cpu.assert_called_once()


def test_batch_worker_collects_per_file_errors(tmp_path):
    """BatchWorkerThread registra archivos fallidos y continúa con los válidos."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "corrupt.png").write_bytes(b"not a real png")
    Image.new("RGB", (16, 16), (255, 0, 0)).save(in_dir / "good.png")

    mock_remover = MagicMock()
    mock_remover.remove_background.return_value = Image.new("RGBA", (16, 16), (255, 0, 0, 255))

    worker = BatchWorkerThread(str(in_dir), str(out_dir), "isnet-anime", True)
    recorded = []
    worker.finishedWithErrors.connect(lambda ok, tot, errs: recorded.append((ok, tot, errs)))

    with patch("app.core.bg_remover.get_bg_remover", return_value=mock_remover):
        worker.run()

    assert len(recorded) == 1
    ok, tot, errs = recorded[0]
    assert tot == 2
    assert ok == 1
    assert len(errs) == 1
    assert errs[0][0] == "corrupt.png"
