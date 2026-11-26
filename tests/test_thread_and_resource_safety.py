"""
Tests unitarios de seguridad de hilos, prevención de saturación de CPU/GPU
y gestión limpia del ciclo de vida de procesos y workers en TimoitaseBG.
"""
import unittest
from unittest.mock import MagicMock, patch
import os
import pytest
from PIL import Image
import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent

from app.config import get_optimized_session_options
from app.core.sam_manager import SAMManager
from app.core.yolo_detector import YOLODetector


@pytest.mark.usefixtures("qapp")
class TestThreadAndResourceSafety(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])

    def test_optimized_session_options_limits_cpu_threads(self):
        """Verifica que ONNX Runtime no sature todos los núcleos de la CPU."""
        opts = get_optimized_session_options()
        self.assertIsNotNone(opts)
        # intra_op_num_threads debe ser <= 4 y >= 1
        self.assertLessEqual(opts.intra_op_num_threads, 4)
        self.assertGreaterEqual(opts.intra_op_num_threads, 1)
        # inter_op_num_threads debe ser 1 para evitar contención
        self.assertEqual(opts.inter_op_num_threads, 1)

    def test_sam_manager_rejects_concurrent_prepare_calls(self):
        """Verifica que SAMManager no lance dos encoders simultáneos en GPU/CPU."""
        mgr = SAMManager()
        img = Image.new("RGB", (100, 100), (255, 0, 0))

        # Simulamos que un cálculo ya está en progreso
        mgr._is_computing = True
        result = mgr.prepare_image(img)
        # Debe rechazar la llamada simultánea sin error para no saturar recursos
        self.assertFalse(result)

        # Restauramos estado
        mgr._is_computing = False

    def test_sam_manager_predict_mask_thread_safety(self):
        """Verifica que predict_mask_at_point maneje adecuadamente el cerrojo y embedding nulo."""
        mgr = SAMManager()
        mgr.clear_embedding()
        # Con embedding nulo debe retornar None de forma segura sin excepción
        mask = mgr.predict_mask_at_point(50, 50)
        self.assertIsNone(mask)

    def test_yolo_detector_lock_exists(self):
        """Verifica que YOLODetector tenga un cerrojo de sincronización activo."""
        detector = YOLODetector()
        self.assertIsNotNone(detector._lock)
        # Simular imagen vacía
        res = detector.detect_subjects(None)
        self.assertEqual(res, [])

    def test_main_window_worker_guards(self):
        """Verifica que MainWindow evite lanzar workers concurrentes duplicados."""
        from app.ui.main_window import MainWindow

        win = MainWindow()
        win.image_state.load_new_image(Image.new("RGB", (100, 100), (255, 255, 255)))

        # Mock de un worker en ejecución
        mock_worker = MagicMock(spec=QThread)
        mock_worker.isRunning.return_value = True

        # 1. Quitar fondo duplicado
        win.active_worker = mock_worker
        with patch("app.ui.main_window.WorkerThread") as mock_thread_cls:
            win.start_remove_bg("isnet-anime")
            mock_thread_cls.assert_not_called()

        # 2. Restauración duplicada
        with patch("app.ui.main_window.WorkerThread") as mock_thread_cls:
            win.start_restore("x2", True)
            mock_thread_cls.assert_not_called()

        # 3. Detección YOLO duplicada
        win.yolo_worker = mock_worker
        with patch("app.core.workers.YoloDetectionWorker") as mock_yolo_cls:
            win.start_detect_subjects()
            mock_yolo_cls.assert_not_called()

        # 4. MobileSAM duplicado
        win.sam_worker = mock_worker
        with patch("app.core.workers.SamEmbeddingWorker") as mock_sam_cls:
            win._ensure_sam_embedding_async()
            mock_sam_cls.assert_not_called()

        # 5. Lote duplicado
        win.batch_worker = mock_worker
        with patch("app.core.workers.BatchWorkerThread") as mock_batch_cls:
            win.start_batch_processing("in", "out", "isnet-anime", True)
            mock_batch_cls.assert_not_called()

        # 6. Limpieza en closeEvent
        real_event = QCloseEvent()
        win.closeEvent(real_event)
        mock_worker.requestInterruption.assert_called()
        mock_worker.wait.assert_called_with(1000)

        win.close()

    def test_ai_coordinator_cancels_save_worker_and_cleans_up_on_finished(self):
        """Verifica cancel_active sobre save_worker, unload_sessions y liberación de referencias en finished."""
        from app.controllers.ai_coordinator import AICoordinator

        coord = AICoordinator()

        class _FakeSignal:
            def __init__(self):
                self.callbacks = []

            def connect(self, cb):
                self.callbacks.append(cb)

            def emit(self):
                for cb in list(self.callbacks):
                    cb()

        mock_save = MagicMock(spec=QThread)
        mock_save.isRunning.return_value = True
        mock_save.finished = _FakeSignal()

        coord._register_worker("save_worker", mock_save)
        self.assertTrue(coord.cancel_active())
        mock_save.requestInterruption.assert_called_once()

        # Al disparar finished debe limpiar la referencia y llamar deleteLater()
        mock_save.finished.emit()
        self.assertIsNone(coord.save_worker)
        mock_save.deleteLater.assert_called_once()

    def test_ai_coordinator_unload_sessions_and_blocking_shutdown(self):
        """Verifica que unload_sessions invoque bg_remover/restorer y shutdown_workers cierre GPU de forma bloqueante."""
        from app.controllers.ai_coordinator import AICoordinator

        coord = AICoordinator()
        mock_bg = MagicMock()
        mock_restorer = MagicMock()
        mock_sam = MagicMock()
        mock_yolo = MagicMock()
        mock_gpu = MagicMock()

        with patch("app.controllers.ai_coordinator.get_bg_remover", return_value=mock_bg), \
             patch("app.controllers.ai_coordinator.get_image_restorer", return_value=mock_restorer), \
             patch("app.core.sam_manager.get_sam_manager", return_value=mock_sam), \
             patch("app.core.yolo_detector.get_yolo_detector", return_value=mock_yolo), \
             patch("app.core.gpu_manager.get_gpu_manager", return_value=mock_gpu):
            coord.unload_sessions()
            mock_bg.unload_sessions.assert_called_once()
            mock_restorer.unload_sessions.assert_called_once()
            mock_sam.unload_sessions.assert_called_once()
            mock_yolo.unload_sessions.assert_called_once()
            mock_gpu.shutdown.assert_called_with(blocking=False)

            mock_gpu.reset_mock()
            coord.shutdown_workers(timeout_ms=500)
            mock_gpu.shutdown.assert_called_with(blocking=True)

    def test_model_action_worker_quantize_rejects_path_traversal(self):
        """Verifica que ModelActionWorker valide el nombre del modelo con _resolve_safe_model_path."""
        from app.core.workers import ModelActionWorker

        worker = ModelActionWorker("quantize", "../../evil.onnx")
        errors = []
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(len(errors), 1)
        self.assertIn("Nombre de modelo inválido", errors[0])


if __name__ == "__main__":
    unittest.main()
