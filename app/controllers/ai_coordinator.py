"""
TimoitaseBG - Coordinador de Inferencia y Workers de IA.
Centraliza la ejecución asíncrona (segmentación, escalado, lote, SAM, YOLO) y el control GPU/CPU.
"""
import logging
from typing import Any, Optional, Tuple

from PIL import Image

from app.core.bg_remover import get_bg_remover
from app.core.restorer import get_image_restorer

logger = logging.getLogger("TimoitaseBG.AICoordinator")


class AICoordinator:
    """Administra el ciclo de vida de los QThreads de IA y el cambio de hardware GPU/CPU."""

    def __init__(self):
        self.active_worker: Optional[Any] = None
        self.batch_worker: Optional[Any] = None
        self.sam_worker: Optional[Any] = None
        self.yolo_worker: Optional[Any] = None
        self.save_worker: Optional[Any] = None

    @staticmethod
    def _is_running(worker: Optional[Any]) -> bool:
        return worker is not None and worker.isRunning()

    def is_active_running(self) -> bool:
        return self._is_running(self.active_worker)

    def is_batch_running(self) -> bool:
        return self._is_running(self.batch_worker)

    def is_sam_running(self) -> bool:
        return self._is_running(self.sam_worker)

    def is_yolo_running(self) -> bool:
        return self._is_running(self.yolo_worker)

    def start_remove_bg_single(
        self,
        image: Image.Image,
        model_id: str,
        enable_clothing_protection: bool,
        worker_cls: Any,
        on_progress,
        on_finished,
        on_failed,
        on_cancelled=None,
    ) -> Optional[Any]:
        if self.is_active_running():
            logger.info("Tarea de inferencia ya en curso, ignorando solicitud duplicada.")
            return None
        worker = worker_cls(
            get_bg_remover().remove_background,
            image,
            model_name=model_id,
            enable_clothing_protection=enable_clothing_protection,
            description=f"Quitar fondo ({model_id})",
        )
        worker.progressPercent.connect(on_progress)
        worker.finishedResult.connect(on_finished)
        worker.failed.connect(on_failed)
        if on_cancelled is not None and hasattr(worker, "cancelled"):
            worker.cancelled.connect(on_cancelled)
        self.active_worker = worker
        worker.start()
        return worker

    def start_remove_bg_sequence(
        self,
        frames: list,
        model_id: str,
        enable_clothing_protection: bool,
        media_label: str,
        on_progress,
        on_progress_text,
        on_finished,
        on_failed,
        on_cancelled=None,
    ) -> Optional[Any]:
        if self.is_active_running():
            logger.info("Tarea de inferencia ya en curso, ignorando solicitud duplicada.")
            return None
        from app.core.workers import GifWorkerThread

        worker = GifWorkerThread(
            frames=frames,
            model_name=model_id,
            enable_clothing_protection=enable_clothing_protection,
            media_label=media_label,
        )
        worker.progress.connect(on_progress)
        worker.progressText.connect(on_progress_text)
        worker.finishedGif.connect(on_finished)
        worker.failed.connect(on_failed)
        if on_cancelled is not None:
            worker.cancelled.connect(on_cancelled)
        self.active_worker = worker
        worker.start()
        return worker

    def start_restore(
        self,
        image: Image.Image,
        scale_mode: str,
        use_tiling: bool,
        description: str,
        worker_cls: Any,
        on_progress,
        on_finished,
        on_failed,
        on_cancelled=None,
    ) -> Optional[Any]:
        if self.is_active_running():
            logger.info("Tarea de restauración ya en curso, ignorando solicitud duplicada.")
            return None
        worker = worker_cls(
            get_image_restorer().restore_image,
            image,
            scale_mode=scale_mode,
            use_tiling=use_tiling,
            description=description,
        )
        worker.progressPercent.connect(on_progress)
        worker.finishedResult.connect(on_finished)
        worker.failed.connect(on_failed)
        if on_cancelled is not None and hasattr(worker, "cancelled"):
            worker.cancelled.connect(on_cancelled)
        self.active_worker = worker
        worker.start()
        return worker

    def start_restore_sequence(
        self,
        frames: list,
        scale_mode: str,
        use_tiling: bool,
        description: str,
        media_label: str,
        on_progress,
        on_progress_text,
        on_finished,
        on_failed,
        on_cancelled=None,
    ) -> Optional[Any]:
        if self.is_active_running():
            logger.info("Tarea de restauración ya en curso, ignorando solicitud duplicada.")
            return None
        from app.core.workers import SequenceRestoreWorker

        worker = SequenceRestoreWorker(
            frames=frames,
            scale_mode=scale_mode,
            use_tiling=use_tiling,
            description=description,
            media_label=media_label,
        )
        worker.progress.connect(on_progress)
        worker.progressText.connect(on_progress_text)
        worker.finishedGif.connect(on_finished)
        worker.failed.connect(on_failed)
        if on_cancelled is not None:
            worker.cancelled.connect(on_cancelled)
        self.active_worker = worker
        worker.start()
        return worker

    def start_track_sequence(
        self,
        frames: list,
        seed_frame_idx: int,
        seed_subject: Optional[Any],
        seed_mask: Optional[Any],
        trim_start: int,
        trim_end: Optional[int],
        on_progress,
        on_progress_text,
        on_finished,
        on_failed,
        on_cancelled=None,
    ) -> Optional[Any]:
        if self.is_active_running():
            logger.info("Tarea de inferencia ya en curso, ignorando solicitud de rastreo.")
            return None
        from app.core.workers import SequenceTrackWorker

        worker = SequenceTrackWorker(
            frames=frames,
            seed_frame_idx=seed_frame_idx,
            seed_subject=seed_subject,
            seed_mask=seed_mask,
            trim_start=trim_start,
            trim_end=trim_end,
            description="Rastreo de sujeto en secuencia",
        )
        worker.progress.connect(on_progress)
        worker.progressText.connect(on_progress_text)
        worker.finishedGif.connect(on_finished)
        worker.failed.connect(on_failed)
        if on_cancelled is not None:
            worker.cancelled.connect(on_cancelled)
        self.active_worker = worker
        worker.start()
        return worker

    def start_batch(
        self,
        input_dir: str,
        output_dir: str,
        model_id: str,
        enable_clothing_protection: bool,
        on_progress,
        on_progress_text,
        on_finished,
        on_failed,
        on_finished_with_errors=None,
        on_cancelled=None,
        pipeline_config: Optional[Any] = None,
    ) -> Optional[Any]:
        if self.is_batch_running():
            logger.info("Procesamiento en lote ya en curso, ignorando solicitud duplicada.")
            return None
        from app.core.workers import BatchWorkerThread

        worker = BatchWorkerThread(
            input_dir,
            output_dir,
            model_id,
            enable_clothing_protection,
            pipeline_config=pipeline_config,
        )
        worker.progressText.connect(on_progress_text)
        worker.progress.connect(on_progress)
        if on_finished_with_errors is not None:
            worker.finishedWithErrors.connect(on_finished_with_errors)
        else:
            worker.finishedResult.connect(on_finished)
        worker.failed.connect(on_failed)
        if on_cancelled is not None:
            worker.cancelled.connect(on_cancelled)
        self.batch_worker = worker
        worker.start()
        return worker

    def start_save_sequence(self, save_fn, on_saved, on_failed) -> Optional[Any]:
        """Exporta animaciones o video en segundo plano sin congelar la UI."""
        if self._is_running(self.save_worker):
            return None
        from app.core.workers import SaveSequenceWorker

        worker = SaveSequenceWorker(save_fn)
        worker.saved.connect(on_saved)
        worker.failed.connect(on_failed)
        self.save_worker = worker
        worker.start()
        return worker

    def start_sam_embedding(
        self,
        image: Image.Image,
        on_progress,
        on_ready,
        on_failed,
    ) -> Optional[Any]:
        if self.is_sam_running():
            return None
        from app.core.workers import SamEmbeddingWorker

        worker = SamEmbeddingWorker(image)
        worker.progressPercent.connect(on_progress)
        worker.embeddingReady.connect(on_ready)
        worker.failed.connect(on_failed)
        self.sam_worker = worker
        worker.start()
        return worker

    def start_yolo_detection(
        self,
        image: Image.Image,
        on_progress,
        on_finished,
        on_failed,
    ) -> Optional[Any]:
        if self.is_yolo_running():
            logger.info("Detección YOLO ya en curso, ignorando solicitud duplicada.")
            return None
        from app.core.workers import YoloDetectionWorker

        worker = YoloDetectionWorker(image)
        worker.progressPercent.connect(on_progress)
        worker.finishedResult.connect(on_finished)
        worker.failed.connect(on_failed)
        self.yolo_worker = worker
        worker.start()
        return worker

    def unload_sessions(self) -> None:
        """Libera sesiones en memoria de MobileSAM, YOLO11 y del GPU worker."""
        from app.core.gpu_manager import get_gpu_manager
        from app.core.sam_manager import get_sam_manager
        from app.core.yolo_detector import get_yolo_detector

        get_sam_manager().unload_sessions()
        get_yolo_detector().unload_sessions()
        get_gpu_manager().shutdown(blocking=False)

    def switch_device_mode(self, is_gpu: bool) -> Tuple[bool, str]:
        """Alterna GPU/CPU y libera sesiones locales de MobileSAM y YOLO11."""
        from app.config import detect_device, get_gpu_name
        from app.core.gpu_manager import get_gpu_manager
        from app.core.sam_manager import get_sam_manager
        from app.core.yolo_detector import get_yolo_detector

        gpu_mgr = get_gpu_manager()
        success = gpu_mgr.set_gpu_enabled(is_gpu)

        get_sam_manager().unload_sessions()
        get_yolo_detector().unload_sessions()

        if is_gpu and success:
            detected = detect_device()
            gpu_name = get_gpu_name() or (detected if "GPU" in detected else "GPU")
            clean_name = (
                gpu_name.replace("NVIDIA ", "")
                .replace("AMD ", "")
                .replace(" Laptop GPU", "")
                .replace("Advanced Micro Devices, Inc. ", "")
            )
            return True, f"gpu_enabled:{clean_name}"
        if not is_gpu:
            return success, "cpu_enabled"
        return False, "gpu_failed"

    def cancel_active(self) -> bool:
        """Solicita cancelación cooperativa a los workers activos sin matar el proceso."""
        any_cancelled = False
        for attr in ("active_worker", "batch_worker", "sam_worker", "yolo_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                any_cancelled = True
        return any_cancelled

    def shutdown_workers(self, timeout_ms: int = 1000) -> None:
        """Solicita interrupción a los workers activos y cierra el proceso GPU dedicado."""
        for attr in ("active_worker", "batch_worker", "sam_worker", "yolo_worker", "save_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(timeout_ms)
        try:
            from app.core.gpu_manager import get_gpu_manager
            get_gpu_manager().shutdown(blocking=False)
        except Exception:
            pass
