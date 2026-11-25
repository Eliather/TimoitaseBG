"""
TimoitaseBG - Hilos de ejecución en segundo plano para inferencia asíncrona no bloqueante.
"""
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger("TimoitaseBG.Workers")


def _close_images(images: List[Image.Image]) -> None:
    """Cierra de forma segura una lista de imágenes PIL liberando buffers."""
    for img in images:
        try:
            img.close()
        except Exception as e:
            logger.debug("Error cerrando frame PIL temporal: %s", e)


def _call_with_optional_kwargs(
    fn: Callable[..., Any],
    *args: Any,
    optional_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Invoca `fn` inyectando kwargs opcionales (ej. `progress_cb`, `cancel_check`) solo si la firma los acepta."""
    call_kwargs = dict(kwargs)
    if optional_kwargs:
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            for key, value in optional_kwargs.items():
                if key not in call_kwargs and (key in sig.parameters or has_var_kw):
                    call_kwargs[key] = value
        except Exception as e:
            logger.debug("Inspección de firma omitida para %s: %s", getattr(fn, "__name__", fn), e)
    return fn(*args, **call_kwargs)


def _gpu_supports_cancel_check() -> bool:
    """Verifica dinámicamente si `GPUWorkerManager.run_inference` acepta `cancel_check`."""
    try:
        from app.core.gpu_manager import GPUWorkerManager

        sig = inspect.signature(GPUWorkerManager.run_inference)
        return "cancel_check" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except Exception as e:
        logger.debug("No se pudo inspeccionar GPUWorkerManager.run_inference: %s", e)
        return False


class _WorkerMixin:
    """Utilidades compartidas de prioridad, progreso y manejo uniforme de excepciones."""

    def _emit_progress(self, percent: int, message: str) -> None:
        pct = max(0, min(100, int(percent)))
        msg = str(message or getattr(self, "description", ""))
        self.progressPercent.emit(pct, msg)  # type: ignore[attr-defined]
        status_sig = getattr(self, "progressStatus", None) or getattr(self, "progress", None)
        if status_sig is not None:
            status_sig.emit(msg)

    def _run_guarded(self, body: Callable[[], None]) -> None:
        try:
            self.setPriority(QThread.Priority.LowPriority)  # type: ignore[attr-defined]
            body()
        except InterruptedError:
            cancel_sig = getattr(self, "cancelled", None)
            if cancel_sig is not None:
                cancel_sig.emit()
        except Exception as e:
            logger.exception("Fallo en worker %s: %s", type(self).__name__, e)
            self.failed.emit(str(e))  # type: ignore[attr-defined]

    def _run_sequence_frames(
        self,
        frames: List[Image.Image],
        step_fn: Callable[[int, int, Image.Image], Image.Image],
    ) -> Optional[List[Image.Image]]:
        """Itera fotogramas verificando cancelación; devuelve None si fue cancelado."""
        total = len(frames)
        out: List[Image.Image] = []
        for idx, frame in enumerate(frames):
            if self.isInterruptionRequested():  # type: ignore[attr-defined]
                self.cancelled.emit()  # type: ignore[attr-defined]
                return None
            out.append(step_fn(idx, total, frame))
        return out


class WorkerThread(_WorkerMixin, QThread):
    """QThread genérico para ejecutar inferencia pesada de IA sin congelar la UI."""

    startedProcessing = Signal(str)
    progress = Signal(str)
    progressPercent = Signal(int, str)    # (porcentaje_0_100, mensaje_etapa)
    finishedResult = Signal(object, str)  # (result_image, action_description)
    cancelled = Signal()
    failed = Signal(str)                  # error_message

    def __init__(self, task_fn: Callable[..., Image.Image], *args, description: str = "Procesando", **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs
        self.description = description

    def run(self) -> None:
        def _exec() -> None:
            self.startedProcessing.emit(f"Iniciando {self.description}")
            self._emit_progress(5, f"Iniciando {self.description}...")

            optional_kw: dict[str, Any] = {
                "progress_cb": self._emit_progress,
                "cancel_check": self.isInterruptionRequested,
            }

            if self.isInterruptionRequested():
                self.cancelled.emit()
                return

            result = _call_with_optional_kwargs(
                self.task_fn,
                *self.args,
                optional_kwargs=optional_kw,
                **self.kwargs,
            )
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self._emit_progress(100, f"{self.description} completado.")
            self.finishedResult.emit(result, self.description)

        self._run_guarded(_exec)


class BatchWorkerThread(_WorkerMixin, QThread):
    """QThread para procesar múltiples imágenes y videos en lote secuencialmente."""

    progress = Signal(int, int)                 # actual, total
    progressText = Signal(str)
    finishedResult = Signal(int, int)           # success, total
    finishedWithErrors = Signal(int, int, list) # success, total, list[(filename, reason)]
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_name: str,
        enable_clothing_protection: bool,
        pipeline_config: Optional[Any] = None,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.enable_clothing_protection = enable_clothing_protection
        self.pipeline_config = pipeline_config
        self.errors: list[tuple[str, str]] = []

    def _process_single_file(self, f: Path, output_path: Path, remover: Any) -> bool:
        """Procesa un archivo individual (video, secuencia animada o imagen). False si fue cancelado."""
        from app.core.gif_processor import (
            extract_gif_frames,
            is_animated_image,
            save_animated_apng,
            save_animated_gif,
            save_animated_webp,
        )
        from app.core.video_processor import (
            extract_video_frames,
            is_video_file,
            process_video_frames,
            save_video_sequence,
        )

        ext = f.suffix.lower()
        if is_video_file(f):
            frames, _durations, fps = extract_video_frames(f)
            processed_frames: List[Image.Image] = []
            try:
                processed_frames = process_video_frames(
                    frames=frames,
                    model_name=self.model_name,
                    preserve_clothing=self.enable_clothing_protection,
                    cancel_check=self.isInterruptionRequested,
                )
                if self.isInterruptionRequested():
                    return False
                out_ext = ext if ext in (".webm", ".mkv", ".mov", ".mp4", ".avi") else ".webm"
                save_video_sequence(processed_frames, output_path / f"{f.stem}_bgrm{out_ext}", fps=fps)
            finally:
                _close_images(processed_frames)
                _close_images(frames)
            return True

        if is_animated_image(f):
            frames, durations, loop = extract_gif_frames(f)
            processed_frames = []
            try:
                for fr in frames:
                    if self.isInterruptionRequested():
                        return False
                    processed_frames.append(
                        _call_with_optional_kwargs(
                            remover.remove_background,
                            fr,
                            optional_kwargs={"cancel_check": self.isInterruptionRequested},
                            model_name=self.model_name,
                            enable_clothing_protection=self.enable_clothing_protection,
                        )
                    )
                if self.isInterruptionRequested():
                    return False
                if ext == ".gif":
                    save_animated_gif(processed_frames, output_path / f"{f.stem}_bgrm.gif", durations=durations, loop=loop)
                elif ext == ".webp":
                    save_animated_webp(processed_frames, output_path / f"{f.stem}_bgrm.webp", durations=durations, loop=loop)
                else:
                    save_animated_apng(processed_frames, output_path / f"{f.stem}_bgrm.png", durations=durations, loop=loop)
            finally:
                _close_images(processed_frames)
                _close_images(frames)
            return True

        with Image.open(f) as raw_img:
            img = raw_img.convert("RGBA")
        try:
            if self.pipeline_config is not None:
                from app.core.batch_pipeline import process_image_pipeline

                res = process_image_pipeline(
                    img,
                    model_name=self.model_name,
                    enable_clothing_protection=self.enable_clothing_protection,
                    pipeline_config=self.pipeline_config,
                    remover=remover,
                )
            else:
                res = _call_with_optional_kwargs(
                    remover.remove_background,
                    img,
                    optional_kwargs={"cancel_check": self.isInterruptionRequested},
                    model_name=self.model_name,
                    enable_clothing_protection=self.enable_clothing_protection,
                )
            try:
                res.save(output_path / f"{f.stem}_bgrm.png")
            finally:
                res.close()
        finally:
            img.close()
        return True

    def run(self) -> None:
        def _exec() -> None:
            from app.core.bg_remover import get_bg_remover
            from app.core.video_processor import VIDEO_EXTENSIONS

            input_path = Path(self.input_dir)
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".apng", *VIDEO_EXTENSIONS}
            files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            total = len(files)
            self.errors = []

            if total == 0:
                self.finishedWithErrors.emit(0, 0, [])
                self.finishedResult.emit(0, 0)
                return

            success_count = 0
            remover = get_bg_remover()

            for i, f in enumerate(files):
                if self.isInterruptionRequested():
                    break
                self.progressText.emit(f"Procesando {i+1}/{total}: {f.name}")
                self.progress.emit(i, total)
                try:
                    if not self._process_single_file(f, output_path, remover):
                        break
                    success_count += 1
                except Exception as e:
                    logger.warning("Error procesando %s: %s", f.name, e)
                    self.errors.append((f.name, str(e)))

            if self.isInterruptionRequested():
                self.cancelled.emit()
                return

            self.progressText.emit("Proceso completado")
            self.progress.emit(total, total)
            self.finishedWithErrors.emit(success_count, total, list(self.errors))
            self.finishedResult.emit(success_count, total)

        self._run_guarded(_exec)


class SamEmbeddingWorker(_WorkerMixin, QThread):
    """QThread para precomputar el embedding MobileSAM en segundo plano sin congelar la UI."""

    progressStatus = Signal(str)
    progressPercent = Signal(int, str)
    embeddingReady = Signal(bool)
    finishedResult = Signal(bool)
    failed = Signal(str)

    def __init__(self, image: Image.Image, image_id: Optional[int] = None):
        super().__init__()
        self.image = image
        self.image_id = image_id

    def run(self) -> None:
        def _exec() -> None:
            from app.core.sam_manager import get_sam_manager

            self._emit_progress(5, "Preparando Varita IA (MobileSAM)...")
            success = get_sam_manager().prepare_image(
                self.image, image_id=self.image_id, progress_cb=self._emit_progress
            )
            self.embeddingReady.emit(success)
            self.finishedResult.emit(success)

        self._run_guarded(_exec)


class YoloDetectionWorker(_WorkerMixin, QThread):
    """QThread para descargar y ejecutar YOLO11-seg en segundo plano."""

    progressStatus = Signal(str)
    progressPercent = Signal(int, str)
    finishedResult = Signal(list)  # List[DetectedSubject]
    failed = Signal(str)

    def __init__(self, image: Image.Image, conf_threshold: float = 0.35):
        super().__init__()
        self.image = image
        self.conf_threshold = conf_threshold

    def run(self) -> None:
        def _exec() -> None:
            from app.core.yolo_detector import get_yolo_detector

            self._emit_progress(5, "Detectando sujetos con IA (YOLO11)...")
            subjects = get_yolo_detector().detect_subjects(
                self.image,
                conf_threshold=self.conf_threshold,
                progress_cb=self._emit_progress,
            )
            self.finishedResult.emit(subjects)

        self._run_guarded(_exec)


class GifWorkerThread(_WorkerMixin, QThread):
    """QThread para eliminar el fondo de todos los fotogramas con estabilización temporal."""

    progress = Signal(int, int)
    progressText = Signal(str)
    finishedGif = Signal(list, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        frames: List[Image.Image],
        model_name: str,
        enable_clothing_protection: bool = True,
        temporal_smoothing: float = 0.22,
        media_label: str = "GIF",
    ):
        super().__init__()
        self.frames = frames
        self.model_name = model_name
        self.enable_clothing_protection = enable_clothing_protection
        self.temporal_smoothing = temporal_smoothing
        self.media_label = media_label

    def run(self) -> None:
        def _exec() -> None:
            from app.core.video_processor import process_video_frames

            def _on_progress(cur: int, total: int) -> None:
                self.progressText.emit(
                    f"⏳ Quitando fondo al {self.media_label} ({cur}/{total} fotogramas) con {self.model_name}..."
                )
                self.progress.emit(cur, total)

            processed = process_video_frames(
                frames=self.frames,
                model_name=self.model_name,
                preserve_clothing=self.enable_clothing_protection,
                temporal_smoothing=self.temporal_smoothing,
                progress_callback=_on_progress,
                cancel_check=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                _close_images(processed)
                self.cancelled.emit()
                return
            self.finishedGif.emit(
                processed,
                f"GIF sin fondo ({len(self.frames)} frames • {self.model_name})",
            )

        self._run_guarded(_exec)


class SequenceRestoreWorker(_WorkerMixin, QThread):
    """QThread para escalar o mejorar nitidez de todos los fotogramas de una secuencia."""

    progress = Signal(int, int)
    progressText = Signal(str)
    finishedGif = Signal(list, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        frames: List[Image.Image],
        scale_mode: str = "x2",
        use_tiling: bool = True,
        description: str = "Escalado",
        media_label: str = "GIF",
    ):
        super().__init__()
        self.frames = frames
        self.scale_mode = scale_mode
        self.use_tiling = use_tiling
        self.description = description
        self.media_label = media_label

    def run(self) -> None:
        def _exec() -> None:
            from app.core.restorer import get_image_restorer

            restorer = get_image_restorer()

            def _step(idx: int, total: int, frame: Image.Image) -> Image.Image:
                self.progressText.emit(f"⏳ {self.description} {self.media_label} ({idx + 1}/{total})...")
                self.progress.emit(idx + 1, total)
                return _call_with_optional_kwargs(
                    restorer.restore_image,
                    frame,
                    optional_kwargs={"cancel_check": self.isInterruptionRequested},
                    scale_mode=self.scale_mode,
                    use_tiling=self.use_tiling,
                )

            processed = self._run_sequence_frames(self.frames, _step)
            if processed is not None:
                self.finishedGif.emit(processed, f"{self.description} ({len(self.frames)} frames)")

        self._run_guarded(_exec)


class SaveSequenceWorker(_WorkerMixin, QThread):
    """QThread para exportar secuencias animadas o video sin congelar la interfaz."""

    saved = Signal(object)  # Path
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, save_fn: Callable[[], Any]):
        super().__init__()
        self._save_fn = save_fn

    def run(self) -> None:
        def _exec() -> None:
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            out = _call_with_optional_kwargs(
                self._save_fn,
                optional_kwargs={"cancel_check": self.isInterruptionRequested},
            )
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self.saved.emit(out)

        self._run_guarded(_exec)


class ModelActionWorker(_WorkerMixin, QThread):
    """QThread para descargar, cuantizar INT8 o verificar SHA-256 de un modelo."""

    progressPercent = Signal(int)
    progressText = Signal(str)
    finishedAction = Signal(str, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, action: str, filename: str):
        super().__init__()
        self.action = action
        self.filename = filename

    def run(self) -> None:
        def _exec() -> None:
            from app.core.model_fetcher import (
                _resolve_safe_model_path,
                ensure_model_file,
                verify_local_model_sha256,
            )

            if self.action == "download":
                def _prog(dl: int, tot: int) -> None:
                    if tot > 0:
                        self.progressPercent.emit(min(100, int(round((dl / tot) * 100))))

                ensure_model_file(
                    self.filename,
                    progress_cb=_prog,
                    status_cb=self.progressText.emit,
                    cancel_check=self.isInterruptionRequested,
                )
                self.finishedAction.emit(self.filename, "ok")
            elif self.action == "quantize":
                from app.core.quantizer import quantize_model

                self.progressPercent.emit(15)
                self.progressText.emit(self.filename)
                src = _resolve_safe_model_path(self.filename)
                if not src.is_file():
                    raise FileNotFoundError(self.filename)
                if not quantize_model(src):
                    raise RuntimeError(self.filename)
                self.progressPercent.emit(100)
                self.finishedAction.emit(self.filename, "quantized")
            elif self.action == "verify":
                self.progressPercent.emit(30)
                ok = verify_local_model_sha256(self.filename)
                self.progressPercent.emit(100)
                status = "sha_ok" if ok is True else ("sha_fail" if ok is False else "sha_none")
                self.finishedAction.emit(self.filename, status)
            else:
                raise ValueError(self.action)

        self._run_guarded(_exec)


class SequenceTrackWorker(_WorkerMixin, QThread):
    """QThread para rastreo temporal de un sujeto (YOLO11 + MobileSAM) sobre una secuencia."""

    progress = Signal(int, int)
    progressText = Signal(str)
    finishedGif = Signal(list, str)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        frames: List[Image.Image],
        seed_frame_idx: int = 0,
        seed_subject: Optional[Any] = None,
        seed_mask: Optional[Any] = None,
        trim_start: int = 0,
        trim_end: Optional[int] = None,
        description: str = "Rastreo de secuencia",
    ):
        super().__init__()
        self.frames = frames
        self.seed_frame_idx = seed_frame_idx
        self.seed_subject = seed_subject
        self.seed_mask = seed_mask
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.description = description

    def run(self) -> None:
        def _exec() -> None:
            from app.core.sequence_tracker import track_subject_across_sequence

            def _prog(pct: int, msg: str) -> None:
                self.progress.emit(int(pct), 100)
                self.progressText.emit(msg)

            out_frames = track_subject_across_sequence(
                frames=self.frames,
                seed_frame_idx=self.seed_frame_idx,
                seed_subject=self.seed_subject,
                seed_mask=self.seed_mask,
                trim_start=self.trim_start,
                trim_end=self.trim_end,
                progress_cb=_prog,
                cancel_check=self.isInterruptionRequested,
            )
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self.finishedGif.emit(out_frames, self.description)

        self._run_guarded(_exec)
