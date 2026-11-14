"""
TimoitaseBG - Hilos de ejecución en segundo plano para inferencia asíncrona no bloqueante.
"""
import inspect
from typing import Callable, Any, Tuple, Dict, Optional, List
from PIL import Image
from PySide6.QtCore import QThread, Signal


class WorkerThread(QThread):
    """
    QThread genérico para ejecutar inferencia pesada de IA sin congelar la interfaz gráfica.
    """

    # Señales emitidas hacia la UI
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

    def _emit_progress(self, percent: int, message: str):
        pct = max(0, min(100, int(percent)))
        msg = str(message or self.description)
        self.progressPercent.emit(pct, msg)
        self.progress.emit(msg)

    def run(self):
        try:
            self.setPriority(QThread.Priority.LowPriority)
            self.startedProcessing.emit(f"Iniciando {self.description}")
            self._emit_progress(5, f"Iniciando {self.description}...")

            call_kwargs = dict(self.kwargs)
            try:
                sig = inspect.signature(self.task_fn)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if "progress_cb" not in call_kwargs and ("progress_cb" in sig.parameters or has_var_kw):
                    call_kwargs["progress_cb"] = self._emit_progress
                if "cancel_check" not in call_kwargs and ("cancel_check" in sig.parameters or has_var_kw):
                    call_kwargs["cancel_check"] = self.isInterruptionRequested
            except Exception:
                pass

            if self.isInterruptionRequested():
                self.cancelled.emit()
                return

            result = self.task_fn(*self.args, **call_kwargs)
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
            self._emit_progress(100, f"{self.description} completado.")
            self.finishedResult.emit(result, self.description)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))

class BatchWorkerThread(QThread):
    """
    QThread para procesar múltiples imágenes y videos en lote secuencialmente.
    """
    progress = Signal(int, int) # actual, total
    progressText = Signal(str)
    finishedResult = Signal(int, int) # success, total
    finishedWithErrors = Signal(int, int, list)  # success, total, list[(filename, reason)]
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

    def run(self):
        from pathlib import Path
        from app.core.bg_remover import get_bg_remover
        import traceback

        try:
            self.setPriority(QThread.Priority.LowPriority)
            input_path = Path(self.input_dir)
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Encontrar imágenes y videos soportados
            from app.core.video_processor import (
                VIDEO_EXTENSIONS,
                is_video_file,
                extract_video_frames,
                process_video_frames,
                save_video_sequence,
            )

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
                    ext = f.suffix.lower()
                    from app.core.gif_processor import (
                        is_animated_image,
                        extract_gif_frames,
                        save_animated_gif,
                        save_animated_webp,
                        save_animated_apng,
                    )

                    if is_video_file(f):
                        frames, _durations, fps = extract_video_frames(f)
                        processed_frames = process_video_frames(
                            frames=frames,
                            model_name=self.model_name,
                            preserve_clothing=self.enable_clothing_protection,
                            cancel_check=self.isInterruptionRequested,
                        )
                        if self.isInterruptionRequested():
                            break

                        out_ext = ext if ext in (".webm", ".mkv", ".mov", ".mp4", ".avi") else ".webm"
                        out_file = output_path / f"{f.stem}_bgrm{out_ext}"
                        save_video_sequence(processed_frames, out_file, fps=fps)

                        for pf in processed_frames:
                            pf.close()
                        for fr in frames:
                            fr.close()
                    elif is_animated_image(f):
                        frames, durations, loop = extract_gif_frames(f)
                        processed_frames = []
                        for fr in frames:
                            if self.isInterruptionRequested():
                                break
                            res_fr = remover.remove_background(
                                fr,
                                model_name=self.model_name,
                                enable_clothing_protection=self.enable_clothing_protection,
                            )
                            processed_frames.append(res_fr)

                        if self.isInterruptionRequested():
                            break

                        if ext == ".gif":
                            out_file = output_path / f"{f.stem}_bgrm.gif"
                            save_animated_gif(processed_frames, out_file, durations=durations, loop=loop)
                        elif ext == ".webp":
                            out_file = output_path / f"{f.stem}_bgrm.webp"
                            save_animated_webp(processed_frames, out_file, durations=durations, loop=loop)
                        else:  # .png, .apng
                            out_file = output_path / f"{f.stem}_bgrm.png"
                            save_animated_apng(processed_frames, out_file, durations=durations, loop=loop)

                        for pf in processed_frames:
                            pf.close()
                        for fr in frames:
                            fr.close()
                    else:
                        with Image.open(f) as raw_img:
                            img = raw_img.convert("RGBA")

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
                            res = remover.remove_background(
                                img,
                                model_name=self.model_name,
                                enable_clothing_protection=self.enable_clothing_protection,
                            )

                        out_file = output_path / f"{f.stem}_bgrm.png"
                        res.save(out_file)
                        res.close()
                        img.close()

                    success_count += 1
                except Exception as e:
                    print(f"Error procesando {f.name}: {e}")
                    self.errors.append((f.name, str(e)))

            if self.isInterruptionRequested():
                self.cancelled.emit()
                return

            self.progressText.emit("Proceso completado")
            self.progress.emit(total, total)
            self.finishedWithErrors.emit(success_count, total, list(self.errors))
            self.finishedResult.emit(success_count, total)

        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))



class SamEmbeddingWorker(QThread):
    """
    QThread para descargar (si es necesario) y calcular el embedding de MobileSAM
    en segundo plano sin congelar la interfaz gráfica.
    """
    progressStatus = Signal(str)
    progressPercent = Signal(int, str)
    embeddingReady = Signal(bool)
    finishedResult = Signal(bool)
    failed = Signal(str)

    def __init__(self, image: Image.Image, image_id: Optional[int] = None):
        super().__init__()
        self.image = image
        self.image_id = image_id

    def _emit_progress(self, percent: int, message: str):
        pct = max(0, min(100, int(percent)))
        msg = str(message)
        self.progressPercent.emit(pct, msg)
        self.progressStatus.emit(msg)

    def run(self):
        from app.core.sam_manager import get_sam_manager
        try:
            self.setPriority(QThread.Priority.LowPriority)
            self._emit_progress(5, "Preparando Varita IA (MobileSAM)...")
            sam_mgr = get_sam_manager()
            success = sam_mgr.prepare_image(
                self.image, image_id=self.image_id, progress_cb=self._emit_progress
            )
            self.embeddingReady.emit(success)
            self.finishedResult.emit(success)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class YoloDetectionWorker(QThread):
    """
    QThread para descargar (si es necesario) y ejecutar YOLO11-seg
    en segundo plano sin congelar la interfaz gráfica.
    """
    progressStatus = Signal(str)
    progressPercent = Signal(int, str)
    finishedResult = Signal(list)  # List[DetectedSubject]
    failed = Signal(str)

    def __init__(self, image: Image.Image, conf_threshold: float = 0.35):
        super().__init__()
        self.image = image
        self.conf_threshold = conf_threshold

    def _emit_progress(self, percent: int, message: str):
        pct = max(0, min(100, int(percent)))
        msg = str(message)
        self.progressPercent.emit(pct, msg)
        self.progressStatus.emit(msg)

    def run(self):
        from app.core.yolo_detector import get_yolo_detector
        try:
            self._emit_progress(5, "Detectando sujetos con IA (YOLO11)...")
            detector = get_yolo_detector()
            subjects = detector.detect_subjects(
                self.image,
                conf_threshold=self.conf_threshold,
                progress_cb=self._emit_progress,
            )
            self.finishedResult.emit(subjects)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class GifWorkerThread(QThread):
    """
    QThread para eliminar el fondo de todos los fotogramas de un GIF animado o Video
    con estabilización temporal de bordes (anti-flicker).
    """
    progress = Signal(int, int)             # (frame_actual, total_frames)
    progressText = Signal(str)
    finishedGif = Signal(list, str)         # (frames_rgba_procesados, descripcion)
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

    def run(self):
        import numpy as np
        from app.core.bg_remover import get_bg_remover
        import traceback

        try:
            self.setPriority(QThread.Priority.LowPriority)
            remover = get_bg_remover()
            total = len(self.frames)
            processed_frames: List[Image.Image] = []
            prev_alpha: Optional[np.ndarray] = None

            for idx, frame in enumerate(self.frames):
                if self.isInterruptionRequested():
                    self.cancelled.emit()
                    return

                self.progressText.emit(
                    f"⏳ Quitando fondo al {self.media_label} ({idx + 1}/{total} fotogramas) con {self.model_name}..."
                )
                self.progress.emit(idx + 1, total)

                out_rgba = remover.remove_background(
                    frame,
                    model_name=self.model_name,
                    enable_clothing_protection=self.enable_clothing_protection,
                )

                # Estabilización temporal ligera entre fotogramas adyacentes para evitar parpadeo en bordes
                if self.temporal_smoothing > 0.0:
                    arr = np.array(out_rgba.convert("RGBA"), dtype=np.uint8)
                    curr_alpha = arr[:, :, 3].astype(np.float32)
                    if prev_alpha is not None and prev_alpha.shape == curr_alpha.shape:
                        # Solo suavizar donde los fotogramas sean similares (evita ghosting en movimientos rápidos)
                        diff = np.abs(curr_alpha - prev_alpha)
                        blend_mask = diff < 45.0
                        w = self.temporal_smoothing
                        curr_alpha = np.where(
                            blend_mask,
                            (1.0 - w) * curr_alpha + w * prev_alpha,
                            curr_alpha,
                        )
                        arr[:, :, 3] = np.clip(np.round(curr_alpha), 0, 255).astype(np.uint8)
                        out_rgba = Image.fromarray(arr, mode="RGBA")
                    prev_alpha = curr_alpha

                processed_frames.append(out_rgba)

            self.finishedGif.emit(
                processed_frames,
                f"GIF sin fondo ({total} frames • {self.model_name})",
            )
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))



class SequenceRestoreWorker(QThread):
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

    def run(self):
        from app.core.restorer import get_image_restorer
        import traceback

        try:
            self.setPriority(QThread.Priority.LowPriority)
            restorer = get_image_restorer()
            total = len(self.frames)
            processed_frames: List[Image.Image] = []

            for idx, frame in enumerate(self.frames):
                if self.isInterruptionRequested():
                    self.cancelled.emit()
                    return
                self.progressText.emit(
                    f"⏳ {self.description} {self.media_label} ({idx + 1}/{total})..."
                )
                self.progress.emit(idx + 1, total)
                out_img = restorer.restore_image(
                    frame,
                    scale_mode=self.scale_mode,
                    use_tiling=self.use_tiling,
                )
                processed_frames.append(out_img)

            self.finishedGif.emit(processed_frames, f"{self.description} ({total} frames)")
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))



class SaveSequenceWorker(QThread):
    """QThread para exportar secuencias animadas o video sin congelar la interfaz."""

    saved = Signal(object)  # Path
    failed = Signal(str)

    def __init__(self, save_fn: Callable[[], Any]):
        super().__init__()
        self._save_fn = save_fn

    def run(self):
        import traceback

        try:
            self.setPriority(QThread.Priority.LowPriority)
            saved_path = self._save_fn()
            self.saved.emit(saved_path)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))



class ModelActionWorker(QThread):
    """QThread para descargar, cuantizar INT8 o verificar SHA-256 de un modelo."""

    progressPercent = Signal(int)
    progressText = Signal(str)
    finishedAction = Signal(str, str)  # (filename, summary_msg)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, action: str, filename: str):
        super().__init__()
        self.action = action
        self.filename = filename

    def run(self):
        import traceback
        from app.config import MODELS_DIR
        from app.core.model_fetcher import ensure_model_file, verify_local_model_sha256

        try:
            self.setPriority(QThread.Priority.LowPriority)
            if self.action == "download":
                def _prog(dl: int, tot: int):
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
                src = MODELS_DIR / self.filename
                if not src.is_file():
                    raise FileNotFoundError(self.filename)
                ok = quantize_model(src)
                if not ok:
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
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))





class SequenceTrackWorker(QThread):
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

    def run(self):
        import traceback
        from app.core.sequence_tracker import track_subject_across_sequence

        try:
            self.setPriority(QThread.Priority.LowPriority)

            def _prog(pct: int, msg: str):
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
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
