"""
TimoitaseBG - Hilos de ejecución en segundo plano para inferencia asíncrona no bloqueante.
"""
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
    finishedResult = Signal(object, str)  # (result_image, action_description)
    failed = Signal(str)                  # error_message

    def __init__(self, task_fn: Callable[..., Image.Image], *args, description: str = "Procesando", **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs
        self.description = description

    def run(self):
        try:
            self.setPriority(QThread.Priority.LowPriority)
            self.startedProcessing.emit(f"Iniciando {self.description}")
            result = self.task_fn(*self.args, **self.kwargs)
            self.finishedResult.emit(result, self.description)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))

class BatchWorkerThread(QThread):
    """
    QThread para procesar múltiples imágenes en lote secuencialmente.
    """
    progress = Signal(int, int) # actual, total
    progressText = Signal(str)
    finishedResult = Signal(int, int) # success, total
    failed = Signal(str)

    def __init__(self, input_dir: str, output_dir: str, model_name: str, enable_clothing_protection: bool):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.enable_clothing_protection = enable_clothing_protection

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
            
            if total == 0:
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
                    
            self.progressText.emit("Proceso completado")
            self.progress.emit(total, total)
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
    embeddingReady = Signal(bool)
    finishedResult = Signal(bool)
    failed = Signal(str)

    def __init__(self, image: Image.Image, image_id: Optional[int] = None):
        super().__init__()
        self.image = image
        self.image_id = image_id

    def run(self):
        from app.core.sam_manager import get_sam_manager
        try:
            self.setPriority(QThread.Priority.LowPriority)
            self.progressStatus.emit("Preparando Varita IA (MobileSAM)...")
            sam_mgr = get_sam_manager()
            success = sam_mgr.prepare_image(self.image, image_id=self.image_id)
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
    finishedResult = Signal(list)  # List[DetectedSubject]
    failed = Signal(str)

    def __init__(self, image: Image.Image, conf_threshold: float = 0.35):
        super().__init__()
        self.image = image
        self.conf_threshold = conf_threshold

    def run(self):
        from app.core.yolo_detector import get_yolo_detector
        try:
            self.progressStatus.emit("Detectando sujetos con IA (YOLO11)...")
            detector = get_yolo_detector()
            subjects = detector.detect_subjects(self.image, conf_threshold=self.conf_threshold)
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
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))



