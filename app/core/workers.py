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

            # Encontrar imágenes
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
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
                    img = Image.open(f)
                    # Convertimos a RGBA de inmediato para homogeneizar
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                        
                    res = remover.remove_background(
                        img, 
                        model_name=self.model_name, 
                        enable_clothing_protection=self.enable_clothing_protection
                    )
                    
                    # Guardar archivo
                    out_file = output_path / f"{f.stem}_bgrm.png"
                    res.save(out_file)
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


