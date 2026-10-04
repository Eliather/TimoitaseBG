"""
TimoitaseBG - Proceso persistente de inferencia dedicado a GPU (gpu_worker).
Se ejecuta como un multiprocessing.Process aislado para proteger el proceso principal
y el sistema operativo contra deadlocks de drivers D3D12/DirectML y picos de VRAM.
"""
import os
import gc
import sys
import time
import logging
import traceback
from typing import Dict, Any, Optional

import numpy as np
import cv2
from PIL import Image

from app.config import APP_DIR, get_gpu_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [GPUWorker:%(process)d] %(message)s",
)
logger = logging.getLogger("TimoitaseBG.GPUWorker")


def get_worker_gpu_providers(preferred_device_id: Optional[int] = None):
    """
    Retorna los proveedores GPU para ONNX Runtime en el worker aislado.
    Prioriza CUDA o DirectML con el device_id indicado o detectado de forma segura.
    """
    try:
        if preferred_device_id is None:
            from app.config import get_best_dml_device_id
            preferred_device_id = get_best_dml_device_id()
        import onnxruntime as ort
        avail = ort.get_available_providers()
        selected = []
        if "CUDAExecutionProvider" in avail:
            selected.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in avail:
            selected.append(("DmlExecutionProvider", {"device_id": preferred_device_id}))
        selected.append("CPUExecutionProvider")
        return selected
    except Exception as e:
        logger.warning(f"Error detectando providers en worker: {e}. Usando CPU...")
        return ["CPUExecutionProvider"]


class PersistentGPUWorker:
    """
    Gestiona el ciclo de vida de los modelos de IA en GPU dentro del proceso worker aislado.
    """

    def __init__(self, preferred_device_id: Optional[int] = None):
        self.device_id = preferred_device_id
        self._cached_sessions: Dict[str, Any] = {}
        self._restorer_session: Optional[Any] = None
        self._restorer_instance: Optional[Any] = None

    def _get_providers(self, override_providers=None):
        if override_providers is not None:
            return override_providers
        return get_worker_gpu_providers(self.device_id)

    def _clear_rembg_sessions(self):
        """Libera explícitamente todas las sesiones y fuerza recolección de basura."""
        for name in list(self._cached_sessions.keys()):
            try:
                del self._cached_sessions[name]
            except Exception:
                pass
        self._cached_sessions.clear()

        if self._restorer_instance is not None:
            try:
                del self._restorer_instance
            except Exception:
                pass
            self._restorer_instance = None

        if self._restorer_session is not None:
            try:
                del self._restorer_session
            except Exception:
                pass
            self._restorer_session = None

        gc.collect()

    def get_bg_session(self, model_name: str = "inspyrenet", providers=None):
        """Compatibilidad con get_bg_session: redirige a la sesión de InSPyReNet."""
        return self.get_inspyrenet_session(providers=providers)

    def get_inspyrenet_session(self, providers=None):
        """Obtiene o inicializa una sesión ONNX para InSPyReNet en GPU Worker."""
        if "inspyrenet" not in self._cached_sessions:
            if self._cached_sessions:
                logger.info("Liberando sesiones previas para cargar inspyrenet...")
                self._clear_rembg_sessions()

            from app.core.model_fetcher import ensure_model_file
            model_path = ensure_model_file("inspyrenet.onnx")
            import onnxruntime as ort
            eff_providers = self._get_providers(providers)
            logger.info(f"Cargando sesión inspyrenet con providers: {eff_providers}...")
            self._cached_sessions["inspyrenet"] = ort.InferenceSession(str(model_path), providers=eff_providers)
        return self._cached_sessions["inspyrenet"]

    def remove_background(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        providers=None,
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
    ) -> Image.Image:
        """
        Ejecuta la eliminación de fondo en el proceso aislado de GPU con fidelidad cromática 100% íntegra.
        - 'isnet-anime': Modelo especializado en Anime/Manga, cel-shading y mangas/ropa blanca.
        - 'u2net_human_seg': Detector semántico de personas y prendas (sin fallos en blanco).
        - 'birefnet-general': BiRefNet para alta definición general.
        - 'inspyrenet': InSPyReNet Swin-B (1024x1024) para fotografía y personas reales con anti-aliasing.
        """
        if image.mode not in ("RGB", "RGBA"):
            input_img = image.convert("RGB")
        else:
            input_img = image.copy()

        eff_providers = self._get_providers(providers)

        if model_name in ("isnet-anime", "birefnet-general", "u2net_human_seg"):
            import rembg
            logger.info(f"[GPUWorker:{os.getpid()}] Ejecutando {model_name} en GPU con providers: {eff_providers}...")
            if model_name not in self._cached_sessions:
                if self._cached_sessions:
                    self._clear_rembg_sessions()
                self._cached_sessions[model_name] = rembg.new_session(model_name, providers=eff_providers)

            session = self._cached_sessions[model_name]
            out = rembg.remove(input_img, session=session)
            final_alpha = np.array(out.split()[-1], dtype=np.uint8)

        else:
            from app.core.bg_remover import (
                _preprocess_inspyrenet,
                _postprocess_inspyrenet_mask,
                postprocess_inspyrenet,
            )
            logger.info(
                f"[GPUWorker:{os.getpid()}] Ejecutando InSPyReNet (Swin-B 1024x1024)..."
            )
            session = self.get_inspyrenet_session(providers=providers)
            tensor, orig_size = _preprocess_inspyrenet(input_img)
            input_name = session.get_inputs()[0].name
            out = session.run(None, {input_name: tensor})[0]
            raw_mask = _postprocess_inspyrenet_mask(out, orig_size)
            cleaned_rgb, final_alpha = postprocess_inspyrenet(input_img, raw_mask)
            input_img = Image.fromarray(cleaned_rgb, mode="RGB")

        # Fusión con Detector Semántico de Ropa y Sujetos (evita agujeros en playeras blancas)
        if enable_clothing_protection and model_name != "u2net_human_seg":
            try:
                import rembg
                human_sess_key = "u2net_human_seg"
                if human_sess_key not in self._cached_sessions:
                    self._cached_sessions[human_sess_key] = rembg.new_session(human_sess_key, providers=eff_providers)
                human_sess = self._cached_sessions[human_sess_key]
                human_out = rembg.remove(input_img, session=human_sess)
                human_alpha = np.array(human_out.split()[-1], dtype=np.uint8)
                if np.any(human_alpha > 50):
                    # Proteger áreas de prendas de vestir confirmadas por el detector semántico
                    final_alpha = np.maximum(final_alpha, human_alpha)
            except Exception as e:
                logger.warning(f"Protección de ropa u2net_human_seg omitida en GPU: {e}")

        if image.mode == "RGBA":
            orig_alpha = np.array(image.split()[-1], dtype=np.uint8)
            final_alpha = np.minimum(final_alpha, orig_alpha)

        output = input_img.convert("RGB").convert("RGBA")
        output.putalpha(Image.fromarray(final_alpha, mode="L"))
        return output

    def restore_image(
        self,
        image: Image.Image,
        scale_mode: str = "x2",
        use_tiling: bool = True,
        tile_size: int = 512,
        tile_pad: int = 16,
        providers=None,
    ) -> Image.Image:
        """
        Ejecuta restauración Real-ESRGAN en el proceso aislado de GPU.
        """
        from app.core.restorer import ImageRestorer

        if self._restorer_instance is None:
            self._restorer_instance = ImageRestorer()
            eff_providers = self._get_providers(providers)
            # Configurar sesión con providers en worker
            import onnxruntime as ort
            from app.core.model_fetcher import ensure_model_file
            model_path = ensure_model_file(ImageRestorer.MODEL_FILENAME)
            logger.info(f"Worker: Inicializando Real-ESRGAN con providers: {eff_providers}...")
            self._restorer_instance._session = ort.InferenceSession(str(model_path), providers=eff_providers)

        return self._restorer_instance.restore_image(
            image,
            scale_mode=scale_mode,
            use_tiling=use_tiling,
            tile_size=tile_size,
            tile_pad=tile_pad,
        )


def gpu_worker_process_loop(request_queue, response_queue, preferred_device_id: Optional[int] = None):
    """
    Bucle principal del proceso persistente de GPU.
    Lee peticiones desde request_queue, ejecuta la tarea en GPU y retorna el resultado
    en response_queue.
    """
    if preferred_device_id is None:
        from app.config import get_best_dml_device_id
        preferred_device_id = get_best_dml_device_id()

    logger.info(f"Proceso GPU Worker iniciado (device_id={preferred_device_id}) y a la espera de peticiones...")
    worker = PersistentGPUWorker(preferred_device_id=preferred_device_id)

    while True:
        try:
            req = request_queue.get()
            if req is None or req.get("command") == "shutdown":
                logger.info("Recibida orden de apagado. Saliendo del worker...")
                worker._clear_rembg_sessions()
                break

            task_id = req.get("task_id", "")
            command = req.get("command", "")
            payload = req.get("payload", {})

            # 1. Comando de verificación simple
            if command == "ping":
                response_queue.put({"task_id": task_id, "status": "ok", "result": "pong"})
                continue

            # 2. Comando de prueba de cuelgue simulado
            if command == "simulate_hang":
                duration = float(payload.get("duration", 60.0))
                logger.warning(f"Simulando cuelgue deliberado por {duration} segundos...")
                time.sleep(duration)
                response_queue.put({"task_id": task_id, "status": "ok", "result": "completed_after_hang"})
                continue

            # 3. Quitar Fondo (individual o ensemble)
            if command == "remove_background":
                image = payload["image"]
                model_name = payload.get("model_name", "isnet-anime")
                providers = payload.get("providers", None)
                ensemble_weight = payload.get("ensemble_weight", 0.5)
                refine_matting = payload.get("refine_matting", False)
                enable_clothing_protection = payload.get("enable_clothing_protection", True)
                result_img = worker.remove_background(
                    image,
                    model_name=model_name,
                    providers=providers,
                    ensemble_weight=ensemble_weight,
                    refine_matting=refine_matting,
                    enable_clothing_protection=enable_clothing_protection,
                )
                response_queue.put({"task_id": task_id, "status": "ok", "result": result_img})
                continue

            # 4. Restauración / Escalado Real-ESRGAN
            if command == "restore_image":
                image = payload["image"]
                scale_mode = payload.get("scale_mode", "x2")
                use_tiling = payload.get("use_tiling", True)
                tile_size = payload.get("tile_size", 512)
                tile_pad = payload.get("tile_pad", 16)
                providers = payload.get("providers", None)
                result_img = worker.restore_image(
                    image,
                    scale_mode=scale_mode,
                    use_tiling=use_tiling,
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                    providers=providers,
                )
                response_queue.put({"task_id": task_id, "status": "ok", "result": result_img})
                continue
            # Comando desconocido
            response_queue.put({
                "task_id": task_id,
                "status": "error",
                "error": f"Comando desconocido: {command}",
            })

        except Exception as e:
            logger.error(f"Error procesando tarea en GPU Worker: {e}")
            tb = traceback.format_exc()
            response_queue.put({
                "task_id": task_id if "task_id" in locals() else "",
                "status": "error",
                "error": str(e),
                "traceback": tb,
            })
