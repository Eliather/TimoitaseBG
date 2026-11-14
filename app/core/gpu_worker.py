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

from app.config import (
    APP_DIR,
    MODELS_DIR,
    configure_ssl_certificates,
    get_gpu_name,
    get_optimized_session_options,
    update_u2net_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [GPUWorker:%(process)d] %(message)s",
)
logger = logging.getLogger("TimoitaseBG.GPUWorker")


def get_worker_gpu_providers(preferred_device_id: Optional[int] = None):
    """
    Retorna los proveedores GPU para ONNX Runtime en el worker aislado,
    añadiendo soporte multiplataforma para NVIDIA (CUDA), AMD (ROCm / DirectML),
    Qualcomm QNN (NPU en Windows ARM64), Apple Silicon/macOS (CoreML) e Intel.
    """
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        selected = []

        # 1. Soporte NVIDIA CUDA (Linux / Windows / Jetson)
        if "CUDAExecutionProvider" in avail:
            selected.append("CUDAExecutionProvider")

        # 2. Soporte AMD ROCm (Principalmente Linux)
        if "ROCMExecutionProvider" in avail:
            selected.append(("ROCMExecutionProvider", {"device_id": preferred_device_id or 0}))

        # 3. Soporte Qualcomm QNN (NPU Hexagon en Windows ARM64)
        if "QNNExecutionProvider" in avail:
            selected.append("QNNExecutionProvider")

        # 4. Soporte Apple Silicon / macOS CoreML
        if "CoreMLExecutionProvider" in avail:
            selected.append("CoreMLExecutionProvider")

        # 5. Soporte Windows DirectML (Compatible universal con AMD Radeon, Intel Arc/Iris, NVIDIA y Qualcomm Adreno)
        if "DmlExecutionProvider" in avail:
            if preferred_device_id is None:
                from app.config import get_best_dml_device_id
                preferred_device_id = get_best_dml_device_id()
            device_id = preferred_device_id if preferred_device_id is not None else 0
            selected.append(("DmlExecutionProvider", {"device_id": device_id}))

        selected.append("CPUExecutionProvider")
        return selected
    except Exception as e:
        logger.warning(f"Error detectando providers multiplataforma en worker: {e}. Usando CPU...")
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

    def _clear_rembg_sessions(self, preserve_clothing_session: bool = False):
        """Libera explícitamente las sesiones en caché y fuerza recolección de basura."""
        from app.config import is_integrated_gpu

        keep_clothing = preserve_clothing_session and (not is_integrated_gpu())
        for name in list(self._cached_sessions.keys()):
            if keep_clothing and name == "u2net_human_seg":
                continue
            try:
                del self._cached_sessions[name]
            except Exception:
                pass
        if not keep_clothing:
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

    def _create_ort_session_with_fallback(self, model_path, eff_providers, model_label: str):
        """Crea sesión ONNX Runtime haciendo fallback limpio QNN -> DirectML -> CPU."""
        import onnxruntime as ort

        sess_options = get_optimized_session_options() or ort.SessionOptions()
        try:
            return ort.InferenceSession(str(model_path), sess_options, providers=eff_providers)
        except Exception as e:
            logger.warning(
                "Fallo al cargar %s con proveedores GPU/CoreML/QNN (%s): %s. "
                "Conmutando limpiamente a DML/CPUExecutionProvider...",
                model_label,
                eff_providers,
                e,
            )
            if any(p == "QNNExecutionProvider" for p in eff_providers) and "DmlExecutionProvider" in ort.get_available_providers():
                try:
                    dml_provs = [("DmlExecutionProvider", {"device_id": self.device_id or 0}), "CPUExecutionProvider"]
                    logger.info("Reintentando %s con DirectML tras fallo en QNN...", model_label)
                    return ort.InferenceSession(str(model_path), sess_options, providers=dml_provs)
                except Exception as dml_err:
                    logger.warning("Fallback DirectML para %s falló (%s). Usando CPU...", model_label, dml_err)
            return ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])

    def get_inspyrenet_session(
        self,
        providers=None,
        force_cpu: bool = False,
        progress_cb=None,
    ):
        """Obtiene o inicializa una sesión ONNX para InSPyReNet en GPU Worker."""
        if force_cpu:
            self._cached_sessions.pop("inspyrenet", None)
            providers = ["CPUExecutionProvider"]

        if "inspyrenet" not in self._cached_sessions:
            if self._cached_sessions or self._restorer_instance is not None:
                logger.info("Liberando sesiones previas de VRAM para cargar inspyrenet...")
                self._clear_rembg_sessions(preserve_clothing_session=True)

            from app.core.model_fetcher import ensure_model_file, make_download_progress_cb

            if progress_cb:
                progress_cb(5, "Verificando modelo InSPyReNet...")

            model_path = ensure_model_file(
                "inspyrenet.onnx",
                progress_cb=make_download_progress_cb(progress_cb, "inspyrenet.onnx", start_pct=5, span_pct=30),
            )
            eff_providers = self._get_providers(providers)
            if progress_cb:
                progress_cb(38, "Cargando modelo InSPyReNet en memoria...")
            logger.info(f"Cargando sesión inspyrenet con providers: {eff_providers}...")
            self._cached_sessions["inspyrenet"] = self._create_ort_session_with_fallback(
                model_path, eff_providers, "sesión inspyrenet"
            )
        return self._cached_sessions["inspyrenet"]

    def remove_background(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        providers=None,
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
        progress_cb=None,
    ) -> Image.Image:
        """
        Ejecuta la eliminación de fondo en el proceso aislado de GPU con fidelidad cromática 100% íntegra.
        - 'isnet-anime': Modelo especializado en Anime/Manga, cel-shading y mangas/ropa blanca.
        - 'birefnet-general': BiRefNet (Dicotomic SOTA 1024x1024 MIT) para alta definición general y bordes complejos.
        - 'u2net_human_seg': Detector semántico de personas y prendas (sin fallos en blanco).
        - 'inspyrenet': InSPyReNet Swin-B (1024x1024) para fotografía y personas reales con anti-aliasing.
        """
        if progress_cb:
            progress_cb(5, f"Preparando imagen para {model_name}...")

        if image.mode not in ("RGB", "RGBA"):
            input_img = image.convert("RGB")
        else:
            input_img = image.copy()

        eff_providers = self._get_providers(providers)

        if model_name in ("isnet-anime", "birefnet-general", "u2net_human_seg"):
            import rembg
            from app.core.model_fetcher import load_rembg_session
            logger.info(f"[GPUWorker:{os.getpid()}] Ejecutando {model_name} en GPU con providers: {eff_providers}...")
            if model_name not in self._cached_sessions:
                if progress_cb:
                    progress_cb(38, f"Cargando modelo {model_name} en memoria...")
                if self._cached_sessions or self._restorer_instance is not None:
                    self._clear_rembg_sessions(preserve_clothing_session=True)
                sess_opts = get_optimized_session_options()
                try:
                    self._cached_sessions[model_name] = load_rembg_session(
                        model_name,
                        providers=eff_providers,
                        sess_opts=sess_opts,
                        progress_cb=progress_cb,
                    )
                except Exception as e:
                    logger.warning(
                        f"Fallo al cargar sesión rembg '{model_name}' con proveedores GPU/CoreML/QNN: {e}. "
                        f"Conmutando a CPUExecutionProvider..."
                    )
                    self._cached_sessions[model_name] = load_rembg_session(
                        model_name,
                        providers=["CPUExecutionProvider"],
                        sess_opts=sess_opts,
                        progress_cb=progress_cb,
                    )

            session = self._cached_sessions[model_name]
            if progress_cb:
                progress_cb(50, f"Ejecutando inferencia con {model_name}...")
            try:
                out = rembg.remove(input_img, session=session)
            except Exception as run_err:
                logger.warning(
                    f"Fallo en inferencia rembg '{model_name}' con acelerador ({run_err}). "
                    f"Reintentando en CPUExecutionProvider..."
                )
                session = load_rembg_session(
                    model_name,
                    providers=["CPUExecutionProvider"],
                    sess_opts=get_optimized_session_options(),
                )
                self._cached_sessions[model_name] = session
                out = rembg.remove(input_img, session=session)
            final_alpha = np.array(out.getchannel("A"), dtype=np.uint8)
            if progress_cb:
                progress_cb(80, "Máscara principal generada...")

        else:
            from app.core.bg_remover import (
                _preprocess_inspyrenet,
                _postprocess_inspyrenet_mask,
                postprocess_inspyrenet,
            )
            logger.info(
                f"[GPUWorker:{os.getpid()}] Ejecutando InSPyReNet (Swin-B 1024x1024)..."
            )
            session = self.get_inspyrenet_session(providers=providers, progress_cb=progress_cb)
            if progress_cb:
                progress_cb(45, "Preprocesando tensor 1024×1024...")
            tensor, orig_size = _preprocess_inspyrenet(input_img)
            input_name = session.get_inputs()[0].name
            if progress_cb:
                progress_cb(55, "Ejecutando red neuronal InSPyReNet (Swin-B)...")
            try:
                out = session.run(None, {input_name: tensor})[0]
            except Exception as run_err:
                logger.warning(
                    f"Fallo durante la inferencia de InSPyReNet ({run_err}). "
                    f"Reintentando ejecución en CPUExecutionProvider..."
                )
                if progress_cb:
                    progress_cb(60, "Conmutando inferencia InSPyReNet a CPU...")
                session = self.get_inspyrenet_session(force_cpu=True, progress_cb=progress_cb)
                out = session.run(None, {input_name: tensor})[0]
            if progress_cb:
                progress_cb(78, "Refinando bordes subpíxel...")
            raw_mask = _postprocess_inspyrenet_mask(out, orig_size)
            cleaned_rgb, final_alpha = postprocess_inspyrenet(input_img, raw_mask)
            input_img = Image.fromarray(cleaned_rgb, mode="RGB")

        def _get_gpu_human_session():
            from app.core.model_fetcher import load_rembg_session

            human_sess_key = "u2net_human_seg"
            if human_sess_key not in self._cached_sessions:
                sess_opts = get_optimized_session_options()
                try:
                    self._cached_sessions[human_sess_key] = load_rembg_session(
                        human_sess_key, providers=eff_providers, sess_opts=sess_opts
                    )
                except Exception as e:
                    logger.warning(
                        "Fallo al cargar protección u2net_human_seg con GPU/CoreML/QNN: %s. "
                        "Conmutando a CPUExecutionProvider...",
                        e,
                    )
                    self._cached_sessions[human_sess_key] = load_rembg_session(
                        human_sess_key,
                        providers=["CPUExecutionProvider"],
                        sess_opts=sess_opts,
                    )
            return self._cached_sessions[human_sess_key]

        from app.core.bg_remover import finalize_removed_background

        return finalize_removed_background(
            orig_image=image,
            input_img=input_img,
            final_alpha=final_alpha,
            model_name=model_name,
            enable_clothing_protection=enable_clothing_protection,
            get_human_session_fn=_get_gpu_human_session,
            progress_cb=progress_cb,
            env_label="GPU",
        )

    def restore_image(
        self,
        image: Image.Image,
        scale_mode: str = "x2",
        use_tiling: bool = True,
        tile_size: int = 512,
        tile_pad: int = 16,
        providers=None,
        progress_cb=None,
    ) -> Image.Image:
        """
        Ejecuta restauración Real-ESRGAN en el proceso aislado de GPU.
        Libera previamente las sesiones de segmentación para evitar picos de VRAM (>4GB).
        """
        from app.core.restorer import ImageRestorer

        if self._restorer_instance is None:
            if self._cached_sessions:
                logger.info("Worker: Liberando sesiones de segmentación para cargar Real-ESRGAN...")
                self._clear_rembg_sessions()

            self._restorer_instance = ImageRestorer()
            eff_providers = self._get_providers(providers)
            from app.core.model_fetcher import ensure_model_file, make_download_progress_cb

            if progress_cb:
                progress_cb(5, "Verificando modelo Real-ESRGAN...")

            model_path = ensure_model_file(
                ImageRestorer.MODEL_FILENAME,
                progress_cb=make_download_progress_cb(progress_cb, "Real-ESRGAN", start_pct=5, span_pct=12),
            )
            if progress_cb:
                progress_cb(18, "Cargando modelo Real-ESRGAN en memoria...")
            logger.info(f"Worker: Inicializando Real-ESRGAN con providers: {eff_providers}...")
            self._restorer_instance._session = self._create_ort_session_with_fallback(
                model_path, eff_providers, "Real-ESRGAN"
            )

        try:
            return self._restorer_instance._restore_image_impl(
                image,
                scale_mode=scale_mode,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_pad=tile_pad,
                force_cpu=False,
                progress_cb=progress_cb,
            )
        except Exception as restorer_err:
            logger.warning(
                f"Fallo en ejecución de Real-ESRGAN con acelerador hardware ({restorer_err}). "
                f"Conmutando automáticamente a CPUExecutionProvider..."
            )
            self._restorer_instance._session = None
            return self._restorer_instance._restore_image_impl(
                image,
                scale_mode=scale_mode,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_pad=tile_pad,
                force_cpu=True,
                progress_cb=progress_cb,
            )


def gpu_worker_process_loop(request_queue, response_queue, preferred_device_id: Optional[int] = None):
    """
    Bucle principal del proceso persistente de GPU.
    Lee peticiones desde request_queue, ejecuta la tarea en GPU y retorna el resultado
    en response_queue.
    """
    # Sincronización explícita de U2NET_HOME y certificados SSL en el subproceso 'spawn'
    update_u2net_env(MODELS_DIR)
    configure_ssl_certificates()

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

            def _emit_progress(percent: int, message: str):
                try:
                    response_queue.put({
                        "task_id": task_id,
                        "status": "progress",
                        "percent": int(percent),
                        "message": str(message),
                    })
                except Exception:
                    pass

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
                    progress_cb=_emit_progress,
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
                    progress_cb=_emit_progress,
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
