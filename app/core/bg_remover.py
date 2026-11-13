"""
TimoitaseBG - Motor de eliminación de fondos y orquestación de sesiones ONNX / rembg.
"""
import logging
import threading
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from app.config import get_inference_providers, get_optimized_session_options
from app.core.postprocess import (  # noqa: F401 (re-exports de compatibilidad)
    _BILINEAR,
    _LANCZOS,
    _guided_filter_box,
    _letterbox_topleft,
    _postprocess_inspyrenet_mask,
    _predict_clothing_mask,
    _preprocess_inspyrenet,
    _unletterbox_topleft,
    apply_guided_filter,
    apply_guided_filter_ambiguous,
    fuse_clothing_protection_core_only,
    generate_trimap,
    letterbox_image,
    postprocess_inspyrenet,
    postprocess_mask,
    unletterbox_mask,
)

logger = logging.getLogger("TimoitaseBG.BGRemover")


class BackgroundRemover:
    """
    Motor de eliminación de fondo basado exclusivamente en InSPyReNet Swin-B (1024x1024)
    con postprocesamiento subpíxel bilateral y despill cromático estilo LocalBG.
    """

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_inspyrenet_session(self, force_cpu: bool = False, progress_cb=None):
        """Obtiene o inicializa una sesión ONNX para el modelo InSPyReNet (Swin-B 1024x1024)."""
        cache_key = "inspyrenet_cpu" if force_cpu else "inspyrenet"
        if cache_key not in self._sessions:
            with self._lock:
                if cache_key not in self._sessions:
                    from app.core.model_fetcher import ensure_model_file

                    if progress_cb:
                        progress_cb(5, "Verificando modelo InSPyReNet...")

                    def _dl_prog(dl: int, tot: int):
                        if progress_cb and tot > 0:
                            pct = 5 + int(round((dl / tot) * 30))
                            dl_mb = dl / (1024 * 1024)
                            tot_mb = tot / (1024 * 1024)
                            progress_cb(
                                min(35, pct),
                                f"Descargando inspyrenet.onnx ({dl_mb:.1f} / {tot_mb:.1f} MB)...",
                            )

                    model_path = ensure_model_file("inspyrenet.onnx", progress_cb=_dl_prog)
                    import onnxruntime as ort
                    providers = ["CPUExecutionProvider"] if force_cpu else get_inference_providers()
                    sess_options = get_optimized_session_options() or ort.SessionOptions()
                    if progress_cb:
                        progress_cb(38, "Cargando modelo InSPyReNet en memoria...")
                    try:
                        self._sessions[cache_key] = ort.InferenceSession(str(model_path), sess_options, providers=providers)
                    except Exception as e:
                        logger.exception(f"No se pudo inicializar inspyrenet con GPU ({e}). Usando CPU...")
                        self._sessions[cache_key] = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
        return self._sessions[cache_key]

    def get_session(self, model_name: str = "inspyrenet", force_cpu: bool = False):
        """Compatibilidad con get_session: redirige a la sesión correspondiente."""
        return self.get_inspyrenet_session(force_cpu=force_cpu)

    def _predict_inspyrenet(self, session, input_img: Image.Image) -> np.ndarray:
        """
        Ejecuta inferencia con InSPyReNet (Swin-B 1024x1024):
        1. Resize directo 1024x1024 bilineal y normalización ImageNet.
        2. Inferencia ONNX.
        3. Resize bicúbico del mapa continuo a dimensiones originales (orig_w, orig_h).
        """
        tensor, orig_size = _preprocess_inspyrenet(input_img)
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: tensor})[0]
        return _postprocess_inspyrenet_mask(out, orig_size)

    def _get_rembg_cpu_session(self, model_name: str, progress_cb=None):
        """Obtiene o reutiliza en caché una sesión de rembg en CPU evitando recargas en lotes."""
        cache_key = f"rembg_cpu_{model_name}"
        if cache_key not in self._sessions:
            with self._lock:
                if cache_key not in self._sessions:
                    from app.core.model_fetcher import load_rembg_session

                    if progress_cb:
                        progress_cb(38, f"Inicializando sesión {model_name} en CPU...")
                    self._sessions[cache_key] = load_rembg_session(
                        model_name,
                        providers=["CPUExecutionProvider"],
                        sess_opts=get_optimized_session_options(),
                        progress_cb=progress_cb,
                    )
        return self._sessions[cache_key]

    def _remove_background_cpu_local(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
        progress_cb=None,
    ) -> Image.Image:
        """
        Ejecución local en CPU con preservación 100% íntegra de colores originales.
        """
        if progress_cb:
            progress_cb(5, f"Preparando imagen para {model_name}...")

        if image.mode not in ("RGB", "RGBA"):
            input_img = image.convert("RGB")
        else:
            input_img = image.copy()

        if model_name in ("isnet-anime", "birefnet-general", "u2net_human_seg"):
            import rembg
            logger.info(f"[CPU Fallback/Local] Ejecutando {model_name} en CPU...")
            sess = self._get_rembg_cpu_session(model_name, progress_cb=progress_cb)
            if progress_cb:
                progress_cb(50, f"Ejecutando inferencia con {model_name}...")
            out = rembg.remove(input_img, session=sess)
            final_alpha = np.array(out.getchannel("A"), dtype=np.uint8)
            if progress_cb:
                progress_cb(80, "Máscara principal generada...")
        else:
            logger.info("[CPU Fallback/Local] Ejecutando InSPyReNet (Swin-B 1024x1024)...")
            session = self.get_inspyrenet_session(force_cpu=True, progress_cb=progress_cb)
            if progress_cb:
                progress_cb(50, "Ejecutando red neuronal InSPyReNet (Swin-B)...")
            raw_mask = self._predict_inspyrenet(session, input_img)
            if progress_cb:
                progress_cb(78, "Refinando bordes subpíxel...")
            cleaned_rgb, final_alpha = postprocess_inspyrenet(input_img, raw_mask)
            input_img = Image.fromarray(cleaned_rgb, mode="RGB")

        # Fusión Core-Only con Detector Semántico de Ropa (se omite en isnet-anime y u2net_human_seg)
        if enable_clothing_protection and model_name not in ("u2net_human_seg", "isnet-anime"):
            try:
                if progress_cb:
                    progress_cb(85, "Aplicando protección semántica de ropa...")
                human_sess = self._get_rembg_cpu_session("u2net_human_seg")
                human_alpha = _predict_clothing_mask(input_img, human_sess)
                if np.any(human_alpha > 50):
                    final_alpha = fuse_clothing_protection_core_only(
                        main_alpha=final_alpha,
                        clothing_alpha=human_alpha,
                        inner_erosion_px=12,
                    )
            except Exception as e:
                logger.warning(f"Protección de ropa u2net_human_seg omitida en CPU: {e}")

        if progress_cb:
            progress_cb(95, "Componiendo canal alfa final...")

        if image.mode == "RGBA":
            orig_alpha = np.array(image.getchannel("A"), dtype=np.uint8)
            final_alpha = np.minimum(final_alpha, orig_alpha)

        output = input_img.convert("RGB").convert("RGBA")
        output.putalpha(Image.fromarray(final_alpha, mode="L"))
        if progress_cb:
            progress_cb(100, "Eliminación de fondo completada.")
        return output

    def remove_background(
        self,
        image: Image.Image,
        model_name: str = "isnet-anime",
        ensemble_weight: float = 0.5,
        refine_matting: bool = False,
        enable_clothing_protection: bool = True,
        progress_cb=None,
    ) -> Image.Image:
        """
        Elimina el fondo de una imagen usando el modelo seleccionado (IS-Net Anime, U2-Net Human, BiRefNet o InSPyReNet).
        Si la GPU está activa, delega al proceso GPU Worker aislado con watchdog.
        Si la GPU está inactiva o no responde antes del timeout, ejecuta automáticamente en CPU.
        """
        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()

        if gpu_mgr.is_gpu_enabled():
            return gpu_mgr.run_inference(
                command="remove_background",
                payload={
                    "image": image,
                    "model_name": model_name,
                    "ensemble_weight": ensemble_weight,
                    "refine_matting": refine_matting,
                    "enable_clothing_protection": enable_clothing_protection,
                },
                cpu_fallback_fn=lambda: self._remove_background_cpu_local(
                    image,
                    model_name=model_name,
                    ensemble_weight=ensemble_weight,
                    refine_matting=refine_matting,
                    enable_clothing_protection=enable_clothing_protection,
                    progress_cb=progress_cb,
                ),
                progress_cb=progress_cb,
            )
        else:
            return self._remove_background_cpu_local(
                image,
                model_name=model_name,
                ensemble_weight=ensemble_weight,
                refine_matting=refine_matting,
                enable_clothing_protection=enable_clothing_protection,
                progress_cb=progress_cb,
            )


# Instancia singleton para reutilización de sesiones
_remover_instance: Optional[BackgroundRemover] = None


def get_bg_remover() -> BackgroundRemover:
    global _remover_instance
    if _remover_instance is None:
        _remover_instance = BackgroundRemover()
    return _remover_instance

