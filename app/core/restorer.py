"""
TimoitaseBG - Módulo de restauración y aumento de resolución (Real-ESRGAN Anime con ONNX Runtime).
"""
import math
import logging
import traceback
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from app.config import get_inference_providers, MODELS_DIR
from app.core.image_utils import separate_alpha, combine_alpha
from app.core.model_fetcher import ensure_model_file

logger = logging.getLogger("TimoitaseBG.Restorer")


def _probe_restorer_gpu_worker(model_path_str: str, providers_list: list) -> None:
    """
    Sonda aislada en subproceso independiente para validar que Real-ESRGAN
    puede inicializarse y ejecutar un tensor de prueba en la GPU sin colgar
    el driver D3D12/TDR ni congelar el sistema operativo.
    Libera explícitamente la memoria VRAM antes de salir.
    """
    import sys
    import gc
    import numpy as np
    session = None
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path_str, providers=providers_list)
        dummy_in = np.zeros((1, 3, 32, 32), dtype=np.float32)
        in_name = session.get_inputs()[0].name
        session.run(None, {in_name: dummy_in})
        # Liberar recursos y VRAM de inmediato
        del session
        gc.collect()
        sys.exit(0)
    except Exception:
        if session is not None:
            del session
        gc.collect()
        sys.exit(1)


class ImageRestorer:
    """
    Restaura y aumenta la resolución de imágenes estilo anime/ilustración usando Real-ESRGAN.
    Soporta aceleración DirectML/CUDA, descomposición en bloques (tiling) y preservación de canal alfa.
    """

    MODEL_FILENAME = "RealESRGAN_x4plus_anime_6B.onnx"

    def __init__(self):
        self._session = None
        self._cpu_session = None

    def _get_cpu_session(self):
        """Inicializa o retorna la sesión garantizada en CPU."""
        if self._cpu_session is None:
            import onnxruntime as ort
            from app.config import get_optimized_session_options
            model_path = ensure_model_file(self.MODEL_FILENAME)
            logger.info("Inicializando sesión local de Real-ESRGAN con CPUExecutionProvider...")
            sess_options = get_optimized_session_options() or ort.SessionOptions()
            self._cpu_session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
        return self._cpu_session

    def _get_session(self, force_cpu: bool = True):
        if not force_cpu and hasattr(self, "_session") and self._session is not None:
            return self._session
        return self._get_cpu_session()

    def restore_image(
        self,
        image: Image.Image,
        scale_mode: str = "x2",
        use_tiling: bool = True,
        tile_size: int = 512,
        tile_pad: int = 16,
    ) -> Image.Image:
        """
        Restaura la imagen aplicando Real-ESRGAN x4.
        Si la GPU está activa, delega al proceso GPU Worker aislado con watchdog.
        Si la GPU no responde, está deshabilitada, o la imagen excede el límite seguro, ejecuta automáticamente en CPU.
        """
        import multiprocessing
        
        orig_w, orig_h = image.size
        # Protección Automática Anti-Saturación de VRAM:
        # Límite seguro para GPU: aprox 2.07 megapíxeles (ej. 1920x1080 o 1440x1440).
        # Imágenes mayores a este límite fuerzan fallback automático a CPU
        # para evitar colapsar la VRAM de tarjetas estándar o disparar TDRs.
        MAX_GPU_PIXELS = 1920 * 1080
        total_pixels = orig_w * orig_h
        force_cpu_safety = total_pixels > MAX_GPU_PIXELS
        
        if force_cpu_safety:
            logger.info(f"Protección Anti-Saturación: Imagen de {orig_w}x{orig_h} ({total_pixels} px) excede límite seguro de GPU. Forzando fallback a CPU.")

        # Si ya se encuentra dentro del proceso aislado GPU Worker, ejecutar directamente
        # para no intentar re-despachar a otro subproceso daemon hijo.
        if multiprocessing.current_process().name != "MainProcess":
            return self._restore_image_impl(
                image, scale_mode, use_tiling, tile_size, tile_pad, force_cpu=False
            )

        from app.core.gpu_manager import get_gpu_manager
        gpu_mgr = get_gpu_manager()

        if gpu_mgr.is_gpu_enabled() and not force_cpu_safety:
            return gpu_mgr.run_inference(
                command="restore_image",
                payload={
                    "image": image,
                    "scale_mode": scale_mode,
                    "use_tiling": use_tiling,
                    "tile_size": tile_size,
                    "tile_pad": tile_pad,
                },
                cpu_fallback_fn=lambda: self._restore_image_impl(
                    image, scale_mode, use_tiling, tile_size, tile_pad, force_cpu=True
                ),
            )
        else:
            return self._restore_image_impl(
                image, scale_mode, use_tiling, tile_size, tile_pad, force_cpu=True
            )

    def _restore_image_impl(
        self,
        image: Image.Image,
        scale_mode: str = "x2",
        use_tiling: bool = True,
        tile_size: int = 512,
        tile_pad: int = 16,
        force_cpu: bool = True,
    ) -> Image.Image:
        orig_w, orig_h = image.size

        # Pre-auditoría de seguridad de tiles antes de procesar o transformar arrays
        if use_tiling:
            expected_tiles_x = math.ceil(orig_w / tile_size)
            expected_tiles_y = math.ceil(orig_h / tile_size)
            expected_total_tiles = expected_tiles_x * expected_tiles_y
            MAX_ALLOWED_TILES = 300
            if expected_total_tiles > MAX_ALLOWED_TILES:
                raise ValueError(
                    f"Imagen demasiado grande para procesar con tiling actual: "
                    f"se calcularon {expected_total_tiles} tiles ({expected_tiles_x}×{expected_tiles_y}) "
                    f"para resolución {orig_w}x{orig_h} px (máximo permitido: {MAX_ALLOWED_TILES}). "
                    f"Reduzca la resolución de la imagen antes de continuar."
                )

        rgb_img, alpha_channel = separate_alpha(image)

        session = self._get_session(force_cpu=force_cpu)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Convertir a array float32 [0.0, 1.0] con formato NCHW contiguo
        img_np = np.ascontiguousarray(np.array(rgb_img, dtype=np.float32) / 255.0)
        h, w, c = img_np.shape
        img_nchw = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))[np.newaxis, ...])

        # Decidir si procesar directamente o con tiling
        if not use_tiling or (h <= tile_size and w <= tile_size):
            out_tensor = session.run([output_name], {input_name: img_nchw})[0]
            out_4x = self._tensor_to_image(out_tensor[0])
        else:
            out_4x = self._process_tiled(session, img_nchw, tile_size, tile_pad, input_name, output_name)

        # Ajuste forzado de dimensiones del output 4x para evitar cualquier desfase por redondeo
        expected_4x_size = (orig_w * 4, orig_h * 4)
        if out_4x.size != expected_4x_size:
            out_4x = out_4x.resize(expected_4x_size, resample=Image.Resampling.LANCZOS)

        # Si había canal alfa original, redimensionarlo forzadamente a las dimensiones EXACTAS de out_4x
        if alpha_channel is not None:
            alpha_4x = alpha_channel.resize(out_4x.size, resample=Image.Resampling.LANCZOS)
            out_4x = combine_alpha(out_4x, alpha_4x)

        # Aplicar modo de escala solicitado
        if scale_mode == "x4":
            result = out_4x
        elif scale_mode == "x2":
            target_size = (orig_w * 2, orig_h * 2)
            result = out_4x.resize(target_size, resample=Image.Resampling.LANCZOS)
        elif scale_mode == "enhance_only":
            target_size = (orig_w, orig_h)
            result = out_4x.resize(target_size, resample=Image.Resampling.LANCZOS)
        else:
            result = out_4x

        # Asegurar consistencia final del canal alfa
        if alpha_channel is not None and result.mode != "RGBA":
            target_alpha = alpha_channel.resize(result.size, resample=Image.Resampling.LANCZOS)
            result = combine_alpha(result, target_alpha)

        return result

    def _process_tiled(
        self,
        session,
        img_nchw: np.ndarray,
        tile_size: int,
        tile_pad: int,
        input_name: str,
        output_name: str,
    ) -> Image.Image:
        """Procesa una imagen grande dividiéndola en bloques solapados contiguos para evitar costuras y picos de memoria."""
        _, _, h, w = img_nchw.shape
        scale = 4

        out_h = h * scale
        out_w = w * scale
        output_accum = np.zeros((3, out_h, out_w), dtype=np.float32)
        weight_accum = np.zeros((1, out_h, out_w), dtype=np.float32)

        tiles_x = math.ceil(w / tile_size)
        tiles_y = math.ceil(h / tile_size)
        total_tiles = tiles_x * tiles_y

        logger.info(
            f"Auditoría de bloques: imagen {w}x{h} px, tile_size={tile_size}, tile_pad={tile_pad}. "
            f"Cuadrícula: {tiles_x} cols × {tiles_y} filas = {total_tiles} tiles en total."
        )

        MAX_ALLOWED_TILES = 300
        if total_tiles > MAX_ALLOWED_TILES:
            raise ValueError(
                f"Imagen demasiado grande para procesar con tiling actual: "
                f"se calcularon {total_tiles} tiles ({tiles_x}×{tiles_y}) "
                f"para resolución {w}x{h} px (máximo permitido: {MAX_ALLOWED_TILES}). "
                f"Reduzca la resolución de la imagen antes de restaurar para prevenir agotamiento de VRAM o RAM."
            )

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Coordenadas del mosaico con solapamiento
                x1 = tx * tile_size
                x2 = min(x1 + tile_size, w)
                y1 = ty * tile_size
                y2 = min(y1 + tile_size, h)

                # Expandir con padding
                px1 = max(x1 - tile_pad, 0)
                px2 = min(x2 + tile_pad, w)
                py1 = max(y1 - tile_pad, 0)
                py2 = min(y2 + tile_pad, h)

                # Extraer tile y forzar C-contiguous en memoria
                tile = np.ascontiguousarray(img_nchw[:, :, py1:py2, px1:px2])
                out_tile = session.run([output_name], {input_name: tile})[0][0]

                # Recortar el padding del resultado
                crop_x1 = (x1 - px1) * scale
                crop_x2 = crop_x1 + (x2 - x1) * scale
                crop_y1 = (y1 - py1) * scale
                crop_y2 = crop_y1 + (y2 - y1) * scale

                cropped_tile = out_tile[:, crop_y1:crop_y2, crop_x1:crop_x2]

                # Acumular en la imagen final
                target_x1 = x1 * scale
                target_x2 = x2 * scale
                target_y1 = y1 * scale
                target_y2 = y2 * scale

                output_accum[:, target_y1:target_y2, target_x1:target_x2] += cropped_tile
                weight_accum[:, target_y1:target_y2, target_x1:target_x2] += 1.0

        weight_accum[weight_accum == 0] = 1.0
        output_final = output_accum / weight_accum
        return self._tensor_to_image(output_final)

    def _tensor_to_image(self, chw_tensor: np.ndarray) -> Image.Image:
        """Convierte un tensor (3, H, W) normalizado a una imagen PIL RGB uint8."""
        clipped = np.clip(chw_tensor * 255.0, 0, 255).astype(np.uint8)
        hwc = np.transpose(clipped, (1, 2, 0))
        return Image.fromarray(np.ascontiguousarray(hwc), mode="RGB")


# Instancia singleton
_restorer_instance: Optional[ImageRestorer] = None


def get_image_restorer() -> ImageRestorer:
    global _restorer_instance
    if _restorer_instance is None:
        _restorer_instance = ImageRestorer()
    return _restorer_instance
