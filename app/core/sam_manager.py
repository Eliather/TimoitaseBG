"""
TimoitaseBG - Gestor de Segmentación Interactiva por IA con MobileSAM.
Permite la selección inteligente de objetos con un solo clic sobre el lienzo.
"""
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Callable, Any
import numpy as np
import cv2
from PIL import Image

from app.config import (
    MODELS_DIR,
    get_inference_providers,
    get_optimized_session_options,
)
from app.core.model_fetcher import ensure_model_file

logger = logging.getLogger("TimoitaseBG.SAMManager")

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


class SAMManager:
    """
    Gestiona las sesiones de inferencia ONNX de MobileSAM (Encoder y Decoder).
    Mantiene en caché el embedding de la imagen activa para permitir
    inferencias interactivas por clic en tiempo real (<15 ms).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.encoder_session = None
        self.decoder_session = None
        self.cached_embedding: Optional[np.ndarray] = None
        self.current_image_id: Optional[int] = None
        self.orig_size: Tuple[int, int] = (0, 0)  # (w, h)
        self.scaled_size: Tuple[int, int] = (0, 0)  # (new_w, new_h)
        self.scale_factor: float = 1.0
        self._is_computing: bool = False

    def is_available(self) -> bool:
        """Indica si los pesos de MobileSAM ya están descargados localmente."""
        enc_path = MODELS_DIR / "mobile_sam_image_encoder.onnx"
        dec_path = MODELS_DIR / "sam_mask_decoder_single.onnx"
        return enc_path.exists() and dec_path.exists()

    def ensure_models(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
    ):
        """Descarga bajo demanda los modelos de MobileSAM si aún no existen."""
        with self._lock:
            if self.encoder_session is None or self.decoder_session is None:
                enc_path = ensure_model_file(
                    "mobile_sam_image_encoder.onnx",
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                dec_path = ensure_model_file(
                    "sam_mask_decoder_single.onnx",
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                self._init_sessions(enc_path, dec_path)

    def _init_sessions(self, enc_path: Path, dec_path: Path):
        """Inicializa las sesiones de ONNX Runtime con proveedores optimizados."""
        import onnxruntime as ort

        providers = get_inference_providers()
        sess_options = get_optimized_session_options() or ort.SessionOptions()

        logger.info(f"Cargando MobileSAM Encoder desde {enc_path.name}...")
        try:
            self.encoder_session = ort.InferenceSession(
                str(enc_path), sess_options, providers=providers
            )
        except Exception as e:
            logger.warning(f"Fallo al cargar Encoder con DirectML, usando CPU: {e}")
            self.encoder_session = ort.InferenceSession(
                str(enc_path), sess_options, providers=["CPUExecutionProvider"]
            )

        logger.info(f"Cargando MobileSAM Decoder desde {dec_path.name}...")
        try:
            self.decoder_session = ort.InferenceSession(
                str(dec_path), sess_options, providers=providers
            )
        except Exception as e:
            logger.warning(f"Fallo al cargar Decoder con DirectML, usando CPU: {e}")
            self.decoder_session = ort.InferenceSession(
                str(dec_path), sess_options, providers=["CPUExecutionProvider"]
            )

    def has_embedding(self) -> bool:
        """Indica si la imagen activa ya tiene su embedding calculado."""
        return self.cached_embedding is not None

    def clear_embedding(self):
        """Libera la caché del embedding de imagen para ahorrar memoria."""
        with self._lock:
            self.cached_embedding = None
            self.current_image_id = None
            self.orig_size = (0, 0)
            self.scaled_size = (0, 0)

    def _preprocess_image(self, pil_img: Image.Image) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocesa la imagen para el encoder de MobileSAM.
        El modelo ONNX de Acly espera entrada de Rango 3: (1024, 1024, 3) en float32 (HWC),
        y contiene internamente la resta de media y división por std.
        """
        w, h = pil_img.size
        scale = 1024.0 / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Redimensionar conservando aspect ratio (lado más largo a 1024)
        resized = pil_img.convert("RGB").resize((new_w, new_h), resample=_BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32)

        # Padding a 1024x1024x3 (HWC, float32 en rango 0-255)
        input_tensor = np.zeros((1024, 1024, 3), dtype=np.float32)
        input_tensor[:new_h, :new_w, :] = rgb

        return input_tensor, scale, new_w, new_h

    def prepare_image(self, pil_img: Image.Image, image_id: Optional[int] = None) -> bool:
        """
        Ejecuta el Image Encoder de MobileSAM para la imagen dada.
        Calcula y almacena en memoria el tensor de embeddings.
        """
        if pil_img is None or pil_img.width <= 0 or pil_img.height <= 0:
            return False

        with self._lock:
            # Si ya está calculado para esta misma instancia, reutilizar
            if image_id is not None and self.current_image_id == image_id and self.cached_embedding is not None:
                return True

            if self._is_computing:
                logger.info("MobileSAM: Ya hay un cálculo de embedding en progreso, ignorando llamada simultánea.")
                return False

            self._is_computing = True

        try:
            if self.encoder_session is None or self.decoder_session is None:
                self.ensure_models()

            w, h = pil_img.size
            input_tensor, scale, new_w, new_h = self._preprocess_image(pil_img)

            # Ejecutar encoder
            enc_input_name = self.encoder_session.get_inputs()[0].name
            outputs = self.encoder_session.run(None, {enc_input_name: input_tensor})
            embedding = outputs[0]  # shape: (1, 256, 64, 64)

            with self._lock:
                self.cached_embedding = embedding
                self.current_image_id = image_id
                self.orig_size = (w, h)
                self.scaled_size = (new_w, new_h)
                self.scale_factor = scale
                self._is_computing = False

            logger.info(f"MobileSAM: Embedding calculado para imagen de {w}x{h} px.")
            return True

        except Exception as e:
            with self._lock:
                self._is_computing = False
            logger.error(f"Error calculando embedding de MobileSAM: {e}")
            raise e

    def predict_mask_at_point(
        self,
        click_x: int,
        click_y: int,
        orig_img_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Ejecuta el Mask Decoder en <15 ms utilizando el embedding en caché.
        Retorna una máscara booleana 2D de dimensiones (orig_h, orig_w).
        Protegido por cerrojo para evitar colisiones con actualizaciones de embedding.
        """
        with self._lock:
            if self.cached_embedding is None or self.decoder_session is None:
                logger.warning("No hay embedding en caché o el decoder no está inicializado.")
                return None

            orig_w, orig_h = orig_img_size or self.orig_size
            scale = self.scale_factor or (1024.0 / max(orig_w, orig_h))
            scaled_x = float(click_x) * scale
            scaled_y = float(click_y) * scale

            # Entradas esperadas por el decoder ONNX de MobileSAM
            point_coords = np.array([[[scaled_x, scaled_y]]], dtype=np.float32)
            point_labels = np.array([[1.0]], dtype=np.float32)
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0.0], dtype=np.float32)
            orig_im_size = np.array([1024.0, 1024.0], dtype=np.float32)

            decoder_inputs = {}
            for inp in self.decoder_session.get_inputs():
                name = inp.name
                if "image_embedding" in name:
                    decoder_inputs[name] = self.cached_embedding
                elif "point_coord" in name:
                    decoder_inputs[name] = point_coords
                elif "point_label" in name:
                    decoder_inputs[name] = point_labels
                elif "mask_input" in name and "has" not in name:
                    decoder_inputs[name] = mask_input
                elif "has_mask_input" in name:
                    decoder_inputs[name] = has_mask_input
                elif "orig_im_size" in name:
                    decoder_inputs[name] = orig_im_size

            try:
                outputs = self.decoder_session.run(None, decoder_inputs)
                raw_mask = outputs[0]

                if raw_mask.ndim == 4:
                    mask_2d = raw_mask[0, 0]
                elif raw_mask.ndim == 3:
                    mask_2d = raw_mask[0]
                else:
                    mask_2d = np.squeeze(raw_mask)

                # Recortar el área activa antes de las dimensiones de padding
                new_w, new_h = self.scaled_size
                if new_w > 0 and new_h > 0 and mask_2d.shape[0] >= new_h and mask_2d.shape[1] >= new_w:
                    crop_mask = mask_2d[:new_h, :new_w]
                else:
                    crop_mask = mask_2d

                resized_mask = cv2.resize(
                    crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )
                return resized_mask > 0.0

            except Exception as e:
                logger.error(f"Error en inferencia del Decoder de MobileSAM: {e}")
                return None


# Instancia singleton global
_sam_manager_instance: Optional[SAMManager] = None


def get_sam_manager() -> SAMManager:
    """Retorna la instancia singleton de SAMManager."""
    global _sam_manager_instance
    if _sam_manager_instance is None:
        _sam_manager_instance = SAMManager()
    return _sam_manager_instance
