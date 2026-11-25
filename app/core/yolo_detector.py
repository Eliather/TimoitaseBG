"""
TimoitaseBG - Detector y segmentador multiobjeto con YOLO11-seg ONNX.
Detecta automáticamente personas, animales y objetos individuales para aislamiento rápido en chips.
"""
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from app.config import get_inference_providers, get_optimized_session_options
from app.core.model_fetcher import ensure_model_file, make_download_progress_cb

logger = logging.getLogger("TimoitaseBG.YOLODetector")

# 80 Clases estándar de COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Emojis y nombres amigables para chips de la UI
CLASS_EMOJIS = {
    "person": "👤", "cat": "🐱", "dog": "🐶", "bird": "🐦", "horse": "🐴",
    "car": "🚗", "motorcycle": "🏍️", "bicycle": "🚲", "airplane": "✈️",
    "bus": "🚌", "train": "🚆", "truck": "🚚", "boat": "⛵",
    "backpack": "🎒", "umbrella": "☂️", "handbag": "👜", "tie": "👔", "suitcase": "🧳",
    "bottle": "🍾", "cup": "☕", "chair": "🪑", "couch": "🛋️", "bed": "🛏️",
    "tv": "📺", "laptop": "💻", "mouse": "🖱️", "keyboard": "⌨️", "cell phone": "📱",
    "book": "📖", "potted plant": "🪴", "clock": "⏰", "scissors": "✂️"
}

CHIP_COLORS = [
    "#3B82F6",  # Azul vibrante
    "#10B981",  # Esmeralda
    "#8B5CF6",  # Púrpura
    "#F59E0B",  # Ámbar
    "#EC4899",  # Rosa
    "#06B6D4",  # Cian
    "#6366F1",  # Índigo
    "#14B8A6",  # Turquesa
]


@dataclass
class DetectedSubject:
    """Representa un objeto individual detectado y segmentado por YOLO11-seg."""
    subject_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) en coordenadas nativas
    mask: np.ndarray  # Máscara booleana 2D de dimensiones (orig_h, orig_w)
    color_hex: str
    emoji: str


class YOLODetector:
    """Motor de detección y segmentación multiobjeto YOLO11-seg."""

    def __init__(self):
        self._session: Optional[Any] = None
        self._model_path: Optional[Path] = None
        self._lock = threading.RLock()

    def ensure_model(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
    ):
        """Descarga bajo demanda el modelo YOLO11n-seg (~11.7MB) si aún no existe."""
        with self._lock:
            if self._session is None:
                model_path = ensure_model_file(
                    "yolo11n-seg.onnx",
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                self._model_path = Path(model_path)
                import onnxruntime as ort
                from app.core.gpu_manager import get_gpu_manager

                gpu_mgr = get_gpu_manager()
                providers = get_inference_providers() if gpu_mgr.is_gpu_enabled() else ["CPUExecutionProvider"]
                sess_options = get_optimized_session_options() or ort.SessionOptions()
                try:
                    self._session = ort.InferenceSession(
                        str(model_path), sess_options, providers=providers
                    )
                except Exception as e:
                    logger.warning(f"Fallo al cargar YOLO11 con GPU ({e}). Usando CPU...")
                    self._session = ort.InferenceSession(
                        str(model_path), sess_options, providers=["CPUExecutionProvider"]
                    )

    def _run_with_cpu_fallback(self, input_name: str, tensor: np.ndarray):
        """Ejecuta la sesión ONNX y hace fallback automático a CPU si falla en GPU."""
        try:
            return self._session.run(None, {input_name: tensor})
        except Exception as exc:
            providers = []
            if hasattr(self._session, "get_providers"):
                try:
                    providers = list(self._session.get_providers())
                except Exception:
                    providers = []
            is_cpu_only = providers == ["CPUExecutionProvider"]
            if is_cpu_only:
                raise
            logger.warning(f"Fallo en inferencia GPU de YOLO11 ({exc}). Reintentando en CPU...")
            import onnxruntime as ort
            from app.core.model_fetcher import resolve_local_model

            model_path = self._model_path or resolve_local_model("yolo11n-seg.onnx")
            if model_path is None:
                raise
            sess_options = get_optimized_session_options() or ort.SessionOptions()
            self._session = ort.InferenceSession(
                str(model_path), sess_options, providers=["CPUExecutionProvider"]
            )
            return self._session.run(None, {input_name: tensor})

    def _get_target_hw(self) -> Tuple[int, int]:
        """Lee dinámicamente (target_h, target_w) del tensor de entrada ONNX (ej. 480x640 o 640x640)."""
        if self._session is not None:
            try:
                inputs = self._session.get_inputs()
                if inputs:
                    shape = getattr(inputs[0], "shape", None)
                    if (
                        isinstance(shape, (list, tuple))
                        and len(shape) >= 4
                        and isinstance(shape[2], int)
                        and isinstance(shape[3], int)
                        and shape[2] > 0
                        and shape[3] > 0
                    ):
                        return int(shape[2]), int(shape[3])
            except Exception:
                pass
        return (640, 640)

    def unload_sessions(self) -> None:
        """Libera la sesión ONNX activa de YOLO11-seg para desocupar VRAM o reiniciar en otro proveedor."""
        with self._lock:
            if self._session is not None:
                self._session = None
        import gc
        gc.collect()

    def _preprocess(
        self, pil_img: Image.Image, target_size: Union[int, Tuple[int, int]] = 640
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int], float]:
        """Letterboxing simétrico a target_h x target_w (cuadrado o rectangular) con relleno 114.0."""
        if isinstance(target_size, (tuple, list)):
            target_h, target_w = int(target_size[0]), int(target_size[1])
        else:
            target_h = target_w = int(target_size)

        orig_w, orig_h = pil_img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        resized = pil_img.convert("RGB").resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32)

        canvas = np.full((target_h, target_w, 3), 114.0, dtype=np.float32)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = rgb

        # Normalización estándar YOLO [0, 1] y NCHW
        norm = canvas / 255.0
        tensor = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
        crop_box = (pad_x, pad_y, new_w, new_h)
        return tensor, crop_box, (orig_w, orig_h), scale

    def detect_subjects(
        self,
        pil_img: Image.Image,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> List[DetectedSubject]:
        """
        Detecta y segmenta todos los sujetos/objetos en la imagen.
        Retorna lista de DetectedSubject con máscara pixel-perfect para cada uno.
        """
        if pil_img is None or pil_img.width <= 0 or pil_img.height <= 0:
            return []

        with self._lock:
            if self._session is None:
                if progress_cb:
                    progress_cb(10, "Verificando modelo YOLO11-seg...")

                self.ensure_model(
                    progress_cb=make_download_progress_cb(progress_cb, "YOLO11-seg", start_pct=10, span_pct=35)
                )

            if progress_cb:
                progress_cb(50, "Preprocesando imagen para detección YOLO11...")
            target_h, target_w = self._get_target_hw()
            tensor, crop_box, (orig_w, orig_h), scale = self._preprocess(pil_img, (target_h, target_w))
            pad_x, pad_y, new_w, new_h = crop_box

            if progress_cb:
                progress_cb(70, "Ejecutando inferencia multiobjeto YOLO11-seg...")
            input_name = self._session.get_inputs()[0].name
            outputs = self._run_with_cpu_fallback(input_name, tensor)
            if progress_cb:
                progress_cb(90, "Generando máscaras de sujetos detectados...")

        # Salida 0: (1, 116, 8400) -> Transponer a (8400, 116)
        output0 = outputs[0]
        output1 = outputs[1]  # Protomasks: (1, 32, 160, 160)

        preds = output0[0].T  # Shape: (8400, 116)
        boxes = preds[:, :4]  # cx, cy, w, h
        scores = preds[:, 4:84]  # 80 clases
        mask_coeffs = preds[:, 84:]  # 32 coeficientes de máscara

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filtrar por confianza mínima
        keep_indices = np.where(confidences >= conf_threshold)[0]
        if len(keep_indices) == 0:
            return []

        boxes_filt = boxes[keep_indices]
        confs_filt = confidences[keep_indices]
        class_ids_filt = class_ids[keep_indices]
        coeffs_filt = mask_coeffs[keep_indices]

        # Convertir cajas a formato [x1, y1, w, h] de forma vectorizada para cv2.dnn.NMSBoxes
        x1_arr = (boxes_filt[:, 0] - boxes_filt[:, 2] * 0.5).astype(np.int32)
        y1_arr = (boxes_filt[:, 1] - boxes_filt[:, 3] * 0.5).astype(np.int32)
        bw_arr = boxes_filt[:, 2].astype(np.int32)
        bh_arr = boxes_filt[:, 3].astype(np.int32)
        boxes_for_nms = np.stack([x1_arr, y1_arr, bw_arr, bh_arr], axis=1).tolist()

        nms_indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confs_filt.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold,
        )

        if len(nms_indices) == 0:
            return []

        if isinstance(nms_indices, np.ndarray):
            nms_indices = nms_indices.flatten().tolist()

        # Multiplicación matricial por lotes (N, proto_c) @ (proto_c, proto_h * proto_w) -> (N, proto_h, proto_w)
        proto_c, proto_h, proto_w = output1.shape[1:]
        protos_flat = output1[0].reshape(proto_c, proto_h * proto_w)
        nms_coeffs = coeffs_filt[nms_indices]
        masks_proto_all = np.dot(nms_coeffs, protos_flat).reshape(-1, proto_h, proto_w)

        scale_px = proto_w / float(target_w)
        scale_py = proto_h / float(target_h)

        results: List[DetectedSubject] = []
        for rank, idx in enumerate(nms_indices):
            cx, cy, bw, bh = boxes_filt[idx]
            x1_in = max(float(pad_x), cx - bw / 2.0)
            y1_in = max(float(pad_y), cy - bh / 2.0)
            x2_in = min(float(pad_x + new_w), cx + bw / 2.0)
            y2_in = min(float(pad_y + new_h), cy + bh / 2.0)

            # Bounding box en coordenadas de la imagen original
            orig_x1 = max(0, min(orig_w, int(round((x1_in - pad_x) / scale))))
            orig_y1 = max(0, min(orig_h, int(round((y1_in - pad_y) / scale))))
            orig_x2 = max(0, min(orig_w, int(round((x2_in - pad_x) / scale))))
            orig_y2 = max(0, min(orig_h, int(round((y2_in - pad_y) / scale))))

            binary_mask = np.zeros((orig_h, orig_w), dtype=bool)
            box_w = orig_x2 - orig_x1
            box_h = orig_y2 - orig_y1
            if box_w > 0 and box_h > 0:
                # Recortar en el espacio prototipo dinámico (proto_h x proto_w)
                px1 = max(0, min(proto_w - 1, int(np.floor(x1_in * scale_px))))
                py1 = max(0, min(proto_h - 1, int(np.floor(y1_in * scale_py))))
                px2 = max(px1 + 1, min(proto_w, int(np.ceil(x2_in * scale_px))))
                py2 = max(py1 + 1, min(proto_h, int(np.ceil(y2_in * scale_py))))

                proto_crop = masks_proto_all[rank, py1:py2, px1:px2]
                # Redimensionar únicamente el ROI del sujeto (sigmoid(x) > 0.5 <=> logit > 0.0)
                patch_logits = cv2.resize(
                    proto_crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR
                )
                binary_mask[orig_y1:orig_y2, orig_x1:orig_x2] = patch_logits > 0.0

            cid = int(class_ids_filt[idx])
            cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else "object"
            conf = float(confs_filt[idx])
            emoji = CLASS_EMOJIS.get(cname, "🎯")
            color_hex = CHIP_COLORS[rank % len(CHIP_COLORS)]

            results.append(
                DetectedSubject(
                    subject_id=rank + 1,
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    bbox=(orig_x1, orig_y1, orig_x2, orig_y2),
                    mask=binary_mask,
                    color_hex=color_hex,
                    emoji=emoji,
                )
            )

        if progress_cb:
            progress_cb(100, "Detección YOLO11 completada.")
        return results


_detector_instance: Optional[YOLODetector] = None


def refine_yolo_subject_mask(
    pil_img: Optional[Image.Image],
    subject: Optional[DetectedSubject],
    sam_manager: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """
    Convierte la máscara gruesa de YOLO11-seg en una máscara alfa HD (uint8 0-255)
    combinando MobileSAM (si tiene embedding listo) y Guided Filter perimetral.
    """
    if pil_img is None or subject is None:
        return None

    orig_w, orig_h = pil_img.size
    coarse_bool = subject.mask
    if coarse_bool.shape != (orig_h, orig_w):
        return None

    base_prob = coarse_bool.astype(np.float32)

    if sam_manager is None:
        from app.core.sam_manager import get_sam_manager
        sam_manager = get_sam_manager()

    if sam_manager is not None and sam_manager.has_embedding():
        sam_prob = sam_manager.predict_mask_from_yolo_subject(
            bbox=subject.bbox,
            coarse_mask=coarse_bool,
            orig_img_size=(orig_w, orig_h),
        )
        if sam_prob is not None:
            sam_bin = sam_prob > 0.5
            inter = np.logical_and(sam_bin, coarse_bool).sum()
            union = np.logical_or(sam_bin, coarse_bool).sum()
            if union > 0 and (inter / union) >= 0.40:
                base_prob = sam_prob

    try:
        from app.core.postprocess import apply_guided_filter

        x1, y1, x2, y2 = [int(v) for v in subject.bbox]
        pad = 24
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(orig_w, x2 + pad), min(orig_h, y2 + pad)

        if rx2 > rx1 and ry2 > ry1:
            crop_rgb = np.array(pil_img.crop((rx1, ry1, rx2, ry2)).convert("RGB"), dtype=np.uint8)
            crop_guide = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            crop_prob = base_prob[ry1:ry2, rx1:rx2]
            crop_refined = apply_guided_filter(
                guide_bgr=crop_guide,
                prob_mask=crop_prob,
                radius=6,
                eps=1e-4,
                band_dilation=12,
            )
            base_prob = base_prob.copy()
            base_prob[ry1:ry2, rx1:rx2] = crop_refined
    except Exception as e:
        logger.warning(f"Guided filter en aislamiento YOLO omitido: {e}")

    return np.clip(np.round(base_prob * 255.0), 0, 255).astype(np.uint8)


def get_yolo_detector() -> YOLODetector:
    """Retorna la instancia singleton de YOLODetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector()
    return _detector_instance
