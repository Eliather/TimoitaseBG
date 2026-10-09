"""
TimoitaseBG - Detector y segmentador multiobjeto con YOLO11-seg ONNX.
Detecta automáticamente personas, animales y objetos individuales para aislamiento rápido en chips.
"""
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import get_inference_providers
from app.core.model_fetcher import ensure_model_file

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
        self._lock = threading.Lock()

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
                import onnxruntime as ort
                providers = get_inference_providers()
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                try:
                    self._session = ort.InferenceSession(
                        str(model_path), sess_options, providers=providers
                    )
                except Exception as e:
                    logger.warning(f"Fallo al cargar YOLO11 con GPU ({e}). Usando CPU...")
                    self._session = ort.InferenceSession(
                        str(model_path), sess_options, providers=["CPUExecutionProvider"]
                    )

    def _preprocess(
        self, pil_img: Image.Image, target_size: int = 640
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int], float]:
        """Letterboxing simétrico a target_size x target_size con relleno 114.0."""
        orig_w, orig_h = pil_img.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        resized = pil_img.convert("RGB").resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.float32)

        canvas = np.full((target_size, target_size, 3), 114.0, dtype=np.float32)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
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
    ) -> List[DetectedSubject]:
        """
        Detecta y segmenta todos los sujetos/objetos en la imagen.
        Retorna lista de DetectedSubject con máscara pixel-perfect para cada uno.
        """
        if pil_img is None or pil_img.width <= 0 or pil_img.height <= 0:
            return []

        if self._session is None:
            self.ensure_model()

        tensor, crop_box, (orig_w, orig_h), scale = self._preprocess(pil_img, 640)
        pad_x, pad_y, new_w, new_h = crop_box

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})

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

        # Convertir cajas a formato [x1, y1, w, h] para cv2.dnn.NMSBoxes
        boxes_for_nms = []
        for box in boxes_filt:
            cx, cy, bw, bh = box
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            boxes_for_nms.append([int(x1), int(y1), int(bw), int(bh)])

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

        # Protomasks a shape (32, 160*160)
        protos_flat = output1[0].reshape(32, 160 * 160)

        results: List[DetectedSubject] = []
        for rank, idx in enumerate(nms_indices):
            cx, cy, bw, bh = boxes_filt[idx]
            x1_640 = max(0, cx - bw / 2.0)
            y1_640 = max(0, cy - bh / 2.0)
            x2_640 = min(640, cx + bw / 2.0)
            y2_640 = min(640, cy + bh / 2.0)

            coeff = coeffs_filt[idx : idx + 1]  # (1, 32)
            # Multiplicación matricial (1, 32) @ (32, 25600) -> (1, 25600)
            mask_160_flat = np.dot(coeff, protos_flat)[0]
            mask_160 = mask_160_flat.reshape(160, 160)
            # Sigmoid
            mask_160 = 1.0 / (1.0 + np.exp(-np.clip(mask_160, -20.0, 20.0)))

            # Redimensionar máscara proto a 640x640
            mask_640 = cv2.resize(mask_160, (640, 640), interpolation=cv2.INTER_LINEAR)

            # Recortar al bounding box en 640
            ix1, iy1, ix2, iy2 = int(x1_640), int(y1_640), int(x2_640), int(y2_640)
            box_mask = np.zeros_like(mask_640)
            box_mask[iy1:iy2, ix1:ix2] = mask_640[iy1:iy2, ix1:ix2]

            # Des-letterbox
            crop_subject = box_mask[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
            orig_mask_f32 = cv2.resize(
                crop_subject, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            binary_mask = orig_mask_f32 > 0.5

            # Bounding box en coordenadas de la imagen original
            orig_x1 = max(0, min(orig_w, int(round((x1_640 - pad_x) / scale))))
            orig_y1 = max(0, min(orig_h, int(round((y1_640 - pad_y) / scale))))
            orig_x2 = max(0, min(orig_w, int(round((x2_640 - pad_x) / scale))))
            orig_y2 = max(0, min(orig_h, int(round((y2_640 - pad_y) / scale))))

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

        return results


_detector_instance: Optional[YOLODetector] = None


def get_yolo_detector() -> YOLODetector:
    """Retorna la instancia singleton de YOLODetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector()
    return _detector_instance
