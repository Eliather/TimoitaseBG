"""
TimoitaseBG - Seguimiento temporal de sujetos en secuencias (Smart Sequences).
Combina detección YOLO11-seg + MobileSAM + Guided Filter y suavizado EMA temporal.
"""
import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.core.yolo_detector import (
    DetectedSubject,
    get_yolo_detector,
    refine_yolo_subject_mask,
)

logger = logging.getLogger("TimoitaseBG.SequenceTracker")


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def _select_best_match(
    candidates: List[DetectedSubject],
    prev_bbox: Tuple[int, int, int, int],
    target_class_id: Optional[int],
    img_size: Tuple[int, int],
) -> Optional[DetectedSubject]:
    if not candidates:
        return None
    w, h = max(1, img_size[0]), max(1, img_size[1])
    diag = float((w ** 2 + h ** 2) ** 0.5)
    pcx = (prev_bbox[0] + prev_bbox[2]) * 0.5
    pcy = (prev_bbox[1] + prev_bbox[3]) * 0.5

    best_subj = None
    best_score = -1.0
    for cand in candidates:
        iou = _bbox_iou(prev_bbox, cand.bbox)
        ccx = (cand.bbox[0] + cand.bbox[2]) * 0.5
        ccy = (cand.bbox[1] + cand.bbox[3]) * 0.5
        dist_norm = (((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5) / diag
        class_bonus = 0.25 if (target_class_id is None or cand.class_id == target_class_id) else -0.15
        score = iou * 1.5 + max(0.0, 1.0 - dist_norm) * 0.6 + class_bonus + cand.confidence * 0.1
        if score > best_score:
            best_score = score
            best_subj = cand
    return best_subj if best_score > 0.25 else None


def _mask_to_bbox(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def track_subject_across_sequence(
    frames: List[Image.Image],
    seed_frame_idx: int = 0,
    seed_subject: Optional[DetectedSubject] = None,
    seed_mask: Optional[np.ndarray] = None,
    trim_start: int = 0,
    trim_end: Optional[int] = None,
    yolo_detector: Optional[Any] = None,
    sam_manager: Optional[Any] = None,
    use_sam_refinement: bool = True,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[Image.Image]:
    """
    Rastrea el sujeto semilla hacia adelante y hacia atrás dentro de [trim_start, trim_end].
    Devuelve una lista de imágenes RGBA con la misma longitud que `frames`.
    """
    total = len(frames)
    if total == 0:
        return []

    s_idx = max(0, min(total - 1, int(trim_start)))
    e_idx = max(s_idx, min(total - 1, total - 1 if trim_end is None else int(trim_end)))
    seed_idx = max(s_idx, min(e_idx, int(seed_frame_idx)))

    detector = yolo_detector or get_yolo_detector()
    if sam_manager is None and use_sam_refinement:
        try:
            from app.core.sam_manager import get_sam_manager
            sam_manager = get_sam_manager()
        except Exception:
            sam_manager = None

    results: List[Image.Image] = [f.convert("RGBA") for f in frames]

    # Inicializar semilla si no viene dada
    seed_img = frames[seed_idx]
    target_class_id: Optional[int] = None
    if seed_subject is not None:
        seed_bbox = seed_subject.bbox
        target_class_id = seed_subject.class_id
        seed_bool = seed_subject.mask
    elif seed_mask is not None:
        seed_bool = np.asarray(seed_mask) > 0
        seed_bbox = _mask_to_bbox(seed_bool) or (0, 0, seed_img.width, seed_img.height)
    else:
        dets = detector.detect(seed_img, conf_threshold=0.25)
        if not dets:
            return results
        seed_subject = max(dets, key=lambda d: d.confidence)
        seed_bbox = seed_subject.bbox
        target_class_id = seed_subject.class_id
        seed_bool = seed_subject.mask

    seed_alpha = (seed_bool.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    if seed_subject is not None:
        if use_sam_refinement and sam_manager is not None:
            try:
                sam_manager.prepare_image(seed_img.convert("RGB"))
            except Exception:
                pass
        refined = refine_yolo_subject_mask(seed_img, seed_subject, sam_manager=sam_manager)
        if refined is not None:
            seed_alpha = refined

    def _apply_alpha(pil_img: Image.Image, alpha_u8: np.ndarray) -> Image.Image:
        rgba = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
        rgba[:, :, 3] = np.minimum(rgba[:, :, 3], alpha_u8)
        return Image.fromarray(rgba, mode="RGBA")

    results[seed_idx] = _apply_alpha(seed_img, seed_alpha)

    active_total = max(1, (e_idx - s_idx) + 1)
    processed_count = 1

    def _step_sequence(indices: List[int]):
        nonlocal processed_count
        prev_bbox = seed_bbox
        prev_prob = seed_alpha.astype(np.float32) / 255.0

        for idx in indices:
            if cancel_check and cancel_check():
                raise InterruptedError("Tracking de secuencia cancelado por el usuario.")

            frm = frames[idx]
            curr_prob: Optional[np.ndarray] = None
            dets: List[DetectedSubject] = []
            try:
                dets = detector.detect(frm, conf_threshold=0.25)
            except Exception as exc:
                logger.warning(f"Detección YOLO falló en fotograma {idx}: {exc}")

            matched = _select_best_match(dets, prev_bbox, target_class_id, frm.size)
            if matched is not None:
                prev_bbox = matched.bbox
                if use_sam_refinement and sam_manager is not None:
                    try:
                        sam_manager.prepare_image(frm.convert("RGB"))
                    except Exception:
                        pass
                alpha_u8 = refine_yolo_subject_mask(frm, matched, sam_manager=sam_manager)
                if alpha_u8 is not None:
                    curr_prob = alpha_u8.astype(np.float32) / 255.0
                else:
                    curr_prob = matched.mask.astype(np.float32)
            elif use_sam_refinement and sam_manager is not None:
                try:
                    sam_manager.prepare_image(frm.convert("RGB"))
                    sam_prob = sam_manager.predict_mask_from_yolo_subject(
                        bbox=prev_bbox,
                        coarse_mask=prev_prob > 0.5,
                        orig_img_size=frm.size,
                    )
                    if sam_prob is not None:
                        curr_prob = sam_prob
                        nb = _mask_to_bbox(curr_prob > 0.5)
                        if nb:
                            prev_bbox = nb
                except Exception:
                    curr_prob = None

            if curr_prob is None:
                # Mantener continuidad si hubo oclusión momentánea en 1 frame
                curr_prob = prev_prob

            if curr_prob.shape == prev_prob.shape:
                smoothed = np.clip(curr_prob * 0.78 + prev_prob * 0.22, 0.0, 1.0)
            else:
                smoothed = curr_prob

            prev_prob = smoothed
            alpha_u8 = np.clip(np.round(smoothed * 255.0), 0, 255).astype(np.uint8)
            results[idx] = _apply_alpha(frm, alpha_u8)

            processed_count += 1
            if progress_cb:
                pct = int(min(99, round((processed_count / active_total) * 100)))
                progress_cb(pct, f"Rastreando sujeto ({processed_count}/{active_total})...")

    forward_indices = list(range(seed_idx + 1, e_idx + 1))
    backward_indices = list(range(seed_idx - 1, s_idx - 1, -1))
    _step_sequence(forward_indices)
    _step_sequence(backward_indices)

    if progress_cb:
        progress_cb(100, "Rastreo de secuencia completado.")
    return results
