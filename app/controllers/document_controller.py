"""
TimoitaseBG - Controlador de Documento, Secuencias, Historial, Portapapeles y Exportación.
Centraliza las operaciones sobre ImageState y SequenceSession sin acoplarse a widgets Qt.
"""
import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from app.core.image_state import ImageState, SequenceSession
from app.core.image_utils import apply_solid_background
from app.core.yolo_detector import refine_yolo_subject_mask

logger = logging.getLogger("TimoitaseBG.DocumentController")


class DocumentController:
    """Administra el documento activo (imagen única o secuencia), historial y E/S."""

    def __init__(self):
        self.image_state: ImageState = ImageState()
        self.sequence_session: Optional[SequenceSession] = None
        self.current_filepath: Optional[Path] = None

    @property
    def has_sequence(self) -> bool:
        return bool(self.sequence_session and self.sequence_session.total_frames > 1)

    def clear_sequence(self) -> None:
        self.sequence_session = None

    def load_pil_image(
        self,
        pil_img: Image.Image,
        filepath: Optional[Path] = None,
        clear_seq: bool = True,
    ) -> Optional[Image.Image]:
        """Carga una imagen PIL estática en el estado actual y reinicia el historial."""
        if pil_img is None or pil_img.width <= 0 or pil_img.height <= 0:
            return None
        if clear_seq:
            self.sequence_session = None
        self.current_filepath = filepath
        return self.image_state.load_new_image(pil_img)

    def open_media(
        self,
        file_path: str,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Carga una imagen estática, animación (.gif, .webp, .apng) o video desde disco.
        Retorna un diccionario descriptor con `kind` in {"missing", "video", "animation", "image"}.
        """
        path = Path(file_path)
        if not path.exists():
            return {"kind": "missing", "path": path}

        from app.core.video_processor import extract_video_frames, is_video_file

        if is_video_file(path):
            if status_cb:
                status_cb(f"⏳ Extrayendo fotogramas del video: {path.name}...")
            frames, durations, fps = extract_video_frames(path)
            if len(frames) >= 1:
                self.sequence_session = SequenceSession(
                    frames, durations, loop=0, fps=fps, is_video=True
                )
                self.current_filepath = path
                self.image_state = self.sequence_session.active_state
                fmt_name = path.suffix.upper().lstrip(".")
                return {
                    "kind": "video",
                    "path": path,
                    "frames_count": len(frames),
                    "fps": fps,
                    "fmt_name": fmt_name,
                    "image": self.image_state.current_image,
                }

        from app.core.gif_processor import extract_gif_frames, is_animated_image

        if is_animated_image(path):
            frames, durations, loop = extract_gif_frames(path)
            if len(frames) > 1:
                self.sequence_session = SequenceSession(
                    frames, durations, loop=loop, is_video=False
                )
                self.current_filepath = path
                self.image_state = self.sequence_session.active_state
                fmt_name = path.suffix.upper().lstrip(".")
                if fmt_name == "PNG":
                    fmt_name = "APNG"
                return {
                    "kind": "animation",
                    "path": path,
                    "frames_count": len(frames),
                    "fmt_name": fmt_name,
                    "image": self.image_state.current_image,
                }

        self.sequence_session = None
        pil_img = Image.open(path)
        loaded = self.load_pil_image(pil_img, filepath=path, clear_seq=True)
        return {
            "kind": "image",
            "path": path,
            "image": loaded,
        }

    def select_sequence_frame(self, new_index: int) -> Tuple[int, int, int, Optional[Image.Image]]:
        """Selecciona el fotograma `new_index` de la secuencia activa y actualiza `image_state`."""
        total = self.sequence_session.total_frames
        idx = max(0, min(new_index, total - 1))
        self.sequence_session.current_index = idx
        self.image_state = self.sequence_session.active_state
        cur_dur = self.sequence_session.durations[idx] if self.sequence_session.durations else 80
        return idx, total, cur_dur, self.image_state.current_image

    def apply_sequence_results(self, processed_frames: list, description: str) -> int:
        """Registra una lista de fotogramas procesados en `sequence_session` y retorna el índice activo."""
        if not processed_frames:
            return 0
        if self.sequence_session and len(self.sequence_session.states) == len(processed_frames):
            cur_idx = self.sequence_session.current_index
            for st, proc_frm in zip(self.sequence_session.states, processed_frames):
                st.push_state(proc_frm, description=description)
            return min(cur_idx, len(processed_frames) - 1)

        durations = self.sequence_session.durations if self.sequence_session else [80] * len(processed_frames)
        loop = self.sequence_session.loop if self.sequence_session else 0
        fps = getattr(self.sequence_session, "fps", None) if self.sequence_session else None
        is_vid = getattr(self.sequence_session, "is_video", False) if self.sequence_session else False
        cur_idx = self.sequence_session.current_index if self.sequence_session else 0
        self.sequence_session = SequenceSession(
            processed_frames, durations, loop, fps=fps, is_video=is_vid
        )
        return min(cur_idx, len(processed_frames) - 1)

    def build_clipboard_png(
        self, bg_color_rgb: Optional[Tuple[int, int, int]] = None
    ) -> Optional[Tuple[bytes, Image.Image]]:
        """Serializa la imagen actual a stream PNG RGBA de 8 bits para el portapapeles."""
        if not self.image_state.has_image:
            return None
        img = self.image_state.current_image
        if not img:
            return None

        if bg_color_rgb is not None:
            img = apply_solid_background(img, bg_color_rgb)
        elif img.mode != "RGBA":
            img = img.convert("RGBA")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), img

    def save_to_path(
        self,
        path: str,
        selected_filter: str = "",
        bg_color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Path]:
        """Guarda la imagen estática o secuencia animada/video al formato correspondiente."""
        img = self.image_state.current_image
        if not img:
            return None

        is_animated_seq = self.has_sequence
        out_path = Path(path)
        ext = out_path.suffix.lower()

        if not ext:
            sf_low = selected_filter.lower()
            if "webm" in sf_low:
                ext = ".webm"
            elif "mkv" in sf_low or "matroska" in sf_low:
                ext = ".mkv"
            elif ".mov" in sf_low or "quicktime" in sf_low:
                ext = ".mov"
            elif "mp4" in sf_low:
                ext = ".mp4"
            elif "avi" in sf_low:
                ext = ".avi"
            elif "zip" in sf_low:
                ext = ".zip"
            elif "webp" in sf_low:
                ext = ".webp"
            elif "gif" in sf_low:
                ext = ".gif"
            elif "apng" in sf_low:
                ext = ".apng"
            elif "jpg" in sf_low or "jpeg" in sf_low:
                ext = ".jpg"
            else:
                ext = ".png"
            out_path = Path(f"{path}{ext}")

        from app.core.gif_processor import save_animated_apng, save_animated_gif, save_animated_webp
        from app.core.video_processor import save_video_sequence

        if is_animated_seq:
            frames_to_save, durations, fps = self.sequence_session.get_export_frames_and_durations()
            loop = self.sequence_session.loop
        else:
            frames_to_save = [img]
            durations = [80]
            loop = 0
            fps = 24.0

        if is_animated_seq and ext in (".webm", ".mkv", ".mov", ".mp4", ".avi", ".m4v", ".ogv", ".zip"):
            save_video_sequence(
                frames_rgba=frames_to_save,
                output_path=out_path,
                fps=fps,
                bg_color_rgb=bg_color_rgb,
                loop=loop,
            )
        elif is_animated_seq and (ext == ".webp" or ("WebP" in selected_filter and ext != ".gif")):
            save_animated_webp(
                frames_rgba=frames_to_save,
                output_path=out_path,
                durations=durations,
                loop=loop,
                bg_color_rgb=bg_color_rgb,
            )
        elif is_animated_seq and (ext == ".gif" or ("GIF" in selected_filter and ext != ".webp")):
            save_animated_gif(
                frames_rgba=frames_to_save,
                output_path=out_path,
                durations=durations,
                loop=loop,
                bg_color_rgb=bg_color_rgb,
            )
        elif is_animated_seq and (ext == ".apng" or ("APNG" in selected_filter and "Fotograma" not in selected_filter)):
            save_animated_apng(
                frames_rgba=frames_to_save,
                output_path=out_path,
                durations=durations,
                loop=loop,
                bg_color_rgb=bg_color_rgb,
            )
        elif ext in (".jpg", ".jpeg"):
            if bg_color_rgb is not None:
                img = apply_solid_background(img, bg_color_rgb)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                bg.save(str(out_path), quality=95)
            else:
                img.convert("RGB").save(str(out_path), quality=95)
        elif ext == ".webp":
            if bg_color_rgb is not None:
                img = apply_solid_background(img, bg_color_rgb)
            img.save(str(out_path), quality=95)
        elif ext == ".gif":
            save_animated_gif(
                frames_rgba=[img],
                output_path=out_path,
                durations=[100],
                loop=0,
                bg_color_rgb=bg_color_rgb,
            )
        else:
            if bg_color_rgb is not None:
                img = apply_solid_background(img, bg_color_rgb)
            img.save(str(out_path))

        return out_path

    def compute_autocrop(self) -> Tuple[Optional[str], Optional[Tuple[Image.Image, Optional[Image.Image], str]]]:
        """Calcula el recorte automático de márgenes transparentes. Retorna (status_error_key, resultado)."""
        if not self.image_state.has_image:
            return None, None
        if not self.image_state.has_transparency:
            return "status_crop_first_remove", None

        current_img = self.image_state.current_image
        alpha = current_img.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            return "status_crop_none", None

        pad = 10
        left = max(0, bbox[0] - pad)
        top = max(0, bbox[1] - pad)
        right = min(current_img.width, bbox[2] + pad)
        bottom = min(current_img.height, bbox[3] + pad)

        crop_box = (left, top, right, bottom)
        cropped = current_img.crop(crop_box)
        if cropped.size == current_img.size:
            return "status_crop_already", None

        brush_ref = self.image_state.current_brush_reference
        if brush_ref:
            brush_ref = brush_ref.crop(crop_box)

        desc = f"Recorte al contenido ({cropped.width}x{cropped.height})"
        return None, (cropped, brush_ref, desc)

    def apply_autocrop(self) -> Tuple[Optional[str], Optional[Tuple[Image.Image, Optional[Image.Image], str]]]:
        """
        Recorta al contenido sobre imagen única o uniformemente sobre toda la secuencia activa.
        Para secuencias calcula la caja envolvente común (union bbox) de todos los fotogramas.
        """
        if not self.has_sequence:
            return self.compute_autocrop()

        states = self.sequence_session.states
        if not any(st.has_transparency for st in states):
            return "status_crop_first_remove", None

        union_box = None
        w0, h0 = states[0].current_image.size
        for st in states:
            img = st.current_image
            if img.mode == "RGBA":
                box = img.getchannel("A").getbbox()
                if box:
                    if union_box is None:
                        union_box = list(box)
                    else:
                        union_box[0] = min(union_box[0], box[0])
                        union_box[1] = min(union_box[1], box[1])
                        union_box[2] = max(union_box[2], box[2])
                        union_box[3] = max(union_box[3], box[3])

        if union_box is None:
            return "status_crop_none", None

        pad = 10
        crop_box = (
            max(0, union_box[0] - pad),
            max(0, union_box[1] - pad),
            min(w0, union_box[2] + pad),
            min(h0, union_box[3] + pad),
        )
        if crop_box == (0, 0, w0, h0):
            return "status_crop_already", None

        desc = f"Recorte al contenido ({crop_box[2] - crop_box[0]}x{crop_box[3] - crop_box[1]})"
        for st in states:
            cropped_fr = st.current_image.crop(crop_box)
            ref = st.current_brush_reference
            cropped_ref = ref.crop(crop_box) if ref is not None else None
            st.push_state(cropped_fr, desc, brush_reference=cropped_ref)

        active = self.image_state
        return None, (active.current_image, active.current_brush_reference, desc)

    def apply_pil_transform(
        self,
        transform_fn,
        description: str,
        update_brush_reference: bool = False,
    ) -> Optional[Image.Image]:
        """Aplica una transformación PIL pura sobre la imagen activa o toda la secuencia."""
        if not self.image_state.has_image:
            return None

        if self.has_sequence:
            for st in self.sequence_session.states:
                img = st.current_image
                if img is None:
                    continue
                out = transform_fn(img)
                ref = transform_fn(st.current_brush_reference) if (update_brush_reference and st.current_brush_reference is not None) else None
                st.push_state(out, description, brush_reference=ref)
            return self.image_state.current_image

        out = transform_fn(self.image_state.current_image)
        ref = (
            transform_fn(self.image_state.current_brush_reference)
            if (update_brush_reference and self.image_state.current_brush_reference is not None)
            else None
        )
        self.image_state.push_state(out, description, brush_reference=ref)
        return self.image_state.current_image

    def apply_edge_refinement(
        self,
        choke_px: int = 0,
        feather_px: int = 0,
        despill_strength: float = 0.0,
    ) -> Optional[Image.Image]:
        """Aplica Choke, Feather y Despill sobre el documento o secuencia activa."""
        from app.core.postprocess import refine_alpha_edges

        return self.apply_pil_transform(
            lambda img: refine_alpha_edges(
                img,
                choke_px=choke_px,
                feather_px=feather_px,
                despill_strength=despill_strength,
            ),
            description=f"Refinar bordes (choke={choke_px}, feather={feather_px})",
        )

    def apply_inpaint_selection(
        self,
        selection_mask: Optional[np.ndarray],
        radius: int = 5,
    ) -> Optional[Image.Image]:
        """Aplica inpainting OpenCV sobre la selección en la imagen activa."""
        from app.core.selection import inpaint_selected_region

        if not self.image_state.has_image or selection_mask is None:
            return None
        curr_np = np.array(self.image_state.current_image.convert("RGBA"), dtype=np.uint8)
        out_np = inpaint_selected_region(curr_np, selection_mask, radius=radius)
        out_img = Image.fromarray(out_np, mode="RGBA")
        self.image_state.push_state(out_img, "Rellenar selección (Inpainting)", brush_reference=out_img)
        return out_img

    def build_isolated_subject(self, subject) -> Optional[Image.Image]:
        """Genera la imagen RGBA con el sujeto detectado aislado y el resto transparente."""
        if not self.image_state.has_image or subject is None:
            return None
        high_res_mask = refine_yolo_subject_mask(self.image_state.current_image, subject)
        if high_res_mask is None:
            return None
        curr = self.image_state.current_image.convert("RGBA")
        curr_arr = np.array(curr, dtype=np.uint8)
        curr_arr[..., 3] = np.minimum(curr_arr[..., 3], high_res_mask)
        return Image.fromarray(curr_arr, "RGBA")

    def apply_isolate_subject_sequence(self, subject) -> Optional[Image.Image]:
        """Aísla el sujeto detectado sobre todos los fotogramas de la secuencia activa."""
        if not self.has_sequence or subject is None:
            return None

        desc = f"Aislar {subject.class_name.capitalize()}"
        for st in self.sequence_session.states:
            img = st.current_image
            if img is None:
                continue
            high_res_mask = refine_yolo_subject_mask(img, subject)
            if high_res_mask is None:
                continue
            curr_arr = np.array(img.convert("RGBA"), dtype=np.uint8)
            curr_arr[..., 3] = np.minimum(curr_arr[..., 3], high_res_mask)
            st.push_state(Image.fromarray(curr_arr, "RGBA"), desc)

        return self.image_state.current_image

    def apply_effects(
        self,
        sticker_width: int = 0,
        sticker_color: Tuple[int, int, int] = (255, 255, 255),
        shadow_opacity: float = 0.0,
        shadow_blur: int = 16,
        shadow_offset_x: int = 10,
        shadow_offset_y: int = 12,
        bg_mode: str = "transparent",
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        bokeh_blur: int = 20,
        bg_image: Optional[Image.Image] = None,
        aspect_preset: str = "original",
    ) -> Optional[Image.Image]:
        """Aplica efectos creativos (sticker, sombra, bokeh/fondo, relación de aspecto)."""
        from app.core.effects import (
            apply_drop_shadow,
            apply_sticker_outline,
            compose_background_mode,
        )

        if not self.image_state.has_image:
            return None

        states = self.sequence_session.states if self.has_sequence else [self.image_state]
        for st in states:
            curr = st.current_image
            if curr is None:
                continue
            orig_ref = st.current_brush_reference
            out = curr.convert("RGBA")
            if int(sticker_width) > 0:
                out = apply_sticker_outline(
                    out,
                    width_px=int(sticker_width),
                    color_rgb=sticker_color,
                    expand_canvas=True,
                )
            if float(shadow_opacity) > 0.0:
                out = apply_drop_shadow(
                    out,
                    offset_x=int(shadow_offset_x),
                    offset_y=int(shadow_offset_y),
                    blur_radius=int(shadow_blur),
                    opacity=float(shadow_opacity),
                    expand_canvas=True,
                )
            out = compose_background_mode(
                out,
                mode=bg_mode,
                bg_color=bg_color,
                original_rgb=orig_ref,
                blur_radius=int(bokeh_blur),
                bg_image=bg_image,
                aspect_preset=aspect_preset,
            )
            st.push_state(out, "Efectos / Estudio Creativo")

        return self.image_state.current_image

    def apply_manual_crop(
        self, left: int, top: int, right: int, bottom: int
    ) -> Optional[Image.Image]:
        """Recorta manualmente la imagen activa o todos los fotogramas de la secuencia."""
        from app.core.effects import crop_image_box

        return self.apply_pil_transform(
            lambda img: crop_image_box(img, left, top, right, bottom),
            description="Recorte manual",
            update_brush_reference=True,
        )

    def apply_transform(
        self,
        rotate_deg: int = 0,
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> Optional[Image.Image]:
        """Rota (en pasos de 90° horarios) o voltea la imagen activa o secuencia."""
        from app.core.effects import rotate_flip_image

        return self.apply_pil_transform(
            lambda img: rotate_flip_image(
                img, rotate_deg=rotate_deg, flip_h=flip_h, flip_v=flip_v
            ),
            description="Transformación geométrica",
            update_brush_reference=True,
        )
