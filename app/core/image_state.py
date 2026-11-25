"""
TimoitaseBG - Gestión de estado e historial de imágenes (Undo / Redo / Reset).
"""
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
from PIL import Image
from app.config import MAX_HISTORY_STATES

# Tipos de operación estructurados para evitar acoplamiento a textos traducidos
OP_ORIGINAL = "original"
OP_BG_REMOVAL = "bg_removal"
OP_MANUAL_EDIT = "manual_edit"
OP_CROP = "crop"
OP_UPSCALE = "upscale"
OP_ENHANCE = "enhance"
OP_RESET = "reset"
OP_OTHER = "other"

DEFAULT_SEQUENCE_MAX_HISTORY = 5


@dataclass(frozen=True)
class HistoryEntryMeta:
    """Metadatos estructurados y caché ligera por entrada del historial."""

    has_transparency: bool
    op_type: str = OP_OTHER
    is_bg_removal: bool = False
    is_upscale: bool = False


def _image_has_transparency(img: Optional[Image.Image]) -> bool:
    """Comprueba si una imagen PIL RGBA contiene al menos un píxel transparente."""
    if img is None or img.mode != "RGBA":
        return False
    extrema = img.getextrema()
    return extrema[3][0] < 255


def _infer_op_metadata(
    description: str,
    op_type: Optional[str] = None,
    is_bg_removal: Optional[bool] = None,
    is_upscale: Optional[bool] = None,
) -> Tuple[str, bool, bool]:
    """Resuelve tipo de operación y flags estructurados manteniendo compatibilidad con descripciones."""
    desc = description or ""
    if op_type is not None:
        resolved_type = op_type
    elif "Quitar fondo" in desc or "sin fondo" in desc or desc.startswith("Aislar "):
        resolved_type = OP_BG_REMOVAL
    elif "Edición Manual" in desc:
        resolved_type = OP_MANUAL_EDIT
    elif "Recorte" in desc:
        resolved_type = OP_CROP
    elif "Escalado" in desc:
        resolved_type = OP_UPSCALE
    elif "Mejora de nitidez" in desc:
        resolved_type = OP_ENHANCE
    elif "Recuperado original" in desc:
        resolved_type = OP_RESET
    elif desc == "Original":
        resolved_type = OP_ORIGINAL
    else:
        resolved_type = OP_OTHER

    resolved_bg_rm = (
        bool(is_bg_removal)
        if is_bg_removal is not None
        else resolved_type in (OP_BG_REMOVAL, OP_MANUAL_EDIT, OP_CROP)
    )
    resolved_upscale = (
        bool(is_upscale)
        if is_upscale is not None
        else resolved_type in (OP_UPSCALE, OP_ENHANCE)
    )
    return resolved_type, resolved_bg_rm, resolved_upscale


class ImageState:
    """
    Gestiona el ciclo de vida de la imagen cargada y el historial de transformaciones.
    Mantiene la imagen original inmutable y un historial acotado para navegación no destructiva.
    """

    def __init__(self, max_history: int = MAX_HISTORY_STATES):
        self.max_history = max(1, int(max_history))
        self._original_image: Optional[Image.Image] = None
        self._original_has_transparency: bool = False
        # Guarda tuplas: (current_image, brush_reference_image)
        self._history: List[Tuple[Image.Image, Image.Image]] = []
        self._current_index: int = -1
        self._descriptions: List[str] = []
        self._metadata: List[HistoryEntryMeta] = []

    @property
    def has_image(self) -> bool:
        """Indica si hay una imagen activa en memoria."""
        return self._current_index >= 0 and self._current_index < len(self._history)

    @property
    def original_image(self) -> Optional[Image.Image]:
        """Copia inmutable de la imagen original inicial."""
        return self._original_image.copy() if self._original_image else None

    @property
    def current_image(self) -> Optional[Image.Image]:
        """Imagen del estado actual apuntado en el historial."""
        if not self.has_image:
            return None
        return self._history[self._current_index][0].copy()

    def peek_current_image(self) -> Optional[Image.Image]:
        """Retorna una referencia de solo lectura al estado actual sin clonar la imagen."""
        if not self.has_image:
            return None
        return self._history[self._current_index][0]

    @property
    def current_brush_reference(self) -> Optional[Image.Image]:
        """Imagen original transformada correspondiente al estado actual (para el pincel)."""
        if not self.has_image:
            return None
        return self._history[self._current_index][1].copy()

    @property
    def current_description(self) -> str:
        """Descripción del estado actual (ej: 'Cargada', 'Fondo quitado')."""
        if 0 <= self._current_index < len(self._descriptions):
            return self._descriptions[self._current_index]
        return ""

    @property
    def current_metadata(self) -> Optional[HistoryEntryMeta]:
        """Metadatos estructurados del estado actual."""
        if 0 <= self._current_index < len(self._metadata):
            return self._metadata[self._current_index]
        return None

    @property
    def current_op_type(self) -> str:
        """Tipo de operación estructurado del estado actual."""
        meta = self.current_metadata
        return meta.op_type if meta is not None else ""

    @property
    def has_transparency(self) -> bool:
        """Indica si la imagen actual posee canal alfa con al menos un píxel transparente (cacheado)."""
        if not self.has_image:
            return False
        if self._current_index < len(self._metadata):
            return self._metadata[self._current_index].has_transparency
        return _image_has_transparency(self._history[self._current_index][0])

    @property
    def has_background_been_removed(self) -> bool:
        """Indica si el historial actual incluye alguna operación destructiva de fondo o borrador manual."""
        if not self.has_image:
            return False
        return any(m.is_bg_removal for m in self._metadata[: self._current_index + 1])

    def has_previous_transparent_state(self) -> bool:
        """
        Comprueba si existe algún estado previo con transparencia en el historial o en el original
        SIN clonar (`.copy()`) ninguna imagen ni recalcular `getextrema()`.
        """
        if not self.has_image:
            return False
        for meta in reversed(self._metadata[: self._current_index]):
            if meta.has_transparency:
                return True
        return self._original_has_transparency

    def get_last_transparent_state(self) -> Optional[Image.Image]:
        """
        Busca en el historial hacia atrás el estado más reciente que posea canal alfa con transparencia.
        Permite restaurar la transparencia tras haber aplicado un color de fondo sólido.
        """
        if not self.has_image:
            return None
        for idx in range(self._current_index - 1, -1, -1):
            if self._metadata[idx].has_transparency:
                return self._history[idx][0].copy()
        if self._original_image is not None and self._original_has_transparency:
            return self._original_image.copy()
        return None

    def load_new_image(self, image: Image.Image) -> Image.Image:
        """
        Carga una nueva imagen, reinicia el historial y establece el original inmutable.
        Optimización de RAM: comparte la referencia base inmutable entre _original_image,
        el estado inicial y brush_reference (las propiedades ya entregan .copy() bajo demanda).
        """
        if image.mode not in ("RGB", "RGBA"):
            converted = image.convert("RGBA" if "A" in image.mode or "transparency" in image.info else "RGB")
        else:
            converted = image.copy()

        orig_has_alpha = _image_has_transparency(converted)
        self._original_image = converted
        self._original_has_transparency = orig_has_alpha
        self._history = [(converted, converted)]
        self._descriptions = ["Original"]
        self._metadata = [
            HistoryEntryMeta(
                has_transparency=orig_has_alpha,
                op_type=OP_ORIGINAL,
                is_bg_removal=False,
                is_upscale=False,
            )
        ]
        self._current_index = 0
        return self.current_image

    def push_state(
        self,
        image: Image.Image,
        description: str = "Modificación",
        brush_reference: Optional[Image.Image] = None,
        *,
        op_type: Optional[str] = None,
        is_bg_removal: Optional[bool] = None,
        is_upscale: Optional[bool] = None,
    ) -> Image.Image:
        """
        Registra una nueva modificación sobre la imagen.
        Descarta cualquier estado posterior si nos encontrábamos en un punto de deshacer.
        """
        if not self.has_image:
            return self.load_new_image(image)

        # Hereda la referencia inmutable del pincel del estado actual si no se provee una nueva
        current_brush_ref = self._history[self._current_index][1]
        if brush_reference is None:
            new_brush = current_brush_ref
        elif brush_reference is current_brush_ref or brush_reference is self._original_image:
            new_brush = brush_reference
        else:
            new_brush = brush_reference.copy()

        # Truncar historial si estábamos en medio de un undo
        self._history = self._history[: self._current_index + 1]
        self._descriptions = self._descriptions[: self._current_index + 1]
        self._metadata = self._metadata[: self._current_index + 1]

        if image is self._original_image:
            new_frame = self._original_image
            has_alpha = self._original_has_transparency
        else:
            new_frame = image.copy()
            has_alpha = _image_has_transparency(new_frame)

        resolved_type, resolved_bg_rm, resolved_upscale = _infer_op_metadata(
            description,
            op_type=op_type,
            is_bg_removal=is_bg_removal,
            is_upscale=is_upscale,
        )

        self._history.append((new_frame, new_brush))
        self._descriptions.append(description)
        self._metadata.append(
            HistoryEntryMeta(
                has_transparency=has_alpha,
                op_type=resolved_type,
                is_bg_removal=resolved_bg_rm,
                is_upscale=resolved_upscale,
            )
        )

        # Si excede el tamaño máximo, podar los estados más antiguos (sin tocar original)
        if len(self._history) > self.max_history:
            self._history.pop(0)
            self._descriptions.pop(0)
            self._metadata.pop(0)

        self._current_index = len(self._history) - 1
        return self.current_image

    def undo(self) -> Optional[Image.Image]:
        """Retrocede un paso en el historial."""
        if not self.can_undo():
            return None
        self._current_index -= 1
        return self.current_image

    def redo(self) -> Optional[Image.Image]:
        """Avanza un paso en el historial."""
        if not self.can_redo():
            return None
        self._current_index += 1
        return self.current_image

    def reset_to_original(self) -> Optional[Image.Image]:
        """
        Restaura la imagen original tal como fue cargada.
        La agrega como un nuevo estado para que esta acción también sea reversible
        sin realizar copias redundantes de `_original_image`.
        """
        if not self.can_reset():
            return None
        return self.push_state(
            self._original_image,
            description="Recuperado original",
            brush_reference=self._original_image,
            op_type=OP_RESET,
            is_bg_removal=False,
            is_upscale=False,
        )

    def can_undo(self) -> bool:
        """Retorna True si hay al menos un estado previo accesible."""
        return self.has_image and self._current_index > 0

    def can_redo(self) -> bool:
        """Retorna True si hay al menos un estado posterior accesible."""
        return self.has_image and self._current_index < len(self._history) - 1

    def can_reset(self) -> bool:
        """Retorna True si la imagen original puede ser restaurada."""
        return self._original_image is not None and self.has_image

    def get_status_summary(self) -> str:
        """Retorna un texto resumen de posición en el historial."""
        if not self.has_image:
            return "Sin imagen"
        return f"Paso {self._current_index + 1}/{len(self._history)}: {self.current_description}"


class SequenceSession:
    """Administra múltiples estados ImageState para fotogramas de un GIF animado o Video."""

    def __init__(
        self,
        frames: List[Image.Image],
        durations: Optional[List[int]] = None,
        loop: int = 0,
        fps: Optional[float] = None,
        is_video: bool = False,
        max_history: int = DEFAULT_SEQUENCE_MAX_HISTORY,
    ):
        self.max_history = max(1, int(max_history))
        self.states: List[ImageState] = []
        for frame in frames:
            st = ImageState(max_history=self.max_history)
            st.load_new_image(frame)
            self.states.append(st)
        self.durations = durations or [80] * len(frames)
        self.loop = loop
        self.is_video = is_video
        if fps is not None and fps > 0:
            self.fps = float(fps)
        else:
            avg_ms = sum(self.durations) / max(1, len(self.durations))
            self.fps = round(1000.0 / max(15.0, avg_ms), 2)
        self.current_index = 0
        self.trim_start: int = 0
        self.trim_end: int = max(0, len(self.states) - 1)
        self.target_fps: float = float(self.fps)

    @property
    def active_state(self) -> ImageState:
        return self.states[self.current_index]

    @property
    def total_frames(self) -> int:
        return len(self.states)

    def set_trim_range(self, start: int, end: int) -> Tuple[int, int]:
        total = self.total_frames
        if total <= 0:
            self.trim_start = 0
            self.trim_end = 0
            return (0, 0)
        s = max(0, min(total - 1, int(start)))
        e = max(s, min(total - 1, int(end)))
        self.trim_start = s
        self.trim_end = e
        return (s, e)

    def set_target_fps(self, fps: float) -> float:
        clamped = max(1.0, min(120.0, float(fps)))
        self.target_fps = round(clamped, 2)
        return self.target_fps

    def get_export_frames_and_durations(self) -> Tuple[List[Image.Image], List[int], float]:
        total = self.total_frames
        if total == 0:
            return [], [], float(self.target_fps or 24.0)
        s = max(0, min(total - 1, int(self.trim_start)))
        e = max(s, min(total - 1, int(self.trim_end)))
        frames = [st.current_image for st in self.states[s : e + 1] if st.has_image]
        fps_out = float(self.target_fps if self.target_fps > 0 else self.fps)
        if abs(fps_out - float(self.fps)) < 1e-2 and len(self.durations) == total:
            durations = list(self.durations[s : e + 1])
        else:
            frame_ms = max(15, int(round(1000.0 / max(1.0, fps_out))))
            durations = [frame_ms] * len(frames)
        return frames, durations, fps_out

    def iter_current_frames_readonly(self) -> Iterator[Image.Image]:
        """Itera sobre los fotogramas actuales en modo solo lectura sin clonar (.copy())."""
        for st in self.states:
            img = st.peek_current_image()
            if img is not None:
                yield img

    def get_all_current_frames_readonly(self) -> List[Image.Image]:
        """Retorna referencias de solo lectura a todos los fotogramas actuales sin clonar."""
        return list(self.iter_current_frames_readonly())

    def get_all_current_frames(self) -> List[Image.Image]:
        return [st.current_image for st in self.states if st.has_image]
