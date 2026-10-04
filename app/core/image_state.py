"""
TimoitaseBG - Gestión de estado e historial de imágenes (Undo / Redo / Reset).
"""
from typing import Optional, List
from PIL import Image
from app.config import MAX_HISTORY_STATES


class ImageState:
    """
    Gestiona el ciclo de vida de la imagen cargada y el historial de transformaciones.
    Mantiene la imagen original inmutable y un historial acotado para navegación no destructiva.
    """

    def __init__(self, max_history: int = MAX_HISTORY_STATES):
        self.max_history = max_history
        self._original_image: Optional[Image.Image] = None
        # Guarda tuplas: (current_image, brush_reference_image)
        self._history: List[Tuple[Image.Image, Image.Image]] = []
        self._current_index: int = -1
        self._descriptions: List[str] = []

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
    def has_transparency(self) -> bool:
        """Indica si la imagen actual posee canal alfa con al menos un píxel transparente."""
        if not self.has_image:
            return False
        curr = self._history[self._current_index][0]
        if curr.mode == "RGBA":
            alpha = curr.split()[-1]
            extrema = alpha.getextrema()
            return extrema[0] < 255
        return False

    @property
    def has_background_been_removed(self) -> bool:
        """Indica si el historial actual incluye alguna operación destructiva de fondo o borrador manual."""
        if not self.has_image:
            return False
        for desc in self._descriptions[:self._current_index + 1]:
            if "Quitar fondo" in desc or "Edición Manual" in desc or "Recorte" in desc:
                return True
        return False

    def get_last_transparent_state(self) -> Optional[Image.Image]:
        """
        Busca en el historial hacia atrás el estado más reciente que posea canal alfa con transparencia.
        Permite restaurar la transparencia tras haber aplicado un color de fondo sólido.
        """
        for img, _ in reversed(self._history[: self._current_index]):
            if img.mode == "RGBA":
                alpha = img.split()[-1]
                if alpha.getextrema()[0] < 255:
                    return img.copy()
        if self._original_image and self._original_image.mode == "RGBA":
            alpha = self._original_image.split()[-1]
            if alpha.getextrema()[0] < 255:
                return self._original_image.copy()
        return None

    def load_new_image(self, image: Image.Image) -> Image.Image:
        """
        Carga una nueva imagen, reinicia el historial y establece el original inmutable.
        """
        # Asegurar formato RGB o RGBA
        if image.mode not in ("RGB", "RGBA"):
            converted = image.convert("RGBA" if "A" in image.mode or "transparency" in image.info else "RGB")
        else:
            converted = image.copy()

        self._original_image = converted.copy()
        self._history = [(converted.copy(), converted.copy())]
        self._descriptions = ["Original"]
        self._current_index = 0
        return self.current_image

    def push_state(self, image: Image.Image, description: str = "Modificación", brush_reference: Optional[Image.Image] = None) -> Image.Image:
        """
        Registra una nueva modificación sobre la imagen.
        Descarta cualquier estado posterior si nos encontrábamos en un punto de deshacer.
        """
        if not self.has_image:
            return self.load_new_image(image)

        # Hereda la referencia del pincel del estado actual si no se provee una nueva
        if brush_reference is None:
            brush_reference = self._history[self._current_index][1]

        # Truncar historial si estábamos en medio de un undo
        self._history = self._history[: self._current_index + 1]
        self._descriptions = self._descriptions[: self._current_index + 1]

        # Asegurar copia independiente
        new_frame = image.copy()
        new_brush = brush_reference.copy() if brush_reference is not self._history[self._current_index][1] else brush_reference
        self._history.append((new_frame, new_brush))
        self._descriptions.append(description)

        # Si excede el tamaño máximo, podar los estados más antiguos (sin tocar original)
        if len(self._history) > self.max_history:
            self._history.pop(0)
            self._descriptions.pop(0)

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
        La agrega como un nuevo estado para que esta acción también sea reversible.
        """
        if not self.can_reset():
            return None
        return self.push_state(self._original_image.copy(), description="Recuperado original", brush_reference=self._original_image.copy())

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
