"""
Tests unitarios para ImageState y SequenceSession:
- Cacheo de has_transparency por entrada del historial
- has_previous_transparent_state() sin clonar imágenes RGBA
- Metadatos estructurados (op_type, is_bg_removal, is_upscale)
- reset_to_original() sin doble copia redundante
- Límite de historial en SequenceSession y getters de solo lectura sin .copy()
"""
from unittest.mock import patch

from PIL import Image

from app.core.image_state import (
    DEFAULT_SEQUENCE_MAX_HISTORY,
    ImageState,
    OP_BG_REMOVAL,
    OP_CROP,
    OP_ORIGINAL,
    OP_RESET,
    OP_UPSCALE,
    SequenceSession,
)


def _make_opaque_rgb(color=(255, 0, 0), size=(32, 32)) -> Image.Image:
    return Image.new("RGB", size, color)


def _make_transparent_rgba(size=(32, 32)) -> Image.Image:
    img = Image.new("RGBA", size, (0, 128, 255, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))
    return img


def test_has_previous_transparent_state_does_not_copy_images():
    state = ImageState()
    state.load_new_image(_make_opaque_rgb())
    assert state.has_transparency is False
    assert state.has_previous_transparent_state() is False

    # Estado con transparencia
    state.push_state(_make_transparent_rgba(), description="Quitar fondo (isnet)", op_type=OP_BG_REMOVAL)
    assert state.has_transparency is True
    assert state.has_previous_transparent_state() is False

    # Estado posterior opaco (p. ej. fondo sólido aplicado)
    state.push_state(_make_opaque_rgb((255, 255, 255)), description="Fondo sólido")
    assert state.has_transparency is False

    # Verificar que has_previous_transparent_state() retorna True SIN llamar Image.Image.copy ni getextrema
    with patch.object(Image.Image, "copy", side_effect=AssertionError("No debe clonar imágenes")) as mock_copy, \
         patch.object(Image.Image, "getextrema", side_effect=AssertionError("No debe recalcular getextrema")) as mock_extrema:
        assert state.has_previous_transparent_state() is True
        assert state.has_transparency is False
        mock_copy.assert_not_called()
        mock_extrema.assert_not_called()


def test_has_transparency_is_cached_per_history_entry():
    state = ImageState()
    state.load_new_image(_make_transparent_rgba())
    assert state.has_transparency is True

    with patch.object(Image.Image, "getextrema", side_effect=AssertionError("Debe usar caché")):
        assert state.has_transparency is True
        assert state.has_previous_transparent_state() is True


def test_structured_metadata_flags_without_spanish_substrings():
    state = ImageState()
    state.load_new_image(_make_opaque_rgb())
    assert state.current_op_type == OP_ORIGINAL
    assert state.has_background_been_removed is False

    # Descripción arbitraria en inglés con op_type estructurado
    state.push_state(_make_transparent_rgba(), description="Remove BG (EN)", op_type=OP_BG_REMOVAL)
    assert state.current_op_type == OP_BG_REMOVAL
    assert state.current_metadata is not None
    assert state.current_metadata.is_bg_removal is True
    assert state.current_metadata.is_upscale is False
    assert state.has_background_been_removed is True

    # Upscale estructurado
    state.push_state(_make_transparent_rgba(size=(64, 64)), description="2x Super-Res", op_type=OP_UPSCALE)
    assert state.current_op_type == OP_UPSCALE
    assert state.current_metadata is not None
    assert state.current_metadata.is_upscale is True
    assert state.has_background_been_removed is True

    # Compatibilidad hacia atrás con descripciones en español sin op_type explícito
    legacy = ImageState()
    legacy.load_new_image(_make_opaque_rgb())
    legacy.push_state(_make_transparent_rgba(), description="Recorte al contenido (16x16)")
    assert legacy.current_op_type == OP_CROP
    assert legacy.has_background_been_removed is True


def test_reset_to_original_avoids_redundant_double_copy():
    state = ImageState()
    state.load_new_image(_make_opaque_rgb((10, 20, 30)))
    state.push_state(_make_transparent_rgba(), description="Quitar fondo")

    # Solo debe ocurrir 1 copia (la retornada al llamador por current_image), no 2 ni 3
    orig_copy = Image.Image.copy
    copy_calls = 0

    def _counting_copy(self_img):
        nonlocal copy_calls
        copy_calls += 1
        return orig_copy(self_img)

    with patch.object(Image.Image, "copy", _counting_copy):
        restored = state.reset_to_original()

    assert restored is not None
    assert state.current_op_type == OP_RESET
    assert copy_calls == 1
    assert state._history[-1][0] is state._original_image
    assert state._history[-1][1] is state._original_image


def test_sequence_session_limits_max_history_and_readonly_getters():
    frames = [_make_opaque_rgb((i * 20, 0, 0)) for i in range(4)]
    session = SequenceSession(frames)

    assert session.max_history == DEFAULT_SEQUENCE_MAX_HISTORY
    assert session.max_history <= 5
    for st in session.states:
        assert st.max_history == session.max_history

    # Empujar más estados que max_history y verificar poda
    target_state = session.states[0]
    for step in range(12):
        target_state.push_state(_make_opaque_rgb((step * 10, 50, 0)), description=f"Paso {step}")
    assert len(target_state._history) == session.max_history
    assert len(target_state._descriptions) == session.max_history
    assert len(target_state._metadata) == session.max_history

    # Getter de solo lectura no debe invocar .copy()
    with patch.object(Image.Image, "copy", side_effect=AssertionError("No debe clonar frames")):
        readonly_frames = session.get_all_current_frames_readonly()
        assert len(readonly_frames) == 4
        assert readonly_frames[0] is session.states[0].peek_current_image()
