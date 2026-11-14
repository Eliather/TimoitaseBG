import io
import logging
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PySide6.QtCore import QByteArray, QMimeData
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog, QMessageBox

from app.i18n import tr
from app.core.image_utils import pil_to_qimage, qimage_to_pil
from app.core.video_processor import VIDEO_EXTENSIONS
from app.ui.components import FramelessPopup

logger = logging.getLogger(__name__)

_IMAGE_GLOBS = "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.apng"
_VIDEO_GLOBS = "*.mp4 *.mov *.webm *.mkv *.avi *.m4v *.wmv *.flv *.mpeg *.mpg *.ogv *.3gp *.ts"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".apng"}

_SEQUENCE_SAVE_FILTERS = {
    ".webm": "Video WebM Libre - VP8/VP9 (*.webm)",
    ".mkv": "Video Matroska Libre - FFV1/VP8 (*.mkv)",
    ".mov": "Video QuickTime MOV - MPEG-4 Libre (*.mov)",
    ".mp4": "Video MP4 - MPEG-4 Libre (*.mp4)",
    ".avi": "Video AVI - Motion JPEG Libre (*.avi)",
    ".webp": "WebP Animado HD - Alfa 8 bits (*.webp)",
    ".gif": "GIF Animado Transparente (*.gif)",
    ".apng": "APNG Animado - Alfa 8 bits (*.apng *.png)",
    ".zip": "Secuencia PNG Transparente en ZIP (*.zip)",
    ".png": "PNG Fotograma Actual (*.png)",
}


def prompt_open_image(window: Any) -> None:
    """Abre un diálogo para seleccionar imagen, animación o video."""
    filters = (
        f"Imágenes y Videos ({_IMAGE_GLOBS} {_VIDEO_GLOBS});;"
        f"Imágenes y Animaciones ({_IMAGE_GLOBS});;"
        f"Videos ({_VIDEO_GLOBS});;"
        "Todos los archivos (*.*)"
    )
    path, _ = QFileDialog.getOpenFileName(window, "Seleccionar Imagen o Video", "", filters)
    if path:
        window.open_image_file(path)


def open_image_file(window: Any, file_path: str) -> None:
    """Carga una imagen estática, animación o video en el documento y sincroniza la UI."""
    try:
        def _progress_status(msg: str) -> None:
            window._set_status_raw(msg)
            QGuiApplication.processEvents()

        info = window.doc.open_media(file_path, status_cb=_progress_status)
        kind = info["kind"]
        if kind == "missing":
            return

        if window.gif_timer.isActive():
            window.gif_timer.stop()

        path = info["path"]
        is_seq = kind in ("video", "animation")
        if hasattr(window, "gif_bar"):
            if is_seq:
                window.gif_bar.configure_sequence(
                    window.sequence_session.total_frames,
                    window.sequence_session.fps,
                )
                visible = info["frames_count"] > 1 if kind == "video" else True
                window.gif_bar.setVisible(visible)
            else:
                window.gif_bar.setVisible(False)

        if is_seq:
            window._navigate_gif_frame(0)

        window._reset_workspace_ui(info["image"], reset_zoom=True)

        if kind == "video":
            window._set_status_raw(
                f"🎬 Video {info['fmt_name']} cargado: {path.name} ({info['frames_count']} fotogramas @ {info['fps']:.1f} FPS)"
            )
        elif kind == "animation":
            window._set_status_raw(
                f"{info['fmt_name']} animado cargado: {path.name} ({info['frames_count']} fotogramas)"
            )
        else:
            window._set_status("status_loaded", name=path.name)
    except Exception as e:
        QMessageBox.critical(window, "Error al abrir archivo", f"No se pudo cargar el archivo:\n{e}")


def copy_to_clipboard(window: Any) -> None:
    """Copia la imagen actual al portapapeles preservando transparencia alfa."""
    built = window.doc.build_clipboard_png(bg_color_rgb=window.canvas.bg_color_rgb)
    if built is None:
        window._set_status("status_open_first")
        return
    png_bytes, img = built

    mime_data = QMimeData()
    byte_array = QByteArray(png_bytes)
    mime_data.setData("image/png", byte_array)
    mime_data.setData("application/x-qt-image", byte_array)
    mime_data.setImageData(pil_to_qimage(img))

    QGuiApplication.clipboard().setMimeData(mime_data)
    window._set_status("status_copied_clipboard")


def _load_pasted_pil(window: Any, pil_img: Image.Image) -> None:
    """Detiene cualquier reproducción activa y carga una imagen PIL pegada."""
    if window.gif_timer.isActive():
        window.gif_timer.stop()
    window.sequence_session = None
    if hasattr(window, "gif_bar"):
        window.gif_bar.setVisible(False)
    window.load_new_pil_image(pil_img, source_name="Imagen desde portapapeles", filepath=None)


def paste_from_clipboard(window: Any) -> None:
    """Pega imagen o archivo multimedia desde el portapapeles (Ctrl+V)."""
    clipboard = QGuiApplication.clipboard()
    mime_data = clipboard.mimeData()

    if mime_data.hasFormat("image/png"):
        try:
            png_bytes = bytes(mime_data.data("image/png"))
            if png_bytes:
                pil_img = Image.open(io.BytesIO(png_bytes))
                _load_pasted_pil(window, pil_img)
                return
        except Exception as e:
            logger.warning("Error al leer stream image/png del portapapeles: %s", e)

    if mime_data.hasImage():
        qimg = clipboard.image()
        if not qimg.isNull() and qimg.width() > 0 and qimg.height() > 0:
            try:
                _load_pasted_pil(window, qimage_to_pil(qimg))
                return
            except Exception as e:
                logger.warning("Error al procesar imagen del portapapeles: %s", e)

    if mime_data.hasUrls():
        urls = mime_data.urls()
        if urls:
            local_path = Path(urls[0].toLocalFile())
            valid_extensions = _IMAGE_EXTENSIONS | set(VIDEO_EXTENSIONS)
            if local_path.is_file() and local_path.suffix.lower() in valid_extensions:
                window.open_image_file(str(local_path))
                return

    window._set_status("status_no_clipboard")


def _pick_save_target(window: Any, is_animated_seq: bool) -> tuple[str, str]:
    """Abre el diálogo de guardado adaptado a secuencia animada o imagen estática."""
    if not is_animated_seq:
        stem = window.current_filepath.stem if window.current_filepath else "anime_processed"
        filters = "PNG transparente (*.png);;GIF (*.gif);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)"
        return QFileDialog.getSaveFileName(window, "Guardar Imagen Como", f"{stem}_edit.png", filters)

    orig_ext = window.current_filepath.suffix.lower() if window.current_filepath else ".webm"
    is_vid = bool(getattr(window.sequence_session, "is_video", False) or orig_ext in VIDEO_EXTENSIONS)
    if orig_ext in (".webm", ".mkv", ".mov", ".mp4", ".avi", ".webp", ".gif", ".apng"):
        default_ext = orig_ext
    elif orig_ext == ".png":
        default_ext = ".apng"
    else:
        default_ext = ".webm" if is_vid else ".webp"

    stem = window.current_filepath.stem if window.current_filepath else "video_processed"
    filters = ";;".join(_SEQUENCE_SAVE_FILTERS.values())
    initial_filter = _SEQUENCE_SAVE_FILTERS.get(default_ext, _SEQUENCE_SAVE_FILTERS[".webm" if is_vid else ".webp"])
    dialog_title = "Guardar Video / Animación Como" if is_vid else "Guardar Animación Como"
    return QFileDialog.getSaveFileName(window, dialog_title, f"{stem}_edit{default_ext}", filters, initial_filter)


def _notify_saved(window: Any, saved_path: Optional[Path]) -> None:
    """Actualiza barra de estado y muestra confirmación tras guardar."""
    if saved_path is None:
        return
    window._set_status("status_saved", name=saved_path.name)
    FramelessPopup(
        window,
        tr("popup_save_success_title"),
        tr("popup_save_success_msg", path=str(saved_path)),
    ).exec()


def prompt_save_image(window: Any) -> None:
    """Guarda la imagen procesada, secuencia animada o video en disco."""
    if not window.image_state.has_image:
        FramelessPopup(window, tr("popup_no_image_title"), tr("popup_no_image_msg")).exec()
        return

    is_animated_seq = bool(window.sequence_session and window.sequence_session.total_frames > 1)
    path, selected_filter = _pick_save_target(window, is_animated_seq)
    if not path:
        return

    try:
        if is_animated_seq:
            window.set_processing(True, tr("status_saving_sequence"))
            window.progress_bar.setRange(0, 0)
            if hasattr(window, "btn_cancel"):
                window.btn_cancel.setVisible(False)

            def _finish_save_ui() -> None:
                window.progress_bar.setRange(0, 100)
                window.set_processing(False)

            def _on_seq_saved(saved_path: Optional[Path]) -> None:
                _finish_save_ui()
                _notify_saved(window, saved_path)

            def _on_seq_save_failed(err_msg: str) -> None:
                _finish_save_ui()
                FramelessPopup(window, tr("popup_error_save_title"), tr("popup_error_save_msg", error=err_msg)).exec()

            window.ai.start_save_sequence(
                save_fn=lambda: window.doc.save_to_path(
                    path,
                    selected_filter=selected_filter,
                    bg_color_rgb=window.canvas.bg_color_rgb,
                ),
                on_saved=_on_seq_saved,
                on_failed=_on_seq_save_failed,
            )
            return

        saved_path = window.doc.save_to_path(
            path,
            selected_filter=selected_filter,
            bg_color_rgb=window.canvas.bg_color_rgb,
        )
        _notify_saved(window, saved_path)
    except Exception as e:
        FramelessPopup(window, tr("popup_error_save_title"), tr("popup_error_save_msg", error=str(e))).exec()
