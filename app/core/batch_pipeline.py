"""
TimoitaseBG - Pipeline encadenado puro para procesamiento por lotes y CLI headless.
Sin dependencias de PySide6.
"""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from PIL import Image


@dataclass
class BatchPipelineConfig:
    """Configuración declarativa de etapas encadenadas para lote y CLI."""

    remove_bg: bool = True
    upscale_mode: str = "none"  # "none" | "x2" | "x4" | "enhance_only"
    autocrop: bool = False
    sticker_width: int = 0
    sticker_color: Tuple[int, int, int] = (255, 255, 255)
    shadow_opacity: float = 0.0
    aspect_preset: str = "original"  # "original" | "512x512" | "1:1" | "16:9" | "9:16"
    bg_color: Optional[Tuple[int, int, int]] = None


def process_image_pipeline(
    img: Image.Image,
    model_name: str = "isnet-anime",
    enable_clothing_protection: bool = True,
    pipeline_config: Optional[BatchPipelineConfig] = None,
    remover: Optional[Any] = None,
    restorer: Optional[Any] = None,
) -> Image.Image:
    """Ejecuta el pipeline secuencial (Fondo -> Escalado -> Autocrop -> Efectos/Aspecto)."""
    cfg = pipeline_config or BatchPipelineConfig()
    out = img.convert("RGBA")

    if cfg.remove_bg:
        if remover is None:
            from app.core.bg_remover import get_bg_remover
            remover = get_bg_remover()
        out = remover.remove_background(
            out,
            model_name=model_name,
            enable_clothing_protection=enable_clothing_protection,
        )

    if cfg.upscale_mode in ("x2", "x4", "enhance_only"):
        if restorer is None:
            from app.core.restorer import get_image_restorer
            restorer = get_image_restorer()
        out = restorer.restore_image(out, scale_mode=cfg.upscale_mode)

    if cfg.autocrop:
        from app.core.image_utils import autocrop_image
        cropped = autocrop_image(out)
        if cropped is not None:
            out = cropped

    if int(cfg.sticker_width) > 0:
        from app.core.effects import apply_sticker_outline
        out = apply_sticker_outline(
            out,
            width_px=int(cfg.sticker_width),
            color_rgb=cfg.sticker_color,
            expand_canvas=True,
        )

    if float(cfg.shadow_opacity) > 0.0:
        from app.core.effects import apply_drop_shadow
        out = apply_drop_shadow(
            out,
            opacity=float(cfg.shadow_opacity),
            expand_canvas=True,
        )

    if cfg.bg_color is not None or (cfg.aspect_preset and cfg.aspect_preset != "original"):
        from app.core.effects import compose_background_mode
        mode = "solid" if cfg.bg_color is not None else "transparent"
        out = compose_background_mode(
            out,
            mode=mode,
            bg_color=cfg.bg_color or (255, 255, 255),
            aspect_preset=cfg.aspect_preset,
        )

    return out
