"""
TimoitaseBG - Interfaz de Línea de Comandos (CLI Headless).
Permite automatización y pipelines sin inicializar PySide6.QtWidgets.
Uso:
  python -m app.cli remove input.png output.png --model birefnet-general
  python -m app.cli restore input.png output.png --scale x2
  python -m app.cli batch ./in ./out --autocrop --aspect 512x512
  python -m app.cli models --list
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from app.config import APP_NAME, APP_VERSION, set_gpu_acceleration
from app.core.batch_pipeline import BatchPipelineConfig, process_image_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timoitasebg-cli",
        description=f"{APP_NAME} v{APP_VERSION} - Headless CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_pipeline_flags(p: argparse.ArgumentParser, include_model: bool = True):
        if include_model:
            p.add_argument(
                "--model",
                default="isnet-anime",
                choices=["isnet-anime", "inspyrenet", "birefnet-general", "u2net_human_seg"],
                help="Segmentation model ID",
            )
            p.add_argument("--no-clothing", action="store_true", help="Disable interior clothing protection")
        p.add_argument(
            "--upscale",
            default="none",
            choices=["none", "x2", "x4", "enhance_only"],
            help="Optional Real-ESRGAN upscale step",
        )
        p.add_argument("--autocrop", action="store_true", help="Trim transparent borders")
        p.add_argument("--sticker-width", type=int, default=0, help="Sticker outline width in px")
        p.add_argument("--shadow", type=float, default=0.0, help="Drop shadow opacity (0.0-1.0)")
        p.add_argument(
            "--aspect",
            default="original",
            choices=["original", "512x512", "1:1", "16:9", "9:16"],
            help="Target canvas aspect/size preset",
        )
        p.add_argument("--cpu", action="store_true", help="Force CPU execution")

    p_rm = sub.add_parser("remove", help="Remove background from a single image")
    p_rm.add_argument("input", type=Path)
    p_rm.add_argument("output", type=Path)
    _add_pipeline_flags(p_rm, include_model=True)

    p_rs = sub.add_parser("restore", help="Upscale/restore a single image")
    p_rs.add_argument("input", type=Path)
    p_rs.add_argument("output", type=Path)
    p_rs.add_argument("--scale", default="x2", choices=["x2", "x4", "enhance_only"])
    p_rs.add_argument("--no-tiling", action="store_true", help="Disable tiled GPU inference")
    p_rs.add_argument("--cpu", action="store_true", help="Force CPU execution")

    p_bt = sub.add_parser("batch", help="Process all images in a folder")
    p_bt.add_argument("input_dir", type=Path)
    p_bt.add_argument("output_dir", type=Path)
    _add_pipeline_flags(p_bt, include_model=True)

    p_md = sub.add_parser("models", help="Inspect, verify, download or quantize local models")
    p_md.add_argument("--list", action="store_true", help="List model inventory")
    p_md.add_argument("--verify", metavar="NAME", help="Verify SHA-256 of FILENAME or 'all'")
    p_md.add_argument("--download", metavar="NAME", help="Download a supported ONNX model")
    p_md.add_argument("--quantize", metavar="NAME", help="Quantize a downloaded model to INT8")

    return parser


def _save_image_out(img: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext in (".jpg", ".jpeg") and img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        bg.save(output_path, quality=95)
    else:
        img.save(output_path)


def _cmd_models(args: argparse.Namespace) -> int:
    from app.config import MODELS_DIR
    from app.core.model_fetcher import (
        ensure_model_file,
        get_models_inventory,
        verify_local_model_sha256,
    )
    from app.core.quantizer import quantize_model

    if args.verify:
        targets = (
            [m["filename"] for m in get_models_inventory()]
            if args.verify == "all"
            else [args.verify]
        )
        all_ok = True
        for fname in targets:
            res = verify_local_model_sha256(fname)
            if res is True:
                print(f"[OK] {fname}")
            elif res is False:
                print(f"[FAIL] {fname}")
                all_ok = False
            else:
                print(f"[SKIP] {fname} (missing or unverified upstream hash)")
        return 0 if all_ok else 1

    if args.download:
        path = ensure_model_file(args.download)
        print(f"Downloaded: {path}")
        return 0

    if args.quantize:
        src = MODELS_DIR / args.quantize
        if not src.is_file():
            print(f"Missing model file: {src}", file=sys.stderr)
            return 1
        ok = quantize_model(src)
        return 0 if ok else 1

    for entry in get_models_inventory():
        sha_txt = "SHA256" if entry["has_sha256"] else "no-sha"
        size_mb = entry["total_bytes"] / (1024 * 1024)
        print(f"{entry['filename']:34s} {entry['active_variant']:8s} {size_mb:6.1f} MB [{sha_txt}]")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "cpu", False):
        set_gpu_acceleration(False)

    if args.command == "models":
        return _cmd_models(args)

    if args.command == "restore":
        if not args.input.is_file():
            print(f"Input file not found: {args.input}", file=sys.stderr)
            return 1
        from app.core.restorer import get_image_restorer
        with Image.open(args.input) as raw:
            img = raw.convert("RGBA")
        out = get_image_restorer().restore_image(
            img, scale_mode=args.scale, use_tiling=not args.no_tiling
        )
        _save_image_out(out, args.output)
        print(f"Saved: {args.output}")
        return 0

    cfg = BatchPipelineConfig(
        remove_bg=True,
        upscale_mode=args.upscale,
        autocrop=args.autocrop,
        sticker_width=args.sticker_width,
        shadow_opacity=args.shadow,
        aspect_preset=args.aspect,
    )

    if args.command == "remove":
        if not args.input.is_file():
            print(f"Input file not found: {args.input}", file=sys.stderr)
            return 1
        with Image.open(args.input) as raw:
            img = raw.convert("RGBA")
        out = process_image_pipeline(
            img,
            model_name=args.model,
            enable_clothing_protection=not args.no_clothing,
            pipeline_config=cfg,
        )
        _save_image_out(out, args.output)
        print(f"Saved: {args.output}")
        return 0

    if args.command == "batch":
        if not args.input_dir.is_dir():
            print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files = [f for f in sorted(args.input_dir.iterdir()) if f.is_file() and f.suffix.lower() in valid_exts]
        failures = 0
        for f in files:
            try:
                with Image.open(f) as raw:
                    img = raw.convert("RGBA")
                out = process_image_pipeline(
                    img,
                    model_name=args.model,
                    enable_clothing_protection=not args.no_clothing,
                    pipeline_config=cfg,
                )
                _save_image_out(out, args.output_dir / f"{f.stem}_bgrm.png")
            except Exception as exc:
                failures += 1
                print(f"[ERROR] {f.name}: {exc}", file=sys.stderr)
        print(f"Processed {len(files) - failures}/{len(files)} images.")
        return 0 if failures == 0 else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
