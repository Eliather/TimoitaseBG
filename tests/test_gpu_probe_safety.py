"""
Tests unitarios de seguridad para detección GPU/DXGI, opciones DirectML,
validación CLI, efectos creativos y utilidades de imagen.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.core import hardware
from app.core.effects import apply_drop_shadow, apply_sticker_outline, compose_background_mode


def test_get_optimized_session_options_dml_and_default():
    import onnxruntime as ort

    opts_default = hardware.get_optimized_session_options()
    assert opts_default is not None

    opts_dml = hardware.get_optimized_session_options(use_dml=True)
    assert opts_dml is not None
    assert opts_dml.enable_mem_pattern is False
    assert opts_dml.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


def test_get_best_dml_device_id_prefers_dxgi_dedicated_vram(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "_best_dml_device_id", None)
    monkeypatch.setattr(hardware, "_is_force_cpu", lambda: False)
    monkeypatch.setattr(hardware, "_is_gpu_accel_enabled", lambda: True)

    fake_adapters = [
        {"index": 0, "name": "Intel(R) UHD Graphics 770", "dedicated_vram": 128 * 1024 * 1024, "flags": 0},
        {"index": 1, "name": "Intel(R) Arc(TM) B580 Graphics", "dedicated_vram": 12 * 1024 * 1024 * 1024, "flags": 0},
    ]
    monkeypatch.setattr(hardware, "_enumerate_dxgi_adapters", lambda: fake_adapters)

    device_id = hardware.get_best_dml_device_id()
    assert device_id == 1
    monkeypatch.setattr(hardware, "_best_dml_device_id", None)


def test_get_best_dml_device_id_registry_fallback_arc_priority(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "_best_dml_device_id", None)
    monkeypatch.setattr(hardware, "_is_force_cpu", lambda: False)
    monkeypatch.setattr(hardware, "_is_gpu_accel_enabled", lambda: True)
    monkeypatch.setattr(hardware, "_enumerate_dxgi_adapters", lambda: [])
    monkeypatch.setattr(
        hardware,
        "_query_windows_gpus_fast",
        lambda: ["Intel(R) Iris(R) Xe Graphics", "Intel(R) Arc(TM) A770 Graphics"],
    )

    device_id = hardware.get_best_dml_device_id()
    assert device_id == 1
    monkeypatch.setattr(hardware, "_best_dml_device_id", None)


@pytest.mark.parametrize(
    "gpu_name,expected_integrated",
    [
        ("Intel(R) Arc(TM) B580 Graphics", False),
        ("NVIDIA Quadro RTX 5000", False),
        ("NVIDIA RTX A4000", False),
        ("NVIDIA TITAN RTX", False),
        ("AMD Radeon Pro W6800", False),
        ("AMD Radeon RX 7900 XT", False),
        ("Intel(R) UHD Graphics 770", True),
        ("Intel(R) UHD Graphics NextGen", True),  # 'xt' en 'NextGen' no debe dar falso positivo
        ("AMD Radeon(TM) Graphics", True),
    ],
)
def test_is_integrated_gpu_classifications(monkeypatch, gpu_name, expected_integrated):
    monkeypatch.setattr(hardware, "is_arm_architecture", lambda: False)
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "_resolve_gpu_name", lambda: gpu_name)
    assert hardware.is_integrated_gpu() is expected_integrated


def test_linux_rocm_smi_and_lspci_parsing(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "is_arm_architecture", lambda: False)
    monkeypatch.setattr(hardware, "_cached_gpu_name", None)

    rocm_out = (
        b"======================== ROCm System Management Interface ========================\n"
        b"GPU[0]\t\t: Card Vendor:\tAdvanced Micro Devices, Inc.\n"
        b"GPU[0]\t\t: Card series:\tAMD Radeon RX 7900 XTX\n"
    )

    def fake_check_output(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError()
        if cmd[0] == "rocm-smi":
            return rocm_out
        raise FileNotFoundError()

    with patch("subprocess.check_output", side_effect=fake_check_output):
        assert hardware.get_gpu_name() == "AMD Radeon RX 7900 XTX"

    monkeypatch.setattr(hardware, "_cached_gpu_name", None)
    lspci_out = (
        b"00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 770 (rev 0c)\n"
        b"01:00.0 VGA compatible controller: NVIDIA Corporation AD104 [GeForce RTX 4070] (rev a1)\n"
    )

    def fake_check_output_lspci(cmd, **kwargs):
        if cmd[0] == "lspci":
            return lspci_out
        raise FileNotFoundError()

    with patch("subprocess.check_output", side_effect=fake_check_output_lspci):
        assert "GeForce RTX 4070" in hardware.get_gpu_name()

    monkeypatch.setattr(hardware, "_cached_gpu_name", None)


def test_cli_validation_and_safe_quantize_path(tmp_path: Path):
    from app.cli import _build_parser, main

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["remove", "in.png", "out.png", "--sticker-width", "250"])
    with pytest.raises(SystemExit):
        parser.parse_args(["remove", "in.png", "out.png", "--sticker-width", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["remove", "in.png", "out.png", "--shadow", "1.5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["remove", "in.png", "out.png", "--shadow", "-0.1"])

    args_ok = parser.parse_args(["remove", "in.png", "out.png", "--sticker-width", "12", "--shadow", "0.5"])
    assert args_ok.sticker_width == 12
    assert args_ok.shadow == pytest.approx(0.5)

    # Path traversal en models --quantize debe ser rechazado con código 1
    rc = main(["models", "--quantize", "../outside.onnx"])
    assert rc == 1


def test_effects_outline_shadow_and_bokeh_clamping():
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            img.putpixel((x, y), (200, 100, 50, 255))

    outlined = apply_sticker_outline(img, width_px=4, color_rgb=(255, 255, 255))
    assert outlined.mode == "RGBA"
    assert outlined.size == (44, 44)

    shadowed = apply_drop_shadow(img, offset_x=3, offset_y=4, blur_radius=5, opacity=0.6)
    assert shadowed.mode == "RGBA"
    assert shadowed.size[0] > img.width and shadowed.size[1] > img.height

    bg_orig = Image.new("RGB", (32, 32), (40, 80, 120))
    bokeh = compose_background_mode(img, mode="bokeh", original_rgb=bg_orig, blur_radius=999)
    assert bokeh.mode == "RGBA"
    assert bokeh.size == (32, 32)
