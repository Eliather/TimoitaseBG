"""Tests unitarios para BackgroundRemover, ImageRestorer, Postprocess, Quantizer, ModelFetcher y download_releases."""
import hashlib
import io
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.core.bg_remover import BackgroundRemover
from app.core.postprocess import (
    _postprocess_inspyrenet_mask,
    fuse_clothing_protection_core_only,
    postprocess_inspyrenet,
    refine_alpha_edges,
)
from app.core.restorer import ImageRestorer


def test_bg_remover_unload_eviction_and_get_session():
    remover = BackgroundRemover()
    sess_biref = MagicMock(name="biref")
    sess_isnet = MagicMock(name="isnet")
    sess_human = MagicMock(name="human")

    def fake_load(model_name, providers, sess_opts=None, progress_cb=None):
        return {
            "birefnet-general": sess_biref,
            "isnet-anime": sess_isnet,
            "u2net_human_seg": sess_human,
        }[model_name]

    with patch("app.core.model_fetcher.load_rembg_session", side_effect=fake_load):
        s1 = remover.get_session("birefnet-general", force_cpu=True)
        assert s1 is sess_biref
        assert "rembg_cpu_birefnet-general" in remover._sessions

        # Cargar auxiliar u2net_human_seg no debe evictar birefnet-general
        sh = remover.get_session("u2net_human_seg", force_cpu=True)
        assert sh is sess_human
        assert "rembg_cpu_birefnet-general" in remover._sessions

        # Cambiar de modelo primario en CPU debe evictar birefnet-general pero mantener u2net_human_seg
        s2 = remover.get_session("isnet-anime", force_cpu=True)
        assert s2 is sess_isnet
        assert "rembg_cpu_isnet-anime" in remover._sessions
        assert "rembg_cpu_birefnet-general" not in remover._sessions
        assert "rembg_cpu_u2net_human_seg" in remover._sessions

    remover.unload_sessions()
    assert remover._sessions == {}


def test_restorer_unload_edge_bleeding_igpu_tile_size_and_uniform_padding():
    restorer = ImageRestorer()
    # Imagen RGBA con sujeto rojo brillante en el centro y fondo negro transparente (0,0,0,0)
    rgba = np.zeros((300, 300, 4), dtype=np.uint8)
    rgba[50:250, 50:250] = [240, 40, 20, 255]
    img = Image.fromarray(rgba, mode="RGBA")

    captured_tiles = []

    class FakeSession:
        def get_inputs(self):
            m = MagicMock()
            m.name = "input"
            return [m]

        def get_outputs(self):
            m = MagicMock()
            m.name = "output"
            return [m]

        def run(self, _out_names, feed):
            tile = feed["input"]
            captured_tiles.append(tile.copy())
            # Escalado x4 sintético por repetición
            up = np.repeat(np.repeat(tile, 4, axis=2), 4, axis=3)
            return [up]

    restorer._cpu_session = FakeSession()

    # Simular iGPU: debe forzar tile_size <= 256 y padding uniforme en todos los tiles
    with patch("app.config.is_integrated_gpu", return_value=True):
        out = restorer._restore_image_impl(
            img,
            scale_mode="enhance_only",
            use_tiling=False,
            tile_size=512,
            tile_pad=16,
            force_cpu=True,
        )

    assert out.mode == "RGBA"
    assert out.size == (300, 300)
    # 300x300 con tile_size=256 -> 2x2 = 4 tiles, TODOS con forma uniforme (1, 3, 288, 288)
    assert len(captured_tiles) == 4
    assert all(t.shape == (1, 3, 288, 288) for t in captured_tiles)
    # Verificar edge-bleeding: los píxeles que eran (0,0,0,0) recibieron color rojo del borde (no negro)
    first_tile_r = captured_tiles[0][0, 0, 0, 0]
    assert first_tile_r > 0.8

    restorer.unload_sessions()
    assert restorer._session is None
    assert restorer._cpu_session is None


def test_postprocess_fixes():
    # 1. _postprocess_inspyrenet_mask: sin sigmoid para leve overshoot [-0.01, 1.01]
    mild = np.full((1, 1, 16, 16), 0.0, dtype=np.float32)
    mild[0, 0, 0, 0] = -0.01
    mild[0, 0, 1, 1] = 1.01
    out_mild = _postprocess_inspyrenet_mask(mild, (16, 16))
    assert float(out_mild.min()) == pytest.approx(0.0, abs=1e-4)
    assert float(out_mild[8, 8]) == pytest.approx(0.0, abs=1e-4)  # si hubiera aplicado sigmoid sería ~0.5

    # Logits reales sí aplican sigmoid
    logits = np.full((1, 1, 16, 16), -6.0, dtype=np.float32)
    logits[0, 0, 4:12, 4:12] = 6.0
    out_logits = _postprocess_inspyrenet_mask(logits, (16, 16))
    assert float(out_logits[8, 8]) > 0.99
    assert float(out_logits[0, 0]) < 0.01

    # 2. postprocess_inspyrenet: floodFill con borde de 1 px de ceros rellena hueco aunque (0,0) sea sujeto
    img = Image.new("RGB", (40, 40), (180, 120, 60))
    raw = np.ones((40, 40), dtype=np.float32)
    raw[15:25, 15:25] = 0.0  # hueco interno rodeado por sujeto que toca (0,0)
    _, alpha = postprocess_inspyrenet(img, raw, antialiasing_radius=0, min_island_area=0)
    assert int(alpha[20, 20]) == 255

    # 3. fuse_clothing_protection_core_only usa clothing_alpha dentro de safe_main_core
    main_a = np.full((80, 80), 255, dtype=np.uint8)
    main_a[30:50, 30:50] = 0  # perforación interna en prenda
    cloth_a = np.zeros((80, 80), dtype=np.uint8)
    cloth_a[28:52, 28:52] = 220  # más estrecho que 2*erosion, pero dentro de safe_main_core
    fused = fuse_clothing_protection_core_only(main_a, cloth_a, inner_erosion_px=12, cloth_confidence_thresh=75)
    assert int(fused[40, 40]) > 150

    # 4. refine_alpha_edges con feather y despill
    rgba_img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    arr = np.array(rgba_img)
    arr[8:24, 8:24] = [200, 100, 50, 255]
    refined = refine_alpha_edges(Image.fromarray(arr, mode="RGBA"), choke_px=1, feather_px=2, despill_strength=0.5)
    assert refined.mode == "RGBA"
    assert refined.size == (32, 32)


def test_quantizer_excludes_realesrgan_and_yolo():
    from app.core.quantizer import quantize_model

    with pytest.raises(ValueError, match="excluido"):
        quantize_model("RealESRGAN_x4plus_anime_6B.onnx")

    with pytest.raises(ValueError, match="excluido"):
        quantize_model("yolo11n-seg.onnx")


def test_model_fetcher_security_and_prefer_int8(tmp_path):
    import app.core.model_fetcher as mf

    fp32 = tmp_path / "inspyrenet.onnx"
    int8 = tmp_path / "inspyrenet_int8.onnx"
    fp32.write_bytes(b"F" * 1_100_000)
    int8.write_bytes(b"I" * 1_100_000)

    with patch.object(mf, "MODELS_DIR", tmp_path):
        # Por defecto prefer_int8=False -> prioriza FP32
        assert mf.resolve_local_model("inspyrenet.onnx") == fp32
        assert mf.resolve_local_model("inspyrenet.onnx", prefer_int8=True) == int8
        # Rechaza path traversal
        with pytest.raises(ValueError):
            mf.resolve_local_model("../secret.onnx")

    # Bloqueo de redirección a HTTP no seguro
    handler = mf._HttpsOnlyRedirectHandler()
    req = urllib.request.Request("https://example.com/model.onnx")
    with pytest.raises(urllib.error.URLError, match="HTTPS"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://evil.example/model.onnx")

    # allow_insecure_fallback=False por defecto -> falla en error SSL sin reintentar
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ssl_err = urllib.error.URLError(ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED"))
    with patch.object(mf, "MODELS_DIR", empty_dir), \
         patch("urllib.request.urlopen", side_effect=ssl_err) as mock_open:
        with pytest.raises(urllib.error.URLError):
            mf.ensure_model_file("inspyrenet.onnx")
        assert mock_open.call_count == 1

    # Límite máximo de bytes en _download_with_context
    class OversizedResp:
        status = 200
        headers = {"Content-Length": "2048"}

        def read(self, _n):
            return b"X" * 1024

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=OversizedResp()):
        with pytest.raises(ValueError, match="límite máximo"):
            mf._download_with_context(
                urllib.request.Request("https://example.com/big.onnx"),
                tmp_path / "big.tmp",
                ssl.create_default_context(),
                max_bytes=1024,
            )


def test_download_releases_security(tmp_path):
    import scripts.download_releases as dr

    assert dr._is_allowed_github_url("https://github.com/Eliather/TimoitaseBG/releases/download/v1/app.msix")
    assert dr._is_allowed_github_url("https://objects.githubusercontent.com/some/path")
    assert not dr._is_allowed_github_url("http://github.com/Eliather/TimoitaseBG/releases/download/v1/app.msix")
    assert not dr._is_allowed_github_url("https://evil.example.com/app.msix")

    # Caso 1: falta SHA256SUMS.txt -> falla cerrado (retorna 1) y escapa tag en URL
    captured_urls = []
    api_payload = json.dumps(
        {
            "assets": [
                {
                    "name": "app.msix",
                    "browser_download_url": "https://github.com/Eliather/TimoitaseBG/releases/download/v1/app.msix",
                    "size": 5,
                }
            ]
        }
    ).encode("utf-8")

    class FakeResp:
        def __init__(self, data: bytes):
            self._buf = io.BytesIO(data)

        def read(self, n=-1):
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=30):
        captured_urls.append(req.full_url)
        return FakeResp(api_payload)

    fake_script = tmp_path / "scripts" / "download_releases.py"
    fake_script.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(dr, "__file__", str(fake_script)), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        rc = dr.download_latest_release("v1.0/../../bad")

    assert rc == 1
    assert "v1.0%2F..%2F..%2Fbad" in captured_urls[0]

    # Caso 2: hash inválido en SHA256SUMS.txt -> borra .tmp y retorna 1
    bad_sums = f"{'0' * 64}  app.msix\n".encode("utf-8")
    api_with_sums = json.dumps(
        {
            "assets": [
                {
                    "name": "SHA256SUMS.txt",
                    "browser_download_url": "https://github.com/Eliather/TimoitaseBG/releases/download/v1/SHA256SUMS.txt",
                    "size": len(bad_sums),
                },
                {
                    "name": "app.msix",
                    "browser_download_url": "https://github.com/Eliather/TimoitaseBG/releases/download/v1/app.msix",
                    "size": 5,
                },
            ]
        }
    ).encode("utf-8")

    def fake_urlopen_bad_hash(req, timeout=30):
        if "api.github.com" in req.full_url:
            return FakeResp(api_with_sums)
        if req.full_url.endswith("SHA256SUMS.txt"):
            return FakeResp(bad_sums)
        return FakeResp(b"hello")

    with patch.object(dr, "__file__", str(fake_script)), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen_bad_hash):
        rc_bad = dr.download_latest_release("v1.0")

    assert rc_bad == 1
    assert not (tmp_path / "releases" / "app.msix").exists()
    assert not (tmp_path / "releases" / "app.msix.tmp").exists()
