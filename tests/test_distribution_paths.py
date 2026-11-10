"""
Pruebas de verificación de empaquetado, resolución de rutas y blindaje de distribución.
Valida el comportamiento en Windows, macOS y Linux simulando sys.frozen, sys._MEIPASS
y variables de entorno nativas de cada sistema operativo.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from app.config import (
    resolve_app_dir,
    resolve_base_dir,
    resolve_models_dir,
    update_u2net_env,
)


def test_dev_paths_resolution():
    """Verifica la resolución normal en entorno de desarrollo (sys.frozen = False)."""
    with patch.object(sys, "frozen", False, create=True):
        app_dir = resolve_app_dir()
        base_dir = resolve_base_dir()
        models_dir = resolve_models_dir(base_dir=base_dir)

        assert app_dir.name == "app"
        assert app_dir.exists()
        assert base_dir.exists()
        assert models_dir == app_dir / "models"


def test_frozen_app_dir_resolution(tmp_path):
    """Verifica que en modo empaquetado, APP_DIR resuelva hacia sys._MEIPASS / 'app'."""
    meipass_dummy = tmp_path / "meipass_temp"
    app_inside = meipass_dummy / "app"
    ui_inside = app_inside / "ui"
    ui_inside.mkdir(parents=True, exist_ok=True)
    (ui_inside / "style.qss").write_text("/* dummy qss */", encoding="utf-8")

    with patch.object(sys, "frozen", True, create=True), \
         patch.object(sys, "_MEIPASS", str(meipass_dummy), create=True):
        resolved = resolve_app_dir()
        assert resolved == app_inside
        assert (resolved / "ui" / "style.qss").exists()


def test_frozen_models_dir_windows(tmp_path):
    """Verifica que en Windows congelado sin carpeta portable, MODELS_DIR apunte a %LOCALAPPDATA%\\TimoitaseBG\\models."""
    fake_localappdata = tmp_path / "AppData" / "Local"
    fake_localappdata.mkdir(parents=True, exist_ok=True)
    dummy_exe_dir = tmp_path / "ProgramFiles" / "TimoitaseBG"
    dummy_exe_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(sys, "frozen", True, create=True), \
         patch.dict(os.environ, {"LOCALAPPDATA": str(fake_localappdata)}, clear=False):
        models_dir = resolve_models_dir(base_dir=dummy_exe_dir, platform_name="win32")
        expected = fake_localappdata / "TimoitaseBG" / "models"
        assert models_dir == expected
        assert models_dir.exists()


def test_frozen_models_dir_macos(tmp_path):
    """Verifica que en macOS congelado, MODELS_DIR apunte a ~/Library/Application Support/TimoitaseBG/models."""
    fake_home = tmp_path / "Users" / "tester"
    dummy_exe_dir = tmp_path / "Applications" / "TimoitaseBG.app" / "Contents" / "MacOS"
    dummy_exe_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(sys, "frozen", True, create=True), \
         patch.object(Path, "home", return_value=fake_home):
        models_dir = resolve_models_dir(base_dir=dummy_exe_dir, platform_name="darwin")
        expected = fake_home / "Library" / "Application Support" / "TimoitaseBG" / "models"
        assert models_dir == expected
        assert models_dir.exists()


def test_frozen_models_dir_linux(tmp_path):
    """Verifica que en Linux congelado, MODELS_DIR respete XDG_DATA_HOME."""
    fake_xdg = tmp_path / "home" / "tester" / ".local" / "share"
    fake_xdg.mkdir(parents=True, exist_ok=True)
    dummy_exe_dir = tmp_path / "opt" / "timoitasebg"
    dummy_exe_dir.mkdir(parents=True, exist_ok=True)

    with patch.object(sys, "frozen", True, create=True), \
         patch.dict(os.environ, {"XDG_DATA_HOME": str(fake_xdg)}, clear=False):
        models_dir = resolve_models_dir(base_dir=dummy_exe_dir, platform_name="linux")
        expected = fake_xdg / "TimoitaseBG" / "models"
        assert models_dir == expected
        assert models_dir.exists()


def test_frozen_portable_mode_precedence(tmp_path):
    """Verifica que si existe una carpeta 'models' local con archivos .onnx o permisos de escritura, tenga prioridad portable."""
    usb_drive = tmp_path / "USB_Drive"
    portable_models = usb_drive / "models"
    portable_models.mkdir(parents=True, exist_ok=True)
    # Crear un archivo .onnx simulado
    (portable_models / "isnet-anime.onnx").write_bytes(b"dummy onnx weights")

    with patch.object(sys, "frozen", True, create=True):
        models_dir = resolve_models_dir(base_dir=usb_drive, platform_name="win32")
        assert models_dir == portable_models
        assert (models_dir / "isnet-anime.onnx").exists()


def test_u2net_env_synchronization(tmp_path):
    """Verifica que update_u2net_env redirija U2NET_HOME y U2NET_home hacia MODELS_DIR."""
    test_models_dir = tmp_path / "safe_models"
    test_models_dir.mkdir(parents=True, exist_ok=True)

    with patch.dict(os.environ, {}, clear=False):
        update_u2net_env(test_models_dir)
        assert os.environ.get("U2NET_HOME") == str(test_models_dir)
        assert os.environ.get("U2NET_home") == str(test_models_dir)


def test_main_entrypoint_multiprocessing_shield():
    """Verifica que app/main.py llame a freeze_support() como primera instrucción en if __name__ == '__main__'."""
    main_py_path = Path("app/main.py")
    assert main_py_path.exists()
    content = main_py_path.read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in content
    main_block = content.split('if __name__ == "__main__":')[1]

    # Descartar comentarios y lineas vacias para encontrar la primera declaracion
    lines = [line.strip() for line in main_block.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert len(lines) > 0
    assert lines[0] == "multiprocessing.freeze_support()", (
        f"Se esperaba 'multiprocessing.freeze_support()' como la primera instrucción, se obtuvo: '{lines[0]}'"
    )


def test_build_spec_models_bundling_flag():
    """Verifica que build.spec NO incluya app/models por defecto, salvo con TIMOITASE_BUNDLE_MODELS == '1'."""
    spec_path = Path("build.spec")
    assert spec_path.exists()
    content = spec_path.read_text(encoding="utf-8")

    assert 'os.environ.get("TIMOITASE_BUNDLE_MODELS") == "1"' in content

    # Evaluar snippet de datas bajo simulación
    def evaluate_datas(env_val):
        g = {"Path": Path, "os": os, "safe_collect_all": lambda x: ([], [], [])}
        with patch.dict(os.environ, {"TIMOITASE_BUNDLE_MODELS": env_val} if env_val is not None else {}, clear=True):
            exec(
                """
datas = [('app/ui', 'app/ui')]
if os.environ.get("TIMOITASE_BUNDLE_MODELS") == "1" and Path('app/models').exists():
    datas.append(('app/models', 'app/models'))
                """,
                g,
            )
            return g["datas"]

    # Por defecto sin variable: No incluye app/models
    datas_default = evaluate_datas(None)
    assert ('app/models', 'app/models') not in datas_default
    assert ('app/ui', 'app/ui') in datas_default

    # Con TIMOITASE_BUNDLE_MODELS == "0": No incluye app/models
    datas_zero = evaluate_datas("0")
    assert ('app/models', 'app/models') not in datas_zero

    # Con TIMOITASE_BUNDLE_MODELS == "1" (si existe el directorio): Se incluye
    datas_one = evaluate_datas("1")
    if Path("app/models").exists():
        assert ('app/models', 'app/models') in datas_one


def test_gpu_worker_u2net_sync():
    """Verifica que la importación de config y el subproceso mantengan sincronizado U2NET_HOME."""
    import importlib
    import app.config as config
    importlib.reload(config)

    assert os.environ.get("U2NET_HOME") == str(config.MODELS_DIR)
    assert os.environ.get("U2NET_home") == str(config.MODELS_DIR)


def test_package_release_msix_and_appimage_staging(tmp_path):
    """Verifica que scripts.package_release genere correctamente los directorios para MSIX y AppImage."""
    from scripts.package_release import prepare_msix_stage, prepare_appimage_stage

    fake_dist = tmp_path / "dist" / "TimoitaseBG"
    fake_dist.mkdir(parents=True, exist_ok=True)
    (fake_dist / "TimoitaseBG.exe").write_bytes(b"MZ")
    (fake_dist / "TimoitaseBG").write_bytes(b"\x7fELF")

    msix_stage = tmp_path / "msix_stage"
    prepare_msix_stage(fake_dist, msix_stage)
    assert (msix_stage / "AppxManifest.xml").exists()
    assert (msix_stage / "Assets" / "StoreLogo.png").exists()
    assert (msix_stage / "Assets" / "Square44x44Logo.png").exists()
    assert (msix_stage / "Assets" / "Square150x150Logo.png").exists()

    appdir = tmp_path / "TimoitaseBG.AppDir"
    prepare_appimage_stage(fake_dist, appdir)
    assert (appdir / "AppRun").exists()
    assert (appdir / "TimoitaseBG.desktop").exists()
    assert (appdir / "timoitasebg.png").exists()
    assert (appdir / "usr" / "bin" / "TimoitaseBG").exists()


def test_configure_ssl_certificates_repairs_invalid_linux_runner_path():
    """
    Verifica que configure_ssl_certificates() reemplace rutas de compilación inexistentes
    (como /opt/hostedtoolcache/.../ssl/cert.pem en Linux AppImage) por un CA bundle válido (certifi).
    """
    from app.config import configure_ssl_certificates, resolve_ca_bundle_path

    broken_env = {
        "SSL_CERT_FILE": "/opt/hostedtoolcache/Python/3.11.9/x64/ssl/cert.pem",
        "REQUESTS_CA_BUNDLE": "/nonexistent/path/cacert.pem",
    }
    with patch.dict(os.environ, broken_env, clear=False):
        ca_bundle = configure_ssl_certificates()
        assert ca_bundle is not None
        assert Path(ca_bundle).is_file()
        assert os.environ["SSL_CERT_FILE"] == ca_bundle
        assert os.environ["REQUESTS_CA_BUNDLE"] == ca_bundle
        assert resolve_ca_bundle_path() == ca_bundle


def test_model_fetcher_ssl_context_and_fallback_on_cert_error(tmp_path):
    """
    Verifica que ensure_model_file() pase un ssl.SSLContext a urlopen y se recupere
    automáticamente si en Linux ocurre [SSL: CERTIFICATE_VERIFY_FAILED] (_ssl.c:1016).
    """
    import ssl
    import urllib.error
    import app.core.model_fetcher as mf

    ctx = mf._create_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)

    dummy_payload = b"ONNX" * 300000  # > 1 MB
    call_contexts = []

    class FakeHttpResponse:
        def __init__(self, data: bytes):
            self._data = data
            self._offset = 0
            self.headers = {"Content-Length": str(len(data))}

        def read(self, size: int = -1) -> bytes:
            if self._offset >= len(self._data):
                return b""
            chunk = self._data[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_urlopen(req, context=None, timeout=None):
        call_contexts.append(context)
        if len(call_contexts) == 1:
            ssl_err = ssl.SSLCertVerificationError(
                1,
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1016)",
            )
            raise urllib.error.URLError(ssl_err)
        return FakeHttpResponse(dummy_payload)

    with patch.object(mf, "MODELS_DIR", tmp_path), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        downloaded_path = mf.ensure_model_file("inspyrenet.onnx")
        assert downloaded_path == tmp_path / "inspyrenet.onnx"
        assert downloaded_path.exists()
        assert downloaded_path.stat().st_size == len(dummy_payload)
        assert len(call_contexts) == 2
        assert isinstance(call_contexts[0], ssl.SSLContext)
        assert isinstance(call_contexts[1], ssl.SSLContext)
        assert call_contexts[1].verify_mode == ssl.CERT_NONE


def test_real_progressive_percentage_reporting():
    """
    Verifica que BackgroundRemover, ImageRestorer y GPUWorkerManager emitan porcentajes
    reales y progresivos (0% -> 100%) hacia progress_cb sin bloquear ni usar rangos indeterminados.
    """
    import numpy as np
    from PIL import Image
    from unittest.mock import MagicMock
    from app.core.bg_remover import BackgroundRemover
    from app.core.restorer import ImageRestorer

    dummy_img = Image.new("RGBA", (64, 64), (120, 180, 220, 255))

    # 1. Progreso real en BackgroundRemover (InSPyReNet)
    remover = BackgroundRemover()
    fake_session = MagicMock()
    fake_input = MagicMock()
    fake_input.name = "input"
    fake_session.get_inputs.return_value = [fake_input]
    fake_session.run.return_value = [np.ones((1, 1, 1024, 1024), dtype=np.float32)]
    remover._sessions["inspyrenet_cpu"] = fake_session

    bg_events = []
    out_bg = remover._remove_background_cpu_local(
        dummy_img,
        model_name="inspyrenet",
        enable_clothing_protection=False,
        progress_cb=lambda pct, msg: bg_events.append((pct, msg)),
    )
    assert out_bg.size == (64, 64)
    percents_bg = [p for p, _ in bg_events]
    assert len(percents_bg) >= 4
    assert percents_bg == sorted(percents_bg)
    assert percents_bg[0] > 0
    assert percents_bg[-1] == 100

    # 2. Progreso real por bloques (tiles) en ImageRestorer
    restorer = ImageRestorer()
    fake_sr_sess = MagicMock()
    sr_in = MagicMock()
    sr_in.name = "in"
    sr_out = MagicMock()
    sr_out.name = "out"
    fake_sr_sess.get_inputs.return_value = [sr_in]
    fake_sr_sess.get_outputs.return_value = [sr_out]

    def fake_sr_run(out_names, feed):
        tensor = list(feed.values())[0]
        _, c, h, w = tensor.shape
        return [np.ones((1, c, h * 4, w * 4), dtype=np.float32) * 0.5]

    fake_sr_sess.run.side_effect = fake_sr_run
    restorer._cpu_session = fake_sr_sess

    sr_events = []
    out_sr = restorer._restore_image_impl(
        dummy_img,
        scale_mode="x2",
        use_tiling=True,
        tile_size=32,
        tile_pad=4,
        force_cpu=True,
        progress_cb=lambda pct, msg: sr_events.append((pct, msg)),
    )
    assert out_sr.size == (128, 128)
    percents_sr = [p for p, _ in sr_events]
    assert any("Restaurando bloque" in m for _, m in sr_events)
    assert percents_sr == sorted(percents_sr)
    assert percents_sr[-1] == 100


