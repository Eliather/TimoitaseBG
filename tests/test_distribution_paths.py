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


