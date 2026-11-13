# -*- mode: python ; coding: utf-8 -*-
"""
TimoitaseBG - Especificación de empaquetado PyInstaller multiplataforma.
Genera ejecutables autocontenidos en modo COLLECT (--onedir) para Windows, macOS y Linux.
"""
import re
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None


def _read_app_version() -> str:
    cfg = Path("app/config.py")
    if cfg.exists():
        m = re.search(r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']', cfg.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            return m.group(1)
    return "1.2.3"


APP_VERSION = _read_app_version()


def safe_collect_all(package_name):
    """Recolecta de forma segura datas, binaries y hiddenimports de un paquete."""
    try:
        return collect_all(package_name)
    except Exception:
        return [], [], []


# 1. Datos estáticos de la aplicación (QSS, iconos vectoriales, traducciones y activos)
datas = [
    ('app/ui', 'app/ui'),
    ('app/locales', 'app/locales'),
]

# Incluir modelos pesados de app/models ÚNICAMENTE si TIMOITASE_BUNDLE_MODELS == "1"
# Esto evita inflar el ejecutable localmente con ~1.5 GB de modelos descargados.
# Por defecto el paquete se mantiene ligero y los modelos se gestionan en MODELS_DIR.
import os
if os.environ.get("TIMOITASE_BUNDLE_MODELS") == "1" and Path('app/models').exists():
    datas.append(('app/models', 'app/models'))

# Recolectar metadatos, binarios y submódulos críticos de dependencias de IA y visión
rembg_datas, rembg_binaries, rembg_hidden = safe_collect_all('rembg')
ort_datas, ort_binaries, ort_hidden = safe_collect_all('onnxruntime')
pooch_datas, pooch_binaries, pooch_hidden = safe_collect_all('pooch')
pymatting_datas, pymatting_binaries, pymatting_hidden = safe_collect_all('pymatting')
certifi_datas, certifi_binaries, certifi_hidden = safe_collect_all('certifi')

datas += rembg_datas + ort_datas + pooch_datas + pymatting_datas + certifi_datas
binaries = rembg_binaries + ort_binaries + pooch_binaries + pymatting_binaries + certifi_binaries

# 2. Hidden imports esenciales para inferencia y UI
hiddenimports = [
    'certifi',
    'onnxruntime',
    'onnxruntime.providers.cpu',
    'rembg',
    'scipy.ndimage',
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageFilter',
    'pooch',
    'pymatting',
    'numpy',
    'unittest',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtSvg',
] + rembg_hidden + ort_hidden + pooch_hidden + pymatting_hidden + certifi_hidden

# Proveedores de aceleración específicos por plataforma
if sys.platform == "win32":
    hiddenimports.append('onnxruntime.providers.dml')
elif sys.platform == "darwin":
    hiddenimports.append('onnxruntime.providers.coreml')

# 3. Exclusiones de paquetes de desarrollo para reducir cientos de MBs
# NOTA: 'unittest' NO debe excluirse porque numpy y scipy.ndimage lo requieren en tiempo de ejecución.
excludes = [
    'pytest',
    'tkinter',
    'matplotlib',
    'notebook',
    'IPython',
    'ipykernel',
    'jupyter',
    'torch',
    'torchvision',
    'tensorboard',
]

a = Analysis(
    ['app/main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_win_icon = Path('build/app_icon.ico')
_mac_icon = Path('build/app_icon.icns') if Path('build/app_icon.icns').exists() else Path('build/app_icon.png')

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TimoitaseBG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_win_icon) if sys.platform == 'win32' and _win_icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TimoitaseBG',
)

# 4. Creación del Bundle .app para macOS con soporte Retina e identificador
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name='TimoitaseBG.app',
        icon=str(_mac_icon) if _mac_icon.exists() else None,
        bundle_identifier='com.eliather.timoitasebg',
        info_plist={
            'CFBundleName': 'TimoitaseBG',
            'CFBundleDisplayName': 'TimoitaseBG',
            'CFBundleIdentifier': 'com.eliather.timoitasebg',
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'NSHumanReadableCopyright': 'Copyright © 2026 Eliather (Autora)',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'NSPrincipalClass': 'NSApplication',
        },
    )
