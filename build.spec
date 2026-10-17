# -*- mode: python ; coding: utf-8 -*-
"""
TimoitaseBG - Especificación de empaquetado PyInstaller multiplataforma.
Genera ejecutables autocontenidos en modo COLLECT (--onedir) para Windows, macOS y Linux.
"""
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None


def safe_collect_all(package_name):
    """Recolecta de forma segura datas, binaries y hiddenimports de un paquete."""
    try:
        return collect_all(package_name)
    except Exception:
        return [], [], []


# 1. Datos estáticos de la aplicación (QSS, iconos vectoriales y activos)
datas = [
    ('app/ui', 'app/ui'),
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

datas += rembg_datas + ort_datas + pooch_datas + pymatting_datas
binaries = rembg_binaries + ort_binaries + pooch_binaries + pymatting_binaries

# 2. Hidden imports esenciales para inferencia y UI
hiddenimports = [
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
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtSvg',
] + rembg_hidden + ort_hidden + pooch_hidden + pymatting_hidden

# Proveedores de aceleración específicos por plataforma
if sys.platform == "win32":
    hiddenimports.append('onnxruntime.providers.dml')
elif sys.platform == "darwin":
    hiddenimports.append('onnxruntime.providers.coreml')

# 3. Exclusiones de paquetes de desarrollo para reducir cientos de MBs
excludes = [
    'pytest',
    'tkinter',
    'matplotlib',
    'notebook',
    'IPython',
    'ipykernel',
    'jupyter',
    'unittest',
    'torch',
    'torchvision',
    'tensorboard',
    'scipy.spatial.cKDTree',
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
        icon=None,
        bundle_identifier='com.eliather.timoitasebg',
        info_plist={
            'CFBundleName': 'TimoitaseBG',
            'CFBundleDisplayName': 'TimoitaseBG',
            'CFBundleIdentifier': 'com.eliather.timoitasebg',
            'CFBundleVersion': '1.2.1',
            'CFBundleShortVersionString': '1.2.1',
            'NSHumanReadableCopyright': 'Copyright © 2026 Eliather (Autora)',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'NSPrincipalClass': 'NSApplication',
        },
    )
