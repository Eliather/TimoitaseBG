"""
TimoitaseBG - Generador de recursos y manifiestos para empaquetado multiplataforma:
- Windows (.msix): Genera AppxManifest.xml y recursos visuales PNG requeridos por MakeAppx.
- Linux (.AppImage): Prepara la estructura AppDir (AppRun, .desktop, icono PNG).
"""
import os
import shutil
import stat
import sys
from pathlib import Path
from PIL import Image, ImageDraw

from app.config import APP_NAME, APP_VERSION, APP_AUTHOR


def generate_brand_icon(size: int, output_path: Path) -> None:
    """Genera un icono PNG de marca (fondo oscuro slate con círculo coral e insignia IA)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fondo redondeado oscuro (#0F172A)
    pad = max(1, size // 16)
    radius = max(4, size // 5)
    draw.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=radius,
        fill=(15, 23, 42, 255),
        outline=(51, 65, 85, 255),
        width=max(1, size // 32),
    )

    # Círculo central coral (#F43F5E)
    c_margin = max(4, size // 4)
    draw.ellipse(
        [c_margin, c_margin, size - c_margin - 1, size - c_margin - 1],
        fill=(244, 63, 94, 255),
    )

    # Destello blanco interior
    s_margin = max(6, int(size * 0.38))
    draw.ellipse(
        [s_margin, s_margin, size - s_margin - 1, size - s_margin - 1],
        fill=(255, 255, 255, 240),
    )

    img.save(output_path, format="PNG")


def prepare_msix_stage(dist_dir: Path, stage_dir: Path) -> None:
    """Prepara el directorio de staging para compilar un paquete Windows .msix con MakeAppx."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    shutil.copytree(dist_dir, stage_dir)

    assets_dir = stage_dir / "Assets"
    generate_brand_icon(50, assets_dir / "StoreLogo.png")
    generate_brand_icon(44, assets_dir / "Square44x44Logo.png")
    generate_brand_icon(150, assets_dir / "Square150x150Logo.png")

    # MSIX requiere versión de 4 partes (Major.Minor.Build.Revision)
    parts = APP_VERSION.split(".")
    while len(parts) < 4:
        parts.append("0")
    msix_version = ".".join(parts[:4])

    manifest_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap rescap">
  <Identity
    Name="Eliather.TimoitaseBG"
    Publisher="CN={APP_AUTHOR}"
    Version="{msix_version}"
    ProcessorArchitecture="x64" />
  <Properties>
    <DisplayName>{APP_NAME}</DisplayName>
    <PublisherDisplayName>{APP_AUTHOR}</PublisherDisplayName>
    <Logo>Assets\\StoreLogo.png</Logo>
    <Description>Estación de trabajo de escritorio para eliminación de fondos en imágenes, GIFs y videos con IA.</Description>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.22631.0" />
  </Dependencies>
  <Resources>
    <Resource Language="es-ES" />
  </Resources>
  <Applications>
    <Application Id="TimoitaseBG" Executable="TimoitaseBG.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="{APP_NAME}"
        Description="Eliminación de fondos con IA para imágenes, GIFs y videos"
        BackgroundColor="#0F172A"
        Square150x150Logo="Assets\\Square150x150Logo.png"
        Square44x44Logo="Assets\\Square44x44Logo.png" />
    </Application>
  </Applications>
  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"""
    (stage_dir / "AppxManifest.xml").write_text(manifest_xml, encoding="utf-8")
    print(f"[MSIX] Directorio de staging preparado en: {stage_dir} (v{msix_version})")


def prepare_appimage_stage(dist_dir: Path, appdir: Path) -> None:
    """Prepara la estructura AppDir para empaquetar en formato .AppImage en Linux."""
    if appdir.exists():
        shutil.rmtree(appdir)

    usr_bin = appdir / "usr" / "bin"
    usr_bin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist_dir, usr_bin)

    generate_brand_icon(256, appdir / "timoitasebg.png")
    shutil.copyfile(appdir / "timoitasebg.png", appdir / ".DirIcon")

    desktop_content = f"""[Desktop Entry]
Type=Application
Name={APP_NAME}
Comment=Eliminación de fondos con IA para imágenes, GIFs y videos
Exec=TimoitaseBG
Icon=timoitasebg
Categories=Graphics;Photography;AudioVideo;
Terminal=false
"""
    (appdir / "TimoitaseBG.desktop").write_text(desktop_content, encoding="utf-8")

    apprun_content = """#!/bin/sh
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/TimoitaseBG" "$@"
"""
    apprun_path = appdir / "AppRun"
    apprun_path.write_text(apprun_content, encoding="utf-8")

    # Asegurar permisos de ejecución
    for target in (apprun_path, usr_bin / "TimoitaseBG"):
        if target.exists():
            st = os.stat(target)
            os.chmod(target, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"[AppImage] Estructura AppDir preparada en: {appdir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.package_release [msix|appimage]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    root = Path(__file__).resolve().parent.parent
    dist_folder = root / "dist" / "TimoitaseBG"

    if mode == "msix":
        prepare_msix_stage(dist_folder, root / "build" / "msix_stage")
    elif mode == "appimage":
        prepare_appimage_stage(dist_folder, root / "build" / "TimoitaseBG.AppDir")
    else:
        print(f"Modo desconocido: {mode}")
        sys.exit(1)
