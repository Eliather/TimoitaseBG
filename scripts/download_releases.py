"""
TimoitaseBG - Descargador automático de binarios de GitHub Releases hacia la carpeta local `releases/`.
Descarga los instaladores compilados por GitHub Actions:
- TimoitaseBG-Windows-x64.msix
- TimoitaseBG-macOS-arm64.dmg (macOS 15 Apple Silicon)
- TimoitaseBG-macOS-x64.dmg (macOS 15 Intel)
- TimoitaseBG-Linux-x64.AppImage
"""
import json
import sys
import urllib.request
from pathlib import Path

REPO = "Eliather/TimoitaseBG"


def download_latest_release(tag: str = "v1.2.1") -> int:
    root = Path(__file__).resolve().parent.parent
    releases_dir = root / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    print(f"Consultando release {tag} en https://github.com/{REPO}/releases/tag/{tag} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TimoitaseBG-ReleaseDownloader"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Aún no está publicado el release {tag} o hubo un error de red: {e}")
        return 1

    assets = data.get("assets", [])
    if not assets:
        print(f"El release {tag} existe pero aún no tiene binarios adjuntos.")
        return 1

    for asset in assets:
        name = asset["name"]
        download_url = asset["browser_download_url"]
        size_mb = asset.get("size", 0) / (1024 * 1024)
        dest = releases_dir / name
        print(f"Descargando {name} ({size_mb:.1f} MB) -> {dest} ...")
        req = urllib.request.Request(download_url, headers={"User-Agent": "TimoitaseBG-ReleaseDownloader"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  [OK] Guardado en {dest}")

    print(f"\nTodos los paquetes fueron descargados en: {releases_dir}")
    return 0


if __name__ == "__main__":
    target_tag = sys.argv[1] if len(sys.argv) > 1 else "v1.2.1"
    sys.exit(download_latest_release(target_tag))
