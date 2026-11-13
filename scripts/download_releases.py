"""
TimoitaseBG - Descargador automático de binarios de GitHub Releases hacia la carpeta local `releases/`.
Descarga los instaladores compilados por GitHub Actions y valida SHA256SUMS.txt si está presente:
- TimoitaseBG-Windows-x64.msix
- TimoitaseBG-Windows-arm64.msix
- TimoitaseBG-macOS-arm64.dmg (macOS 15 Apple Silicon)
- TimoitaseBG-macOS-x64.dmg (macOS 15 Intel)
- TimoitaseBG-Linux-x64.AppImage
- TimoitaseBG-Linux-arm64.AppImage
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from app.config import APP_VERSION

REPO = "Eliather/TimoitaseBG"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _parse_checksums(checksums_path: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    if not checksums_path.exists():
        return sums
    for line in checksums_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def download_latest_release(tag: str) -> int:
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
        expected_size = asset.get("size", 0)
        size_mb = expected_size / (1024 * 1024)
        dest = releases_dir / name
        if dest.exists() and dest.stat().st_size == expected_size and expected_size > 0:
            print(f"[SKIP] {name} ({size_mb:.1f} MB) ya existe en {dest}")
            continue
        print(f"Descargando {name} ({size_mb:.1f} MB) -> {dest} ...")
        req = urllib.request.Request(download_url, headers={"User-Agent": "TimoitaseBG-ReleaseDownloader"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  [OK] Guardado en {dest}")

    checksums = _parse_checksums(releases_dir / "SHA256SUMS.txt")
    for fname, expected_hash in checksums.items():
        fpath = releases_dir / fname
        if fpath.exists():
            actual = _sha256_file(fpath)
            if actual != expected_hash:
                print(f"  [ERROR] Checksum inválido para {fname}")
                return 1
            print(f"  [SHA-256 OK] {fname}")

    print(f"\nTodos los paquetes fueron descargados en: {releases_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga instaladores de GitHub Releases.")
    parser.add_argument(
        "tag",
        nargs="?",
        default=f"v{APP_VERSION}",
        help=f"Etiqueta de release a descargar (por defecto: v{APP_VERSION})",
    )
    args = parser.parse_args()
    sys.exit(download_latest_release(args.tag))

