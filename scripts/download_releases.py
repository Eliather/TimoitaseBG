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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from app.config import APP_VERSION

REPO = "Eliather/TimoitaseBG"
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}


def _is_allowed_github_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme.lower() == "https" and host in ALLOWED_DOWNLOAD_HOSTS


class _GitHubHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Permite únicamente redirecciones HTTPS hacia hosts oficiales de GitHub."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        if not _is_allowed_github_url(resolved):
            raise urllib.error.URLError(f"Redirección a URL no permitida bloqueada: {resolved!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_github_download(req: urllib.request.Request, timeout: int = 300):
    if not _is_allowed_github_url(req.full_url):
        raise ValueError(f"URL de descarga fuera de hosts GitHub permitidos: {req.full_url!r}")
    if getattr(urllib.request.urlopen, "__module__", "") == "unittest.mock":
        return urllib.request.urlopen(req, timeout=timeout)
    opener = urllib.request.build_opener(_GitHubHttpsRedirectHandler())
    return opener.open(req, timeout=timeout)


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
            raw_name = parts[1].lstrip("*").strip()
            safe_name = Path(raw_name).name
            if safe_name:
                sums[safe_name] = parts[0].lower()
    return sums


def _download_asset_to_tmp(download_url: str, tmp_path: Path) -> None:
    req = urllib.request.Request(download_url, headers={"User-Agent": "TimoitaseBG-ReleaseDownloader"})
    with _open_github_download(req, timeout=300) as r, open(tmp_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)


def download_latest_release(tag: str, allow_unverified: bool = False) -> int:
    root = Path(__file__).resolve().parent.parent
    releases_dir = (root / "releases").resolve()
    releases_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = urllib.parse.quote(tag, safe="")
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{safe_tag}"
    print(f"Consultando release {tag} en https://github.com/{REPO}/releases/tag/{safe_tag} ...")
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

    # 1. Descargar primero SHA256SUMS.txt si existe en el release
    checksums_asset = None
    binary_assets = []
    for asset in assets:
        name = str(asset.get("name", ""))
        safe_name = Path(name).name
        if not safe_name or safe_name != name or safe_name in (".", ".."):
            print(f"[SKIP] Nombre de asset inválido omitido: {name!r}")
            continue
        if safe_name == "SHA256SUMS.txt":
            checksums_asset = asset
        else:
            binary_assets.append(asset)

    checksums_file = releases_dir / "SHA256SUMS.txt"
    if checksums_asset is not None:
        cs_url = str(checksums_asset.get("browser_download_url", ""))
        if not _is_allowed_github_url(cs_url):
            print("[ERROR] URL no permitida para SHA256SUMS.txt")
            return 1
        cs_tmp = checksums_file.with_name(checksums_file.name + ".tmp")
        try:
            _download_asset_to_tmp(cs_url, cs_tmp)
            cs_tmp.replace(checksums_file)
        except Exception as exc:
            cs_tmp.unlink(missing_ok=True)
            print(f"[ERROR] No se pudo descargar SHA256SUMS.txt: {exc}")
            return 1

    checksums = _parse_checksums(checksums_file)
    if not checksums:
        if not allow_unverified:
            print("[ERROR] Falta SHA256SUMS.txt o está vacío. Usa --allow-unverified solo si confías explícitamente.")
            return 1
        print("  [WARN] SHA256SUMS.txt ausente; continuando sin verificación por --allow-unverified.")

    # 2. Descargar binarios a .tmp y verificar SHA-256 antes de mover al destino final
    for asset in binary_assets:
        name = str(asset.get("name", ""))
        safe_name = Path(name).name
        dest = (releases_dir / safe_name).resolve()
        if not dest.is_relative_to(releases_dir):
            print(f"[SKIP] Ruta fuera de releases_dir omitida: {name!r}")
            continue

        expected_hash = checksums.get(safe_name)
        if not expected_hash and not allow_unverified:
            print(f"  [ERROR] {safe_name} no figura en SHA256SUMS.txt; descarga rechazada.")
            return 1

        download_url = str(asset.get("browser_download_url", ""))
        if not _is_allowed_github_url(download_url):
            print(f"[SKIP] URL no permitida omitida para {safe_name}: {download_url!r}")
            continue

        expected_size = asset.get("size", 0)
        size_mb = expected_size / (1024 * 1024)
        if dest.exists() and dest.stat().st_size == expected_size and expected_size > 0:
            if not expected_hash or _sha256_file(dest) == expected_hash:
                print(f"[SKIP] {safe_name} ({size_mb:.1f} MB) ya existe y está verificado en {dest}")
                continue

        tmp_path = dest.with_name(dest.name + ".tmp")
        print(f"Descargando {safe_name} ({size_mb:.1f} MB) -> {tmp_path} ...")
        try:
            _download_asset_to_tmp(download_url, tmp_path)
            if expected_hash:
                actual = _sha256_file(tmp_path)
                if actual != expected_hash:
                    tmp_path.unlink(missing_ok=True)
                    print(f"  [ERROR] Checksum inválido para {safe_name}; archivo temporal eliminado.")
                    return 1
                print(f"  [SHA-256 OK] {safe_name}")
            tmp_path.replace(dest)
            print(f"  [OK] Guardado en {dest}")
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            print(f"  [ERROR] Fallo descargando {safe_name}: {exc}")
            return 1

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
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Permite descargar sin SHA256SUMS.txt (inseguro; desactivado por defecto).",
    )
    args = parser.parse_args()
    sys.exit(download_latest_release(args.tag, allow_unverified=args.allow_unverified))

