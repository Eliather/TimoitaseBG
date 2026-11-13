"""
TimoitaseBG - Carga ligera de internacionalización desde archivos JSON en app/locales/.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import APP_DIR, APP_AUTHOR, APP_NAME

logger = logging.getLogger("TimoitaseBG.i18n")

_SUPPORTED_LANG_ORDER = ("es", "en", "ru", "zh")


def _resolve_locales_dir() -> Path:
    candidate = APP_DIR / "locales"
    if candidate.is_dir():
        return candidate
    return Path(__file__).resolve().parent / "locales"


def _load_translations() -> Dict[str, Dict[str, str]]:
    locales_dir = _resolve_locales_dir()
    loaded: Dict[str, Dict[str, str]] = {}

    if locales_dir.is_dir():
        json_files = sorted(
            locales_dir.glob("*.json"),
            key=lambda p: (
                _SUPPORTED_LANG_ORDER.index(p.stem)
                if p.stem in _SUPPORTED_LANG_ORDER
                else len(_SUPPORTED_LANG_ORDER),
                p.stem,
            ),
        )
        for path in json_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    loaded[path.stem] = {str(k): str(v) for k, v in data.items()}
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"No se pudo cargar idioma '{path.name}': {e}")

    return loaded


TRANSLATIONS: Dict[str, Dict[str, str]] = _load_translations()


class Translator:
    def __init__(self):
        self.current_lang = "es"
        self._load_lang_preference()

    def _get_qsettings(self):
        try:
            from PySide6.QtCore import QSettings
            return QSettings(APP_AUTHOR, APP_NAME)
        except Exception:
            return None

    def _load_lang_preference(self):
        settings = self._get_qsettings()
        if settings is not None:
            try:
                saved = settings.value("ui/language", None)
                if isinstance(saved, str) and saved in TRANSLATIONS:
                    self.current_lang = saved
                    return
            except Exception as e:
                logger.debug(f"No se pudo leer idioma desde QSettings: {e}")

        # Migración silenciosa desde archivo legado ~/.timoitasebg_lang.json
        legacy_file = Path(os.path.expanduser("~")) / ".timoitasebg_lang.json"
        if legacy_file.is_file():
            try:
                data = json.loads(legacy_file.read_text(encoding="utf-8"))
                lang = data.get("lang", "es")
                if lang in TRANSLATIONS:
                    self.current_lang = lang
                    if settings is not None:
                        settings.setValue("ui/language", lang)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug(f"Error leyendo preferencia legada de idioma: {e}")

    def _save_lang_preference(self):
        settings = self._get_qsettings()
        if settings is not None:
            try:
                settings.setValue("ui/language", self.current_lang)
                settings.sync()
                return
            except Exception as e:
                logger.warning(f"No se pudo guardar idioma en QSettings: {e}")

        legacy_file = Path(os.path.expanduser("~")) / ".timoitasebg_lang.json"
        try:
            legacy_file.write_text(
                json.dumps({"lang": self.current_lang}),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"No se pudo persistir preferencia de idioma: {e}")

    def set_language(self, lang_code: str):
        if lang_code in TRANSLATIONS:
            self.current_lang = lang_code
            self._save_lang_preference()

    def get(self, key: str, **kwargs) -> str:
        lang_dict = TRANSLATIONS.get(self.current_lang, TRANSLATIONS.get("en", {}))
        fallback_es = TRANSLATIONS.get("es", {})
        text = lang_dict.get(key, fallback_es.get(key, key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError:
                return text
        return text


_translator = Translator()


def tr(key: str, **kwargs) -> str:
    return _translator.get(key, **kwargs)


def get_current_language() -> str:
    return _translator.current_lang


def set_language(lang_code: str):
    _translator.set_language(lang_code)


def get_available_languages() -> List[Tuple[str, str]]:
    return [(code, data.get("lang_name", code)) for code, data in TRANSLATIONS.items()]
