"""Per-chat language selection for the whole bot.

The bot ships English-first (DEFAULT_LANG); any chat can switch with /lang. The setting is stored
per chat_id and persisted to lang_state.json so it survives restarts. Every user-facing string and
every model-facing prompt is selected by the chat's language.

Usage pattern everywhere (keep it uniform — this is the convention all modules follow):

    import i18n

    lang = i18n.get_lang(chat_id)          # at the top of a Telegram handler
    await msg.reply_text(i18n.L(lang, STR, "no_active_book"))   # module-local STR table

where STR is a module-level table:

    STR = {
        "en": {"no_active_book": "No active book. Send a book file (FB2/EPUB)."},
        "ru": {"no_active_book": "Активной книги нет. Пришли файл книги (FB2/EPUB)."},
    }

Deep functions that build MODEL prompts (query.prepare, research, recall) take an explicit `lang`
argument rather than looking it up, so they stay pure and testable.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("t800.i18n")

LANGS = ("en", "ru")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "en").strip().lower()
if DEFAULT_LANG not in LANGS:
    DEFAULT_LANG = "en"

_STATE_FILE = Path(__file__).resolve().parent / "lang_state.json"
_cache: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = {str(k): v for k, v in json.loads(_STATE_FILE.read_text(encoding="utf-8")).items()}
        except Exception:
            _cache = {}
    return _cache


def normalize(lang: str | None) -> str:
    """Accept 'en'/'ru', 'english'/'русский', 'eng'/'рус', locale-ish 'en-US' — anything else -> None."""
    if not lang:
        return ""
    s = lang.strip().lower()
    if s.startswith(("en", "англ", "eng")):
        return "en"
    if s.startswith(("ru", "рус")):
        return "ru"
    return ""


def get_lang(chat_id: int | None) -> str:
    if chat_id is None:
        return DEFAULT_LANG
    return _load().get(str(chat_id), DEFAULT_LANG)


def set_lang(chat_id: int, lang: str) -> str:
    """Set and persist; returns the normalized value actually stored (DEFAULT_LANG if unrecognized)."""
    norm = normalize(lang) or DEFAULT_LANG
    data = _load()
    data[str(chat_id)] = norm
    try:
        _STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        log.warning("failed to persist lang_state", exc_info=True)
    return norm


def L(lang: str, table: dict, key: str, **fmt) -> str:
    """Pick table[lang][key], falling back to English, then to the key itself. Formats with **fmt."""
    entry = table.get(lang, {}).get(key)
    if entry is None:
        entry = table.get("en", {}).get(key, key)
    return entry.format(**fmt) if fmt else entry
