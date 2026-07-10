"""user_id -> display name map for the chat memory.

The dossier axis needs to bridge three name spaces: Telegram user_id (exact, what the bot knows),
@username (what people type in /profile), and the display name the export stamps on every message
(what actually appears inside window texts). Without the bridge the LLM gets windows full of
"Alexey: ..." while being told the person is "brokenthreephaseswitchboard" — and honestly reports
it can't find any quotes (seen live). Built during JSON backfill; persisted next to the bot state.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger("t800.chatmem")

_FILE = Path(__file__).resolve().parent.parent / "chatmem_names.json"
_cache: dict[str, dict[str, str]] | None = None  # {chat_id_str: {user_id_str: display_name}}


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def save_names(chat_id: int, mapping: dict[int, str]) -> None:
    data = _load()
    bucket = data.setdefault(str(chat_id), {})
    for uid, name in mapping.items():
        if uid and name:
            bucket[str(uid)] = name
    try:
        _FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        log.warning("failed to persist names map", exc_info=True)


def get_name(chat_id: int, user_id: int) -> str | None:
    return _load().get(str(chat_id), {}).get(str(user_id))
