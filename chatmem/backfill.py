"""Backfill: Telegram Desktop HTML export -> dialogue windows -> chat_memory collection.

Text-only pass (cheap, minutes). Media placeholders stay in window texts; the media grind
(vision/whisper) adds its own points later without touching windows. Idempotent: window point ids
derive from message-id ranges, re-running upserts the same points.

IMPORTANT: embedded Qdrant is single-process. While the bot is running it owns qdrant_data, so
backfill must run INSIDE the bot (the /memload command). The CLI form below only works with the
bot stopped:

    python -m chatmem.backfill "<export dir>" <chat_id>
"""

import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable

import i18n
from reader import embedder

from . import names, store
from .telegram_export import parse_export
from .windows import build_windows

log = logging.getLogger("t800.chatmem")

BATCH = 32

# Progress / summary strings, selected by the chat's language.
STR = {
    "en": {
        "parsed": "parsed {n} messages ({span}), windows: {win} (p50 {p50} chars). Embedding...",
        "progress": "memory: {done}/{total} windows (~{secs}s left)",
        "summary": (
            "Chat archive ingested in {secs}s: {win} dialogue windows, "
            "memory points total: {total}. Media ({media} items) will be digested separately."
        ),
    },
    "ru": {
        "parsed": "распарсено {n} сообщений ({span}), окон: {win} (p50 {p50} знаков). Эмбеддинг...",
        "progress": "память: {done}/{total} окон (~{secs} сек осталось)",
        "summary": (
            "Архив чата усвоен за {secs} сек: {win} окон диалога, "
            "всего точек памяти: {total}. Медиа ({media} шт.) будут дожёваны отдельно."
        ),
    },
}


async def run(
    export_dir: str,
    chat_id: int,
    progress: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Returns a one-line human summary; optionally reports progress along the way."""
    lang = i18n.get_lang(chat_id)

    async def tick(text: str) -> None:
        log.info("backfill: %s", text)
        if progress:
            try:
                await progress(text)
            except Exception:
                log.warning("backfill progress callback failed", exc_info=True)

    messages = await asyncio.to_thread(parse_export, export_dir)
    log.info("messages: %s", messages)
    # id -> display name bridge (JSON exports carry from_id) — /profile depends on it.
    names.save_names(chat_id, {m.author_id: m.author for m in messages if m.author_id})
    windows = build_windows(messages, chat_id)
    sizes = sorted(len(w.text) for w in windows)
    span = f"{messages[0].ts:%d.%m} — {messages[-1].ts:%d.%m}"
    await tick(i18n.L(lang, STR, "parsed", n=len(messages), span=span,
                      win=len(windows), p50=sizes[len(sizes) // 2]))

    t0 = time.monotonic()
    last_tick = 0.0
    for i in range(0, len(windows), BATCH):
        batch = windows[i : i + BATCH]
        embs = await embedder.embed_texts([w.text for w in batch])
        await store.upsert_windows(batch, embs)
        done = min(i + BATCH, len(windows))
        rate = done / max(0.1, time.monotonic() - t0)
        if time.monotonic() - last_tick > 15:
            last_tick = time.monotonic()
            await tick(i18n.L(lang, STR, "progress", done=done, total=len(windows),
                              secs=int((len(windows) - done) / max(rate, 0.01))))

    total = await store.count_points(chat_id)
    media = sum(1 for m in messages if m.kind in ("photo", "voice", "video"))
    summary = i18n.L(lang, STR, "summary", secs=f"{time.monotonic() - t0:.0f}",
                     win=len(windows), total=total, media=media)
    await tick(summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(sys.argv[1], int(sys.argv[2])))
