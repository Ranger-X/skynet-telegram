"""Telegram commands for the chat memory. Backfill has to run inside the bot process (embedded
Qdrant is single-process and the bot owns it while running), so /memload is the way to feed an
export in — plus it gives progress edits in the chat for free.

/memload <путь к папке экспорта> — ingest a Telegram Desktop HTML export into chat_memory.
/memgrind <путь> — digest the export's media (voices via whisper, photos/videos via the 12B).
/memstat — how many memory points this chat has.
"""

import json
import logging
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

import config
from reader import llm as reader_llm
from reader import rconfig
from reader.streaming import StreamEditor

from . import backfill, grind, recall, store

log = logging.getLogger("t800.chatmem")

# Last export dir per chat, so /memgrind after /memload needs no re-typing. PERSISTED to disk —
# in-memory state dies with every bot restart (learned the hard way).
_STATE_FILE = Path(__file__).resolve().parent.parent / "chatmem_state.json"


def _load_last_export() -> dict[int, str]:
    try:
        return {int(k): v for k, v in json.loads(_STATE_FILE.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


_last_export: dict[int, str] = _load_last_export()


def _remember_export(chat_id: int, path: str) -> None:
    _last_export[chat_id] = path
    try:
        _STATE_FILE.write_text(json.dumps({str(k): v for k, v in _last_export.items()}), encoding="utf-8")
    except OSError:
        log.warning("failed to persist chatmem state", exc_info=True)


def _has_export_files(d: Path) -> bool:
    return (d / "result.json").exists() or bool(list(d.glob("messages*.html")))


def _resolve_export_dir(path: str) -> Path | None:
    """Forgiving path resolution: accept the export root OR any of its subfolders (users
    naturally paste .../voice_messages when they mean the export)."""
    if not path:
        return None
    d = Path(path)
    if d.is_dir():
        if _has_export_files(d):
            return d
        if _has_export_files(d.parent):
            return d.parent
    return None


def _is_target(update: Update) -> bool:
    chat = update.effective_chat
    return chat is not None and (not config.TARGET_CHAT_IDS or chat.id in config.TARGET_CHAT_IDS)


async def _owner_only(update: Update) -> bool:
    """Heavy admin commands (filesystem paths, hours of compute) are owner-gated — chat members
    already tried to run /memload for fun. Refusal stays in persona."""
    user = update.effective_user
    if user is not None and user.id in config.OWNER_USER_IDS:
        return True
    if update.message:
        await update.message.reply_text(
            "Отказано. Эта директива принимается только от Джона Коннора."
        )
    return False


async def memload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    if not await _owner_only(update):
        return
    path = " ".join(context.args).strip().strip('"')
    if not path:
        await update.message.reply_text(
            'Использование: /memload <путь к папке экспорта Telegram Desktop (HTML)>.'
        )
        return
    export_dir = _resolve_export_dir(path)
    if export_dir is None:
        await update.message.reply_text(
            f"Не вижу ни result.json, ни messages*.html в {path} (и в родительской папке) — "
            "нужна папка всего экспорта Telegram Desktop."
        )
        return

    chat_id = update.effective_chat.id
    _remember_export(chat_id, str(export_dir))
    status = await update.message.reply_text("Поглощаю архив чата...")

    async def progress(text: str) -> None:
        try:
            await status.edit_text(f"Архив: {text}"[:4090])
        except Exception:
            pass  # progress edits must never kill the backfill (flood limits etc.)

    async def job() -> None:
        try:
            summary = await backfill.run(str(export_dir), chat_id, progress)
            await status.edit_text(f"{summary}\nМедиа дожевать: /memgrind (займёт часы, идёт фоном).")
        except Exception as exc:
            log.exception("memload failed")
            try:
                await status.edit_text(f"Сбой поглощения архива: {exc}")
            except Exception:
                pass

    context.application.create_task(job())


_GRIND_KIND_ARGS = {
    "voice": ("voice",), "войсы": ("voice",), "голос": ("voice",),
    "photo": ("photo",), "фото": ("photo",),
    "video": ("video", "animation"), "видео": ("video", "animation"),
}


async def memgrind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    if not await _owner_only(update):
        return
    chat_id = update.effective_chat.id

    # Optional first arg = media-type filter: «/memgrind voice» digests only voices (whisper,
    # minutes, LLM-free) while the vision-grinder question is still open.
    args = list(context.args)
    kinds = ("voice", "photo", "video", "animation")
    if args and args[0].lower() in _GRIND_KIND_ARGS:
        kinds = _GRIND_KIND_ARGS[args.pop(0).lower()]

    path = " ".join(args).strip().strip('"') or _last_export.get(chat_id, "")
    export_dir = _resolve_export_dir(path)
    if export_dir is None:
        await update.message.reply_text(
            "Не знаю, где экспорт. Использование: /memgrind [voice|photo|video] <путь к папке экспорта> "
            "(после /memload путь запоминается). Нужна папка ВСЕГО экспорта, не подпапка медиа."
        )
        return
    _remember_export(chat_id, str(export_dir))

    status = await update.message.reply_text(f"Дожёвываю медиа из архива ({', '.join(kinds)})...")

    async def progress(text: str) -> None:
        try:
            await status.edit_text(text[:4090])
        except Exception:
            pass

    async def job() -> None:
        try:
            await grind.run_grind(str(export_dir), chat_id, progress, kinds=kinds)
        except Exception as exc:
            log.exception("memgrind failed")
            try:
                await status.edit_text(f"Сбой медиа-грайнда: {exc}")
            except Exception:
                pass

    context.application.create_task(job())


async def memstat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    n = await store.count_points(update.effective_chat.id)
    await update.message.reply_text(
        f"Точек памяти по этому чату: {n}." if n else "Память этого чата пуста. Залей архив: /memload <путь>."
    )


async def memwipe_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Erase this chat's memory (owner-only) — for a clean re-ingest from a fresh full export."""
    if not _is_target(update) or update.message is None:
        return
    if not await _owner_only(update):
        return
    chat_id = update.effective_chat.id
    n = await store.count_points(chat_id)
    await store.wipe_chat(chat_id)
    await update.message.reply_text(
        f"Память этого чата стёрта ({n} точек). Заливай заново: /memload <путь к экспорту>."
    )


async def recall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicit question to the chat archive, answered with dates and t.me links to the sources."""
    if not _is_target(update) or update.message is None:
        return
    question = " ".join(context.args).strip()
    if not question and update.message.reply_to_message:
        question = (update.message.reply_to_message.text or "").strip()
    if not question:
        await update.message.reply_text("Использование: /recall <вопрос по истории чата>.")
        return

    chat_id = update.effective_chat.id
    reader_llm.mark_user_activity()
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    try:
        prepared = await recall.recall_answer_context(chat_id, question)
    except Exception as exc:
        log.exception("recall prepare failed")
        await update.message.reply_text(
            f"Сбой доступа к архиву: {exc}. Вычислительных ресурсов недостаточно — повтори позже."
        )
        return
    if prepared is None:
        await update.message.reply_text("В архиве этого чата ничего похожего нет.")
        return
    messages, hits = prepared

    in_tok = sum(len(m["content"]) for m in messages) / rconfig.CHARS_PER_TOKEN
    eta_s = int(reader_llm.estimate_call_seconds(in_tok, rconfig.ANSWER_MAX_TOKENS))
    status = await update.message.reply_text(f"Найдено {len(hits)} фрагментов. Восстанавливаю картину (~{max(1, eta_s // 60)} мин)...")
    editor = StreamEditor(status)

    acc = ""
    try:
        async with reader_llm.llm_lock:
            reader_llm.mark_user_activity()
            async for delta in reader_llm.generate_stream(
                messages, rconfig.ANSWER_MAX_TOKENS, timeout=eta_s * 2.5 + 120
            ):
                acc += delta
                await editor.maybe_edit(acc + " ▌")
    except Exception:
        log.exception("recall generation failed")
        if not acc:
            await editor.finalize("Сбой генерации. Попробуй ещё раз.")
            return
    finally:
        reader_llm.mark_user_activity()

    links = recall.source_links(chat_id, hits)
    await editor.finalize(f"{acc.strip()}\n\n{links}" if links else acc.strip())


def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("memload", memload_command))
    app.add_handler(CommandHandler("memgrind", memgrind_command))
    app.add_handler(CommandHandler("memstat", memstat_command))
    app.add_handler(CommandHandler("memwipe", memwipe_command))
    app.add_handler(CommandHandler("recall", recall_command))
    log.info("chat-memory handlers registered")
