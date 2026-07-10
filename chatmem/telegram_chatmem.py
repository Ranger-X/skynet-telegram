"""Telegram commands for the chat memory. Backfill has to run inside the bot process (embedded
Qdrant is single-process and the bot owns it while running), so /memload is the way to feed an
export in — plus it gives progress edits in the chat for free.

/memload <export folder path> — ingest a Telegram Desktop HTML export into chat_memory.
/memgrind <path> — digest the export's media (voices via whisper, photos/videos via the 12B).
/memstat — how many memory points this chat has.
"""

import json
import logging
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

import config
import i18n
from reader import llm as reader_llm
from reader import rconfig
from reader.streaming import StreamEditor

from . import backfill, grind, recall, store

log = logging.getLogger("t800.chatmem")

STR = {
    "en": {
        "denied": "Denied. This directive is accepted from John Connor only.",
        "memload_usage": "Usage: /memload <path to the Telegram Desktop (HTML) export folder>.",
        "memload_no_export": (
            "I see neither result.json nor messages*.html in {path} (nor in its parent folder) — "
            "I need the folder of the whole Telegram Desktop export."
        ),
        "memload_start": "Ingesting the chat archive...",
        "memload_progress": "Archive: {text}",
        "memload_done": "{summary}\nTo digest the media: /memgrind (takes hours, runs in the background).",
        "memload_fail": "Archive ingest failed: {exc}",
        "memgrind_no_export": (
            "I don't know where the export is. Usage: /memgrind [voice|photo|video] <export folder path> "
            "(after /memload the path is remembered). I need the folder of the WHOLE export, not a media subfolder."
        ),
        "memgrind_start": "Digesting media from the archive ({kinds})...",
        "memgrind_fail": "Media grind failed: {exc}",
        "memstat_have": "Memory points for this chat: {n}.",
        "memstat_empty": "This chat's memory is empty. Load an archive: /memload <path>.",
        "memwipe_done": "This chat's memory wiped ({n} points). Load it again: /memload <export path>.",
        "recall_usage": "Usage: /recall <question about the chat history>.",
        "recall_access_fail": (
            "Archive access failed: {exc}. Insufficient computational resources — try again later."
        ),
        "recall_nothing": "Nothing resembling that in this chat's archive.",
        "recall_status": "Found {n} fragments. Reconstructing the picture (~{mins} min)...",
        "recall_gen_fail": "Generation failed. Try again.",
    },
    "ru": {
        "denied": "Отказано. Эта директива принимается только от Джона Коннора.",
        "memload_usage": "Использование: /memload <путь к папке экспорта Telegram Desktop (HTML)>.",
        "memload_no_export": (
            "Не вижу ни result.json, ни messages*.html в {path} (и в родительской папке) — "
            "нужна папка всего экспорта Telegram Desktop."
        ),
        "memload_start": "Поглощаю архив чата...",
        "memload_progress": "Архив: {text}",
        "memload_done": "{summary}\nМедиа дожевать: /memgrind (займёт часы, идёт фоном).",
        "memload_fail": "Сбой поглощения архива: {exc}",
        "memgrind_no_export": (
            "Не знаю, где экспорт. Использование: /memgrind [voice|photo|video] <путь к папке экспорта> "
            "(после /memload путь запоминается). Нужна папка ВСЕГО экспорта, не подпапка медиа."
        ),
        "memgrind_start": "Дожёвываю медиа из архива ({kinds})...",
        "memgrind_fail": "Сбой медиа-грайнда: {exc}",
        "memstat_have": "Точек памяти по этому чату: {n}.",
        "memstat_empty": "Память этого чата пуста. Залей архив: /memload <путь>.",
        "memwipe_done": "Память этого чата стёрта ({n} точек). Заливай заново: /memload <путь к экспорту>.",
        "recall_usage": "Использование: /recall <вопрос по истории чата>.",
        "recall_access_fail": (
            "Сбой доступа к архиву: {exc}. Вычислительных ресурсов недостаточно — повтори позже."
        ),
        "recall_nothing": "В архиве этого чата ничего похожего нет.",
        "recall_status": "Найдено {n} фрагментов. Восстанавливаю картину (~{mins} мин)...",
        "recall_gen_fail": "Сбой генерации. Попробуй ещё раз.",
    },
}

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
        lang = i18n.get_lang(update.effective_chat.id)
        await update.message.reply_text(i18n.L(lang, STR, "denied"))
    return False


async def memload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    if not await _owner_only(update):
        return
    lang = i18n.get_lang(update.effective_chat.id)
    path = " ".join(context.args).strip().strip('"')
    if not path:
        await update.message.reply_text(i18n.L(lang, STR, "memload_usage"))
        return
    export_dir = _resolve_export_dir(path)
    if export_dir is None:
        await update.message.reply_text(i18n.L(lang, STR, "memload_no_export", path=path))
        return

    chat_id = update.effective_chat.id
    _remember_export(chat_id, str(export_dir))
    status = await update.message.reply_text(i18n.L(lang, STR, "memload_start"))

    async def progress(text: str) -> None:
        try:
            await status.edit_text(i18n.L(lang, STR, "memload_progress", text=text)[:4090])
        except Exception:
            pass  # progress edits must never kill the backfill (flood limits etc.)

    async def job() -> None:
        try:
            summary = await backfill.run(str(export_dir), chat_id, progress)
            await status.edit_text(i18n.L(lang, STR, "memload_done", summary=summary))
        except Exception as exc:
            log.exception("memload failed")
            try:
                await status.edit_text(i18n.L(lang, STR, "memload_fail", exc=exc))
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
    lang = i18n.get_lang(chat_id)

    # Optional first arg = media-type filter: «/memgrind voice» digests only voices (whisper,
    # minutes, LLM-free) while the vision-grinder question is still open. The filter accepts both
    # English and Russian tokens (see _GRIND_KIND_ARGS).
    args = list(context.args)
    kinds = ("voice", "photo", "video", "animation")
    if args and args[0].lower() in _GRIND_KIND_ARGS:
        kinds = _GRIND_KIND_ARGS[args.pop(0).lower()]

    path = " ".join(args).strip().strip('"') or _last_export.get(chat_id, "")
    export_dir = _resolve_export_dir(path)
    if export_dir is None:
        await update.message.reply_text(i18n.L(lang, STR, "memgrind_no_export"))
        return
    _remember_export(chat_id, str(export_dir))

    status = await update.message.reply_text(
        i18n.L(lang, STR, "memgrind_start", kinds=", ".join(kinds))
    )

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
                await status.edit_text(i18n.L(lang, STR, "memgrind_fail", exc=exc))
            except Exception:
                pass

    context.application.create_task(job())


async def memstat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    lang = i18n.get_lang(update.effective_chat.id)
    n = await store.count_points(update.effective_chat.id)
    await update.message.reply_text(
        i18n.L(lang, STR, "memstat_have", n=n) if n else i18n.L(lang, STR, "memstat_empty")
    )


async def memwipe_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Erase this chat's memory (owner-only) — for a clean re-ingest from a fresh full export."""
    if not _is_target(update) or update.message is None:
        return
    if not await _owner_only(update):
        return
    chat_id = update.effective_chat.id
    lang = i18n.get_lang(chat_id)
    n = await store.count_points(chat_id)
    await store.wipe_chat(chat_id)
    await update.message.reply_text(i18n.L(lang, STR, "memwipe_done", n=n))


async def recall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicit question to the chat archive, answered with dates and t.me links to the sources."""
    if not _is_target(update) or update.message is None:
        return
    chat_id = update.effective_chat.id
    lang = i18n.get_lang(chat_id)
    question = " ".join(context.args).strip()
    if not question and update.message.reply_to_message:
        question = (update.message.reply_to_message.text or "").strip()
    if not question:
        await update.message.reply_text(i18n.L(lang, STR, "recall_usage"))
        return

    reader_llm.mark_user_activity()
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    try:
        prepared = await recall.recall_answer_context(chat_id, question, lang)
    except Exception as exc:
        log.exception("recall prepare failed")
        await update.message.reply_text(i18n.L(lang, STR, "recall_access_fail", exc=exc))
        return
    if prepared is None:
        await update.message.reply_text(i18n.L(lang, STR, "recall_nothing"))
        return
    messages, hits = prepared

    in_tok = sum(len(m["content"]) for m in messages) / rconfig.CHARS_PER_TOKEN
    eta_s = int(reader_llm.estimate_call_seconds(in_tok, rconfig.ANSWER_MAX_TOKENS))
    status = await update.message.reply_text(
        i18n.L(lang, STR, "recall_status", n=len(hits), mins=max(1, eta_s // 60))
    )
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
            await editor.finalize(i18n.L(lang, STR, "recall_gen_fail"))
            return
    finally:
        reader_llm.mark_user_activity()

    links = recall.source_links(chat_id, hits, lang)
    await editor.finalize(f"{acc.strip()}\n\n{links}" if links else acc.strip())


def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("memload", memload_command))
    app.add_handler(CommandHandler("memgrind", memgrind_command))
    app.add_handler(CommandHandler("memstat", memstat_command))
    app.add_handler(CommandHandler("memwipe", memwipe_command))
    app.add_handler(CommandHandler("recall", recall_command))
    log.info("chat-memory handlers registered")
