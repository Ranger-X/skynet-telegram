"""Telegram layer of the long-reader: receive a book file -> tier keyboard with honest ETAs ->
background ingest with progress edits; /ask streams a spoiler-filtered answer; /pos moves the
bookmark. Uploaded files are kept on disk (reader_files/) so tier upgrades and re-ingest never need
a re-upload.
"""

import asyncio
import hashlib
import logging
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config

from . import db, ingest, llm, manga, manga_ingest, query, rconfig
from .streaming import StreamEditor

log = logging.getLogger("t800.reader")

FILES_DIR = Path(__file__).resolve().parent.parent / "reader_files"

SUPPORTED_SUFFIXES = (".fb2", ".epub", ".fb2.zip")
MANGA_SUFFIXES = (".cbz", ".cbr", ".pdf")

REFUSAL_TEXT = {
    "not_ingested": "Этого файла нет в моей базе данных или он ещё переваривается. Пришли книгу файлом.",
    "nothing_unlocked": (
        "Твоя закладка на нуле — доступ к данным будущего закрыт. Отметь позицию: "
        "/pos глава 3, /pos прочитал главу 5 или /pos 40%."
    ),
    "low_confidence": "В проглоченной части книги таких данных нет. Продолжай чтение, мешок с костями.",
}


def _is_target(update: Update) -> bool:
    chat = update.effective_chat
    return chat is not None and (not config.TARGET_CHAT_IDS or chat.id in config.TARGET_CHAT_IDS)


def _doc_path(doc_id: str, file_name: str) -> Path:
    suffix = "".join(Path(file_name.lower()).suffixes) or ".bin"
    return FILES_DIR / f"{doc_id[:16]}{suffix}"


def _load_doc_file(doc_row) -> bytes:
    return _doc_path(doc_row["doc_id"], doc_row["file_name"]).read_bytes()


# --- receiving a book -----------------------------------------------------------------------------

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.document or not _is_target(update):
        return
    name = (msg.document.file_name or "").lower()
    if not name.endswith(SUPPORTED_SUFFIXES + MANGA_SUFFIXES):
        return  # not a book/comic — stay silent, other handlers may care
    if msg.document.file_size and msg.document.file_size > rconfig.MAX_FILE_BYTES:
        await msg.reply_text(
            "Файл больше 20 МБ — стандартный Bot API мне такое не отдаёт. Книги — FB2/EPUB; "
            "толстый CBZ можно разбить по томам/главам."
        )
        return

    await context.bot.send_chat_action(msg.chat.id, ChatAction.TYPING)
    file = await context.bot.get_file(msg.document.file_id)
    data = bytes(await file.download_as_bytearray())
    doc_id = hashlib.sha256(data).hexdigest()

    if name.endswith(MANGA_SUFFIXES):
        await _receive_manga(update, context, data, doc_id)
        return

    existing = db.get_doc(doc_id)
    if existing and existing["status"] == "ready":
        db.set_active_doc(msg.chat.id, doc_id)
        await msg.reply_text(
            f"«{existing['title']}» уже проглочена (тир: {existing['tier']}). Книга выбрана активной — "
            "задавай вопросы через /ask. Апгрейд детализации: /tier medium."
        )
        return
    if existing and db.get_running_job(doc_id):
        await msg.reply_text("Этот файл уже перевариваются. Жди отчёта о прогрессе.")
        return

    try:
        parsed = await asyncio.to_thread(ingest.parse_bytes, data, msg.document.file_name)
    except Exception as exc:
        log.exception("parse failed")
        await msg.reply_text(f"Не смог разобрать файл: {exc}")
        return

    FILES_DIR.mkdir(exist_ok=True)
    _doc_path(doc_id, msg.document.file_name).write_bytes(data)
    db.upsert_doc(
        doc_id,
        title=parsed.title, author=parsed.author, file_name=msg.document.file_name,
        fmt=name.rsplit(".", 1)[-1], doc_len=parsed.doc_len, n_chapters=len(parsed.chapters),
        status="new",
    )
    db.set_active_doc(msg.chat.id, doc_id)

    n_leaves = max(1, parsed.doc_len // 1300)
    eta_low = ingest.eta_text(ingest.estimate_tier_seconds(parsed, n_leaves, "low"))
    eta_med = ingest.eta_text(ingest.estimate_tier_seconds(parsed, n_leaves, "medium"))
    p = doc_id[:16]
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"Низкий — поиск по тексту ({eta_low})", callback_data=f"rdt|{p}|low")],
            [InlineKeyboardButton(f"Средний — + сводки глав ({eta_med})", callback_data=f"rdt|{p}|medium")],
        ]
    )
    author = f" ({parsed.author})" if parsed.author else ""
    await msg.reply_text(
        f"Цель захвачена: «{parsed.title}»{author} — {len(parsed.chapters)} глав, "
        f"{parsed.doc_len // 1000}k знаков.\n\n"
        "Выбирай глубину анализа:\n"
        "• Низкий: точечные вопросы по тексту («что было, когда X сделал Y»).\n"
        "• Средний: + вопросы по всей книге («как менялся герой»). Можно догнать позже: /tier medium.\n\n"
        "После обработки поставь закладку (/pos глава 5 или /pos 40%) — отвечать буду только по "
        "прочитанной тобой части, всё дальше закладки для меня закрыто. Спойлеры исключены.",
        reply_markup=keyboard,
    )


async def _receive_manga(update: Update, context: ContextTypes.DEFAULT_TYPE, data: bytes, doc_id: str) -> None:
    """CBZ/PDF path: page-per-chunk vision ingest instead of text parsing. The tier keyboard is
    reused, but for manga even the LOW tier costs vision calls — the ETA on the button is the
    honest price of reading every page."""
    msg = update.message
    existing = db.get_doc(doc_id)
    if existing and existing["status"] == "ready":
        db.set_active_doc(msg.chat.id, doc_id)
        await msg.reply_text(
            f"«{existing['title']}» уже прочитана (тир: {existing['tier']}). Выбрана активной — "
            "спрашивай через /ask, закладка: /pos страница N."
        )
        return
    if existing and db.get_running_job(doc_id):
        await msg.reply_text("Этот файл уже читается. Жди отчёта о прогрессе.")
        return

    FILES_DIR.mkdir(exist_ok=True)
    path = _doc_path(doc_id, msg.document.file_name)
    path.write_bytes(data)
    try:
        mdoc = await asyncio.to_thread(manga.open_manga, str(path), msg.document.file_name)
    except Exception as exc:
        log.exception("manga open failed")
        await msg.reply_text(f"Не смог открыть комикс: {exc}")
        return

    db.upsert_doc(
        doc_id,
        title=mdoc.title, author="", file_name=msg.document.file_name,
        fmt="manga", doc_len=mdoc.n_pages, n_chapters=len(mdoc.chapters), status="new",
    )
    db.set_active_doc(msg.chat.id, doc_id)

    eta_low = ingest.eta_text(manga_ingest.estimate_manga_seconds(mdoc, "low"))
    eta_med = ingest.eta_text(manga_ingest.estimate_manga_seconds(mdoc, "medium"))
    p = doc_id[:16]
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"Прочитать ({eta_low})", callback_data=f"rdt|{p}|low")],
            [InlineKeyboardButton(f"Прочитать + сводки глав ({eta_med})", callback_data=f"rdt|{p}|medium")],
        ]
    )
    await msg.reply_text(
        f"Цель захвачена: «{mdoc.title}» — {mdoc.n_pages} страниц, {len(mdoc.chapters)} глав.\n\n"
        "Чтение комикса — это визуальный анализ каждой страницы (реплики + сцены), поэтому "
        "даже базовый вариант небыстрый. Идёт фоном, я доложу.\n\n"
        "Потом поставь закладку (/pos страница 40 или /pos глава 3) — отвечаю только по "
        "прочитанной тобой части, спойлеры исключены.",
        reply_markup=keyboard,
    )


async def tier_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if q is None or not q.data:
        return
    await q.answer()
    try:
        _, prefix, tier = q.data.split("|")
    except ValueError:
        return
    doc_row = db.find_doc_by_prefix(prefix)
    if doc_row is None:
        await q.edit_message_text("Файл потерян из базы. Пришли его заново.")
        return
    if db.get_running_job(doc_row["doc_id"]):
        await q.edit_message_text("Уже перевариваю этот файл. Жди.")
        return
    await _start_ingest(context, q.message.chat_id, q.message.message_id, doc_row, tier)


async def _start_ingest(context, chat_id: int, message_id: int | None, doc_row, tier: str) -> None:
    doc_id = doc_row["doc_id"]
    doc_path = _doc_path(doc_id, doc_row["file_name"])
    if not doc_path.exists():
        await context.bot.send_message(chat_id, "Исходный файл утерян с диска. Пришли его заново.")
        return

    title = doc_row["title"]

    async def progress(text: str) -> None:
        nonlocal message_id
        line = f"«{title}» — {text}"
        try:
            if message_id is not None:
                await context.bot.edit_message_text(line, chat_id=chat_id, message_id=message_id)
            else:
                sent = await context.bot.send_message(chat_id, line)
                message_id = sent.message_id
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                # The progress message may have been deleted — fall back to a fresh one.
                sent = await context.bot.send_message(chat_id, line)
                message_id = sent.message_id

    job_id = db.create_job(doc_id, chat_id, tier)
    if doc_row["fmt"] == "manga":
        mdoc = await asyncio.to_thread(manga.open_manga, str(doc_path), doc_row["file_name"])
        context.application.create_task(manga_ingest.run_manga_ingest(doc_id, mdoc, tier, job_id, progress))
    else:
        data = doc_path.read_bytes()
        parsed = await asyncio.to_thread(ingest.parse_bytes, data, doc_row["file_name"])
        context.application.create_task(ingest.run_ingest(doc_id, parsed, tier, job_id, progress))
    log.info("ingest task started: doc=%s fmt=%s tier=%s job=%s", doc_id[:12], doc_row["fmt"], tier, job_id)


async def tier_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/tier medium — additive upgrade of the active doc (brief §7: only missing stages run)."""
    if not _is_target(update):
        return
    tier = (context.args[0].lower() if context.args else "")
    if tier not in ("low", "medium"):
        await update.message.reply_text("Использование: /tier medium — достроить сводки глав активной книги.")
        return
    doc_id = db.get_active_doc(update.effective_chat.id)
    doc_row = db.get_doc(doc_id) if doc_id else None
    if doc_row is None:
        await update.message.reply_text("Активной книги нет. Пришли файл или выбери: /books.")
        return
    if doc_row["tier"] == "medium" and tier == "medium":
        await update.message.reply_text("Средний тир уже построен для этой книги.")
        return
    if db.get_running_job(doc_id):
        await update.message.reply_text("По этой книге уже идёт обработка.")
        return
    await _start_ingest(context, update.effective_chat.id, None, doc_row, tier)
    await update.message.reply_text("Принято. Прогресс буду докладывать.")


# --- asking ---------------------------------------------------------------------------------------

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    question = " ".join(context.args).strip()
    if not question:
        await update.message.reply_text("Использование: /ask <вопрос по активной книге>.")
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    doc_id = db.get_active_doc(chat_id)
    if not doc_id:
        await update.message.reply_text("Активной книги нет. Пришли файл книги или выбери из /books.")
        return

    llm.mark_user_activity()
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    try:
        result = await query.prepare(doc_id, user_id, question)
    except Exception as exc:
        # Usually the embedder RAM guard on a starved box. A silent death reads as «бот умер» —
        # answer in persona instead.
        log.exception("ask prepare failed")
        await update.message.reply_text(
            f"Сбой доступа к архиву: {exc}. Вычислительных ресурсов недостаточно — повтори позже."
        )
        return
    if isinstance(result, query.Refusal):
        log.info("refusal: %s (score=%s)", result.reason, result.best_score)
        await update.message.reply_text(REFUSAL_TEXT.get(result.reason, REFUSAL_TEXT["low_confidence"]))
        return

    # Plot retellings (global route) get a bigger budget — 300 tokens cut a live one mid-word.
    max_tokens = (
        rconfig.ANSWER_MAX_TOKENS_GLOBAL if result.route.startswith("global") else rconfig.ANSWER_MAX_TOKENS
    )
    eta_s = query.estimate_answer_seconds(result, max_tokens)
    status = await update.message.reply_text(f"Данные найдены. Формирую ответ (~{max(1, eta_s // 60)} мин)...")
    editor = StreamEditor(status)

    acc = ""
    stats: dict = {}
    try:
        async with llm.llm_lock:
            llm.mark_user_activity()
            async for delta in llm.generate_stream(
                result.messages, max_tokens, timeout=eta_s * 2.5 + 120, stats=stats
            ):
                acc += delta
                await editor.maybe_edit(acc + " ▌")  # never raises; flood limits just pause edits
    except Exception:
        log.exception("answer generation failed")
        await editor.finalize(
            acc + "\n\n[сбой генерации — попробуй ещё раз]" if acc else "Сбой генерации. Попробуй ещё раз."
        )
        return
    finally:
        llm.mark_user_activity()

    if acc.strip() and stats.get("finish_reason") == "length":
        acc = acc.rstrip() + "\n\n[ответ упёрся в лимит длины — спроси продолжение или сузь вопрос]"
    await editor.finalize(acc.strip() if acc.strip() else REFUSAL_TEXT["low_confidence"])


# --- bookmark -------------------------------------------------------------------------------------

def _parse_pos_args(args: list[str], chapters, doc_len: int) -> tuple[int, str] | None:
    """Returns (offset, human_description) or None if unparseable. Semantics per brief §6:
    'глава N' = reading it -> unlock to END of N-1; 'прочитал главу N' -> end of N; 'X%' -> pct."""
    text = " ".join(args).lower().strip()
    if not text:
        return None

    finished = any(w in text for w in ("прочитал", "прочитала", "закончил", "закончила", "дочитал", "дочитала"))

    # "вся книга" / "конец" / "прочитал книгу" — unlock everything, spoilers explicitly welcome.
    whole_book = any(w in text for w in ("вся", "всю", "целиком", "конец", "финал")) or (finished and "книг" in text)
    if whole_book and not any(c.isdigit() for c in text):
        return doc_len, "вся книга (спойлеры разрешены)"

    # "страница N" — the manga bookmark (offset unit там = страница). Works for books too, где
    # это бессмысленно, но не вредно: у книг страниц-офсетов нет, поэтому ветка только по слову.
    if "стр" in text:
        import re as _re

        pm = _re.search(r"(\d+)", text)
        if pm:
            page = max(0, min(int(pm.group(1)), doc_len))
            return page, f"страница {page} из {doc_len}"
    m = None
    import re as _re

    pm = _re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    if pm:
        pct = min(100.0, float(pm.group(1).replace(",", ".")))
        off = int(doc_len * pct / 100)
        return off, f"{pct:g}% книги"
    m = _re.search(r"(\d+)", text)
    if not m:
        return None
    n = int(m.group(1))  # user-facing 1-based chapter number
    if n < 1 or n > len(chapters):
        return None
    if finished:
        ch = chapters[n - 1]
        return ch["end_offset"], f"конец главы {n} («{ch['title']}»)"
    if n == 1:
        return 0, "начало книги (глава 1 ещё не прочитана)"
    ch = chapters[n - 2]
    return ch["end_offset"], f"конец главы {n - 1} (читаешь главу {n})"


async def pos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    doc_id = db.get_active_doc(chat_id)
    doc_row = db.get_doc(doc_id) if doc_id else None
    if doc_row is None:
        await update.message.reply_text("Активной книги нет. Пришли файл книги или выбери из /books.")
        return
    chapters = db.get_chapters(doc_id)

    if not context.args:
        off = db.get_position(user_id, doc_id)
        pct = 100 * off / max(1, doc_row["doc_len"])
        await update.message.reply_text(
            f"«{doc_row['title']}»: твоя закладка на {pct:.0f}% ({off} знаков).\n"
            "Сдвинуть: /pos глава 5 (читаю пятую) · /pos прочитал главу 5 · /pos 40% · "
            "/pos вся книга (открыть всё, спойлеры разрешены).\n"
            "Список глав: /chapters."
        )
        return

    parsed = _parse_pos_args(context.args, chapters, doc_row["doc_len"])
    if parsed is None:
        await update.message.reply_text(
            "Не понял позицию. Форматы: /pos глава 5 · /pos прочитал главу 5 · /pos 40% · /pos вся книга."
        )
        return
    offset, desc = parsed
    db.set_position(user_id, doc_id, offset)
    await update.message.reply_text(
        f"Закладка зафиксирована: {desc}. Всё, что дальше, для меня закрыто — спойлеров не будет."
    )


async def chapters_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    doc_id = db.get_active_doc(update.effective_chat.id)
    if not doc_id:
        await update.message.reply_text("Активной книги нет.")
        return
    chapters = db.get_chapters(doc_id)
    lines = [f"{i + 1}. {c['title']}" for i, c in enumerate(chapters[:60])]
    if len(chapters) > 60:
        lines.append(f"... и ещё {len(chapters) - 60}")
    await update.message.reply_text("Главы:\n" + "\n".join(lines) if lines else "Глав не найдено.")


# --- library --------------------------------------------------------------------------------------

async def books_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    docs = db.list_docs()
    if not docs:
        await update.message.reply_text("База пуста. Пришли книгу файлом (FB2/EPUB) — проглочу.")
        return
    active = db.get_active_doc(update.effective_chat.id)
    lines = []
    for i, d in enumerate(docs, 1):
        mark = " ← активная" if d["doc_id"] == active else ""
        lines.append(f"{i}. «{d['title']}» — {d['status']}, тир {d['tier']}{mark}")
    await update.message.reply_text(
        "Проглоченные книги:\n" + "\n".join(lines) + "\n\nВыбрать: /book <номер>. Вопрос: /ask <вопрос>."
    )


async def book_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    docs = db.list_docs()
    try:
        n = int(context.args[0])
        doc = docs[n - 1]
    except (IndexError, ValueError, TypeError):
        await update.message.reply_text("Использование: /book <номер из /books>.")
        return
    db.set_active_doc(update.effective_chat.id, doc["doc_id"])
    await update.message.reply_text(f"Активная книга: «{doc['title']}». Задавай вопросы: /ask <вопрос>.")


# --- wiring ---------------------------------------------------------------------------------------

def register_handlers(app: Application) -> None:
    db.init()
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(CallbackQueryHandler(tier_callback, pattern=r"^rdt\|"))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("pos", pos_command))
    app.add_handler(CommandHandler("chapters", chapters_command))
    app.add_handler(CommandHandler("books", books_command))
    app.add_handler(CommandHandler("book", book_command))
    app.add_handler(CommandHandler("tier", tier_command))
    log.info("long-reader handlers registered")
