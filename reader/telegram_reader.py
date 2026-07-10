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
import i18n

from . import db, ingest, llm, manga, manga_ingest, query, rconfig
from .streaming import StreamEditor

log = logging.getLogger("t800.reader")

FILES_DIR = Path(__file__).resolve().parent.parent / "reader_files"

SUPPORTED_SUFFIXES = (".fb2", ".epub", ".fb2.zip")
MANGA_SUFFIXES = (".cbz", ".cbr", ".pdf")

# Retrieval-level refusals (distinct from the model's in-context "no such data" line). Bilingual.
REFUSAL_TEXT = {
    "en": {
        "not_ingested": (
            "This file isn't in my database, or it's still being digested. Send the book as a file."
        ),
        "nothing_unlocked": (
            "Your bookmark is at zero — access to future data is locked. Mark your position: "
            "/pos chapter 3, /pos read chapter 5, or /pos 40%."
        ),
        "low_confidence": (
            "There's no such data in the part of the book I've swallowed. Keep reading, bag of bones."
        ),
    },
    "ru": {
        "not_ingested": "Этого файла нет в моей базе данных или он ещё переваривается. Пришли книгу файлом.",
        "nothing_unlocked": (
            "Твоя закладка на нуле — доступ к данным будущего закрыт. Отметь позицию: "
            "/pos глава 3, /pos прочитал главу 5 или /pos 40%."
        ),
        "low_confidence": "В проглоченной части книги таких данных нет. Продолжай чтение, мешок с костями.",
    },
}

# All other user-facing strings of this layer. Bilingual via i18n.L.
STR = {
    "en": {
        "too_big": (
            "File is over 20 MB — the standard Bot API won't hand me anything that large. Books go "
            "as FB2/EPUB; a fat CBZ can be split by volume or chapter."
        ),
        "book_ready": (
            "\"{title}\" is already swallowed (tier: {tier}). Book set as active — ask questions "
            "via /ask. Upgrade the detail: /tier medium."
        ),
        "already_digesting": "This file is already being digested. Wait for the progress report.",
        "parse_failed": "Couldn't parse the file: {exc}",
        "btn_low": "Low — text search ({eta})",
        "btn_medium": "Medium — + chapter summaries ({eta})",
        "book_captured": (
            "Target acquired: \"{title}\"{author} — {n_chapters} chapters, {kchars}k characters.\n\n"
            "Pick the analysis depth:\n"
            "• Low: pinpoint questions on the text (\"what happened when X did Y\").\n"
            "• Medium: + whole-book questions (\"how did the hero change\"). You can add it later: "
            "/tier medium.\n\n"
            "After processing, set a bookmark (/pos chapter 5 or /pos 40%) — I'll answer only from "
            "the part you've read; everything past the bookmark is closed to me. No spoilers."
        ),
        "manga_ready": (
            "\"{title}\" is already read (tier: {tier}). Set as active — ask via /ask, bookmark: "
            "/pos page N."
        ),
        "already_reading": "This file is already being read. Wait for the progress report.",
        "manga_open_failed": "Couldn't open the comic: {exc}",
        "btn_manga_low": "Read ({eta})",
        "btn_manga_medium": "Read + chapter summaries ({eta})",
        "manga_captured": (
            "Target acquired: \"{title}\" — {n_pages} pages, {n_chapters} chapters.\n\n"
            "Reading a comic means a visual analysis of every page (dialogue + scenes), so even the "
            "basic option isn't fast. It runs in the background; I'll report back.\n\n"
            "Then set a bookmark (/pos page 40 or /pos chapter 3) — I answer only from the part "
            "you've read, no spoilers."
        ),
        "file_lost": "File lost from the database. Send it again.",
        "already_digesting_short": "Already digesting this file. Wait.",
        "source_lost": "The source file is gone from disk. Send it again.",
        "progress_prefix": "\"{title}\" — {text}",
        "tier_usage": "Usage: /tier medium — build the chapter summaries for the active book.",
        "no_active_book_books": "No active book. Send a file or pick one: /books.",
        "medium_already": "The medium tier is already built for this book.",
        "already_processing": "This book is already being processed.",
        "tier_accepted": "Acknowledged. I'll report progress.",
        "ask_usage": "Usage: /ask <question about the active book>.",
        "no_active_book_ask": "No active book. Send a book file or pick one from /books.",
        "prepare_failed": (
            "Archive access failure: {exc}. Insufficient compute resources — try again later."
        ),
        "answer_status": "Data found. Composing an answer (~{mins} min)...",
        "gen_failed_tail": "\n\n[generation failure — try again]",
        "gen_failed_full": "Generation failure. Try again.",
        "length_note": (
            "\n\n[answer hit the length limit — ask for a continuation or narrow the question]"
        ),
        "no_active_book_pos": "No active book. Send a book file or pick one from /books.",
        "pos_status": (
            "\"{title}\": your bookmark is at {pct:.0f}% ({off} characters).\n"
            "Move it: /pos chapter 5 (reading chapter five) · /pos read chapter 5 · /pos 40% · "
            "/pos whole book (open everything, spoilers allowed).\n"
            "Chapter list: /chapters."
        ),
        "pos_unparsed": (
            "Didn't get the position. Formats: /pos chapter 5 · /pos read chapter 5 · /pos 40% · "
            "/pos whole book."
        ),
        "pos_fixed": "Bookmark set: {desc}. Everything past it is closed to me — no spoilers.",
        "no_active_book_short": "No active book.",
        "chapters_more": "... and {n} more",
        "chapters_header": "Chapters:\n{lines}",
        "chapters_none": "No chapters found.",
        "books_empty": "Library is empty. Send a book file (FB2/EPUB) — I'll swallow it.",
        "books_active_mark": " ← active",
        "books_line": "{i}. \"{title}\" — {status}, tier {tier}{mark}",
        "books_header": "Swallowed books:\n{lines}\n\nPick: /book <number>. Ask: /ask <question>.",
        "book_usage": "Usage: /book <number from /books>.",
        "book_active": "Active book: \"{title}\". Ask your questions: /ask <question>.",
    },
    "ru": {
        "too_big": (
            "Файл больше 20 МБ — стандартный Bot API мне такое не отдаёт. Книги — FB2/EPUB; "
            "толстый CBZ можно разбить по томам/главам."
        ),
        "book_ready": (
            "«{title}» уже проглочена (тир: {tier}). Книга выбрана активной — задавай вопросы через "
            "/ask. Апгрейд детализации: /tier medium."
        ),
        "already_digesting": "Этот файл уже перевариваются. Жди отчёта о прогрессе.",
        "parse_failed": "Не смог разобрать файл: {exc}",
        "btn_low": "Низкий — поиск по тексту ({eta})",
        "btn_medium": "Средний — + сводки глав ({eta})",
        "book_captured": (
            "Цель захвачена: «{title}»{author} — {n_chapters} глав, {kchars}k знаков.\n\n"
            "Выбирай глубину анализа:\n"
            "• Низкий: точечные вопросы по тексту («что было, когда X сделал Y»).\n"
            "• Средний: + вопросы по всей книге («как менялся герой»). Можно догнать позже: "
            "/tier medium.\n\n"
            "После обработки поставь закладку (/pos глава 5 или /pos 40%) — отвечать буду только по "
            "прочитанной тобой части, всё дальше закладки для меня закрыто. Спойлеры исключены."
        ),
        "manga_ready": (
            "«{title}» уже прочитана (тир: {tier}). Выбрана активной — спрашивай через /ask, "
            "закладка: /pos страница N."
        ),
        "already_reading": "Этот файл уже читается. Жди отчёта о прогрессе.",
        "manga_open_failed": "Не смог открыть комикс: {exc}",
        "btn_manga_low": "Прочитать ({eta})",
        "btn_manga_medium": "Прочитать + сводки глав ({eta})",
        "manga_captured": (
            "Цель захвачена: «{title}» — {n_pages} страниц, {n_chapters} глав.\n\n"
            "Чтение комикса — это визуальный анализ каждой страницы (реплики + сцены), поэтому "
            "даже базовый вариант небыстрый. Идёт фоном, я доложу.\n\n"
            "Потом поставь закладку (/pos страница 40 или /pos глава 3) — отвечаю только по "
            "прочитанной тобой части, спойлеры исключены."
        ),
        "file_lost": "Файл потерян из базы. Пришли его заново.",
        "already_digesting_short": "Уже перевариваю этот файл. Жди.",
        "source_lost": "Исходный файл утерян с диска. Пришли его заново.",
        "progress_prefix": "«{title}» — {text}",
        "tier_usage": "Использование: /tier medium — достроить сводки глав активной книги.",
        "no_active_book_books": "Активной книги нет. Пришли файл или выбери: /books.",
        "medium_already": "Средний тир уже построен для этой книги.",
        "already_processing": "По этой книге уже идёт обработка.",
        "tier_accepted": "Принято. Прогресс буду докладывать.",
        "ask_usage": "Использование: /ask <вопрос по активной книге>.",
        "no_active_book_ask": "Активной книги нет. Пришли файл книги или выбери из /books.",
        "prepare_failed": (
            "Сбой доступа к архиву: {exc}. Вычислительных ресурсов недостаточно — повтори позже."
        ),
        "answer_status": "Данные найдены. Формирую ответ (~{mins} мин)...",
        "gen_failed_tail": "\n\n[сбой генерации — попробуй ещё раз]",
        "gen_failed_full": "Сбой генерации. Попробуй ещё раз.",
        "length_note": "\n\n[ответ упёрся в лимит длины — спроси продолжение или сузь вопрос]",
        "no_active_book_pos": "Активной книги нет. Пришли файл книги или выбери из /books.",
        "pos_status": (
            "«{title}»: твоя закладка на {pct:.0f}% ({off} знаков).\n"
            "Сдвинуть: /pos глава 5 (читаю пятую) · /pos прочитал главу 5 · /pos 40% · "
            "/pos вся книга (открыть всё, спойлеры разрешены).\n"
            "Список глав: /chapters."
        ),
        "pos_unparsed": (
            "Не понял позицию. Форматы: /pos глава 5 · /pos прочитал главу 5 · /pos 40% · /pos вся книга."
        ),
        "pos_fixed": "Закладка зафиксирована: {desc}. Всё, что дальше, для меня закрыто — спойлеров не будет.",
        "no_active_book_short": "Активной книги нет.",
        "chapters_more": "... и ещё {n}",
        "chapters_header": "Главы:\n{lines}",
        "chapters_none": "Глав не найдено.",
        "books_empty": "База пуста. Пришли книгу файлом (FB2/EPUB) — проглочу.",
        "books_active_mark": " ← активная",
        "books_line": "{i}. «{title}» — {status}, тир {tier}{mark}",
        "books_header": "Проглоченные книги:\n{lines}\n\nВыбрать: /book <номер>. Вопрос: /ask <вопрос>.",
        "book_usage": "Использование: /book <номер из /books>.",
        "book_active": "Активная книга: «{title}». Задавай вопросы: /ask <вопрос>.",
    },
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
    lang = i18n.get_lang(update.effective_chat.id)
    if msg.document.file_size and msg.document.file_size > rconfig.MAX_FILE_BYTES:
        await msg.reply_text(i18n.L(lang, STR, "too_big"))
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
            i18n.L(lang, STR, "book_ready", title=existing["title"], tier=existing["tier"])
        )
        return
    if existing and db.get_running_job(doc_id):
        await msg.reply_text(i18n.L(lang, STR, "already_digesting"))
        return

    try:
        parsed = await asyncio.to_thread(ingest.parse_bytes, data, msg.document.file_name)
    except Exception as exc:
        log.exception("parse failed")
        await msg.reply_text(i18n.L(lang, STR, "parse_failed", exc=exc))
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
    eta_low = ingest.eta_text(ingest.estimate_tier_seconds(parsed, n_leaves, "low"), lang)
    eta_med = ingest.eta_text(ingest.estimate_tier_seconds(parsed, n_leaves, "medium"), lang)
    p = doc_id[:16]
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(i18n.L(lang, STR, "btn_low", eta=eta_low), callback_data=f"rdt|{p}|low")],
            [InlineKeyboardButton(i18n.L(lang, STR, "btn_medium", eta=eta_med), callback_data=f"rdt|{p}|medium")],
        ]
    )
    author = f" ({parsed.author})" if parsed.author else ""
    await msg.reply_text(
        i18n.L(
            lang, STR, "book_captured",
            title=parsed.title, author=author, n_chapters=len(parsed.chapters),
            kchars=parsed.doc_len // 1000,
        ),
        reply_markup=keyboard,
    )


async def _receive_manga(update: Update, context: ContextTypes.DEFAULT_TYPE, data: bytes, doc_id: str) -> None:
    """CBZ/PDF path: page-per-chunk vision ingest instead of text parsing. The tier keyboard is
    reused, but for manga even the LOW tier costs vision calls — the ETA on the button is the
    honest price of reading every page."""
    msg = update.message
    lang = i18n.get_lang(update.effective_chat.id)
    existing = db.get_doc(doc_id)
    if existing and existing["status"] == "ready":
        db.set_active_doc(msg.chat.id, doc_id)
        await msg.reply_text(
            i18n.L(lang, STR, "manga_ready", title=existing["title"], tier=existing["tier"])
        )
        return
    if existing and db.get_running_job(doc_id):
        await msg.reply_text(i18n.L(lang, STR, "already_reading"))
        return

    FILES_DIR.mkdir(exist_ok=True)
    path = _doc_path(doc_id, msg.document.file_name)
    path.write_bytes(data)
    try:
        mdoc = await asyncio.to_thread(manga.open_manga, str(path), msg.document.file_name)
    except Exception as exc:
        log.exception("manga open failed")
        await msg.reply_text(i18n.L(lang, STR, "manga_open_failed", exc=exc))
        return

    db.upsert_doc(
        doc_id,
        title=mdoc.title, author="", file_name=msg.document.file_name,
        fmt="manga", doc_len=mdoc.n_pages, n_chapters=len(mdoc.chapters), status="new",
    )
    db.set_active_doc(msg.chat.id, doc_id)

    eta_low = ingest.eta_text(manga_ingest.estimate_manga_seconds(mdoc, "low"), lang)
    eta_med = ingest.eta_text(manga_ingest.estimate_manga_seconds(mdoc, "medium"), lang)
    p = doc_id[:16]
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(i18n.L(lang, STR, "btn_manga_low", eta=eta_low), callback_data=f"rdt|{p}|low")],
            [InlineKeyboardButton(i18n.L(lang, STR, "btn_manga_medium", eta=eta_med), callback_data=f"rdt|{p}|medium")],
        ]
    )
    await msg.reply_text(
        i18n.L(
            lang, STR, "manga_captured",
            title=mdoc.title, n_pages=mdoc.n_pages, n_chapters=len(mdoc.chapters),
        ),
        reply_markup=keyboard,
    )


async def tier_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if q is None or not q.data:
        return
    await q.answer()
    lang = i18n.get_lang(update.effective_chat.id)
    try:
        _, prefix, tier = q.data.split("|")
    except ValueError:
        return
    doc_row = db.find_doc_by_prefix(prefix)
    if doc_row is None:
        await q.edit_message_text(i18n.L(lang, STR, "file_lost"))
        return
    if db.get_running_job(doc_row["doc_id"]):
        await q.edit_message_text(i18n.L(lang, STR, "already_digesting_short"))
        return
    await _start_ingest(context, q.message.chat_id, q.message.message_id, doc_row, tier, lang)


async def _start_ingest(context, chat_id: int, message_id: int | None, doc_row, tier: str, lang: str) -> None:
    doc_id = doc_row["doc_id"]
    doc_path = _doc_path(doc_id, doc_row["file_name"])
    if not doc_path.exists():
        await context.bot.send_message(chat_id, i18n.L(lang, STR, "source_lost"))
        return

    title = doc_row["title"]

    async def progress(text: str) -> None:
        nonlocal message_id
        line = i18n.L(lang, STR, "progress_prefix", title=title, text=text)
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
        context.application.create_task(manga_ingest.run_manga_ingest(doc_id, mdoc, tier, job_id, progress, lang))
    else:
        data = doc_path.read_bytes()
        parsed = await asyncio.to_thread(ingest.parse_bytes, data, doc_row["file_name"])
        context.application.create_task(ingest.run_ingest(doc_id, parsed, tier, job_id, progress, lang))
    log.info("ingest task started: doc=%s fmt=%s tier=%s job=%s", doc_id[:12], doc_row["fmt"], tier, job_id)


async def tier_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/tier medium — additive upgrade of the active doc (brief §7: only missing stages run)."""
    if not _is_target(update):
        return
    lang = i18n.get_lang(update.effective_chat.id)
    tier = (context.args[0].lower() if context.args else "")
    if tier not in ("low", "medium"):
        await update.message.reply_text(i18n.L(lang, STR, "tier_usage"))
        return
    doc_id = db.get_active_doc(update.effective_chat.id)
    doc_row = db.get_doc(doc_id) if doc_id else None
    if doc_row is None:
        await update.message.reply_text(i18n.L(lang, STR, "no_active_book_books"))
        return
    if doc_row["tier"] == "medium" and tier == "medium":
        await update.message.reply_text(i18n.L(lang, STR, "medium_already"))
        return
    if db.get_running_job(doc_id):
        await update.message.reply_text(i18n.L(lang, STR, "already_processing"))
        return
    await _start_ingest(context, update.effective_chat.id, None, doc_row, tier, lang)
    await update.message.reply_text(i18n.L(lang, STR, "tier_accepted"))


# --- asking ---------------------------------------------------------------------------------------

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    lang = i18n.get_lang(update.effective_chat.id)
    question = " ".join(context.args).strip()
    if not question:
        await update.message.reply_text(i18n.L(lang, STR, "ask_usage"))
        return
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    doc_id = db.get_active_doc(chat_id)
    if not doc_id:
        await update.message.reply_text(i18n.L(lang, STR, "no_active_book_ask"))
        return

    llm.mark_user_activity()
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    try:
        result = await query.prepare(doc_id, user_id, question, lang)
    except Exception as exc:
        # Usually the embedder RAM guard on a starved box. A silent death reads as "the bot died" —
        # answer in persona instead.
        log.exception("ask prepare failed")
        await update.message.reply_text(i18n.L(lang, STR, "prepare_failed", exc=exc))
        return
    if isinstance(result, query.Refusal):
        log.info("refusal: %s (score=%s)", result.reason, result.best_score)
        key = result.reason if result.reason in REFUSAL_TEXT["en"] else "low_confidence"
        await update.message.reply_text(i18n.L(lang, REFUSAL_TEXT, key))
        return

    # Plot retellings (global route) get a bigger budget — 300 tokens cut a live one mid-word.
    max_tokens = (
        rconfig.ANSWER_MAX_TOKENS_GLOBAL if result.route.startswith("global") else rconfig.ANSWER_MAX_TOKENS
    )
    eta_s = query.estimate_answer_seconds(result, max_tokens)
    status = await update.message.reply_text(
        i18n.L(lang, STR, "answer_status", mins=max(1, eta_s // 60))
    )
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
            acc + i18n.L(lang, STR, "gen_failed_tail") if acc else i18n.L(lang, STR, "gen_failed_full")
        )
        return
    finally:
        llm.mark_user_activity()

    if acc.strip() and stats.get("finish_reason") == "length":
        acc = acc.rstrip() + i18n.L(lang, STR, "length_note")
    await editor.finalize(acc.strip() if acc.strip() else i18n.L(lang, REFUSAL_TEXT, "low_confidence"))


# --- bookmark -------------------------------------------------------------------------------------

def _parse_pos_args(args: list[str], chapters, doc_len: int, lang: str = "en") -> tuple[int, str] | None:
    """Returns (offset, human_description) or None if unparseable. Semantics per brief §6:
    'chapter N' / 'глава N' = reading it -> unlock to END of N-1; 'read chapter N' / 'прочитал
    главу N' -> end of N; 'X%' -> pct. The matcher and the descriptions are bilingual."""
    text = " ".join(args).lower().strip()
    if not text:
        return None

    finished = any(w in text for w in (
        "прочитал", "прочитала", "закончил", "закончила", "дочитал", "дочитала",
        "read", "finished", "done", "completed",
    ))

    # "whole book" / "вся книга" / "the end" / "конец" — unlock everything, spoilers explicitly welcome.
    whole_book = (
        any(w in text for w in ("вся", "всю", "целиком", "конец", "финал"))
        or any(w in text for w in ("whole book", "entire book", "whole thing", "everything", "the end", "finale"))
        or (finished and ("книг" in text or "book" in text))
    )
    if whole_book and not any(c.isdigit() for c in text):
        return doc_len, ("вся книга (спойлеры разрешены)" if lang == "ru" else "whole book (spoilers allowed)")

    # "page N" / "страница N" — the manga bookmark (offset unit here = page). Works for books too,
    # where it's meaningless but harmless: books have no page-offsets, so this branch is word-gated.
    if "стр" in text or "page" in text:
        import re as _re

        pm = _re.search(r"(\d+)", text)
        if pm:
            page = max(0, min(int(pm.group(1)), doc_len))
            return page, (f"страница {page} из {doc_len}" if lang == "ru" else f"page {page} of {doc_len}")
    m = None
    import re as _re

    pm = _re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    if pm:
        pct = min(100.0, float(pm.group(1).replace(",", ".")))
        off = int(doc_len * pct / 100)
        return off, (f"{pct:g}% книги" if lang == "ru" else f"{pct:g}% of the book")
    m = _re.search(r"(\d+)", text)
    if not m:
        return None
    n = int(m.group(1))  # user-facing 1-based chapter number
    if n < 1 or n > len(chapters):
        return None
    if finished:
        ch = chapters[n - 1]
        return ch["end_offset"], (
            f"конец главы {n} («{ch['title']}»)" if lang == "ru"
            else f"end of chapter {n} (\"{ch['title']}\")"
        )
    if n == 1:
        return 0, (
            "начало книги (глава 1 ещё не прочитана)" if lang == "ru"
            else "start of the book (chapter 1 not read yet)"
        )
    ch = chapters[n - 2]
    return ch["end_offset"], (
        f"конец главы {n - 1} (читаешь главу {n})" if lang == "ru"
        else f"end of chapter {n - 1} (reading chapter {n})"
    )


async def pos_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    lang = i18n.get_lang(update.effective_chat.id)
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    doc_id = db.get_active_doc(chat_id)
    doc_row = db.get_doc(doc_id) if doc_id else None
    if doc_row is None:
        await update.message.reply_text(i18n.L(lang, STR, "no_active_book_pos"))
        return
    chapters = db.get_chapters(doc_id)

    if not context.args:
        off = db.get_position(user_id, doc_id)
        pct = 100 * off / max(1, doc_row["doc_len"])
        await update.message.reply_text(
            i18n.L(lang, STR, "pos_status", title=doc_row["title"], pct=pct, off=off)
        )
        return

    parsed = _parse_pos_args(context.args, chapters, doc_row["doc_len"], lang)
    if parsed is None:
        await update.message.reply_text(i18n.L(lang, STR, "pos_unparsed"))
        return
    offset, desc = parsed
    db.set_position(user_id, doc_id, offset)
    await update.message.reply_text(i18n.L(lang, STR, "pos_fixed", desc=desc))


async def chapters_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    lang = i18n.get_lang(update.effective_chat.id)
    doc_id = db.get_active_doc(update.effective_chat.id)
    if not doc_id:
        await update.message.reply_text(i18n.L(lang, STR, "no_active_book_short"))
        return
    chapters = db.get_chapters(doc_id)
    lines = [f"{i + 1}. {c['title']}" for i, c in enumerate(chapters[:60])]
    if len(chapters) > 60:
        lines.append(i18n.L(lang, STR, "chapters_more", n=len(chapters) - 60))
    if lines:
        await update.message.reply_text(i18n.L(lang, STR, "chapters_header", lines="\n".join(lines)))
    else:
        await update.message.reply_text(i18n.L(lang, STR, "chapters_none"))


# --- library --------------------------------------------------------------------------------------

async def books_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    lang = i18n.get_lang(update.effective_chat.id)
    docs = db.list_docs()
    if not docs:
        await update.message.reply_text(i18n.L(lang, STR, "books_empty"))
        return
    active = db.get_active_doc(update.effective_chat.id)
    mark_active = i18n.L(lang, STR, "books_active_mark")
    lines = []
    for i, d in enumerate(docs, 1):
        mark = mark_active if d["doc_id"] == active else ""
        lines.append(
            i18n.L(lang, STR, "books_line", i=i, title=d["title"], status=d["status"], tier=d["tier"], mark=mark)
        )
    await update.message.reply_text(i18n.L(lang, STR, "books_header", lines="\n".join(lines)))


async def book_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_target(update) or update.message is None:
        return
    lang = i18n.get_lang(update.effective_chat.id)
    docs = db.list_docs()
    try:
        n = int(context.args[0])
        doc = docs[n - 1]
    except (IndexError, ValueError, TypeError):
        await update.message.reply_text(i18n.L(lang, STR, "book_usage"))
        return
    db.set_active_doc(update.effective_chat.id, doc["doc_id"])
    await update.message.reply_text(i18n.L(lang, STR, "book_active", title=doc["title"]))


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
