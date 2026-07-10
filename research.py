"""Deep research command (/research, /deep_research): multi-angle web search -> fetch pages ->
pick the most relevant paragraphs -> ONE synthesis generation with numbered citations, streamed
into the chat. Built for settling group-chat arguments.

Shaped by the same hardware reality as the long-reader (see long-reader-architecture.md §1a):
generation on the local 12B costs minutes, so the middle of the pipeline is LLM-free — search
angles are fixed heuristics (no query-generation call), paragraph relevance is scored by the
reader's cross-encoder (seconds on CPU), and the LLM runs exactly once at the end. Synthesis goes
local-first (streamed); if llama-server is down it falls back to the OpenRouter path (one edit,
no stream).
"""

import asyncio
import logging
import re
import time
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from lxml import html as lxml_html
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

import config
from openrouter_client import get_reply
from reader import llm as reader_llm
from reader import rconfig, reranker
from reader.streaming import StreamEditor

log = logging.getLogger("t800.research")

# Search angles for a debate-shaped topic. The bare topic plus these suffixes cover the "both
# sides + hard data" spread without spending a generation call on query planning.
ANGLES = ["", "аргументы за и против", "факты статистика исследования", "критика опровержение мифы"]
RESULTS_PER_ANGLE = 4
MAX_SOURCES = 6          # pages actually fetched
MAX_PARAS_PER_PAGE = 30  # lexical-prefilter cap before the cross-encoder sees anything
RERANK_CANDIDATES = 28   # paragraphs scored by the cross-encoder
CONTEXT_PARAS = 10       # paragraphs fed to the synthesis prompt
PARA_MAX_CHARS = 900
REPORT_MAX_TOKENS = 450
FETCH_TIMEOUT = 12.0

RESEARCH_SYSTEM = (
    "Ты — Т-800, кибернетический организм, проводишь исследование по запросу людей из чата. "
    "Отвечаешь по-военному чётко, слегка надменно, но строго по данным. "
    "ЖЕЛЕЗНЫЕ ПРАВИЛА: опирайся ТОЛЬКО на приведённые фрагменты источников; на каждый факт ставь "
    "ссылку вида [n] по номеру источника; если данные противоречат друг другу — скажи об этом прямо; "
    "если данных мало — признай это, не выдумывай. Структура ответа: 1) вердикт в одну-две фразы; "
    "2) ключевые факты и аргументы сторон со ссылками; 3) итог: чья позиция подтверждается данными. "
    "Без markdown-заголовков, обычный связный текст. Не пиши список источников в конце — он будет "
    "добавлен автоматически."
)


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[а-яa-zё0-9-]{4,}", text.lower())}


async def _search(topic: str) -> list[dict]:
    """Multi-angle DDGS search, deduped by URL, briefly ranked by how many angles agree."""

    def _run() -> list[dict]:
        found: dict[str, dict] = {}
        with DDGS() as ddgs:
            for angle in ANGLES:
                q = f"{topic} {angle}".strip()
                try:
                    for r in ddgs.text(q, max_results=RESULTS_PER_ANGLE, region="ru-ru"):
                        url = r.get("href") or r.get("url") or ""
                        if not url:
                            continue
                        entry = found.setdefault(
                            url, {"url": url, "title": r.get("title", ""), "snippet": r.get("body", ""), "hits": 0}
                        )
                        entry["hits"] += 1
                except Exception:
                    log.warning("search angle failed: %r", q, exc_info=True)
        return sorted(found.values(), key=lambda e: -e["hits"])

    return await asyncio.to_thread(_run)


def _extract_paragraphs(html_text: str) -> list[str]:
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return []
    for bad in tree.xpath("//script|//style|//nav|//header|//footer|//aside|//form"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    paras = []
    for node in tree.xpath("//p"):
        text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
        if 200 <= len(text):
            paras.append(text[:PARA_MAX_CHARS])
        if len(paras) >= MAX_PARAS_PER_PAGE:
            break
    return paras


async def _fetch_pages(sources: list[dict]) -> None:
    """Fill each source's 'paras' in place. Failures leave the snippet as the only material."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) T800-research/1.0"}
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers) as client:

        async def one(src: dict) -> None:
            src["paras"] = []
            try:
                r = await client.get(src["url"])
                if r.status_code == 200 and "html" in (r.headers.get("content-type") or ""):
                    src["paras"] = await asyncio.to_thread(_extract_paragraphs, r.text)
            except Exception:
                log.info("fetch failed: %s", src["url"])

        await asyncio.gather(*(one(s) for s in sources))


async def _select_paragraphs(topic: str, sources: list[dict]) -> list[dict]:
    """Lexical prefilter -> cross-encoder rerank -> top paragraphs, each tagged with its source
    index (1-based) so [n] citations in the report map to the numbered source list."""
    topic_terms = _tokenize(topic)
    candidates: list[dict] = []
    for i, src in enumerate(sources, 1):
        material = src.get("paras") or ([src["snippet"]] if src.get("snippet") else [])
        for p in material:
            overlap = len(topic_terms & _tokenize(p))
            candidates.append({"text": p, "src": i, "overlap": overlap})
    if not candidates:
        return []
    candidates.sort(key=lambda c: -c["overlap"])
    candidates = candidates[:RERANK_CANDIDATES]
    try:
        ranked = await reranker.rerank(topic, candidates)
    except Exception:
        log.warning("reranker unavailable for research, using lexical order", exc_info=True)
        ranked = candidates
    # Cap per-source dominance: no more than half the context from one page.
    picked: list[dict] = []
    per_src: dict[int, int] = {}
    for c in ranked:
        if per_src.get(c["src"], 0) >= CONTEXT_PARAS // 2:
            continue
        picked.append(c)
        per_src[c["src"]] = per_src.get(c["src"], 0) + 1
        if len(picked) >= CONTEXT_PARAS:
            break
    return picked


def _build_messages(topic: str, picked: list[dict]) -> list[dict]:
    blocks = "\n\n".join(f"[{c['src']}] {c['text']}" for c in picked)
    user_msg = (
        f"Материалы из источников (номер в скобках = номер источника):\n\n{blocks}\n\n"
        f"Тема исследования / предмет спора: {topic}\n\n"
        "Проведи разбор по правилам из системной инструкции."
    )
    return [
        {"role": "system", "content": RESEARCH_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def _sources_footer(sources: list[dict], picked: list[dict]) -> str:
    used = {c["src"] for c in picked}
    lines = []
    for i, src in enumerate(sources, 1):
        if i in used:
            domain = urlparse(src["url"]).netloc
            lines.append(f"[{i}] {domain} — {src['url']}")
    return "Источники:\n" + "\n".join(lines) if lines else ""


async def research_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    if config.TARGET_CHAT_IDS and update.effective_chat.id not in config.TARGET_CHAT_IDS:
        return

    topic = " ".join(context.args).strip()
    if not topic and update.message.reply_to_message:
        topic = (update.message.reply_to_message.text or update.message.reply_to_message.caption or "").strip()
    if not topic:
        await update.message.reply_text(
            "Использование: /research <тема или суть спора>, либо ответь командой на сообщение с темой."
        )
        return
    topic = topic[:400]

    reader_llm.mark_user_activity()
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    status = await update.message.reply_text(f"Запускаю протокол исследования: «{topic}». Сканирую сеть...")
    editor = StreamEditor(status)

    async def edit(text: str) -> None:
        await editor.finalize(text)

    t0 = time.monotonic()
    results = await _search(topic)
    if not results:
        await edit("Сеть молчит: источников по теме не найдено. Переформулируй запрос.")
        return
    sources = results[:MAX_SOURCES]
    await edit(f"Найдено источников: {len(results)}. Извлекаю данные из {len(sources)}...")

    await _fetch_pages(sources)
    picked = await _select_paragraphs(topic, sources)
    if not picked:
        await edit("Источники нашлись, но пригодных данных в них нет. Переформулируй запрос.")
        return
    log.info(
        "research %r: %d sources, %d paras picked in %.1fs",
        topic[:50], len(sources), len(picked), time.monotonic() - t0,
    )

    messages = _build_messages(topic, picked)
    in_tok = sum(len(m["content"]) for m in messages) / rconfig.CHARS_PER_TOKEN
    eta_s = int(reader_llm.estimate_call_seconds(in_tok, REPORT_MAX_TOKENS))
    await edit(f"Данные собраны ({len(picked)} фрагментов из {len({c['src'] for c in picked})} источников). "
               f"Анализирую (~{max(1, eta_s // 60)} мин)...")

    footer = _sources_footer(sources, picked)
    acc = ""
    try:
        async with reader_llm.llm_lock:
            reader_llm.mark_user_activity()
            async for delta in reader_llm.generate_stream(
                messages, REPORT_MAX_TOKENS, timeout=eta_s * 2.5 + 120
            ):
                acc += delta
                await editor.maybe_edit(acc + " ▌")  # never raises; flood limits just pause edits
    except Exception:
        # Only GENERATION failures land here now (edit errors are contained in StreamEditor) —
        # a partial stream can't be trusted, so regenerate through the standard reply path.
        log.warning("local research synthesis failed, falling back to get_reply", exc_info=True)
        acc = ""
    finally:
        reader_llm.mark_user_activity()

    if not acc.strip():
        # Cloud (or local-retry) fallback. Generous budget: get_reply itself caps the local
        # attempt at LOCAL_TIMEOUT_SECONDS, the rest is for the cloud call on a slow free tier.
        try:
            acc = await asyncio.wait_for(get_reply(messages), timeout=240)
        except Exception:
            log.exception("research synthesis failed on all paths")
            await edit("Анализ сорвался: вычислительные мощности недоступны. Попробуй позже.")
            return

    await edit(f"{acc.strip()}\n\n{footer}" if footer else acc.strip())
