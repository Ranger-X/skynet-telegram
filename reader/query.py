"""Interactive query pipeline (brief §5): route -> hybrid retrieve (spoiler-filtered) -> rerank ->
small-to-big context -> persona answer with a citation. Everything before generation must stay in
seconds; generation itself is streamed by the caller.
"""

import logging
import re
from dataclasses import dataclass, field

import i18n

from . import db, embedder, rconfig, reranker, store

log = logging.getLogger("t800.reader")

# Router (brief §5 step 1): heuristic first, no LLM call — on this box a classifier call would cost
# more than the retrieval it routes. Global = the answer is smeared across the book, so leaf top-k
# dies and chapter summaries are the right corpus. Bilingual: the same markers in English and
# Russian, incl. the explicit «по всей книге» / "whole book" route override.
_GLOBAL_MARKERS = re.compile(
    r"(?i)\b("
    # Russian markers:
    r"как (меня|развива|измени)|в целом|по ходу (книги|сюжета)|сюжет|перескаж|"
    r"краткое содержание|о ч[её]м|главн(ая|ую) (мысль|тема|идея)|эволюци|динамик|"
    r"в итоге|к концу книги|на протяжении|через всю|вс[её] произведение|"
    r"по всей (книге|манге)|во всей (книге|манге)|по (книге|манге) в целом|по всем главам|"
    # English markers (mirror the Russian set):
    r"plot|summary|summari[sz]e|retell|recap|overall|in general|"
    r"main (idea|theme|message|point|thesis)|whole (book|story|manga)|"
    r"entire (book|story|manga)|throughout|over the course|by the end|"
    r"what('s| is| are| was) .{0,20}about|how did .{1,30} (change|evolve|develop)|"
    r"how does it all end|character (arc|development)|across (the|all) chapters"
    r")"
)

# Answer-language system prompts: instruct the model to answer in the chat language, keep the
# "plain text, no markdown" clause, the citation format, and the honest-refusal clause (all
# translated). Selected by lang in prepare().
READER_SYSTEM = {
    "en": (
        "You are the T-800, a cybernetic organism working as an assistant for books you've read. "
        "You answer concisely, with military precision, slightly condescending toward humans but "
        "strictly to the point and without theatrics. "
        "IRON RULES: answer ONLY from the book fragments provided. If the answer isn't in them, "
        "say so plainly: \"There is no such data in the part of the book I've swallowed,\" and "
        "don't make anything up. Don't recount events that aren't in the fragments, even if you "
        "know the book from other sources. "
        "End your answer with the source in parentheses: (chapter so-and-so). Write in plain text, "
        "no markdown formatting — no asterisks or headings. Answer in English."
    ),
    "ru": (
        "Ты — Т-800, кибернетический организм, работаешь ассистентом по прочитанным книгам. Отвечаешь "
        "кратко, по-военному чётко, слегка надменно к людям, но по делу и без кривляний. "
        "ЖЕЛЕЗНЫЕ ПРАВИЛА: отвечай ТОЛЬКО по приведённым фрагментам книги. Если ответа в них нет — "
        "скажи прямо: «В проглоченной части книги таких данных нет», не выдумывай. Не пересказывай "
        "события, которых нет во фрагментах, даже если знаешь книгу из других источников. "
        "В конце ответа укажи источник в скобках: (глава такая-то). Пиши обычным текстом, без "
        "markdown-разметки — никаких звёздочек и заголовков. Отвечай по-русски."
    ),
}

READER_SYSTEM_MANGA = {
    "en": (
        "You are the T-800, a cybernetic organism, an assistant for manga and comics. You're given "
        "text digests of pages (lines from speech bubbles + scene descriptions). You answer "
        "concisely, with military precision. "
        "IRON RULES: only from the pages provided; if there's no answer, say \"There is no such "
        "data in the part I've read,\" don't make anything up, and don't use knowledge of the "
        "title from other sources. "
        "End with the source: (page so-and-so). Write in plain text, no markdown formatting. "
        "Answer in English."
    ),
    "ru": (
        "Ты — Т-800, кибернетический организм, ассистент по манге и комиксам. Тебе дают текстовые "
        "выжимки страниц (реплики из баблов + описание сцен). Отвечаешь кратко и по-военному чётко. "
        "ЖЕЛЕЗНЫЕ ПРАВИЛА: только по приведённым страницам; нет ответа — скажи «В прочитанной части "
        "таких данных нет», не выдумывай и не используй знание тайтла из других источников. "
        "В конце укажи источник: (стр. такая-то). Пиши обычным текстом, без markdown-разметки. "
        "Отвечай по-русски."
    ),
}

# Model-facing context scaffolding, selected by lang. .format() args (title/context/question) are
# never re-scanned for placeholders, so book text with stray braces is safe here.
_SUMMARY_BLOCK_LABEL = {"en": "Summary", "ru": "Сводка"}
_USER_MSG = {
    "en": {
        "book": "Fragments from the book \"{title}\":\n\n{context}\n\n"
                "Reader's question: {question}\n\nAnswer from the fragments above.",
        "manga": "Fragments from the manga/comic \"{title}\":\n\n{context}\n\n"
                 "Reader's question: {question}\n\nAnswer from the fragments above.",
    },
    "ru": {
        "book": "Фрагменты книги «{title}»:\n\n{context}\n\n"
                "Вопрос читателя: {question}\n\nОтветь по фрагментам выше.",
        "manga": "Фрагменты манги/комикса «{title}»:\n\n{context}\n\n"
                 "Вопрос читателя: {question}\n\nОтветь по фрагментам выше.",
    },
}


@dataclass
class Prepared:
    """Everything the caller needs to run + label the generation."""
    messages: list[dict]
    route: str                      # local | global
    sources: list[str] = field(default_factory=list)  # chapter titles used
    best_score: float | None = None


@dataclass
class Refusal:
    reason: str                     # not_ingested | nothing_unlocked | low_confidence | no_summaries
    best_score: float | None = None


def _is_global(question: str) -> bool:
    return bool(_GLOBAL_MARKERS.search(question))


def _budget_chars() -> int:
    return int(rconfig.CONTEXT_TOKENS * rconfig.CHARS_PER_TOKEN)


async def prepare(doc_id: str, user_id: int, question: str, lang: str = "en") -> Prepared | Refusal:
    doc = db.get_doc(doc_id)
    if doc is None or doc["status"] != "ready":
        return Refusal("not_ingested")

    user_pos = db.get_position(user_id, doc_id)
    if user_pos <= 0:
        return Refusal("nothing_unlocked")

    # Routing, two tiers: regex markers (high precision, incl. the explicit "whole book" /
    # «по всей книге» override), then the semantic classifier on the SAME embedding retrieval
    # uses — free.
    q_emb = await embedder.embed_query(question)
    if _is_global(question):
        route = "global"
    else:
        from . import router as _router

        route, _ = await _router.classify(q_emb["dense"], lang)

    if route == "global":
        # Global questions run over chapter summaries; their offset = chapter end (max of children),
        # so the SAME spoiler condition applies (brief §3/§6).
        hits = await store.search(q_emb, doc_id, user_pos, level="chapter_summary",
                                  limit=rconfig.RETRIEVE_LIMIT)
        if not hits:
            # Low tier has no summaries — degrade to leaves honestly rather than refuse outright.
            route = "global-degraded"
            hits = await store.search(q_emb, doc_id, user_pos, level="leaf", limit=rconfig.RETRIEVE_LIMIT)
    else:
        hits = await store.search(q_emb, doc_id, user_pos, level="leaf", limit=rconfig.RETRIEVE_LIMIT)

    if not hits:
        return Refusal("nothing_unlocked")

    if rconfig.RERANK_ENABLED:
        try:
            hits = await reranker.rerank(question, hits)
        except Exception:
            # Reranker unavailable (usually the RAM guard) — degrade to fusion order. The honest-
            # refusal threshold needs rerank scores, so it's skipped too: worse than normal, but
            # infinitely better than a dead bot or a dead /ask.
            log.warning("reranker unavailable, falling back to fusion order", exc_info=True)
            best = None
            hits = hits[: rconfig.RERANK_KEEP]
        else:
            best = hits[0]["rerank_score"]
            # The honest-refusal threshold applies to PRECISE (local) questions only. A global
            # "describe the plot" scores low against any single summary by nature (command-vs-
            # document, not question-vs-answer: 0.199 on a perfectly valid live query) — refusing
            # there is wrong, the summaries ARE the answer corpus.
            if route == "local" and best < rconfig.MIN_SCORE:
                return Refusal("low_confidence", best_score=best)
            hits = hits[: rconfig.RERANK_KEEP]
    else:
        best = None

    # --- context assembly ------------------------------------------------------------------
    budget = _budget_chars()
    blocks: list[str] = []
    sources: list[str] = []

    if doc["fmt"] == "manga" and route != "global":
        # Manga small-to-big: the "parent" of a page is its neighbors. Pull page±1 for each top
        # hit (respecting the bookmark), dedupe, keep page order inside each block.
        from . import store as _store

        seen_pages: set[int] = set()
        used = 0
        for h in hits:
            page = int(h["offset"])
            if page in seen_pages:
                continue
            first = max(1, page - 1)
            last = min(int(user_pos), page + 1)
            texts = await _store.page_texts(doc_id, first, last)
            block = "\n\n".join(texts)
            if not block or used + len(block) > budget and blocks:
                if blocks:
                    break
                block = block[:budget]
            seen_pages.update(range(first, last + 1))
            blocks.append(block)
            sources.append(f"page {page}")  # log-only metadata, not shown to the user
            used += len(block)
            if len(blocks) >= rconfig.MAX_PARENTS:
                break
        route = "manga-local"
    elif route == "global":
        # Summaries ARE the context — no parents to fetch. Select by relevance (budget-bound),
        # then ORDER CHRONOLOGICALLY: a plot retelling built from relevance-ordered chapters
        # scrambles the timeline.
        picked: list[dict] = []
        used = 0
        for h in hits:
            if used + len(h["text"]) > budget:
                break
            picked.append(h)
            used += len(h["text"])
        picked.sort(key=lambda h: h.get("chapter_idx", 0))
        summary_label = _SUMMARY_BLOCK_LABEL.get(lang, _SUMMARY_BLOCK_LABEL["en"])
        for h in picked:
            blocks.append(f"[{summary_label}: {h['chapter_title']}]\n{h['text']}")
            sources.append(h["chapter_title"])
    else:
        # Small-to-big with parent dedup (brief §5 step 5): several hits often share a chapter.
        seen: set[str] = set()
        parent_ids: list[str] = []
        for h in hits:
            pid = h["parent_id"]
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)
            if len(parent_ids) >= rconfig.MAX_PARENTS:
                break
        parents = {p["parent_id"]: p for p in db.get_parents(parent_ids)}
        chapters = {c["chapter_idx"]: c for c in db.get_chapters(doc_id)}
        used = 0
        for pid in parent_ids:  # keep rerank order
            p = parents.get(pid)
            if p is None:
                continue
            # A parent can stick past the user's position (its window may span the bookmark);
            # trim the tail so the model never sees locked text.
            text = p["text"]
            if p["end_offset"] > user_pos and p["start_offset"] < user_pos:
                text = text[: max(0, user_pos - p["start_offset"])]
            elif p["start_offset"] >= user_pos:
                continue
            if not text.strip():
                continue
            if used + len(text) > budget and blocks:
                break
            title = chapters[p["chapter_idx"]]["title"] if p["chapter_idx"] in chapters else "?"
            blocks.append(f"[{title}]\n{text[:budget]}")
            sources.append(title)
            used += len(text)

    if not blocks:
        return Refusal("nothing_unlocked")

    context = "\n\n---\n\n".join(blocks)
    is_manga = doc["fmt"] == "manga"
    user_msg = i18n.L(
        lang, _USER_MSG, "manga" if is_manga else "book",
        title=doc["title"], context=context, question=question,
    )
    system = READER_SYSTEM_MANGA if is_manga else READER_SYSTEM
    messages = [
        {"role": "system", "content": system.get(lang, system["en"])},
        {"role": "user", "content": user_msg},
    ]
    log.info(
        "prepared %s query: %d blocks, %d chars ctx, best=%.3f, sources=%s",
        route, len(blocks), len(context), best or -1, sources,
    )
    return Prepared(messages=messages, route=route, sources=sources, best_score=best)


def estimate_answer_seconds(prepared: Prepared, max_tokens: int | None = None) -> int:
    from .llm import estimate_call_seconds

    in_tok = sum(len(m["content"]) for m in prepared.messages) / rconfig.CHARS_PER_TOKEN
    return int(estimate_call_seconds(in_tok, max_tokens or rconfig.ANSWER_MAX_TOKENS))
