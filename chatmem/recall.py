"""Retrieval side of the chat memory.

Two consumers with different budgets (the hardware lesson: every retrieved token is prefill):

- INLINE: when a live message smells like a reference to the past («помнишь», «кто предлагал»...),
  the persona's prompt gets a compact memory block — top-3 windows by hybrid RRF, NO cross-encoder
  (speed; fusion is good enough for «напомнить контекст»), injected at the TAIL of the prompt so
  the prefix cache survives. Costs ~20-30s of prefill only on triggered messages.

- /recall: an explicit question to the archive. Full pipeline (rerank included), streamed answer
  with dates/authors and t.me/c/ links to the original messages — supergroup message ids are
  linkable for members, so citations can jump to the source.
"""

import asyncio
import logging
import re
from datetime import datetime

from reader import embedder, rconfig, reranker

from . import store

log = logging.getLogger("t800.chatmem")

TRIGGER_RE = re.compile(
    r"(?i)\b(помнишь|помните|припомни|вспомни|кто (говорил|писал|предлагал|кидал|скидывал|обещал)|"
    r"в прошлый раз|мы (же )?(обсуждали|говорили|решили|спорили)|о ч[ёе]м (мы|тут|вы)|"
    r"что (мы )?решили|когда (мы|вы|кто-то))\b"
)

INLINE_LIMIT = 3
INLINE_MAX_CHARS = 1500  # ~450-500 tokens of extra prefill, worst case ~30s

RECALL_SYSTEM = (
    "Ты — Т-800, архивариус этого чата. Тебе дают фрагменты реальной переписки и вопрос. "
    "Отвечай только по фрагментам: кто, что и когда (дата) говорил. Если в фрагментах ответа "
    "нет — скажи прямо «в архиве этого нет». Не выдумывай. Формат: короткий ответ по существу, "
    "с датами и именами, обычным текстом без markdown. В конце НЕ пиши список ссылок — он будет добавлен автоматически."
)


def is_memory_trigger(text: str) -> bool:
    return bool(TRIGGER_RE.search(text))


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y")


async def inline_memory_block(chat_id: int, query: str) -> str | None:
    """Compact context block for the persona prompt, or None when nothing relevant."""
    q_emb = await embedder.embed_query(query)
    hits = await store.search(q_emb, chat_id, limit=INLINE_LIMIT * 2)
    if not hits:
        return None
    parts: list[str] = []
    used = 0
    for h in hits:
        t = h["text"]
        if used + len(t) > INLINE_MAX_CHARS:
            t = t[: max(0, INLINE_MAX_CHARS - used)]
        if not t:
            break
        parts.append(f"— [{_fmt_ts(h['ts_start'])}] {t}")
        used += len(t)
        if len(parts) >= INLINE_LIMIT or used >= INLINE_MAX_CHARS:
            break
    if not parts:
        return None
    return (
        "Фрагменты из архива этого чата, возможно относящиеся к последнему сообщению "
        "(используй, только если реально в тему):\n" + "\n".join(parts)
    )


async def recall_answer_context(chat_id: int, question: str) -> tuple[list[dict], list[dict]] | None:
    """-> (messages for generation, used hits) or None when the archive has nothing."""
    q_emb = await embedder.embed_query(question)
    hits = await store.search(q_emb, chat_id, limit=rconfig.RETRIEVE_LIMIT)
    if not hits:
        return None
    if rconfig.RERANK_ENABLED:
        try:
            hits = await reranker.rerank(question, hits)
            hits = [h for h in hits[:6] if h.get("rerank_score", 0) >= 0.05]
        except Exception:
            log.warning("reranker unavailable for /recall, using fusion order", exc_info=True)
            hits = hits[:5]
    else:
        hits = hits[:5]
    if not hits:
        return None

    budget = int(rconfig.CONTEXT_TOKENS * rconfig.CHARS_PER_TOKEN)
    blocks: list[str] = []
    used_hits: list[dict] = []
    used = 0
    for i, h in enumerate(hits, 1):
        t = h["text"][:1400]
        if used + len(t) > budget and blocks:
            break
        blocks.append(f"[{i}] ({_fmt_ts(h['ts_start'])}) {t}")
        used_hits.append(h)
        used += len(t)

    user_msg = (
        "Фрагменты архива чата:\n\n" + "\n\n".join(blocks) +
        f"\n\nВопрос: {question}\n\nОтветь по фрагментам, указывая даты и кто что говорил."
    )
    messages = [
        {"role": "system", "content": RECALL_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    return messages, used_hits


# --- dossier corpus for /profile -----------------------------------------------------------------

# Probe queries for a DIVERSE portrait: one semantic query would return near-duplicates; probing
# different facets (with the author filter) collects a spread of characteristic moments.
PROFILE_PROBES = [
    "мнение, спор, точка зрения, несогласие",
    "шутка, юмор, мем, стёб",
    "вкусы: музыка, еда, фильмы, игры",
    "планы, обещания, договорённости",
    "работа, деньги, покупки",
    "личная жизнь, отношения, быт",
]
PROFILE_PER_PROBE = 2
PROFILE_MAX_BLOCKS = 8
PROFILE_MAX_CHARS = 5200  # ~1600 tok: /profile also carries persona + chat history in the slot


async def known_authors(chat_id: int) -> set[str]:
    """Distinct author names stored for this chat (display names, lowercased)."""
    from qdrant_client import models as qm

    from reader.store import _get_client, _qlock

    def _scan() -> set[str]:
        names: set[str] = set()
        offset = None
        while True:
            points, offset = _get_client().scroll(
                store.COLLECTION,
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))
                ]),
                with_payload=["authors"], with_vectors=False, limit=512, offset=offset,
            )
            for p in points:
                names.update((p.payload or {}).get("authors", []))
            if offset is None:
                return names

    async with _qlock:
        return await asyncio.to_thread(_scan)


def resolve_author(fragment: str, authors: set[str]) -> str | None:
    """'илья' / '@ilya_m' -> 'илья миронов' (export stores display names, the bot knows
    username/first_name — bridge them fuzzily)."""
    frag = fragment.lower().lstrip("@").strip()
    if not frag:
        return None
    if frag in authors:
        return frag
    candidates = [a for a in authors if frag in a or a in frag]
    if not candidates:
        parts = [a for a in authors if any(w.startswith(frag) or frag.startswith(w) for w in a.split())]
        candidates = parts
    return min(candidates, key=len) if candidates else None


async def profile_corpus(chat_id: int, name_fragment: str,
                         user_id: int | None = None) -> tuple[str, list[str]] | None:
    """-> (resolved display name, diverse memory blocks) or None if the author isn't in memory.
    Prefers filtering by exact numeric user_id (present in JSON-export windows); falls back to
    the fuzzy display-name bridge for HTML-export data."""
    author = None
    use_id = None
    if user_id is not None:
        q_probe = await embedder.embed_query(PROFILE_PROBES[0])
        if await store.search(q_probe, chat_id, limit=1, author_id=user_id):
            use_id = user_id
            # The DISPLAY name is what window texts actually contain («Алексей: ...») — telling
            # the LLM the @username instead makes it honestly "find no quotes" (seen live).
            from . import names

            author = names.get_name(chat_id, user_id) or name_fragment.lower() or str(user_id)
    if use_id is None:
        authors = await known_authors(chat_id)
        author = resolve_author(name_fragment, authors)
        if author is None:
            return None

    seen: set[tuple] = set()
    blocks: list[str] = []
    used = 0
    for probe in PROFILE_PROBES:
        if len(blocks) >= PROFILE_MAX_BLOCKS or used >= PROFILE_MAX_CHARS:
            break
        q_emb = await embedder.embed_query(probe)
        hits = await store.search(q_emb, chat_id, limit=PROFILE_PER_PROBE * 2,
                                  author=author, author_id=use_id)
        for h in hits[:PROFILE_PER_PROBE]:
            key = (h.get("msg_id_first"), h.get("ts_start"))
            if key in seen:
                continue
            seen.add(key)
            t = h["text"][:900]
            if used + len(t) > PROFILE_MAX_CHARS and blocks:
                break
            blocks.append(f"[{_fmt_ts(h['ts_start'])}] {t}")
            used += len(t)
    if not blocks:
        return None
    return author, blocks


def source_links(chat_id: int, hits: list[dict]) -> str:
    """t.me/c/<internal>/<msg_id> links work for supergroup members — citations that jump home."""
    cid = str(chat_id)
    if not cid.startswith("-100"):
        return ""
    internal = cid[4:]
    lines = [
        f"[{i}] {_fmt_ts(h['ts_start'])} — https://t.me/c/{internal}/{h['msg_id_first']}"
        for i, h in enumerate(hits, 1)
    ]
    return "Источники:\n" + "\n".join(lines)
