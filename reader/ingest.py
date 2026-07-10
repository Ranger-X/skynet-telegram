"""Ingest pipeline (brief §4): parse -> chunk -> embed -> (medium tier) chapter summaries.
Runs as a plain asyncio task inside the bot process (no celery/redis — Windows box, one user;
brief §1a) with progress streamed to the chat and state in SQLite, so a crash is restartable.

Tiers are additive (brief §7): low = pure retrieval (0 LLM calls), medium = + chapter summaries.
Upgrading re-runs only the missing stage; embeddings are never recomputed (same point ids ->
idempotent upserts).
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from . import db, embedder, llm, rconfig, store
from .chunking import Leaf, chunk_document
from .parsing import ParsedDoc, parse_document

log = logging.getLogger("t800.reader")

# Measured-on-first-run embedding throughput (chunks/s) used for the low-tier ETA; prior = the
# real ingest of the test book after batch/max_length tuning (0.63 ch/s ≈ 15 min per 400-page
# book); updated in-process after every ingest.
_embed_chunks_per_s = 0.6

EMBED_BATCH = 32         # chunks per embed_texts call; progress ticks between calls
PROGRESS_EVERY_S = 12    # min seconds between progress edits (Telegram flood limits)

SUMMARY_PROMPT = (
    "Ниже текст главы «{title}» из книги «{book}». Составь СЖАТУЮ сводку главы: ключевые события, "
    "кто из персонажей что сделал, важные факты и изменения ситуации. Пиши связным текстом, "
    "по-русски, без вступлений и без своих оценок. Не больше 8-10 предложений."
    "\n\n{text}"
)


def estimate_tier_seconds(parsed: ParsedDoc, n_leaves: int, tier: str) -> int:
    """ETA under the MEASURED speeds (brief §7 formula, amended: prefill is a first-class term)."""
    total = n_leaves / _embed_chunks_per_s
    if tier == "medium":
        for ch in parsed.chapters:
            in_tok = min(len(ch.text), rconfig.SUMMARY_INPUT_CHARS) / rconfig.CHARS_PER_TOKEN
            total += llm.estimate_call_seconds(in_tok + 150, rconfig.SUMMARY_MAX_TOKENS)
    return int(total)


def eta_text(seconds: int) -> str:
    if seconds < 90:
        return f"~{seconds} с"
    if seconds < 5400:
        return f"~{max(1, round(seconds / 60))} мин"
    return f"~{seconds / 3600:.1f} ч"


async def run_ingest(
    doc_id: str,
    parsed: ParsedDoc,
    tier: str,
    job_id: int,
    progress: Callable[[str], Awaitable[None]],
) -> None:
    """Full (or incremental) ingest of an already-parsed document. `progress` gets human-readable
    status lines, already throttled here."""
    global _embed_chunks_per_s
    last_edit = 0.0

    async def tick(text: str, force: bool = False) -> None:
        nonlocal last_edit
        db.update_job(job_id, progress=text)
        if force or time.monotonic() - last_edit > PROGRESS_EVERY_S:
            last_edit = time.monotonic()
            try:
                await progress(text)
            except Exception:  # a failed edit must never kill the ingest
                log.warning("progress edit failed", exc_info=True)

    try:
        doc = db.get_doc(doc_id)
        already_embedded = await store.count_points(doc_id) > 0 and doc and doc["status"] == "ready"
        db.set_doc_status(doc_id, "ingesting")

        leaves, parents = chunk_document(doc_id, parsed)

        if not already_embedded:
            db.update_job(job_id, stage="embed")
            db.save_chapters(
                doc_id,
                [
                    {"chapter_idx": i, "title": c.title, "start_offset": c.start_offset, "end_offset": c.end_offset}
                    for i, c in enumerate(parsed.chapters)
                ],
            )
            db.save_parents(
                doc_id,
                [
                    {"parent_id": p.parent_id, "chapter_idx": p.chapter_idx,
                     "start_offset": p.start_offset, "end_offset": p.end_offset, "text": p.text}
                    for p in parents
                ],
            )

            # Resume support: skip leaves already in the store from an interrupted run
            # (deterministic chunking -> stable ids -> safe to skip).
            done_ids = await store.existing_chunk_ids(doc_id)
            todo_leaves = [l for l in leaves if l.chunk_id not in done_ids]
            n_skipped = len(leaves) - len(todo_leaves)
            if n_skipped:
                log.info("ingest resume: %d/%d chunks already stored", n_skipped, len(leaves))

            await tick(f"Проглатываю: {len(leaves)} фрагментов, строю поисковый индекс...", force=True)
            t0 = time.monotonic()
            for i in range(0, len(todo_leaves), EMBED_BATCH):
                batch = todo_leaves[i : i + EMBED_BATCH]
                embs = await embedder.embed_texts([l.text for l in batch])
                await store.upsert_leaves(batch, embs)
                done = n_skipped + min(i + EMBED_BATCH, len(todo_leaves))
                await tick(f"Проглатываю: индекс {done}/{len(leaves)} фрагментов...")
            elapsed = time.monotonic() - t0
            if elapsed > 5 and todo_leaves:
                _embed_chunks_per_s = max(0.05, len(todo_leaves) / elapsed)
                log.info("embed throughput: %.2f chunks/s", _embed_chunks_per_s)

        if tier == "medium":
            db.update_job(job_id, stage="summarize")
            chapters = db.get_chapters(doc_id)
            todo = [c for c in chapters if not c["summary"]]
            n_all = len(chapters)

            def _summaries_eta(remaining: list) -> str:
                secs = sum(
                    llm.estimate_call_seconds(
                        min(len(parsed.chapters[c["chapter_idx"]].text), rconfig.SUMMARY_INPUT_CHARS)
                        / rconfig.CHARS_PER_TOKEN + 150,
                        rconfig.SUMMARY_MAX_TOKENS,
                    )
                    for c in remaining
                )
                return eta_text(int(secs))

            for n, ch in enumerate(todo, 1):
                ch_obj = parsed.chapters[ch["chapter_idx"]]
                text = ch_obj.text[: rconfig.SUMMARY_INPUT_CHARS]
                prompt = SUMMARY_PROMPT.format(title=ch_obj.title, book=parsed.title, text=text)
                # force=True: these ticks are minutes apart by nature (one summary ≈ 4-5 min on this
                # box), so the throttle only hurts — without force the FIRST one lands right after
                # the last embed tick and gets swallowed, leaving the message frozen on "индекс
                # N/N" for the whole first summary (seen live).
                await tick(
                    f"Строю сводки глав: {n}/{len(todo)} (всего глав {n_all}, осталось "
                    f"{_summaries_eta(todo[n - 1:])})...",
                    force=True,
                )
                # Contention protocol: never fire while the user is mid-conversation, and never
                # overlap a reader answer (brief §11).
                await llm.wait_for_quiet()
                async with llm.llm_lock:
                    try:
                        summary = await llm.generate(
                            [{"role": "user", "content": prompt}],
                            max_tokens=rconfig.SUMMARY_MAX_TOKENS,
                            timeout=llm.estimate_call_seconds(
                                len(prompt) / rconfig.CHARS_PER_TOKEN, rconfig.SUMMARY_MAX_TOKENS
                            ) * 2 + 60,
                        )
                    except Exception:
                        log.exception("summary failed for chapter %s, continuing", ch["chapter_idx"])
                        continue
                db.set_chapter_summary(doc_id, ch["chapter_idx"], summary)
                # Summary offset = END of its chapter = max of children (brief §3): the same spoiler
                # condition that guards leaves guards it.
                s_leaf = Leaf(
                    chunk_id=f"{doc_id[:16]}:s:{ch['chapter_idx']}",
                    parent_id="", doc_id=doc_id, text=summary,
                    offset=ch["end_offset"], start_offset=ch["start_offset"],
                    chapter_idx=ch["chapter_idx"], chapter_title=ch_obj.title,
                    level="chapter_summary",
                )
                embs = await embedder.embed_texts([summary])
                await store.upsert_leaves([s_leaf], embs)
            db.set_doc_tier(doc_id, "medium")

        db.set_doc_status(doc_id, "ready")
        db.update_job(job_id, stage="done", status="done")
        await tick(
            "Проглочено. Теперь поставь закладку — докуда дочитал: /pos глава 5 или /pos 40%. "
            "Дальше спрашивай через /ask — отвечаю только по прочитанной тобой части, спойлеров не будет.",
            force=True,
        )
    except Exception as exc:
        log.exception("ingest failed for %s", doc_id[:12])
        db.set_doc_status(doc_id, "error", error=str(exc))
        db.update_job(job_id, stage="error", status="error", error=str(exc))
        try:
            await progress(f"Сбой при поглощении файла: {exc}")
        except Exception:
            pass


def parse_bytes(data: bytes, file_name: str) -> ParsedDoc:
    """Thread-friendly wrapper (parsing a book is CPU work, seconds-scale)."""
    return parse_document(data, file_name)
