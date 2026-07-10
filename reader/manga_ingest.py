"""Manga ingest: page images -> VLM text (bubbles + scene) -> the same spoiler-filtered RAG the
books live in. Resumable per page (deterministic chunk ids, described pages skipped). Tiers:
low = describe + embed every page; medium = + chapter summaries generated from the page TEXTS
(cheap — by then the expensive vision part is already done).
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import i18n

from . import db, embedder, grinder, llm, manga, rconfig, store
from .chunking import Leaf
from .ingest import SUMMARY_PROMPT, eta_text
from .manga import MangaDoc

log = logging.getLogger("t800.reader")

# Measured seconds per page describe; prior for the 12B path, self-corrects after each run.
_sec_per_page = 45.0

# User-facing manga-ingest progress lines. Bilingual via i18n.L.
STR = {
    "en": {
        "reading_start": "Reading \"{title}\": {n} pages{skipped}, will take ~{eta}...",
        "already_read": " ({n} already read)",
        "reading_progress": "Reading: page {done}/{total} ({failed} failed), {eta} left...",
        "estimating_rate": "estimating the pace",
        "summaries": "Building chapter summaries: {n}/{total}...",
        "done": (
            "\"{title}\" has been read ({described} pages absorbed, {failed} failures). "
            "Set a bookmark: /pos page 30 or /pos chapter 2 — then ask via /ask. No spoilers."
        ),
        "read_failed": "Failure while reading the comic: {exc}",
    },
    "ru": {
        "reading_start": "Читаю «{title}»: {n} страниц{skipped}, займёт ~{eta}...",
        "already_read": " (уже прочитано {n})",
        "reading_progress": "Читаю: стр. {done}/{total} (ошибок {failed}), осталось {eta}...",
        "estimating_rate": "оцениваю темп",
        "summaries": "Строю сводки глав: {n}/{total}...",
        "done": (
            "«{title}» прочитана ({described} страниц усвоено, {failed} сбоев). "
            "Поставь закладку: /pos страница 30 или /pos глава 2 — и спрашивай через /ask. Спойлеров не будет."
        ),
        "read_failed": "Сбой чтения комикса: {exc}",
    },
}


def estimate_manga_seconds(doc: MangaDoc, tier: str, pages_left: int | None = None) -> int:
    n = doc.n_pages if pages_left is None else pages_left
    total = n * _sec_per_page
    if tier == "medium":
        # summaries: input = the chapter's page texts (~150 tok/page), output ~250 tok
        for ch in doc.chapters:
            in_tok = (ch.last_page - ch.first_page + 1) * 150
            total += llm.estimate_call_seconds(min(in_tok, 2800), rconfig.SUMMARY_MAX_TOKENS)
    return int(total)


def _page_chunk_id(doc_id: str, page: int) -> str:
    return f"{doc_id[:16]}:mp:{page}"


async def run_manga_ingest(
    doc_id: str,
    mdoc: MangaDoc,
    tier: str,
    job_id: int,
    progress: Callable[[str], Awaitable[None]],
    lang: str = "en",
) -> None:
    global _sec_per_page
    last_edit = 0.0

    async def tick(text: str, force: bool = False) -> None:
        nonlocal last_edit
        db.update_job(job_id, progress=text)
        if force or time.monotonic() - last_edit > 15:
            last_edit = time.monotonic()
            try:
                await progress(text)
            except Exception:
                log.warning("manga progress edit failed", exc_info=True)

    try:
        db.set_doc_status(doc_id, "ingesting")
        db.save_chapters(
            doc_id,
            [
                {"chapter_idx": i, "title": c.title, "start_offset": c.first_page, "end_offset": c.last_page}
                for i, c in enumerate(mdoc.chapters)
            ],
        )

        chapters = db.get_chapters(doc_id)

        def chapter_of(page: int):
            for c in chapters:
                if c["start_offset"] <= page <= c["end_offset"]:
                    return c
            return chapters[-1]

        done_ids = await store.existing_chunk_ids(doc_id)
        todo = [p for p in range(1, mdoc.n_pages + 1) if _page_chunk_id(doc_id, p) not in done_ids]
        n_skipped = mdoc.n_pages - len(todo)
        skipped = i18n.L(lang, STR, "already_read", n=n_skipped) if n_skipped else ""
        await tick(
            i18n.L(lang, STR, "reading_start", title=mdoc.title, n=len(todo),
                   skipped=skipped, eta=eta_text(int(len(todo) * _sec_per_page), lang)),
            force=True,
        )

        t0 = time.monotonic()
        described = 0
        failed = 0
        for page in todo:
            try:
                jpeg = await asyncio.to_thread(manga.page_jpeg, mdoc, page)
                text = await grinder.describe_image(jpeg, grinder.MANGA_PAGE_PROMPT)
                if not text.strip():
                    raise RuntimeError("empty describe")
                ch = chapter_of(page)
                leaf = Leaf(
                    chunk_id=_page_chunk_id(doc_id, page), parent_id="", doc_id=doc_id,
                    # Index-internal page marker (embedded + fed to the model as context, never
                    # shown verbatim) — English single version per the i18n contract.
                    text=f"[page {page}]\n{text}", offset=page, start_offset=page,
                    chapter_idx=ch["chapter_idx"], chapter_title=ch["title"],
                )
                embs = await embedder.embed_texts([leaf.text])
                await store.upsert_leaves([leaf], embs)
                described += 1
            except Exception:
                failed += 1
                log.exception("manga page %d failed", page)

            done_total = n_skipped + described
            rate = described / max(1.0, time.monotonic() - t0)
            eta = (
                eta_text(int((len(todo) - described - failed) / max(rate, 0.001)), lang)
                if described else i18n.L(lang, STR, "estimating_rate")
            )
            await tick(i18n.L(lang, STR, "reading_progress", done=done_total,
                              total=mdoc.n_pages, failed=failed, eta=eta))

        if described > 3:
            _sec_per_page = max(3.0, (time.monotonic() - t0) / described)
            log.info("manga describe rate: %.1f s/page", _sec_per_page)

        if tier == "medium":
            db.update_job(job_id, stage="summarize")
            todo_ch = [c for c in db.get_chapters(doc_id) if not c["summary"]]
            for n, ch in enumerate(todo_ch, 1):
                page_texts = await store.page_texts(doc_id, ch["start_offset"], ch["end_offset"])
                if not page_texts:
                    continue
                body = "\n\n".join(page_texts)[: rconfig.SUMMARY_INPUT_CHARS]
                prompt = SUMMARY_PROMPT[lang].format(title=ch["title"], book=mdoc.title, text=body)
                await tick(i18n.L(lang, STR, "summaries", n=n, total=len(todo_ch)), force=True)
                await llm.wait_for_quiet()
                async with llm.llm_lock:
                    try:
                        summary = await llm.generate(
                            [{"role": "user", "content": prompt}],
                            max_tokens=rconfig.SUMMARY_MAX_TOKENS, timeout=600,
                        )
                    except Exception:
                        log.exception("manga chapter summary failed, continuing")
                        continue
                db.set_chapter_summary(doc_id, ch["chapter_idx"], summary)
                s_leaf = Leaf(
                    chunk_id=f"{doc_id[:16]}:ms:{ch['chapter_idx']}", parent_id="", doc_id=doc_id,
                    text=summary, offset=ch["end_offset"], start_offset=ch["start_offset"],
                    chapter_idx=ch["chapter_idx"], chapter_title=ch["title"], level="chapter_summary",
                )
                embs = await embedder.embed_texts([summary])
                await store.upsert_leaves([s_leaf], embs)
            db.set_doc_tier(doc_id, "medium")

        db.set_doc_status(doc_id, "ready")
        db.update_job(job_id, stage="done", status="done")
        await tick(
            i18n.L(lang, STR, "done", title=mdoc.title, described=described, failed=failed),
            force=True,
        )
        log.info("manga ingest done: %s, %d described, %d failed", doc_id[:12], described, failed)
    except Exception as exc:
        log.exception("manga ingest failed for %s", doc_id[:12])
        db.set_doc_status(doc_id, "error", error=str(exc))
        db.update_job(job_id, stage="error", status="error", error=str(exc))
        try:
            await progress(i18n.L(lang, STR, "read_failed", exc=exc))
        except Exception:
            pass
