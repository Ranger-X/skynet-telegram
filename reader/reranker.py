"""bge-reranker-v2-m3 cross-encoder on CPU, lazy-loaded (brief §5 step 3 / §9: RAM is the reason to
load late — ~2.3 GB that only the chat phase needs). Loaded through sentence-transformers
CrossEncoder, NOT FlagEmbedding's FlagReranker — the latter breaks against transformers 5.x
(its tokenizer path calls the removed prepare_for_model). Scores are sigmoid-normalized to 0..1,
which is what rconfig.MIN_SCORE (the honest-refusal threshold) is calibrated against.

If scoring RETRIEVE_LIMIT pairs proves too slow on this CPU (brief flags 10+ s as the risk), the
knob order is: fewer candidates -> rerank only on ambiguous fusion scores -> READER_RERANK=false.
"""

import asyncio
import logging
import time

log = logging.getLogger("t800.reader")

_model = None
# One cross-encoder forward at a time (same reasoning as the embedder lock: two concurrent torch
# forwards on this 6-core CPU are slower than serialized ones).
_rlock = asyncio.Lock()


MIN_AVAILABLE_GB = 3.0  # fp32 weights ~2.3 GB + headroom


def _ram_guard() -> None:
    """Same protection as the embedder's: a graceful 'no reranker right now' beats an OOM'd bot.
    Callers treat a rerank failure as 'use fusion order' — quality degrades, nothing dies."""
    import psutil

    available_gb = psutil.virtual_memory().available / 1024**3
    if available_gb < MIN_AVAILABLE_GB:
        raise RuntimeError(
            f"не хватает RAM для реранкера: доступно {available_gb:.1f} ГБ, нужно ~{MIN_AVAILABLE_GB} ГБ"
        )


def _get_model():
    global _model
    if _model is None:
        _ram_guard()
        from sentence_transformers import CrossEncoder

        log.info("loading bge-reranker-v2-m3 (first use)...")
        # num_labels=1 => CrossEncoder applies sigmoid by default: predict() returns 0..1.
        _model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", max_length=512)
        log.info("reranker loaded")
    return _model


def _score(query: str, texts: list[str]) -> list[float]:
    model = _get_model()
    t0 = time.monotonic()
    scores = model.predict([(query, t) for t in texts], batch_size=8, show_progress_bar=False)
    log.info("rerank: %d pairs in %.1fs", len(texts), time.monotonic() - t0)
    return [float(s) for s in scores]


async def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Attach a normalized cross-encoder score to each candidate and sort best-first."""
    if not candidates:
        return []
    async with _rlock:
        scores = await asyncio.to_thread(_score, query, [c["text"] for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
