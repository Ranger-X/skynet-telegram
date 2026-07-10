"""bge-m3 embeddings on CPU (brief §5: multilingual is non-negotiable for Russian books; one
forward pass yields BOTH dense and sparse-lexical vectors, which is what the hybrid search fuses).

The model (~2.3 GB RAM) is lazy-loaded on first use and kept resident. Encoding is CPU-bound —
always call the async wrappers, which push work to a thread so the bot keeps answering.
"""

import asyncio
import logging

log = logging.getLogger("t800.reader")

_model = None
# One encode at a time: two concurrent torch forwards (ingest batch + /ask query) would thrash the
# 6-core CPU and slow BOTH; serializing is strictly faster here.
_elock = asyncio.Lock()


MIN_AVAILABLE_GB = 1.1  # fp16 weights ~1.2 GB. Calibration history: 2.2 bounced a viable grind at
# 2.0 available; 1.7 bounced /ask at 1.5 and even the startup preload at 1.23. The number that
# matters for NOT DYING is commit headroom, not physical: with ~13 GB free virtual (measured
# 2026-07-10) a 1.2 GB load can't OOM the process — worst case Windows pages colder llama pages
# out and prefill slows. So the guard now only blocks truly desperate states.


def _ram_guard() -> None:
    """Refuse to load rather than OOM the whole bot: with llama-server resident (~11 GB working
    set) free RAM can dip under 0.5 GB, and a 2+ GB lazy model load at that moment KILLS the
    process (seen live: /profile's first RAG call took the bot down). psutil 'available' counts
    Windows standby pages (the mmap'd GGUF cache is reclaimable), so a pass here means the load
    can evict cache instead of dying."""
    import psutil

    available_gb = psutil.virtual_memory().available / 1024**3
    if available_gb < MIN_AVAILABLE_GB:
        raise RuntimeError(
            f"не хватает RAM для эмбеддера: доступно {available_gb:.1f} ГБ, нужно ~{MIN_AVAILABLE_GB} ГБ"
        )


def _get_model():
    global _model
    if _model is None:
        _ram_guard()
        import torch

        from FlagEmbedding import BGEM3FlagModel

        # Explicit: all 6 physical cores for the forward pass (defaults sometimes undershoot on
        # hybrid-scheduler Windows), single-threaded interop to avoid oversubscription.
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        log.info("loading bge-m3 embedder (first use)...")
        # fp16 halves resident RAM (2.3 -> ~1.2 GB) at the cost of slower CPU inference — the
        # right trade next to an 11 GB llama-server. Embedding quality is unaffected for our use.
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cpu")
        log.info("bge-m3 loaded (fp16)")
    return _model


def _encode(texts: list[str], batch_size: int) -> list[dict]:
    model = _get_model()
    # max_length 640 covers the biggest leaf (~1800 chars ≈ 560 tok) with margin; the previous 1024
    # padded every batch to nothing but zeros and measured ~2x slower on this CPU.
    out = model.encode(
        texts, batch_size=batch_size, max_length=640,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    results = []
    for dense, lex in zip(out["dense_vecs"], out["lexical_weights"]):
        # lexical_weights: {token_id_str: weight}; Qdrant wants parallel int/float arrays.
        indices = [int(k) for k in lex]
        values = [float(v) for v in lex.values()]
        results.append({"dense": dense.tolist(), "sparse_indices": indices, "sparse_values": values})
    return results


async def embed_texts(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Embed a batch of chunk texts. batch_size stays small: peak RAM during encode scales with it,
    and this box runs the 12B alongside."""
    async with _elock:
        return await asyncio.to_thread(_encode, texts, batch_size)


async def embed_query(text: str) -> dict:
    async with _elock:
        return (await asyncio.to_thread(_encode, [text], 1))[0]


def preload() -> None:
    """Optional warm-up so the first ingest doesn't pay model-load latency mid-progress."""
    _get_model()
