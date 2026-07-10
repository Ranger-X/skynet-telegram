"""Qdrant in embedded mode (qdrant-client with a local path — no server, no Docker; brief §1a).
One collection holds every document's chunks; doc_id is a payload filter. Named dense + sparse
vectors, RRF fusion server-side (well, library-side), and the payload-range condition that IS the
spoiler filter (brief §6).
"""

import asyncio
import logging
from pathlib import Path

from qdrant_client import QdrantClient, models

from .chunking import Leaf, point_id

log = logging.getLogger("t800.reader")

STORE_PATH = Path(__file__).resolve().parent.parent / "qdrant_data"
COLLECTION = "reader_chunks"
DENSE_DIM = 1024  # bge-m3

_client: QdrantClient | None = None
# Embedded QdrantLocal is not thread-safe; ingest upserts and /ask searches can otherwise land in
# two executor threads at once. All store ops funnel through this lock.
_qlock = asyncio.Lock()


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(STORE_PATH))
        if not _client.collection_exists(COLLECTION):
            _client.create_collection(
                COLLECTION,
                vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            log.info("created qdrant collection %s at %s", COLLECTION, STORE_PATH)
    return _client


def _upsert(leaves: list[Leaf], embeddings: list[dict]) -> None:
    client = _get_client()
    points = []
    for leaf, emb in zip(leaves, embeddings):
        points.append(
            models.PointStruct(
                id=point_id(leaf.chunk_id),
                vector={
                    "dense": emb["dense"],
                    "sparse": models.SparseVector(indices=emb["sparse_indices"], values=emb["sparse_values"]),
                },
                payload={
                    "chunk_id": leaf.chunk_id,
                    "parent_id": leaf.parent_id,
                    "doc_id": leaf.doc_id,
                    "text": leaf.text,
                    "offset": leaf.offset,
                    "start_offset": leaf.start_offset,
                    "chapter_idx": leaf.chapter_idx,
                    "chapter_title": leaf.chapter_title,
                    "level": leaf.level,
                },
            )
        )
    client.upsert(COLLECTION, points)


async def upsert_leaves(leaves: list[Leaf], embeddings: list[dict]) -> None:
    async with _qlock:
        await asyncio.to_thread(_upsert, leaves, embeddings)


def _delete_doc(doc_id: str) -> None:
    _get_client().delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        ),
    )


async def delete_doc(doc_id: str) -> None:
    async with _qlock:
        await asyncio.to_thread(_delete_doc, doc_id)


def _search(query_emb: dict, doc_id: str, max_offset: int, level: str, limit: int) -> list[dict]:
    """Hybrid dense+sparse retrieval fused with RRF, constrained to one doc, one level, and — the
    spoiler filter — only chunks whose offset lies at or before the reader's position."""
    flt = models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="level", match=models.MatchValue(value=level)),
            models.FieldCondition(key="offset", range=models.Range(lte=max_offset)),
        ]
    )
    res = _get_client().query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=query_emb["dense"], using="dense", filter=flt, limit=limit * 2),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_emb["sparse_indices"], values=query_emb["sparse_values"]
                ),
                using="sparse", filter=flt, limit=limit * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [{"score": p.score, **(p.payload or {})} for p in res.points]


async def search(query_emb: dict, doc_id: str, max_offset: int, level: str = "leaf", limit: int = 24) -> list[dict]:
    async with _qlock:
        return await asyncio.to_thread(_search, query_emb, doc_id, max_offset, level, limit)


def _existing_chunk_ids(doc_id: str) -> set[str]:
    """chunk_ids already in the store for this doc — lets a restarted ingest resume instead of
    re-embedding from zero (chunking is deterministic, so ids are stable)."""
    client = _get_client()
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            with_payload=["chunk_id"], with_vectors=False, limit=512, offset=offset,
        )
        ids.update(p.payload["chunk_id"] for p in points if p.payload)
        if offset is None:
            return ids


async def existing_chunk_ids(doc_id: str) -> set[str]:
    async with _qlock:
        return await asyncio.to_thread(_existing_chunk_ids, doc_id)


def _page_texts(doc_id: str, first: int, last: int) -> list[str]:
    """Leaf texts of a manga doc in [first..last] page range, page order. Powers neighbor-page
    context assembly and chapter summaries (offset == page number for manga)."""
    client = _get_client()
    out: list[tuple[int, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="level", match=models.MatchValue(value="leaf")),
                models.FieldCondition(key="offset", range=models.Range(gte=first, lte=last)),
            ]),
            with_payload=["offset", "text"], with_vectors=False, limit=256, offset=offset,
        )
        for p in points:
            out.append((p.payload["offset"], p.payload["text"]))
        if offset is None:
            break
    return [t for _, t in sorted(out)]


async def page_texts(doc_id: str, first: int, last: int) -> list[str]:
    async with _qlock:
        return await asyncio.to_thread(_page_texts, doc_id, first, last)


def _count(doc_id: str) -> int:
    return _get_client().count(
        COLLECTION,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        ),
        exact=True,
    ).count


async def count_points(doc_id: str) -> int:
    async with _qlock:
        return await asyncio.to_thread(_count, doc_id)
