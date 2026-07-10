"""Chat-memory collection in the SAME embedded Qdrant instance the reader uses.

Embedded Qdrant allows exactly one client per storage path per process, so this module deliberately
borrows reader.store's client and lock instead of opening its own. Separate collection, same
hybrid dense+sparse layout, RRF fusion. Payload filters: chat_id always, author optionally
(the dossier axis), timestamp range optionally (the «что было вчера» axis).
"""

import asyncio
import logging

from qdrant_client import models

from reader.store import _get_client, _qlock

from .windows import Window

log = logging.getLogger("t800.chatmem")

COLLECTION = "chat_memory"
DENSE_DIM = 1024  # bge-m3


def _ensure_collection() -> None:
    client = _get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        log.info("created qdrant collection %s", COLLECTION)


def _upsert_windows(windows: list[Window], embeddings: list[dict]) -> None:
    _ensure_collection()
    points = []
    for w, emb in zip(windows, embeddings):
        points.append(
            models.PointStruct(
                id=w.point_id,
                vector={
                    "dense": emb["dense"],
                    "sparse": models.SparseVector(indices=emb["sparse_indices"], values=emb["sparse_values"]),
                },
                payload={
                    "kind": "window",
                    "chat_id": w.chat_id,
                    "ts_start": w.ts_start.timestamp(),
                    "ts_end": w.ts_end.timestamp(),
                    "msg_id_first": w.msg_id_first,
                    "msg_id_last": w.msg_id_last,
                    "authors": [a.lower() for a in w.authors],
                    "author_ids": w.author_ids,
                    "text": w.text,
                },
            )
        )
    _get_client().upsert(COLLECTION, points)


async def upsert_windows(windows: list[Window], embeddings: list[dict]) -> None:
    async with _qlock:
        await asyncio.to_thread(_upsert_windows, windows, embeddings)


def _upsert_media_point(point_id: str, text: str, payload: dict, emb: dict) -> None:
    """A described/transcribed media item as its own memory point (kind=media)."""
    _ensure_collection()
    _get_client().upsert(
        COLLECTION,
        [
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": emb["dense"],
                    "sparse": models.SparseVector(indices=emb["sparse_indices"], values=emb["sparse_values"]),
                },
                payload={"kind": "media", "text": text, **payload},
            )
        ],
    )


async def upsert_media_point(point_id: str, text: str, payload: dict, emb: dict) -> None:
    async with _qlock:
        await asyncio.to_thread(_upsert_media_point, point_id, text, payload, emb)


def _search(query_emb: dict, chat_id: int, limit: int, author: str | None,
            ts_from: float | None, ts_to: float | None, author_id: int | None = None) -> list[dict]:
    must: list = [models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
    if author_id:
        must.append(models.FieldCondition(key="author_ids", match=models.MatchValue(value=author_id)))
    elif author:
        must.append(models.FieldCondition(key="authors", match=models.MatchValue(value=author.lower())))
    if ts_from is not None or ts_to is not None:
        must.append(models.FieldCondition(key="ts_start", range=models.Range(gte=ts_from, lte=ts_to)))
    flt = models.Filter(must=must)
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


async def search(query_emb: dict, chat_id: int, limit: int = 16, author: str | None = None,
                 ts_from: float | None = None, ts_to: float | None = None,
                 author_id: int | None = None) -> list[dict]:
    async with _qlock:
        return await asyncio.to_thread(_search, query_emb, chat_id, limit, author, ts_from, ts_to, author_id)


def _wipe_chat(chat_id: int) -> None:
    """Delete every memory point of one chat (full re-ingest from a fresh export)."""
    _ensure_collection()
    _get_client().delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))
            ])
        ),
    )


async def wipe_chat(chat_id: int) -> None:
    async with _qlock:
        await asyncio.to_thread(_wipe_chat, chat_id)


def _count(chat_id: int) -> int:
    _ensure_collection()
    return _get_client().count(
        COLLECTION,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
        ),
        exact=True,
    ).count


async def count_points(chat_id: int) -> int:
    async with _qlock:
        return await asyncio.to_thread(_count, chat_id)
