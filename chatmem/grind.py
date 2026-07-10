"""Media grind: turn the export's voices/photos/videos into attributed text memory points.

Queue order is cheapest-first: voices (whisper, LLM-free, seconds each) -> photos (12B vision,
~half a minute each, politeness-gated) -> videos/animations (frames + 12B, the slowest). Every
item becomes its own memory point («артём кинул фото (03.07): два кота дерутся за шаверму») —
windows are never rewritten, so nothing gets re-embedded.

Resumable: point ids derive from the media path; already-stored ids are skipped. The whole run
lives inside the bot process (embedded Qdrant) and yields to live chat between LLM calls exactly
like the book-summary ingest does.
"""

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import config
from qdrant_client import models

from reader import embedder
from reader import llm as reader_llm
from reader.store import _get_client, _qlock

from . import store, transcribe
from .telegram_export import ExportMessage, parse_export

log = logging.getLogger("t800.chatmem")

PHOTO_MAX_SIDE = 640      # dossier descriptions don't need full-res; keeps encoder+prefill sane
VIDEO_FRAMES = 5
DESCRIBE_MAX_TOKENS = 120

PHOTO_PROMPT = (
    "Опиши это изображение одним-двумя предложениями по-русски: что происходит/что изображено. "
    "Если на картинке есть читаемый текст (мем, скриншот) — процитируй его суть. Без вступлений."
)
VIDEO_PROMPT = (
    "Это кадры одного видео по порядку. Опиши одним-двумя предложениями, что в нём происходит. "
    "Без вступлений."
)


def _media_point_id(chat_id: int, media_path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chatmedia:{chat_id}:{media_path}"))


def _existing_media_ids(chat_id: int) -> set[str]:
    client = _get_client()
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            store.COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id)),
                models.FieldCondition(key="kind", match=models.MatchValue(value="media")),
            ]),
            with_payload=False, with_vectors=False, limit=512, offset=offset,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def _jpeg_of(path: Path) -> bytes:
    """Photo -> bounded JPEG via ffmpeg (also flattens odd formats the same way stickers taught us)."""
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "o.jpg")
        subprocess.run(
            [config.FFMPEG_EXE, "-y", "-i", str(path),
             "-vf", f"scale='min({PHOTO_MAX_SIDE},iw)':-2", "-frames:v", "1", dst],
            capture_output=True, check=True,
        )
        return open(dst, "rb").read()


def _frames_of(path: Path, n: int) -> list[bytes]:
    with tempfile.TemporaryDirectory() as td:
        pr = subprocess.run(
            [config.FFPROBE_EXE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        try:
            dur = float(pr.stdout.strip())
        except ValueError:
            dur = 0.0
        fps = max(0.1, n / dur) if dur > 0 else 2.0
        pat = os.path.join(td, "f_%03d.jpg")
        subprocess.run(
            [config.FFMPEG_EXE, "-y", "-i", str(path),
             "-vf", f"fps={fps:.4f},scale='min({config.FRAME_MAX_WIDTH},iw)':-2",
             "-frames:v", str(n), pat],
            capture_output=True, check=True,
        )
        out = []
        for i in range(1, n + 1):
            p = os.path.join(td, f"f_{i:03d}.jpg")
            if os.path.exists(p):
                out.append(open(p, "rb").read())
        return out


async def _describe_images(images: list[bytes], prompt: str) -> str:
    """Single image -> the grinder (GPU Qwen when up, main 12B politely otherwise). Multi-image
    (video frames) stays on the main 12B: the grinder's small ctx (2560) can't hold 5 frames."""
    if len(images) == 1:
        from reader import grinder

        return await grinder.describe_image(images[0], prompt, max_tokens=DESCRIBE_MAX_TOKENS)
    content = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = base64.b64encode(img).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    await reader_llm.wait_for_quiet()
    async with reader_llm.llm_lock:
        return await reader_llm.generate(
            [{"role": "user", "content": content}],
            max_tokens=DESCRIBE_MAX_TOKENS,
            timeout=config.LOCAL_VISION_TIMEOUT_SECONDS + 60,
        )


async def _store_point(m: ExportMessage, chat_id: int, text: str) -> None:
    emb = (await embedder.embed_texts([text]))[0]
    await store.upsert_media_point(
        _media_point_id(chat_id, m.media_path),
        text,
        {
            "chat_id": chat_id,
            "ts_start": m.ts.timestamp(),
            "ts_end": m.ts.timestamp(),
            "msg_id_first": m.msg_id,
            "msg_id_last": m.msg_id,
            "authors": [m.author.lower()],
            "media": m.kind,
        },
        emb,
    )


async def run_grind(
    export_dir: str,
    chat_id: int,
    progress: Callable[[str], Awaitable[None]],
    kinds: tuple[str, ...] = ("voice", "photo", "video", "animation"),
) -> str:
    export_root = Path(export_dir)
    messages = await asyncio.to_thread(parse_export, export_dir)

    async with _qlock:
        done_ids = await asyncio.to_thread(_existing_media_ids, chat_id)

    queue: list[ExportMessage] = []
    for m in messages:
        if m.kind in kinds and m.media_path and (export_root / m.media_path).exists():
            if _media_point_id(chat_id, m.media_path) not in done_ids:
                queue.append(m)
    order = {"voice": 0, "photo": 1, "video": 2, "animation": 3}
    queue.sort(key=lambda m: (order.get(m.kind, 9), m.ts))
    counts = {k: sum(1 for m in queue if m.kind == k) for k in kinds}
    await progress(f"Медиа в очереди: {counts} (уже усвоено: {len(done_ids)}). Начинаю.")

    # Warm the embedder BEFORE anything else: it's the smallest model in the chain, but if its
    # RAM guard refuses, every transcription/description would be wasted work (seen live: whisper
    # loaded first, ate the last GB, embedder bounced, all results died at the store step).
    try:
        await embedder.embed_texts(["прогрев"])
    except Exception as exc:
        await progress(f"Грайнд отменён: эмбеддер не загрузился ({exc}). Освободи RAM и повтори.")
        return "aborted: embedder unavailable"

    t0 = time.monotonic()
    done = 0
    failed = 0
    last_tick = 0.0
    for m in queue:
        path = export_root / m.media_path
        stamp = m.ts.strftime("%d.%m")
        try:
            if m.kind == "voice":
                tr = await transcribe.transcribe(str(path))
                if not tr:
                    raise RuntimeError("empty transcript")
                text = f"{m.author} (голосовое, {stamp}): {tr}"
            elif m.kind == "photo":
                jpeg = await asyncio.to_thread(_jpeg_of, path)
                desc = await _describe_images([jpeg], PHOTO_PROMPT)
                caption = f" Подпись: {m.text}" if m.text else ""
                text = f"{m.author} кинул фото ({stamp}): {desc}{caption}"
            else:  # video | animation
                frames = await asyncio.to_thread(_frames_of, path, VIDEO_FRAMES)
                if not frames:
                    raise RuntimeError("no frames")
                desc = await _describe_images(frames, VIDEO_PROMPT)
                noun = "гифку" if m.kind == "animation" else "видео"
                text = f"{m.author} кинул {noun} ({stamp}): {desc}"
            await _store_point(m, chat_id, text)
            done += 1
        except Exception:
            failed += 1
            log.exception("grind failed on %s", m.media_path)

        if time.monotonic() - last_tick > 20:
            last_tick = time.monotonic()
            if done:
                rate = done / max(0.1, time.monotonic() - t0)
                eta = f"осталось ~{max(1, int((len(queue) - done - failed) / rate) // 60)} мин"
            else:
                eta = "оцениваю темп"  # no completions yet — a rate of 0 would print garbage ETA
            await progress(f"Медиа: {done}/{len(queue)} усвоено (ошибок {failed}), {eta}.")

    summary = f"Медиа-грайнд завершён за {(time.monotonic() - t0) / 60:.0f} мин: {done} усвоено, {failed} сбоев."
    log.info(summary)  # progress() only edits the Telegram message — the log needs its own line
    await progress(summary)
    return summary
