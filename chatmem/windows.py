"""Cut a message stream into dialogue windows — the unit of chat memory.

A lone «лол» is useless as a retrieval chunk; a window of 10-15 consecutive messages carries the
actual exchange. Windows close on a silence gap (a new «session»), on message count, or on size.
Media inside a window stays as an attributed placeholder line («артём кинул трек: …») — real
descriptions/transcripts arrive later from the grind pipeline as SEPARATE memory points, so
windows never need re-embedding.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .telegram_export import ExportMessage

GAP_MINUTES = 30      # silence longer than this starts a new window (session boundary)
MAX_MESSAGES = 15
MAX_CHARS = 1200


@dataclass
class Window:
    chat_id: int
    ts_start: datetime
    ts_end: datetime
    msg_id_first: int
    msg_id_last: int
    authors: list[str]
    text: str
    author_ids: list[int] = field(default_factory=list)  # exact ids — JSON exports only

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chatwin:{self.chat_id}:{self.msg_id_first}:{self.msg_id_last}"))


def render_line(m: ExportMessage) -> str:
    """One message -> one attributed line of window text."""
    stamp = m.ts.strftime("%d.%m %H:%M")
    body = m.text or ""
    if m.kind == "photo":
        body = f"[фото] {body}".strip()
    elif m.kind == "sticker":
        body = f"[стикер {m.media_note}]".strip() if m.media_note else "[стикер]"
    elif m.kind == "voice":
        body = f"[голосовое {m.duration_s} с] {body}".strip()
    elif m.kind == "video":
        body = f"[видео] {body}".strip()
    elif m.kind == "animation":
        body = f"[гифка] {body}".strip()
    elif m.kind == "audio_file":
        body = f"[кинул трек: {m.media_note}] {body}".strip()
    elif m.kind == "file":
        body = f"[файл: {m.media_note}] {body}".strip()
    return f"[{stamp}] {m.author}: {body}"


def build_windows(messages: list[ExportMessage], chat_id: int) -> list[Window]:
    windows: list[Window] = []
    buf: list[ExportMessage] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        authors = sorted({m.author for m in buf})
        author_ids = sorted({m.author_id for m in buf if m.author_id})
        windows.append(
            Window(
                chat_id=chat_id,
                ts_start=buf[0].ts, ts_end=buf[-1].ts,
                msg_id_first=buf[0].msg_id, msg_id_last=buf[-1].msg_id,
                authors=authors, author_ids=author_ids,
                text="\n".join(render_line(m) for m in buf),
            )
        )
        buf, size = [], 0

    for m in messages:
        line_len = len(m.text) + 30
        gap = buf and (m.ts - buf[-1].ts).total_seconds() > GAP_MINUTES * 60
        if gap or len(buf) >= MAX_MESSAGES or (buf and size + line_len > MAX_CHARS):
            flush()
        buf.append(m)
        size += line_len
    flush()
    return windows
