"""Parser for Telegram Desktop HTML chat exports (messages*.html).

Telegram's export exists in two flavors; ours is HTML (the JSON one is nicer, but the media is
already downloaded next to this HTML — reparsing beats re-exporting). The HTML is stable and
class-annotated:

    div.message.service            — group created / user invited / photo changed (skipped)
    div.message.default[.joined]   — a real message; .joined = same author as previous (from_name
                                     omitted -> carried over)
      .pull_right.date[title]      — full timestamp in the title attribute (with UTC offset)
      .from_name                   — author display name
      .text                        — message text (links/formatting flattened to plain text)
      .media_wrap a[href]          — media, classified by css class + link target + title text
      .reply_to                    — "In reply to <a href=#go_to_message123>" — reply linkage

Media classes seen in the wild: media_photo, media_voice_message, media_audio_file, media_video,
media_file; stickers arrive as media_photo/media_video whose .title is literally "Sticker" (emoji
in .status). Animations are .mp4 in video_files/. Video messages (кружки) are media_video with
round thumbnails — distinguished by the "media_video" class + duration in .status.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lxml import html as lxml_html

log = logging.getLogger("t800.chatmem")


@dataclass
class ExportMessage:
    msg_id: int
    ts: datetime
    author: str
    text: str = ""
    kind: str = "text"          # text | photo | sticker | voice | video | animation | audio_file | file
    media_path: str | None = None   # relative to the export root
    media_note: str = ""        # emoji for stickers, "performer — title" for audio, duration etc.
    duration_s: int = 0         # voice/video/round duration when the export shows it
    reply_to: int | None = None
    author_id: int | None = None    # numeric user id — present only in JSON exports (from_id)


_DUR_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d\d)$")  # h:mm:ss or m:ss


def _parse_duration(s: str) -> int:
    m = _DUR_RE.match(s.strip())
    if not m:
        return 0
    h, mm, ss = m.groups()
    return (int(h or 0)) * 3600 + int(mm) * 60 + int(ss)


def _clean(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def _classify_media(a_node) -> tuple[str, str, str, int]:
    """-> (kind, media_path, media_note, duration_s) for one media_wrap link."""
    href = a_node.get("href", "")
    classes = a_node.get("class", "")
    title = _clean(a_node.find_class("title")[0]) if a_node.find_class("title") else ""
    status = _clean(a_node.find_class("status")[0]) if a_node.find_class("status") else ""

    if title == "Sticker":
        return "sticker", href, status, 0  # status carries the sticker's emoji
    if "media_voice_message" in classes:
        return "voice", href, "", _parse_duration(status)
    if "media_audio_file" in classes:
        return "audio_file", href, title if title != "Audio file" else Path(href).stem, _parse_duration(status)
    if "media_video" in classes:
        dur = _parse_duration(status)
        kind = "animation" if href.lower().endswith((".gif.mp4", ".gif")) else "video"
        return kind, href, "", dur
    if "media_photo" in classes:
        return "photo", href, "", 0
    if "media_animated" in classes or href.lower().endswith(".tgs"):
        return "sticker", href, status, 0
    # Plain file attachments (books, mp3 sent as document, ...)
    return "file", href, title, _parse_duration(status)


# --- JSON flavor (result.json) --------------------------------------------------------------------
# The machine-readable export. Richer than HTML: numeric from_id per message (exact identity — the
# HTML flavor only has display names), clean media_type, sticker_emoji, performer/title. Preferred
# when present.

_JSON_MEDIA_KIND = {
    "voice_message": "voice",
    "video_message": "video",   # кружок
    "video_file": "video",
    "animation": "animation",
    "sticker": "sticker",
    "audio_file": "audio_file",
}


def _json_text(raw) -> str:
    """'text' is either a plain string or a list of strings and entity dicts — flatten."""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts = []
        for piece in raw:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict):
                parts.append(piece.get("text", ""))
        return re.sub(r"\s+", " ", "".join(parts)).strip()
    return ""


def _json_media_path(value) -> str | None:
    """Media fields hold a relative path OR a '(File not included...)' placeholder when the
    export was made without downloading that media type."""
    if isinstance(value, str) and value and not value.startswith("("):
        return value
    return None


def parse_result_json(json_path: Path) -> list[ExportMessage]:
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    messages: list[ExportMessage] = []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue  # service entries (joins, pins, group photo) carry no dialogue value
        try:
            ts = datetime.fromisoformat(m["date"])
            msg_id = int(m["id"])
        except (KeyError, ValueError):
            continue

        author = (m.get("from") or "?").strip()
        author_id = None
        from_id = m.get("from_id") or ""
        if isinstance(from_id, str) and from_id.startswith("user"):
            try:
                author_id = int(from_id[4:])
            except ValueError:
                pass

        text = _json_text(m.get("text"))
        if m.get("forwarded_from"):
            text = f"[переслал от {m['forwarded_from']}] {text}".strip()

        reply_to = m.get("reply_to_message_id")
        duration = int(m.get("duration_seconds") or 0)

        kind = "text"
        media_path = None
        media_note = ""
        if m.get("photo"):
            # Keep the message even when the file wasn't downloaded ('(File not included...)') —
            # the [фото] placeholder is dialogue signal; only media_path becomes None.
            kind, media_path = "photo", _json_media_path(m.get("photo"))
        elif m.get("media_type"):
            kind = _JSON_MEDIA_KIND.get(m["media_type"], "file")
            media_path = _json_media_path(m.get("file"))
            if kind == "sticker":
                media_note = m.get("sticker_emoji", "")
            elif kind == "audio_file":
                performer, title = m.get("performer", ""), m.get("title", "")
                media_note = " — ".join(x for x in (performer, title) if x) or m.get("file_name", "")
            elif kind == "file":
                media_note = m.get("file_name", "")
        elif _json_media_path(m.get("file")):
            kind, media_path = "file", _json_media_path(m.get("file"))
            media_note = m.get("file_name", "")

        if kind == "text" and not text:
            continue
        messages.append(
            ExportMessage(
                msg_id=msg_id, ts=ts, author=author, text=text, kind=kind,
                media_path=media_path, media_note=media_note, duration_s=duration,
                reply_to=reply_to, author_id=author_id,
            )
        )

    messages.sort(key=lambda m: (m.ts, m.msg_id))
    log.info("parsed JSON export: %d messages", len(messages))
    return messages


def parse_export(export_dir: str | Path) -> list[ExportMessage]:
    export_dir = Path(export_dir)
    # JSON flavor wins when present: exact author ids beat display-name guessing.
    result_json = export_dir / "result.json"
    if result_json.exists():
        return parse_result_json(result_json)

    pages = sorted(
        export_dir.glob("messages*.html"),
        key=lambda p: int(re.search(r"messages(\d*)", p.stem).group(1) or 1),
    )
    if not pages:
        raise FileNotFoundError(f"no result.json or messages*.html under {export_dir}")

    messages: list[ExportMessage] = []
    last_author = "?"
    for page in pages:
        tree = lxml_html.fromstring(page.read_bytes())
        for node in tree.find_class("message"):
            classes = node.get("class", "")
            if "default" not in classes:
                continue  # service messages carry no dialogue value
            try:
                msg_id = int(node.get("id", "message-0").split("-")[-1])
            except ValueError:
                continue

            date_node = node.find_class("date")
            ts_raw = date_node[0].get("title", "") if date_node else ""
            # "16.06.2026 22:11:53 UTC+05:00" -> naive local datetime (the chat lives in one TZ)
            try:
                ts = datetime.strptime(ts_raw.split(" UTC")[0], "%d.%m.%Y %H:%M:%S")
            except ValueError:
                continue

            # Author = the from_name OUTSIDE any .forwarded block (the forwarder). The from_name
            # INSIDE .forwarded is the original author with a datetime glued on — never let it
            # poison last_author (it also breaks attribution of subsequent .joined messages).
            fwd_from = None
            for fn in node.find_class("from_name"):
                inside_fwd = any("forwarded" in (anc.get("class") or "") for anc in fn.iterancestors("div"))
                name = _clean(fn).split(" via ")[0]
                if inside_fwd:
                    # "Apollo 26.06.2026 23:35:04" -> "Apollo"
                    fwd_from = re.sub(r"\s*\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$", "", name)
                else:
                    last_author = name
            author = last_author

            text = ""
            for t in node.find_class("text"):
                text = _clean(t)
                break
            if fwd_from:
                text = f"[переслал от {fwd_from}] {text}".strip()

            reply_to = None
            for r in node.find_class("reply_to"):
                m = re.search(r"go_to_message(\d+)", lxml_html.tostring(r, encoding="unicode"))
                if m:
                    reply_to = int(m.group(1))
                break

            # Media comes in three wrapper flavors: a.media_* (voice/video/audio/files, and video
            # stickers posing as media_photo with title "Sticker"), a.photo_wrap (real photos),
            # a.sticker_wrap (static/animated stickers).
            kind = path = note = None
            dur = 0
            for a in node.iter("a"):
                cls = a.get("class") or ""
                if "media" in cls and "media_wrap" not in cls:
                    kind, path, note, dur = _classify_media(a)
                    break
                if "photo_wrap" in cls:
                    kind, path, note = "photo", a.get("href", ""), ""
                    break
                if "sticker_wrap" in cls:
                    kind, path, note = "sticker", a.get("href", ""), ""
                    break

            if kind:
                messages.append(
                    ExportMessage(
                        msg_id=msg_id, ts=ts, author=author, text=text, kind=kind,
                        media_path=path, media_note=note, duration_s=dur, reply_to=reply_to,
                    )
                )
            elif text:
                messages.append(
                    ExportMessage(msg_id=msg_id, ts=ts, author=author, text=text, reply_to=reply_to)
                )

    messages.sort(key=lambda m: (m.ts, m.msg_id))
    log.info("parsed export: %d messages from %d html pages", len(messages), len(pages))
    return messages


if __name__ == "__main__":
    import sys
    from collections import Counter

    msgs = parse_export(sys.argv[1])
    kinds = Counter(m.kind for m in msgs)
    authors = Counter(m.author for m in msgs)
    voice_s = sum(m.duration_s for m in msgs if m.kind == "voice")
    video_s = sum(m.duration_s for m in msgs if m.kind == "video")
    print(f"messages: {len(msgs)}  |  {msgs[0].ts} .. {msgs[-1].ts}")
    print("kinds:", dict(kinds.most_common()))
    print("authors:", dict(authors.most_common()))
    print(f"voice total: {voice_s}s ({voice_s / 60:.0f} min), video total: {video_s}s ({video_s / 60:.0f} min)")
    with_text = sum(1 for m in msgs if m.text)
    print(f"messages with text: {with_text}, total text chars: {sum(len(m.text) for m in msgs)}")
