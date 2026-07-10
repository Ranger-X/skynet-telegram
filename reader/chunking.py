"""Turn a parsed chapter tree into leaf chunks (embedded, searched) and parent windows (fetched
for generation — small-to-big, brief §4). Leaves are ~300-500 tokens cut on sentence boundaries
(razdel); parents are ~2k-token windows of consecutive leaves within one chapter. Every piece
carries the monotonic char offset that drives the spoiler filter.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass

from razdel import sentenize

from .parsing import ParsedDoc

log = logging.getLogger("t800.reader")

# Char targets derived from ~3.3 chars/token for Russian prose. Exact token counts don't matter —
# only that leaves stay small enough to search sharply and parents fit the generation budget.
LEAF_TARGET_CHARS = 1300   # ~400 tokens
LEAF_MAX_CHARS = 1800      # hard cap ~550 tokens
PARENT_TARGET_CHARS = 6000 # ~1800 tokens; 2-3 parents fit the ~4k in-context budget with room to spare


@dataclass
class Leaf:
    chunk_id: str
    parent_id: str
    doc_id: str
    text: str
    offset: int          # max char offset of the leaf's text — the spoiler-filter key
    start_offset: int
    chapter_idx: int
    chapter_title: str
    level: str = "leaf"  # leaf | chapter_summary


@dataclass
class Parent:
    parent_id: str
    doc_id: str
    chapter_idx: int
    start_offset: int
    end_offset: int
    text: str


def point_id(chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs/ints; derive a stable UUID from the string id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _split_sentences(paragraph: str) -> list[str]:
    return [s.text for s in sentenize(paragraph)] or [paragraph]


def chunk_document(doc_id: str, parsed: ParsedDoc) -> tuple[list[Leaf], list[Parent]]:
    leaves: list[Leaf] = []
    parents: list[Parent] = []

    for ch_idx, ch in enumerate(parsed.chapters):
        # 1) Cut the chapter into leaves on sentence boundaries, tracking char offsets.
        ch_leaf_texts: list[tuple[str, int]] = []  # (text, start_offset_in_doc)
        buf: list[str] = []
        buf_len = 0
        cursor = ch.start_offset  # char offset where the CURRENT buffer starts

        def flush():
            nonlocal buf, buf_len, cursor
            if buf:
                text = " ".join(buf)
                ch_leaf_texts.append((text, cursor))
                cursor += buf_len
                buf, buf_len = [], 0

        pos_in_chapter = 0  # chars consumed of ch.text so far, mirrors the "\n" paragraph joints
        for p_i, para in enumerate(ch.paragraphs):
            for sent in _split_sentences(para):
                if buf and buf_len + len(sent) + 1 > LEAF_MAX_CHARS:
                    flush()
                elif buf and buf_len >= LEAF_TARGET_CHARS:
                    flush()
                if not buf:
                    cursor = ch.start_offset + pos_in_chapter
                buf.append(sent)
                buf_len += len(sent) + 1
                pos_in_chapter += len(sent) + 1
            # Rough alignment: paragraph joint. (Sentence joins inside razdel may differ from the
            # original by a char or two of whitespace — irrelevant at spoiler-filter granularity.)
            pos_in_chapter = min(pos_in_chapter, len(ch.text) if p_i == len(ch.paragraphs) - 1 else pos_in_chapter)
        flush()

        # 2) Group consecutive leaves into parent windows within the chapter.
        window: list[tuple[str, int]] = []
        window_len = 0
        windows: list[list[tuple[str, int]]] = []
        for text, start in ch_leaf_texts:
            if window and window_len + len(text) > PARENT_TARGET_CHARS:
                windows.append(window)
                window, window_len = [], 0
            window.append((text, start))
            window_len += len(text) + 1
        if window:
            windows.append(window)

        for w_idx, w in enumerate(windows):
            parent_id = f"{doc_id[:16]}:p:{ch_idx}:{w_idx}"
            w_text = "\n".join(t for t, _ in w)
            w_start = w[0][1]
            w_end = min(w[-1][1] + len(w[-1][0]), ch.end_offset)
            parents.append(
                Parent(
                    parent_id=parent_id, doc_id=doc_id, chapter_idx=ch_idx,
                    start_offset=w_start, end_offset=w_end, text=w_text,
                )
            )
            for l_idx, (text, start) in enumerate(w):
                end = min(start + len(text), ch.end_offset)
                leaves.append(
                    Leaf(
                        chunk_id=f"{doc_id[:16]}:l:{ch_idx}:{w_idx}:{l_idx}",
                        parent_id=parent_id, doc_id=doc_id, text=text,
                        offset=end, start_offset=start,
                        chapter_idx=ch_idx, chapter_title=ch.title,
                    )
                )

    log.info(
        "chunked doc %s: %d leaves, %d parents, %d chapters",
        doc_id[:12], len(leaves), len(parents), len(parsed.chapters),
    )
    return leaves, parents


def doc_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
