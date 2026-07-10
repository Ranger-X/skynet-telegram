"""Parse FB2/EPUB into a chapter tree with positional anchors.

The offset unit is CHARS of normalized plain text, one monotonic counter per document, fixed at
parse time (see brief §3) — bookmarks must survive any later change of tokenizer or embedder.
Normalization: paragraph text is whitespace-collapsed; paragraphs are joined with a single "\\n".
Every downstream offset (chunks, chapters, positions, spoiler filter) counts chars of exactly this
text, so parsing must stay deterministic for a given file.
"""

import logging
import re
from dataclasses import dataclass, field

from lxml import etree, html as lxml_html

log = logging.getLogger("t800.reader")

# Chapters bigger than this get split into virtual sections (~3-5k tokens each, brief §4.1):
# they play the role of chapters for summaries and small-to-big, and keep the KV budget sane.
VIRTUAL_SECTION_CHARS = 14000


@dataclass
class Chapter:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    # Filled in by finalize():
    start_offset: int = 0
    end_offset: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


@dataclass
class ParsedDoc:
    title: str
    author: str
    chapters: list[Chapter]
    doc_len: int  # total chars of normalized text (per-chapter texts + the "\n" chapter joints)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --- FB2 -----------------------------------------------------------------------------------------

_FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"


def _fb2_tag(el) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _fb2_section_title(section) -> str:
    t = section.find(f"{{{_FB2_NS}}}title")
    if t is None:
        return ""
    return _norm(" ".join(t.itertext()))


def _fb2_collect_sections(el, out: list[Chapter], inherited_title: str = "") -> None:
    """Depth-first over <section>. A section that contains child sections is a part/book divider —
    its title is folded into the children's titles; a section with none is a leaf chapter."""
    title = _fb2_section_title(el) or inherited_title
    child_sections = [c for c in el if _fb2_tag(c) == "section"]
    if child_sections:
        for c in child_sections:
            _fb2_collect_sections(c, out, inherited_title=title)
        return
    paragraphs = []
    for node in el.iter():
        if _fb2_tag(node) in ("p", "subtitle", "text-author"):
            text = _norm(" ".join(node.itertext()))
            if text:
                paragraphs.append(text)
    if paragraphs:
        out.append(Chapter(title=title, paragraphs=paragraphs))


def parse_fb2(data: bytes) -> ParsedDoc:
    root = etree.fromstring(data, parser=etree.XMLParser(recover=True, huge_tree=True))
    ns = {"fb": _FB2_NS}

    def _first(xpath: str) -> str:
        nodes = root.xpath(xpath, namespaces=ns)
        return _norm(" ".join(nodes[0].itertext())) if nodes else ""

    title = _first("//fb:description/fb:title-info/fb:book-title")
    first_name = _first("//fb:description/fb:title-info/fb:author/fb:first-name")
    last_name = _first("//fb:description/fb:title-info/fb:author/fb:last-name")
    author = _norm(f"{first_name} {last_name}")

    chapters: list[Chapter] = []
    for body in root.findall(f"{{{_FB2_NS}}}body"):
        if body.get("name") == "notes":  # endnotes would pollute retrieval AND leak late-book text
            continue
        for section in body.findall(f"{{{_FB2_NS}}}section"):
            _fb2_collect_sections(section, chapters)

    return _finalize(title or "Без названия", author, chapters)


# --- EPUB ----------------------------------------------------------------------------------------

def parse_epub(data: bytes) -> ParsedDoc:
    # ebooklib insists on a file path — feed it via a temp file.
    import tempfile

    from ebooklib import ITEM_DOCUMENT, epub

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})
    finally:
        import os

        os.unlink(tmp_path)

    title = _norm(" ".join(v[0] for v in book.get_metadata("DC", "title") or [("", {})] if v[0]))
    author = _norm(" ".join(v[0] for v in book.get_metadata("DC", "creator") or [("", {})] if v[0]))

    # Spine order = reading order. Each spine document becomes a chapter; its title is the first
    # heading found inside (falls back to a numbered label at finalize time).
    items_by_id = {it.id: it for it in book.get_items_of_type(ITEM_DOCUMENT)}
    chapters: list[Chapter] = []
    for spine_id, _linear in book.spine:
        it = items_by_id.get(spine_id)
        if it is None:
            continue
        try:
            tree = lxml_html.fromstring(it.get_content())
        except etree.ParserError:
            continue
        for bad in tree.xpath("//script|//style"):
            bad.getparent().remove(bad)
        heading = tree.xpath("(//h1|//h2|//h3)[1]")
        ch_title = _norm(" ".join(heading[0].itertext())) if heading else ""
        paragraphs = []
        blocks = tree.xpath("//p | //blockquote | //li") or [tree]
        for node in blocks:
            text = _norm(" ".join(node.itertext()))
            if text:
                paragraphs.append(text)
        if heading and paragraphs and paragraphs[0] == ch_title:
            paragraphs = paragraphs[1:]
        if paragraphs:
            chapters.append(Chapter(title=ch_title, paragraphs=paragraphs))

    return _finalize(title or "Без названия", author, chapters)


# --- shared finalization -------------------------------------------------------------------------

def _split_virtual(ch: Chapter, max_chars: int = VIRTUAL_SECTION_CHARS) -> list[Chapter]:
    """Split an oversized (or only) chapter into virtual sections on paragraph boundaries."""
    if len(ch.text) <= max_chars:
        return [ch]
    parts: list[Chapter] = []
    buf: list[str] = []
    size = 0
    for p in ch.paragraphs:
        if buf and size + len(p) > max_chars:
            parts.append(Chapter(title="", paragraphs=buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 1
    if buf:
        parts.append(Chapter(title="", paragraphs=buf))
    base = ch.title or "Раздел"
    for i, part in enumerate(parts, 1):
        part.title = f"{base} ({i}/{len(parts)})"
    return parts


def _finalize(title: str, author: str, chapters: list[Chapter]) -> ParsedDoc:
    # Oversized chapters (or a chapterless wall of text) become virtual sections.
    split: list[Chapter] = []
    for ch in chapters:
        split.extend(_split_virtual(ch))
    chapters = split

    offset = 0
    for i, ch in enumerate(chapters):
        if not ch.title:
            ch.title = f"Глава {i + 1}"
        ch.start_offset = offset
        offset += len(ch.text)
        ch.end_offset = offset  # exclusive of the joint "\n" below
        offset += 1  # the "\n" joining chapters in the monotonic char stream

    doc_len = max(0, offset - 1)
    log.info("parsed '%s': %d chapters, %d chars", title, len(chapters), doc_len)
    return ParsedDoc(title=title, author=author, chapters=chapters, doc_len=doc_len)


def parse_document(data: bytes, file_name: str) -> ParsedDoc:
    name = file_name.lower()
    if name.endswith(".fb2"):
        return parse_fb2(data)
    if name.endswith(".epub"):
        return parse_epub(data)
    if name.endswith(".zip") and ".fb2" in name:  # the common .fb2.zip packaging
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            inner = next((n for n in z.namelist() if n.lower().endswith(".fb2")), None)
            if inner:
                return parse_fb2(z.read(inner))
    raise ValueError(f"Неподдерживаемый формат файла: {file_name} (жду .fb2, .fb2.zip или .epub)")
