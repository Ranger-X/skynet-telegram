"""Manga/comics as a long-reader document: CBZ (zip of page images) and image-only PDF.

The page is everything here: offset unit = PAGE NUMBER (1-based, monotonic — the spoiler filter
works unchanged), leaf chunk = one page's extracted text (bubbles + panel gist), context assembly
pulls neighboring pages by offset range instead of the books' parent windows. Chapters come from
the CBZ's top-level folders (scanlation convention) or fall back to virtual chapters of ~20 pages;
they feed /pos, /chapters and medium-tier summaries exactly like book chapters.

Pages are re-extracted lazily from the stored source file (reader_files/) — nothing rendered is
persisted.
"""

import io
import logging
import os
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("t800.reader")

PAGE_MAX_SIDE = 1400        # bubbles must stay readable; below ~1100px scanlation text smears
JPEG_QUALITY = 87
VIRTUAL_CHAPTER_PAGES = 20  # when the archive has no folder structure
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

MANGA_SUFFIXES = (".cbz", ".cbr", ".pdf")

# CBR = RAR: no stdlib support, extract once via 7-Zip into a sibling cache dir and serve pages
# from files. 7z handles both rar4 and rar5.
SEVEN_ZIP = os.environ.get("SEVEN_ZIP_EXE", r"C:\Program Files\7-Zip\7z.exe")


@dataclass
class MangaChapter:
    title: str
    first_page: int  # 1-based, inclusive
    last_page: int


@dataclass
class MangaDoc:
    title: str
    n_pages: int
    chapters: list[MangaChapter]
    fmt: str                    # cbz | pdf
    source_path: str            # where page_jpeg() re-reads from


def _natural_key(s: str) -> list:
    """'ch2/p10.jpg' after 'ch2/p9.jpg', not before — numeric-aware ordering."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _cbz_page_names(path: str) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(IMAGE_EXTS) and not n.startswith("__MACOSX")
            and "/." not in n and not n.rsplit("/", 1)[-1].startswith(".")
        ]
    return sorted(names, key=_natural_key)


def _chapters_from_names(names: list[str]) -> list[MangaChapter]:
    """Top-level folder = chapter (scanlation packs are usually ch01/, ch02/, ...). A flat archive
    (or a single folder) gets virtual chapters so summaries and /pos have units to work with."""
    groups: list[tuple[str, int]] = []  # (folder, count) in page order
    for n in names:
        folder = n.split("/")[0] if "/" in n else ""
        if groups and groups[-1][0] == folder:
            groups[-1] = (folder, groups[-1][1] + 1)
        else:
            groups.append((folder, 1))

    chapters: list[MangaChapter] = []
    if len(groups) > 1:
        page = 1
        for folder, count in groups:
            chapters.append(MangaChapter(title=folder or "Без раздела", first_page=page, last_page=page + count - 1))
            page += count
        return chapters

    total = len(names)
    for start in range(1, total + 1, VIRTUAL_CHAPTER_PAGES):
        end = min(start + VIRTUAL_CHAPTER_PAGES - 1, total)
        chapters.append(MangaChapter(title=f"Страницы {start}–{end}", first_page=start, last_page=end))
    return chapters


def _cbr_cache_dir(path: str) -> Path:
    return Path(path + "_pages")


def _cbr_page_names(path: str) -> list[str]:
    """Extract-once cache: the CBR is unpacked next to itself on first open; pages are then plain
    files. Relative posix paths keep the chapter-from-folder logic shared with CBZ."""
    cache = _cbr_cache_dir(path)
    if not cache.is_dir() or not any(cache.rglob("*")):
        cache.mkdir(exist_ok=True)
        proc = subprocess.run(
            [SEVEN_ZIP, "x", "-y", f"-o{cache}", str(path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise ValueError(f"7-Zip не смог распаковать CBR: {proc.stderr[:200]}")
    names = [
        p.relative_to(cache).as_posix()
        for p in cache.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    return sorted(names, key=_natural_key)


def open_manga(path: str, file_name: str) -> MangaDoc:
    name = file_name.lower()
    title = re.sub(r"[_\-.]+", " ", file_name.rsplit(".", 1)[0]).strip()
    if name.endswith(".cbz"):
        pages = _cbz_page_names(path)
        if not pages:
            raise ValueError("в CBZ не нашлось ни одной страницы-изображения")
        return MangaDoc(title=title, n_pages=len(pages), chapters=_chapters_from_names(pages),
                        fmt="cbz", source_path=path)
    if name.endswith(".cbr"):
        pages = _cbr_page_names(path)
        if not pages:
            raise ValueError("в CBR не нашлось ни одной страницы-изображения")
        return MangaDoc(title=title, n_pages=len(pages), chapters=_chapters_from_names(pages),
                        fmt="cbr", source_path=path)
    if name.endswith(".pdf"):
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            n = doc.page_count
            # A PDF with a real text layer is a BOOK, not a comic — refuse so it isn't burned
            # through the expensive vision path by mistake.
            text_chars = sum(len(doc[i].get_text()) for i in range(min(5, n)))
            if text_chars > 500:
                raise ValueError(
                    "этот PDF — текстовый (книга), а не комикс; текстовые PDF пока не поддерживаются"
                )
        chapters = _chapters_from_names([f"p{i:05d}" for i in range(n)])  # always virtual for PDF
        return MangaDoc(title=title, n_pages=n, chapters=chapters, fmt="pdf", source_path=path)
    raise ValueError(f"не манга/комикс: {file_name} (жду .cbz или .pdf)")


def page_jpeg(doc: MangaDoc, page_idx: int) -> bytes:
    """1-based page -> normalized JPEG (≤PAGE_MAX_SIDE on the long side). Blocking; call via
    asyncio.to_thread."""
    from PIL import Image

    if doc.fmt == "cbz":
        names = _cbz_page_names(doc.source_path)
        with zipfile.ZipFile(doc.source_path) as z:
            raw = z.read(names[page_idx - 1])
        img = Image.open(io.BytesIO(raw))
    elif doc.fmt == "cbr":
        names = _cbr_page_names(doc.source_path)
        img = Image.open(_cbr_cache_dir(doc.source_path) / names[page_idx - 1])
    else:  # pdf
        import fitz

        with fitz.open(doc.source_path) as pdf:
            page = pdf[page_idx - 1]
            zoom = PAGE_MAX_SIDE / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    img = img.convert("RGB")
    scale = PAGE_MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
