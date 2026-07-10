"""SQLite state for the long-reader: documents, chapters, parent chunks, reading positions, ingest
jobs. This is the reader's OWN store — the bot's existing state.json is not touched. Parent chunk
TEXT lives here (small-to-big fetches it by id); leaf chunk text lives in Qdrant payloads.

All functions are synchronous sqlite3 (fast, single-user); call the write-heavy ones via
asyncio.to_thread from handlers if they ever show up in traces — in practice every call here is
milliseconds on this scale (one book = hundreds of rows).
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "reader.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id      TEXT PRIMARY KEY,           -- sha256 of the file content: re-upload == same doc
    title       TEXT,
    author      TEXT,
    file_name   TEXT,
    fmt         TEXT,                       -- fb2 | epub
    doc_len     INTEGER,                    -- total chars of normalized plain text (offset unit)
    n_chapters  INTEGER,
    tier        TEXT DEFAULT 'low',         -- highest tier ingested so far: low | medium
    status      TEXT DEFAULT 'new',         -- new | ingesting | ready | error
    error       TEXT,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS chapters (
    doc_id       TEXT,
    chapter_idx  INTEGER,
    title        TEXT,
    start_offset INTEGER,
    end_offset   INTEGER,                   -- max offset of the chapter's text (spoiler math)
    summary      TEXT,                      -- filled by the medium tier; NULL on low
    PRIMARY KEY (doc_id, chapter_idx)
);

CREATE TABLE IF NOT EXISTS parents (
    parent_id    TEXT PRIMARY KEY,
    doc_id       TEXT,
    chapter_idx  INTEGER,
    start_offset INTEGER,
    end_offset   INTEGER,
    text         TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    user_id    INTEGER,
    doc_id     TEXT,
    offset     INTEGER,                     -- unlocked up to and including this char offset
    updated_at REAL,
    PRIMARY KEY (user_id, doc_id)
);

-- Which document /ask targets in a given chat. One active doc per chat keeps the UX command-free.
CREATE TABLE IF NOT EXISTS active_doc (
    chat_id INTEGER PRIMARY KEY,
    doc_id  TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     TEXT,
    chat_id    INTEGER,
    tier       TEXT,
    stage      TEXT,                        -- parse | embed | summarize | done | error
    progress   TEXT,                        -- free-form progress line shown to the user
    status     TEXT DEFAULT 'running',      -- running | done | error
    error      TEXT,
    created_at REAL,
    updated_at REAL
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


# --- docs ---------------------------------------------------------------------------------------

def get_doc(doc_id: str) -> sqlite3.Row | None:
    with _conn() as conn:
        return conn.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,)).fetchone()


def find_doc_by_prefix(prefix: str) -> sqlite3.Row | None:
    """Telegram callback_data is capped at 64 bytes, so buttons carry a doc_id prefix, not the
    full sha256. 16 hex chars = collision-proof at 'a few books' scale."""
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM docs WHERE doc_id LIKE ? LIMIT 1", (prefix + "%",)
        ).fetchone()


def list_docs() -> list[sqlite3.Row]:
    with _conn() as conn:
        return conn.execute("SELECT * FROM docs ORDER BY created_at").fetchall()


def upsert_doc(doc_id: str, **fields) -> None:
    fields.setdefault("created_at", time.time())
    with _conn() as conn:
        existing = conn.execute("SELECT doc_id FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if existing:
            sets = ", ".join(f"{k}=?" for k in fields if k != "created_at")
            vals = [v for k, v in fields.items() if k != "created_at"]
            if sets:
                conn.execute(f"UPDATE docs SET {sets} WHERE doc_id=?", (*vals, doc_id))
        else:
            cols = ["doc_id", *fields]
            conn.execute(
                f"INSERT INTO docs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                (doc_id, *fields.values()),
            )


def set_doc_status(doc_id: str, status: str, error: str | None = None) -> None:
    with _conn() as conn:
        conn.execute("UPDATE docs SET status=?, error=? WHERE doc_id=?", (status, error, doc_id))


def set_doc_tier(doc_id: str, tier: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE docs SET tier=? WHERE doc_id=?", (tier, doc_id))


# --- chapters / parents -------------------------------------------------------------------------

def save_chapters(doc_id: str, chapters: list[dict]) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM chapters WHERE doc_id=?", (doc_id,))
        conn.executemany(
            "INSERT INTO chapters (doc_id, chapter_idx, title, start_offset, end_offset) VALUES (?,?,?,?,?)",
            [(doc_id, c["chapter_idx"], c["title"], c["start_offset"], c["end_offset"]) for c in chapters],
        )


def get_chapters(doc_id: str) -> list[sqlite3.Row]:
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM chapters WHERE doc_id=? ORDER BY chapter_idx", (doc_id,)
        ).fetchall()


def set_chapter_summary(doc_id: str, chapter_idx: int, summary: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE chapters SET summary=? WHERE doc_id=? AND chapter_idx=?",
            (summary, doc_id, chapter_idx),
        )


def save_parents(doc_id: str, parents: list[dict]) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM parents WHERE doc_id=?", (doc_id,))
        conn.executemany(
            "INSERT INTO parents (parent_id, doc_id, chapter_idx, start_offset, end_offset, text) "
            "VALUES (?,?,?,?,?,?)",
            [
                (p["parent_id"], doc_id, p["chapter_idx"], p["start_offset"], p["end_offset"], p["text"])
                for p in parents
            ],
        )


def get_parents(parent_ids: list[str]) -> list[sqlite3.Row]:
    if not parent_ids:
        return []
    q = ",".join("?" * len(parent_ids))
    with _conn() as conn:
        return conn.execute(f"SELECT * FROM parents WHERE parent_id IN ({q})", parent_ids).fetchall()


# --- positions / active doc ---------------------------------------------------------------------

def get_position(user_id: int, doc_id: str) -> int:
    """Unlocked-up-to char offset; 0 = nothing read yet (safe default — the whole book is spoilers)."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT offset FROM positions WHERE user_id=? AND doc_id=?", (user_id, doc_id)
        ).fetchone()
        return row["offset"] if row else 0


def set_position(user_id: int, doc_id: str, offset: int) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO positions (user_id, doc_id, offset, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id, doc_id) DO UPDATE SET offset=excluded.offset, updated_at=excluded.updated_at",
            (user_id, doc_id, offset, time.time()),
        )


def get_active_doc(chat_id: int) -> str | None:
    with _conn() as conn:
        row = conn.execute("SELECT doc_id FROM active_doc WHERE chat_id=?", (chat_id,)).fetchone()
        return row["doc_id"] if row else None


def set_active_doc(chat_id: int, doc_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO active_doc (chat_id, doc_id) VALUES (?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET doc_id=excluded.doc_id",
            (chat_id, doc_id),
        )


# --- jobs ---------------------------------------------------------------------------------------

def create_job(doc_id: str, chat_id: int, tier: str) -> int:
    now = time.time()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (doc_id, chat_id, tier, stage, progress, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, chat_id, tier, "parse", "", now, now),
        )
        return cur.lastrowid


def update_job(job_id: int, stage: str | None = None, progress: str | None = None,
               status: str | None = None, error: str | None = None) -> None:
    sets, vals = ["updated_at=?"], [time.time()]
    for col, val in (("stage", stage), ("progress", progress), ("status", status), ("error", error)):
        if val is not None:
            sets.append(f"{col}=?")
            vals.append(val)
    with _conn() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", (*vals, job_id))


def get_running_job(doc_id: str) -> sqlite3.Row | None:
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE doc_id=? AND status='running' ORDER BY job_id DESC", (doc_id,)
        ).fetchone()
