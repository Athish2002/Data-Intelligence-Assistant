"""
dia/dictionary_store.py
─────────────────────────
Local SQLite persistence for the data dictionary (business glossary). This is
the one thing in the app that survives across sessions — everything else
(the uploaded dataset, its rows, chat history, the retrieval index built from
them) stays in-memory only, exactly like before this feature existed.

Callers must only write here after confirming
dia.compliance.scan_dataset_privacy() found nothing — see
dia.ingestion.data_dictionary.DataDictionarySource, which annotates that
decision as meta["safe_to_persist"] but deliberately does not act on it
itself, keeping "ingest" and "persist" cleanly separated.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DICTIONARY_DB_PATH

__all__ = ["save_entries", "load_entries", "delete_entry", "clear_all", "count_entries"]

_DDL = """
CREATE TABLE IF NOT EXISTS dictionary_terms (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    term         TEXT NOT NULL,
    definition   TEXT NOT NULL,
    source_label TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE (term COLLATE NOCASE)
);
"""


def _connect() -> sqlite3.Connection:
    path = Path(DICTIONARY_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(_DDL)
    return conn


def save_entries(entries: list[dict[str, str]], source_label: str) -> int:
    """
    Insert or update (by case-insensitive term) each entry. Returns the number
    of entries written. Caller must have already confirmed the PII gate passed.
    """
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    with _connect() as conn:
        for e in entries:
            term = str(e.get("term", "")).strip()
            definition = str(e.get("definition", "")).strip()
            if not term:
                continue
            conn.execute(
                """
                INSERT INTO dictionary_terms (term, definition, source_label, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(term) DO UPDATE SET
                    definition=excluded.definition,
                    source_label=excluded.source_label,
                    updated_at=excluded.updated_at
                """,
                (term, definition, source_label, now, now),
            )
            written += 1
    return written


def load_entries() -> list[dict[str, Any]]:
    """Return every persisted term, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT term, definition, source_label, created_at, updated_at "
            "FROM dictionary_terms ORDER BY id ASC"
        ).fetchall()
    cols = ("term", "definition", "source_label", "created_at", "updated_at")
    return [dict(zip(cols, row, strict=True)) for row in rows]


def delete_entry(term: str) -> bool:
    """Delete one term (case-insensitive). Returns True if a row was removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM dictionary_terms WHERE term = ? COLLATE NOCASE", (term,))
        return cur.rowcount > 0


def clear_all() -> int:
    """Delete every persisted term. Returns the number of rows removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM dictionary_terms")
        return cur.rowcount


def count_entries() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM dictionary_terms").fetchone()[0])
