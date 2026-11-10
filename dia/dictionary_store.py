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

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS dictionary_terms (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    term         TEXT NOT NULL,
    definition   TEXT NOT NULL,
    source_label TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE (tenant_id, term COLLATE NOCASE)
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_dict_tenant ON dictionary_terms(tenant_id);
"""

_DDL = _TABLE_DDL + _INDEX_DDL


def _connect() -> sqlite3.Connection:
    path = Path(DICTIONARY_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='dictionary_terms'")
    row = cur.fetchone()
    if row is None:
        conn.executescript(_DDL)
    elif "UNIQUE (tenant_id, term" not in row[0] and "UNIQUE(tenant_id, term" not in row[0]:
        # Migrate legacy table with single-tenant constraint to multi-tenant
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS dictionary_terms_new (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id    TEXT NOT NULL DEFAULT 'default',
            term         TEXT NOT NULL,
            definition   TEXT NOT NULL,
            source_label TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            UNIQUE (tenant_id, term COLLATE NOCASE)
        );
        INSERT OR IGNORE INTO dictionary_terms_new (tenant_id, term, definition, source_label, created_at, updated_at)
        SELECT COALESCE(tenant_id, 'default'), term, definition, source_label, created_at, updated_at FROM dictionary_terms;
        DROP TABLE dictionary_terms;
        ALTER TABLE dictionary_terms_new RENAME TO dictionary_terms;
        CREATE INDEX IF NOT EXISTS idx_dict_tenant ON dictionary_terms(tenant_id);
        """)
    else:
        conn.execute(_INDEX_DDL)
    return conn


def save_entries(entries: list[dict[str, str]], source_label: str, tenant_id: str = "default") -> int:
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
                INSERT INTO dictionary_terms (tenant_id, term, definition, source_label, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, term) DO UPDATE SET
                    definition=excluded.definition,
                    source_label=excluded.source_label,
                    updated_at=excluded.updated_at
                """,
                (tenant_id, term, definition, source_label, now, now),
            )
            written += 1
    return written


def load_entries(tenant_id: str = "default") -> list[dict[str, Any]]:
    """Return every persisted term, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT term, definition, source_label, created_at, updated_at "
            "FROM dictionary_terms WHERE tenant_id = ? ORDER BY id ASC",
            (tenant_id,),
        ).fetchall()
    cols = ("term", "definition", "source_label", "created_at", "updated_at")
    return [dict(zip(cols, row, strict=True)) for row in rows]


def delete_entry(term: str, tenant_id: str = "default") -> bool:
    """Delete one term (case-insensitive). Returns True if a row was removed."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM dictionary_terms WHERE tenant_id = ? AND term = ? COLLATE NOCASE",
            (tenant_id, term),
        )
        return cur.rowcount > 0


def clear_all(tenant_id: str = "default") -> int:
    """Delete every persisted term. Returns the number of rows removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM dictionary_terms WHERE tenant_id = ?", (tenant_id,))
        return cur.rowcount


def count_entries(tenant_id: str = "default") -> int:
    with _connect() as conn:
        return int(
            conn.execute("SELECT COUNT(*) FROM dictionary_terms WHERE tenant_id = ?", (tenant_id,)).fetchone()[0]
        )
