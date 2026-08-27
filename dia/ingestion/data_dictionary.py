"""
dia/ingestion/data_dictionary.py
──────────────────────────────────
Ingestion source: a business-glossary CSV (term, definition) that grounds
the RAG chat copilot alongside a dataset's own schema/profiling chunks.

Deliberately does NOT persist anything itself — dia.ingestion never touches
disk (see local_csv.py/url_ingestion.py). It only parses, validates shape,
and runs the PII gate, annotating meta["safe_to_persist"]. The caller
(dia.dictionary_store, invoked from app.py / api/server.py) decides whether
to actually write to disk based on that flag, keeping "ingest" and
"persist" cleanly separated — the same separation IngestionResult already
enforces for every other source.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..compliance import scan_dataset_privacy
from ..exceptions import ValidationError
from ..validators import validate_magic_bytes, validate_uploaded_file
from .base import IngestionResult, IngestionSource
from .local_csv import _parse_csv

__all__ = ["DataDictionarySource"]

_REQUIRED_COLS = {"term", "definition"}


class DataDictionarySource(IngestionSource):
    """Parses and PII-gates a business-glossary CSV (columns: term, definition)."""

    def load(self, uploaded_file: Any = None, **kwargs: Any) -> IngestionResult:
        validate_uploaded_file(uploaded_file)
        raw_bytes: bytes = uploaded_file.read()
        validate_magic_bytes(raw_bytes)

        df = _parse_csv(raw_bytes)
        df.columns = pd.Index(str(c).strip().lower() for c in df.columns)

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValidationError(
                f"Data dictionary CSV must have columns {sorted(_REQUIRED_COLS)}; "
                f"missing {sorted(missing)}. Found: {list(df.columns)}."
            )

        df = df[["term", "definition"]].dropna(subset=["term"]).copy()
        df["term"] = df["term"].astype(str).str.strip()
        df["definition"] = df["definition"].astype(str).str.strip()
        df = df[df["term"] != ""].reset_index(drop=True)
        if df.empty:
            raise ValidationError("Data dictionary CSV contains no usable term/definition rows.")

        pii_report = scan_dataset_privacy(df)
        meta: dict[str, Any] = {
            "n_terms": len(df),
            "safe_to_persist": pii_report["pii_columns_count"] == 0,
            "pii_gate_report": pii_report,
        }
        return IngestionResult(
            df=df,
            source_label=f"Data Dictionary — {getattr(uploaded_file, 'name', 'file')}",
            meta=meta,
        )
