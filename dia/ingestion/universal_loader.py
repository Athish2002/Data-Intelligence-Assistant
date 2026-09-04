"""
dia/ingestion/universal_loader.py
─────────────────────────────────
Universal, resilient multi-format data ingestion engine.
Supports:
- CSV, TSV (with automatic delimiter sniffing: comma, semicolon, tab, pipe)
- Apache Parquet columnar binary
- JSON & JSON Lines (records, split, normalized)
- Excel spreadsheets (.xlsx, .xls)
- Multi-encoding resilience (utf-8, utf-8-sig BOM stripping, latin-1, cp1252)
- Header normalization and column deduplication
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from ..config import MAX_FILE_BYTES
from ..exceptions import DataLoadError, ValidationError
from .base import IngestionResult, IngestionSource

log = logging.getLogger("dia.ingestion.universal_loader")

_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1")


def detect_delimiter(sample_text: str) -> str:
    """
    Intelligently detects delimiter (comma, semicolon, tab, pipe)
    by analyzing the consistency of field counts across lines.
    """
    sample_lines = [line for line in sample_text.splitlines()[:20] if line.strip()]
    if not sample_lines:
        return ","

    # Try csv.Sniffer first
    try:
        sniffer = csv.Sniffer()
        join_str = chr(10)
        dialect = sniffer.sniff(join_str.join(sample_lines), delimiters=[",", ";", "\t", "|"])
        if dialect.delimiter in [",", ";", "\t", "|"]:
            return dialect.delimiter
    except Exception:
        pass

    # Fallback heuristic: count delimiter consistency across sample lines
    candidates = [",", ";", "	", "|"]
    best_delim = ","
    max_score = -1

    for cand in candidates:
        counts = [line.count(cand) for line in sample_lines]
        if len(counts) > 1 and all(c > 0 for c in counts):
            # Check consistency (low standard deviation / equal counts)
            if len(set(counts)) == 1:
                score = counts[0] * 10
            else:
                score = sum(counts) / len(counts)
            if score > max_score:
                max_score = score
                best_delim = cand

    return best_delim


def sanitize_column_headers(columns: list[Any]) -> list[str]:
    """
    Normalizes column headers: strips whitespace, replaces newlines/tabs with spaces,
    ensures valid string names, and deduplicates repeated column names.
    """
    cleaned: list[str] = []
    seen: dict[str, int] = {}

    for idx, c in enumerate(columns):
        if c is None or pd.isna(c) or str(c).strip() == "":
            name = f"column_{idx + 1}"
        else:
            name = re.sub(r"\s+", " ", str(c)).strip()
            if not name:
                name = f"column_{idx + 1}"

        if name in seen:
            seen[name] += 1
            unique_name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
            unique_name = name

        cleaned.append(unique_name)

    return cleaned


class UniversalLoader(IngestionSource):
    """Universal multi-format data ingestion engine."""

    def load(self, raw_bytes: bytes, filename: str, **kwargs: Any) -> IngestionResult:
        if not raw_bytes:
            raise ValidationError("Uploaded file is empty.")

        if len(raw_bytes) > MAX_FILE_BYTES:
            max_mb = MAX_FILE_BYTES / (1024 ** 2)
            actual_mb = len(raw_bytes) / (1024 ** 2)
            raise ValidationError(
                f"File size exceeds limit ({actual_mb:.1f} MB > {max_mb:.0f} MB). "
                f"Increase DIA_MAX_FILE_MB to load larger datasets."
            )

        _, ext = os.path.splitext(filename.lower())
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        df: pd.DataFrame
        meta: dict[str, Any] = {
            "filename": filename,
            "file_size_bytes": len(raw_bytes),
            "file_format": ext.lstrip("."),
        }

        try:
            # ── 1. Apache Parquet ─────────────────────────────────────────────
            if ext == ".parquet":
                df = pd.read_parquet(io.BytesIO(raw_bytes))
                meta["encoding"] = "binary"
                meta["delimiter"] = "N/A"

            # ── 2. Excel Spreadsheet ──────────────────────────────────────────
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(io.BytesIO(raw_bytes))
                meta["encoding"] = "binary"
                meta["delimiter"] = "N/A"

            # ── 3. JSON / JSON Lines ──────────────────────────────────────────
            elif ext in (".json", ".jsonl"):
                text = None
                used_encoding = "utf-8"
                for enc in _ENCODINGS:
                    try:
                        text = raw_bytes.decode(enc)
                        used_encoding = enc
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue

                if text is None:
                    raise DataLoadError(f"Could not decode JSON file '{filename}' with supported encodings.")

                meta["encoding"] = used_encoding
                meta["delimiter"] = "N/A"

                try:
                    df = pd.read_json(io.StringIO(text), orient="records")
                except Exception:
                    try:
                        df = pd.read_json(io.StringIO(text), lines=True)
                    except Exception:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            df = pd.DataFrame(parsed)
                        elif isinstance(parsed, dict):
                            # Try common json wrapper keys or normalize
                            if "data" in parsed and isinstance(parsed["data"], list):
                                df = pd.DataFrame(parsed["data"])
                            elif "records" in parsed and isinstance(parsed["records"], list):
                                df = pd.DataFrame(parsed["records"])
                            else:
                                df = pd.json_normalize(parsed)
                        else:
                            raise DataLoadError(f"Unsupported JSON structure in '{filename}'")

            # ── 4. Delimited Text (CSV, TSV) ──────────────────────────────────
            else:
                text = None
                used_encoding = "utf-8"
                for enc in _ENCODINGS:
                    try:
                        text = raw_bytes.decode(enc)
                        used_encoding = enc
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue

                if text is None:
                    raise DataLoadError(
                        f"Unable to decode '{filename}'. Tried encodings: {', '.join(_ENCODINGS)}"
                    )

                delimiter = "	" if ext == ".tsv" else detect_delimiter(text[:8192])
                meta["encoding"] = used_encoding
                meta["delimiter"] = "\t" if delimiter == "	" else delimiter

                # Try parsing with engine='c', fallback to engine='python'
                try:
                    df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="c", low_memory=False)
                except Exception:
                    df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python")

        except Exception as exc:
            if isinstance(exc, (ValidationError, DataLoadError)):
                raise
            raise DataLoadError(f"Failed to load dataset '{filename}': {str(exc)}") from exc

        # ── 5. Sanitize Headers & Validate Shape ──────────────────────────────
        df.columns = sanitize_column_headers(list(df.columns))

        if df.empty or df.shape[0] == 0:
            raise ValidationError(f"Dataset '{filename}' is empty (0 rows).")

        if df.shape[1] < 2:
            raise ValidationError(
                f"Dataset '{filename}' has fewer than 2 columns ({df.shape[1]} found). "
                f"Machine learning workflows require at least one feature column and one target column."
            )

        meta["n_rows"] = int(df.shape[0])
        meta["n_cols"] = int(df.shape[1])
        meta["columns"] = list(df.columns)

        log.info(
            "UniversalLoader successfully ingested '%s': %d rows, %d cols, format=%s, encoding=%s, delim=%s",
            filename, df.shape[0], df.shape[1], meta["file_format"], meta["encoding"], meta.get("delimiter")
        )

        return IngestionResult(
            df=df,
            source_label=f"File Upload ({meta['file_format'].upper()}): {filename}",
            meta=meta,
        )
