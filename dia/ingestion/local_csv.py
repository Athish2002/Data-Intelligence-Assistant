"""
dia/ingestion/local_csv.py
──────────────────────────
Ingestion source: local CSV file upload (Streamlit UploadedFile).
Replaces the old dia/data_loader.py with a typed, validated source.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

from ..config import MAX_FILE_BYTES
from ..exceptions import DataLoadError, ValidationError
from ..validators import (
    sanitise_column_names,
    validate_dataframe_shape,
    validate_magic_bytes,
    validate_uploaded_file,
)
from .base import IngestionResult, IngestionSource

log = logging.getLogger("dia.ingestion.local_csv")

_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


class LocalCSVSource(IngestionSource):
    """Load a CSV from a Streamlit UploadedFile object."""

    def load(self, uploaded_file: Any = None, **kwargs: Any) -> IngestionResult:
        """
        Validate, read, and return a DataFrame from an uploaded CSV file.

        Parameters
        ----------
        uploaded_file : Streamlit UploadedFile (or any BinaryIO)

        Returns
        -------
        IngestionResult

        Raises
        ------
        ValidationError  – file type, size, shape problems
        DataLoadError    – encoding / parse failure
        """
        # 1. Structural validation
        validate_uploaded_file(uploaded_file)
        log.info("Reading uploaded file: %s", getattr(uploaded_file, "name", "unknown"))

        # 2. Read raw bytes
        raw_bytes: bytes = uploaded_file.read()
        file_size = len(raw_bytes)

        if file_size > MAX_FILE_BYTES:
            max_mb = MAX_FILE_BYTES / (1024 ** 2)
            actual_mb = file_size / (1024 ** 2)
            raise ValidationError(
                f"File is {actual_mb:.1f} MB which exceeds the "
                f"{max_mb:.0f} MB upload limit. "
                "Please sample your data first."
            )

        # 3. Magic-byte content check
        validate_magic_bytes(raw_bytes)

        # 4. Robust parsing and sanitization (handles malformed lines, encodings, dirty currencies, null tokens)
        from ..data_sanitizer import robust_parse_csv_bytes
        df, sanitize_report = robust_parse_csv_bytes(raw_bytes)

        # 5. Sanitise column names (strip null bytes / control chars)
        df.columns = pd.Index(sanitise_column_names(list(df.columns)))

        # 6. Shape sanity checks
        validate_dataframe_shape(df)

        log.info(
            "Loaded CSV: %d rows × %d columns (%.2f MB)",
            df.shape[0], df.shape[1], file_size / (1024 ** 2),
        )

        meta: dict[str, Any] = {
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 ** 2), 2),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "encoding": sanitize_report.get("detected_encoding", "utf-8"),
            "delimiter": sanitize_report.get("detected_delimiter", ","),
            "sanitize_report": sanitize_report,
        }

        return IngestionResult(
            df=df,
            source_label=f"Local CSV — {getattr(uploaded_file, 'name', 'file')}",
            meta=meta,
        )


def _parse_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Try multiple encodings and return the first successful parse."""
    buf = io.BytesIO(raw_bytes)
    last_error: Exception | None = None

    for enc in _ENCODINGS:
        try:
            buf.seek(0)
            df = pd.read_csv(buf, encoding=enc, low_memory=False)
            log.debug("CSV parsed with encoding=%s", enc)
            return df
        except UnicodeDecodeError as exc:
            log.debug("Encoding %s failed: %s", enc, exc)
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            log.warning("CSV parse error with encoding=%s: %s", enc, exc)
            last_error = exc

    raise DataLoadError(
        "Could not parse the CSV file. "
        "Please ensure it is UTF-8 or Latin-1 encoded and is a valid CSV. "
        f"Last error: {last_error}"
    )
