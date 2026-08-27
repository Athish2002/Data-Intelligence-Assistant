"""
dia/validators.py
─────────────────
Centralised input validation layer.

All validators raise `ValidationError` on failure so callers
can catch one typed exception instead of bare ValueError.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from .config import GOAL_MAX_CHARS, MAX_COLS, MAX_MODELS, MAX_ROWS
from .exceptions import ValidationError

if TYPE_CHECKING:
    pass

__all__ = [
    "validate_uploaded_file",
    "validate_goal_text",
    "validate_target_column",
    "validate_model_selection",
    "validate_dataframe_shape",
    "validate_sql_identifier",
    "sanitise_column_names",
    "detect_pii_columns",
]

# A bare identifier or one schema-qualified level (e.g. "orders" or "public.orders").
# Deliberately does not allow quoting/backticks/semicolons/whitespace — table names
# are interpolated directly into SQL (SQL doesn't support parameter placeholders for
# identifiers), so this is the injection guard for that one field. The free-text
# `query` field is a different, intentionally-arbitrary-SQL feature and is not
# covered by this validator.
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

# ─── PII tokens ───────────────────────────────────────────────────────────────

_PII_TOKENS: frozenset[str] = frozenset({
    "email", "mail", "phone", "mobile", "ssn", "social", "security",
    "passport", "national", "dob", "birth", "birthday", "age",
    "address", "street", "zip", "postcode", "credit", "card",
    "cvv", "iban", "account", "ip", "latitude", "longitude", "location",
    "gender", "race", "ethnicity", "religion", "salary",
})

# ─── Dangerous CSV injection prefixes ────────────────────────────────────────

_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# ─── Allowed MIME types for uploads ──────────────────────────────────────────

_ALLOWED_MIMETYPES: frozenset[str] = frozenset({
    "text/csv",
    "text/plain",
    "application/csv",
    "application/vnd.ms-excel",
    "",  # some browsers send empty type for .csv
})


# ─── File validation ──────────────────────────────────────────────────────────

def validate_uploaded_file(uploaded_file: Any) -> None:
    """
    Validate a Streamlit UploadedFile before reading.

    Checks:
    - File object is not None
    - MIME type is in the allowed set
    - File has a non-empty name ending in .csv

    Raises
    ------
    ValidationError
    """
    if uploaded_file is None:
        raise ValidationError("No file was uploaded.")

    name: str = getattr(uploaded_file, "name", "") or ""
    if not name.lower().endswith(".csv"):
        raise ValidationError(
            f"File '{html.escape(name)}' does not have a .csv extension. "
            "Only CSV files are supported."
        )

    mime: str = getattr(uploaded_file, "type", "") or ""
    if mime and mime not in _ALLOWED_MIMETYPES:
        raise ValidationError(
            f"Unexpected file type '{html.escape(mime)}'. "
            "Please upload a CSV file."
        )


def validate_magic_bytes(raw_bytes: bytes) -> None:
    """
    Reject files whose first bytes look like known binary formats.

    Raises
    ------
    ValidationError  – if the file starts with a binary magic signature.
    """
    # Known binary magic signatures: PK (zip/xlsx), PDF, ELF, PE, PNG, JPEG
    binary_signatures = [
        (b"PK\x03\x04", "ZIP / XLSX / DOCX"),
        (b"%PDF",       "PDF"),
        (b"\x7fELF",   "ELF binary"),
        (b"MZ",         "Windows executable"),
        (b"\x89PNG",   "PNG image"),
        (b"\xff\xd8\xff", "JPEG image"),
        (b"GIF8",      "GIF image"),
        (b"\x1f\x8b",  "gzip archive"),
        (b"BZh",       "bzip2 archive"),
    ]
    header = raw_bytes[:8]
    for sig, label in binary_signatures:
        if header.startswith(sig):
            raise ValidationError(
                f"The uploaded file appears to be a {label}, not a CSV. "
                "Please upload a plain CSV text file."
            )


# ─── Goal text validation ─────────────────────────────────────────────────────

def validate_goal_text(goal: str) -> str:
    """
    Sanitise and validate the user's plain-text ML goal.

    Returns the cleaned goal string.

    Raises
    ------
    ValidationError  – if the goal is empty or too long.
    """
    if not goal or not goal.strip():
        raise ValidationError("Please enter a prediction goal before running analysis.")

    if len(goal) > GOAL_MAX_CHARS:
        raise ValidationError(
            f"Goal text is {len(goal):,} characters, which exceeds the "
            f"{GOAL_MAX_CHARS:,} character limit. Please be more concise."
        )

    # Strip null bytes and control characters (keep newlines and tabs as spaces)
    cleaned = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf") else ch
        for ch in goal
        if ch != "\x00"
    ).strip()

    if not cleaned:
        raise ValidationError("Goal text contains only non-printable characters.")

    return cleaned


# ─── Target column validation ─────────────────────────────────────────────────

def validate_target_column(col: str, df: pd.DataFrame, min_non_null: int = 10) -> None:
    """
    Verify the target column exists and has enough non-null values.

    Raises
    ------
    ValidationError
    """
    if col not in df.columns:
        raise ValidationError(
            f"Target column '{col}' was not found in the dataset. "
            f"Available columns: {list(df.columns[:10])}"
        )

    non_null = df[col].notna().sum()
    if non_null < min_non_null:
        raise ValidationError(
            f"Target column '{col}' has only {non_null} non-null value(s). "
            f"At least {min_non_null} non-null values are required for training."
        )


# ─── DataFrame shape validation ───────────────────────────────────────────────

def validate_dataframe_shape(df: pd.DataFrame) -> None:
    """
    Enforce row / column hard caps to prevent resource exhaustion.

    Raises
    ------
    ValidationError
    """
    n_rows, n_cols = df.shape

    if n_rows == 0:
        raise ValidationError("The uploaded dataset contains no rows.")

    if n_cols < 2:
        raise ValidationError(
            f"The dataset has only {n_cols} column(s). "
            "At least 2 columns (1 feature + 1 target) are required."
        )

    if n_rows > MAX_ROWS:
        raise ValidationError(
            f"Dataset has {n_rows:,} rows, exceeding the {MAX_ROWS:,} row limit. "
            "Please sample your data before uploading."
        )

    if n_cols > MAX_COLS:
        raise ValidationError(
            f"Dataset has {n_cols:,} columns, exceeding the {MAX_COLS:,} column limit."
        )


# ─── SQL identifier validation ────────────────────────────────────────────────

def validate_sql_identifier(name: str, field_label: str = "table name") -> str:
    """
    Validate that *name* is safe to interpolate directly into SQL as an
    identifier (table/schema name) — SQL has no parameter-placeholder syntax
    for identifiers, so this is the injection guard for that one field.

    Returns the stripped name unchanged. Only bare identifiers or one
    schema-qualified level (e.g. "orders" or "public.orders") are accepted —
    no quoting, semicolons, or whitespace.

    Raises
    ------
    ValidationError
    """
    cleaned = name.strip()
    if not _SQL_IDENTIFIER_RE.match(cleaned):
        raise ValidationError(
            f"Invalid {field_label} '{name}'. Only letters, digits, underscores, "
            "and one optional 'schema.table' qualifier are allowed."
        )
    return cleaned


# ─── Model selection validation ───────────────────────────────────────────────

def validate_model_selection(
    selected_keys: Sequence[str],
    registry: dict[str, Any],
) -> list[str]:
    """
    Validate and cap the user's model selection.

    Returns a cleaned list of valid model keys (at most MAX_MODELS).

    Raises
    ------
    ValidationError  – if no valid keys are selected.
    """
    valid = [k for k in selected_keys if k in registry]

    if not valid:
        raise ValidationError(
            "No valid models selected for this task type. "
            "Please choose at least one model from the sidebar."
        )

    if len(valid) > MAX_MODELS:
        valid = valid[:MAX_MODELS]

    return valid


# ─── Column name sanitisation ─────────────────────────────────────────────────

def sanitise_column_names(columns: list[str]) -> list[str]:
    """
    Strip null bytes and control characters from column names.

    Also replaces empty names with 'unnamed_N'.
    """
    cleaned = []
    for i, col in enumerate(columns):
        safe = "".join(
            ch for ch in str(col) if ch >= " " and ch != "\x7f"
        ).strip()
        cleaned.append(safe if safe else f"unnamed_{i}")
    return cleaned


# ─── PII detection ────────────────────────────────────────────────────────────

def detect_pii_columns(columns: list[str]) -> list[str]:
    """
    Return a list of column names that may contain PII.

    Uses token matching against a curated PII keyword set.
    """
    pii_cols = []
    for col in columns:
        tokens = set(re.findall(r"[a-z]+", col.lower()))
        if tokens & _PII_TOKENS:
            pii_cols.append(col)
    return pii_cols
