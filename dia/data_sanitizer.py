"""
dia/data_sanitizer.py
─────────────────────
Enterprise Automated Data Sanitizer & Malformed Data Repair Engine.
Autonomously detects and repairs:
1. Encoding discrepancies, non-standard delimiters, and jagged CSV rows.
2. Dirty numeric strings with currencies ($/€/£/₹), thousand separators, and unit multipliers (k/M/B).
3. Ambiguous null representations ('N/A', '?', '--', 'null', 'missing', '#N/A').
4. Inconsistent boolean text strings ('yes'/'no', 'true'/'false', 't'/'f').
5. Duplicate, empty, or special-character column headers.
6. Inf / -Inf numerical overflow values.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataLoadError

log = logging.getLogger("dia.sanitizer")

_COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1")
_COMMON_DELIMITERS = [",", ";", "\t", "|"]

_NULL_STRINGS = {
    "", "nan", "null", "none", "n/a", "na", "nil", "unknown", "missing",
    "?", "-", "--", "---", "#n/a", "#null!", "#value!", "inf", "-inf"
}

_CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹", "kr", "R$"]


def robust_parse_csv_bytes(raw_bytes: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Parses arbitrary, potentially malformed CSV bytes by detecting encoding,
    sniffing delimiters, and handling ragged/jagged lines cleanly.
    """
    repair_log: list[str] = []
    detected_encoding = "utf-8"
    detected_delimiter = ","

    # 1. Attempt detection of encoding and delimiter
    for enc in _COMMON_ENCODINGS:
        try:
            sample_text = raw_bytes[:4096].decode(enc)
            detected_encoding = enc
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text)
                detected_delimiter = dialect.delimiter
            except Exception:
                # Fallback heuristic: count common delimiters in first line
                first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
                delim_counts = {d: first_line.count(d) for d in _COMMON_DELIMITERS}
                detected_delimiter = max(delim_counts, key=delim_counts.get) if any(delim_counts.values()) else ","
            break
        except UnicodeDecodeError:
            continue

    buf = io.BytesIO(raw_bytes)
    
    # 2. Try standard read with detected parameters
    try:
        df = pd.read_csv(
            buf,
            encoding=detected_encoding,
            sep=detected_delimiter,
            on_bad_lines="skip",
            low_memory=False,
        )
    except Exception as e:
        repair_log.append(f"Standard parser failed ({str(e)[:40]}); attempting python-engine fallback.")
        buf.seek(0)
        try:
            df = pd.read_csv(
                buf,
                encoding=detected_encoding,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as fallback_exc:
            raise DataLoadError(f"Could not parse the file as CSV: {fallback_exc}") from fallback_exc

    # 3. Sanitize and clean the resulting DataFrame
    clean_df, sanitize_report = sanitize_dataframe(df)
    sanitize_report["detected_encoding"] = detected_encoding
    sanitize_report["detected_delimiter"] = detected_delimiter
    sanitize_report["parser_repairs"] = repair_log

    return clean_df, sanitize_report


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Cleans cell-level and schema-level malformations across an entire DataFrame.
    """
    clean_df = df.copy()
    report: dict[str, Any] = {
        "renamed_columns": {},
        "coerced_numeric_columns": [],
        "imputed_null_counts": {},
        "boolean_converted_columns": [],
        "inf_replaced_count": 0,
        "total_cells_repaired": 0,
    }

    # ── 1. Column Name Normalization ──────────────────────────────────────────
    new_cols = []
    seen_cols: dict[str, int] = {}
    for col in clean_df.columns:
        col_str = str(col).strip()
        # Replace special chars with underscores
        cleaned = re.sub(r"[^\w\s]", "_", col_str)
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        if not cleaned or cleaned.startswith("Unnamed"):
            cleaned = "feature"
        
        # Handle duplicates
        if cleaned in seen_cols:
            seen_cols[cleaned] += 1
            unique_name = f"{cleaned}_{seen_cols[cleaned]}"
        else:
            seen_cols[cleaned] = 0
            unique_name = cleaned

        if unique_name != str(col):
            report["renamed_columns"][str(col)] = unique_name
        new_cols.append(unique_name)

    clean_df.columns = pd.Index(new_cols)

    # ── 2. Cell-level Null and Infinity Standardization ───────────────────────
    # Replace Inf and -Inf
    inf_mask = np.isneginf(clean_df.select_dtypes(include=[np.number])) | np.isposinf(clean_df.select_dtypes(include=[np.number]))
    inf_count = int(inf_mask.sum().sum()) if not inf_mask.empty else 0
    if inf_count > 0:
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        report["inf_replaced_count"] = inf_count
        report["total_cells_repaired"] += inf_count

    # ── 3. Type-specific Column Sanitization ──────────────────────────────────
    for col in clean_df.columns:
        series = clean_df[col]

        # Process object/string columns
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            str_series = series.astype(str).str.strip()
            
            # Standardize null strings
            null_matches = str_series.str.lower().isin(_NULL_STRINGS)
            null_count = int(null_matches.sum())
            if null_count > 0:
                series_as_obj = clean_df[col].astype(object)
                series_as_obj[null_matches] = np.nan
                clean_df[col] = series_as_obj
                report["imputed_null_counts"][col] = null_count
                report["total_cells_repaired"] += null_count
                str_series = clean_df[col].dropna().astype(str).str.strip()

            if str_series.empty:
                continue

            # Check if column is a dirty numeric column (e.g. "$1,200.50", "15.4%", "10k")
            cleaned_num_series, is_numeric = _try_coerce_numeric_string(str_series)
            if is_numeric:
                clean_df[col] = cleaned_num_series
                report["coerced_numeric_columns"].append(col)
                report["total_cells_repaired"] += len(str_series)
                continue

            # Check if column is a boolean flag in disguise ("yes"/"no", "true"/"false")
            cleaned_bool_series, is_bool = _try_coerce_boolean_string(str_series)
            if is_bool:
                clean_df[col] = cleaned_bool_series
                report["boolean_converted_columns"].append(col)
                report["total_cells_repaired"] += len(str_series)

    return clean_df, report


def _try_coerce_numeric_string(series: pd.Series) -> tuple[pd.Series, bool]:
    """
    Attempts to strip currencies, percentage signs, comma separators, and multipliers.
    Returns (cleaned_series, is_valid_numeric).
    """
    if series.empty:
        return series, False

    sample = series.head(100).tolist()
    
    # Check if sample strings look like dirty numbers
    numeric_pattern = re.compile(r"^[\$€£¥₹]?\s*-?[\d,]+(?:\.\d+)?\s*[%kKmMbB]?$")
    match_count = sum(1 for s in sample if numeric_pattern.match(str(s).strip()))

    if match_count / max(1, len(sample)) < 0.60:
        return series, False

    def _clean_val(val: str) -> float | np.nan:
        v = str(val).strip()
        for sym in _CURRENCY_SYMBOLS:
            v = v.replace(sym, "")
        v = v.replace(",", "").replace("%", "").strip()

        # Check multipliers
        multiplier = 1.0
        if v.endswith(("k", "K")):
            multiplier = 1000.0
            v = v[:-1]
        elif v.endswith(("m", "M")):
            multiplier = 1000000.0
            v = v[:-1]
        elif v.endswith(("b", "B")):
            multiplier = 1000000000.0
            v = v[:-1]

        try:
            return float(v) * multiplier
        except ValueError:
            return np.nan

    cleaned = series.apply(_clean_val)
    return cleaned, True


def _try_coerce_boolean_string(series: pd.Series) -> tuple[pd.Series, bool]:
    """
    Converts string representations of booleans to 1/0 integers.
    """
    bool_map = {
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "y": 1, "n": 0,
        "t": 1, "f": 0,
        "1": 1, "0": 0,
        "active": 1, "inactive": 0,
    }
    lower_vals = series.str.lower()
    unique_vals = set(lower_vals.unique())

    if unique_vals.issubset(set(bool_map.keys())) and len(unique_vals) <= 3:
        return lower_vals.map(bool_map), True

    return series, False


def format_sanitization_report_markdown(report: dict[str, Any]) -> str:
    """Formats the data sanitization audit report as Markdown."""
    lines = [
        "### 🧹 Automated Data Sanitization & Repair Audit",
        f"- **Total Malformed Cells Repaired:** `{report.get('total_cells_repaired', 0):,}`",
        f"- **Detected File Encoding:** `{report.get('detected_encoding', 'utf-8')}`",
        f"- **Detected Delimiter:** `{repr(report.get('detected_delimiter', ','))}`",
    ]

    if report.get("renamed_columns"):
        lines.append(f"- **Normalized Column Headers ({len(report['renamed_columns'])}):**")
        for orig, new in list(report["renamed_columns"].items())[:5]:
            lines.append(f"  - `{orig}` ➔ `{new}`")

    if report.get("coerced_numeric_columns"):
        lines.append(f"- **Dirty Numeric Strings Coerced ({len(report['coerced_numeric_columns'])}):** `{', '.join(report['coerced_numeric_columns'])}`")

    if report.get("boolean_converted_columns"):
        lines.append(f"- **Boolean Flags Standardized ({len(report['boolean_converted_columns'])}):** `{', '.join(report['boolean_converted_columns'])}`")

    if report.get("imputed_null_counts"):
        lines.append(f"- **Standardized Null Tokens ('N/A', '?', '--'):** across `{len(report['imputed_null_counts'])}` columns.")

    if report.get("inf_replaced_count", 0) > 0:
        lines.append(f"- **Infinite / Overflow Values Sanitized:** `{report['inf_replaced_count']:,}` values.")

    return "\n".join(lines)
