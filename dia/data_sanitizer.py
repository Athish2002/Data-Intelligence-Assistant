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
    Cleans cell-level and schema-level malformations across an entire DataFrame:
    - Normalizes column names, trims invisible whitespace and control characters.
    - Replaces Inf and -Inf with NaN and clips extreme float values outside [-1e15, 1e15].
    - Standardizes ambiguous null representations ('N/A', '?', '--', etc.).
    - Detects and coerces dirty numeric and currency strings ($1,200, 15%, 10k).
    - Converts boolean strings ('yes'/'no', 'true'/'false') to 1/0 flags.
    - Homogenizes mixed-type columns (e.g. ['int', 'str']) to uniform strings or numerics.
    - Prunes 100% all-NaN columns to prevent empty-array pipeline crashes.
    """
    clean_df = df.copy()
    report: dict[str, Any] = {
        "renamed_columns": {},
        "coerced_numeric_columns": [],
        "imputed_null_counts": {},
        "boolean_converted_columns": [],
        "inf_replaced_count": 0,
        "mixed_type_homogenized_columns": [],
        "all_nan_dropped_columns": [],
        "zero_variance_columns": [],
        "total_cells_repaired": 0,
    }

    # ── 1. Column Name Normalization ──────────────────────────────────────────
    new_cols = []
    seen_cols: dict[str, int] = {}
    for col in clean_df.columns:
        col_str = str(col).strip().replace("\u00a0", " ")
        # Replace non-word chars with underscores
        cleaned = re.sub(r"[^\w\s]", "_", col_str)
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        if not cleaned or cleaned.startswith("Unnamed"):
            cleaned = "feature"

        # Handle case-insensitive duplicates
        clean_lower = cleaned.lower()
        if clean_lower in seen_cols:
            seen_cols[clean_lower] += 1
            unique_name = f"{cleaned}_{seen_cols[clean_lower]}"
        else:
            seen_cols[clean_lower] = 0
            unique_name = cleaned

        if unique_name != str(col):
            report["renamed_columns"][str(col)] = unique_name
        new_cols.append(unique_name)

    clean_df.columns = pd.Index(new_cols)

    # ── 2. Cell-level Null and Infinity Standardization ───────────────────────
    # Replace Inf and -Inf in numeric columns
    numeric_dtypes = clean_df.select_dtypes(include=[np.number])
    if not numeric_dtypes.empty:
        inf_mask = np.isneginf(numeric_dtypes) | np.isposinf(numeric_dtypes)
        inf_count = int(inf_mask.sum().sum())
        if inf_count > 0:
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
            report["inf_replaced_count"] = inf_count
            report["total_cells_repaired"] += inf_count

        # Clip extreme floats outside [-1e15, 1e15] to prevent 64-bit float overflows
        for ncol in clean_df.select_dtypes(include=[np.number]).columns:
            clean_df[ncol] = clean_df[ncol].clip(lower=-1e15, upper=1e15)

    # ── 3. Type-specific Column Sanitization ──────────────────────────────────
    for col in list(clean_df.columns):
        series = clean_df[col]

        # Process object/string columns
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            # Normalize whitespace and strip control characters
            def _clean_str_cell(val: Any) -> Any:
                if pd.isna(val) or val is None:
                    return np.nan
                s = str(val).replace("\u00a0", " ").strip()
                s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
                if s.lower() in _NULL_STRINGS:
                    return np.nan
                return s

            clean_df[col] = series.apply(_clean_str_cell)
            null_count = int(clean_df[col].isna().sum() - series.isna().sum())
            if null_count > 0:
                report["imputed_null_counts"][col] = null_count
                report["total_cells_repaired"] += null_count

            non_null = clean_df[col].dropna()
            if non_null.empty:
                continue

            str_series = non_null.astype(str)

            # Check if column is a dirty numeric column (e.g. "$1,200.50", "15.4%", "10k")
            cleaned_num_series, is_numeric = _try_coerce_numeric_string(str_series)
            if is_numeric:
                num_col = pd.Series(np.nan, index=clean_df.index, dtype=float)
                num_col.loc[non_null.index] = cleaned_num_series.values
                clean_df[col] = num_col
                report["coerced_numeric_columns"].append(col)
                report["total_cells_repaired"] += len(str_series)
                continue

            # Check if column is a boolean flag in disguise ("yes"/"no", "true"/"false")
            cleaned_bool_series, is_bool = _try_coerce_boolean_string(str_series)
            if is_bool:
                bool_col = pd.Series(np.nan, index=clean_df.index, dtype=float)
                bool_col.loc[non_null.index] = cleaned_bool_series.values
                clean_df[col] = bool_col
                report["boolean_converted_columns"].append(col)
                report["total_cells_repaired"] += len(str_series)
                continue

            # Check mixed types in non-null entries
            type_set = {type(v) for v in non_null}
            # Attempt numeric conversion only if values are directly parseable as numeric without destroying strings
            num_coerced = pd.to_numeric(str_series, errors="coerce")
            valid_num_ratio = num_coerced.notna().sum() / max(1, len(str_series))

            if valid_num_ratio >= 0.75 and any(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in non_null):
                # Column is predominantly genuine numeric with occasional dirty/corrupted entries: coerce to float64
                num_col = pd.Series(np.nan, index=clean_df.index, dtype=float)
                num_col.loc[non_null.index] = num_coerced.values
                clean_df[col] = num_col.clip(lower=-1e15, upper=1e15)
                report["coerced_numeric_columns"].append(col)
                report["total_cells_repaired"] += len(str_series)
                continue

            # Otherwise, strictly homogenize all non-null values to uniform clean Python strings!
            # This guarantees scikit-learn OneHotEncoder / OrdinalEncoder will never see mixed types!
            clean_df[col] = clean_df[col].astype(object).apply(lambda v: str(v) if pd.notna(v) and v is not None else np.nan)
            if len(type_set) > 1 or any(not isinstance(v, str) for v in non_null):
                report["mixed_type_homogenized_columns"].append(col)
                report["total_cells_repaired"] += len(non_null)

    # ── 4. Prune All-NaN Columns ──────────────────────────────────────────────
    all_nan_cols = [c for c in clean_df.columns if clean_df[c].isna().all()]
    if all_nan_cols and len(all_nan_cols) < len(clean_df.columns):
        clean_df = clean_df.drop(columns=all_nan_cols)
        report["all_nan_dropped_columns"] = all_nan_cols

    # Detect zero-variance columns for audit reporting
    for c in clean_df.columns:
        if clean_df[c].nunique(dropna=True) <= 1:
            report["zero_variance_columns"].append(c)

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

    if report.get("mixed_type_homogenized_columns"):
        lines.append(f"- **Mixed-Type Categoricals Homogenized ({len(report['mixed_type_homogenized_columns'])}):** `{', '.join(report['mixed_type_homogenized_columns'])}`")

    if report.get("all_nan_dropped_columns"):
        lines.append(f"- **Empty (100% NaN) Columns Pruned ({len(report['all_nan_dropped_columns'])}):** `{', '.join(report['all_nan_dropped_columns'])}`")

    return "\n".join(lines)

