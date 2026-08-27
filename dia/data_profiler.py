"""
dia/data_profiler.py
────────────────────
Generates a rich profile of a DataFrame:
  • per-column stats (dtype, null %, unique count, cardinality)
  • inferred column roles (ID, date, target candidate, duration, text, numeric)
  • target type detection (classification vs regression)
  • data readiness report
"""

from __future__ import annotations

import re

import pandas as pd

from .utils import confidence_label

# ─── Column role definitions ──────────────────────────────────────────────────

ROLE_ID = "identifier"
ROLE_DATE = "date / time"
ROLE_DURATION = "duration / tenure"
ROLE_TARGET_CANDIDATE = "target candidate"
ROLE_CATEGORICAL = "categorical feature"
ROLE_NUMERIC = "numeric feature"
ROLE_TEXT = "free text"
ROLE_BINARY = "binary flag"
ROLE_CONSTANT = "constant (useless)"
ROLE_HIGH_CARDINALITY = "high-cardinality categorical"

# Column-name tokens that suggest identifiers
_ID_TOKENS = {"id", "key", "uuid", "guid", "code", "number", "num", "no", "ref"}
# Column-name tokens that suggest dates
_DATE_TOKENS = {"date", "time", "at", "on", "created", "updated", "timestamp", "dt", "year", "month", "day"}
# Column-name tokens that suggest duration/tenure
_DURATION_TOKENS = {"tenure", "duration", "age", "days", "weeks", "months", "years", "length", "period", "elapsed"}
# Column-name tokens that suggest possible targets
_TARGET_TOKENS = {
    "churn", "fraud", "label", "target", "class", "output", "result",
    "status", "flag", "default", "survived", "outcome", "y", "price",
    "cost", "revenue", "salary", "sales", "score", "demand",
}

# High-cardinality threshold (fraction of unique values vs total rows)
_HIGH_CARD_RATIO = 0.50
# Low cardinality for binary detection
_BINARY_MAX_UNIQUE = 2


def _tokenize_col(col: str) -> set[str]:
    """Split a column name into lowercase tokens."""
    return set(re.findall(r"[a-z]+", col.lower()))


# ─── Per-column stats ─────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per column.

    Columns returned:
        column, dtype, null_pct, unique_count, unique_ratio, sample_values
    """
    records = []
    n = len(df)

    for col in df.columns:
        series = df[col]
        null_count = series.isna().sum()
        unique_count = series.nunique(dropna=True)

        records.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "null_count": int(null_count),
                "null_pct": round(100 * null_count / n, 1) if n else 0.0,
                "unique_count": int(unique_count),
                "unique_ratio": round(unique_count / n, 4) if n else 0.0,
                "sample_values": _sample_values(series),
            }
        )

    return pd.DataFrame(records)


def _sample_values(series: pd.Series, k: int = 3) -> str:
    """Return up to k non-null sample values as a comma-separated string."""
    vals = series.dropna().unique()[:k]
    return ", ".join(str(v) for v in vals)


# ─── Column role inference ─────────────────────────────────────────────────────

def infer_column_roles(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate the profile DataFrame with an inferred role and confidence score.

    Returns the same DataFrame with two new columns: role, confidence.
    """
    n = len(df)
    roles = []
    confidences = []
    explanations = []

    for _, row in profile.iterrows():
        col = row["column"]
        dtype = row["dtype"]
        null_pct = row["null_pct"]
        unique_count = int(row["unique_count"])
        unique_ratio = float(row["unique_ratio"])
        tokens = _tokenize_col(col)

        role, conf, explanation = _infer_role(
            col, tokens, dtype, n, unique_count, unique_ratio, null_pct, df[col]
        )
        roles.append(role)
        confidences.append(conf)
        explanations.append(explanation)

    result = profile.copy()
    result["inferred_role"] = roles
    result["confidence"] = confidences
    result["confidence_label"] = [confidence_label(c) for c in confidences]
    result["explanation"] = explanations
    return result


def _infer_role(
    col: str,
    tokens: set,
    dtype: str,
    n_rows: int,
    unique_count: int,
    unique_ratio: float,
    null_pct: float,
    series: pd.Series,
) -> tuple[str, float, str]:
    """Core heuristic engine – returns (role, confidence, explanation)."""

    # ── Constant column ───────────────────────────────────────────────────────
    if unique_count <= 1:
        return ROLE_CONSTANT, 0.99, "Only one unique value – not useful for ML."

    # ── Binary flag ───────────────────────────────────────────────────────────
    if unique_count == 2:
        if tokens & _TARGET_TOKENS:
            return (
                ROLE_TARGET_CANDIDATE, 0.88,
                "Binary column with target-like name – strong target candidate.",
            )
        return ROLE_BINARY, 0.80, "Binary column – likely a flag or indicator."

    # ── Identifier ─────────────────────────────────────────────────────────────
    # Only flag as identifier for integer/object columns – not continuous floats
    is_float = "float" in dtype
    if not is_float and tokens & _ID_TOKENS and unique_ratio > 0.90:
        return (
            ROLE_ID, 0.92,
            f"Column name contains ID tokens and has {unique_ratio:.0%} unique values.",
        )
    # Near-unique non-float column → likely an identifier
    if not is_float and unique_ratio >= 0.99:
        return ROLE_ID, 0.85, "Near-unique values suggest this is an identifier."
    # Near-unique FLOAT → continuous numeric feature (e.g. price, charge)
    if is_float and unique_ratio > 0.80:
        return ROLE_NUMERIC, 0.85, "High-cardinality float column – likely a continuous numeric feature."

    # ── Date / time ───────────────────────────────────────────────────────────
    if tokens & _DATE_TOKENS:
        return ROLE_DATE, 0.82, "Column name contains date/time tokens."
    if "datetime" in dtype or "date" in dtype:
        return ROLE_DATE, 0.95, "Column has datetime dtype."
    # Try parsing a sample as date
    if dtype == "object":
        sample = series.dropna().head(10)
        try:
            pd.to_datetime(sample)
            return ROLE_DATE, 0.70, "Values look like dates (parsed successfully)."
        except Exception:  # noqa: S110 — routine "is this parseable as a date" probe, not a failure
            pass

    # ── Duration / tenure ─────────────────────────────────────────────────────
    if tokens & _DURATION_TOKENS:
        return (
            ROLE_DURATION, 0.80,
            "Column name suggests a duration or tenure measurement.",
        )

    # ── Target candidate ──────────────────────────────────────────────────────
    if tokens & _TARGET_TOKENS:
        return (
            ROLE_TARGET_CANDIDATE, 0.78,
            "Column name contains common target keywords.",
        )

    # ── High-cardinality categorical ──────────────────────────────────────────
    if dtype == "object" and unique_ratio > _HIGH_CARD_RATIO:
        return (
            ROLE_HIGH_CARDINALITY, 0.75,
            f"String column with {unique_ratio:.0%} unique values – likely free text or IDs.",
        )

    # ── Free text ─────────────────────────────────────────────────────────────
    if dtype == "object":
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len and avg_len > 40:
            return ROLE_TEXT, 0.72, "Long string values suggest free-text content."
        return ROLE_CATEGORICAL, 0.70, "Low-cardinality string column – categorical feature."

    # ── Numeric feature ───────────────────────────────────────────────────────
    if "int" in dtype or "float" in dtype:
        return ROLE_NUMERIC, 0.75, "Numeric column – likely a feature."

    return ROLE_CATEGORICAL, 0.50, "Could not determine role confidently."


# ─── Target type detection ────────────────────────────────────────────────────

def detect_target_type(df: pd.DataFrame, target_col: str) -> dict:
    """
    Determine whether the target column leads to classification or regression.

    Returns
    -------
    dict with keys: task_type, reason, n_classes (for classification)
    """
    series = df[target_col].dropna()
    dtype = str(series.dtype)
    unique_count = series.nunique()
    unique_ratio = unique_count / len(series) if len(series) else 0

    # String dtype → classification
    if "object" in dtype or "bool" in dtype or "category" in dtype:
        return {
            "task_type": "classification",
            "reason": f"Target has string/boolean dtype ({dtype}).",
            "n_classes": unique_count,
        }

    # Integer with very few unique values → classification
    if "int" in dtype and unique_count <= 20:
        return {
            "task_type": "classification",
            "reason": f"Integer target with only {unique_count} unique values.",
            "n_classes": unique_count,
        }

    # Float or high-unique integer → regression
    if "float" in dtype or unique_ratio > 0.05:
        return {
            "task_type": "regression",
            "reason": (
                f"Target has {unique_count} unique values "
                f"({unique_ratio:.1%} of rows) – treating as continuous."
            ),
            "n_classes": None,
        }

    # Default: classification
    return {
        "task_type": "classification",
        "reason": "Could not determine conclusively; defaulting to classification.",
        "n_classes": unique_count,
    }


# ─── Data readiness report ─────────────────────────────────────────────────────

def generate_readiness_report(
    df: pd.DataFrame,
    annotated_profile: pd.DataFrame,
    goal_info: dict,
    target_col: str,
    task_type: str,
) -> dict:
    """
    Produce a high-level data-readiness verdict.

    Returns
    -------
    dict with keys:
        verdict       : "Ready" | "Partially Ready" | "Not Ready"
        score         : int 0–100
        useful_features : list[str]
        missing_signals : list[str]
        leakage_risk    : list[str]
        summary_lines   : list[str]
    """
    n_rows, n_cols = df.shape
    target_row = annotated_profile[annotated_profile["column"] == target_col]
    target_null_pct = float(target_row["null_pct"].iloc[0]) if not target_row.empty else 0.0

    useful_features = []
    leakage_risk = []
    missing_signals = []
    issues = []
    score = 100

    for _, row in annotated_profile.iterrows():
        col = row["column"]
        role = row["inferred_role"]
        null_pct = row["null_pct"]

        if col == target_col:
            continue

        # Useful feature candidates
        if role in (ROLE_NUMERIC, ROLE_DURATION, ROLE_BINARY, ROLE_CATEGORICAL):
            useful_features.append(col)

        # High null penalty
        if null_pct > 50:
            issues.append(f"'{col}' is {null_pct:.0f}% null.")
            score -= 5

        # Leakage risk: identifiers, dates, high-cardinality strings
        if role in (ROLE_ID, ROLE_HIGH_CARDINALITY, ROLE_CONSTANT):
            leakage_risk.append(col)

        # Date columns without feature engineering
        if role == ROLE_DATE:
            leakage_risk.append(col)  # Raw dates can leak future information

    # Row count checks
    if n_rows < 100:
        issues.append(f"Only {n_rows} rows – model may overfit significantly.")
        score -= 20
    elif n_rows < 500:
        issues.append(f"{n_rows} rows is small; results may be unstable.")
        score -= 10

    # Target null check
    if target_null_pct > 10:
        issues.append(f"Target column '{target_col}' has {target_null_pct:.0f}% missing values.")
        score -= 15

    # Feature count
    if len(useful_features) == 0:
        missing_signals.append("No obvious numeric or categorical features found.")
        score -= 20
    elif len(useful_features) < 3:
        missing_signals.append("Very few useful features detected; predictions may be weak.")
        score -= 5

    score = max(0, min(100, score))

    if score >= 75:
        verdict = "✅ Ready"
    elif score >= 45:
        verdict = "⚠️ Partially Ready"
    else:
        verdict = "❌ Not Ready"

    summary_lines = issues or ["Dataset looks generally suitable for the stated goal."]

    return {
        "verdict": verdict,
        "score": score,
        "useful_features": useful_features[:10],
        "missing_signals": missing_signals,
        "leakage_risk": list(set(leakage_risk))[:10],
        "summary_lines": summary_lines,
    }
