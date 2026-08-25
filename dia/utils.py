"""
dia/utils.py
────────────
General-purpose helpers used across the application.
All constants are imported from dia.config for single-source-of-truth.
"""

from __future__ import annotations

import re
from typing import Sequence

from .config import (
    BINARY_MAX_UNIQUE,
    HIGH_CARD_RATIO,
    MAX_FILE_BYTES,
    MAX_MODELS,
)

__all__ = [
    "limit_models",
    "is_plotly_available",
    "is_seaborn_available",
    "is_shap_available",
    "is_sentence_transformers_available",
    "clean_column_name",
    "clean_column_names",
    "confidence_label",
    "truncate",
    # Re-export constants so callers that previously used utils still work
    "MAX_FILE_BYTES",
    "MAX_MODELS",
]


# ─── Model cap ────────────────────────────────────────────────────────────────

def limit_models(selected: Sequence[str], max_models: int = MAX_MODELS) -> list[str]:
    """Return at most *max_models* from *selected*, preserving order."""
    return list(selected)[:max_models]


# ─── Availability probes ──────────────────────────────────────────────────────

def is_plotly_available() -> bool:
    """Return True if plotly is importable."""
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


def is_seaborn_available() -> bool:
    """Return True if seaborn is importable."""
    try:
        import seaborn  # noqa: F401
        return True
    except ImportError:
        return False


def is_shap_available() -> bool:
    """Return True if shap is importable."""
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def is_sentence_transformers_available() -> bool:
    """Return True if sentence_transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def is_torch_available() -> bool:
    """Return True if torch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ─── Column name utilities ─────────────────────────────────────────────────────

def clean_column_name(col: str) -> str:
    """Lowercase, strip, replace spaces/dashes with underscores."""
    return re.sub(r"[\s\-]+", "_", col.strip().lower())


def clean_column_names(columns: Sequence[str]) -> list[str]:
    """Apply :func:`clean_column_name` to each element of *columns*."""
    return [clean_column_name(c) for c in columns]


# ─── Confidence helpers ────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    """Convert a 0–1 score to a human-readable confidence label."""
    if score >= 0.85:
        return "High"
    if score >= 0.60:
        return "Medium"
    return "Low"


# ─── Text helpers ──────────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 80) -> str:
    """Truncate *text* to *max_len* characters, appending '…' if needed."""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
