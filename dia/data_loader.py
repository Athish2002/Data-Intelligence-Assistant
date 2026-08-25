"""
dia/data_loader.py
──────────────────
Backward-compatible shim — delegates to the new ingestion package.
New code should import from dia.ingestion directly.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from .ingestion.local_csv import LocalCSVSource

__all__ = ["load_csv"]


def load_csv(
    uploaded_file: Any,
    max_bytes: int | None = None,  # kept for API compatibility, enforced in source
) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV from *uploaded_file*.

    .. deprecated::
        Use ``dia.ingestion.LocalCSVSource().load(uploaded_file=f)`` directly.

    Returns
    -------
    df   : pd.DataFrame
    meta : dict
    """
    warnings.warn(
        "load_csv() is deprecated. Use dia.ingestion.LocalCSVSource().load() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = LocalCSVSource().load(uploaded_file=uploaded_file)
    return result.df, result.meta
