"""
dia/ingestion/base.py
─────────────────────
Abstract base class and shared types for all ingestion sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class IngestionResult:
    """Returned by every ingestion source on success."""

    df: pd.DataFrame
    """The loaded DataFrame."""

    source_label: str
    """Human-readable description of the data source."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Optional metadata: file size, row/col count, encoding, query, etc."""


class IngestionSource(ABC):
    """
    Abstract base for a data ingestion source.

    Subclasses implement `load()` and `render_config_ui()`.
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> IngestionResult:
        """
        Load data and return an IngestionResult.

        Raises
        ------
        IngestionError   – connection / auth / parse failure
        ValidationError  – bad user input (wrong file type, size exceeded, etc.)
        """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if all required dependencies are importable."""
        return True
