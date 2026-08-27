"""
dia/exceptions.py
─────────────────
Custom exception hierarchy for Data Intelligence Assistant.

Typed exceptions let callers (app.py) handle each failure mode
differently and show contextually correct UI messages.
"""

from __future__ import annotations


class DIAError(Exception):
    """Base exception for all DIA errors."""


class DataLoadError(DIAError):
    """Raised when a data source cannot be read or parsed."""


class ValidationError(DIAError):
    """Raised when user-supplied input fails validation checks."""


class ModelTrainingError(DIAError):
    """Raised when model training fails unrecoverably."""


class ConfigurationError(DIAError):
    """Raised when required configuration or credentials are missing."""


class IngestionError(DIAError):
    """Raised when a cloud / SQL data source fails to connect or fetch."""


class LLMProviderError(DIAError):
    """Raised when an LLM provider call fails at runtime (timeout, rate limit, bad response)."""


class PIIWarning(UserWarning):
    """Issued when potential PII columns are detected in the dataset."""
