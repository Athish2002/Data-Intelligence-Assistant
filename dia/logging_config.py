"""
dia/logging_config.py
─────────────────────
Configures structured logging for the entire application.

Call `setup_logging()` once at startup (app.py).
Every module then uses `logging.getLogger(__name__)`.

PII Guard
─────────
Column *values* are NEVER logged — only column names, shapes, and types.
"""

from __future__ import annotations

import logging
import sys

from .config import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def setup_logging() -> None:
    """Initialise root logger. Safe to call multiple times (idempotent)."""
    global _configured
    if _configured:
        return

    level = getattr(logging, LOG_LEVEL, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger("dia")
    root.setLevel(level)
    # Avoid duplicate handlers if Streamlit hot-reloads the module
    if not root.handlers:
        root.addHandler(handler)

    # Silence noisy third-party loggers unless we're in DEBUG mode
    if level > logging.DEBUG:
        for noisy in ("urllib3", "botocore", "boto3", "google", "azure", "snowflake"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper: returns a child logger of 'dia'."""
    return logging.getLogger(f"dia.{name}")
