"""
dia/ingestion/sql_ingestion.py
──────────────────────────────
Ingestion source: SQL databases via SQLAlchemy.

Supports any SQLAlchemy-compatible database:
  PostgreSQL   → postgresql+psycopg2://user:pass@host:5432/db
  MySQL        → mysql+pymysql://user:pass@host:3306/db
  SQLite       → sqlite:///path/to/file.db
  SQL Server   → mssql+pyodbc://user:pass@host/db?driver=...
  and more.

The query or table name is specified by the user.
Credentials are NEVER logged — only the host/db portion of the DSN is shown.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from ..config import CLOUD_TIMEOUT_S, MAX_ROWS
from ..exceptions import ConfigurationError, IngestionError, ValidationError
from ..validators import sanitise_column_names, validate_dataframe_shape, validate_sql_identifier
from .base import IngestionResult, IngestionSource

log = logging.getLogger("dia.ingestion.sql")

# Regex to strip credentials from a DSN for safe logging
_DSN_REDACT_RE = re.compile(r"(://[^:@/]+:)[^@]+(@)", re.IGNORECASE)


def _safe_dsn(dsn: str) -> str:
    """Return DSN with password replaced by ***."""
    return _DSN_REDACT_RE.sub(r"\1***\2", dsn)


class SQLSource(IngestionSource):
    """Load data from a SQL database using SQLAlchemy."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import sqlalchemy  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        connection_string: str = "",
        query: str = "",
        table_name: str = "",
        limit: int | None = None,
        **kwargs: Any,
    ) -> IngestionResult:
        """
        Execute a SQL query or fetch an entire table.

        Parameters
        ----------
        connection_string : SQLAlchemy DSN  (required)
        query             : SQL SELECT statement (takes priority over table_name)
        table_name        : table to fetch entirely (used if query is empty)
        limit             : max rows to fetch (None = no limit, capped at MAX_ROWS)

        Raises
        ------
        ConfigurationError  – missing connection string
        ValidationError     – no query or table name provided
        IngestionError      – connection or query failure
        """
        if not self.is_available():
            raise ConfigurationError(
                "SQLAlchemy is not installed. "
                "Run: pip install sqlalchemy psycopg2-binary pymysql"
            )

        connection_string = connection_string.strip()
        if not connection_string:
            raise ConfigurationError("A SQLAlchemy connection string is required.")

        query = re.sub(r"[\s;]+$", "", query)
        table_name = table_name.strip()

        if not query and not table_name:
            raise ValidationError("Provide either a SQL query or a table name.")

        # Enforce row limit
        row_limit = min(limit, MAX_ROWS) if limit else MAX_ROWS

        if query:
            # The query itself is an intentionally-arbitrary user-supplied SQL
            # statement (this source's whole purpose), run against the same
            # database the user just supplied credentials for — not a value
            # smuggled in from a less-trusted party. Only wrapped to apply a
            # row limit without rewriting the user's own SQL.
            safe_sql = f"SELECT * FROM ({query}) _dia_subq LIMIT {row_limit}"  # noqa: S608
        else:
            # Unlike `query`, `table_name` is meant to be a bare identifier, and
            # SQL has no parameter-placeholder syntax for identifiers — validate
            # it looks like one before interpolating.
            table_name = validate_sql_identifier(table_name, field_label="table name")
            safe_sql = f"SELECT * FROM {table_name} LIMIT {row_limit}"  # noqa: S608 — validated above

        log.info(
            "Connecting to SQL source: %s  query_len=%d",
            _safe_dsn(connection_string),
            len(safe_sql),
        )

        try:
            import sqlalchemy as sa  # noqa: PLC0415

            engine = sa.create_engine(
                connection_string,
                connect_args={"connect_timeout": CLOUD_TIMEOUT_S}
                if "postgresql" in connection_string or "mysql" in connection_string
                else {},
            )

            with engine.connect() as conn:
                df = pd.read_sql(sa.text(safe_sql), conn)

        except Exception as exc:  # noqa: BLE001
            # Strip any credentials that may appear in the exception message
            safe_msg = _DSN_REDACT_RE.sub(r"\1***\2", str(exc))
            raise IngestionError(f"SQL query failed: {safe_msg}") from exc

        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        log.info("SQL query returned %d rows × %d cols", df.shape[0], df.shape[1])

        source_ref = table_name or "custom query"
        meta: dict[str, Any] = {
            "source": _safe_dsn(connection_string),
            "query": safe_sql[:200],
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }

        return IngestionResult(
            df=df,
            source_label=f"SQL — {source_ref}",
            meta=meta,
        )
