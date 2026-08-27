"""
dia/ingestion/snowflake_ingestion.py
─────────────────────────────────────
Ingestion source: Snowflake data warehouse.

Requires: pip install snowflake-connector-python
Credentials are NEVER logged.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config import MAX_ROWS
from ..exceptions import ConfigurationError, IngestionError, ValidationError
from ..validators import sanitise_column_names, validate_dataframe_shape, validate_sql_identifier
from .base import IngestionResult, IngestionSource

log = logging.getLogger("dia.ingestion.snowflake")


class SnowflakeSource(IngestionSource):
    """Run a query against a Snowflake warehouse and return a DataFrame."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import snowflake.connector  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        account: str = "",
        user: str = "",
        password: str = "",
        warehouse: str = "",
        database: str = "",
        schema: str = "PUBLIC",
        role: str = "",
        query: str = "",
        table_name: str = "",
        **kwargs: Any,
    ) -> IngestionResult:
        """
        Connect to Snowflake and execute a SQL query.

        Parameters
        ----------
        account   : Snowflake account identifier (e.g. 'xy12345.us-east-1')
        user      : Snowflake username
        password  : Snowflake password (not logged)
        warehouse : Virtual warehouse name
        database  : Database name
        schema    : Schema name (default PUBLIC)
        role      : Optional role to assume
        query     : SQL SELECT statement (takes priority over table_name)
        table_name: Table to fetch (used if query is empty)

        Raises
        ------
        ConfigurationError  – SDK not installed or missing credentials
        ValidationError     – missing query/table
        IngestionError      – connection / query failure
        """
        if not self.is_available():
            raise ConfigurationError(
                "snowflake-connector-python is not installed. "
                "Run: pip install snowflake-connector-python"
            )

        missing = [f for f, v in [
            ("account", account), ("user", user),
            ("password", password), ("warehouse", warehouse), ("database", database)
        ] if not v.strip()]
        if missing:
            raise ConfigurationError(
                f"Missing required Snowflake fields: {', '.join(missing)}"
            )

        query = query.strip()
        table_name = table_name.strip()
        if not query and not table_name:
            raise ValidationError("Provide either a SQL query or a table name.")

        if query:
            # Intentionally-arbitrary user-supplied SQL (this source's whole
            # purpose), run against the same warehouse the user just supplied
            # credentials for. Only wrapped to apply a row limit.
            safe_sql = f"SELECT * FROM ({query}) _dia LIMIT {MAX_ROWS}"  # noqa: S608
        else:
            # `table_name` is meant to be a bare identifier, not arbitrary SQL —
            # validate it looks like one before interpolating (SQL has no
            # parameter-placeholder syntax for identifiers).
            table_name = validate_sql_identifier(table_name, field_label="table name")
            safe_sql = f"SELECT * FROM {table_name} LIMIT {MAX_ROWS}"  # noqa: S608 — validated above

        log.info(
            "Connecting to Snowflake: account=%s  user=%s  db=%s",
            account.strip(), user.strip(), database.strip(),
        )

        try:
            import snowflake.connector  # noqa: PLC0415

            connect_kwargs: dict[str, Any] = {
                "account": account.strip(),
                "user": user.strip(),
                "password": password.strip(),
                "warehouse": warehouse.strip(),
                "database": database.strip(),
                "schema": schema.strip() or "PUBLIC",
                "client_session_keep_alive": False,
            }
            if role.strip():
                connect_kwargs["role"] = role.strip()

            conn = snowflake.connector.connect(**connect_kwargs)
            try:
                df = pd.read_sql(safe_sql, conn)
            finally:
                conn.close()

        except (ConfigurationError, ValidationError):
            raise
        except Exception as exc:  # noqa: BLE001
            safe_msg = str(exc)
            # Never include password in error string
            safe_msg = safe_msg.replace(password.strip(), "***")
            raise IngestionError(f"Snowflake error: {safe_msg}") from exc

        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        log.info("Snowflake query returned: %d rows × %d cols", df.shape[0], df.shape[1])

        meta: dict[str, Any] = {
            "account": account.strip(),
            "database": database.strip(),
            "schema": schema.strip(),
            "warehouse": warehouse.strip(),
            "query": safe_sql[:200],
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }

        return IngestionResult(
            df=df,
            source_label=f"Snowflake — {database.strip()}.{schema.strip()}",
            meta=meta,
        )
