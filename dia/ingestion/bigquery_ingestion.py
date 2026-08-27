"""
dia/ingestion/bigquery_ingestion.py
────────────────────────────────────
Ingestion source: Google BigQuery.

Requires: pip install google-cloud-bigquery pandas-gbq
Auth: service account JSON string OR GOOGLE_APPLICATION_CREDENTIALS env var.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from ..config import CLOUD_TIMEOUT_S, MAX_ROWS
from ..exceptions import ConfigurationError, IngestionError, ValidationError
from ..validators import sanitise_column_names, validate_dataframe_shape
from .base import IngestionResult, IngestionSource

log = logging.getLogger("dia.ingestion.bigquery")


class BigQuerySource(IngestionSource):
    """Run a BigQuery SQL query and return results as a DataFrame."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            from google.cloud import bigquery  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        project: str = "",
        query: str = "",
        service_account_json: str = "",
        location: str = "US",
        **kwargs: Any,
    ) -> IngestionResult:
        """
        Execute a BigQuery SQL query and return the result.

        Parameters
        ----------
        project              : GCP project ID (required)
        query                : Standard SQL query (required)
        service_account_json : Service account JSON as a string (optional)
        location             : BigQuery dataset location (default US)

        Raises
        ------
        ConfigurationError  – SDK not installed or missing project/query
        ValidationError     – empty query
        IngestionError      – BigQuery error
        """
        if not self.is_available():
            raise ConfigurationError(
                "google-cloud-bigquery is not installed. "
                "Run: pip install google-cloud-bigquery db-dtypes"
            )

        project = project.strip()
        query = query.strip()

        if not project:
            raise ConfigurationError("GCP project ID is required for BigQuery.")
        if not query:
            raise ValidationError("A SQL query is required.")

        # Enforce row limit by wrapping the query. `query` is intentionally-
        # arbitrary user-supplied SQL (this source's whole purpose), run
        # against the same project the user just authenticated against.
        limited_query = f"SELECT * FROM ({query}) _dia LIMIT {MAX_ROWS}"  # noqa: S608
        log.info("Running BigQuery query on project=%s  len=%d", project, len(query))

        try:
            from google.cloud import bigquery  # noqa: PLC0415
            from google.oauth2 import service_account  # noqa: PLC0415

            if service_account_json.strip():
                creds_dict = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
                )
                client = bigquery.Client(project=project, credentials=credentials)
            else:
                client = bigquery.Client(project=project)

            job_config = bigquery.QueryJobConfig(
                use_query_cache=True,
                location=location,
            )
            job = client.query(limited_query, job_config=job_config)
            df = job.to_dataframe(timeout=CLOUD_TIMEOUT_S)

        except (ConfigurationError, ValidationError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"BigQuery error: {exc}") from exc

        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        log.info("BigQuery returned: %d rows × %d cols", df.shape[0], df.shape[1])

        meta: dict[str, Any] = {
            "project": project,
            "query": query[:200],
            "location": location,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }

        return IngestionResult(
            df=df,
            source_label=f"BigQuery — {project}",
            meta=meta,
        )
