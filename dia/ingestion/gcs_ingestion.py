"""
dia/ingestion/gcs_ingestion.py
──────────────────────────────
Ingestion source: Google Cloud Storage blob.

Requires: pip install google-cloud-storage
Auth: service account JSON string OR GOOGLE_APPLICATION_CREDENTIALS env var.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pandas as pd

from ..config import MAX_FILE_BYTES
from ..exceptions import ConfigurationError, IngestionError, ValidationError
from ..validators import sanitise_column_names, validate_dataframe_shape
from .base import IngestionResult, IngestionSource
from .local_csv import _parse_csv

log = logging.getLogger("dia.ingestion.gcs")


class GCSSource(IngestionSource):
    """Fetch a CSV from a Google Cloud Storage bucket."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            from google.cloud import storage  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        bucket: str = "",
        blob_name: str = "",
        service_account_json: str = "",
        project: str = "",
        **kwargs: Any,
    ) -> IngestionResult:
        """
        Download and parse a CSV blob from GCS.

        Parameters
        ----------
        bucket               : GCS bucket name
        blob_name            : path to blob within bucket
        service_account_json : service account JSON as a string (optional;
                               falls back to GOOGLE_APPLICATION_CREDENTIALS)
        project              : GCP project ID (optional)

        Raises
        ------
        ConfigurationError  – google-cloud-storage not installed
        ValidationError     – missing bucket or blob
        IngestionError      – GCS error
        """
        if not self.is_available():
            raise ConfigurationError(
                "google-cloud-storage is not installed. "
                "Run: pip install google-cloud-storage"
            )

        bucket = bucket.strip()
        blob_name = blob_name.strip()

        if not bucket:
            raise ValidationError("GCS bucket name is required.")
        if not blob_name:
            raise ValidationError("GCS blob name (path) is required.")

        log.info("Fetching gs://%s/%s", bucket, blob_name)

        try:
            from google.cloud import storage  # noqa: PLC0415
            from google.oauth2 import service_account  # noqa: PLC0415

            if service_account_json.strip():
                creds_dict = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                client = storage.Client(
                    project=project or creds_dict.get("project_id"),
                    credentials=credentials,
                )
            else:
                # Falls back to GOOGLE_APPLICATION_CREDENTIALS env var or ADC
                client = storage.Client(project=project or None)

            gcs_bucket = client.bucket(bucket)
            blob = gcs_bucket.blob(blob_name)

            # Check size before downloading
            blob.reload()
            obj_size = blob.size or 0
            if obj_size > MAX_FILE_BYTES:
                max_mb = MAX_FILE_BYTES / (1024 ** 2)
                raise ValidationError(
                    f"GCS blob is {obj_size / (1024 ** 2):.1f} MB, "
                    f"exceeding the {max_mb:.0f} MB limit."
                )

            buf = io.BytesIO()
            blob.download_to_file(buf)
            raw_bytes = buf.getvalue()

        except (ValidationError, ConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"GCS error: {exc}") from exc

        df = _parse_csv(raw_bytes)
        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        log.info("GCS blob loaded: %d rows × %d cols", df.shape[0], df.shape[1])

        meta: dict[str, Any] = {
            "bucket": bucket,
            "blob": blob_name,
            "file_size_mb": round(len(raw_bytes) / (1024 ** 2), 2),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }

        return IngestionResult(
            df=df,
            source_label=f"GCS — gs://{bucket}/{blob_name}",
            meta=meta,
        )
