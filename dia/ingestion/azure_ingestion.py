"""
dia/ingestion/azure_ingestion.py
────────────────────────────────
Ingestion source: Azure Blob Storage.

Requires: pip install azure-storage-blob
Auth: connection string OR account name + SAS token / account key.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

from ..config import MAX_FILE_BYTES
from ..exceptions import ConfigurationError, IngestionError, ValidationError
from ..validators import sanitise_column_names, validate_dataframe_shape
from .base import IngestionResult, IngestionSource
from .local_csv import _parse_csv

log = logging.getLogger("dia.ingestion.azure")


class AzureBlobSource(IngestionSource):
    """Fetch a CSV from Azure Blob Storage."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            from azure.storage.blob import BlobServiceClient  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        connection_string: str = "",
        account_name: str = "",
        account_key: str = "",
        sas_token: str = "",
        container_name: str = "",
        blob_name: str = "",
        **kwargs: Any,
    ) -> IngestionResult:
        """
        Download and parse a CSV blob from Azure.

        Parameters
        ----------
        connection_string : Azure storage connection string (takes priority)
        account_name      : Storage account name (alternative auth)
        account_key       : Account key (alternative auth)
        sas_token         : SAS token (alternative auth)
        container_name    : Blob container name (required)
        blob_name         : Blob path within container (required)

        Raises
        ------
        ConfigurationError  – SDK not installed
        ValidationError     – missing required params
        IngestionError      – Azure access failure
        """
        if not self.is_available():
            raise ConfigurationError(
                "azure-storage-blob is not installed. "
                "Run: pip install azure-storage-blob"
            )

        container_name = container_name.strip()
        blob_name = blob_name.strip()

        if not container_name:
            raise ValidationError("Azure container name is required.")
        if not blob_name:
            raise ValidationError("Azure blob name is required.")
        if not connection_string and not account_name:
            raise ConfigurationError(
                "Provide either an Azure connection string or account name + key/SAS token."
            )

        log.info(
            "Fetching blob: container=%s  blob=%s  account=%s",
            container_name,
            blob_name,
            account_name or "[from connection string]",
        )

        try:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

            if connection_string.strip():
                client = BlobServiceClient.from_connection_string(connection_string.strip())
            elif sas_token.strip():
                account_url = f"https://{account_name}.blob.core.windows.net"
                client = BlobServiceClient(
                    account_url=account_url, credential=sas_token.strip()
                )
            else:
                account_url = f"https://{account_name}.blob.core.windows.net"
                client = BlobServiceClient(
                    account_url=account_url, credential=account_key.strip()
                )

            blob_client = client.get_blob_client(
                container=container_name, blob=blob_name
            )

            # Check size via properties
            props = blob_client.get_blob_properties()
            obj_size = props.size or 0
            if obj_size > MAX_FILE_BYTES:
                max_mb = MAX_FILE_BYTES / (1024 ** 2)
                raise ValidationError(
                    f"Azure blob is {obj_size / (1024 ** 2):.1f} MB, "
                    f"exceeding the {max_mb:.0f} MB limit."
                )

            buf = io.BytesIO()
            blob_client.download_blob().readinto(buf)
            raw_bytes = buf.getvalue()

        except (ValidationError, ConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001
            safe_msg = str(exc).replace(account_key or "", "***").replace(sas_token or "", "***")
            raise IngestionError(f"Azure Blob error: {safe_msg}") from exc

        df = _parse_csv(raw_bytes)
        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        log.info("Azure blob loaded: %d rows × %d cols", df.shape[0], df.shape[1])

        meta: dict[str, Any] = {
            "account": account_name or "[connection string]",
            "container": container_name,
            "blob": blob_name,
            "file_size_mb": round(len(raw_bytes) / (1024 ** 2), 2),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }

        return IngestionResult(
            df=df,
            source_label=f"Azure Blob — {container_name}/{blob_name}",
            meta=meta,
        )
