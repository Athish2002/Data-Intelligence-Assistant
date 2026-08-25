"""
dia/ingestion/s3_ingestion.py
─────────────────────────────
Ingestion source: AWS S3 bucket object.

Requires: pip install boto3
Credentials: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars,
             OR an IAM role (EC2/ECS/Lambda), OR explicit params.
Credentials are NEVER logged.
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

log = logging.getLogger("dia.ingestion.s3")


class S3Source(IngestionSource):
    """Fetch a CSV from an AWS S3 bucket."""

    @classmethod
    def is_available(cls) -> bool:
        try:
            import boto3  # noqa: F401
            return True
        except ImportError:
            return False

    def load(
        self,
        bucket: str = "",
        key: str = "",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str = "us-east-1",
        endpoint_url: str | None = None,
        **kwargs: Any,
    ) -> IngestionResult:
        """
        Download and parse a CSV object from S3.

        Parameters
        ----------
        bucket              : S3 bucket name
        key                 : S3 object key (path inside bucket)
        aws_access_key_id   : optional — falls back to env vars / IAM role
        aws_secret_access_key: optional
        aws_session_token   : optional (for STS temporary credentials)
        region_name         : AWS region (default us-east-1)
        endpoint_url        : optional custom endpoint (e.g. MinIO)

        Raises
        ------
        ConfigurationError  – boto3 not installed
        ValidationError     – missing bucket or key
        IngestionError      – S3 access or download failure
        """
        if not self.is_available():
            raise ConfigurationError(
                "boto3 is not installed. Run: pip install boto3"
            )

        bucket = bucket.strip()
        key = key.strip()

        if not bucket:
            raise ValidationError("S3 bucket name is required.")
        if not key:
            raise ValidationError("S3 object key is required.")

        log.info("Fetching s3://%s/%s", bucket, key)

        try:
            import boto3  # noqa: PLC0415
            from botocore.exceptions import BotoCoreError, ClientError  # noqa: PLC0415

            session = boto3.session.Session(
                aws_access_key_id=aws_access_key_id or None,
                aws_secret_access_key=aws_secret_access_key or None,
                aws_session_token=aws_session_token or None,
                region_name=region_name,
            )
            s3 = session.client("s3", endpoint_url=endpoint_url or None)

            # HEAD first to check size before downloading
            try:
                head = s3.head_object(Bucket=bucket, Key=key)
                obj_size = head.get("ContentLength", 0)
                if obj_size > MAX_FILE_BYTES:
                    max_mb = MAX_FILE_BYTES / (1024 ** 2)
                    raise ValidationError(
                        f"S3 object is {obj_size / (1024 ** 2):.1f} MB, "
                        f"exceeding the {max_mb:.0f} MB limit."
                    )
            except (BotoCoreError, ClientError) as exc:
                log.debug("HEAD object failed (may lack s3:GetObjectAttributes): %s", exc)
                obj_size = 0  # proceed — will catch size during download

            buf = io.BytesIO()
            s3.download_fileobj(bucket, key, buf)
            raw_bytes = buf.getvalue()

        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            safe_msg = str(exc)
            # Never include credentials in error messages
            for secret_kw in ("password", "secret", "token", "key"):
                if secret_kw in safe_msg.lower():
                    safe_msg = "[credentials redacted — check AWS permissions]"
                    break
            raise IngestionError(f"S3 error: {safe_msg}") from exc

        if len(raw_bytes) > MAX_FILE_BYTES:
            max_mb = MAX_FILE_BYTES / (1024 ** 2)
            raise ValidationError(
                f"Downloaded object exceeds the {max_mb:.0f} MB limit."
            )

        df = _parse_csv(raw_bytes)
        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        log.info("S3 object loaded: %d rows × %d cols", df.shape[0], df.shape[1])

        meta: dict[str, Any] = {
            "bucket": bucket,
            "key": key,
            "region": region_name,
            "file_size_mb": round(len(raw_bytes) / (1024 ** 2), 2),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        }

        return IngestionResult(
            df=df,
            source_label=f"S3 — s3://{bucket}/{key}",
            meta=meta,
        )
