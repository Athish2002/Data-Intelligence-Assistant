"""
dia/ingestion/__init__.py
─────────────────────────
Multi-source data ingestion package.

Supported sources
─────────────────
  local_csv    – CSV file upload (Streamlit UploadedFile)
  url          – Public HTTP/HTTPS URL pointing to a CSV
  s3           – AWS S3 bucket object
  gcs          – Google Cloud Storage blob
  azure        – Azure Blob Storage
  sql          – SQLAlchemy-compatible databases (PostgreSQL, MySQL, SQLite, MSSQL)
  bigquery     – Google BigQuery table or query
  snowflake    – Snowflake table or query

All optional cloud sources are imported lazily so missing dependencies
don't crash the application — they only show an installation hint.
"""

from __future__ import annotations

from .base import IngestionResult, IngestionSource
from .local_csv import LocalCSVSource
from .url_ingestion import URLSource
from .sql_ingestion import SQLSource

__all__ = [
    "IngestionResult",
    "IngestionSource",
    "LocalCSVSource",
    "URLSource",
    "SQLSource",
]

# Cloud sources are available at runtime if optional deps are installed
try:
    from .s3_ingestion import S3Source
    __all__.append("S3Source")
except ImportError:
    S3Source = None  # type: ignore[assignment,misc]

try:
    from .gcs_ingestion import GCSSource
    __all__.append("GCSSource")
except ImportError:
    GCSSource = None  # type: ignore[assignment,misc]

try:
    from .azure_ingestion import AzureBlobSource
    __all__.append("AzureBlobSource")
except ImportError:
    AzureBlobSource = None  # type: ignore[assignment,misc]

try:
    from .bigquery_ingestion import BigQuerySource
    __all__.append("BigQuerySource")
except ImportError:
    BigQuerySource = None  # type: ignore[assignment,misc]

try:
    from .snowflake_ingestion import SnowflakeSource
    __all__.append("SnowflakeSource")
except ImportError:
    SnowflakeSource = None  # type: ignore[assignment,misc]


# ─── Source registry ──────────────────────────────────────────────────────────

SOURCE_REGISTRY: dict[str, dict] = {
    "demo_sample": {
        "label": "🧪 Built-in Demo Datasets",
        "description": "Quickly load curated benchmark datasets (Telecom Churn, Credit Risk, Real Estate).",
        "cls": None,
        "available": True,
        "requires": None,
    },
    "local_csv": {
        "label": "📂 Local CSV Upload",
        "description": "Upload a CSV file from your computer (max 500 MB).",
        "cls": LocalCSVSource,
        "available": True,
        "requires": None,
    },
    "url": {
        "label": "🌐 URL / HTTP",
        "description": "Load a CSV from any public HTTP or HTTPS URL.",
        "cls": URLSource,
        "available": True,
        "requires": None,
    },
    "sql": {
        "label": "🗄️ SQL Database",
        "description": "Connect to PostgreSQL, MySQL, SQLite, or SQL Server via SQLAlchemy.",
        "cls": SQLSource,
        "available": True,
        "requires": "sqlalchemy",
    },
    "s3": {
        "label": "☁️ AWS S3",
        "description": "Fetch a CSV from an Amazon S3 bucket.",
        "cls": S3Source,
        "available": S3Source is not None,
        "requires": "boto3",
    },
    "gcs": {
        "label": "☁️ Google Cloud Storage",
        "description": "Fetch a CSV from a GCS bucket.",
        "cls": GCSSource,
        "available": GCSSource is not None,
        "requires": "google-cloud-storage",
    },
    "azure": {
        "label": "☁️ Azure Blob Storage",
        "description": "Fetch a CSV from Azure Blob Storage.",
        "cls": AzureBlobSource,
        "available": AzureBlobSource is not None,
        "requires": "azure-storage-blob",
    },
    "bigquery": {
        "label": "☁️ Google BigQuery",
        "description": "Run a BigQuery SQL query and load results as a DataFrame.",
        "cls": BigQuerySource,
        "available": BigQuerySource is not None,
        "requires": "google-cloud-bigquery",
    },
    "snowflake": {
        "label": "❄️ Snowflake",
        "description": "Connect to a Snowflake warehouse and run a SQL query.",
        "cls": SnowflakeSource,
        "available": SnowflakeSource is not None,
        "requires": "snowflake-connector-python",
    },
}
