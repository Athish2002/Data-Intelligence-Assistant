"""
dia/config.py
─────────────
Single source of truth for all application constants.
Every value can be overridden via environment variables so the app
is 12-factor compliant without touching source code.
"""

from __future__ import annotations

import os

# ─── File / upload limits ─────────────────────────────────────────────────────

MAX_FILE_MB: int = int(os.getenv("DIA_MAX_FILE_MB", "500"))
MAX_FILE_BYTES: int = MAX_FILE_MB * 1024 * 1024

MAX_ROWS: int = int(os.getenv("DIA_MAX_ROWS", "10000000"))   # 10 M rows
MAX_COLS: int = int(os.getenv("DIA_MAX_COLS", "10000"))       # 10 K columns

# ─── Model training ───────────────────────────────────────────────────────────

MAX_MODELS: int = int(os.getenv("DIA_MAX_MODELS", "4"))
TEST_SPLIT_RATIO: float = float(os.getenv("DIA_TEST_SPLIT", "0.20"))
RANDOM_STATE: int = int(os.getenv("DIA_RANDOM_STATE", "42"))

# ─── Explainability ───────────────────────────────────────────────────────────

SHAP_SAMPLE_SIZE: int = int(os.getenv("DIA_SHAP_SAMPLE", "200"))
TOP_FEATURES: int = int(os.getenv("DIA_TOP_FEATURES", "20"))

# ─── Goal parsing ─────────────────────────────────────────────────────────────

GOAL_MAX_CHARS: int = int(os.getenv("DIA_GOAL_MAX_CHARS", "1000"))

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("DIA_LOG_LEVEL", "INFO").upper()

# ─── Network / cloud timeouts (seconds) ──────────────────────────────────────

HTTP_TIMEOUT_S: int = int(os.getenv("DIA_HTTP_TIMEOUT", "30"))
CLOUD_TIMEOUT_S: int = int(os.getenv("DIA_CLOUD_TIMEOUT", "60"))

# ─── Cardinality thresholds ───────────────────────────────────────────────────

HIGH_CARD_RATIO: float = float(os.getenv("DIA_HIGH_CARD_RATIO", "0.50"))
BINARY_MAX_UNIQUE: int = 2

# ─── Retrieval / RAG ──────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME: str = os.getenv("DIA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K: int = int(os.getenv("DIA_RAG_TOP_K", "5"))

# ─── LLM providers (optional — chat/RAG generation) ──────────────────────────

OLLAMA_HOST: str = os.getenv("DIA_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("DIA_OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT_S: int = int(os.getenv("DIA_OLLAMA_TIMEOUT", "30"))

GROQ_API_KEY: str = os.getenv("DIA_GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("DIA_GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_API_KEY: str = os.getenv("DIA_GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("DIA_GEMINI_MODEL", "gemini-2.5-flash-lite")

LLM_MAX_TOOL_ITERATIONS: int = int(os.getenv("DIA_LLM_MAX_TOOL_ITERATIONS", "3"))

# ─── Local persistence (data dictionary only — datasets are never persisted) ──

DICTIONARY_DB_PATH: str = os.getenv(
    "DIA_DICTIONARY_DB_PATH",
    os.path.join(os.path.expanduser("~"), ".dia", "dictionary.sqlite3"),
)

# ─── Network, Server & Environment Configuration ──────────────────────────────

HOST: str = os.getenv("DIA_HOST", "0.0.0.0")
PORT: int = int(os.getenv("DIA_PORT", "8000"))
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("DIA_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# ─── Enterprise Authentication & RBAC ─────────────────────────────────────────

AUTH_ENABLED: bool = os.getenv("DIA_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
JWT_SECRET: str = os.getenv("DIA_JWT_SECRET", "dia-enterprise-insecure-secret-change-in-production")
JWT_ALGORITHM: str = os.getenv("DIA_JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.getenv("DIA_JWT_EXPIRE_MINUTES", "1440"))

DEFAULT_ROLE: str = os.getenv("DIA_DEFAULT_ROLE", "Admin")
DEFAULT_TENANT_ID: str = os.getenv("DIA_DEFAULT_TENANT_ID", "default")
DEFAULT_ORG_ID: str = os.getenv("DIA_DEFAULT_ORG_ID", "default")
API_KEYS_CONFIG: str = os.getenv("DIA_API_KEYS", "")

# ─── Pluggable Storage Abstraction (Local Disk, NFS, Cloud S3, K8s PVC) ───────

STORAGE_TYPE: str = os.getenv("DIA_STORAGE_TYPE", "local").lower()
STORAGE_PATH: str = os.getenv(
    "DIA_STORAGE_PATH",
    os.path.join(os.path.expanduser("~"), ".dia", "storage"),
)
S3_BUCKET: str = os.getenv("DIA_S3_BUCKET", "dia-storage")
S3_ENDPOINT_URL: str | None = os.getenv("DIA_S3_ENDPOINT_URL", None)
S3_REGION: str = os.getenv("DIA_S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID: str | None = os.getenv("DIA_S3_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", None))
S3_SECRET_ACCESS_KEY: str | None = os.getenv("DIA_S3_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", None))
S3_PREFIX: str = os.getenv("DIA_S3_PREFIX", "dia")


