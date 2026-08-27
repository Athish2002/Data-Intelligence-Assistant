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
