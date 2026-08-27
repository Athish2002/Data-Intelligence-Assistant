"""
dia/llm
───────
Pluggable LLM provider layer. Mirrors dia/ingestion's shape: an abstract
base, one module per backend, and a small registry.

Providers are tried local-first (Ollama), then free-tier hosted (Groq,
Gemini). get_default_provider() returns None when nothing is available,
which callers must treat as "stay in deterministic/offline mode" — never
as an error.
"""

from __future__ import annotations

import logging
from typing import Any

from ..exceptions import ConfigurationError
from .base import LLMProvider, LLMResponse
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider

log = logging.getLogger("dia.llm")

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OllamaProvider",
    "GroqProvider",
    "GeminiProvider",
    "PROVIDER_REGISTRY",
    "get_provider",
    "get_default_provider",
]

# Priority order for get_default_provider(): local-first, then free-hosted.
PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "ollama": {
        "label": "Ollama (local, free, private)",
        "cls": OllamaProvider,
        "requires": None,
    },
    "groq": {
        "label": "Groq (hosted, free tier, fast)",
        "cls": GroqProvider,
        "requires": "DIA_GROQ_API_KEY",
    },
    "gemini": {
        "label": "Gemini (hosted, free tier)",
        "cls": GeminiProvider,
        "requires": "DIA_GEMINI_API_KEY",
    },
}

# Deliberately no "available" key precomputed at import time (unlike
# dia.ingestion.SOURCE_REGISTRY): Ollama's availability is a live network
# fact that can change mid-session, so every caller must call
# is_available() fresh rather than trust a cached boolean.


def get_provider(key: str) -> LLMProvider:
    """Instantiate a provider by registry key, regardless of availability."""
    entry = PROVIDER_REGISTRY.get(key)
    if entry is None:
        raise ConfigurationError(f"Unknown LLM provider '{key}'.")
    return entry["cls"]()


def get_default_provider() -> LLMProvider | None:
    """
    Try Ollama -> Groq -> Gemini in priority order. Returns None if none are
    available, meaning: no credentials configured and no local server running.
    Callers must treat None as "stay in deterministic mode", not as an error.
    """
    for key in ("ollama", "groq", "gemini"):
        try:
            if PROVIDER_REGISTRY[key]["cls"].is_available():
                return PROVIDER_REGISTRY[key]["cls"]()
        except Exception:
            log.debug("Provider '%s' availability check raised; trying the next one.", key, exc_info=True)
            continue
    return None
