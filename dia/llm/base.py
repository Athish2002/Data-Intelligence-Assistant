"""
dia/llm/base.py
────────────────
Abstract base class and shared types for all LLM provider backends.
Mirrors dia/ingestion/base.py's IngestionSource / IngestionResult shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Returned by every provider's generate() call, wire-format already normalized."""

    text: str
    provider: str
    model: str

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Normalized regardless of backend wire format:
    [{"id": str, "name": str, "arguments": dict[str, Any]}, ...]
    """

    raw: dict[str, Any] = field(default_factory=dict)
    """The raw provider response, kept for debugging/logging only."""


class LLMProvider(ABC):
    """
    Abstract base for a pluggable LLM backend.

    Subclasses implement `generate()` and `is_available()`.
    """

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResponse:
        """
        Send a chat-style message list (OpenAI-style role/content dicts) and
        return a normalized LLMResponse.

        Raises
        ------
        LLMProviderError   – network/timeout/rate-limit/malformed response
        ConfigurationError – credentials missing or the backend SDK isn't installed
        """

    @classmethod
    def is_available(cls) -> bool:
        """
        Return True if this backend can be used right now.

        For a local backend this is a live reachability check (the server
        might not be running); for a hosted backend it's a credential
        presence check (a bad key still fails at call time, not here).
        """
        return True
