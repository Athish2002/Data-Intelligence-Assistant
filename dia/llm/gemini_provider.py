"""
dia/llm/gemini_provider.py
────────────────────────────
Hosted LLM backend via Google's Gemini API free tier. Refactored out of the
inline call that used to live in dia/llm_context.py.

Scope cut for v1: no tool-calling support here (tool_calls is always empty).
Ollama and Groq are the two tool-capable backends; Gemini only ever answers
directly from whatever context is already in the prompt. Messages are
flattened into one text transcript rather than mapped onto the SDK's
multi-turn history API, since there's no multi-turn tool loop to support yet.
"""

from __future__ import annotations

from typing import Any

from ..config import GEMINI_API_KEY, GEMINI_MODEL
from ..exceptions import ConfigurationError, LLMProviderError
from .base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    @classmethod
    def is_available(cls) -> bool:
        """Requires both a configured key and the optional SDK being installed."""
        if not GEMINI_API_KEY.strip():
            return False
        try:
            import google.generativeai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResponse:
        if not self.api_key.strip():
            raise ConfigurationError("Gemini API key is missing (set DIA_GEMINI_API_KEY).")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ConfigurationError(
                "google-generativeai is not installed; pip install google-generativeai."
            ) from exc

        transcript = "\n\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages
        )

        try:
            genai.configure(api_key=self.api_key.strip())
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                transcript,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            text = (response.text or "").strip()
        except Exception as exc:
            raise LLMProviderError(f"Gemini request failed: {exc}") from exc

        return LLMResponse(text=text, provider=self.name, model=self.model, tool_calls=[], raw={})
