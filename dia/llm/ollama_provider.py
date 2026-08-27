"""
dia/llm/ollama_provider.py
───────────────────────────
Local LLM backend via Ollama's REST API (https://ollama.com). Free, fully
offline once a model is pulled — no API key. Uses `requests`, already a
hard dependency, so this backend needs no new package.
"""

from __future__ import annotations

from typing import Any

import requests

from ..config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_S
from ..exceptions import LLMProviderError
from .base import LLMProvider, LLMResponse


def _to_ollama_message(m: dict[str, Any]) -> dict[str, Any]:
    """Translate a generic chat message into Ollama's /api/chat wire shape."""
    out: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {"function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in m["tool_calls"]
        ]
    return out


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> None:
        self.host = host.rstrip("/")
        self.model = model

    @classmethod
    def is_available(cls) -> bool:
        """Live check: a local Ollama server must actually be running right now."""
        try:
            resp = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_S)
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMProviderError(f"Ollama timed out after {OLLAMA_TIMEOUT_S}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama returned a non-JSON response.") from exc

        message = data.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        calls = [
            {
                "id": f"call_{i}",
                "name": (tc.get("function") or {}).get("name", ""),
                "arguments": (tc.get("function") or {}).get("arguments", {}) or {},
            }
            for i, tc in enumerate(raw_calls)
        ]

        return LLMResponse(
            text=message.get("content", "") or "",
            provider=self.name,
            model=self.model,
            tool_calls=calls,
            raw=data,
        )
