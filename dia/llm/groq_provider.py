"""
dia/llm/groq_provider.py
─────────────────────────
Hosted LLM backend via Groq's free-tier, OpenAI-compatible REST API
(https://groq.com). Serves open-weight models on fast LPU hardware. Uses
`requests` directly rather than the `groq` SDK, so this backend needs no
new package.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from ..config import GROQ_API_KEY, GROQ_MODEL
from ..exceptions import ConfigurationError, LLMProviderError
from .base import LLMProvider, LLMResponse

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_TIMEOUT_S = 30


def _to_groq_message(m: dict[str, Any]) -> dict[str, Any]:
    """Translate a generic chat message into Groq's OpenAI-compatible wire shape."""
    out: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
            }
            for tc in m["tool_calls"]
        ]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id", "")
        if m.get("name"):
            out["name"] = m["name"]
    return out


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    @classmethod
    def is_available(cls) -> bool:
        """Credential presence only — a network ping would itself burn a rate-limited call."""
        return bool(GROQ_API_KEY.strip())

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResponse:
        if not self.api_key.strip():
            raise ConfigurationError("Groq API key is missing (set DIA_GROQ_API_KEY).")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_groq_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(_ENDPOINT, json=payload, headers=headers, timeout=_TIMEOUT_S)
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMProviderError(f"Groq timed out after {_TIMEOUT_S}s.") from exc
        except requests.exceptions.HTTPError as exc:
            raise LLMProviderError(f"Groq request failed ({resp.status_code}): {resp.text[:200]}") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMProviderError("Groq returned a non-JSON response.") from exc

        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        raw_calls = message.get("tool_calls") or []

        calls: list[dict[str, Any]] = []
        for tc in raw_calls:
            fn = tc.get("function") or {}
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except (TypeError, ValueError):
                args = {}
            calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})

        return LLMResponse(
            text=message.get("content", "") or "",
            provider=self.name,
            model=self.model,
            tool_calls=calls,
            raw=data,
        )
