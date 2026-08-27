"""
tests/test_llm_providers.py
────────────────────────────
Unit tests for the dia/llm provider abstraction. No real network calls or
running Ollama server required — requests.get/post are monkeypatched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from dia.exceptions import ConfigurationError, LLMProviderError
from dia.llm import PROVIDER_REGISTRY, get_default_provider
from dia.llm.gemini_provider import GeminiProvider
from dia.llm.groq_provider import GroqProvider
from dia.llm.ollama_provider import OllamaProvider


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")


class TestOllamaProvider:
    def test_is_available_true_when_server_responds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))
        assert OllamaProvider.is_available() is True

    def test_is_available_false_when_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_conn_error(*a: Any, **k: Any) -> Any:
            raise requests.exceptions.ConnectionError("no server")

        monkeypatch.setattr(requests, "get", raise_conn_error)
        assert OllamaProvider.is_available() is False

    def test_generate_normalizes_dict_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeResponse(200, {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "describe_column", "arguments": {"column": "age"}}}],
            }
        })
        monkeypatch.setattr(requests, "post", lambda *a, **k: fake)
        resp = OllamaProvider().generate([{"role": "user", "content": "hi"}])
        assert resp.provider == "ollama"
        assert resp.tool_calls == [{"id": "call_0", "name": "describe_column", "arguments": {"column": "age"}}]

    def test_generate_raises_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_timeout(*a: Any, **k: Any) -> Any:
            raise requests.exceptions.Timeout("slow")

        monkeypatch.setattr(requests, "post", raise_timeout)
        with pytest.raises(LLMProviderError, match="timed out"):
            OllamaProvider().generate([{"role": "user", "content": "hi"}])


class TestGroqProvider:
    def test_is_available_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dia.llm.groq_provider.GROQ_API_KEY", "")
        assert GroqProvider.is_available() is False
        monkeypatch.setattr("dia.llm.groq_provider.GROQ_API_KEY", "sk-test")
        assert GroqProvider.is_available() is True

    def test_generate_parses_json_string_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeResponse(200, {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {"name": "value_counts", "arguments": json.dumps({"column": "region"})},
                    }],
                }
            }]
        })
        monkeypatch.setattr(requests, "post", lambda *a, **k: fake)
        resp = GroqProvider(api_key="sk-test").generate([{"role": "user", "content": "hi"}])
        assert resp.tool_calls == [{"id": "call_abc", "name": "value_counts", "arguments": {"column": "region"}}]

    def test_generate_without_key_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            GroqProvider(api_key="").generate([{"role": "user", "content": "hi"}])

    def test_generate_raises_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(429, text="rate limited"))
        with pytest.raises(LLMProviderError):
            GroqProvider(api_key="sk-test").generate([{"role": "user", "content": "hi"}])


class TestGeminiProvider:
    def test_is_available_false_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dia.llm.gemini_provider.GEMINI_API_KEY", "")
        assert GeminiProvider.is_available() is False

    def test_generate_without_key_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            GeminiProvider(api_key="").generate([{"role": "user", "content": "hi"}])


class TestProviderRegistry:
    def test_get_default_provider_prefers_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(OllamaProvider, "is_available", classmethod(lambda cls: True))
        monkeypatch.setattr(GroqProvider, "is_available", classmethod(lambda cls: True))
        monkeypatch.setattr(GeminiProvider, "is_available", classmethod(lambda cls: True))
        assert get_default_provider().name == "ollama"

    def test_get_default_provider_falls_back_to_groq(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(OllamaProvider, "is_available", classmethod(lambda cls: False))
        monkeypatch.setattr(GroqProvider, "is_available", classmethod(lambda cls: True))
        monkeypatch.setattr(GeminiProvider, "is_available", classmethod(lambda cls: True))
        assert get_default_provider().name == "groq"

    def test_get_default_provider_none_when_all_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(OllamaProvider, "is_available", classmethod(lambda cls: False))
        monkeypatch.setattr(GroqProvider, "is_available", classmethod(lambda cls: False))
        monkeypatch.setattr(GeminiProvider, "is_available", classmethod(lambda cls: False))
        assert get_default_provider() is None

    def test_registry_keys_match_provider_names(self) -> None:
        for key, entry in PROVIDER_REGISTRY.items():
            assert entry["cls"]().name == key
