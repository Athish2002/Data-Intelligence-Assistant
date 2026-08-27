"""
tests/test_chat_analyst.py
─────────────────────────────
Unit tests for dia/chat_analyst.py's RAG-grounded answer_with_rag(), using a
hand-written fake LLMProvider so the tool-loop logic is verified without any
real network call or running Ollama server.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from dia.chat_analyst import (
    _dispatch_query_dataframe,
    answer_with_rag,
    execute_natural_language_query,
)
from dia.exceptions import LLMProviderError
from dia.llm.base import LLMResponse
from dia.retrieval import build_session_index


class _FakeProvider:
    """Duck-typed LLMProvider: scripted responses, no network."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.name = "fake"
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]], *, tools=None, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


class _RaisingProvider:
    name = "fake"

    def generate(self, messages: list[dict[str, Any]], *, tools=None, **kwargs: Any) -> LLMResponse:
        raise LLMProviderError("simulated outage")


class TestAnswerWithRagFallback:
    def test_no_index_matches_deterministic_answer_exactly(self, churn_df: pd.DataFrame) -> None:
        query = "average monthly_charges"
        expected = execute_natural_language_query(churn_df, query, "churn_flag")
        result = answer_with_rag(churn_df, query, "churn_flag", rag_index=None)
        assert result["text"] == expected["text"]
        assert result["engine"] == "deterministic"
        assert result["sources"] is None

    def test_empty_index_matches_deterministic_answer(self, churn_df: pd.DataFrame) -> None:
        from dia.retrieval import SimpleVectorIndex

        query = "average monthly_charges"
        expected = execute_natural_language_query(churn_df, query, "churn_flag")
        result = answer_with_rag(churn_df, query, "churn_flag", rag_index=SimpleVectorIndex([]))
        assert result["text"] == expected["text"]
        assert result["engine"] == "deterministic"

    def test_index_without_llm_provider_adds_sources_only(
        self, churn_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("dia.llm.get_default_provider", lambda: None)
        index = build_session_index(churn_df)
        result = answer_with_rag(churn_df, "monthly charges", "churn_flag", rag_index=index)
        assert result["engine"] == "retrieval_only"
        assert result["sources"]


class TestAnswerWithRagLlmToolLoop:
    def test_llm_answers_directly_without_tool_call(self, churn_df: pd.DataFrame) -> None:
        provider = _FakeProvider([LLMResponse(text="Churn correlates with contract length.", provider="fake", model="x")])
        index = build_session_index(churn_df)
        result = answer_with_rag(churn_df, "what drives churn?", "churn_flag", rag_index=index, llm_provider=provider)
        assert result["engine"] == "llm:fake"
        assert "correlates" in result["text"]
        assert result["sources"]

    def test_llm_drives_tool_call_then_final_answer(self, churn_df: pd.DataFrame) -> None:
        tool_call_response = LLMResponse(
            text="",
            provider="fake",
            model="x",
            tool_calls=[{"id": "call_0", "name": "query_dataframe", "arguments": {"operation": "describe_column", "column": "monthly_charges"}}],
        )
        final_response = LLMResponse(text="The average monthly charge is around $70.", provider="fake", model="x")
        provider = _FakeProvider([tool_call_response, final_response])
        index = build_session_index(churn_df)

        result = answer_with_rag(churn_df, "what's the average monthly charge?", "churn_flag", rag_index=index, llm_provider=provider)

        assert provider.calls == 2
        assert result["engine"] == "llm:fake"
        assert "average monthly charge" in result["text"]

    def test_provider_failure_falls_back_to_retrieval_only(self, churn_df: pd.DataFrame) -> None:
        index = build_session_index(churn_df)
        result = answer_with_rag(churn_df, "monthly charges", "churn_flag", rag_index=index, llm_provider=_RaisingProvider())
        assert result["engine"] == "retrieval_only"
        assert result["sources"]
        assert "fallback_reason" in result

    def test_exhausting_iterations_without_final_answer_falls_back(self, churn_df: pd.DataFrame) -> None:
        # Always returns a tool call, never a final text-only answer.
        looping_call = LLMResponse(
            text="",
            provider="fake",
            model="x",
            tool_calls=[{"id": "call_0", "name": "query_dataframe", "arguments": {"operation": "row_count"}}],
        )
        provider = _FakeProvider([looping_call])
        index = build_session_index(churn_df)
        result = answer_with_rag(churn_df, "monthly charges", "churn_flag", rag_index=index, llm_provider=provider)
        assert result["engine"] == "retrieval_only"
        assert "fallback_reason" in result


class TestDispatchQueryDataframe:
    def test_row_count(self, churn_df: pd.DataFrame) -> None:
        assert _dispatch_query_dataframe(churn_df, {"operation": "row_count"}, set())["row_count"] == len(churn_df)

    def test_describe_column_numeric(self, churn_df: pd.DataFrame) -> None:
        res = _dispatch_query_dataframe(churn_df, {"operation": "describe_column", "column": "monthly_charges"}, set())
        assert "mean" in res and res["column"] == "monthly_charges"

    def test_unknown_column_returns_error_not_exception(self, churn_df: pd.DataFrame) -> None:
        res = _dispatch_query_dataframe(churn_df, {"operation": "describe_column", "column": "does_not_exist"}, set())
        assert "error" in res

    def test_pii_column_refused_for_describe(self, churn_df: pd.DataFrame) -> None:
        res = _dispatch_query_dataframe(
            churn_df, {"operation": "describe_column", "column": "monthly_charges"}, {"monthly_charges"}
        )
        assert "error" in res

    def test_pii_column_still_allows_row_and_null_count(self, churn_df: pd.DataFrame) -> None:
        res = _dispatch_query_dataframe(churn_df, {"operation": "null_count", "column": "monthly_charges"}, {"monthly_charges"})
        assert "error" not in res

    def test_groupby_agg_rejects_non_numeric_target(self, churn_df: pd.DataFrame) -> None:
        res = _dispatch_query_dataframe(
            churn_df, {"operation": "groupby_agg", "group_by": "churn_flag", "column": "joined_on", "agg": "mean"}, set()
        )
        assert "error" in res

    def test_unknown_operation_returns_error(self, churn_df: pd.DataFrame) -> None:
        res = _dispatch_query_dataframe(churn_df, {"operation": "drop_table"}, set())
        assert "error" in res
