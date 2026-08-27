"""
tests/test_retrieval.py
─────────────────────────
Unit tests for dia/retrieval.py's chunking, PII redaction, and the
semantic/lexical fallback search behavior.
"""

from __future__ import annotations

import pandas as pd
import pytest

import dia.retrieval as retrieval
from dia.retrieval import build_session_index


class TestBuildSessionIndex:
    def test_produces_one_chunk_per_column_plus_overview(self, churn_df: pd.DataFrame) -> None:
        index = build_session_index(churn_df)
        column_chunks = [c for c in index.chunks if c.source_type == "column_profile"]
        overview_chunks = [c for c in index.chunks if c.source_type == "dataset_overview"]
        assert len(column_chunks) == len(churn_df.columns)
        assert len(overview_chunks) == 1

    def test_readiness_and_target_chunks_included_when_provided(self, churn_df: pd.DataFrame) -> None:
        readiness = {
            "verdict": "Ready",
            "score": 80,
            "summary_lines": ["Looks fine."],
            "leakage_risk": ["customer_id"],
            "missing_signals": [],
            "useful_features": ["tenure_months"],
        }
        target_type_info = {"task_type": "classification", "reason": "binary target", "n_classes": 2}
        index = build_session_index(
            churn_df, readiness=readiness, target_col="churn_flag", target_type_info=target_type_info
        )
        source_types = {c.source_type for c in index.chunks}
        assert "readiness" in source_types
        assert "target" in source_types

    def test_dictionary_entries_become_chunks(self, tiny_df: pd.DataFrame) -> None:
        entries = [{"term": "MRR", "definition": "Monthly recurring revenue."}]
        index = build_session_index(tiny_df, dictionary_entries=entries)
        dict_chunks = [c for c in index.chunks if c.source_type == "dictionary"]
        assert len(dict_chunks) == 1
        assert "MRR" in dict_chunks[0].text
        assert "Monthly recurring revenue" in dict_chunks[0].text

    def test_never_raises_on_missing_optional_args(self, tiny_df: pd.DataFrame) -> None:
        index = build_session_index(tiny_df)
        assert len(index) > 0

    def test_pii_column_values_are_redacted_from_chunk_text(self) -> None:
        df = pd.DataFrame({
            "email": ["alice@example.com", "bob@example.com", "carol@example.com"],
            "purchase_count": [1, 2, 3],
        })
        index = build_session_index(df)
        email_chunk = next(c for c in index.chunks if c.metadata.get("column") == "email")
        assert "@example.com" not in email_chunk.text
        assert "redacted" in email_chunk.text.lower()

    def test_explicit_pii_columns_override_the_cheap_default(self) -> None:
        # "notes" wouldn't be caught by the header-token heuristic, but a caller
        # that already ran the fuller compliance scanner can still force redaction.
        df = pd.DataFrame({"notes": ["secret info a", "secret info b"], "amount": [1, 2]})
        index = build_session_index(df, pii_columns=["notes"])
        notes_chunk = next(c for c in index.chunks if c.metadata.get("column") == "notes")
        assert "redacted" in notes_chunk.text.lower()


class TestSimpleVectorIndexSearch:
    def test_relevant_chunk_ranks_above_irrelevant_chunk(self, churn_df: pd.DataFrame) -> None:
        index = build_session_index(churn_df)
        results = index.search("monthly charges billing amount", top_k=3)
        assert results, "expected at least one search result"
        top_columns = [r.chunk.metadata.get("column") for r in results]
        assert "monthly_charges" in top_columns

    def test_empty_index_returns_empty_results(self) -> None:
        index = retrieval.SimpleVectorIndex([])
        assert index.search("anything") == []

    def test_lexical_fallback_used_when_embeddings_unavailable(
        self, churn_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(retrieval, "is_sentence_transformers_available", lambda: False)
        index = build_session_index(churn_df)
        assert index.is_semantic is False
        results = index.search("monthly charges", top_k=3)
        assert results
        assert any(r.chunk.metadata.get("column") == "monthly_charges" for r in results)

    def test_search_with_no_token_overlap_returns_empty_in_lexical_mode(
        self, tiny_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(retrieval, "is_sentence_transformers_available", lambda: False)
        index = build_session_index(tiny_df)
        assert index.search("zzzznonexistentzzzz") == []
