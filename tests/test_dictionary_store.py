"""
tests/test_dictionary_store.py
─────────────────────────────────
Unit tests for dia/dictionary_store.py's local SQLite persistence. Every
test runs against a throwaway sqlite file so the developer's real
~/.dia/dictionary.sqlite3 is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dia.dictionary_store as dictionary_store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dictionary_store, "DICTIONARY_DB_PATH", str(tmp_path / "test_dictionary.sqlite3"))


class TestDictionaryStore:
    def test_save_and_load_round_trip(self) -> None:
        n = dictionary_store.save_entries(
            [{"term": "MRR", "definition": "Monthly recurring revenue."}], source_label="test.csv"
        )
        assert n == 1
        entries = dictionary_store.load_entries()
        assert len(entries) == 1
        assert entries[0]["term"] == "MRR"
        assert entries[0]["definition"] == "Monthly recurring revenue."

    def test_upsert_is_case_insensitive(self) -> None:
        dictionary_store.save_entries([{"term": "MRR", "definition": "First definition."}], source_label="a.csv")
        dictionary_store.save_entries([{"term": "mrr", "definition": "Updated definition."}], source_label="b.csv")
        entries = dictionary_store.load_entries()
        assert len(entries) == 1
        assert entries[0]["definition"] == "Updated definition."

    def test_delete_entry_is_case_insensitive(self) -> None:
        dictionary_store.save_entries([{"term": "ARR", "definition": "Annual recurring revenue."}], source_label="a.csv")
        assert dictionary_store.delete_entry("arr") is True
        assert dictionary_store.load_entries() == []

    def test_delete_missing_entry_returns_false(self) -> None:
        assert dictionary_store.delete_entry("does-not-exist") is False

    def test_clear_all(self) -> None:
        dictionary_store.save_entries(
            [{"term": "A", "definition": "a"}, {"term": "B", "definition": "b"}], source_label="a.csv"
        )
        removed = dictionary_store.clear_all()
        assert removed == 2
        assert dictionary_store.count_entries() == 0

    def test_count_entries(self) -> None:
        assert dictionary_store.count_entries() == 0
        dictionary_store.save_entries([{"term": "A", "definition": "a"}], source_label="a.csv")
        assert dictionary_store.count_entries() == 1
