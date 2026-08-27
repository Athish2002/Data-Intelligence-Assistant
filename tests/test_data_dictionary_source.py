"""
tests/test_data_dictionary_source.py
────────────────────────────────────────
Unit tests for the DataDictionarySource ingestion source, including the
real dia.compliance.scan_dataset_privacy() PII gate (not mocked — that gate
is the actual safety property under test).
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from dia.exceptions import ValidationError
from dia.ingestion.data_dictionary import DataDictionarySource
from tests.conftest import MockUploadedFile


def _make_csv(rows: list[tuple[str, str]], columns: tuple[str, str] = ("term", "definition")) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows, columns=list(columns)).to_csv(buf, index=False)
    return buf.getvalue()


class TestDataDictionarySource:
    source = DataDictionarySource()

    def test_loads_clean_glossary(self) -> None:
        csv_bytes = _make_csv([("MRR", "Monthly recurring revenue."), ("Churn", "Customer cancellation rate.")])
        result = self.source.load(uploaded_file=MockUploadedFile(csv_bytes, name="glossary.csv"))
        assert result.meta["n_terms"] == 2
        assert result.meta["safe_to_persist"] is True
        assert list(result.df.columns) == ["term", "definition"]

    def test_rejects_missing_required_columns(self) -> None:
        csv_bytes = _make_csv([("MRR", "x")], columns=("word", "meaning"))
        f = MockUploadedFile(csv_bytes, name="bad.csv")
        with pytest.raises(ValidationError, match="term"):
            self.source.load(uploaded_file=f)

    def test_rejects_all_blank_terms(self) -> None:
        csv_bytes = _make_csv([("", "some definition")])
        f = MockUploadedFile(csv_bytes, name="empty.csv")
        with pytest.raises(ValidationError, match="no usable"):
            self.source.load(uploaded_file=f)

    def test_flags_bare_emails_as_unsafe_to_persist(self) -> None:
        # All-matching values so the real scanner's sample-ratio threshold fires
        # regardless of exact sample size (see dia/compliance.py's match_count check).
        csv_bytes = _make_csv([
            ("Contact A", "alice@example.com"),
            ("Contact B", "bob@example.com"),
            ("Contact C", "carol@example.com"),
        ])
        result = self.source.load(uploaded_file=MockUploadedFile(csv_bytes, name="glossary.csv"))
        assert result.meta["safe_to_persist"] is False
        assert result.meta["pii_gate_report"]["pii_columns_count"] > 0

    def test_rejects_none_file(self) -> None:
        with pytest.raises(ValidationError):
            self.source.load(uploaded_file=None)
