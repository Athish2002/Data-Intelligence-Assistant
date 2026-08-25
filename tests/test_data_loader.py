"""
tests/test_data_loader.py
─────────────────────────
Unit tests for the LocalCSVSource ingestion source.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from dia.exceptions import DataLoadError, ValidationError
from dia.ingestion.local_csv import LocalCSVSource
from tests.conftest import MockUploadedFile


def _make_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


class TestLocalCSVSource:
    source = LocalCSVSource()

    def test_loads_valid_csv(self, mock_csv_file: MockUploadedFile) -> None:
        result = self.source.load(uploaded_file=mock_csv_file)
        assert result.df.shape[0] > 0
        assert result.df.shape[1] > 1

    def test_rejects_oversized_file(self) -> None:
        huge = MockUploadedFile(b"a" * (501 * 1024 * 1024), name="big.csv")
        with pytest.raises(ValidationError, match="exceeds"):
            self.source.load(uploaded_file=huge)

    def test_rejects_zip_magic_bytes(self) -> None:
        fake_csv = MockUploadedFile(b"PK\x03\x04" + b"x" * 100, name="evil.csv")
        with pytest.raises(ValidationError, match="ZIP"):
            self.source.load(uploaded_file=fake_csv)

    def test_rejects_none_file(self) -> None:
        with pytest.raises(ValidationError):
            self.source.load(uploaded_file=None)

    def test_rejects_wrong_extension(self) -> None:
        f = MockUploadedFile(b"a,b\n1,2", name="data.xlsx")
        with pytest.raises(ValidationError, match=".csv"):
            self.source.load(uploaded_file=f)

    def test_rejects_empty_csv(self) -> None:
        empty = MockUploadedFile(b"", name="empty.csv")
        # Empty bytes → pd.read_csv error → DataLoadError
        with pytest.raises((ValidationError, DataLoadError)):
            self.source.load(uploaded_file=empty)

    def test_rejects_single_column_csv(self) -> None:
        single_col = MockUploadedFile(b"only_col\n1\n2\n3", name="single.csv")
        with pytest.raises(ValidationError, match="2 column"):
            self.source.load(uploaded_file=single_col)

    def test_sanitises_null_bytes_in_column_names(self) -> None:
        # Create CSV with null byte in header (via raw bytes)
        bad_csv = b"col\x00name,value\n1,2\n3,4"
        f = MockUploadedFile(bad_csv, name="bad.csv")
        result = self.source.load(uploaded_file=f)
        for col in result.df.columns:
            assert "\x00" not in col

    def test_meta_contains_expected_keys(self, mock_csv_file: MockUploadedFile) -> None:
        result = self.source.load(uploaded_file=mock_csv_file)
        for key in ("file_size_bytes", "file_size_mb", "n_rows", "n_cols"):
            assert key in result.meta

    def test_latin1_encoded_csv(self) -> None:
        """Ensures encoding fallback handles Latin-1 content."""
        latin1_csv = "col1,col2\ncafé,résumé\n".encode("latin-1")
        f = MockUploadedFile(latin1_csv, name="latin.csv")
        result = self.source.load(uploaded_file=f)
        assert result.df.shape == (1, 2)
