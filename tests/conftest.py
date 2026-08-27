"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the DIA test suite.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

# ─── Sample DataFrames ────────────────────────────────────────────────────────

@pytest.fixture()
def churn_df() -> pd.DataFrame:
    """Binary classification dataset (300 rows, 6 columns)."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "customer_id":     range(n),
        "tenure_months":   rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20.0, 120.0, n),
        "num_services":    rng.integers(1, 8, n),
        "joined_on":       pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "churn_flag":      rng.choice([0, 1], n, p=[0.75, 0.25]),
    })


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    """Regression dataset (200 rows)."""
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "sqft":      rng.integers(500, 4000, n).astype(float),
        "bedrooms":  rng.integers(1, 6, n),
        "bathrooms": rng.uniform(1.0, 4.0, n),
        "age_years": rng.integers(1, 80, n),
        "price":     rng.uniform(100_000.0, 900_000.0, n),
    })


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    """Minimal 2-column, 5-row DataFrame."""
    return pd.DataFrame({"feature": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})


@pytest.fixture()
def all_null_col_df() -> pd.DataFrame:
    """DataFrame with one fully null column."""
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [None, None, None],
        "target": [0, 1, 0],
    })


# ─── Mock UploadedFile ────────────────────────────────────────────────────────

class MockUploadedFile:
    """Minimal mock for Streamlit UploadedFile."""

    def __init__(self, content: bytes, name: str = "test.csv", mime: str = "text/csv") -> None:
        self._buf = io.BytesIO(content)
        self.name = name
        self.type = mime

    def read(self) -> bytes:
        return self._buf.read()

    def seek(self, pos: int) -> None:
        self._buf.seek(pos)


@pytest.fixture()
def csv_bytes(churn_df: pd.DataFrame) -> bytes:
    """Churn DataFrame serialised to CSV bytes."""
    buf = io.BytesIO()
    churn_df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture()
def mock_csv_file(csv_bytes: bytes) -> MockUploadedFile:
    """A mock Streamlit UploadedFile wrapping the churn CSV."""
    return MockUploadedFile(csv_bytes, name="churn.csv", mime="text/csv")
