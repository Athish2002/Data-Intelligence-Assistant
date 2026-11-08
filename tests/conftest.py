"""
tests/conftest.py - Shared pytest fixtures and utilities for the DIA test suite.
"""

from __future__ import annotations

import io
import os
import threading
import time
import urllib.request

import numpy as np
import pandas as pd
import pytest

# Base URL
DIA_BASE_URL = os.environ.get("DIA_BASE_URL", "http://127.0.0.1:8000")


def is_server_running(url: str = "http://127.0.0.1:8000/api/v1/health") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.8) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_server_running():
    """Ensure local server is running for browser/HTTP tests."""
    if is_server_running():
        yield
        return

    import uvicorn
    from api.server import app

    for attempt in range(3):
        try:
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
            server = uvicorn.Server(config)
            t = threading.Thread(target=server.run, daemon=True)
            t.start()

            for _ in range(50):
                time.sleep(0.1)
                if is_server_running():
                    break
            if is_server_running():
                break
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    yield



def launch_browser(playwright_instance):
    """Launch headless browser with Chrome/Edge/Chromium fallback."""
    for channel in ["chrome", "msedge"]:
        try:
            return playwright_instance.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return playwright_instance.chromium.launch(headless=True)


def relative_luminance(r: int, g: int, b: int) -> float:
    """Calculates WCAG 2.2 relative luminance for an sRGB color."""
    def channel_linear(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel_linear(r) + 0.7152 * channel_linear(g) + 0.0722 * channel_linear(b)


def contrast_ratio(rgb1: tuple, rgb2: tuple) -> float:
    """Computes the WCAG contrast ratio between two RGB tuples."""
    lum1 = relative_luminance(*rgb1)
    lum2 = relative_luminance(*rgb2)
    l_max = max(lum1, lum2)
    l_min = min(lum1, lum2)
    return (l_max + 0.05) / (l_min + 0.05)


def parse_rgb(rgb_str: str) -> tuple:
    """Parses rgb/rgba CSS string into an integer (r, g, b) tuple."""
    clean = rgb_str.replace("rgba(", "").replace("rgb(", "").replace(")", "").strip()
    parts = [p.strip() for p in clean.split(",")][:3]
    return tuple(int(float(p)) for p in parts)


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