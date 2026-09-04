"""
tests/test_universal_ingestion.py
─────────────────────────────────
Automated test suite for Agile Sprint 2: Environment Versatility & Resilient Ingestion.
Tests:
- European semicolon-delimited CSVs
- Tab-separated TSV files
- Apache Parquet columnar binary datasets
- JSON records and JSON Lines (JSONL) datasets
- Multi-encoding resilience (UTF-8 with BOM, Latin-1, CP1252)
- Header normalization and column deduplication
- Root /health and /api/v1/health endpoints with dynamic platform detection
"""

import io
import json
import platform
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.server import app
from dia.config import HOST, PORT, CORS_ORIGINS
from dia.ingestion.universal_loader import UniversalLoader, detect_delimiter, sanitize_column_headers


class TestUniversalIngestion:
    @pytest.fixture
    def loader(self):
        return UniversalLoader()

    def test_semicolon_delimited_csv(self, loader):
        csv_text = "id;revenue;churn\n1;1500.50;No\n2;2300.00;Yes\n3;850.25;No\n"
        res = loader.load(raw_bytes=csv_text.encode("utf-8"), filename="financial_data.csv")
        assert res.meta["delimiter"] == ";"
        assert res.df.shape == (3, 3)
        assert list(res.df.columns) == ["id", "revenue", "churn"]
        assert float(res.df["revenue"].iloc[0]) == 1500.50

    def test_tab_separated_tsv(self, loader):
        tsv_text = "user_id\tclick_count\tconverted\nU101\t15\t1\nU102\t3\t0\n"
        res = loader.load(raw_bytes=tsv_text.encode("utf-8"), filename="tracking.tsv")
        assert res.df.shape == (2, 3)
        assert list(res.df.columns) == ["user_id", "click_count", "converted"]

    def test_parquet_columnar_binary(self, loader):
        df_orig = pd.DataFrame({
            "sensor_id": ["S1", "S2", "S3", "S4"],
            "temperature": [22.4, 23.1, 21.8, 24.5],
            "alert": [0, 0, 0, 1],
        })
        buf = io.BytesIO()
        df_orig.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        res = loader.load(raw_bytes=parquet_bytes, filename="telemetry.parquet")
        assert res.meta["file_format"] == "parquet"
        assert res.df.shape == (4, 3)
        assert list(res.df.columns) == ["sensor_id", "temperature", "alert"]
        assert res.df["temperature"].iloc[0] == 22.4

    def test_json_records(self, loader):
        data = [
            {"account_id": "ACC1", "balance": 5000, "status": "active"},
            {"account_id": "ACC2", "balance": 12000, "status": "premium"},
        ]
        json_bytes = json.dumps(data).encode("utf-8")
        res = loader.load(raw_bytes=json_bytes, filename="accounts.json")
        assert res.df.shape == (2, 3)
        assert list(res.df.columns) == ["account_id", "balance", "status"]

    def test_json_lines(self, loader):
        jsonl_text = '{"event": "login", "latency_ms": 42}\n{"event": "checkout", "latency_ms": 128}\n'
        res = loader.load(raw_bytes=jsonl_text.encode("utf-8"), filename="events.jsonl")
        assert res.df.shape == (2, 2)
        assert list(res.df.columns) == ["event", "latency_ms"]

    def test_utf8_bom_stripping(self, loader):
        # UTF-8 with BOM prefix \xef\xbb\xbf
        bom_bytes = b"\xef\xbb\xbfname,score\nAlice,95\nBob,88\n"
        res = loader.load(raw_bytes=bom_bytes, filename="students.csv")
        assert list(res.df.columns) == ["name", "score"]
        assert res.df["name"].iloc[0] == "Alice"

    def test_latin1_encoding_accents(self, loader):
        french_text = "employé;département;salaire\nFrançois;Ventes;45000\nHélène;R&D;58000\n"
        latin1_bytes = french_text.encode("latin-1")
        res = loader.load(raw_bytes=latin1_bytes, filename="personnel.csv")
        assert res.df.shape == (2, 3)
        assert "salaire" in res.df.columns

    def test_header_sanitization_and_deduplication(self):
        raw_headers = ["  user name  ", "user name", "col\twith\nnewlines", "", None]
        cleaned = sanitize_column_headers(raw_headers)
        assert cleaned[0] == "user name"
        assert cleaned[1] == "user name_1"
        assert cleaned[2] == "col with newlines"
        assert cleaned[3] == "column_4"
        assert cleaned[4] == "column_5"


class TestEnvironmentVersatility:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_root_health_probe(self, client):
        """Kubernetes and AWS ALB liveness probe path /health."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["platform"] == platform.system()
        assert data["cpu_cores"] > 0

    def test_api_v1_health_probe(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["platform"] == platform.system()

    def test_multipart_upload_tsv(self, client):
        tsv_content = "feature1\tfeature2\ttarget\n1.0\t2.0\t1\n3.0\t4.0\t0\n"
        resp = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("dataset.tsv", io.BytesIO(tsv_content.encode("utf-8")), "text/tab-separated-values")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_rows"] == 2
        assert data["n_cols"] == 3
        assert "session_id" in data

    def test_config_defaults(self):
        assert HOST is not None
        assert PORT > 0
        assert len(CORS_ORIGINS) > 0
