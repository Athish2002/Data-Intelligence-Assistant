"""
tests/test_system_metrics_and_memory.py
───────────────────────────────────────
Comprehensive test suite for:
1. Thread-safe BoundedSessionStore (LRU eviction, TTL pruning, metadata reporting).
2. Hardware worker safety bounding (safe_n_jobs).
3. /api/v1/system/metrics endpoint (process RSS, system RAM, CPU load, active sessions).
4. /api/v1/system/gc endpoint (garbage collection sweep and memory reclaim).
5. /api/v1/system/sessions/{id} session eviction endpoint.
"""

import time
import pandas as pd
import pytest

from api.server import (
    SESSION_STORE,
    delete_session,
    get_system_metrics,
    ingest_demo_dataset,
    trigger_system_garbage_collection,
)
from api.schemas import DemoIngestRequest
from dia.model_trainer import safe_n_jobs
from dia.session_manager import BoundedSessionStore


def test_safe_n_jobs_bounding():
    """Verify safe_n_jobs strictly caps concurrency to max 4 to avoid process explosion."""
    assert safe_n_jobs(-1) <= 4
    assert safe_n_jobs(0) <= 4
    assert safe_n_jobs(16) <= 4
    assert safe_n_jobs(1) == 1
    assert safe_n_jobs(2) <= 2


def test_bounded_session_store_lru_eviction():
    """Verify BoundedSessionStore evicts oldest sessions when exceeding capacity."""
    store = BoundedSessionStore(max_sessions=3, ttl_seconds=300)

    # Insert 3 sessions
    for i in range(3):
        store[f"s{i}"] = {"df": pd.DataFrame({"a": [1, 2, 3]}), "goal": f"test {i}"}

    assert len(store) == 3
    assert "s0" in store
    assert "s1" in store
    assert "s2" in store

    # Access s0 to make s1 the LRU session
    _ = store["s0"]

    # Insert 4th session - s1 should be evicted (LRU)
    store["s3"] = {"df": pd.DataFrame({"a": [4, 5, 6]}), "goal": "test 3"}
    assert len(store) == 3
    assert "s1" not in store  # Evicted
    assert "s0" in store      # Retained due to recent access
    assert "s2" in store
    assert "s3" in store


def test_bounded_session_store_ttl_expiration():
    """Verify BoundedSessionStore prunes sessions older than TTL."""
    store = BoundedSessionStore(max_sessions=5, ttl_seconds=1)  # 1 second TTL
    store["s_temp"] = {"df": pd.DataFrame({"x": [1]}), "goal": "temporary"}
    assert "s_temp" in store

    # Wait for TTL to elapse
    time.sleep(1.2)
    assert "s_temp" not in store
    assert len(store) == 0


def test_system_metrics_endpoint():
    """Verify GET /api/v1/system/metrics returns accurate hardware telemetry."""
    # Ingest a demo benchmark to ensure at least 1 session
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Telecom Customer Churn"))
    sid = ingest_res.session_id

    metrics = get_system_metrics()
    assert metrics.status == "ok"
    assert metrics.process_memory_rss_mb > 0
    assert metrics.system_memory_total_gb > 0
    assert metrics.system_memory_used_gb > 0
    assert 0 <= metrics.system_memory_percent <= 100
    assert metrics.cpu_cores_logical > 0
    assert metrics.active_sessions_count >= 1
    assert metrics.max_sessions_capacity == 5
    assert metrics.session_ttl_minutes == 30
    assert len(metrics.sessions_detail) >= 1

    # Verify session detail fields
    detail = next(d for d in metrics.sessions_detail if d.session_id == sid)
    assert detail.n_rows > 0
    assert detail.n_cols > 0
    assert detail.memory_mb >= 0


def test_system_gc_endpoint():
    """Verify POST /api/v1/system/gc triggers garbage collection successfully."""
    gc_res = trigger_system_garbage_collection()
    assert gc_res.status == "ok"
    assert gc_res.unreachable_objects_collected >= 0
    assert gc_res.current_rss_mb > 0


def test_system_session_delete_endpoint():
    """Verify DELETE /api/v1/system/sessions/{id} frees specific session."""
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Telecom Customer Churn"))
    sid = ingest_res.session_id
    assert sid in SESSION_STORE

    del_res = delete_session(sid)
    assert del_res["status"] == "success"
    assert sid not in SESSION_STORE
