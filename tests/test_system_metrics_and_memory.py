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


def test_session_manager_pop_preserves_data():
    """Verify BoundedSessionStore.pop() returns session data without clearing nested state."""
    store = BoundedSessionStore(max_sessions=3, ttl_seconds=600)
    test_df = pd.DataFrame({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    pipe_payload = {
        "best_model_label": "Random Forest Classifier",
        "score": 0.954,
        "status": "trained",
        "metrics": {"accuracy": 0.954, "f1": 0.951},
    }
    store["sess_pop_test"] = {
        "df": test_df,
        "goal": "Test session eviction",
        "custom_key": 42,
        "pipeline_result": pipe_payload,
    }

    assert "sess_pop_test" in store
    popped = store.pop("sess_pop_test")

    assert popped is not None, "pop() should return the stored session"
    assert isinstance(popped, dict), "popped value must be a dictionary"
    assert popped.get("custom_key") == 42
    assert "df" in popped
    assert "pipeline_result" in popped
    assert popped["pipeline_result"] == pipe_payload
    assert "sess_pop_test" not in store, "session must be evicted from internal store"


def test_session_manager_concurrent_pop_and_set():
    """Stress test BoundedSessionStore thread-safety under concurrent pop and insertion."""
    import concurrent.futures
    store = BoundedSessionStore(max_sessions=10, ttl_seconds=300)

    def worker(idx):
        sid = f"thread_sess_{idx}"
        store[sid] = {"df": pd.DataFrame({"x": [idx]}), "id": idx}
        time.sleep(0.001)
        return store.pop(sid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(40)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 40
    assert len(store) <= 10


def test_autoencoder_input_row_capping_50k():
    """Verify deep autoencoder caps inputs exceeding 50,000 rows to bound tensor memory."""
    import numpy as np
    from dia.deep_autoencoder import train_tabular_autoencoder

    n_rows = 50_050
    n_cols = 4
    rng = np.random.RandomState(42)
    X = rng.randn(n_rows, n_cols).astype(np.float32)
    feat_names = [f"col_{i}" for i in range(n_cols)]

    res = train_tabular_autoencoder(
        X,
        feat_names,
        epochs=1,
        batch_size=1024,
        eval_batch_size=1024,
    )
    assert res["status"] == "success"
    assert res["latent_embeddings_shape"][0] == 50_000
    assert np.isfinite(res["final_reconstruction_loss"])

