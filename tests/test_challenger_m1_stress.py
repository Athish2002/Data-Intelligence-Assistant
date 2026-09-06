"""
tests/test_challenger_m1_stress.py
───────────────────────────────────
Adversarial challenge test suite for Milestone M1 offline logic audit.
Designed by Challenger 1 (critic, specialist).

Focus Areas:
1. SQL semicolon stripping with whitespace/repetition combinations.
2. Drift monitor and server endpoint behavior on tiny datasets (0 <= len <= 5).
3. Autoencoder batch size edge cases (len % batch_size == 1, len < batch_size, batch_size == 1).
4. Causal uplift T-Learner single-class subsets and split imbalances.
5. SessionManager.pop() preservation of data across concurrent/serial access and nested structures.
6. Zero-variance column behavior and warning suppression in insights.py.
"""

import concurrent.futures
import sqlite3
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from api.server import SESSION_STORE, get_drift_monitor
from dia.causal_engine import estimate_uplift_t_learner
from dia.drift_monitor import calculate_drift_report
from dia.insights import generate_smart_insights
from dia.session_manager import BoundedSessionStore
from dia.utils import is_torch_available


# ─────────────────────────────────────────────────────────────────────────────
# 1. SQL Semicolon Stripping
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_semicolon_stripping_patterns():
    """Adversarially probe SQL semicolon stripping against SQLite engine."""
    import re

    queries_to_test = [
        "SELECT 1 AS val;",
        "SELECT 1 AS val;   ",
        "SELECT 1 AS val;\t\n  ",
        "SELECT 1 AS val;;;",
        "SELECT 1 AS val; ;  ",
        "SELECT 1 AS val;\n; \t",
        "SELECT ';' AS val;",
    ]

    for raw_q in queries_to_test:
        cleaned = re.sub(r"[\s;]+$", "", raw_q)
        safe_sql = f"SELECT * FROM ({cleaned}) _dia_subq LIMIT 10"

        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        try:
            cur.execute(safe_sql)
            rows = cur.fetchall()
            assert len(rows) == 1
            execution_passed = True
        except Exception:
            execution_passed = False
        finally:
            conn.close()

        assert execution_passed, f"Query failed unexpectedly: {raw_q!r} -> {safe_sql!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Small Dataset (len <= 5) in Drift Monitor & Server
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_rows", [0, 1, 2, 3, 4, 5])
def test_drift_monitor_and_server_small_datasets(n_rows):
    """Verify drift_monitor and get_drift_monitor handle len <= 5 without ZeroDivisionError or crashes."""
    df = pd.DataFrame({
        "num1": np.arange(n_rows, dtype=float),
        "num2": np.arange(n_rows, dtype=float) * 2.5,
        "cat1": [f"group_{i % 2}" for i in range(n_rows)],
    })

    # Test server endpoint
    sid = f"challenger_drift_{n_rows}"
    SESSION_STORE[sid] = {"df": df}
    resp = get_drift_monitor(sid)
    assert resp.status == "success"
    assert resp.psi_score >= 0.0
    assert resp.drift_status in ["NO DRIFT DETECTED", "MODERATE DRIFT", "SEVERE DRIFT"]

    # Test engine function directly across multiple slice combinations
    half = max(1, n_rows // 2)
    if half >= n_rows:
        half = max(1, n_rows - 1)
    ref = df.iloc[:half]
    cur = df.iloc[half:]
    rep = calculate_drift_report(ref, cur)
    assert "drift_detected" in rep
    assert "anomalies" in rep
    assert rep["anomalies"]["anomalies_detected"] >= 0
    assert rep["anomalies"]["anomaly_pct"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Batch Size Edge Cases in Deep Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def test_deep_autoencoder_batch_size_edge_cases():
    """Verify TabularAutoencoderNet handling across varied batch sizes and sample counts."""
    if not is_torch_available():
        pytest.skip("PyTorch is not available in environment.")

    from dia.deep_autoencoder import train_tabular_autoencoder

    feat_names = ["col1", "col2", "col3", "col4"]
    rng = np.random.RandomState(123)

    # Edge Case A: len < 10 returns insufficient_data
    X_small = rng.randn(9, 4).astype(np.float32)
    res_small = train_tabular_autoencoder(X_small, feat_names, epochs=2, batch_size=32)
    assert res_small["status"] == "insufficient_data"

    # Edge Case B: len % batch_size == 1 (e.g. 33 samples, batch_size 32)
    X_33 = rng.randn(33, 4).astype(np.float32)
    res_33 = train_tabular_autoencoder(X_33, feat_names, epochs=2, batch_size=32)
    assert res_33["status"] == "success"
    assert len(res_33["loss_curve"]) == 2

    # Edge Case C: len % batch_size == 1 (e.g. 65 samples, batch_size 32)
    X_65 = rng.randn(65, 4).astype(np.float32)
    res_65 = train_tabular_autoencoder(X_65, feat_names, epochs=2, batch_size=32)
    assert res_65["status"] == "success"

    # Edge Case D: len == batch_size (exact match, e.g. 32 samples, batch_size 32)
    X_32 = rng.randn(32, 4).astype(np.float32)
    res_32 = train_tabular_autoencoder(X_32, feat_names, epochs=2, batch_size=32)
    assert res_32["status"] == "success"

    # Edge Case E: len < batch_size but >= 10 (e.g. 15 samples, batch_size 32)
    X_15 = rng.randn(15, 4).astype(np.float32)
    res_15 = train_tabular_autoencoder(X_15, feat_names, epochs=2, batch_size=32)
    assert res_15["status"] == "success"

    # Edge Case F: batch_size > 1000 (batch_size >> len)
    res_large_b = train_tabular_autoencoder(X_15, feat_names, epochs=2, batch_size=1024)
    assert res_large_b["status"] == "success"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Causal Uplift T-Learner with Single-Class Subsets
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_uplift_single_class_subsets_adversarial():
    """Verify estimate_uplift_t_learner returns insufficient_data for any single-class split."""
    rng = np.random.RandomState(42)
    X = rng.randn(24, 4)
    feat_names = ["f1", "f2", "f3", "f4"]
    clf = LogisticRegression()

    # Scenario 1: Treatment group has only class 1, Control has {0, 1}
    y_s1 = np.array([1]*12 + [0]*6 + [1]*6)
    t_s1 = np.array([1]*12 + [0]*12)
    res_s1 = estimate_uplift_t_learner(clf, X, y_s1, t_s1, feat_names)
    assert res_s1["status"] == "insufficient_data"

    # Scenario 2: Control group has only class 0, Treatment has {0, 1}
    y_s2 = np.array([0]*6 + [1]*6 + [0]*12)
    t_s2 = np.array([1]*12 + [0]*12)
    res_s2 = estimate_uplift_t_learner(clf, X, y_s2, t_s2, feat_names)
    assert res_s2["status"] == "insufficient_data"

    # Scenario 3: Both groups have only class 0
    y_s3 = np.array([0]*24)
    t_s3 = np.array([1]*12 + [0]*12)
    res_s3 = estimate_uplift_t_learner(clf, X, y_s3, t_s3, feat_names)
    assert res_s3["status"] == "insufficient_data"

    # Scenario 4: Both groups have only class 1
    y_s4 = np.array([1]*24)
    t_s4 = np.array([1]*12 + [0]*12)
    res_s4 = estimate_uplift_t_learner(clf, X, y_s4, t_s4, feat_names)
    assert res_s4["status"] == "insufficient_data"

    # Scenario 5: Random Forest classifier also safely handled
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    res_s5 = estimate_uplift_t_learner(rf, X, y_s1, t_s1, feat_names)
    assert res_s5["status"] == "insufficient_data"

    # Scenario 6: Valid balanced distribution succeeds
    y_valid = np.array([0, 1]*6 + [0, 1]*6)
    res_valid = estimate_uplift_t_learner(clf, X, y_valid, t_s1, feat_names)
    assert res_valid["status"] == "success"
    assert "average_treatment_effect_ate" in res_valid
    assert "uplift_quadrant_distribution" in res_valid


# ─────────────────────────────────────────────────────────────────────────────
# 5. SessionManager.pop() Preservation across Concurrent/Serial Access
# ─────────────────────────────────────────────────────────────────────────────

def test_session_manager_pop_serial_and_nested_eviction_defect():
    """Verify SessionManager.pop() preservation of data and expose shallow copy defect."""
    store = BoundedSessionStore(max_sessions=5, ttl_seconds=600)

    # Session with nested pipeline_result
    test_pipe = {"best_model_label": "LightGBM", "score": 0.945, "status": "trained"}
    test_df = pd.DataFrame({"colA": [10, 20, 30], "colB": [40, 50, 60]})
    store["sess_pipe"] = {
        "df": test_df,
        "goal": "Classification",
        "pipeline_result": test_pipe,
    }

    popped = store.pop("sess_pipe")
    assert popped is not None
    assert "df" in popped
    assert len(popped["df"]) == 3
    assert popped.get("goal") == "Classification"

    # Verify nested dictionary preservation:
    # BoundedSessionStore.pop() transfers full ownership of the session to the caller.
    assert popped.get("pipeline_result") == test_pipe, "pipeline_result must be preserved intact during pop()"


def test_session_manager_concurrent_pop_and_set():
    """Verify BoundedSessionStore thread-safety under concurrent access."""
    store = BoundedSessionStore(max_sessions=10, ttl_seconds=300)

    for i in range(10):
        store[f"sess_{i}"] = {"df": pd.DataFrame({"x": [i]}), "id": i}

    def worker_pop(session_id):
        return store.pop(session_id)

    def worker_set(session_id):
        store[session_id] = {"df": pd.DataFrame({"y": [1]}), "id": session_id}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures_pop = [executor.submit(worker_pop, f"sess_{i}") for i in range(10)]
        futures_set = [executor.submit(worker_set, f"sess_new_{i}") for i in range(10)]
        results_pop = [f.result() for f in futures_pop]
        concurrent.futures.wait(futures_set)

    # All popped sessions must return a dict (not None), and store length <= 10
    for res in results_pop:
        assert isinstance(res, dict)
    assert len(store) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# 6. Zero-Variance Column Behavior in insights.py
# ─────────────────────────────────────────────────────────────────────────────

def test_insights_zero_variance_numerical_stability():
    """Verify insights generation on zero-variance columns does not emit RuntimeWarning."""
    df = pd.DataFrame({
        "const_zero": [0.0] * 50,
        "const_val": [42.123] * 50,
        "near_zero_variance": [1.0 + 1e-11 * (i % 2) for i in range(50)],
        "normal_feat": np.linspace(1, 100, 50),
        "target": [0, 1] * 25,
    })

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        insights = generate_smart_insights(df, target_col="target", task_type="classification")

    # Check for any RuntimeWarning regarding division by zero or invalid value
    divide_warnings = [
        w for w in recorded_warnings
        if issubclass(w.category, RuntimeWarning) and "divide" in str(w.message).lower()
    ]
    assert len(divide_warnings) == 0, f"Encountered RuntimeWarning: {[str(w.message) for w in divide_warnings]}"
    assert isinstance(insights, list)
    assert len(insights) > 0


def test_insights_constant_target():
    """Verify insights generation when the target column itself has zero variance."""
    df = pd.DataFrame({
        "normal_feat": np.linspace(1, 100, 50),
        "another_feat": np.random.randn(50),
        "target_const": [1] * 50,
    })

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        insights = generate_smart_insights(df, target_col="target_const", task_type="classification")

    divide_warnings = [
        w for w in recorded_warnings
        if issubclass(w.category, RuntimeWarning) and "divide" in str(w.message).lower()
    ]
    assert len(divide_warnings) == 0, f"Encountered RuntimeWarning: {[str(w.message) for w in divide_warnings]}"
    assert isinstance(insights, list)
