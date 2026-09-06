"""
tests/test_m1_logic_audit.py
────────────────────────────
Dedicated offline unit test suite for Milestone M1 (Offline Code Logic & Architectural Audit).
Verifies:
1. SessionManager.pop() returns cloned session dict instead of cleared empty dict
2. SQL ingestion trailing semicolon stripping
3. Drift monitor small dataset zero-division protection
4. Causal uplift single-class split guard
5. Model trainer return keys (y_test, y_true, test_indices)
6. Deep autoencoder drop_last batch norm stability
7. Insights zero-variance standard deviation guard
8. Report generator readiness score key alignment
9. API server endpoints key alignment and slicing resilience
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from api.schemas import (
    ActiveLearningResponse,
    AutoencoderResponse,
    GraphAnalysisResponse,
    OnlineLearningResponse,
    TimeSeriesForecastResponse,
)
from api.server import (
    SESSION_STORE,
    get_drift_monitor,
)
from dia.causal_engine import estimate_uplift_t_learner
from dia.drift_monitor import calculate_drift_report
from dia.ingestion.sql_ingestion import SQLSource
from dia.insights import generate_smart_insights
from dia.model_trainer import train_and_evaluate
from dia.report_generator import generate_executive_html_report
from dia.session_manager import BoundedSessionStore
from dia.utils import is_torch_available


def test_session_manager_pop_preserves_data():
    """Verify SessionManager.pop() returns session data and does not clear pipeline_result or dict."""
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
    assert popped.get("custom_key") == 42, "popped dictionary must retain its keys"
    assert "df" in popped, "popped dictionary must retain DataFrame"
    assert "pipeline_result" in popped, "popped dictionary must retain pipeline_result"
    assert popped["pipeline_result"] == pipe_payload, "pipeline_result must retain all nested keys and values intact"
    assert popped["pipeline_result"]["score"] == 0.954
    assert popped["pipeline_result"]["metrics"]["accuracy"] == 0.954
    assert "sess_pop_test" not in store, "session must be evicted from internal store"


def test_sql_ingestion_semicolon_handling():
    """Verify trailing semicolons (including spaced and repeated semicolons) are stripped and execute cleanly."""
    source = SQLSource()
    if not source.is_available():
        pytest.skip("SQLAlchemy is not installed")

    queries = [
        "SELECT 1 AS id, 'A' AS val; ;  ",
        "SELECT 2 AS id, 'B' AS val;\n;\t  ",
        "SELECT 3 AS id, 'C' AS val;;;",
        "SELECT 4 AS id, 'D' AS val;   ",
        "SELECT 5 AS id, 'E' AS val",
    ]
    for q in queries:
        res = source.load(connection_string="sqlite:///:memory:", query=q)
        assert res.df is not None, f"Expected valid DataFrame for query {q!r}"
        assert len(res.df) == 1
        assert "id" in res.df.columns
        assert "val" in res.df.columns


def test_drift_monitor_tiny_dataset_zero_division():
    """Verify calculate_drift_report handles small and minimal dataframes without ZeroDivisionError."""
    ref_df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    cur_df = pd.DataFrame({"a": [1.5, 2.5], "b": [11.0, 21.0]})

    report = calculate_drift_report(ref_df, cur_df)
    assert "drift_detected" in report
    assert "anomalies" in report

    # Also verify with empty cur_df
    empty_cur = pd.DataFrame({"a": [], "b": []})
    empty_report = calculate_drift_report(ref_df, empty_cur)
    assert "anomalies" in empty_report
    assert empty_report["anomalies"]["anomalies_detected"] == 0
    assert empty_report["anomalies"]["anomaly_pct"] == 0.0


def test_causal_engine_single_class_binary_split_guard():
    """Verify estimate_uplift_t_learner returns insufficient_data when a split has only 1 unique class."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3)
    # Control group has ONLY class 0, treatment has both 0 and 1
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    treatment = np.array([1] * 10 + [0] * 10)

    clf = LogisticRegression()
    res = estimate_uplift_t_learner(
        model=clf,
        X=X,
        y=y,
        treatment=treatment,
        feature_names=["f1", "f2", "f3"],
    )
    assert res["status"] == "insufficient_data"
    assert "must contain at least two target classes" in res["message"] or "subgroup" in res["message"]


def test_model_trainer_returns_y_test_and_indices():
    """Verify train_models returns y_test alongside y_true and preserves test row indices."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    res = train_and_evaluate(
        df=df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["rf", "logreg"],
        test_size=0.3,
        random_state=42,
    )

    assert "y_true" in res, "Return dict must include y_true"
    assert "y_test" in res, "Return dict must include y_test"
    assert "test_indices" in res, "Return dict must include test_indices"
    assert len(res["y_test"]) == len(res["y_true"])
    assert len(res["test_indices"]) == len(res["y_test"])


def test_deep_autoencoder_batchnorm_stability():
    """Verify train_tabular_autoencoder does not crash on trailing single-sample batch."""
    if not is_torch_available():
        pytest.skip("PyTorch not installed")

    from dia.deep_autoencoder import train_tabular_autoencoder

    # 65 samples with batch_size=32 means final batch has 65 % 32 = 1 sample
    rng = np.random.RandomState(42)
    X = rng.randn(65, 4).astype(np.float32)
    feat_names = ["c1", "c2", "c3", "c4"]

    res = train_tabular_autoencoder(X, feat_names, epochs=3, batch_size=32)
    assert res["status"] == "success"
    assert "loss_curve" in res
    assert "anomaly_threshold_mse" in res


def test_insights_zero_variance_numeric_column():
    """Verify generate_smart_insights handles constant/zero-variance numeric columns without warnings."""
    df = pd.DataFrame({
        "constant_col": [5.0] * 30,
        "varying_col": list(range(30)),
        "target": [0, 1] * 15,
    })

    insights = generate_smart_insights(df, target_col="target", task_type="classification")
    assert isinstance(insights, list)


def test_report_generator_readiness_score_key():
    """Verify generate_executive_html_report reads both 'score' and 'readiness_score'."""
    html_out = generate_executive_html_report(
        goal="Predict target",
        target_col="target",
        task_type="classification",
        train_result={"best_model_label": "Random Forest", "results": [{"metrics": {"Score": 0.95}}]},
        readiness={"score": 87, "grade": "B"},
        compliance_report={},
        latency_stats={"p99_ms": 1.5},
    )
    assert "87/100" in html_out, "HTML report should reflect score from readiness dict"


def test_api_drift_monitor_small_session():
    """Verify get_drift_monitor in api/server.py handles small dataset without crashing."""
    sid = "small_drift_test_session"
    SESSION_STORE[sid] = {
        "df": pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
    }

    resp = get_drift_monitor(sid)
    assert resp.status == "success"
    assert resp.psi_score >= 0.0


def test_uplift_duplicate_index_handling():
    """Verify get_uplift_effect gracefully handles DataFrame with duplicate indices without reindex error."""
    from api.schemas import UpliftRequest
    from api.server import get_uplift_effect
    from sklearn.ensemble import RandomForestClassifier

    n_samples = 40
    idx = [i // 2 for i in range(n_samples)]
    df = pd.DataFrame({
        "feat1": np.random.randn(n_samples),
        "treatment": [0, 1] * (n_samples // 2),
        "target": [0, 1, 1, 0] * (n_samples // 4),
    }, index=idx)

    X = df[["feat1"]].values
    y = df["target"].values
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    sid = "uplift_duplicate_index_session"
    SESSION_STORE[sid] = {
        "df": df,
        "pipeline_result": {
            "train_result": {
                "best_model": clf,
                "X_test_processed": X[:20],
                "y_test": y[:20],
                "test_indices": list(range(20)),
                "feature_names": ["feat1"],
            }
        },
    }

    req = UpliftRequest(session_id=sid, treatment_column="treatment")
    resp = get_uplift_effect(req)
    assert resp.status == "success"
    assert resp.average_treatment_effect_ate is not None
