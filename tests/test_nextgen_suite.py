"""
tests/test_nextgen_suite.py
───────────────────────────
Comprehensive test suite verifying Next-Generation Algorithmic & Governance Engines:
L1. Algorithmic Recourse & Counterfactual Explanations (dia/recourse_engine.py)
L2. Monte Carlo Macroeconomic Stress Testing (dia/stress_testing.py)
L3. Conformal Prediction & Epistemic Uncertainty Bounds (dia/conformal_uncertainty.py)
L4. Automated Symbolic Feature Discovery (dia/symbolic_features.py)
L5. Drift Sentinel Multivariate MMD & Governance Triaging (dia/drift_sentinel.py)
U4. Executive Audit Dossier Export Route (api/server.py)
"""

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from starlette.testclient import TestClient

from dia.recourse_engine import (
    AlgorithmicRecourseEngine,
    compute_recourse,
    RecourseResult,
    RecourseAction,
)
from dia.stress_testing import (
    MonteCarloStressTester,
    run_stress_test,
    StressTestReport,
    ScenarioResult,
)
from dia.conformal_uncertainty import (
    ConformalUncertaintyEngine,
    evaluate_conformal_bounds,
    ConformalCalibrationReport,
    ConformalInstanceResult,
)
from dia.symbolic_features import (
    SymbolicFeatureDiscovery,
    discover_symbolic_features,
    SymbolicDiscoveryReport,
    DiscoveredFormula,
)
from dia.drift_sentinel import (
    DriftSentinel,
    audit_drift_sentinel,
    DriftSentinelReport,
    FeatureDriftDetail,
)
from api.server import app


# ─── L1: Algorithmic Recourse Tests ──────────────────────────────────────────

def test_recourse_engine_counterfactual_flip():
    """Verify that recourse optimization flips prediction to target label."""
    np.random.seed(42)
    # Binary dataset: feature x1 > 0 means class 1, x1 <= 0 means class 0
    X = np.random.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)
    feature_names = ["income", "credit_score", "tenure"]
    df = pd.DataFrame(X, columns=feature_names)

    model = LogisticRegression()
    model.fit(X, y)

    # An instance firmly predicted as class 0
    query = {"income": -2.0, "credit_score": -1.0, "tenure": 0.0}
    
    engine = AlgorithmicRecourseEngine(max_iterations=100)
    result = engine.compute_recourse(
        model=model,
        x_input=query,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
    )

    assert isinstance(result, RecourseResult)
    assert result.original_prediction == 0
    assert result.target_prediction == 1
    assert result.total_recourse_cost >= 0.0
    assert len(result.actions) > 0
    assert len(result.counterfactual_vector) == len(feature_names)

    # Ensure counterfactual produces the desired prediction
    cf_vec = np.array([[result.counterfactual_vector[f] for f in feature_names]])
    assert model.predict(cf_vec)[0] == 1


def test_recourse_engine_immutable_features():
    """Verify that immutable features are strictly locked during recourse search."""
    np.random.seed(42)
    X = np.random.randn(150, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = ["income", "debt", "customer_age"]
    df = pd.DataFrame(X, columns=feature_names)

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)

    query = {"income": -1.5, "debt": 1.5, "customer_age": 45.0}
    result = compute_recourse(
        model=model,
        x_input=query,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
        immutable_features=["customer_age"],
    )

    # customer_age must not have changed at all
    assert result.counterfactual_vector["customer_age"] == query["customer_age"]
    for act in result.actions:
        assert act.feature != "customer_age"


# ─── L2: Monte Carlo Stress Testing Tests ────────────────────────────────────

def test_stress_testing_scenarios_and_var():
    """Verify Monte Carlo simulation under macroeconomic shock scenarios."""
    np.random.seed(42)
    # Correlated portfolio features
    mean = [50000, 700, 30]
    cov = [
        [1e8, 2e5, 1e4],
        [2e5, 2500, 50],
        [1e4, 50, 25],
    ]
    X = np.random.multivariate_normal(mean, cov, size=200)
    feature_names = ["income", "score", "tenure"]
    df = pd.DataFrame(X, columns=feature_names)

    # Outcome model
    y = ((df["income"] * 0.001 + df["score"] * 0.1 - 70) > 40).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(df, y)

    result = run_stress_test(
        model=model,
        df=df,
        feature_names=feature_names,
        task_type="classification",
    )

    assert isinstance(result, StressTestReport)
    assert result.n_simulations == 250
    assert len(result.scenarios) > 0

    for sc in result.scenarios:
        assert isinstance(sc, ScenarioResult)
        assert sc.var_95 >= 0.0
        assert sc.var_99 >= sc.var_95
        assert sc.cvar_95 >= sc.var_95
        assert len(sc.tail_distribution) > 0


def test_stress_testing_custom_shocks():
    """Verify custom shock vectors."""
    np.random.seed(42)
    X = np.random.randn(150, 2)
    feature_names = ["feat_a", "feat_b"]
    df = pd.DataFrame(X, columns=feature_names)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    tester = MonteCarloStressTester(n_simulations=100)
    report = tester.run_stress_test(
        model=model,
        df=df,
        feature_names=feature_names,
        custom_shocks={"feat_a": -0.5, "feat_b": 0.3},
    )

    assert any(s.scenario_name in ("CUSTOM_SHOCK_POLICY", "CUSTOM_USER_SHOCK") for s in report.scenarios)


# ─── L3: Conformal Uncertainty Bounds Tests ──────────────────────────────────

def test_conformal_uncertainty_calibration_coverage():
    """Verify split conformal prediction sets satisfy empirical finite-sample coverage."""
    np.random.seed(42)
    X = np.random.randn(300, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    feature_names = ["x1", "x2", "x3"]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    model = LogisticRegression()
    model.fit(X, y)

    alpha = 0.10
    report = evaluate_conformal_bounds(
        model=model,
        df=df,
        target_col="target",
        feature_names=feature_names,
        alpha=alpha,
        task_type="classification",
    )

    assert isinstance(report, ConformalCalibrationReport)
    assert report.coverage_guarantee_pct == 90.0
    # Conformal guarantee: empirical coverage should be near 1 - alpha
    assert report.empirical_coverage >= 0.75
    assert len(report.sample_evaluations) > 0
    for sample in report.sample_evaluations:
        assert isinstance(sample, ConformalInstanceResult)
        assert len(sample.conformal_set) >= 1
        assert sample.coverage_guarantee_pct == 90.0


# ─── L4: Symbolic Feature Discovery Tests ────────────────────────────────────

def test_symbolic_feature_discovery():
    """Verify algebraic interaction generation, MI ranking, and SQL synthesis."""
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(1, 10, n)
    x2 = np.random.uniform(1, 10, n)
    x3 = np.random.randn(n)
    # Non-linear interaction: x1 / x2
    y = (x1 / x2) + 0.05 * np.random.randn(n)

    df = pd.DataFrame({"ratio_num": x1, "ratio_den": x2, "noise": x3, "target": y})

    report = discover_symbolic_features(
        df=df,
        target_col="target",
        max_candidates=40,
        top_k=5,
    )

    assert isinstance(report, SymbolicDiscoveryReport)
    assert report.target_column == "target"
    assert len(report.formulas) > 0
    assert "SELECT" in report.consolidated_sql_view
    assert "FROM raw_features" in report.consolidated_sql_view
    assert len(report.consolidated_python_transform) > 0

    # Verify discovered formulas
    first_f = report.formulas[0]
    assert isinstance(first_f, DiscoveredFormula)
    assert len(first_f.base_features) >= 1
    assert first_f.mutual_info_score >= 0.0


# ─── L5: Drift Sentinel Tests ────────────────────────────────────────────────

def test_drift_sentinel_in_distribution_healthy():
    """Verify that samples from identical distribution yield healthy status."""
    np.random.seed(42)
    ref = pd.DataFrame(np.random.randn(200, 3), columns=["f1", "f2", "f3"])
    cur = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])

    sentinel = DriftSentinel(p_value_threshold=0.05, psi_threshold=0.25)
    report = sentinel.audit_drift(reference_df=ref, current_df=cur)

    assert isinstance(report, DriftSentinelReport)
    assert report.overall_sentinel_status in ("PASS_HEALTHY", "WARN_MONITOR")
    assert report.multivariate_mmd_score >= 0.0
    assert report.total_features_monitored == 3


def test_drift_sentinel_shifted_distribution_triggers_critical():
    """Verify that heavily shifted distribution triggers warning or critical action."""
    np.random.seed(42)
    ref = pd.DataFrame(np.random.normal(0, 1, size=(200, 3)), columns=["f1", "f2", "f3"])
    # Heavy shift of 3 standard deviations
    cur = pd.DataFrame(np.random.normal(3.0, 1, size=(100, 3)), columns=["f1", "f2", "f3"])

    report = audit_drift_sentinel(reference_df=ref, current_df=cur)

    assert report.overall_sentinel_status in ("WARN_MONITOR", "CRITICAL_DRIFT")
    assert report.drifting_feature_count >= 2
    assert report.governance_action in ("ACTIVATE_SAFE_FALLBACK", "TRIGGER_RETRAIN", "WARN_MONITOR")


# ─── API Endpoints & Dossier Export Tests ─────────────────────────────────────

def test_api_nextgen_endpoints():
    """Verify all next-gen REST endpoints via FastAPI TestClient."""
    client = TestClient(app)

    # 1. Ingest a standard demo dataset to seed an active session
    ingest_resp = client.post("/api/v1/ingest/demo", json={"demo_name": "Telecom Customer Churn"})
    assert ingest_resp.status_code == 200
    session_id = ingest_resp.json()["session_id"]

    # 2. GET /api/v1/recourse/sample/{session_id}
    sample_resp = client.get(f"/api/v1/recourse/sample/{session_id}")
    assert sample_resp.status_code == 200
    sample_data = sample_resp.json()
    assert "sample_row" in sample_data
    assert "feature_names" in sample_data

    # 3. POST /api/v1/recourse/counterfactual
    cf_payload = {
        "session_id": session_id,
        "row_index": 0,
        "desired_class": 1,
    }
    cf_resp = client.post("/api/v1/recourse/counterfactual", json=cf_payload)
    assert cf_resp.status_code == 200
    cf_data = cf_resp.json()
    assert "counterfactual_vector" in cf_data
    assert "actions" in cf_data

    # 4. POST /api/v1/simulation/stress-test
    stress_payload = {
        "session_id": session_id,
        "n_simulations": 100,
    }
    stress_resp = client.post("/api/v1/simulation/stress-test", json=stress_payload)
    assert stress_resp.status_code == 200
    stress_data = stress_resp.json()
    assert "scenarios" in stress_data
    assert "overall_resilience_grade" in stress_data

    # 5. POST /api/v1/uncertainty/conformal-bounds
    conformal_payload = {
        "session_id": session_id,
        "alpha": 0.10,
    }
    conf_resp = client.post("/api/v1/uncertainty/conformal-bounds", json=conformal_payload)
    assert conf_resp.status_code == 200
    conf_data = conf_resp.json()
    assert conf_data["coverage_guarantee_pct"] == 90.0
    assert "sample_evaluations" in conf_data

    # 6. POST /api/v1/features/symbolic-discovery/{session_id}
    symb_payload = {
        "session_id": session_id,
        "max_candidates": 30,
        "top_k": 3,
    }
    symb_resp = client.post(f"/api/v1/features/symbolic-discovery/{session_id}", json=symb_payload)
    assert symb_resp.status_code == 200
    symb_data = symb_resp.json()
    assert "formulas" in symb_data
    assert "consolidated_sql_view" in symb_data

    # 7. POST /api/v1/monitoring/drift-sentinel
    drift_payload = {
        "session_id": session_id,
        "sample_fraction": 0.35,
        "synthetic_shift_strength": 0.5,
    }
    drift_resp = client.post("/api/v1/monitoring/drift-sentinel", json=drift_payload)
    assert drift_resp.status_code == 200
    drift_data = drift_resp.json()
    assert "multivariate_mmd_score" in drift_data
    assert "governance_action" in drift_data
    assert "actionable_recommendations" in drift_data

    # 8. GET /api/v1/export/{session_id}/dossier
    dossier_resp = client.get(f"/api/v1/export/{session_id}/dossier")
    assert dossier_resp.status_code == 200
    assert "text/html" in dossier_resp.headers.get("content-type", "")
    html_text = dossier_resp.text
    assert "Executive Audit Dossier" in html_text
    assert "Data Intelligence Assistant" in html_text
    assert "REGULATORY INTEGRITY PASSED" in html_text
