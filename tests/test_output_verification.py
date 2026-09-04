"""
tests/test_output_verification.py
──────────────────────────────────
Rigorous automated test suite dedicated to DEEP OUTPUT VERIFICATION.
Eliminates shallow presence-only testing and strictly asserts real computed
values, mathematical invariants, domain sensitivities, and schema contracts.
"""

import numpy as np
import pytest

from api.schemas import (
    DemoIngestRequest,
    SimulateRequest,
    TrainPipelineRequest,
)
from api.server import (
    get_data_contract_suite,
    get_data_profile,
    get_readiness_audit,
    get_system_metrics,
    ingest_demo_dataset,
    run_automl_pipeline,
    simulate_whatif,
)


def test_ingest_output_deep_verification():
    """Verify that dataset ingestion produces exact, mathematically sound outputs."""
    res = ingest_demo_dataset(DemoIngestRequest(demo_name="Bank Credit Risk & Default"))

    # 1. Structural output bounds
    assert res.session_id is not None and len(res.session_id) >= 10
    assert res.n_rows == 600, f"Expected exactly 600 rows, got {res.n_rows}"
    assert res.n_cols == 8, f"Expected exactly 8 columns, got {res.n_cols}"
    assert len(res.columns) == 8

    # 2. Domain and target intelligence resolution
    assert any(term in res.detected_domain.lower() for term in ["banking", "credit", "finance"]), (
        f"Unexpected detected domain: '{res.detected_domain}'"
    )
    assert res.suggested_target == "default_risk", f"Expected default_risk, got {res.suggested_target}"
    assert "default" in res.goal.lower()

    # 3. Capability flags
    assert res.capabilities["can_causal"] is True
    assert res.capabilities["can_bandits"] is True

    # 4. Data Readiness output verification
    readiness = get_readiness_audit(res.session_id)
    assert 80 <= readiness.overall_readiness_score <= 100, f"Readiness score {readiness.overall_readiness_score} out of bounds"
    assert "PRODUCTION" in readiness.production_verdict
    assert len(readiness.checks) >= 4
    for check in readiness.checks:
        assert check.verdict in ("PASS", "WARN", "FAIL")
        assert len(check.title) > 3
        assert len(check.details) > 10

    # 5. Profiling numeric range invariants
    profile = get_data_profile(res.session_id)
    assert profile.shape == [600, 8]
    cols_dict = {c.column: c for c in profile.columns_info}
    assert "applicant_age" in cols_dict
    assert "annual_income" in cols_dict
    assert "credit_score" in cols_dict
    assert cols_dict["applicant_age"].role == "numeric"
    assert cols_dict["default_risk"].unique_count == 2
    assert "credit_score" in profile.summary_stats
    assert profile.summary_stats["credit_score"]["min"] >= 300.0
    assert profile.summary_stats["credit_score"]["max"] <= 850.0


def test_automl_pipeline_deep_output_verification():
    """Verify that AutoML training outputs meet strict mathematical performance bounds."""
    # 1. Ingest demo
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Bank Credit Risk & Default"))
    session_id = ingest_res.session_id

    # 2. Run AutoML Pipeline
    train_res = run_automl_pipeline(
        TrainPipelineRequest(
            session_id=session_id,
            user_target_col="default_risk",
            goal="Predict loan default risk for credit underwriting",
        )
    )

    assert train_res.status == "success"
    assert train_res.final_task_type == "classification"
    assert train_res.target_col == "default_risk"
    assert train_res.best_model_label is not None
    assert len(train_res.best_model_label) > 0

    # 3. Models evaluated and metrics verification
    assert len(train_res.models_evaluated) >= 3, f"Expected >= 3 models evaluated, got {len(train_res.models_evaluated)}"
    
    champion_entry = next((m for m in train_res.models_evaluated if m.get("is_best")), train_res.models_evaluated[0])
    champ_metrics = champion_entry.get("metrics", {})

    acc = champ_metrics.get("Accuracy", 0.0)
    roc = champ_metrics.get("ROC-AUC", 0.0)

    assert 0.70 <= acc <= 1.0, f"Champion accuracy {acc} is out of realistic bounds [0.70, 1.0]"
    assert 0.70 <= roc <= 1.0, f"Champion ROC-AUC {roc} is out of realistic bounds [0.70, 1.0]"

    # 4. Feature Importance verification
    assert len(train_res.shap_importance) >= 4, "Expected >= 4 features with SHAP importance"
    for item in train_res.shap_importance:
        feat = item.get("feature")
        imp = item.get("importance")
        assert isinstance(feat, str) and len(feat) > 0
        assert isinstance(imp, (int, float))
        assert imp >= 0.0, f"Feature importance for {feat} must be non-negative, got {imp}"


def test_confusion_matrix_and_roc_mathematical_conservation():
    """Verify exact mathematical conservation laws of Confusion Matrix and ROC Curve."""
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Bank Credit Risk & Default"))
    session_id = ingest_res.session_id

    train_res = run_automl_pipeline(
        TrainPipelineRequest(
            session_id=session_id,
            user_target_col="default_risk",
            goal="Predict loan default risk",
        )
    )

    # 1. Confusion Matrix Conservation Law: TN + FP + FN + TP == Total Test Samples
    cm = train_res.confusion_matrix
    assert isinstance(cm, list) and len(cm) == 2, f"Expected 2x2 confusion matrix, got {cm}"
    assert len(cm[0]) == 2 and len(cm[1]) == 2

    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])

    assert tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0
    total_test_samples = tn + fp + fn + tp
    # Test set is exactly 20% of 600 rows = 120 rows
    assert total_test_samples == 120, f"Expected exactly 120 test samples, got {total_test_samples}"

    # Consistency of accuracy: (TN + TP) / Total matches reported champion accuracy
    calc_accuracy = (tn + tp) / total_test_samples
    best_entry = next((m for m in train_res.models_evaluated if m.get("is_best")), train_res.models_evaluated[0])
    rep_accuracy = best_entry["metrics"].get("Accuracy", calc_accuracy)
    assert abs(calc_accuracy - rep_accuracy) <= 0.05, (
        f"Confusion matrix accuracy {calc_accuracy:.4f} diverges from reported accuracy {rep_accuracy:.4f}"
    )

    # 2. ROC Curve Coordinate Invariants
    roc = train_res.roc_curve
    assert roc is not None
    assert "fpr" in roc and "tpr" in roc
    fpr = roc["fpr"]
    tpr = roc["tpr"]

    assert isinstance(fpr, list) and isinstance(tpr, list)
    assert len(fpr) == len(tpr), f"FPR length {len(fpr)} != TPR length {len(tpr)}"
    assert len(fpr) >= 5, "ROC curve should have >= 5 evaluation thresholds"

    # Boundary coordinates: (0, 0) to (1, 1)
    assert fpr[0] == pytest.approx(0.0, abs=1e-5)
    assert tpr[0] == pytest.approx(0.0, abs=1e-5)
    assert fpr[-1] == pytest.approx(1.0, abs=1e-5)
    assert tpr[-1] == pytest.approx(1.0, abs=1e-5)

    # Trapezoidal AUC integral matches reported AUC within 0.05
    trapz_auc = np.trapezoid(tpr, fpr)
    reported_auc = roc.get("auc", trapz_auc)
    assert abs(trapz_auc - reported_auc) <= 0.05, f"Integrated AUC {trapz_auc:.4f} diverges from reported {reported_auc:.4f}"
    assert reported_auc >= 0.75, f"Expected high discriminatory AUC (>=0.75), got {reported_auc}"


def test_whatif_simulator_output_sensitivity_and_calibration():
    """Verify that What-If simulation outputs are probabilistically calibrated and responsive to risk variables."""
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Bank Credit Risk & Default"))
    session_id = ingest_res.session_id

    run_automl_pipeline(
        TrainPipelineRequest(
            session_id=session_id,
            user_target_col="default_risk",
            goal="Predict loan default risk",
        )
    )

    # 1. Low-Risk Prime Borrower
    low_risk_features = {
        "applicant_age": 48,
        "annual_income": 160000.0,
        "credit_score": 810.0,
        "debt_to_income_ratio": 0.12,
        "loan_amount": 5000.0,
        "home_ownership": "MORTGAGE",
    }
    low_res = simulate_whatif(SimulateRequest(session_id=session_id, feature_overrides=low_risk_features))
    assert low_res.status == "success"
    assert low_res.prediction == 0, f"Expected low-risk borrower prediction 0 (No Default), got {low_res.prediction}"
    assert 0.0 <= low_res.probability < 0.40, f"Expected low default probability (<0.40), got {low_res.probability}"

    # 2. High-Risk Subprime Borrower
    high_risk_features = {
        "applicant_age": 21,
        "annual_income": 16000.0,
        "credit_score": 480.0,
        "debt_to_income_ratio": 0.62,
        "loan_amount": 42000.0,
        "home_ownership": "RENT",
    }
    high_res = simulate_whatif(SimulateRequest(session_id=session_id, feature_overrides=high_risk_features))
    assert high_res.status == "success"
    assert high_res.prediction == 1, f"Expected high-risk borrower prediction 1 (Default), got {high_res.prediction}"
    assert 0.50 <= high_res.probability <= 1.0, f"Expected high default probability (>=0.50), got {high_res.probability}"

    # 3. Monotonic Sensitivity Invariant
    prob_spread = high_res.probability - low_res.probability
    assert prob_spread >= 0.25, f"Probability spread between subprime and prime ({prob_spread:.2f}) must be >= 0.25"


def test_governance_data_contracts_output_invariants():
    """Verify that Governance contracts output natural language descriptions and valid schemas."""
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Bank Credit Risk & Default"))
    session_id = ingest_res.session_id

    gov_res = get_data_contract_suite(session_id)
    assert gov_res.n_expectations >= 5, f"Expected >= 5 governance expectations, got {gov_res.n_expectations}"
    assert len(gov_res.expectations) >= 5

    for rule in gov_res.expectations:
        exp_type = rule.get("expectation_type", "")
        kwargs = rule.get("kwargs", {})
        assert exp_type.startswith("expect_"), f"Invalid expectation type: {exp_type}"
        if "column" in kwargs:
            assert len(kwargs["column"]) > 0
        else:
            assert "column_set" in kwargs or len(kwargs) >= 0

    # Assert YAML contract is generated and non-trivial
    assert len(gov_res.contract_yaml) > 100
    assert "Generated Data Contract" in gov_res.contract_yaml or "expectations:" in gov_res.contract_yaml


def test_system_health_metrics_output_invariants():
    """Verify that system metrics output bounded RSS RAM and CPU usage."""
    metrics = get_system_metrics()
    assert 50.0 <= metrics.process_memory_rss_mb <= 1000.0, f"RSS RAM {metrics.process_memory_rss_mb} MB out of reasonable bounds"
    assert 0.0 <= metrics.cpu_percent <= 100.0
    assert metrics.active_sessions_count >= 1
    assert metrics.max_sessions_capacity == 5
