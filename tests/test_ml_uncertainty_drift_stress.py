"""
tests/test_ml_uncertainty_drift_stress.py
─────────────────────────────────────────
Targeted tests for ML Uncertainty, Drift Governance & Active Learning fixes:
- M1: Conformal prediction with string target labels & graceful serialization
- M2: Population Stability Index (PSI) out-of-bounds drift detection
- M3: Active learning 1D uncertainty scoring prioritization (p=0.5 vs p=0.99)
- M6: Empirical feature perturbation correlation in stress testing & zero-variance fallback
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dia.active_learning import sample_uncertain_predictions
from dia.conformal_uncertainty import ConformalUncertaintyEngine, evaluate_conformal_bounds
from dia.drift_monitor import calculate_psi
from dia.stress_testing import MonteCarloStressTester


def test_conformal_string_labels():
    """M1: Test ConformalUncertaintyEngine handles string labels correctly."""
    X = pd.DataFrame({
        "feature_1": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5, 1.2, 8.8],
        "feature_2": [0.5, 0.8, 0.6, 7.2, 8.1, 7.9, 0.7, 8.0],
    })
    y = np.array(["cat", "cat", "cat", "dog", "dog", "dog", "cat", "dog"])

    clf = LogisticRegression().fit(X, y)
    engine = ConformalUncertaintyEngine(alpha=0.10)

    # 1. Calibration on string labels
    q_hat = engine.calibrate(clf, X, y, task_type="classification")
    assert 0.0 <= q_hat <= 1.0

    # 2. Evaluation on test instances
    report = engine.evaluate_instances(clf, X, y)
    assert report.empirical_coverage >= 0.75
    assert len(report.sample_evaluations) == len(X)

    for item in report.sample_evaluations:
        assert isinstance(item.predicted_label, str)
        assert item.predicted_label in ["cat", "dog"]
        assert isinstance(item.conformal_set, list)
        for cls in item.conformal_set:
            assert isinstance(cls, str)
            assert cls in ["cat", "dog"]

    # 3. Test functional wrapper with DataFrame
    df = X.copy()
    df["target"] = y
    rep_df = evaluate_conformal_bounds(
        clf,
        df,
        target_col="target",
        feature_names=["feature_1", "feature_2"],
        alpha=0.10,
    )
    assert rep_df.executive_verdict == "CONFORMAL_COVERAGE_VERIFIED"


def test_drift_psi_out_of_bounds_detection():
    """M2: Test calculate_psi detects massive drift when distributions are non-overlapping."""
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(0, 1, 1000))
    s2 = pd.Series(rng.normal(10, 1, 1000))

    psi_ood = calculate_psi(s1, s2)
    assert psi_ood > 0.5, f"Expected PSI > 0.5 for N(0,1) vs N(10,1), got {psi_ood}"

    # Negative extreme shift
    s3 = pd.Series(rng.normal(-10, 1, 1000))
    psi_neg_ood = calculate_psi(s1, s3)
    assert psi_neg_ood > 0.5, f"Expected PSI > 0.5 for N(0,1) vs N(-10,1), got {psi_neg_ood}"

    # In-distribution should have low PSI (< 0.10)
    s_in = pd.Series(rng.normal(0, 1, 1000))
    psi_in = calculate_psi(s1, s_in)
    assert psi_in < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi_in}"


def test_active_learning_prioritization_ambiguity():
    """M3: Test active learning gives highest uncertainty to p=0.5 over confident predictions."""
    class Mock1DProbaModel:
        def predict_proba(self, X):
            # Row 0: 0.50 (maximum ambiguity)
            # Row 1: 0.99 (highly confident)
            # Row 2: 0.55 (mild ambiguity)
            return np.array([[0.50], [0.99], [0.55]])

        def predict(self, X):
            return np.array(["pos", "pos", "pos"])

    raw_df = pd.DataFrame({"id": [101, 102, 103], "val": [10, 20, 30]})
    X_proc = np.array([[10], [20], [30]])

    res = sample_uncertain_predictions(Mock1DProbaModel(), raw_df, X_proc, n_samples=3)
    queue = res["review_queue"]

    assert len(queue) == 3

    # Row 0 (p=0.50) must have uncertainty_score ~ 1.0 and be first in queue
    assert queue[0]["original_row_index"] == 0
    assert pytest.approx(queue[0]["uncertainty_score"], abs=1e-3) == 1.0

    # Row 2 (p=0.55) must be second in queue (score = 1.0 - 0.05*2 = 0.90)
    assert queue[1]["original_row_index"] == 2
    assert pytest.approx(queue[1]["uncertainty_score"], abs=1e-3) == 0.90

    # Row 1 (p=0.99) must be last in queue (score = 1.0 - 0.49*2 = 0.02)
    assert queue[2]["original_row_index"] == 1
    assert pytest.approx(queue[2]["uncertainty_score"], abs=1e-3) == 0.02

    # Verify model_prediction handles string classes without TypeError/ValueError
    for item in queue:
        assert item["model_prediction"] == "pos"


def test_stress_testing_empirical_correlation_and_fallback():
    """M6: Test empirical correlation calculation and zero-variance fallback."""
    # 1. Empirical correlation test: feature 1 strongly drives adverse outcome
    class SensitiveDriverModel:
        def predict_proba(self, X):
            f1 = X["f1"].values
            probs = 1.0 / (1.0 + np.exp(-f1))
            return np.column_stack([1.0 - probs, probs])

    df = pd.DataFrame({
        "f1": np.random.default_rng(42).normal(0, 1, 60),
        "f2": np.random.default_rng(42).normal(0, 1, 60),
    })

    tester = MonteCarloStressTester(n_simulations=40, random_seed=42)
    report = tester.run_stress_test(
        SensitiveDriverModel(),
        df,
        feature_names=["f1", "f2"],
        task_type="classification",
        adverse_class=1,
        loss_threshold=0.5,
    )

    for scen in report.scenarios:
        assert len(scen.top_risk_drivers) > 0
        for driver in scen.top_risk_drivers:
            assert "feature" in driver
            assert "sensitivity" in driver
            assert driver["sensitivity"] >= 0.0

    # 2. Zero-variance fallback test
    class ConstantOutcomeModel:
        def predict_proba(self, X):
            n = len(X)
            # Always outputs exactly 0.8 for adverse class => rate is constant 1.0 across all trials
            return np.column_stack([np.zeros(n), np.ones(n)])

    report_const = tester.run_stress_test(
        ConstantOutcomeModel(),
        df,
        feature_names=["f1", "f2"],
        task_type="classification",
        adverse_class=1,
        loss_threshold=0.5,
    )

    for scen in report_const.scenarios:
        for driver in scen.top_risk_drivers:
            # When arr_rates has 0 variance, sensitivity falls back to abs(rate_delta)
            assert pytest.approx(driver["sensitivity"], abs=1e-3) == abs(scen.rate_delta)
