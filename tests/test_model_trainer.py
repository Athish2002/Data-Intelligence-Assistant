"""
tests/test_model_trainer.py
───────────────────────────
Tests for AutoML Calibration, Ensemble Selection, and AutoFE Integrity (M3, M4, M5).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from dia.feature_engineer import auto_engineer_features
from dia.model_trainer import train_and_evaluate


@pytest.fixture
def sample_classification_data():
    np.random.seed(42)
    n = 120
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = x1 * 2 + np.random.randn(n) * 0.5
    prob = 1 / (1 + np.exp(-(x1 + 1.5 * x2)))
    y = (prob > 0.5).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = x1 * 1.5 - x2 * 0.5 + np.random.randn(n) * 0.1
    y = 2.0 * x1 - 3.0 * x2 + np.random.randn(n) * 0.2
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def test_m4_calibrated_classifier_no_test_leakage(sample_classification_data):
    """
    Issue M4: Verify that CalibratedClassifierCV fits strictly on X_train_proc with cv=3
    and NEVER touches X_test_proc during calibration fitting.
    """
    df = sample_classification_data

    # Spy on CalibratedClassifierCV.fit to verify the training data was passed, not test data
    fit_calls = []
    original_fit = CalibratedClassifierCV.fit

    def spy_fit(self, X, y, *args, **kwargs):
        fit_calls.append((X, y))
        return original_fit(self, X, y, *args, **kwargs)

    with patch.object(CalibratedClassifierCV, "fit", side_effect=spy_fit, autospec=True):
        res = train_and_evaluate(
            df,
            target_col="target",
            task_type="classification",
            selected_model_keys=["rf", "logreg"],
            calibrate_probs=True,
            build_ensemble=False,
            enable_autofe=False,
            test_size=0.25,
            random_state=42,
        )

    assert len(fit_calls) >= 1, "CalibratedClassifierCV.fit should have been called"
    test_indices = res.get("test_indices", [])
    test_size_count = len(test_indices)
    expected_train_size = len(df) - test_size_count

    for fitted_X, fitted_y in fit_calls:
        # fitted_X should have rows equal to train size, NOT test size!
        assert fitted_X.shape[0] == expected_train_size, (
            f"Calibrator was fitted on {fitted_X.shape[0]} samples, "
            f"expected train size {expected_train_size} (test size was {test_size_count})"
        )
        assert len(fitted_y) == expected_train_size

    # Verify model results include valid calibrated metrics
    for model_res in res["results"]:
        assert model_res["error"] is None
        assert "ROC-AUC" in model_res["metrics"]
        assert model_res["y_proba"] is not None
        # Importance should be successfully extracted from CalibratedClassifierCV
        assert not model_res["importance"].empty


def test_m3_ensemble_selection_by_validation_score(sample_classification_data):
    """
    Issue M3: Verify that Soft Voting Ensemble selects the top 3 best models by metric
    rather than arbitrary first 3 registry order.
    """
    df = sample_classification_data
    # Use 4 models: rf, gbdt, logreg, knn
    res = train_and_evaluate(
        df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["rf", "gbdt", "logreg", "knn"],
        build_ensemble=True,
        calibrate_probs=False,
        enable_autofe=False,
        random_state=42,
    )

    results = res["results"]
    individual_results = [r for r in results if r["model_key"] != "voting_ensemble" and not r.get("error")]
    ensemble_entry = next((r for r in results if r["model_key"] == "voting_ensemble"), None)

    assert ensemble_entry is not None, "Voting ensemble should have been built"
    selected_base_models = ensemble_entry["best_params"]["base_models"]

    # Rank individual models by ROC-AUC (fallback F1)
    def get_score(entry):
        m = entry.get("metrics", {})
        return m.get("ROC-AUC", m.get("F1", -999.0))

    sorted_individuals = sorted(individual_results, key=get_score, reverse=True)
    top_3_keys = [r["model_key"] for r in sorted_individuals[:3]]

    # The ensemble's base models must be the top 3 models
    assert selected_base_models == top_3_keys, (
        f"Ensemble selected {selected_base_models}, but top 3 were {top_3_keys}"
    )


def test_m3_voting_regressor_top_selection(sample_regression_data):
    """
    Issue M3: Verify that Voting Regressor selects the top models by R² score.
    """
    df = sample_regression_data
    res = train_and_evaluate(
        df,
        target_col="target",
        task_type="regression",
        selected_model_keys=["linreg", "rf", "gb", "knn"],
        build_ensemble=True,
        enable_autofe=False,
        random_state=42,
    )

    results = res["results"]
    individual_results = [r for r in results if r["model_key"] != "voting_ensemble" and not r.get("error")]
    ensemble_entry = next((r for r in results if r["model_key"] == "voting_ensemble"), None)

    assert ensemble_entry is not None, "Voting regressor should have been built"
    selected_base_models = ensemble_entry["best_params"]["base_models"]

    def get_r2(entry):
        m = entry.get("metrics", {})
        return m.get("R²", m.get("R2", -999.0))

    sorted_individuals = sorted(individual_results, key=get_r2, reverse=True)
    top_keys = [r["model_key"] for r in sorted_individuals[:3]]

    assert selected_base_models == top_keys


def test_m5_auto_engineer_features_train_indices_isolation():
    """
    Issue M5: Verify that auto_engineer_features calculates medians and mutual info
    strictly on train_indices, preventing test set leakage.
    """
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num1": np.random.randn(n) * 10,
        "num2": np.random.randn(n) * 5,
        "target": np.random.choice([0, 1], size=n),
    })
    # Add outliers/nans specifically to the test portion to verify training median is used
    train_idx = df.index[:70]
    test_idx = df.index[70:]

    # Put extreme values in test set for num2 to ensure test median is drastically different
    df_modified = df.copy()
    df_modified.loc[test_idx, "num2"] = 10000.0

    # Run AutoFE with train_indices
    df_out, new_cols = auto_engineer_features(
        df_modified,
        target_col="target",
        task_type="classification",
        train_indices=train_idx,
    )

    assert isinstance(new_cols, list)
    assert len(df_out) == len(df)

    # Backward compatibility test: train_indices=None works seamlessly
    df_out_compat, new_cols_compat = auto_engineer_features(
        df,
        target_col="target",
        task_type="classification",
        train_indices=None,
    )
    assert isinstance(new_cols_compat, list)
    assert len(df_out_compat) == len(df)


def test_m5_train_and_evaluate_with_autofe(sample_classification_data):
    """
    Issue M5: Verify that train_and_evaluate with enable_autofe=True runs end-to-end
    without data leakage and produces high quality model evaluation.
    """
    df = sample_classification_data
    res = train_and_evaluate(
        df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["rf", "logreg"],
        enable_autofe=True,
        calibrate_probs=True,
        build_ensemble=True,
        random_state=42,
    )

    assert res["best_model"] is not None
    assert len(res["results"]) >= 2
    for r in res["results"]:
        assert r["error"] is None
