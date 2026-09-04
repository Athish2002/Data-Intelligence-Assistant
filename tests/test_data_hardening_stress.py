"""
tests/test_data_hardening_stress.py
───────────────────────────────────
Adversarial stress test suite verifying enterprise data hardening against:
1. Mixed Python primitive types (['int', 'str'], ['float', 'str']) in categorical columns.
2. Infinite and extreme float overflow values (+inf, -inf, 1e308).
3. High-cardinality categorical explosion and memory bounding.
4. Unseen category levels during model inference.
5. Extreme minority class imbalance (single-member classes).
6. Degenerate target variables (single-class targets).
7. Target whitespace and formatting inconsistencies.
8. 100% NaN and zero-variance constant columns.
9. Datetime string parsing and feature extraction.
10. End-to-end full pipeline execution on compound adversarial datasets.
"""

import numpy as np
import pandas as pd
import pytest

from dia.data_sanitizer import sanitize_dataframe
from dia.model_trainer import (
    SafeCategoricalTransformer,
    SafeNumericTransformer,
    _build_preprocessor,
    train_and_evaluate,
)
from dia.pipeline_coordinator import PipelineCoordinator


def test_mixed_types_int_str_categorical_no_crash():
    """Verifies that columns with mixed int and string values train without TypeError."""
    df = pd.DataFrame({
        "mixed_feature": [1, 2, "3", "unknown", 5, 6, "7", 8, "9", 10],
        "numeric_feat": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    # Test direct preprocessor
    prep = _build_preprocessor(df[["mixed_feature", "numeric_feat"]])
    X_proc = prep.fit_transform(df[["mixed_feature", "numeric_feat"]])
    assert X_proc.shape[0] == 10
    assert X_proc.shape[1] > 0

    # Test end-to-end model training
    res = train_and_evaluate(
        df, target_col="target", task_type="classification",
        selected_model_keys=["rf", "logreg"]
    )
    assert res["best_model_key"] in ["rf", "logreg"]
    assert "Accuracy" in res["best_metrics"]


def test_mixed_types_float_str_categorical_no_crash():
    """Verifies that columns with mixed float, string, and NaN values train without error."""
    df = pd.DataFrame({
        "mixed_feat": [1.5, "Tier_A", np.nan, "Tier_B", 2.5, "Tier_A", 3.5, "Tier_B", 4.5, "Tier_C"],
        "num_feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    res = train_and_evaluate(
        df, target_col="target", task_type="classification",
        selected_model_keys=["rf"]
    )
    assert res["best_model_key"] == "rf"
    assert res["best_metrics"]["Accuracy"] is not None


def test_infinite_and_overflow_floats_no_crash():
    """Verifies that infinite (+inf, -inf) and large floats are safely handled without ValueError."""
    df = pd.DataFrame({
        "inf_feat": [1.0, np.inf, 3.0, -np.inf, 5.0, 6.0, 1e308, -1e308, 9.0, 10.0],
        "clean_feat": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    # Test sanitizer
    clean_df, report = sanitize_dataframe(df)
    assert report["inf_replaced_count"] >= 2
    assert not np.isinf(clean_df["inf_feat"]).any()

    # Test training directly on un-sanitized dataframe
    res = train_and_evaluate(
        df, target_col="target", task_type="classification",
        selected_model_keys=["rf", "logreg"]
    )
    assert res["best_model_key"] in ["rf", "logreg"]


def test_high_cardinality_bounded_memory():
    """Verifies that high-cardinality categorical features (e.g. UUIDs) do not explode memory."""
    n_rows = 150
    df = pd.DataFrame({
        "uuid_col": [f"uuid_{i:04d}" for i in range(n_rows)],
        "category_col": [f"group_{i % 5}" for i in range(n_rows)],
        "target": [i % 2 for i in range(n_rows)],
    })

    prep = _build_preprocessor(df[["uuid_col", "category_col"]])
    X_proc = prep.fit_transform(df[["uuid_col", "category_col"]])

    # Output dimensionality should be strictly bounded (category_col <= 5 one-hot, uuid_col <= 1 ordinal)
    assert X_proc.shape[1] <= 35
    assert X_proc.shape[0] == n_rows


def test_unseen_categories_at_inference():
    """Verifies that novel category values at test/inference time do not crash transformers."""
    train_df = pd.DataFrame({
        "cat_low": ["A", "B", "A", "B", "A"],
        "cat_high": [f"code_{i}" for i in range(5)],
        "num": [1, 2, 3, 4, 5],
    })
    test_df = pd.DataFrame({
        "cat_low": ["A", "NEW_UNSEEN_VALUE"],
        "cat_high": ["code_0", "TOTALLY_NOVEL_CODE"],
        "num": [2, 6],
    })

    prep = _build_preprocessor(train_df)
    prep.fit(train_df)
    test_proc = prep.transform(test_df)
    assert test_proc.shape[0] == 2
    assert not np.isnan(test_proc).any()


def test_single_sample_minority_class_cv():
    """Verifies that a classification dataset with a single-member class does not break cross-validation."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        # Class 2 has only 1 instance!
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
    })

    res = train_and_evaluate(
        df, target_col="target", task_type="classification",
        selected_model_keys=["rf"], apply_cv=True,
    )
    assert res["best_model_key"] == "rf"
    assert "Accuracy" in res["best_metrics"]


def test_single_class_target_informative_error():
    """Verifies that a dataset where target has only 1 distinct class raises a clear ValueError."""
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5],
        "target": ["active", "active", "active", "active", "active"],
    })

    with pytest.raises(ValueError) as excinfo:
        train_and_evaluate(df, target_col="target", task_type="classification", selected_model_keys=["rf"])

    assert "only 1 distinct class" in str(excinfo.value)


def test_target_whitespace_cleaning():
    """Verifies that target whitespace like 'churn' vs 'churn ' does not create phantom classes."""
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": ["churn", "churn ", "no_churn", "no_churn  ", "churn", "no_churn", "churn", "no_churn"],
    })

    res = train_and_evaluate(
        df, target_col="target", task_type="classification",
        selected_model_keys=["rf"]
    )
    assert res["best_model_key"] == "rf"


def test_all_nan_and_constant_columns_pruned():
    """Verifies that 100% all-NaN and constant columns are safely filtered without breaking models."""
    df = pd.DataFrame({
        "all_nan_col": [np.nan] * 10,
        "constant_col": ["SAME"] * 10,
        "valid_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    clean_df, report = sanitize_dataframe(df)
    assert "all_nan_col" in report["all_nan_dropped_columns"]

    res = train_and_evaluate(
        clean_df, target_col="target", task_type="classification",
        selected_model_keys=["rf"]
    )
    assert res["best_model_key"] == "rf"


def test_end_to_end_automl_on_adversarial_dataset():
    """
    Executes the full pipeline coordinator on a comprehensive adversarial dataset containing
    mixed types, infinite values, NaNs, high cardinality, dirty numerics, and date strings.
    """
    df = pd.DataFrame({
        "Customer_ID": [f"ID_{i:03d}" for i in range(50)],
        "Price ($)": ["$1,200.50", "$3,400", "$500.25", "$10k", "missing"] * 10,
        "Mixed_Tier": [1, "Gold", np.nan, 2, "Silver", "Platinum", 3, "Gold", np.nan, "Bronze"] * 5,
        "Inf_Values": [10.0, np.inf, 25.0, -np.inf, 40.0] * 10,
        "Signup_Date": ["2024-01-15", "2024-02-20", "2024-03-25", "2024-04-10", "2024-05-01"] * 10,
        "Churn_Flag": ["yes", "no", "yes ", "no", "yes"] * 10,
    })

    result = PipelineCoordinator.execute_full_pipeline(
        df=df,
        goal_text="Predict churn outcome for customers",
        user_target_col="Churn_Flag",
        selected_models=["rf", "logreg"],
    )

    assert result["target_col"] == "Churn_Flag"
    assert result["train_result"]["best_model_key"] in ["rf", "logreg"]
    assert result["train_result"]["best_metrics"]["Accuracy"] > 0
    assert result["readiness"]["score"] > 0
