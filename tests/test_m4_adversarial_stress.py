"""
tests/test_m4_adversarial_stress.py

Milestone M4 Test Suite: Adversarial Edge Cases, Extreme Stress, and Resilience Testing.
Verifies offline robustness against:
1. Unicode, non-ASCII, and emoji column names.
2. String payloads containing null bytes, script tags, and SQL injection strings.
3. Extreme floating point numbers (+1e300, -1e300, subnormals, NaNs, Infs).
4. Highly unbalanced binary classification targets (95:5).
5. Boolean and text target formats.
6. Noise-only high-dimensional feature matrices with PipelineCoordinator.
"""
import numpy as np
import pandas as pd
import pytest

from dia.data_profiler import profile_dataframe
from dia.data_sanitizer import sanitize_dataframe
from dia.model_trainer import train_and_evaluate
from dia.pipeline_coordinator import PipelineCoordinator


def test_unicode_and_emoji_column_names():
    """Verify data sanitizer, profiler, and trainer handle unicode, diacritics, and emoji columns."""
    df = pd.DataFrame({
        "用户_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "prénom": ["Jean", "Pierre", "Marie", "Chloé", "Luc", "Sophie", "Antoine", "Julie"],
        "âge_années": [25, 30, 35, 40, 45, 50, 55, 60],
        "emoji_💰_score": [100.5, 200.2, 150.8, 300.0, 250.4, 400.1, 350.0, 500.2],
        "cible_target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    # 1. Sanitize
    sanitized_df, report = sanitize_dataframe(df)
    assert len(sanitized_df) == 8
    assert "cible_target" in sanitized_df.columns

    # 2. Profile
    profile = profile_dataframe(sanitized_df)
    assert len(profile) == 5
    assert "column" in profile.columns
    assert "unique_count" in profile.columns

    # 3. Train
    train_res = train_and_evaluate(
        sanitized_df,
        target_col="cible_target",
        task_type="classification",
        selected_model_keys=["rf", "logreg"],
    )
    assert train_res["best_model_key"] in ["rf", "logreg"]
    assert "Accuracy" in train_res["best_metrics"]


def test_malicious_string_payloads_in_features():
    """Verify input containing XSS tags, null bytes, and SQL injection payloads are handled safely."""
    df = pd.DataFrame({
        "input_text": [
            "<script>alert('xss')</script>",
            "SELECT * FROM users WHERE '1'='1';",
            "DROP TABLE sessions;--",
            "normal text\x00with null byte",
            "Robert'); DROP TABLE Students;--",
            "<img src=x onerror=alert(1)>",
            "admin' --",
            "clean string value",
        ],
        "num_val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    sanitized_df, report = sanitize_dataframe(df)
    assert len(sanitized_df) == 8
    # Null bytes should be stripped
    assert "\x00" not in str(sanitized_df["input_text"].iloc[3])

    # Model training should proceed without executing or failing on strings
    train_res = train_and_evaluate(
        sanitized_df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["rf"],
    )
    assert "best_model_key" in train_res
    assert "best_metrics" in train_res


def test_extreme_floating_point_bounds_and_subnormals():
    """Verify model trainer and sanitizers survive extreme float values, infs, and subnormals."""
    df = pd.DataFrame({
        "f_large": [1e300, -1e300, 1e250, -1e250, 0.0, 1.0, -1.0, 2.0],
        "f_subnormal": [1e-300, -1e-300, 1e-250, -1e-250, 0.0, 1e-10, -1e-10, 2e-10],
        "f_infs": [np.inf, -np.inf, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        "cat_clean": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    sanitized_df, report = sanitize_dataframe(df)
    # Ensure infs were capped/handled
    assert not np.isinf(sanitized_df["f_large"]).any()
    assert not np.isinf(sanitized_df["f_infs"]).any()

    train_res = train_and_evaluate(
        sanitized_df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["rf", "logreg"],
    )
    assert "best_model_key" in train_res
    assert "best_metrics" in train_res


def test_severe_class_imbalance():
    """Verify model trainer handles severely unbalanced binary targets (e.g., 95:5) without crashing."""
    np.random.seed(42)
    n = 100
    y = np.zeros(n, dtype=int)
    y[:5] = 1  # only 5 positives out of 100

    df = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.uniform(10, 100, n),
        "target": y,
    })

    train_res = train_and_evaluate(
        df,
        target_col="target",
        task_type="classification",
        selected_model_keys=["rf", "logreg"],
    )
    assert "best_model_key" in train_res
    assert "Accuracy" in train_res["best_metrics"]


def test_boolean_and_string_targets():
    """Verify target columns formatted as boolean or strings are cleanly transformed and modeled."""
    # 1. Boolean target
    df_bool = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "feat_b": ["x", "y", "x", "y", "x", "y", "x", "y"],
        "target_bool": [True, False, True, False, True, False, True, False],
    })
    res_bool = train_and_evaluate(
        df_bool,
        target_col="target_bool",
        task_type="classification",
        selected_model_keys=["rf"],
    )
    assert "best_model_key" in res_bool

    # 2. String target
    df_str = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "target_str": ["Churn", "Retain", "Churn", "Retain", "Churn", "Retain", "Churn", "Retain"],
    })
    res_str = train_and_evaluate(
        df_str,
        target_col="target_str",
        task_type="classification",
        selected_model_keys=["rf"],
    )
    assert "best_model_key" in res_str


def test_noise_only_high_dimensional_features():
    """Verify AutoML and feature engineering remain stable on 50 noise features."""
    np.random.seed(123)
    n_samples = 40
    n_features = 50
    noise_matrix = np.random.randn(n_samples, n_features)
    col_names = [f"noise_{i}" for i in range(n_features)]

    df = pd.DataFrame(noise_matrix, columns=col_names)
    df["target"] = np.random.choice([0, 1], size=n_samples)

    result = PipelineCoordinator.execute_full_pipeline(
        df=df,
        goal_text="Predict binary outcome on high-dimensional noise",
        user_target_col="target",
        selected_models=["rf", "logreg"],
    )

    assert result["pipeline_executed"] is True
    assert "best_model_key" in result["train_result"]
    assert len(result["train_result"]["results"]) > 0
