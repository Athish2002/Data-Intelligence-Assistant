"""
tests/test_data_profiler.py
───────────────────────────
Unit tests for dia/data_profiler.py
"""

from __future__ import annotations

import pandas as pd

from dia.data_profiler import (
    detect_target_type,
    generate_readiness_report,
    infer_column_roles,
    profile_dataframe,
)
from dia.goal_parser import parse_goal


class TestProfileDataframe:
    def test_returns_one_row_per_column(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df)
        assert len(profile) == len(churn_df.columns)

    def test_null_pct_correct(self) -> None:
        df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, 4]})
        profile = profile_dataframe(df)
        row = profile[profile["column"] == "a"].iloc[0]
        assert row["null_pct"] == 75.0

    def test_unique_count_correct(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df)
        row = profile[profile["column"] == "churn_flag"].iloc[0]
        assert row["unique_count"] == 2


class TestInferColumnRoles:
    def test_identifier_detection(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        id_row = annotated[annotated["column"] == "customer_id"].iloc[0]
        assert id_row["inferred_role"] == "identifier"

    def test_duration_detection(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        tenure_row = annotated[annotated["column"] == "tenure_months"].iloc[0]
        assert tenure_row["inferred_role"] == "duration / tenure"

    def test_numeric_feature_for_float(self, churn_df: pd.DataFrame) -> None:
        """monthly_charges (float) should NOT be identified as an identifier."""
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        row = annotated[annotated["column"] == "monthly_charges"].iloc[0]
        assert row["inferred_role"] == "numeric feature"

    def test_target_candidate_detection(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        churn_row = annotated[annotated["column"] == "churn_flag"].iloc[0]
        assert churn_row["inferred_role"] in ("target candidate", "binary flag")

    def test_constant_column_detected(self) -> None:
        df = pd.DataFrame({"const": [1, 1, 1, 1], "feature": [1, 2, 3, 4]})
        profile = profile_dataframe(df)
        annotated = infer_column_roles(df, profile)
        const_row = annotated[annotated["column"] == "const"].iloc[0]
        assert const_row["inferred_role"] == "constant (useless)"

    def test_confidence_label_present(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        assert "confidence_label" in annotated.columns
        assert all(v in ("High", "Medium", "Low") for v in annotated["confidence_label"])


class TestDetectTargetType:
    def test_binary_int_is_classification(self, churn_df: pd.DataFrame) -> None:
        result = detect_target_type(churn_df, "churn_flag")
        assert result["task_type"] == "classification"

    def test_string_target_is_classification(self) -> None:
        # Object dtype with a small number of unique values → classification
        df = pd.DataFrame({"label": ["cat", "dog", "bird"] * 100, "x": range(300)})
        result = detect_target_type(df, "label")
        assert result["task_type"] == "classification"

    def test_float_target_is_regression(self, regression_df: pd.DataFrame) -> None:
        result = detect_target_type(regression_df, "price")
        assert result["task_type"] == "regression"

    def test_high_unique_int_is_regression(self) -> None:
        df = pd.DataFrame({"price": list(range(500)), "feat": [1.0] * 500})
        result = detect_target_type(df, "price")
        assert result["task_type"] == "regression"

    def test_result_contains_reason(self, churn_df: pd.DataFrame) -> None:
        result = detect_target_type(churn_df, "churn_flag")
        assert "reason" in result
        assert len(result["reason"]) > 0


class TestGenerateReadinessReport:
    def test_returns_expected_keys(self, churn_df: pd.DataFrame) -> None:
        from dia.data_profiler import profile_dataframe, infer_column_roles
        gi = parse_goal("predict churn", columns=churn_df.columns.tolist())
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        r = generate_readiness_report(churn_df, annotated, gi, "churn_flag", "classification")
        for key in ("verdict", "score", "useful_features", "leakage_risk", "missing_signals"):
            assert key in r

    def test_score_is_bounded(self, churn_df: pd.DataFrame) -> None:
        from dia.data_profiler import profile_dataframe, infer_column_roles
        gi = parse_goal("predict churn", columns=churn_df.columns.tolist())
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        r = generate_readiness_report(churn_df, annotated, gi, "churn_flag", "classification")
        assert 0 <= r["score"] <= 100

    def test_small_dataset_lowers_score(self) -> None:
        from dia.data_profiler import profile_dataframe, infer_column_roles
        df = pd.DataFrame({"feat": range(50), "target": [0, 1] * 25})
        gi = parse_goal("predict target")
        profile = profile_dataframe(df)
        annotated = infer_column_roles(df, profile)
        r = generate_readiness_report(df, annotated, gi, "target", "classification")
        assert r["score"] < 90  # penalty for small dataset

    def test_leakage_risk_includes_identifiers(self, churn_df: pd.DataFrame) -> None:
        from dia.data_profiler import profile_dataframe, infer_column_roles
        gi = parse_goal("predict churn", columns=churn_df.columns.tolist())
        profile = profile_dataframe(churn_df)
        annotated = infer_column_roles(churn_df, profile)
        r = generate_readiness_report(churn_df, annotated, gi, "churn_flag", "classification")
        # customer_id should appear in leakage risk
        assert "customer_id" in r["leakage_risk"]
