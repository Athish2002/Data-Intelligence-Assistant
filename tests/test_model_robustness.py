"""
tests/test_model_robustness.py
==============================
Tests for edge-case target handling, auto-fallback from string regression to classification,
and prevention of n_samples=0 train_test_split errors.
"""

import numpy as np
import pandas as pd
import pytest
from dia.column_resolver import resolve_column
from dia.goal_parser import parse_goal
from dia.model_trainer import train_and_evaluate
from dia.pipeline_coordinator import PipelineCoordinator


class TestModelRobustness:
    def test_goal_parser_conversion_probability(self):
        res = parse_goal("Predict customer purchase conversion probability")
        assert res["task_type"] == "classification"

    def test_column_resolver_conversion_synonyms(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "age": [25, 40, 32],
            "converted": [0, 1, 1],
        })
        col, conf, match_type = resolve_column("conversion", df)
        assert col == "converted"
        assert conf > 0.7

    def test_regression_on_string_auto_fallback(self):
        df = pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "target": ["Yes", "No"] * 25,
        })
        result = train_and_evaluate(
            df=df,
            target_col="target",
            task_type="regression",
            selected_model_keys=["rf", "logreg"],
            test_size=0.2,
        )
        assert result is not None
        assert "best_model" in result
        assert len(result["results"]) > 0

    def test_pipeline_coordinator_robust_execution_ecom_goal(self):
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i}" for i in range(100)],
            "cart_amount": np.random.uniform(10, 500, 100),
            "items_in_cart": np.random.randint(1, 10, 100),
            "converted": np.random.choice(["Yes", "No"], 100),
        })
        res = PipelineCoordinator.execute_full_pipeline(
            df=df,
            goal_text="Predict customer purchase conversion probability",
            user_target_col=None,
        )
        assert res is not None
        assert res["target_col"] == "converted"
        assert res["final_task_type"] == "classification"
        assert "train_result" in res
        assert res["train_result"]["best_model_label"] is not None
