"""
tests/test_goal_parser.py
─────────────────────────
Unit tests for dia/goal_parser.py
"""

from __future__ import annotations


from dia.goal_parser import parse_goal


class TestParseGoal:
    # ── Classification detection ─────────────────────────────────────────────
    def test_churn_is_classification(self) -> None:
        result = parse_goal("predict customer churn")
        assert result["task_type"] == "classification"

    def test_fraud_is_classification(self) -> None:
        result = parse_goal("detect fraudulent transactions")
        assert result["task_type"] == "classification"

    def test_spam_is_classification(self) -> None:
        result = parse_goal("classify spam emails")
        assert result["task_type"] == "classification"

    # ── Regression detection ─────────────────────────────────────────────────
    def test_price_is_regression(self) -> None:
        result = parse_goal("predict house price")
        assert result["task_type"] == "regression"

    def test_salary_is_regression(self) -> None:
        result = parse_goal("estimate employee salary")
        assert result["task_type"] == "regression"

    def test_revenue_is_regression(self) -> None:
        result = parse_goal("forecast next month revenue")
        assert result["task_type"] == "regression"

    # ── Confidence ───────────────────────────────────────────────────────────
    def test_confidence_between_0_and_1(self) -> None:
        result = parse_goal("predict churn")
        assert 0.0 <= result["confidence"] <= 1.0

    # ── Target candidates ────────────────────────────────────────────────────
    def test_target_candidates_from_columns(self) -> None:
        cols = ["customer_id", "tenure", "churn_flag"]
        result = parse_goal("predict churn", columns=cols)
        assert "churn_flag" in result["target_candidates"]

    def test_no_columns_returns_empty_candidates(self) -> None:
        result = parse_goal("predict price")
        assert result["target_candidates"] == []

    # ── Edge cases ───────────────────────────────────────────────────────────
    def test_empty_goal_returns_unknown(self) -> None:
        result = parse_goal("")
        assert result["task_type"] == "unknown"
        assert result["confidence"] == 0.0

    def test_whitespace_only_goal_returns_unknown(self) -> None:
        result = parse_goal("   ")
        assert result["task_type"] == "unknown"

    def test_gibberish_goal_returns_unknown(self) -> None:
        result = parse_goal("xyzzy florp zab")
        assert result["task_type"] in ("unknown", "classification", "regression")

    def test_no_semantic_flag(self) -> None:
        """use_semantic=False should still return keyword-based result."""
        result = parse_goal("predict churn", use_semantic=False)
        assert result["task_type"] == "classification"
        assert result["method"] == "keyword"

    def test_explanation_is_string(self) -> None:
        result = parse_goal("predict house price")
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0
