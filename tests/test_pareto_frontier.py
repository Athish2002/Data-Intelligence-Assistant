"""
tests/test_pareto_frontier.py
──────────────────────────────
Tests for:
1. Exact 3D Lebesgue Hypervolume Calculation in `dia/pareto_frontier.py`
2. Directional Monotonicity Constraints in `dia/recourse_engine.py`
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dia.pareto_frontier import (
    ParetoFrontierResult,
    ParetoPoint,
    _calculate_hypervolume_3d,
    compute_pareto_frontier,
)
from dia.recourse_engine import (
    AlgorithmicRecourseEngine,
    RecourseResult,
    _parse_direction_constraint,
    compute_recourse,
)

# ---------------------------------------------------------------------------
# Task 1: Exact 3D Lebesgue Hypervolume Calculation Tests
# ---------------------------------------------------------------------------

def test_hypervolume_3d_analytical_geometry():
    """
    Verifies exact dimension slicing against an analytically calculated 3D volume.
    Two boxes with known inclusion-exclusion volume:
      B1: [0, 10] x [0, 5] x [0, 2] -> Vol = 100
      B2: [0, 5] x [0, 10] x [0, 4] -> Vol = 200
      B1 n B2: [0, 5] x [0, 5] x [0, 2] -> Vol = 50
      Union Vol = 100 + 200 - 50 = 250.0
    Nadir: (0.0, 10.0, 0.0)
    Points:
      P1: profit=10, default_risk=5 (v = 10 - 5 = 5), fairness=2
      P2: profit=5, default_risk=0 (v = 10 - 0 = 10), fairness=4
    """
    nadir = (0.0, 10.0, 0.0)
    p1 = ParetoPoint(model_name="P1", roi_profit=10.0, default_risk=5.0, fairness_ratio=2.0)
    p2 = ParetoPoint(model_name="P2", roi_profit=5.0, default_risk=0.0, fairness_ratio=4.0)

    hv = _calculate_hypervolume_3d([p1, p2], nadir)
    assert hv == 250.0, f"Expected exact volume 250.0, got {hv}"


def test_hypervolume_3d_duplicate_invariance():
    """
    CRITICAL INVARIANCE: Adding identical duplicate points must NOT inflate hypervolume!
    Verifies that n duplicates produce the exact same Lebesgue measure.
    """
    nadir = (1000.0, 0.30, 0.40)
    points = [
        ParetoPoint(model_name="M1", roi_profit=50000.0, default_risk=0.20, fairness_ratio=0.75),
        ParetoPoint(model_name="M2", roi_profit=35000.0, default_risk=0.10, fairness_ratio=0.85),
        ParetoPoint(model_name="M3", roi_profit=20000.0, default_risk=0.05, fairness_ratio=0.95),
    ]

    base_hv = _calculate_hypervolume_3d(points, nadir)
    assert base_hv > 0.0

    # Add 1 duplicate of M1
    points_dup1 = points + [points[0]]
    hv_dup1 = _calculate_hypervolume_3d(points_dup1, nadir)
    assert hv_dup1 == pytest.approx(base_hv, abs=1e-4), (
        f"Hypervolume inflated with 1 duplicate: {hv_dup1} vs {base_hv}"
    )

    # Add 10 duplicates of each point
    points_dup10 = points * 10
    hv_dup10 = _calculate_hypervolume_3d(points_dup10, nadir)
    assert hv_dup10 == pytest.approx(base_hv, abs=1e-4), (
        f"Hypervolume inflated with 10 duplicates: {hv_dup10} vs {base_hv}"
    )


def test_hypervolume_3d_dominated_point_invariance():
    """
    Adding points that are strictly dominated within the positive orthant
    must NOT inflate the hypervolume.
    """
    nadir = (0.0, 1.0, 0.0)
    p_opt1 = ParetoPoint(model_name="Opt1", roi_profit=10.0, default_risk=0.2, fairness_ratio=0.8)
    p_opt2 = ParetoPoint(model_name="Opt2", roi_profit=5.0, default_risk=0.1, fairness_ratio=0.9)

    base_hv = _calculate_hypervolume_3d([p_opt1, p_opt2], nadir)

    # p_dom is strictly dominated by p_opt1 in all dimensions (profit=8<10, risk=0.3>0.2, fairness=0.7<0.8)
    p_dom = ParetoPoint(model_name="Dom", roi_profit=8.0, default_risk=0.3, fairness_ratio=0.7)

    hv_with_dom = _calculate_hypervolume_3d([p_opt1, p_opt2, p_dom], nadir)
    assert hv_with_dom == pytest.approx(base_hv, abs=1e-4), (
        f"Hypervolume changed when adding dominated point: {hv_with_dom} vs {base_hv}"
    )


def test_hypervolume_3d_edge_cases():
    """Edge cases: empty list, zero volume, negative relative coordinates."""
    nadir = (100.0, 0.1, 0.5)
    # Empty points
    assert _calculate_hypervolume_3d([], nadir) == 0.0

    # Points strictly at or below nadir (u <= 0, v <= 0, or w <= 0)
    below_nadir = [
        ParetoPoint(model_name="Below", roi_profit=50.0, default_risk=0.2, fairness_ratio=0.3)
    ]
    assert _calculate_hypervolume_3d(below_nadir, nadir) == 0.0


def test_compute_pareto_frontier_full_pipeline():
    """Integration test for compute_pareto_frontier with exact hypervolume."""
    models_data = [
        {"model_name": "HighProfit", "roi_profit": 60000.0, "default_risk": 0.20, "fairness_ratio": 0.70},
        {"model_name": "Balanced", "roi_profit": 45000.0, "default_risk": 0.08, "fairness_ratio": 0.88},
        {"model_name": "MinRisk", "roi_profit": 25000.0, "default_risk": 0.03, "fairness_ratio": 0.92},
        {"model_name": "Dominated", "roi_profit": 20000.0, "default_risk": 0.25, "fairness_ratio": 0.60},
    ]

    res = compute_pareto_frontier(models_data)
    assert isinstance(res, ParetoFrontierResult)
    assert len(res.frontier_points) == 3
    assert res.hypervolume > 0.0

    # Adding a duplicate model configuration in input data must produce the same hypervolume
    models_data_with_dup = models_data + [models_data[1]]
    res_dup = compute_pareto_frontier(models_data_with_dup)
    assert res_dup.hypervolume == pytest.approx(res.hypervolume, abs=1e-4)


# ---------------------------------------------------------------------------
# Task 2: Directional Monotonicity Constraints Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_loan_model():
    """
    Creates a simple reference dataset and trained logistic regression model:
      features: ['tenure', 'income', 'debt_ratio']
      label: approval (1) vs denial (0)
      approval probability increases with tenure and income, decreases with debt_ratio.
    """
    np.random.seed(42)
    n = 200
    tenure = np.random.uniform(1.0, 10.0, n)
    income = np.random.uniform(20.0, 120.0, n)
    debt_ratio = np.random.uniform(0.1, 0.9, n)

    # Logit: 0.4*tenure + 0.05*income - 5.0*debt_ratio - 1.0
    logit = 0.4 * tenure + 0.05 * income - 5.0 * debt_ratio - 1.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (prob >= 0.5).astype(int)

    df = pd.DataFrame({"tenure": tenure, "income": income, "debt_ratio": debt_ratio, "y": y})
    feature_names = ["tenure", "income", "debt_ratio"]

    model = LogisticRegression()
    model.fit(df[feature_names], y)

    return model, df, feature_names


def test_recourse_monotonicity_increasing_tenure(trained_loan_model):
    """
    Verifies that when direction_constraints={'tenure': 'increasing'},
    tenure is NEVER decreased in candidate moves or the final recourse actions.
    """
    model, df, feature_names = trained_loan_model

    # Instance with low probability of approval
    x_input = pd.Series({"tenure": 3.0, "income": 35.0, "debt_ratio": 0.70})

    # 1. Compute with increasing constraint on tenure
    res = compute_recourse(
        model=model,
        x_input=x_input,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
        target_prob_threshold=0.60,
        direction_constraints={"tenure": "increasing"},
    )

    assert isinstance(res, RecourseResult)
    # Check counterfactual vector
    proposed_tenure = res.counterfactual_vector.get("tenure", 3.0)
    assert proposed_tenure >= 3.0, f"Tenure decreased from 3.0 to {proposed_tenure} despite increasing constraint!"

    # Check actions list
    for action in res.actions:
        if action.feature == "tenure":
            assert action.direction == "INCREASE", f"Action on tenure was {action.direction}, expected INCREASE"
            assert action.delta >= 0.0, f"Action delta on tenure was negative: {action.delta}"
            assert action.proposed_value >= action.original_value


def test_recourse_monotonicity_decreasing_debt_ratio(trained_loan_model):
    """
    Verifies that when direction_constraints={'debt_ratio': 'decreasing'},
    debt_ratio is NEVER increased.
    """
    model, df, feature_names = trained_loan_model
    x_input = pd.Series({"tenure": 2.0, "income": 40.0, "debt_ratio": 0.80})

    res = compute_recourse(
        model=model,
        x_input=x_input,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
        target_prob_threshold=0.60,
        direction_constraints={"debt_ratio": "decreasing"},
    )

    assert isinstance(res, RecourseResult)
    proposed_debt = res.counterfactual_vector.get("debt_ratio", 0.80)
    assert proposed_debt <= 0.80 + 1e-6, f"Debt ratio increased from 0.80 to {proposed_debt}!"

    for action in res.actions:
        if action.feature == "debt_ratio":
            assert action.direction == "DECREASE"
            assert action.delta <= 0.0
            assert action.proposed_value <= action.original_value


def test_recourse_direction_constraint_string_variants(trained_loan_model):
    """
    Tests various constraint representations:
      '>=', 'increasing', '+' -> all enforce non-decreasing
      '<=', 'decreasing', '-' -> all enforce non-increasing
    """
    assert _parse_direction_constraint(">=") == "increasing"
    assert _parse_direction_constraint("increasing") == "increasing"
    assert _parse_direction_constraint("+") == "increasing"
    assert _parse_direction_constraint("INCREASE") == "increasing"

    assert _parse_direction_constraint("<=") == "decreasing"
    assert _parse_direction_constraint("decreasing") == "decreasing"
    assert _parse_direction_constraint("-") == "decreasing"
    assert _parse_direction_constraint("DECREASE") == "decreasing"

    model, df, feature_names = trained_loan_model
    x_input = pd.Series({"tenure": 2.5, "income": 30.0, "debt_ratio": 0.75})

    # Test with '+' and '-'
    res = compute_recourse(
        model=model,
        x_input=x_input,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
        target_prob_threshold=0.60,
        direction_constraints={"tenure": "+", "debt_ratio": "-"},
    )

    assert res.counterfactual_vector["tenure"] >= 2.5 - 1e-6
    assert res.counterfactual_vector["debt_ratio"] <= 0.75 + 1e-6


def test_recourse_engine_class_level_constraints(trained_loan_model):
    """
    Verifies that direction_constraints passed to AlgorithmicRecourseEngine.__init__
    are preserved and enforced during compute_recourse.
    """
    model, df, feature_names = trained_loan_model
    x_input = pd.Series({"tenure": 4.0, "income": 30.0, "debt_ratio": 0.80})

    engine = AlgorithmicRecourseEngine(
        direction_constraints={"tenure": "increasing", "debt_ratio": "<="}
    )
    res = engine.compute_recourse(
        model=model,
        x_input=x_input,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
        target_prob_threshold=0.55,
    )

    assert res.counterfactual_vector["tenure"] >= 4.0 - 1e-6
    assert res.counterfactual_vector["debt_ratio"] <= 0.80 + 1e-6


def test_recourse_already_compliant_instance(trained_loan_model):
    """An instance that already satisfies the target probability requires 0 actions."""
    model, df, feature_names = trained_loan_model
    # Instance with very favorable metrics
    x_input = pd.Series({"tenure": 9.5, "income": 115.0, "debt_ratio": 0.15})

    res = compute_recourse(
        model=model,
        x_input=x_input,
        reference_df=df,
        feature_names=feature_names,
        desired_class=1,
        target_prob_threshold=0.55,
        direction_constraints={"tenure": "increasing"},
    )

    assert res.feasibility == "OPTIMAL_RECOURSE_FOUND"
    assert res.total_recourse_cost == 0.0
    assert len(res.actions) == 0
    assert res.counterfactual_vector["tenure"] == 9.5


def test_recourse_categorical_monotonicity():
    """
    Verifies that for categorical features with ordered categories,
    directional monotonicity constraints prevent decreasing category codes.
    """
    # Create categorical dataset with 3 tiers
    categories = ["Tier1_Bronze", "Tier2_Silver", "Tier3_Gold"]
    np.random.seed(42)
    n = 150
    tier_idx = np.random.choice([0, 1, 2], n)
    tier_col = [categories[i] for i in tier_idx]
    score = np.random.uniform(10.0, 50.0, n)
    # Target: higher tier and higher score -> class 1
    y = ((tier_idx * 15.0 + score) >= 45.0).astype(int)

    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    df = pd.DataFrame({
        "tier": pd.Series(tier_col, dtype=cat_type),
        "score": score,
        "y": y,
    })

    # Simple mock model for demonstration
    class CategoricalMockModel:
        def predict_proba(self, X):
            probs = []
            for _, row in X.iterrows():
                t = str(row["tier"])
                s = float(row["score"])
                code = categories.index(t) if t in categories else 0
                val = code * 0.35 + s * 0.015
                p1 = min(0.99, max(0.01, val))
                probs.append([1.0 - p1, p1])
            return np.array(probs)

    mock_model = CategoricalMockModel()
    x_input = pd.Series({"tier": "Tier2_Silver", "score": 20.0})

    res = compute_recourse(
        model=mock_model,
        x_input=x_input,
        reference_df=df,
        feature_names=["tier", "score"],
        desired_class=1,
        target_prob_threshold=0.60,
        direction_constraints={"tier": "increasing"},
    )

    # If tier changed, it must be Tier3_Gold, never Tier1_Bronze
    proposed_tier = res.counterfactual_vector.get("tier", "Tier2_Silver")
    assert proposed_tier in ["Tier2_Silver", "Tier3_Gold"], (
        f"Tier decreased from Tier2_Silver to {proposed_tier} despite increasing constraint!"
    )
    for action in res.actions:
        if action.feature == "tier":
            assert action.proposed_value == "Tier3_Gold"
            assert action.direction in ["INCREASE", "SWITCH"]

