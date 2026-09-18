"""
Multi-Objective Pareto Frontier "Flight Simulator".

Implements:
1. Non-dominated Pareto sorting across 3 conflicting business dimensions:
   - (1) Net Profit / ROI (maximize)
   - (2) Risk / Default Rate (minimize)
   - (3) Algorithmic Fairness / Demographic Parity Ratio (maximize toward 1.0)
2. Knee-point detection using weighted Euclidean & Chebyshev distance to ideal utopia point.
3. Classification of extreme points: Max Profit, Min Risk, Max Fairness, and Balanced Knee.
4. Threshold flight simulator with protected group demographic parity evaluation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models / Interface Contracts
# ---------------------------------------------------------------------------

class ParetoPoint(BaseModel):
    """Represents an evaluated model configuration or threshold policy point."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model_name: str = "model"
    threshold: Optional[float] = None
    roi_profit: float
    default_risk: float
    fairness_ratio: float  # Demographic Parity / Disparate Impact ratio [0.0, 1.0]
    is_pareto_optimal: bool = False
    classification: str = "SUB_OPTIMAL"  # MAX_PROFIT, MIN_RISK, MAX_FAIRNESS, BALANCED_KNEE, SUB_OPTIMAL
    metrics: Dict[str, float] = Field(default_factory=dict)
    label: Optional[str] = None

    # Properties for aliases
    @property
    def profit(self) -> float:
        return self.roi_profit

    @property
    def risk_rate(self) -> float:
        return self.default_risk


class ParetoFrontierResult(BaseModel):
    """Result of multi-objective Pareto optimization across Profit, Risk, and Fairness."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    frontier_points: List[ParetoPoint]
    all_evaluated_points: List[ParetoPoint]
    knee_point: Optional[ParetoPoint] = None
    max_profit_point: Optional[ParetoPoint] = None
    min_risk_point: Optional[ParetoPoint] = None
    max_fairness_point: Optional[ParetoPoint] = None
    recommended_point: Optional[ParetoPoint] = None
    dimensions: List[str] = Field(
        default_factory=lambda: ["roi_profit", "default_risk", "fairness_ratio"]
    )
    hypervolume: float = 0.0
    protected_attribute: Optional[str] = None
    sensitive_groups: List[str] = Field(default_factory=list)
    flight_recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Algorithmic Routines
# ---------------------------------------------------------------------------

def _extract_metric(item: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    """Extracts a numeric metric from a dictionary checking multiple alias keys."""
    for k in keys:
        if k in item and item[k] is not None:
            try:
                val = float(item[k])
                if not (math.isnan(val) or math.isinf(val)):
                    return val
            except (ValueError, TypeError):
                continue
    # Check nested metrics dictionary if present
    if "metrics" in item and isinstance(item["metrics"], dict):
        for k in keys:
            if k in item["metrics"] and item["metrics"][k] is not None:
                try:
                    val = float(item["metrics"][k])
                    if not (math.isnan(val) or math.isinf(val)):
                        return val
                except (ValueError, TypeError):
                    continue
    return default


def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """
    Checks if point a dominates point b in 3 objectives:
    - roi_profit: maximize
    - default_risk: minimize
    - fairness_ratio: maximize
    """
    # Weak dominance: a must be at least as good as b in all dimensions
    weak = (
        a.roi_profit >= b.roi_profit and
        a.default_risk <= b.default_risk and
        a.fairness_ratio >= b.fairness_ratio
    )
    if not weak:
        return False

    # Strict dominance: a must be strictly better than b in at least one dimension
    strict = (
        a.roi_profit > b.roi_profit or
        a.default_risk < b.default_risk or
        a.fairness_ratio > b.fairness_ratio
    )
    return strict


def _calculate_hypervolume_3d(points: List[ParetoPoint], nadir: Tuple[float, float, float]) -> float:
    """
    Approximates 3D dominated hypervolume of normalized frontier points.
    nadir: (min_profit, max_risk, min_fairness)
    """
    if not points:
        return 0.0

    # Project to positive box [0, 1]^3
    # Objectives: x = profit (max), y = 1 - risk (max), z = fairness (max)
    volumes: List[float] = []
    for p in points:
        v_x = max(0.0, p.roi_profit - nadir[0])
        v_y = max(0.0, nadir[1] - p.default_risk)
        v_z = max(0.0, p.fairness_ratio - nadir[2])
        volumes.append(v_x * v_y * v_z)

    # Return union approximation (upper bound on non-dominated set hypervolume)
    if not volumes:
        return 0.0
    return float(round(np.mean(volumes) * len(volumes) / (1.0 + len(volumes) * 0.05), 4))


# ---------------------------------------------------------------------------
# Pareto Frontier Computation
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    models_data: List[Dict[str, Any]],
    profit_weight: float = 1.0,
    risk_weight: float = 1.0,
    fairness_weight: float = 1.0,
) -> ParetoFrontierResult:
    """
    Computes the non-dominated Pareto frontier and identifies optimal trade-off knee points.

    Args:
        models_data: List of model metric records containing profit, risk, and fairness.
        profit_weight: Business preference weight for ROI/profit.
        risk_weight: Business preference weight for risk minimization.
        fairness_weight: Business preference weight for algorithmic fairness.

    Returns:
        ParetoFrontierResult with classified points and flight recommendations.
    """
    dimensions = ["roi_profit", "default_risk", "fairness_ratio"]

    # Defensive handling: empty list
    if not models_data:
        return ParetoFrontierResult(
            frontier_points=[],
            all_evaluated_points=[],
            knee_point=None,
            max_profit_point=None,
            min_risk_point=None,
            max_fairness_point=None,
            recommended_point=None,
            dimensions=dimensions,
            hypervolume=0.0,
            sensitive_groups=[],
            flight_recommendations=["No model configurations evaluated."],
        )

    # 1. Parse and standardize all input records
    evaluated_points: List[ParetoPoint] = []
    for idx, item in enumerate(models_data):
        m_name = str(item.get("model_name") or item.get("name") or item.get("label") or f"Model_{idx + 1}")
        thresh = item.get("threshold")
        if thresh is not None:
            try:
                thresh = float(thresh)
            except (ValueError, TypeError):
                thresh = None

        profit = _extract_metric(item, ["roi_profit", "profit", "roi", "net_profit"], default=0.0)
        risk = _extract_metric(item, ["default_risk", "risk_rate", "risk", "default_rate"], default=0.0)
        fairness = _extract_metric(item, ["fairness_ratio", "fairness", "demographic_parity_ratio", "disparate_impact"], default=1.0)
        fairness = max(0.0, min(1.0, fairness))

        raw_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        numeric_metrics = {str(k): float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
        numeric_metrics["roi_profit"] = profit
        numeric_metrics["default_risk"] = risk
        numeric_metrics["fairness_ratio"] = fairness

        point = ParetoPoint(
            model_name=m_name,
            threshold=thresh,
            roi_profit=round(profit, 4),
            default_risk=round(risk, 4),
            fairness_ratio=round(fairness, 4),
            is_pareto_optimal=False,
            classification="SUB_OPTIMAL",
            metrics=numeric_metrics,
            label=m_name,
        )
        evaluated_points.append(point)

    # Defensive handling: single model
    if len(evaluated_points) == 1:
        single = evaluated_points[0]
        single.is_pareto_optimal = True
        single.classification = "BALANCED_KNEE"
        return ParetoFrontierResult(
            frontier_points=[single],
            all_evaluated_points=[single],
            knee_point=single,
            max_profit_point=single,
            min_risk_point=single,
            max_fairness_point=single,
            recommended_point=single,
            dimensions=dimensions,
            hypervolume=1.0,
            flight_recommendations=[f"Single model evaluated: {single.model_name} selected as balanced baseline."],
        )

    # 2. Non-Dominated Sorting (Pareto Optimality)
    n = len(evaluated_points)
    is_dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i != j and not is_dominated[i]:
                if _dominates(evaluated_points[j], evaluated_points[i]):
                    is_dominated[i] = True
                    break

    frontier_indices = [i for i in range(n) if not is_dominated[i]]
    for idx in frontier_indices:
        evaluated_points[idx].is_pareto_optimal = True

    frontier_points = [evaluated_points[i] for i in frontier_indices]

    # 3. Find Extreme Points on Frontier
    max_profit_pt = max(frontier_points, key=lambda p: p.roi_profit)
    min_risk_pt = min(frontier_points, key=lambda p: p.default_risk)
    max_fairness_pt = max(frontier_points, key=lambda p: p.fairness_ratio)

    # 4. Knee Point Detection (Normalized Distance to Utopia Point)
    # Utopia Point: max_profit, min_risk, max_fairness = (1.0, 0.0, 1.0)
    all_profits = [p.roi_profit for p in evaluated_points]
    all_risks = [p.default_risk for p in evaluated_points]
    all_fairness = [p.fairness_ratio for p in evaluated_points]

    min_p, max_p = min(all_profits), max(all_profits)
    min_r, max_r = min(all_risks), max(all_risks)
    min_f, max_f = min(all_fairness), max(all_fairness)

    range_p = max(1e-6, max_p - min_p)
    range_r = max(1e-6, max_r - min_r)
    range_f = max(1e-6, max_f - min_f)

    # Normalize weights so sum = 1.0
    total_w = max(1e-6, profit_weight + risk_weight + fairness_weight)
    w_p = profit_weight / total_w
    w_r = risk_weight / total_w
    w_f = fairness_weight / total_w

    best_dist = float("inf")
    knee_pt = frontier_points[0]

    for p in frontier_points:
        norm_p = (p.roi_profit - min_p) / range_p
        norm_r = (p.default_risk - min_r) / range_r
        norm_f = (p.fairness_ratio - min_f) / range_f

        # Distance to ideal Utopia point (profit=1, risk=0, fairness=1)
        dist_euclidean = math.sqrt(
            w_p * ((1.0 - norm_p) ** 2) +
            w_r * ((norm_r - 0.0) ** 2) +
            w_f * ((1.0 - norm_f) ** 2)
        )
        if dist_euclidean < best_dist:
            best_dist = dist_euclidean
            knee_pt = p

    # 5. Classify Frontier Points
    for p in frontier_points:
        if p is knee_pt:
            p.classification = "BALANCED_KNEE"
        elif p is max_profit_pt:
            p.classification = "MAX_PROFIT"
        elif p is min_risk_pt:
            p.classification = "MIN_RISK"
        elif p is max_fairness_pt:
            p.classification = "MAX_FAIRNESS"
        else:
            p.classification = "PARETO_OPTIMAL"

    # Hypervolume calculation
    nadir = (min_p, max_r, min_f)
    hv = _calculate_hypervolume_3d(frontier_points, nadir)

    # 6. Strategic Flight Recommendations
    recommendations: List[str] = [
        f"Balanced Knee Point selected: '{knee_pt.model_name}' achieves Net Profit ${knee_pt.roi_profit:,.2f} "
        f"with Risk Rate {knee_pt.default_risk:.2%} and Fairness Ratio {knee_pt.fairness_ratio:.2f}.",
    ]
    if max_profit_pt != knee_pt:
        profit_delta = max_profit_pt.roi_profit - knee_pt.roi_profit
        risk_delta = max_profit_pt.default_risk - knee_pt.default_risk
        recommendations.append(
            f"Max Profit variant '{max_profit_pt.model_name}' offers +${profit_delta:,.2f} profit "
            f"at the expense of +{risk_delta:.2%} risk."
        )
    if min_risk_pt != knee_pt:
        risk_saving = knee_pt.default_risk - min_risk_pt.default_risk
        recommendations.append(
            f"Min Risk variant '{min_risk_pt.model_name}' reduces risk by {risk_saving:.2%} "
            f"(Fairness: {min_risk_pt.fairness_ratio:.2f})."
        )
    if max_fairness_pt.fairness_ratio >= 0.80:
        recommendations.append(
            f"Demographic Parity compliance: '{max_fairness_pt.model_name}' satisfies EEOC Four-Fifths Rule "
            f"with Fairness Ratio {max_fairness_pt.fairness_ratio:.2f}."
        )

    return ParetoFrontierResult(
        frontier_points=frontier_points,
        all_evaluated_points=evaluated_points,
        knee_point=knee_pt,
        max_profit_point=max_profit_pt,
        min_risk_point=min_risk_pt,
        max_fairness_point=max_fairness_pt,
        recommended_point=knee_pt,
        dimensions=dimensions,
        hypervolume=hv,
        flight_recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Simulator Object for Threshold Sweeps
# ---------------------------------------------------------------------------

class ParetoFrontierSimulator:
    """Multi-objective Pareto optimizer balancing Profit, Risk, and Demographic Parity."""

    def __init__(self, steps: int = 80):
        self.steps = steps

    def compute_frontier(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        protected_series: Optional[pd.Series] = None,
        cost_fp: float = 20.0,
        cost_fn: float = 150.0,
        val_tp: float = 100.0,
        val_tn: float = 0.0,
        w_profit: float = 0.4,
        w_risk: float = 0.3,
        w_fairness: float = 0.3,
    ) -> ParetoFrontierResult:
        """
        Sweeps classification decision thresholds to construct the empirical Pareto frontier.
        """
        y_t = np.asarray(y_true).astype(int)
        y_p = np.asarray(y_prob).astype(float)
        n_samples = len(y_t)

        if n_samples == 0:
            return compute_pareto_frontier([])

        thresholds = np.linspace(0.01, 0.99, num=self.steps)
        models_data: List[Dict[str, Any]] = []

        # Identify sensitive groups if provided
        groups = None
        group_names: List[str] = []
        if protected_series is not None:
            groups = np.asarray(protected_series)
            group_names = [str(g) for g in np.unique(groups)]

        for tau in thresholds:
            y_pred = (y_p >= tau).astype(int)

            tp = int(np.sum((y_t == 1) & (y_pred == 1)))
            fp = int(np.sum((y_t == 0) & (y_pred == 1)))
            tn = int(np.sum((y_t == 0) & (y_pred == 0)))
            fn = int(np.sum((y_t == 1) & (y_pred == 0)))

            # 1. Net Profit
            profit = float(tp * val_tp + tn * val_tn - fp * cost_fp - fn * cost_fn)

            # 2. Risk / Default Rate (False Negative Rate or False Discovery Rate)
            actual_positives = tp + fn
            risk = float(fn / actual_positives) if actual_positives > 0 else 0.0

            # 3. Demographic Parity Fairness Ratio: min(P(Y_hat=1|G=g)) / max(P(Y_hat=1|G=g))
            if groups is not None and len(group_names) > 1:
                rates: List[float] = []
                for g in np.unique(groups):
                    mask = (groups == g)
                    g_size = np.sum(mask)
                    if g_size > 0:
                        acceptance_rate = float(np.mean(y_pred[mask]))
                        rates.append(acceptance_rate)
                if rates:
                    min_r = min(rates)
                    max_r = max(rates)
                    fairness = float(min_r / (max_r + 1e-6))
                    fairness = max(0.0, min(1.0, fairness))
                else:
                    fairness = 1.0
            else:
                # If no protected attribute, baseline fairness is 1.0 - absolute divergence from prevalence
                fairness = float(1.0 - abs(np.mean(y_pred) - np.mean(y_t)))
                fairness = max(0.0, min(1.0, fairness))

            models_data.append(
                {
                    "model_name": f"Threshold {tau:.2f}",
                    "threshold": round(float(tau), 4),
                    "roi_profit": round(profit, 2),
                    "default_risk": round(risk, 4),
                    "fairness_ratio": round(fairness, 4),
                    "metrics": {
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "accuracy": round((tp + tn) / n_samples, 4),
                    },
                }
            )

        res = compute_pareto_frontier(
            models_data=models_data,
            profit_weight=w_profit,
            risk_weight=w_risk,
            fairness_weight=w_fairness,
        )
        if protected_series is not None:
            res.protected_attribute = getattr(protected_series, "name", "protected_group") or "protected_group"
            res.sensitive_groups = group_names

        return res
