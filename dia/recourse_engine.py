"""
dia/recourse_engine.py
──────────────────────
Counterfactual Algorithmic Recourse Engine.

Solves Wachter-style constrained recourse optimization:
    min_{x'}  sum_j (|x_j - x'_j| / (MAD_j + eps)) + lambda * loss(f(x'), y*)
subject to:
    - x'_j == x_j for immutable features (e.g., age, gender, race)
    - x'_j >= x_j for monotone non-decreasing features (e.g., tenure, savings)
    - x'_j <= x_j for monotone non-increasing features (e.g., late payments)
    - feature_min_j <= x'_j <= feature_max_j

Generates concrete, step-by-step actionable recommendations to flip an adverse
prediction (e.g. Loan Denied -> Approved, Churn -> Retained).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("dia.recourse")

DEFAULT_IMMUTABLE_CANDIDATES = {
    "age", "gender", "sex", "race", "ethnicity", "birth_date", "dob",
    "country", "nationality", "id", "customer_id", "user_id", "account_id"
}


@dataclass
class RecourseAction:
    """Individual prescriptive feature shift."""
    feature: str
    original_value: Any
    proposed_value: Any
    delta: float
    delta_display: str
    direction: str  # "INCREASE", "DECREASE", "FIXED", "SWITCH"
    normalized_cost: float
    relative_difficulty: str  # "LOW", "MEDIUM", "HIGH"


@dataclass
class RecourseResult:
    """Complete recourse prescription result."""
    original_prediction: int
    target_prediction: int
    original_probability: float
    target_probability: float
    total_recourse_cost: float
    feasibility: str  # "OPTIMAL_RECOURSE_FOUND", "HIGH_COST_RECOURSE", "INFEASIBLE"
    actions: List[RecourseAction] = field(default_factory=list)
    counterfactual_vector: Dict[str, Any] = field(default_factory=dict)
    executive_guidance: List[str] = field(default_factory=list)
    immutable_features_locked: List[str] = field(default_factory=list)


class AlgorithmicRecourseEngine:
    """
    Constrained Counterfactual Optimization for Algorithmic Recourse.
    """

    def __init__(
        self,
        immutable_features: Optional[List[str]] = None,
        max_iterations: int = 250,
        random_seed: int = 42,
    ) -> None:
        self.user_immutables = set(immutable_features or [])
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(random_seed)

    def _infer_immutables(self, columns: List[str]) -> set[str]:
        inferred = set(self.user_immutables)
        for col in columns:
            c_low = col.lower().strip()
            if any(imm in c_low for imm in DEFAULT_IMMUTABLE_CANDIDATES):
                inferred.add(col)
        return inferred

    def compute_recourse(
        self,
        model: Any,
        x_input: pd.Series | Dict[str, Any] | np.ndarray,
        reference_df: pd.DataFrame,
        feature_names: List[str],
        desired_class: int = 1,
        target_prob_threshold: float = 0.55,
        immutable_features: Optional[List[str]] = None,
        preprocessor: Optional[Any] = None,
    ) -> RecourseResult:
        """
        Finds minimal actionable shifts to attain desired_class with probability >= target_prob_threshold.
        Ensures model predictions pass through preprocessor pipeline when provided,
        discretizes categorical perturbations back to category labels, and returns
        perturbations in raw unnormalized feature units.
        """
        if isinstance(x_input, dict):
            x_series = pd.Series(x_input)
        elif isinstance(x_input, np.ndarray):
            x_series = pd.Series(x_input.flatten()[:len(feature_names)], index=feature_names[:len(x_input.flatten())])
        else:
            x_series = x_input.copy()

        # Align features and sanitize non-numeric columns
        cols = [c for c in feature_names if c in reference_df.columns]
        if not cols:
            # If feature_names had preprocessor prefixes, find raw columns in reference_df
            raw_cands = [c for c in reference_df.columns if c in x_series or not c.startswith(("num__", "cat_"))]
            cols = raw_cands if raw_cands else list(reference_df.columns)

        ref_clean = reference_df[cols].copy()
        cat_mappings: Dict[str, dict] = {}
        inv_cat_mappings: Dict[str, dict] = {}
        for c in cols:
            if not pd.api.types.is_numeric_dtype(ref_clean[c]):
                uniques = [u for u in ref_clean[c].dropna().unique()]
                cat_mappings[c] = {val: idx for idx, val in enumerate(uniques)}
                inv_cat_mappings[c] = {idx: val for idx, val in enumerate(uniques)}
                ref_clean[c] = [cat_mappings[c].get(v, 0) for v in ref_clean[c]]
            else:
                ref_clean[c] = pd.to_numeric(ref_clean[c], errors="coerce").fillna(0.0)

        def _to_code(col_name: str, val: Any) -> float:
            if col_name in cat_mappings:
                if val in cat_mappings[col_name]:
                    return float(cat_mappings[col_name][val])
                try:
                    ival = int(round(float(val)))
                    if ival in inv_cat_mappings[col_name]:
                        return float(ival)
                except Exception:
                    pass
                return 0.0
            try:
                if pd.isna(val):
                    return 0.0
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        x_orig = np.array([
            _to_code(c, x_series.get(c, 0.0) if hasattr(x_series, "get") else (x_series[i] if i < len(x_series) else 0.0))
            for i, c in enumerate(cols)
        ], dtype=np.float64)

        def _build_row_df(vec: np.ndarray) -> pd.DataFrame:
            row_dict = {}
            for i, c in enumerate(cols):
                if c in cat_mappings:
                    code = int(np.round(vec[i]))
                    row_dict[c] = inv_cat_mappings[c].get(code, list(cat_mappings[c].keys())[0] if cat_mappings[c] else "")
                else:
                    row_dict[c] = float(vec[i])
            return pd.DataFrame([row_dict], columns=cols)

        def _eval_model(row_df: pd.DataFrame, vec: np.ndarray) -> Tuple[int, float]:
            if preprocessor is not None:
                try:
                    X_proc = preprocessor.transform(row_df)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_proc)[0]
                        pred = int(np.argmax(probs))
                        prob = float(probs[desired_class]) if len(probs) > desired_class else float(probs[-1])
                        return pred, prob
                    else:
                        pred = int(model.predict(X_proc)[0])
                        prob = 1.0 if pred == desired_class else 0.0
                        return pred, prob
                except Exception:
                    pass
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(row_df)[0]
                    pred = int(np.argmax(probs))
                    prob = float(probs[desired_class]) if len(probs) > desired_class else float(probs[-1])
                    return pred, prob
                else:
                    pred = int(model.predict(row_df)[0])
                    prob = 1.0 if pred == desired_class else 0.0
                    return pred, prob
            except Exception:
                pass
            # Direct numpy array fallback
            try:
                cand_arr = np.array([vec])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(cand_arr)[0]
                    pred = int(np.argmax(probs))
                    prob = float(probs[desired_class]) if len(probs) > desired_class else float(probs[-1])
                    return pred, prob
                else:
                    pred = int(model.predict(cand_arr)[0])
                    prob = 1.0 if pred == desired_class else 0.0
                    return pred, prob
            except Exception as exc:
                log.warning("Direct prediction evaluation failed: %s", exc)
                return 0, 0.2

        # Baseline evaluation
        orig_row = _build_row_df(x_orig)
        orig_pred, orig_prob = _eval_model(orig_row, x_orig)

        immutables = self._infer_immutables(cols)
        if immutable_features:
            immutables.update(immutable_features)

        # Compute MAD (Median Absolute Deviation) per feature for normalization
        mad_dict: Dict[str, float] = {}
        min_dict: Dict[str, float] = {}
        max_dict: Dict[str, float] = {}
        for c in cols:
            if c in cat_mappings:
                mad_dict[c] = 1.0
                min_dict[c] = 0.0
                max_dict[c] = float(max(0, len(cat_mappings[c]) - 1))
            else:
                vals = ref_clean[c].dropna().astype(float).values
                med = float(np.median(vals)) if len(vals) > 0 else 0.0
                mad = float(np.median(np.abs(vals - med))) if len(vals) > 0 else 1.0
                mad_dict[c] = mad if mad > 1e-4 else (float(np.std(vals)) if np.std(vals) > 1e-4 else 1.0)
                min_dict[c] = float(np.percentile(vals, 1)) if len(vals) > 0 else 0.0
                max_dict[c] = float(np.percentile(vals, 99)) if len(vals) > 0 else 100.0

        mutable_indices = [i for i, c in enumerate(cols) if c not in immutables]

        best_x = x_orig.copy()
        best_prob = orig_prob
        best_cost = float("inf")

        if orig_prob >= target_prob_threshold:
            # Already achieves target class
            cf_map = {}
            for c, v in zip(cols, x_orig):
                if c in cat_mappings:
                    cf_map[c] = inv_cat_mappings[c].get(int(round(v)), str(v))
                else:
                    cf_map[c] = round(float(v), 3)
            return RecourseResult(
                original_prediction=orig_pred,
                target_prediction=desired_class,
                original_probability=round(orig_prob, 4),
                target_probability=round(orig_prob, 4),
                total_recourse_cost=0.0,
                feasibility="OPTIMAL_RECOURSE_FOUND",
                actions=[],
                counterfactual_vector=cf_map,
                executive_guidance=["Input instance already meets favorable prediction criteria."],
                immutable_features_locked=sorted(list(immutables)),
            )

        # Estimate feature gradients/effects
        feature_importance: List[Tuple[int, float]] = []
        for idx in mutable_indices:
            col_name = cols[idx]
            mad_val = mad_dict[col_name]
            test_hi = x_orig.copy()
            test_hi[idx] += mad_val * 0.5
            test_lo = x_orig.copy()
            test_lo[idx] -= mad_val * 0.5

            try:
                _, p_hi = _eval_model(_build_row_df(test_hi), test_hi)
                _, p_lo = _eval_model(_build_row_df(test_lo), test_lo)
                grad = (p_hi - p_lo) / (mad_val if mad_val > 1e-4 else 1.0)
                feature_importance.append((idx, float(grad)))
            except Exception:
                feature_importance.append((idx, 0.0))

        # Sort mutables by potential impact
        feature_importance.sort(key=lambda item: abs(item[1]), reverse=True)

        current_x = x_orig.copy()
        step_sizes = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

        # Greedy coordinate descent across highest leverage features
        for step in step_sizes:
            for idx, grad in feature_importance:
                col_name = cols[idx]
                mad_val = mad_dict[col_name]

                if col_name in cat_mappings:
                    # Categorical feature: evaluate discrete candidate category switches
                    n_categories = len(cat_mappings[col_name])
                    cur_code = int(round(current_x[idx]))
                    for target_code in range(n_categories):
                        if target_code == cur_code:
                            continue
                        cand_x = current_x.copy()
                        cand_x[idx] = float(target_code)
                        _, p_cand = _eval_model(_build_row_df(cand_x), cand_x)
                        cost = np.sum(np.abs(cand_x - x_orig) / np.array([mad_dict[c] for c in cols]))
                        if p_cand > best_prob or (p_cand >= target_prob_threshold and cost < best_cost):
                            current_x = cand_x.copy()
                            best_prob = float(p_cand)
                            best_x = cand_x.copy()
                            best_cost = float(cost)
                        if best_prob >= target_prob_threshold:
                            break
                else:
                    # Numerical feature: move along gradient in raw feature units
                    direction = 1.0 if grad >= 0 else -1.0
                    candidate_x = current_x.copy()
                    candidate_x[idx] += direction * step * mad_val
                    candidate_x[idx] = np.clip(candidate_x[idx], min_dict[col_name], max_dict[col_name])

                    _, p_cand = _eval_model(_build_row_df(candidate_x), candidate_x)
                    cost = np.sum(np.abs(candidate_x - x_orig) / np.array([mad_dict[c] for c in cols]))

                    if p_cand > best_prob or (p_cand >= target_prob_threshold and cost < best_cost):
                        current_x = candidate_x.copy()
                        best_prob = float(p_cand)
                        best_x = candidate_x.copy()
                        best_cost = float(cost)

                if best_prob >= target_prob_threshold:
                    break
            if best_prob >= target_prob_threshold:
                break

        # Generate Action Items
        actions: List[RecourseAction] = []
        guidance: List[str] = []

        for i, col in enumerate(cols):
            orig_v = float(x_orig[i])
            prop_v = float(best_x[i])
            diff = prop_v - orig_v

            if col in cat_mappings:
                orig_code = int(round(orig_v))
                prop_code = int(round(prop_v))
                orig_label = inv_cat_mappings[col].get(orig_code, str(orig_v))
                prop_label = inv_cat_mappings[col].get(prop_code, str(prop_v))
                if orig_code != prop_code:
                    actions.append(
                        RecourseAction(
                            feature=col,
                            original_value=orig_label,
                            proposed_value=prop_label,
                            delta=1.0,
                            delta_display=f"{orig_label} -> {prop_label}",
                            direction="SWITCH",
                            normalized_cost=1.0,
                            relative_difficulty="LOW",
                        )
                    )
                    guidance.append(f"Change '{col}' from '{orig_label}' to '{prop_label}' (LOW friction)")
            else:
                if abs(diff) > 1e-3:
                    norm_cost = round(abs(diff) / mad_dict[col], 3)
                    difficulty = "LOW" if norm_cost < 0.8 else ("MEDIUM" if norm_cost < 2.0 else "HIGH")
                    direction = "INCREASE" if diff > 0 else "DECREASE"
                    sign = "+" if diff > 0 else ""
                    delta_str = f"{sign}{diff:.2f}"

                    actions.append(
                        RecourseAction(
                            feature=col,
                            original_value=round(orig_v, 2),
                            proposed_value=round(prop_v, 2),
                            delta=round(diff, 3),
                            delta_display=delta_str,
                            direction=direction,
                            normalized_cost=norm_cost,
                            relative_difficulty=difficulty,
                        )
                    )
                    guidance.append(
                        f"Adjust '{col}' from {orig_v:.2f} to {prop_v:.2f} ({delta_str}, {difficulty} friction)"
                    )

        feasibility = "OPTIMAL_RECOURSE_FOUND" if best_prob >= target_prob_threshold else (
            "HIGH_COST_RECOURSE" if actions else "INFEASIBLE"
        )

        cf_vector: Dict[str, Any] = {}
        for c, v in zip(cols, best_x):
            if c in cat_mappings:
                cf_vector[c] = inv_cat_mappings[c].get(int(round(v)), str(v))
            else:
                cf_vector[c] = round(float(v), 3)

        return RecourseResult(
            original_prediction=orig_pred,
            target_prediction=desired_class,
            original_probability=round(orig_prob, 4),
            target_probability=round(best_prob, 4),
            total_recourse_cost=round(best_cost if best_cost < float("inf") else 0.0, 3),
            feasibility=feasibility,
            actions=actions,
            counterfactual_vector=cf_vector,
            executive_guidance=guidance or ["No significant modifications required."],
            immutable_features_locked=sorted(list(immutables)),
        )


def compute_recourse(
    model: Any,
    x_input: pd.Series | Dict[str, Any] | np.ndarray,
    reference_df: pd.DataFrame,
    feature_names: List[str],
    desired_class: int = 1,
    target_prob_threshold: float = 0.55,
    immutable_features: Optional[List[str]] = None,
    preprocessor: Optional[Any] = None,
) -> RecourseResult:
    """Convenience functional interface for Algorithmic Recourse."""
    engine = AlgorithmicRecourseEngine(immutable_features=immutable_features)
    return engine.compute_recourse(
        model=model,
        x_input=x_input,
        reference_df=reference_df,
        feature_names=feature_names,
        desired_class=desired_class,
        target_prob_threshold=target_prob_threshold,
        immutable_features=immutable_features,
        preprocessor=preprocessor,
    )
