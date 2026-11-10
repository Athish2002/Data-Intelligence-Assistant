"""
dia/causal_engine.py
────────────────────
Causal Machine Learning & Decision Intelligence Engine.
Provides Counterfactual Explanations, Uplift Modeling (T-Learner),
and Causal Directed Dependency Estimation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

log = logging.getLogger("dia.causal")


def generate_counterfactual(
    model: Any,
    instance: pd.Series,
    feature_names: list[str],
    desired_outcome: int | float,
    X_reference: np.ndarray,
    max_iter: int = 200,
    step_size: float = 0.05,
) -> dict[str, Any]:
    """
    Finds the minimal actionable perturbation to flip an instance's prediction
    to the desired outcome using gradient-free coordinate search.
    """
    x_orig = instance[feature_names].values.astype(float)
    x_curr = x_orig.copy()

    # Pre-calculate feature standard deviations and ranges for normalization
    stds = np.std(X_reference, axis=0)
    stds[stds == 0] = 1.0
    mins = np.min(X_reference, axis=0)
    maxs = np.max(X_reference, axis=0)

    orig_pred = model.predict(x_orig.reshape(1, -1))[0]
    if hasattr(model, "predict_proba"):
        orig_proba = model.predict_proba(x_orig.reshape(1, -1))[0]
    else:
        orig_proba = None

    orig_confidence = float(np.max(orig_proba)) if orig_proba is not None else None

    if orig_pred == desired_outcome:
        return {
            "status": "already_desired",
            "message": "Instance already satisfies the desired outcome.",
            "original_prediction": orig_pred,
            "original_prediction_confidence": orig_confidence,
            "counterfactual_prediction": orig_pred,
            "perturbations": [],
        }

    best_cf = x_curr.copy()
    found = False

    # Target probability indexing for classification
    desired_idx: int | None = None
    if orig_proba is not None:
        if hasattr(model, "classes_"):
            classes_arr = getattr(model, "classes_", None)
            if classes_arr is not None:
                matches = np.where(classes_arr == desired_outcome)[0]
                if len(matches) > 0:
                    desired_idx = int(matches[0])
        if desired_idx is None:
            if isinstance(desired_outcome, (int, np.integer)) and 0 <= int(desired_outcome) < len(orig_proba):
                desired_idx = int(desired_outcome)
            else:
                desired_idx = 1 if len(orig_proba) > 1 else 0

    def _calc_score(cand: np.ndarray) -> tuple[float, bool]:
        pred_cand = model.predict(cand.reshape(1, -1))[0]
        is_desired = bool(pred_cand == desired_outcome)
        dist = float(np.sum(((cand - x_orig) / stds) ** 2))

        if hasattr(model, "predict_proba") and desired_idx is not None:
            probs = model.predict_proba(cand.reshape(1, -1))[0]
            p_desired = float(probs[desired_idx]) if desired_idx < len(probs) else float(probs[-1])
            score = (1.0 - p_desired) + 0.05 * dist
        elif hasattr(model, "decision_function"):
            df = model.decision_function(cand.reshape(1, -1))
            val = float(df[0]) if np.ndim(df) == 1 else float(df[0][0])
            is_pos = (
                desired_outcome == 1
                or (hasattr(model, "classes_") and len(model.classes_) == 2 and desired_outcome == model.classes_[1])
            )
            target_margin = val if is_pos else -val
            p_desired = 1.0 / (1.0 + np.exp(-np.clip(target_margin, -20.0, 20.0)))
            score = (1.0 - p_desired) + 0.05 * dist
        else:
            # Regression or predict-only models: margin to target threshold
            margin = abs(float(pred_cand) - float(desired_outcome))
            score = margin + 0.05 * dist

        return score, is_desired

    # Coordinate search
    for _iteration in range(max_iter):
        pred = model.predict(best_cf.reshape(1, -1))[0]
        if pred == desired_outcome:
            found = True
            break

        # Perturb each feature towards reference median / distribution
        diffs = []
        for i in range(len(feature_names)):
            candidate_pos = best_cf.copy()
            candidate_neg = best_cf.copy()

            step = step_size * stds[i]
            candidate_pos[i] = min(maxs[i], candidate_pos[i] + step)
            candidate_neg[i] = max(mins[i], candidate_neg[i] - step)

            score_pos, match_pos = _calc_score(candidate_pos)
            if match_pos:
                best_cf = candidate_pos
                found = True
                break

            score_neg, match_neg = _calc_score(candidate_neg)
            if match_neg:
                best_cf = candidate_neg
                found = True
                break

            diffs.append((score_pos, candidate_pos))
            diffs.append((score_neg, candidate_neg))

        if found:
            break

        # Move along candidate minimizing guidance score
        if diffs:
            diffs.sort(key=lambda x: x[0])
            best_cf = diffs[0][1]

    cf_pred = model.predict(best_cf.reshape(1, -1))[0]
    if cf_pred == desired_outcome:
        found = True

    # Calculate changes
    changes = []
    for i, name in enumerate(feature_names):
        delta = best_cf[i] - x_orig[i]
        if abs(delta) > 1e-4:
            pct_change = (delta / max(1e-4, abs(x_orig[i]))) * 100.0 if abs(x_orig[i]) > 1e-4 else 0.0
            changes.append({
                "feature": name,
                "original_value": round(float(x_orig[i]), 2),
                "counterfactual_value": round(float(best_cf[i]), 2),
                "delta": round(float(delta), 2),
                "pct_change": round(float(pct_change), 1),
                "action": "Increase" if delta > 0 else "Decrease",
            })

    cf_pred = model.predict(best_cf.reshape(1, -1))[0]
    return {
        "status": "success" if found else "approximate",
        "original_prediction": orig_pred,
        "original_prediction_confidence": orig_confidence,
        "counterfactual_prediction": cf_pred,
        "desired_outcome": desired_outcome,
        "total_features_modified": len(changes),
        "perturbations": sorted(changes, key=lambda x: abs(x["pct_change"]), reverse=True),
    }


def estimate_uplift_t_learner(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Implements a Causal T-Learner (Two-Model Approach) to estimate Individual Treatment Effect (ITE)
    and categorize users into Uplift quadrants: Persuadables, Sure Things, Lost Causes, Sleeping Dogs.
    """
    idx_treat = (treatment == 1)
    idx_ctrl = (treatment == 0)

    if sum(idx_treat) < 5 or sum(idx_ctrl) < 5:
        return {
            "status": "insufficient_data",
            "message": "Treatment groups require at least 5 samples each for Causal Uplift estimation.",
        }

    # Fit Model 1 on Treatment group, Model 0 on Control group
    m_treat = clone(model)
    m_ctrl = clone(model)

    y_treat = y[idx_treat]
    y_ctrl = y[idx_ctrl]

    is_classification = hasattr(model, "predict_proba") or hasattr(model, "classes_")
    if is_classification and (len(np.unique(y_treat)) < 2 or len(np.unique(y_ctrl)) < 2):
        return {
            "status": "insufficient_data",
            "message": "Both treatment and control groups must contain at least two target classes for causal uplift estimation.",
        }

    try:
        m_treat.fit(X[idx_treat], y_treat)
        m_ctrl.fit(X[idx_ctrl], y_ctrl)
    except ValueError as e:
        return {
            "status": "insufficient_data",
            "message": f"Unable to fit causal models on subgroup splits: {str(e)}",
        }

    # Predict factual and counterfactual outcomes
    if hasattr(m_treat, "predict_proba"):
        p_treat = m_treat.predict_proba(X)[:, 1]
        p_ctrl = m_ctrl.predict_proba(X)[:, 1]
    else:
        p_treat = m_treat.predict(X)
        p_ctrl = m_ctrl.predict(X)

    ite = p_treat - p_ctrl
    ate = float(np.mean(ite))

    # Segment users into 4 Uplift Quadrants
    # Persuadable: positive impact of treatment
    # Sure Thing: positive outcome regardless
    # Lost Cause: negative outcome regardless
    # Sleeping Dog / Do-not-disturb: treatment causes negative outcome
    is_prob = bool(np.all((p_treat >= 0.0) & (p_treat <= 1.0)) and np.all((p_ctrl >= 0.0) & (p_ctrl <= 1.0)))
    thresh = 0.5 if is_prob else float(np.median(p_ctrl))

    quadrants = []
    for pt, pc in zip(p_treat, p_ctrl, strict=True):
        if pt >= thresh and pc < thresh:
            quadrants.append("Persuadables (High ROI Target)")
        elif pt >= thresh and pc >= thresh:
            quadrants.append("Sure Things (Organic Success)")
        elif pt < thresh and pc < thresh:
            quadrants.append("Lost Causes (No Intervention ROI)")
        else:
            quadrants.append("Sleeping Dogs (Do-Not-Disturb)")

    quad_counts = pd.Series(quadrants).value_counts().to_dict()

    return {
        "status": "success",
        "average_treatment_effect_ate": round(ate, 4),
        "ite_mean": round(float(np.mean(ite)), 4),
        "ite_std": round(float(np.std(ite)), 4),
        "uplift_quadrant_distribution": quad_counts,
        "recommended_action": (
            "Deploy targeted intervention: High concentration of Persuadables detected."
            if quad_counts.get("Persuadables (High ROI Target)", 0) > len(X) * 0.2
            else "Standard campaign strategy recommended."
        ),
    }
