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
    medians = np.median(X_reference, axis=0)
    mins = np.min(X_reference, axis=0)
    maxs = np.max(X_reference, axis=0)

    orig_pred = model.predict(x_orig.reshape(1, -1))[0]
    if hasattr(model, "predict_proba"):
        orig_proba = model.predict_proba(x_orig.reshape(1, -1))[0]
    else:
        orig_proba = None

    if orig_pred == desired_outcome:
        return {
            "status": "already_desired",
            "message": "Instance already satisfies the desired outcome.",
            "original_prediction": orig_pred,
            "counterfactual_prediction": orig_pred,
            "perturbations": [],
        }

    best_cf = x_curr.copy()
    found = False

    # Coordinate search
    for iteration in range(max_iter):
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

            pred_pos = model.predict(candidate_pos.reshape(1, -1))[0]
            pred_neg = model.predict(candidate_neg.reshape(1, -1))[0]

            if pred_pos == desired_outcome:
                best_cf = candidate_pos
                found = True
                break
            if pred_neg == desired_outcome:
                best_cf = candidate_neg
                found = True
                break

            # Distance tracking
            dist_pos = np.sum(((candidate_pos - x_orig) / stds) ** 2)
            dist_neg = np.sum(((candidate_neg - x_orig) / stds) ** 2)
            diffs.append((dist_pos, candidate_pos))
            diffs.append((dist_neg, candidate_neg))

        if found:
            break

        # Move along least distance direction
        diffs.sort(key=lambda x: x[0])
        best_cf = diffs[0][1]

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

    m_treat.fit(X[idx_treat], y[idx_treat])
    m_ctrl.fit(X[idx_ctrl], y[idx_ctrl])

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
    # Persuadable: positive impact of treatment (p_treat >= 0.5, p_ctrl < 0.5)
    # Sure Thing: positive outcome regardless (p_treat >= 0.5, p_ctrl >= 0.5)
    # Lost Cause: negative outcome regardless (p_treat < 0.5, p_ctrl < 0.5)
    # Sleeping Dog / Do-not-disturb: treatment causes negative outcome (p_treat < 0.5, p_ctrl >= 0.5)
    quadrants = []
    for pt, pc in zip(p_treat, p_ctrl):
        if pt >= 0.5 and pc < 0.5:
            quadrants.append("Persuadables (High ROI Target)")
        elif pt >= 0.5 and pc >= 0.5:
            quadrants.append("Sure Things (Organic Success)")
        elif pt < 0.5 and pc < 0.5:
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
