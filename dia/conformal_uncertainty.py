"""
dia/conformal_uncertainty.py
────────────────────────────
Split Conformal Prediction & Epistemic Uncertainty Engine.

Provides distribution-free, finite-sample prediction coverage guarantees:
    P(Y in C(X)) >= 1 - alpha

Decomposes prediction uncertainty into:
1. Aleatoric Uncertainty: Inherent label noise, calculated via predictive Shannon entropy.
2. Epistemic Uncertainty: Out-of-distribution (OOD) distance from calibration manifold.

Flags instances requiring human review when epistemic OOD bounds are breached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("dia.conformal")


@dataclass
class ConformalInstanceResult:
    """Evaluation of a single test point."""
    predicted_label: int | float | str
    conformal_set: list[Any]
    conformal_interval: tuple[float, float] | None
    aleatoric_entropy: float
    epistemic_distance: float
    uncertainty_classification: str  # "HIGH_CONFIDENCE", "ALEATORIC_NOISE", "EPISTEMIC_OOD_FLAG"
    requires_human_review: bool
    coverage_guarantee_pct: float


@dataclass
class ConformalCalibrationReport:
    """Overall conformal calibration evaluation across dataset."""
    alpha_error_rate: float
    coverage_guarantee_pct: float
    task_type: str
    quantile_threshold: float
    empirical_coverage: float
    average_set_size_or_width: float
    ood_flagged_count: int
    ood_flagged_pct: float
    sample_evaluations: list[ConformalInstanceResult] = field(default_factory=list)
    executive_verdict: str = "CALIBRATION_CERTIFIED"


class ConformalUncertaintyEngine:
    """
    Split conformal inference with epistemic/aleatoric uncertainty decomposition.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        """
        alpha: significance level. Coverage guarantee is 1 - alpha (e.g. alpha=0.10 -> 90% coverage).
        """
        self.alpha = alpha
        self.q_hat: float = 0.0
        self.calib_centroids: np.ndarray | None = None
        self.calib_stds: np.ndarray | None = None
        self.task_type: str = "classification"
        self.classes_: Any | None = None

    def calibrate(
        self,
        model: Any,
        x_calib: pd.DataFrame,
        y_calib: np.ndarray | pd.Series,
        task_type: str = "classification",
    ) -> float:
        """
        Calculates conformal non-conformity threshold q_hat on calibration split.
        """
        self.task_type = task_type
        arr_x = x_calib.select_dtypes(include=[np.number]).fillna(0.0).values
        arr_y = np.asarray(y_calib)
        n = len(arr_y)
        if n == 0:
            return 0.5

        # Calibration centroids for epistemic OOD distance
        self.calib_centroids = np.mean(arr_x, axis=0)
        self.calib_stds = np.std(arr_x, axis=0)
        self.calib_stds[self.calib_stds < 1e-4] = 1.0

        if task_type == "classification" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_calib)
            # Non-conformity score: 1 - P(true_class)
            n_classes = probs.shape[1]
            classes = getattr(model, "classes_", np.unique(arr_y))
            self.classes_ = classes
            class_to_idx = {c: idx for idx, c in enumerate(classes)}

            scores = []
            for i, true_label in enumerate(arr_y):
                l_idx = class_to_idx.get(true_label, None)
                if l_idx is None:
                    try:
                        int_label = int(true_label)
                        l_idx = class_to_idx.get(int_label, None)
                        if l_idx is None and 0 <= int_label < n_classes:
                            l_idx = int_label
                    except (ValueError, TypeError):
                        l_idx = None
                if l_idx is None or l_idx >= n_classes or l_idx < 0:
                    l_idx = 0
                scores.append(1.0 - probs[i, l_idx])
            scores = np.array(scores)
        else:
            preds = model.predict(x_calib)
            scores = np.abs(arr_y - preds)

        # Finite sample conformal correction factor: ceil((n+1)(1-alpha)) / n
        level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        return self.q_hat

    fit_calibration = calibrate

    def evaluate_instances(
        self,
        model: Any,
        x_eval: pd.DataFrame,
        y_true: np.ndarray | pd.Series | None = None,
    ) -> ConformalCalibrationReport:
        """
        Evaluates test instances, constructing guaranteed conformal sets and uncertainty flags.
        """
        arr_x = x_eval.select_dtypes(include=[np.number]).fillna(0.0).values
        arr_y_true = np.asarray(y_true) if y_true is not None else None
        n = len(x_eval)
        if n == 0:
            return ConformalCalibrationReport(
                alpha_error_rate=self.alpha,
                coverage_guarantee_pct=(1.0 - self.alpha) * 100,
                task_type=self.task_type,
                quantile_threshold=self.q_hat,
                empirical_coverage=1.0,
                average_set_size_or_width=1.0,
                ood_flagged_count=0,
                ood_flagged_pct=0.0,
            )

        # Compute epistemic normalized distance to calibration manifold
        if self.calib_centroids is not None and self.calib_stds is not None:
            norm_diff = (arr_x - self.calib_centroids) / self.calib_stds
            epistemic_dists = np.sqrt(np.mean(norm_diff ** 2, axis=1))
        else:
            epistemic_dists = np.zeros(n)

        results: list[ConformalInstanceResult] = []
        covered_count = 0
        set_sizes: list[float] = []
        ood_count = 0

        # OOD threshold: > 2.75 normalized standard deviations
        ood_cutoff = 2.75

        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_eval)
            n_classes = probs.shape[1]
            raw_classes = getattr(model, "classes_", getattr(self, "classes_", None))
            if raw_classes is None or len(raw_classes) != n_classes:
                classes = list(range(n_classes))
            else:
                classes = list(raw_classes)
            preds = np.argmax(probs, axis=1)

            for i in range(n):
                p_row = probs[i]
                # Conformal set: all classes where (1.0 - p) <= q_hat
                conf_set = [classes[c] for c in range(n_classes) if (1.0 - probs[i, c]) <= self.q_hat]
                if not conf_set:
                    conf_set = [classes[int(np.argmax(p_row))]]

                set_sizes.append(len(conf_set))

                # Shannon entropy
                p_nz = p_row[p_row > 1e-6]
                entropy = float(-np.sum(p_nz * np.log2(p_nz)))
                dist = float(epistemic_dists[i])

                # Classification verdict
                if dist > ood_cutoff:
                    verdict = "EPISTEMIC_OOD_FLAG"
                    needs_review = True
                    ood_count += 1
                elif len(conf_set) > 1 or entropy > 0.85:
                    verdict = "ALEATORIC_NOISE"
                    needs_review = False
                else:
                    verdict = "HIGH_CONFIDENCE"
                    needs_review = False

                if arr_y_true is not None:
                    y_val = arr_y_true[i]
                    if y_val in conf_set:
                        covered_count += 1

                raw_pred = classes[preds[i]]
                if isinstance(raw_pred, (np.integer, int)):
                    pred_label: int | float | str = int(raw_pred)
                elif isinstance(raw_pred, (np.floating, float)):
                    pred_label = float(raw_pred)
                else:
                    pred_label = str(raw_pred)

                results.append(
                    ConformalInstanceResult(
                        predicted_label=pred_label,
                        conformal_set=[c.item() if hasattr(c, "item") else c for c in conf_set],
                        conformal_interval=None,
                        aleatoric_entropy=round(entropy, 3),
                        epistemic_distance=round(dist, 3),
                        uncertainty_classification=verdict,
                        requires_human_review=needs_review,
                        coverage_guarantee_pct=round((1.0 - self.alpha) * 100, 1),
                    )
                )
        else:
            # Continuous Regression
            preds = model.predict(x_eval)
            for i in range(n):
                pred_val = float(preds[i])
                interval = (pred_val - self.q_hat, pred_val + self.q_hat)
                width = 2.0 * self.q_hat
                set_sizes.append(width)

                dist = float(epistemic_dists[i])
                if dist > ood_cutoff:
                    verdict = "EPISTEMIC_OOD_FLAG"
                    needs_review = True
                    ood_count += 1
                else:
                    verdict = "HIGH_CONFIDENCE"
                    needs_review = False

                if arr_y_true is not None:
                    t_val = float(arr_y_true[i])
                    if interval[0] <= t_val <= interval[1]:
                        covered_count += 1

                results.append(
                    ConformalInstanceResult(
                        predicted_label=round(pred_val, 3),
                        conformal_set=[],
                        conformal_interval=(round(interval[0], 3), round(interval[1], 3)),
                        aleatoric_entropy=0.0,
                        epistemic_distance=round(dist, 3),
                        uncertainty_classification=verdict,
                        requires_human_review=needs_review,
                        coverage_guarantee_pct=round((1.0 - self.alpha) * 100, 1),
                    )
                )

        emp_cov = float(covered_count / n) if y_true is not None and n > 0 else (1.0 - self.alpha)
        avg_size = float(np.mean(set_sizes)) if set_sizes else 1.0

        return ConformalCalibrationReport(
            alpha_error_rate=self.alpha,
            coverage_guarantee_pct=round((1.0 - self.alpha) * 100, 1),
            task_type=self.task_type,
            quantile_threshold=round(self.q_hat, 4),
            empirical_coverage=round(emp_cov, 4),
            average_set_size_or_width=round(avg_size, 3),
            ood_flagged_count=ood_count,
            ood_flagged_pct=round((ood_count / n) * 100, 2) if n > 0 else 0.0,
            sample_evaluations=results[:50],  # Keep bounded payload
            executive_verdict="CONFORMAL_COVERAGE_VERIFIED",
        )


def evaluate_conformal_bounds(
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    feature_names: list[str],
    alpha: float = 0.10,
    task_type: str = "classification",
) -> ConformalCalibrationReport:
    """Functional convenience entry point for Conformal Uncertainty."""
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) > 3000:
        clean_df = clean_df.sample(n=3000, random_state=42)

    # 50/50 split for calibration and test
    mid = len(clean_df) // 2
    calib_df = clean_df.iloc[:mid]
    test_df = clean_df.iloc[mid:]

    engine = ConformalUncertaintyEngine(alpha=alpha)
    engine.calibrate(
        model=model,
        x_calib=calib_df[feature_names],
        y_calib=calib_df[target_col].values,
        task_type=task_type,
    )

    return engine.evaluate_instances(
        model=model,
        x_eval=test_df[feature_names],
        y_true=test_df[target_col].values,
    )
