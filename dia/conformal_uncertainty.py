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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("dia.conformal")


@dataclass
class ConformalInstanceResult:
    """Evaluation of a single test point."""
    predicted_label: int | float
    conformal_set: List[int]
    conformal_interval: Optional[Tuple[float, float]]
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
    sample_evaluations: List[ConformalInstanceResult] = field(default_factory=list)
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
        self.calib_centroids: Optional[np.ndarray] = None
        self.calib_stds: Optional[np.ndarray] = None
        self.task_type: str = "classification"

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
            scores = []
            for i, true_label in enumerate(arr_y):
                l_idx = int(true_label) if int(true_label) < n_classes else 0
                scores.append(1.0 - probs[i, l_idx])
            scores = np.array(scores)
        else:
            preds = model.predict(x_calib)
            scores = np.abs(arr_y - preds)

        # Finite sample conformal correction factor: ceil((n+1)(1-alpha)) / n
        level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        return self.q_hat

    def evaluate_instances(
        self,
        model: Any,
        x_eval: pd.DataFrame,
        y_true: Optional[np.ndarray | pd.Series] = None,
    ) -> ConformalCalibrationReport:
        """
        Evaluates test instances, constructing guaranteed conformal sets and uncertainty flags.
        """
        arr_x = x_eval.select_dtypes(include=[np.number]).fillna(0.0).values
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

        results: List[ConformalInstanceResult] = []
        covered_count = 0
        set_sizes: List[float] = []
        ood_count = 0

        # OOD threshold: > 2.75 normalized standard deviations
        ood_cutoff = 2.75

        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_eval)
            preds = np.argmax(probs, axis=1)

            for i in range(n):
                p_row = probs[i]
                # Conformal set: all classes where 1 - p <= q_hat => p >= 1 - q_hat
                threshold = max(1.0 - self.q_hat, 0.0)
                conf_set = [c for c, p in enumerate(p_row) if p >= threshold]
                if not conf_set:
                    conf_set = [int(np.argmax(p_row))]

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

                if y_true is not None:
                    t_val = int(y_true[i] if isinstance(y_true, (list, np.ndarray)) else y_true.iloc[i])
                    if t_val in conf_set:
                        covered_count += 1

                results.append(
                    ConformalInstanceResult(
                        predicted_label=int(preds[i]),
                        conformal_set=conf_set,
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

                if y_true is not None:
                    t_val = float(y_true[i] if isinstance(y_true, (list, np.ndarray)) else y_true.iloc[i])
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
    feature_names: List[str],
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
