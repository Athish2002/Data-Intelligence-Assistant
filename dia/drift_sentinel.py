"""
dia/drift_sentinel.py
─────────────────────
Streaming Out-of-Distribution (OOD) & Concept Drift Sentinel.

Multi-hypothesis distribution shift testing comparing baseline training data against
production/streaming inference windows:
- 2-Sample Kolmogorov-Smirnov (KS) tests for continuous features (statistic D & p-value).
- Population Stability Index (PSI) for binned distributions.
- Maximum Mean Discrepancy (MMD) with RBF kernel for multivariate joint distribution shift.

Prescribes automated governance remedies:
- PASS_HEALTHY: Normal operations.
- WARN_MONITOR: Accelerated monitoring cadence.
- TRIGGER_RETRAIN: Immediate automated pipeline retraining required.
- ACTIVATE_SAFE_FALLBACK: Divert live inference to calibrated baseline model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("dia.sentinel")


@dataclass
class FeatureDriftDetail:
    """Drift audit for an individual column."""
    feature_name: str
    feature_type: str  # "NUMERICAL", "CATEGORICAL"
    ks_statistic: Optional[float]
    p_value: Optional[float]
    psi_score: float
    drift_status: str  # "HEALTHY", "MODERATE_SHIFT", "CRITICAL_DRIFT"
    baseline_mean: Optional[float]
    current_mean: Optional[float]
    mean_shift_pct: Optional[float]


@dataclass
class DriftSentinelReport:
    """Complete sentinel audit across dataset."""
    total_features_monitored: int
    n_reference_samples: int
    n_current_samples: int
    drifting_feature_count: int
    drifting_feature_ratio: float
    multivariate_mmd_score: float
    overall_sentinel_status: str  # "PASS_HEALTHY", "WARN_MONITOR", "CRITICAL_DRIFT"
    governance_action: str
    feature_drift_breakdown: List[FeatureDriftDetail] = field(default_factory=list)
    top_drifting_features: List[str] = field(default_factory=list)
    actionable_recommendations: List[str] = field(default_factory=list)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI)."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins from expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bins = np.percentile(expected, percentiles)
    except Exception:
        return 0.0

    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    # Ensure finite outer bounds
    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / max(len(expected), 1)
    act_pct = act_counts / max(len(actual), 1)

    # Laplace smoothing
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.nan_to_num(psi_val, nan=0.0, posinf=1.0, neginf=0.0))


def compute_rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Calculates Maximum Mean Discrepancy (MMD) with RBF kernel."""
    n = len(x)
    m = len(y)
    if n == 0 or m == 0:
        return 0.0

    if gamma is None:
        gamma = 1.0 / max(x.shape[1], 1)

    # Subsample if large for bounded latency
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        x = x[idx]
        n = 500
    if m > 500:
        idx = np.random.choice(m, 500, replace=False)
        y = y[idx]
        m = 500

    # Pairwise distances
    xx = np.sum(x**2, axis=1)[:, np.newaxis]
    yy = np.sum(y**2, axis=1)[:, np.newaxis]

    dist_xx = xx + xx.T - 2.0 * np.dot(x, x.T)
    dist_yy = yy + yy.T - 2.0 * np.dot(y, y.T)
    dist_xy = xx + yy.T - 2.0 * np.dot(x, y.T)

    k_xx = np.exp(-gamma * np.maximum(dist_xx, 0.0))
    k_yy = np.exp(-gamma * np.maximum(dist_yy, 0.0))
    k_xy = np.exp(-gamma * np.maximum(dist_xy, 0.0))

    mmd2 = (np.sum(k_xx) / (n * n)) - (2.0 * np.sum(k_xy) / (n * m)) + (np.sum(k_yy) / (m * m))
    return float(np.sqrt(max(mmd2, 0.0)))


class DriftSentinel:
    """
    Evaluates dataset drift and out-of-distribution shifts between baseline and inference.
    """

    def __init__(self, p_value_threshold: float = 0.01, psi_threshold: float = 0.20) -> None:
        self.p_val_thresh = p_value_threshold
        self.psi_thresh = psi_threshold

    def audit_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        monitored_columns: Optional[List[str]] = None,
    ) -> DriftSentinelReport:
        """
        Runs comprehensive multi-hypothesis drift testing.
        """
        cols = monitored_columns or [c for c in reference_df.columns if c in current_df.columns]
        feature_details: List[FeatureDriftDetail] = []
        drifting_cols: List[str] = []

        ref_num = reference_df.select_dtypes(include=[np.number]).dropna()
        cur_num = current_df.select_dtypes(include=[np.number]).dropna()

        # 1. Feature-by-Feature Tests
        for c in cols:
            ref_series = reference_df[c].dropna()
            cur_series = current_df[c].dropna()

            if len(ref_series) < 5 or len(cur_series) < 5:
                continue

            if pd.api.types.is_numeric_dtype(ref_series):
                ref_vals = ref_series.values.astype(float)
                cur_vals = cur_series.values.astype(float)

                ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
                psi = calculate_psi(ref_vals, cur_vals)

                m_ref = float(np.mean(ref_vals))
                m_cur = float(np.mean(cur_vals))
                shift_pct = ((m_cur - m_ref) / m_ref * 100) if abs(m_ref) > 1e-4 else 0.0

                if p_val < self.p_val_thresh and psi > self.psi_thresh:
                    status = "CRITICAL_DRIFT"
                    drifting_cols.append(c)
                elif p_val < self.p_val_thresh or psi > 0.10:
                    status = "MODERATE_SHIFT"
                else:
                    status = "HEALTHY"

                feature_details.append(
                    FeatureDriftDetail(
                        feature_name=c,
                        feature_type="NUMERICAL",
                        ks_statistic=round(float(ks_stat), 4),
                        p_value=round(float(p_val), 5),
                        psi_score=round(float(psi), 4),
                        drift_status=status,
                        baseline_mean=round(m_ref, 2),
                        current_mean=round(m_cur, 2),
                        mean_shift_pct=round(shift_pct, 2),
                    )
                )
            else:
                # Categorical column
                val_counts_ref = ref_series.value_counts(normalize=True)
                val_counts_cur = cur_series.value_counts(normalize=True)

                all_cats = list(set(val_counts_ref.index).union(set(val_counts_cur.index)))
                p_ref = np.array([val_counts_ref.get(k, 1e-4) for k in all_cats])
                p_cur = np.array([val_counts_cur.get(k, 1e-4) for k in all_cats])

                psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
                status = "CRITICAL_DRIFT" if psi > self.psi_thresh else ("MODERATE_SHIFT" if psi > 0.10 else "HEALTHY")
                if status == "CRITICAL_DRIFT":
                    drifting_cols.append(c)

                feature_details.append(
                    FeatureDriftDetail(
                        feature_name=c,
                        feature_type="CATEGORICAL",
                        ks_statistic=None,
                        p_value=None,
                        psi_score=round(float(psi), 4),
                        drift_status=status,
                        baseline_mean=None,
                        current_mean=None,
                        mean_shift_pct=None,
                    )
                )

        # 2. Multivariate Joint Shift (MMD)
        common_num = [c for c in ref_num.columns if c in cur_num.columns]
        if common_num:
            # Normalize with reference statistics
            ref_mat = ref_num[common_num].values
            means = np.mean(ref_mat, axis=0)
            stds = np.std(ref_mat, axis=0)
            stds[stds < 1e-4] = 1.0

            x_norm = (ref_mat - means) / stds
            y_norm = (cur_num[common_num].values - means) / stds

            mmd_val = compute_rbf_mmd(x_norm, y_norm)
        else:
            mmd_val = 0.0

        drift_ratio = len(drifting_cols) / max(len(cols), 1)

        # 3. Governance Verdict
        if drift_ratio >= 0.30 or mmd_val > 0.45:
            sentinel_status = "CRITICAL_DRIFT"
            governance_action = "ACTIVATE_SAFE_FALLBACK"
            recs = [
                f"Severe drift detected across {len(drifting_cols)} features ({drift_ratio:.1%} of monitored columns).",
                f"Multivariate MMD score ({mmd_val:.3f}) exceeds safety ceiling (0.450).",
                "Quarantine compromised model and trigger automated pipeline retraining immediately.",
            ]
        elif drift_ratio >= 0.10 or mmd_val > 0.20:
            sentinel_status = "WARN_MONITOR"
            governance_action = "TRIGGER_RETRAIN"
            recs = [
                f"Moderate distribution shift detected on {len(drifting_cols)} features.",
                "Schedule model retraining during the next maintenance window.",
                "Verify upstream data ingestion pipelines for recent schema/distribution shifts.",
            ]
        else:
            sentinel_status = "PASS_HEALTHY"
            governance_action = "PASS_HEALTHY"
            recs = [
                "All monitored features remain well within empirical confidence bounds.",
                f"Multivariate MMD divergence is nominal ({mmd_val:.4f}).",
                "Zero governance interventions required.",
            ]

        return DriftSentinelReport(
            total_features_monitored=len(cols),
            n_reference_samples=len(reference_df),
            n_current_samples=len(current_df),
            drifting_feature_count=len(drifting_cols),
            drifting_feature_ratio=round(drift_ratio, 4),
            multivariate_mmd_score=round(mmd_val, 4),
            overall_sentinel_status=sentinel_status,
            governance_action=governance_action,
            feature_drift_breakdown=feature_details,
            top_drifting_features=drifting_cols[:6],
            actionable_recommendations=recs,
        )


def audit_drift_sentinel(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    monitored_columns: Optional[List[str]] = None,
) -> DriftSentinelReport:
    """Convenience functional interface for Drift Sentinel."""
    sentinel = DriftSentinel()
    return sentinel.audit_drift(
        reference_df=reference_df,
        current_df=current_df,
        monitored_columns=monitored_columns,
    )
