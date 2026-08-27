"""
dia/drift_monitor.py
────────────────────
Production Data Drift & Anomaly Detection Engine.
Calculates Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI),
Wasserstein Distance, and Isolation Forest anomaly scores between baseline and production data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

log = logging.getLogger("dia.drift_monitor")


def calculate_psi(expected: pd.Series, actual: pd.Series, num_buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between expected (baseline) and actual (production).
    """
    try:
        if pd.api.types.is_numeric_dtype(expected):
            # Bin continuous data
            quantiles = np.linspace(0, 1, num_buckets + 1)
            bins = np.percentile(expected.dropna(), quantiles * 100)
            bins = np.unique(bins)
            if len(bins) < 2:
                return 0.0

            exp_counts = pd.cut(expected, bins=bins, include_lowest=True).value_counts(normalize=True)
            act_counts = pd.cut(actual, bins=bins, include_lowest=True).value_counts(normalize=True)
        else:
            exp_counts = expected.value_counts(normalize=True)
            act_counts = actual.value_counts(normalize=True)

        all_cats = list(set(exp_counts.index).union(set(act_counts.index)))
        psi_val = 0.0

        for cat in all_cats:
            e = exp_counts.get(cat, 0.0001) or 0.0001
            a = act_counts.get(cat, 0.0001) or 0.0001
            psi_val += (a - e) * np.log(a / e)

        return float(max(0.0, psi_val))
    except Exception as e:
        log.debug("PSI calculation failed: %s", e)
        return 0.0


def calculate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str | None = None,
) -> dict:
    """
    Analyze statistical drift between reference (training) and current (production/new batch) datasets.
    """
    column_reports: list[dict] = []
    drifted_count = 0

    shared_cols = [c for c in reference_df.columns if c in current_df.columns and c != target_col]

    for col in shared_cols:
        ref_s = reference_df[col].dropna()
        cur_s = current_df[col].dropna()

        if len(ref_s) < 5 or len(cur_s) < 5:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s)

        if is_numeric:
            # 1. Kolmogorov-Smirnov Test
            ks_stat, p_val = stats.ks_2samp(ref_s, cur_s)
            # 2. Wasserstein Distance
            w_dist = stats.wasserstein_distance(ref_s, cur_s)
            # 3. PSI
            psi_val = calculate_psi(ref_s, cur_s)

            is_drifted = bool(p_val < 0.05 or psi_val >= 0.2)
            if is_drifted:
                drifted_count += 1

            column_reports.append({
                "column": col,
                "type": "Numeric",
                "drift_detected": is_drifted,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 4),
                "psi": round(float(psi_val), 4),
                "wasserstein_distance": round(float(w_dist), 4),
                "ref_mean": round(float(ref_s.mean()), 2),
                "cur_mean": round(float(cur_s.mean()), 2),
                "ref_std": round(float(ref_s.std()), 2),
                "cur_std": round(float(cur_s.std()), 2),
            })

        else:
            # Categorical Drift via PSI
            psi_val = calculate_psi(ref_s, cur_s)
            is_drifted = bool(psi_val >= 0.2)
            if is_drifted:
                drifted_count += 1

            column_reports.append({
                "column": col,
                "type": "Categorical",
                "drift_detected": is_drifted,
                "ks_statistic": None,
                "p_value": None,
                "psi": round(float(psi_val), 4),
                "wasserstein_distance": None,
                "ref_unique": int(ref_s.nunique()),
                "cur_unique": int(cur_s.nunique()),
            })

    # Anomaly Detection using Isolation Forest on continuous features
    num_cols = reference_df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c in current_df.columns and c != target_col]
    anomaly_summary = {"anomalies_detected": 0, "anomaly_pct": 0.0}

    if len(num_cols) >= 2:
        try:
            ref_clean = reference_df[num_cols].fillna(reference_df[num_cols].median())
            cur_clean = current_df[num_cols].fillna(reference_df[num_cols].median())

            iso = IsolationForest(contamination=0.05, random_state=42)
            iso.fit(ref_clean)
            preds = iso.predict(cur_clean)
            anom_count = int(np.sum(preds == -1))
            anom_pct = round(float(anom_count / len(current_df) * 100), 2)

            anomaly_summary = {
                "anomalies_detected": anom_count,
                "anomaly_pct": anom_pct,
            }
        except Exception as e:
            log.warning("Isolation Forest anomaly scan failed: %s", e)

    overall_drift_pct = round(float((drifted_count / max(1, len(column_reports))) * 100), 1)

    return {
        "drift_detected": drifted_count > 0,
        "drifted_columns_count": drifted_count,
        "total_columns_evaluated": len(column_reports),
        "drift_percentage": overall_drift_pct,
        "column_reports": column_reports,
        "anomalies": anomaly_summary,
        "ref_row_count": len(reference_df),
        "cur_row_count": len(current_df),
    }
