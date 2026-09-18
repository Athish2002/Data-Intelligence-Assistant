"""
dia/leakage_detector.py
───────────────────────
Automated Target Leakage & Data Poisoning Sleuth.

Detects subtle, catastrophic failure modes in tabular datasets before model training:
1. Information-Theoretic Leakage: Mutual Information ratio I(X; Y) / H(Y).
2. Quasi-Target Proxies: Pearson, Spearman, Correlation Ratio (eta), and Cramer's V (|r| >= 0.95).
3. Temporal Chronology Violations: Feature timestamps occurring after target outcomes (t_feature > t_target).
4. High-Cardinality ID Memorization: Identifier columns risking decision tree memorization (unique ratio > 0.90).
5. Data Poisoning & Label Inversion: Duplicate feature records with contradictory target labels.

Quarantine Categorization:
- QUARANTINE_CRITICAL: Immediate removal recommended (MI >= 0.85, |r| >= 0.95, future timestamps).
- QUARANTINE_WARNING: Feature flagged for review (0.70 <= MI < 0.85, 0.85 <= |r| < 0.95, high-cardinality IDs).
- CLEAN: Safe for downstream feature engineering and model training.

Full defensive handling: NaNs, constant variance, single-class target, tiny N (< 5), empty DataFrames.
Strict resource hygiene: Explicit gc.collect(), bounded memory footprint.
"""

from __future__ import annotations

import gc
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder


# ─── Enumerations & Data Contracts ─────────────────────────────────────────────

class LeakageSeverity(str, Enum):
    """Quarantine severity levels for detected features."""
    CLEAN = "CLEAN"
    WARNING = "QUARANTINE_WARNING"
    CRITICAL = "QUARANTINE_CRITICAL"


class FeatureAssessment(BaseModel):
    """Detailed forensic evaluation of an individual candidate feature."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature: str
    risk_level: str  # "CLEAN", "QUARANTINE_WARNING", "QUARANTINE_CRITICAL"
    leakage_score: float = 0.0  # Normalized risk metric in [0.0, 1.0]
    metric_name: str = ""
    metric_value: float = 0.0
    reason: str = ""
    recommended_action: Literal["DROP", "REVIEW", "KEEP"] = "KEEP"


class PoisoningReport(BaseModel):
    """Forensic report on contradictory labels and malicious or corrupted label injections."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conflict_count: int = 0
    conflict_ratio: float = 0.0
    anomalous_label_clusters: int = 0
    suspicion_score: float = 0.0  # 0.0 to 100.0
    details: List[str] = Field(default_factory=list)


class LeakageReport(BaseModel):
    """
    Comprehensive target leakage and data integrity audit report.
    Compatible with both functional and object-oriented API contracts.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    has_leakage: bool = False
    quarantine_features: List[str] = Field(default_factory=list)
    mi_scores: Dict[str, float] = Field(default_factory=dict)
    correlation_scores: Dict[str, float] = Field(default_factory=dict)
    temporal_leakage_detected: bool = False
    id_memorization_features: List[str] = Field(default_factory=list)
    poisoning_detected: bool = False
    quarantine_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

    # Extended compatibility fields
    target_col: str = ""
    overall_leakage_risk_score: float = 0.0
    leakage_detected: bool = False
    quarantined_features: List[str] = Field(default_factory=list)
    clean_features: List[str] = Field(default_factory=list)
    feature_assessments: List[FeatureAssessment] = Field(default_factory=list)
    poisoning_report: PoisoningReport = Field(default_factory=PoisoningReport)


# Dataclass aliases for downstream consumers expecting dataclass results
LeakageAnalysisResult = LeakageReport


# ─── Heuristic Regex Patterns ─────────────────────────────────────────────────

_ID_PATTERN = re.compile(
    r"(?i)(^id$|_id$|^id_|^uuid$|_uuid$|^guid$|_guid$|^hash$|_hash$|^token$|_token$|"
    r"^key$|_key$|account_num|cust_id|customer_id|ssn|serial_no|record_id|client_id)"
)

_FUTURE_EVENT_PATTERN = re.compile(
    r"(?i)(future_|next_|post_|after_|churn_date|discharge_date|cancellation_date|post_event|termination_date)"
)


# ─── Mathematical Helpers ──────────────────────────────────────────────────────

def _compute_entropy(y_series: pd.Series) -> float:
    """
    Compute Shannon entropy H(Y) in nats for a discrete or categorical series.
    Returns 0.0 if series is empty or has zero variance (single class).
    """
    clean_y = y_series.dropna()
    if len(clean_y) <= 1:
        return 0.0
    
    counts = clean_y.value_counts(normalize=True).values
    if len(counts) <= 1:
        return 0.0
    
    # Shannon entropy: H(Y) = -sum(p * ln(p))
    entropy_val = -float(np.sum(counts * np.log(counts + 1e-12)))
    return max(0.0, entropy_val)


def _compute_discrete_mutual_info(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """
    Directly compute exact discrete mutual information I(X; Y) in nats from contingency counts.
    Mathematically rigorous and avoids KNN parameter failure on small N.
    """
    n = len(x_arr)
    if n <= 1:
        return 0.0

    # Build 2D contingency table using factorized indices
    df_temp = pd.DataFrame({"x": x_arr, "y": y_arr})
    contingency = pd.crosstab(df_temp["x"], df_temp["y"]).values
    del df_temp

    if contingency.size == 0:
        return 0.0

    p_xy = contingency / float(n)
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    # Non-zero entries only
    nz_mask = p_xy > 0
    if not np.any(nz_mask):
        return 0.0

    outer = p_x @ p_y
    mi = float(np.sum(p_xy[nz_mask] * np.log(p_xy[nz_mask] / (outer[nz_mask] + 1e-15) + 1e-15)))
    return max(0.0, mi)


def _compute_correlation_ratio(cat_x: pd.Series, num_y: pd.Series) -> float:
    """
    Calculate the correlation ratio eta for categorical X predicting numeric Y.
    eta = sqrt(SS_between / SS_total)
    """
    valid_mask = cat_x.notna() & num_y.notna()
    if valid_mask.sum() < 3:
        return 0.0

    x_clean = cat_x[valid_mask]
    y_clean = num_y[valid_mask].astype(float)

    y_mean = y_clean.mean()
    ss_total = np.sum((y_clean - y_mean) ** 2)
    if ss_total <= 1e-12:
        return 0.0

    # Group means and group sizes
    group_stats = y_clean.groupby(x_clean, observed=True).agg(["count", "mean"])
    ss_between = np.sum(group_stats["count"] * (group_stats["mean"] - y_mean) ** 2)

    eta = math.sqrt(max(0.0, min(1.0, ss_between / ss_total)))
    return float(eta)


def _compute_cramers_v(cat_x: pd.Series, cat_y: pd.Series) -> float:
    """
    Calculate Cramer's V association metric for two categorical variables.
    """
    valid_mask = cat_x.notna() & cat_y.notna()
    n = int(valid_mask.sum())
    if n < 3:
        return 0.0

    df_temp = pd.DataFrame({"x": cat_x[valid_mask], "y": cat_y[valid_mask]})
    contingency = pd.crosstab(df_temp["x"], df_temp["y"]).values
    del df_temp

    r, c = contingency.shape
    if r <= 1 or c <= 1:
        return 0.0

    try:
        chi2 = float(stats.chi2_contingency(contingency, correction=False)[0])
        denom = n * min(r - 1, c - 1)
        if denom <= 0:
            return 0.0
        v = math.sqrt(max(0.0, min(1.0, chi2 / denom)))
        return float(v)
    except Exception:
        return 0.0


# ─── Target Leakage Sleuth Class ──────────────────────────────────────────────

class TargetLeakageSleuth:
    """
    Information-Theoretic Target Leakage & Data Poisoning Sleuth.

    Analyzes tabular datasets for target contamination, proxy variables,
    future timestamp leaks, ID memorization risks, and adversarial data poisoning.
    """

    def __init__(
        self,
        mi_threshold_critical: float = 0.85,
        mi_threshold_warning: float = 0.70,
        corr_threshold_critical: float = 0.95,
        corr_threshold_warning: float = 0.85,
    ) -> None:
        self.mi_threshold_critical = float(mi_threshold_critical)
        self.mi_threshold_warning = float(mi_threshold_warning)
        self.corr_threshold_critical = float(corr_threshold_critical)
        self.corr_threshold_warning = float(corr_threshold_warning)

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str = "classification",
        date_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
    ) -> LeakageReport:
        """
        Executes exhaustive leakage, proxy, chronology, and poisoning checks.

        Args:
            df: Input pandas DataFrame to audit.
            target_col: Name of the target variable column.
            task_type: Either 'classification' or 'regression'.
            date_cols: Optional list of datetime columns.
            time_col: Optional reference timestamp column for temporal event ordering.

        Returns:
            LeakageReport containing granular forensic assessments and quarantine recommendations.
        """
        # 1. Defensive Guard: Empty or Missing Target
        if df is None or df.empty or len(df) == 0 or len(df.columns) == 0:
            return LeakageReport(
                has_leakage=False,
                target_col=target_col or "",
                details={"status": "EMPTY_DATAFRAME", "message": "DataFrame is empty."},
            )

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

        n_rows = len(df)
        target_series = df[target_col]

        # 2. Defensive Guard: Target column completely null
        if target_series.isna().all():
            return LeakageReport(
                has_leakage=False,
                target_col=target_col,
                details={"status": "ALL_NULL_TARGET", "message": f"Target column '{target_col}' is entirely NaN."},
            )

        # 3. Determine Effective Task Type & Target Properties
        clean_target = target_series.dropna()
        n_unique_target = clean_target.nunique()

        is_discrete_target = (
            task_type == "classification"
            or pd.api.types.is_object_dtype(clean_target)
            or pd.api.types.is_string_dtype(clean_target)
            or pd.api.types.is_bool_dtype(clean_target)
            or pd.api.types.is_categorical_dtype(clean_target)
            or n_unique_target <= max(5, int(0.05 * n_rows))
        )

        # Target entropy H(Y) in nats
        if is_discrete_target:
            h_y = _compute_entropy(clean_target)
            encoded_y = pd.factorize(clean_target)[0]
        else:
            # For continuous targets, discretize into equal-frequency quantiles to compute discrete entropy
            n_bins = min(10, max(2, n_rows // 5))
            try:
                binned_y = pd.qcut(clean_target, q=n_bins, duplicates="drop")
                h_y = _compute_entropy(binned_y)
                encoded_y = pd.factorize(binned_y)[0]
            except Exception:
                h_y = 1.0
                encoded_y = (clean_target.values > clean_target.median()).astype(int)

        # 4. Feature Columns to inspect
        feature_cols = [c for c in df.columns if c != target_col]
        if time_col and time_col in feature_cols:
            pass  # Keep time_col in feature inspection for correlation/MI, but also use for temporal order

        mi_scores: Dict[str, float] = {}
        corr_scores: Dict[str, float] = {}
        id_features: List[str] = []
        assessments: List[FeatureAssessment] = []
        quarantine_features: List[str] = []
        clean_features: List[str] = []
        recommendations: List[Dict[str, Any]] = []
        temporal_leakage_detected = False

        # 5. Temporal Chronology Violations Detection
        ref_time_col = time_col
        if not ref_time_col and date_cols:
            ref_time_col = date_cols[0]

        # Scan for date columns if not provided
        detected_date_cols: List[str] = []
        for col in feature_cols:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                detected_date_cols.append(col)
            elif "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
                try:
                    sample = df[col].dropna().head(10)
                    if not sample.empty and pd.to_datetime(sample, errors="coerce").notna().sum() >= len(sample) * 0.8:
                        detected_date_cols.append(col)
                except Exception:
                    pass

        # Check temporal order against reference timestamp
        if ref_time_col and ref_time_col in df.columns:
            try:
                t_ref = pd.to_datetime(df[ref_time_col], errors="coerce")
                for d_col in detected_date_cols:
                    if d_col == ref_time_col:
                        continue
                    t_feat = pd.to_datetime(df[d_col], errors="coerce")
                    valid_time_mask = t_feat.notna() & t_ref.notna()
                    if valid_time_mask.sum() >= 3:
                        future_count = (t_feat[valid_time_mask] > t_ref[valid_time_mask]).sum()
                        future_ratio = future_count / float(valid_time_mask.sum())
                        if future_ratio > 0.05:  # > 5% events in the future
                            temporal_leakage_detected = True
                            rec = {
                                "feature": d_col,
                                "risk_level": LeakageSeverity.CRITICAL.value,
                                "metric_name": "Temporal Event Inversion",
                                "metric_value": round(float(future_ratio), 4),
                                "reason": f"Temporal Chronology Violation: {future_ratio*100:.1f}% of records have {d_col} timestamp strictly occurring AFTER reference {ref_time_col}.",
                                "action": "DROP",
                            }
                            recommendations.append(rec)
                            assessments.append(FeatureAssessment(
                                feature=d_col,
                                risk_level=LeakageSeverity.CRITICAL.value,
                                leakage_score=round(float(future_ratio), 4),
                                metric_name="Temporal Event Inversion",
                                metric_value=round(float(future_ratio), 4),
                                reason=rec["reason"],
                                recommended_action="DROP",
                            ))
                            if d_col not in quarantine_features:
                                quarantine_features.append(d_col)
            except Exception:
                pass

        # Also check future naming regex
        for col in feature_cols:
            if _FUTURE_EVENT_PATTERN.search(col):
                temporal_leakage_detected = True
                if col not in quarantine_features:
                    rec = {
                        "feature": col,
                        "risk_level": LeakageSeverity.CRITICAL.value,
                        "metric_name": "Future Event Semantics",
                        "metric_value": 1.0,
                        "reason": f"Feature name '{col}' denotes a post-outcome or future event.",
                        "action": "DROP",
                    }
                    recommendations.append(rec)
                    assessments.append(FeatureAssessment(
                        feature=col,
                        risk_level=LeakageSeverity.CRITICAL.value,
                        leakage_score=1.0,
                        metric_name="Future Event Semantics",
                        metric_value=1.0,
                        reason=rec["reason"],
                        recommended_action="DROP",
                    ))
                    quarantine_features.append(col)

        # 6. Per-Feature Information-Theoretic & Correlation Analysis
        for col in feature_cols:
            feat_series = df[col]

            # A. Constant or All-Null Feature
            if feat_series.isna().all() or feat_series.nunique(dropna=True) <= 1:
                mi_scores[col] = 0.0
                corr_scores[col] = 0.0
                clean_features.append(col)
                continue

            # B. High-Cardinality ID Memorization Check
            u_ratio = feat_series.nunique(dropna=True) / float(max(1, n_rows))
            is_id_name = bool(_ID_PATTERN.search(col))
            is_high_card_id = False

            if is_id_name and u_ratio > 0.50:
                is_high_card_id = True
            elif u_ratio > 0.90 and n_rows >= 10:
                # If unique ratio > 0.90 on non-float or integer/string
                if pd.api.types.is_object_dtype(feat_series) or pd.api.types.is_string_dtype(feat_series) or pd.api.types.is_integer_dtype(feat_series):
                    is_high_card_id = True

            if is_high_card_id:
                id_features.append(col)

            # C. Mutual Information Ratio: I(X; Y) / H(Y)
            mi_ratio = 0.0
            if h_y > 1e-7 and n_rows >= 3:
                try:
                    # Align valid indices
                    valid_feat_mask = feat_series.notna() & target_series.notna()
                    if valid_feat_mask.sum() >= 3:
                        x_sub = feat_series[valid_feat_mask]
                        y_sub = encoded_y[valid_feat_mask.values]

                        is_num_feat = pd.api.types.is_numeric_dtype(x_sub) and not pd.api.types.is_bool_dtype(x_sub)

                        if not is_num_feat:
                            # Discrete X, Discrete Y: Compute exact discrete mutual information
                            x_fac = pd.factorize(x_sub)[0]
                            raw_mi = _compute_discrete_mutual_info(x_fac, y_sub)
                        else:
                            # Continuous X, Discrete Y: Use mutual_info_classif with bounded neighbors
                            x_vals = x_sub.values.astype(float)
                            finite_mask = np.isfinite(x_vals)
                            if finite_mask.sum() >= 4:
                                x_mat = x_vals[finite_mask].reshape(-1, 1)
                                y_mat = y_sub[finite_mask]
                                k_neighbors = min(3, max(1, len(x_mat) - 1))
                                mi_res = mutual_info_classif(x_mat, y_mat, n_neighbors=k_neighbors, random_state=42)
                                raw_mi = float(mi_res[0])
                            else:
                                # Fallback to binned discrete MI
                                x_fac = pd.qcut(x_vals, q=min(5, len(x_vals)), duplicates="drop").codes
                                raw_mi = _compute_discrete_mutual_info(x_fac, y_sub)

                        # Normalize by target entropy
                        mi_ratio = max(0.0, min(1.0, raw_mi / h_y))
                except Exception:
                    mi_ratio = 0.0

            mi_scores[col] = round(float(mi_ratio), 4)

            # D. Correlation Proxy Check
            corr_val = 0.0
            try:
                valid_mask = feat_series.notna() & target_series.notna()
                if valid_mask.sum() >= 3:
                    x_sub = feat_series[valid_mask]
                    y_sub = target_series[valid_mask]

                    is_x_num = pd.api.types.is_numeric_dtype(x_sub) and not pd.api.types.is_bool_dtype(x_sub)
                    is_y_num = pd.api.types.is_numeric_dtype(y_sub) and not pd.api.types.is_bool_dtype(y_sub)

                    if is_x_num and is_y_num:
                        # Pearson & Spearman
                        p_val = stats.pearsonr(x_sub.values.astype(float), y_sub.values.astype(float))[0]
                        s_val = stats.spearmanr(x_sub.values.astype(float), y_sub.values.astype(float))[0]
                        p_clean = abs(p_val) if np.isfinite(p_val) else 0.0
                        s_clean = abs(s_val) if np.isfinite(s_val) else 0.0
                        corr_val = max(p_clean, s_clean)
                    elif (not is_x_num) and is_y_num:
                        # Categorical X, Numeric Y: Correlation Ratio eta
                        corr_val = _compute_correlation_ratio(x_sub, y_sub)
                    elif is_x_num and (not is_y_num):
                        # Numeric X, Categorical Y: Correlation Ratio eta of Y on X
                        corr_val = _compute_correlation_ratio(y_sub, x_sub)
                    else:
                        # Categorical X, Categorical Y: Cramer's V
                        corr_val = _compute_cramers_v(x_sub, y_sub)
            except Exception:
                corr_val = 0.0

            corr_scores[col] = round(float(corr_val), 4)

            # E. Severity Assessment & Quarantine Assignment
            # Check Critical
            is_critical = False
            is_warning = False
            crit_reasons: List[str] = []
            warn_reasons: List[str] = []

            if corr_val >= self.corr_threshold_critical:
                is_critical = True
                crit_reasons.append(f"Quasi-target proxy detected: correlation = {corr_val:.4f} >= {self.corr_threshold_critical}")
            elif corr_val >= self.corr_threshold_warning:
                is_warning = True
                warn_reasons.append(f"High correlation with target: {corr_val:.4f} >= {self.corr_threshold_warning}")

            if mi_ratio >= self.mi_threshold_critical:
                is_critical = True
                crit_reasons.append(f"Target leakage: Normalized Mutual Information I(X;Y)/H(Y) = {mi_ratio:.4f} >= {self.mi_threshold_critical}")
            elif mi_ratio >= self.mi_threshold_warning:
                is_warning = True
                warn_reasons.append(f"Elevated Mutual Information I(X;Y)/H(Y) = {mi_ratio:.4f} >= {self.mi_threshold_warning}")

            if is_high_card_id:
                if u_ratio >= 0.95 and is_id_name:
                    is_warning = True
                    warn_reasons.append(f"Identifier memorization: unique ratio = {u_ratio*100:.1f}% with ID naming pattern.")
                else:
                    is_warning = True
                    warn_reasons.append(f"High-cardinality ID risk: unique ratio = {u_ratio*100:.1f}%.")

            # Finalize Triage for this feature
            if is_critical:
                severity = LeakageSeverity.CRITICAL.value
                action = "DROP"
                reason_str = " | ".join(crit_reasons)
                if col not in quarantine_features:
                    quarantine_features.append(col)
            elif is_warning:
                severity = LeakageSeverity.WARNING.value
                action = "REVIEW"
                reason_str = " | ".join(warn_reasons)
                if col not in quarantine_features:
                    quarantine_features.append(col)
            else:
                severity = LeakageSeverity.CLEAN.value
                action = "KEEP"
                reason_str = "Feature shows no target leakage or memorization risks."
                clean_features.append(col)

            # Compute normalized leakage score for this feature
            feature_leakage_score = max(mi_ratio, corr_val, 0.75 if is_high_card_id else 0.0)

            assess = FeatureAssessment(
                feature=col,
                risk_level=severity,
                leakage_score=round(float(feature_leakage_score), 4),
                metric_name="Combined Leakage Score",
                metric_value=round(float(max(mi_ratio, corr_val)), 4),
                reason=reason_str,
                recommended_action=action,
            )
            assessments.append(assess)

            if severity != LeakageSeverity.CLEAN.value:
                recommendations.append({
                    "feature": col,
                    "risk_level": severity,
                    "metric_name": "Combined Leakage Score",
                    "metric_value": assess.metric_value,
                    "reason": reason_str,
                    "action": action,
                })

        # 7. Data Poisoning & Duplicate Contradictory Label Sleuth
        poisoning_detected = False
        conflict_count = 0
        conflict_ratio = 0.0
        suspicion_score = 0.0
        poisoning_details: List[str] = []

        if len(feature_cols) > 0 and n_rows >= 2:
            try:
                # Find identical feature vectors with differing labels
                dup_mask = df.duplicated(subset=feature_cols, keep=False)
                if dup_mask.any():
                    dup_df = df[dup_mask]
                    # Group by all feature columns and count distinct target labels
                    grouped = dup_df.groupby(feature_cols, observed=True)[target_col].nunique()
                    conflicts = grouped[grouped > 1]
                    conflict_count = int(len(conflicts))

                    if conflict_count > 0:
                        poisoning_detected = True
                        # Total conflicting rows involved
                        conflicting_rows = int(grouped[grouped > 1].sum())
                        conflict_ratio = float(min(1.0, conflicting_rows / float(n_rows)))
                        suspicion_score = float(min(100.0, conflict_ratio * 400.0 + conflict_count * 10.0))
                        poisoning_details.append(
                            f"Detected {conflict_count} identical feature clusters with contradictory target labels ({conflicting_rows} rows affected)."
                        )
                        # Sample details
                        poisoning_details.append(
                            "Potential malicious label flip injection or corrupted logging pipeline."
                        )
            except Exception as e:
                # Incase grouping on certain complex types raises, fall back gracefully
                pass

        poisoning_rep = PoisoningReport(
            conflict_count=conflict_count,
            conflict_ratio=round(conflict_ratio, 4),
            anomalous_label_clusters=conflict_count,
            suspicion_score=round(suspicion_score, 2),
            details=poisoning_details,
        )

        # 8. Overall Leakage Risk Score Calculation (0.0 to 100.0)
        n_crit = sum(1 for a in assessments if a.risk_level == LeakageSeverity.CRITICAL.value)
        n_warn = sum(1 for a in assessments if a.risk_level == LeakageSeverity.WARNING.value)

        risk_score = 0.0
        risk_score += n_crit * 35.0
        risk_score += n_warn * 15.0
        if temporal_leakage_detected:
            risk_score += 25.0
        if poisoning_detected:
            risk_score += min(30.0, suspicion_score * 0.3)

        overall_risk_score = min(100.0, round(risk_score, 1))
        has_leakage = (n_crit > 0 or n_warn > 0 or temporal_leakage_detected or poisoning_detected)

        # Build final report
        report = LeakageReport(
            has_leakage=has_leakage,
            quarantine_features=quarantine_features,
            mi_scores=mi_scores,
            correlation_scores=corr_scores,
            temporal_leakage_detected=temporal_leakage_detected,
            id_memorization_features=id_features,
            poisoning_detected=poisoning_detected,
            quarantine_recommendations=recommendations,
            details={
                "task_type": "classification" if is_discrete_target else "regression",
                "target_entropy_nats": round(float(h_y), 4),
                "total_features_evaluated": len(feature_cols),
                "critical_quarantine_count": n_crit,
                "warning_quarantine_count": n_warn,
                "clean_features_count": len(clean_features),
            },
            target_col=target_col,
            overall_leakage_risk_score=overall_risk_score,
            leakage_detected=has_leakage,
            quarantined_features=quarantine_features,
            clean_features=clean_features,
            feature_assessments=assessments,
            poisoning_report=poisoning_rep,
        )

        # Proactive memory cleanup
        gc.collect()

        return report

    def get_quarantined_dataframe(
        self,
        df: pd.DataFrame,
        analysis: LeakageReport,
        drop_critical_only: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a sanitized DataFrame with quarantined features removed.

        Args:
            df: Original pandas DataFrame.
            analysis: LeakageReport generated by analyze().
            drop_critical_only: If True, only drops QUARANTINE_CRITICAL features;
                                otherwise drops both CRITICAL and WARNING features.

        Returns:
            Sanitized DataFrame without quarantined columns.
        """
        if df is None or df.empty:
            return df

        if drop_critical_only:
            cols_to_drop = [
                a.feature for a in analysis.feature_assessments
                if a.risk_level == LeakageSeverity.CRITICAL.value and a.feature in df.columns
            ]
        else:
            cols_to_drop = [c for c in analysis.quarantine_features if c in df.columns]

        if not cols_to_drop:
            return df.copy(deep=False)

        sanitized_df = df.drop(columns=cols_to_drop)
        gc.collect()
        return sanitized_df


# ─── Functional API ───────────────────────────────────────────────────────────

def detect_leakage(
    df: pd.DataFrame,
    target_col: str,
    time_col: Optional[str] = None,
    task_type: str = "classification",
) -> LeakageReport:
    """
    Standard interface function to detect target leakage and data poisoning.

    Args:
        df: Input pandas DataFrame.
        target_col: Name of the target column.
        time_col: Optional reference timestamp column.
        task_type: Either 'classification' or 'regression'.

    Returns:
        LeakageReport with comprehensive leakage analysis.
    """
    sleuth = TargetLeakageSleuth()
    return sleuth.analyze(df=df, target_col=target_col, task_type=task_type, time_col=time_col)


def quarantine_features(
    df: pd.DataFrame,
    report: LeakageReport,
    drop_critical_only: bool = False,
) -> pd.DataFrame:
    """
    Quarantines (drops) high-risk leakage features from the dataset.

    Args:
        df: Original DataFrame.
        report: LeakageReport object from detect_leakage.
        drop_critical_only: If True, only drops CRITICAL features.

    Returns:
        Cleaned pandas DataFrame.
    """
    sleuth = TargetLeakageSleuth()
    return sleuth.get_quarantined_dataframe(df=df, analysis=report, drop_critical_only=drop_critical_only)
