"""
dia/feature_engineer.py
───────────────────────
Automated Feature Engineering (AutoFE) Engine.
Discovers and constructs non-linear interaction terms, datetime signals,
and log transformations, then filters them using Mutual Information.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger("dia.feature_engineer")


def auto_engineer_features(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    max_new_features: int = 8,
    train_indices: pd.Index | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Automatically creates high-signal engineered features from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target variable name.
    task_type : str
        'classification' or 'regression'.
    max_new_features : int
        Maximum number of engineered features to retain.
    train_indices : pd.Index | None
        Optional train split indices. When provided, imputation medians and
        Mutual Information feature selection are strictly computed on these indices
        to eliminate data leakage into test sets.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Transformed DataFrame with selected engineered features, and a list of new feature names.
    """
    df_out = df.copy()
    candidate_features: dict[str, pd.Series] = {}
    date_cols_extracted: list[str] = []

    if train_indices is not None:
        if not isinstance(train_indices, pd.Index):
            train_indices = pd.Index(train_indices)
        train_indices = df_out.index.intersection(train_indices)
        if len(train_indices) == 0:
            train_indices = None

    # 1. Date/Time feature extraction
    for col in df_out.columns:
        if col == target_col:
            continue
        if pd.api.types.is_datetime64_any_dtype(df_out[col]) or "date" in col.lower() or "time" in col.lower():
            try:
                dt_series = pd.to_datetime(df_out[col], errors="coerce")
                if dt_series.notna().sum() > len(df_out) * 0.5:
                    candidate_features[f"fe_{col}_year"] = dt_series.dt.year.fillna(0)
                    candidate_features[f"fe_{col}_month"] = dt_series.dt.month.fillna(0)
                    candidate_features[f"fe_{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0)
                    candidate_features[f"fe_{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                    date_cols_extracted.append(col)
            except Exception as e:
                log.debug("Could not parse date column %s: %s", col, e)

    # 2. Numeric skewness & transformations
    num_cols = [
        c for c in df_out.select_dtypes(include=["number"]).columns
        if c != target_col and df_out[c].nunique() > 1
    ]

    for col in num_cols:
        series = df_out[col].dropna()
        if len(series) > 10 and series.min() >= 0:
            skew = series.skew()
            if abs(skew) > 1.5:
                candidate_features[f"fe_log_{col}"] = np.log1p(df_out[col].clip(lower=0))

    # 3. Numeric interaction & ratio terms between top correlated pairs
    if len(num_cols) >= 2:
        corr_matrix = df_out[num_cols].corr().abs()
        # Find top correlated feature pairs (to combine into ratios or products)
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val) and 0.1 < val < 0.95:
                    pairs.append((c1, c2, float(val)))

        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:4]

        for c1, c2, _ in top_pairs:
            # Interaction product
            candidate_features[f"fe_{c1}_x_{c2}"] = df_out[c1] * df_out[c2]
            # Safe ratio
            denominator = df_out[c2].replace(0, np.nan)
            if train_indices is not None:
                c1_median = df_out.loc[train_indices, c1].median()
            else:
                c1_median = df_out[c1].median()
            ratio = (df_out[c1] / denominator).fillna(c1_median).fillna(0.0)
            candidate_features[f"fe_ratio_{c1}_div_{c2}"] = ratio

    if date_cols_extracted:
        df_out.drop(columns=date_cols_extracted, inplace=True, errors="ignore")

    if not candidate_features:
        return df_out, []

    candidates_df = pd.DataFrame(candidate_features, index=df_out.index)
    candidates_df = candidates_df.replace([np.inf, -np.inf], np.nan)
    if train_indices is not None:
        impute_medians = candidates_df.loc[train_indices].median(numeric_only=True)
    else:
        impute_medians = candidates_df.median(numeric_only=True)
    candidates_df = candidates_df.fillna(impute_medians).fillna(0.0)
    candidates_df = candidates_df.clip(lower=-1e15, upper=1e15)

    # 4. Filter with Mutual Information against the target variable
    try:
        y_raw = df_out[target_col].dropna()
        valid_idx = y_raw.index
        if train_indices is not None:
            train_valid = valid_idx.intersection(train_indices)
            if len(train_valid) >= 2:
                valid_idx = train_valid

        X_candidates = candidates_df.loc[valid_idx]
        y_eval = y_raw.loc[valid_idx]

        if task_type == "classification":
            le = LabelEncoder()
            y_enc = le.fit_transform(y_eval.astype(str))
            mi_scores = mutual_info_classif(X_candidates, y_enc, random_state=42)
        else:
            y_num = pd.to_numeric(y_eval, errors="coerce").fillna(0)
            mi_scores = mutual_info_regression(X_candidates, y_num, random_state=42)

        scores_series = pd.Series(mi_scores, index=candidates_df.columns).sort_values(ascending=False)
        selected_cols = scores_series[scores_series > 0.001].head(max_new_features).index.tolist()

        if not selected_cols:
            selected_cols = scores_series.head(min(3, len(scores_series))).index.tolist()

        for c in selected_cols:
            df_out[c] = candidates_df[c]

        log.info("AutoFE generated %d new features: %s", len(selected_cols), selected_cols)
        return df_out, selected_cols

    except Exception as e:
        log.warning("AutoFE selection failed (%s). Adding top 3 raw candidates.", e)
        fallback_cols = list(candidates_df.columns)[:3]
        for c in fallback_cols:
            df_out[c] = candidates_df[c]
        return df_out, fallback_cols
