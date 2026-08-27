"""
dia/insights.py
───────────────
Generates automated smart insights based on data profiling and correlation with the target.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("dia.insights")


def generate_smart_insights(
    df: pd.DataFrame, target_col: str, task_type: str, max_insights: int = 4
) -> list[dict[str, str]]:
    """
    Generate natural language insights by finding correlations with the target.
    Returns a list of dicts: [{"title": str, "description": str, "type": "positive"|"negative"|"neutral"}]
    """
    insights = []

    # ── Handle Regression (Continuous Target) ─────────────────────────────────
    if task_type == "regression":
        try:
            target_series = pd.to_numeric(df[target_col], errors="coerce")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            correlations = []
            for col in numeric_cols:
                if col == target_col:
                    continue
                col_series = pd.to_numeric(df[col], errors="coerce")
                valid_idx = ~(target_series.isna() | col_series.isna())
                if valid_idx.sum() > 10:
                    corr = np.corrcoef(col_series[valid_idx], target_series[valid_idx])[0, 1]
                    if not np.isnan(corr):
                        correlations.append((col, corr))

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            for col, corr in correlations[:max_insights]:
                strength = "strong" if abs(corr) > 0.6 else ("moderate" if abs(corr) > 0.3 else "weak")
                direction = "positive" if corr > 0 else "negative"
                type_ = "positive" if corr > 0 else "negative"

                insights.append({
                    "title": f"{direction.title()} correlation with {col}",
                    "description": f"There is a {strength} {direction} correlation ({corr:.2f}) between `{col}` and the target `{target_col}`. As `{col}` increases, the target tends to {'increase' if corr > 0 else 'decrease'}.",
                    "type": type_
                })
        except Exception:
            log.debug("Could not compute regression-target insights.", exc_info=True)

    # ── Handle Classification (Categorical/Binary Target) ─────────────────────
    else:
        try:
            # For simplicity, if target is binary, treat as 0/1 to find correlation
            unique_targets = df[target_col].dropna().unique()
            if len(unique_targets) == 2:
                target_series = (df[target_col] == unique_targets[1]).astype(int)
                target_name = str(unique_targets[1])

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                correlations = []
                for col in numeric_cols:
                    if col == target_col:
                        continue
                    col_series = pd.to_numeric(df[col], errors="coerce")
                    valid_idx = ~col_series.isna()
                    if valid_idx.sum() > 10:
                        corr = np.corrcoef(col_series[valid_idx], target_series[valid_idx])[0, 1]
                        if not np.isnan(corr):
                            correlations.append((col, corr))

                correlations.sort(key=lambda x: abs(x[1]), reverse=True)

                for col, corr in correlations[:max_insights]:
                    strength = "strongly" if abs(corr) > 0.15 else "slightly"
                    direction = "higher" if corr > 0 else "lower"
                    type_ = "positive" if corr > 0 else "negative"

                    median_val = df[col].median()

                    insights.append({
                        "title": f"Impact of {col}",
                        "description": f"Records with {direction} `{col}` (relative to the median of {median_val:,.2f}) are {strength} more likely to have `{target_col}` = **{target_name}**. (Point-Biserial correlation: {corr:.2f})",
                        "type": type_
                    })
        except Exception:
            log.debug("Could not compute classification-target insights.", exc_info=True)

    # Fallback if no correlations found
    if not insights:
        insights.append({
            "title": "Complex Relationships",
            "description": "No strong linear correlations were found. The model is likely relying on complex, non-linear interactions between multiple features to make predictions.",
            "type": "neutral"
        })

    return insights
