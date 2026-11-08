"""
dia/chat_analyst.py
───────────────────
Autonomous Conversational Data Copilot.
Parses natural language analytical queries into structured analytical intents,
executes sandboxed vectorized calculations with statistical confidence measures,
and generates rich text summaries, tables, and interactive Plotly visualizations.
Works completely standalone (offline deterministic mode) with optional LLM reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import LLM_MAX_TOOL_ITERATIONS
from .retrieval import SimpleVectorIndex
from .validators import detect_pii_columns

log = logging.getLogger("dia.chat_analyst")


class AnalyticalIntent(str, Enum):
    SEGMENT_COMPARISON = "segment_comparison"
    OUTLIER_INVESTIGATION = "outlier_investigation"
    TOP_K_CONTRIBUTORS = "top_k_contributors"
    BIVARIATE_RELATIONSHIP = "bivariate_relationship"
    DISTRIBUTION_PROFILE = "distribution_profile"
    GENERAL_PROFILE = "general_profile"


# ─── Robust Column & Entity Extraction ───────────────────────────────────────

def _find_matching_columns(query: str, columns: list[str]) -> list[str]:
    """
    Find matching column names from query tokens.
    Prioritizes longer and exact matches to prevent substring shadowing.
    """
    q_norm = query.lower()
    q_tokens = set(re.findall(r"\w+", q_norm))

    matches: list[tuple[int, int, str]] = []  # (priority_score, length, col_name)

    # Sort columns by length descending to match composite names first
    sorted_cols = sorted(columns, key=lambda c: len(c), reverse=True)

    for col in sorted_cols:
        col_lower = col.lower()
        col_clean = col_lower.replace("_", " ")

        # Direct exact occurrence of column name or space-separated variant
        if col_lower in q_norm or col_clean in q_norm:
            score = 3
        else:
            col_words = set(re.findall(r"\w+", col_lower))
            overlap = len(col_words & q_tokens)
            if overlap == len(col_words) and len(col_words) > 0:
                score = 2
            elif overlap > 0 and len(col_words) > 1:
                score = 1
            else:
                score = 0

        if score > 0:
            matches.append((score, len(col), col))

    # Sort by score descending, then length descending
    matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [col for _, _, col in matches]


def _find_matching_column(query: str, columns: list[str]) -> str | None:
    """Find the single best matching column name from query tokens."""
    matches = _find_matching_columns(query, columns)
    return matches[0] if matches else None


def _classify_intent(
    query: str,
    matched_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
) -> AnalyticalIntent:
    """Classify user query into a primary analytical intent type."""
    q = query.strip().lower()

    # Outlier investigation keywords
    outlier_keywords = ["outlier", "outliers", "anomaly", "anomalies", "anomalous", "extreme", "extremes", "unusual", "tukey", "z-score", "heavy tail"]
    if any(k in q for k in outlier_keywords):
        return AnalyticalIntent.OUTLIER_INVESTIGATION

    # Bivariate keywords or correlation matrix
    bivariate_keywords = ["correlation", "correlate", "relationship", " vs ", "versus", "scatter", "covariance", "association", "bivariate"]
    if any(k in q for k in bivariate_keywords):
        return AnalyticalIntent.BIVARIATE_RELATIONSHIP

    # Top-K / Leaderboard keywords
    top_k_keywords = ["top ", "bottom ", "leaders", "ranking", "rank", "contributors", "pareto", "biggest ", "smallest ", "highest ", "lowest "]
    has_top_k_kw = any(k in q for k in top_k_keywords)
    if has_top_k_kw:
        # Check if ranking categories
        matched_cat = [c for c in matched_cols if c in cat_cols]
        if matched_cat or any(k in q for k in ["category", "segment", "class", "group"]):
            return AnalyticalIntent.TOP_K_CONTRIBUTORS

    # Segment comparison / Groupby keywords
    segment_keywords = ["group by", "grouped by", "segment by", "segmented by", "broken down by", "breakdown by", "compare", "comparison", "across", " per ", " by "]
    has_segment_kw = any(k in q for k in segment_keywords)
    matched_cat = [c for c in matched_cols if c in cat_cols]
    matched_num = [c for c in matched_cols if c in num_cols]

    if has_segment_kw and matched_cat and matched_num:
        return AnalyticalIntent.SEGMENT_COMPARISON
    if any(k in q for k in ["group by", "grouped by", "broken down by"]):
        return AnalyticalIntent.SEGMENT_COMPARISON

    # Distribution & Summary keywords
    dist_keywords = ["distribution", "histogram", "spread", "density", "skew", "skewness", "average", "mean", "median", "summary", "describe", "stats", "statistics", "std", "min", "max"]
    if any(k in q for k in dist_keywords) or matched_cols:
        return AnalyticalIntent.DISTRIBUTION_PROFILE

    return AnalyticalIntent.GENERAL_PROFILE


# ─── Analytical Executors ───────────────────────────────────────────────────

def _execute_segment_comparison(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
) -> dict[str, Any]:
    """Execute segmented aggregation with Standard Errors, IQR, and statistical hypothesis tests."""
    sub_df = df[[group_col, metric_col]].dropna()
    sub_df[metric_col] = pd.to_numeric(sub_df[metric_col], errors="coerce")
    sub_df = sub_df.dropna()

    if sub_df.empty:
        return {
            "text": f"No valid records found for grouping `{metric_col}` by `{group_col}`.",
            "table": None,
            "figure": None,
        }

    # Vectorized statistical aggregation
    agg_df = sub_df.groupby(group_col)[metric_col].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Median="median",
        Q25=lambda s: s.quantile(0.25),
        Q75=lambda s: s.quantile(0.75),
    ).reset_index()

    agg_df["Std"] = agg_df["Std"].fillna(0.0)
    agg_df["SEM"] = agg_df["Std"] / np.sqrt(np.maximum(1, agg_df["Count"]))
    agg_df["IQR"] = agg_df["Q75"] - agg_df["Q25"]
    agg_df["CI95_Lower"] = (agg_df["Mean"] - 1.96 * agg_df["SEM"]).round(2)
    agg_df["CI95_Upper"] = (agg_df["Mean"] + 1.96 * agg_df["SEM"]).round(2)
    agg_df = agg_df.sort_values(by="Mean", ascending=False).reset_index(drop=True)

    top_row = agg_df.iloc[0]
    bot_row = agg_df.iloc[-1]
    n_groups = len(agg_df)

    # Statistical significance testing
    group_samples = [group_data[metric_col].values for _, group_data in sub_df.groupby(group_col) if len(group_data) >= 2]
    stat_msg = ""

    if len(group_samples) == 2:
        # Welch's t-test and Cohen's d
        t_stat, p_val = stats.ttest_ind(group_samples[0], group_samples[1], equal_var=False)
        s_pooled = np.sqrt((np.var(group_samples[0], ddof=1) + np.var(group_samples[1], ddof=1)) / 2.0 + 1e-12)
        cohen_d = abs(np.mean(group_samples[0]) - np.mean(group_samples[1])) / s_pooled
        sig_str = "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant (p >= 0.05)"
        stat_msg = f"Welch's t-test: t = {t_stat:.2f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.2f} ({sig_str})."
    elif len(group_samples) >= 3:
        # One-way ANOVA
        try:
            f_stat, p_val = stats.f_oneway(*group_samples)
            sig_str = "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant (p >= 0.05)"
            stat_msg = f"One-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f} ({sig_str})."
        except Exception:
            stat_msg = ""

    text = (
        f"**Segment Comparison: `{metric_col}` by `{group_col}` ({n_groups} segments, N = {len(sub_df):,}):**\n"
        f"- **Top Segment:** **`{top_row[group_col]}`** with Mean = **{top_row['Mean']:,.2f}** ± {top_row['SEM']:,.2f} "
        f"(95% CI: [{top_row['CI95_Lower']:,.2f}, {top_row['CI95_Upper']:,.2f}], Median = {top_row['Median']:,.2f}, IQR = {top_row['IQR']:,.2f}).\n"
        f"- **Lowest Segment:** **`{bot_row[group_col]}`** with Mean = **{bot_row['Mean']:,.2f}** ± {bot_row['SEM']:,.2f} "
        f"(Median = {bot_row['Median']:,.2f}).\n"
    )
    if stat_msg:
        text += f"- **Hypothesis Test:** {stat_msg}\n"

    table_df = agg_df[[group_col, "Count", "Mean", "SEM", "CI95_Lower", "CI95_Upper", "Median", "IQR"]].round(2)

    figure = None
    try:
        import plotly.express as px
        figure = px.bar(
            agg_df.head(15),
            x=group_col,
            y="Mean",
            error_y="SEM",
            color="Mean",
            title=f"Mean {metric_col} by {group_col} (with ±SEM Error Bars)",
            labels={"Mean": f"Mean {metric_col}", group_col: str(group_col)},
            color_continuous_scale="Blues",
        )
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
    except Exception:
        log.debug("Plotly render error in segment comparison", exc_info=True)

    return {"text": text, "table": table_df, "figure": figure}


def _execute_outlier_investigation(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Execute vectorized Tukey IQR & Z-score outlier detection."""
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = len(s)
    if n < 5:
        return {"text": f"Insufficient numeric observations in `{col}` for outlier analysis.", "table": None, "figure": None}

    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))
    iqr = float(q75 - q25)
    mean_val = float(s.mean())
    std_val = float(s.std(ddof=1)) if s.std(ddof=1) > 1e-8 else 1.0

    lower_fence = q25 - 1.5 * iqr
    upper_fence = q75 + 1.5 * iqr
    extreme_lower = q25 - 3.0 * iqr
    extreme_upper = q75 + 3.0 * iqr

    outliers_mask = (s < lower_fence) | (s > upper_fence)
    n_outliers = int(outliers_mask.sum())
    pct_outliers = (n_outliers / n) * 100.0

    extreme_mask = (s < extreme_lower) | (s > extreme_upper)
    n_extreme = int(extreme_mask.sum())

    z_scores = (s - mean_val) / std_val
    z_anomalies = int((z_scores.abs() > 3.0).sum())

    skewness = float(stats.skew(s))
    skew_desc = "heavily right-skewed" if skewness > 1.0 else ("heavily left-skewed" if skewness < -1.0 else ("moderately skewed" if abs(skewness) > 0.5 else "approximately symmetric"))

    text = (
        f"**Outlier & Anomaly Investigation for `{col}` (N = {n:,}):**\n"
        f"- **Tukey IQR Fences [1.5×IQR]:** Lower = **{lower_fence:,.2f}**, Upper = **{upper_fence:,.2f}** (IQR = {iqr:,.2f}).\n"
        f"- **Detected Outliers:** **{n_outliers:,}** records (**{pct_outliers:.2f}%** of sample).\n"
        f"- **Severe Anomalies [3.0×IQR]:** {n_extreme:,} extreme outliers; **{z_anomalies:,}** records with |z-score| > 3.0.\n"
        f"- **Distribution Shape:** Skewness = **{skewness:.2f}** ({skew_desc})."
    )

    # Top extreme sample table
    outlier_records = df.loc[s[outliers_mask].index].copy()
    outlier_records["_z_score"] = z_scores[outliers_mask].round(2)
    outlier_records = outlier_records.sort_values(by="_z_score", key=abs, ascending=False).head(10)
    outlier_records.rename(columns={"_z_score": "Z-Score"}, inplace=True)

    figure = None
    try:
        import plotly.express as px
        figure = px.box(
            df.sample(min(n, 1000), random_state=42),
            y=col,
            points="outliers",
            title=f"Boxplot & Outlier Spread for {col}",
        )
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
    except Exception:
        log.debug("Plotly render error in outlier investigation", exc_info=True)

    return {"text": text, "table": outlier_records if not outlier_records.empty else None, "figure": figure}


def _execute_top_k_contributors(
    df: pd.DataFrame,
    dim_col: str,
    metric_col: str,
    k: int = 5,
    ascending: bool = False,
) -> dict[str, Any]:
    """Execute Top-K Pareto contribution analysis."""
    sub_df = df[[dim_col, metric_col]].dropna()
    sub_df[metric_col] = pd.to_numeric(sub_df[metric_col], errors="coerce")
    sub_df = sub_df.dropna()

    if sub_df.empty:
        return {"text": f"No valid data for ranking `{dim_col}` by `{metric_col}`.", "table": None, "figure": None}

    grouped = sub_df.groupby(dim_col)[metric_col].sum().reset_index()
    total_val = float(grouped[metric_col].sum())
    grouped["Share_Pct"] = (grouped[metric_col] / (total_val + 1e-12)) * 100.0

    grouped = grouped.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)
    grouped["Cumulative_Pct"] = grouped["Share_Pct"].cumsum().round(2)
    grouped["Rank"] = range(1, len(grouped) + 1)

    top_slice = grouped.head(k)
    cum_share = float(top_slice["Share_Pct"].sum())
    leader_name = str(top_slice.iloc[0][dim_col])
    leader_val = float(top_slice.iloc[0][metric_col])
    leader_share = float(top_slice.iloc[0]["Share_Pct"])

    direction_str = "Lowest" if ascending else "Top"
    text = (
        f"**{direction_str} {k} Contributors: `{dim_col}` ranked by `{metric_col}` (Total: {total_val:,.2f}):**\n"
        f"- The top {len(top_slice)} segments account for **{cum_share:.1f}%** of total `{metric_col}`.\n"
        f"- **#1 Leader:** **`{leader_name}`** with **{leader_val:,.2f}** ({leader_share:.1f}% share).\n"
    )

    table_df = top_slice[["Rank", dim_col, metric_col, "Share_Pct", "Cumulative_Pct"]].rename(
        columns={"Share_Pct": "Share (%)", "Cumulative_Pct": "Cumulative Share (%)"}
    ).round(2)

    figure = None
    try:
        import plotly.express as px
        figure = px.bar(
            top_slice,
            x=dim_col,
            y=metric_col,
            color="Share (%)",
            title=f"{direction_str} {k} Categories in {dim_col} by {metric_col}",
            text_auto=".2s",
            color_continuous_scale="Viridis",
        )
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
    except Exception:
        log.debug("Plotly render error in top-k contributors", exc_info=True)

    return {"text": text, "table": table_df, "figure": figure}


def _execute_bivariate_relationship(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Execute bivariate relationship analysis with Pearson CI, Spearman rank, and OLS fit."""
    sub_df = df[[col1, col2]].dropna()
    x = pd.to_numeric(sub_df[col1], errors="coerce")
    y = pd.to_numeric(sub_df[col2], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid].to_numpy(dtype=float)
    y = y[valid].to_numpy(dtype=float)

    n = len(x)
    if n < 5:
        return {"text": f"Insufficient matching numeric data between `{col1}` and `{col2}`.", "table": None, "figure": None}

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    # 95% Confidence Interval for Pearson r
    z = np.arctanh(np.clip(r_pearson, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(max(1, n - 3))
    ci_l = float(np.tanh(z - 1.96 * se_z))
    ci_u = float(np.tanh(z + 1.96 * se_z))

    # OLS parameters
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    slope = float(r_pearson * (std_y / (std_x + 1e-12)))
    intercept = float(np.mean(y) - slope * np.mean(x))
    r2 = float(r_pearson ** 2)

    strength = "strong" if abs(r_pearson) >= 0.6 else ("moderate" if abs(r_pearson) >= 0.3 else "weak")
    direction = "positive" if r_pearson > 0 else "negative"

    text = (
        f"**Bivariate Analysis: `{col1}` vs `{col2}` (N = {n:,}):**\n"
        f"- **Pearson Correlation:** **r = {r_pearson:+.4f}** ({strength} {direction}, 95% CI: [{ci_l:+.3f}, {ci_u:+.3f}], p = {p_pearson:.4e}).\n"
        f"- **Spearman Rank Correlation:** **ρ = {r_spearman:+.4f}** (monotonic alignment, p = {p_spearman:.4e}).\n"
        f"- **OLS Fit:** `{col2}` = {slope:+.4f} × `{col1}` + {intercept:+.2f} (explains **{r2 * 100:.1f}%** of variance).\n"
    )

    figure = None
    try:
        import plotly.express as px
        sample_df = df[[col1, col2] + ([target_col] if target_col and target_col in df.columns else [])].dropna()
        sample_df = sample_df.sample(min(len(sample_df), 500), random_state=42)
        figure = px.scatter(
            sample_df,
            x=col1,
            y=col2,
            color=target_col if target_col and target_col in sample_df.columns else None,
            trendline="ols",
            title=f"{col1} vs {col2} (OLS Trendline)",
            opacity=0.7,
        )
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
    except Exception:
        log.debug("Plotly render error in bivariate relationship", exc_info=True)

    return {"text": text, "table": None, "figure": figure}


def _execute_distribution_profile(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Execute distribution profiling with central tendency, IQR, skewness, and normality."""
    s = df[col].dropna()

    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        n = len(s_num)
        if n == 0:
            return {"text": f"Column `{col}` contains no valid numeric entries.", "table": None, "figure": None}

        mean_val = float(s_num.mean())
        std_val = float(s_num.std(ddof=1)) if n > 1 else 0.0
        sem_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
        median_val = float(s_num.median())
        q25 = float(s_num.quantile(0.25))
        q75 = float(s_num.quantile(0.75))
        iqr_val = float(q75 - q25)
        min_val = float(s_num.min())
        max_val = float(s_num.max())

        skewness = float(stats.skew(s_num)) if n >= 3 else 0.0
        skew_desc = "right-skewed" if skewness > 0.5 else ("left-skewed" if skewness < -0.5 else "symmetric")

        norm_msg = ""
        if n >= 20:
            try:
                _, p_norm = stats.normaltest(s_num)
                norm_desc = "normally distributed" if p_norm > 0.05 else "non-normal"
                norm_msg = f"Normality test p = {p_norm:.4f} ({norm_desc})."
            except Exception:
                norm_msg = ""

        text = (
            f"📊 **Distribution Profile for `{col}` (N = {n:,}):**\n"
            f"- **Central Tendency:** Mean = **{mean_val:,.2f}** ± {sem_val:,.2f} (95% CI: [{mean_val - 1.96 * sem_val:,.2f}, {mean_val + 1.96 * sem_val:,.2f}]), Median = **{median_val:,.2f}**.\n"
            f"- **Dispersion:** Std Dev = **{std_val:,.2f}**, IQR = **{iqr_val:,.2f}** (Q1 = {q25:,.2f}, Q3 = {q75:,.2f}).\n"
            f"- **Range:** Min = **{min_val:,.2f}**, Max = **{max_val:,.2f}**.\n"
            f"- **Shape:** Skewness = **{skewness:.2f}** ({skew_desc}). {norm_msg}"
        )

        figure = None
        try:
            import plotly.express as px
            figure = px.histogram(
                df,
                x=col,
                marginal="box",
                title=f"Distribution of {col}",
                color_discrete_sequence=["#6366f1"],
            )
            figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
        except Exception:
            log.debug("Plotly render error in distribution profile", exc_info=True)

        return {"text": text, "table": None, "figure": figure}
    else:
        counts = s.value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["Percentage (%)"] = ((counts["Count"] / len(s)) * 100.0).round(2)
        top_cat = counts.iloc[0][col]
        top_count = counts.iloc[0]["Count"]
        top_pct = counts.iloc[0]["Percentage (%)"]

        text = (
            f"**Categorical Profile for `{col}` ({len(counts)} unique categories, N = {len(s):,}):**\n"
            f"- **Top Mode Category:** **`{top_cat}`** with **{top_count:,}** occurrences (**{top_pct:.1f}%** share).\n"
        )

        figure = None
        try:
            import plotly.express as px
            figure = px.bar(counts.head(10), x=col, y="Count", title=f"Top Categories in {col}", color="Count")
            figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
        except Exception:
            log.debug("Plotly render error in categorical profile", exc_info=True)

        return {"text": text, "table": counts.head(10), "figure": figure}


# ─── Master Dispatch: execute_natural_language_query ─────────────────────────

def execute_natural_language_query(
    df: pd.DataFrame,
    query: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    """
    Execute a natural language analytical query on the dataset.
    Classifies analytical intent and executes vectorized statistical calculations.

    Returns
    -------
    dict with:
        text     : str - plain-English analytical answer with confidence statistics
        table    : pd.DataFrame | None - tabular result if applicable
        figure   : plotly.graph_objects.Figure | None - interactive chart if applicable
    """
    if df.empty:
        return {"text": "The dataset is empty. Please upload or provide a populated dataset.", "table": None, "figure": None}

    q = query.strip()
    cols = list(df.columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    matched_cols = _find_matching_columns(q, cols)
    intent = _classify_intent(q, matched_cols, num_cols, cat_cols)

    try:
        # 1. Outlier Investigation
        if intent == AnalyticalIntent.OUTLIER_INVESTIGATION:
            target_c = (
                next((c for c in matched_cols if c in num_cols), None)
                or (num_cols[0] if num_cols else None)
            )
            if target_c:
                return _execute_outlier_investigation(df, target_c)

        # 2. Bivariate / Correlation
        if intent == AnalyticalIntent.BIVARIATE_RELATIONSHIP:
            matched_num = [c for c in matched_cols if c in num_cols]
            if len(matched_num) >= 2:
                return _execute_bivariate_relationship(df, matched_num[0], matched_num[1], target_col)
            elif "matrix" in q.lower() or len(num_cols) >= 2:
                corr_df = df[num_cols].corr().round(3)
                figure = None
                try:
                    import plotly.express as px
                    figure = px.imshow(
                        corr_df,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation Matrix",
                    )
                    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
                except Exception:
                    log.debug("Correlation heatmap render error", exc_info=True)
                return {
                    "text": f"Calculated the pairwise correlation matrix across {len(num_cols)} numeric features.",
                    "table": corr_df,
                    "figure": figure,
                }

        # 3. Top-K Contributors
        if intent == AnalyticalIntent.TOP_K_CONTRIBUTORS:
            k_match = re.search(r"\b(?:top|bottom|lowest|highest)\s+(\d+)\b", q.lower())
            k = int(k_match.group(1)) if k_match else 5
            ascending = any(w in q.lower() for w in ["bottom", "lowest", "worst", "smallest"])

            dim_c = next((c for c in matched_cols if c in cat_cols), (cat_cols[0] if cat_cols else None))
            metric_c = next((c for c in matched_cols if c in num_cols), (num_cols[0] if num_cols else None))

            if dim_c and metric_c and dim_c != metric_c:
                return _execute_top_k_contributors(df, dim_c, metric_c, k=k, ascending=ascending)

        # 4. Segment Comparison
        if intent == AnalyticalIntent.SEGMENT_COMPARISON:
            grp_c = next((c for c in matched_cols if c in cat_cols), None) or next((c for c in matched_cols if c in cols), (cat_cols[0] if cat_cols else None))
            metric_c = next((c for c in matched_cols if c in num_cols and c != grp_c), (num_cols[0] if num_cols else None))

            if grp_c and metric_c and grp_c != metric_c:
                return _execute_segment_comparison(df, grp_c, metric_c)

        # 5. Distribution Profile
        if intent == AnalyticalIntent.DISTRIBUTION_PROFILE:
            target_c = (
                next((c for c in matched_cols if c in num_cols), None)
                or (matched_cols[0] if matched_cols else None)
                or (num_cols[0] if num_cols else (cols[0] if cols else None))
            )
            if target_c:
                return _execute_distribution_profile(df, target_c)

    except Exception:
        log.warning("Analytical execution failed; falling back to general snapshot", exc_info=True)

    # 6. Fallback General Snapshot
    n_rows, n_cols = df.shape
    missing_sum = int(df.isnull().sum().sum())
    return {
        "text": (
            f"I analyzed your query: *\"{query}\"*. Here is a quick snapshot of the dataset:\n"
            f"- **Rows & Columns:** {n_rows:,} rows, {n_cols} columns\n"
            f"- **Numeric features:** {len(num_cols)} ({', '.join(num_cols[:4])}{'...' if len(num_cols) > 4 else ''})\n"
            f"- **Categorical features:** {len(cat_cols)} ({', '.join(cat_cols[:4])}{'...' if len(cat_cols) > 4 else ''})\n"
            f"- **Total Missing Values:** {missing_sum:,}\n\n"
            f"💡 *Tip: Try asking queries like: 'Average {num_cols[0] if num_cols else 'metric'} by {cat_cols[0] if cat_cols else 'category'}', 'Outliers in {num_cols[0] if num_cols else 'metric'}', or 'Correlation between {num_cols[0] if len(num_cols) > 0 else 'col1'} and {num_cols[1] if len(num_cols) > 1 else 'col2'}'.*"
        ),
        "table": None,
        "figure": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RAG-grounded copilot — retrieval always runs, an LLM tool loop runs on top
# of it when a provider is available, and execute_natural_language_query()
# above is always the floor nothing here can fall below.
# ═══════════════════════════════════════════════════════════════════════════

_QUERY_DATAFRAME_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": (
                "Run a read-only aggregate query against the dataset. Never returns raw rows — "
                "only aggregates (counts, means, correlations). Use this whenever the user asks "
                "for a specific number rather than a general explanation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "describe_column", "value_counts", "groupby_agg",
                            "correlation", "row_count", "null_count",
                        ],
                    },
                    "column": {"type": "string", "description": "Primary column name."},
                    "group_by": {"type": "string", "description": "Column to group by (groupby_agg only)."},
                    "agg": {
                        "type": "string",
                        "enum": ["mean", "median", "sum", "min", "max", "count"],
                        "description": "Aggregation function (groupby_agg only).",
                    },
                    "column_b": {"type": "string", "description": "Second column name (correlation only)."},
                    "top_n": {"type": "integer", "description": "Max rows to return, capped at 50."},
                },
                "required": ["operation"],
            },
        },
    }
]

_AGG_FUNCS = {
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "sum": lambda s: s.sum(),
    "min": lambda s: s.min(),
    "max": lambda s: s.max(),
    "count": lambda s: s.count(),
}


def _dispatch_query_dataframe(df: pd.DataFrame, args: dict[str, Any], pii_columns: set[str]) -> dict[str, Any]:
    """
    Closed-form, allow-listed dispatch for the query_dataframe tool.

    No eval/exec, no dynamic getattr(df, name) — every operation is an
    explicit branch, every column argument is membership-checked against
    df.columns before use, and PII columns are refused for anything except
    row_count/null_count. Results are aggregates only, capped, JSON-safe.
    A bad or hallucinated call returns {"error": ...} rather than raising,
    so the model can self-correct on its next turn.
    """
    op = args.get("operation")
    try:
        top_n = min(int(args.get("top_n") or 10), 50)
    except (TypeError, ValueError):
        top_n = 10

    def _valid_column(name: Any, allow_pii: bool = False) -> str | None:
        if not isinstance(name, str) or name not in df.columns:
            return None
        if not allow_pii and name in pii_columns:
            return None
        return name

    if op == "row_count":
        return {"row_count": int(len(df))}

    if op == "null_count":
        col = _valid_column(args.get("column"), allow_pii=True)
        if col is None:
            return {"error": f"Unknown column: {args.get('column')!r}"}
        return {"column": col, "null_count": int(df[col].isna().sum())}

    if op == "describe_column":
        col = _valid_column(args.get("column"))
        if col is None:
            return {"error": f"Column not available for this operation: {args.get('column')!r}"}
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            return {
                "column": col,
                "mean": float(series.mean()) if len(series) else None,
                "median": float(series.median()) if len(series) else None,
                "std": float(series.std()) if len(series) else None,
                "min": float(series.min()) if len(series) else None,
                "max": float(series.max()) if len(series) else None,
                "count": int(series.count()),
            }
        mode = series.mode()
        return {
            "column": col,
            "unique_count": int(series.nunique()),
            "top_value": str(mode.iloc[0]) if not mode.empty else None,
        }

    if op == "value_counts":
        col = _valid_column(args.get("column"))
        if col is None:
            return {"error": f"Column not available for this operation: {args.get('column')!r}"}
        counts = df[col].value_counts().head(top_n)
        return {"column": col, "counts": {str(k): int(v) for k, v in counts.items()}}

    if op == "groupby_agg":
        group_col = _valid_column(args.get("group_by"))
        target_col = _valid_column(args.get("column"))
        agg_name = args.get("agg", "mean")
        agg_fn = _AGG_FUNCS.get(agg_name)
        if group_col is None or target_col is None:
            return {"error": "group_by and column must both be valid, non-PII column names."}
        if agg_fn is None:
            return {"error": f"Unsupported aggregation: {agg_name!r}"}
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return {"error": f"Column '{target_col}' is not numeric; cannot aggregate."}
        grouped = df.groupby(group_col)[target_col].apply(agg_fn).sort_values(ascending=False).head(top_n)
        return {
            "group_by": group_col,
            "column": target_col,
            "agg": agg_name,
            "result": {str(k): float(v) for k, v in grouped.items()},
        }

    if op == "correlation":
        col_a = _valid_column(args.get("column"))
        col_b = _valid_column(args.get("column_b"))
        if col_a is None or col_b is None:
            return {"error": "Both 'column' and 'column_b' must be valid, non-PII numeric column names."}
        if not (pd.api.types.is_numeric_dtype(df[col_a]) and pd.api.types.is_numeric_dtype(df[col_b])):
            return {"error": "Both columns must be numeric for correlation."}
        corr = df[[col_a, col_b]].dropna().corr().iloc[0, 1]
        return {"column_a": col_a, "column_b": col_b, "correlation": None if pd.isna(corr) else round(float(corr), 4)}

    return {"error": f"Unknown operation: {op!r}"}


def _run_llm_tool_loop(
    df: pd.DataFrame,
    query: str,
    hits: list[Any],
    sources: list[dict[str, Any]],
    provider: Any,
    pii_columns: set[str],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    context_block = "\n".join(f"- {h.chunk.text}" for h in hits)
    system_prompt = (
        "You are a data analysis copilot. Answer the user's question about their dataset "
        "using ONLY the schema/context below and the query_dataframe tool for actual numbers. "
        "Never invent statistics you haven't retrieved or queried for. Keep answers to 2-4 sentences.\n\n"
        f"Dataset context:\n{context_block}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for _ in range(LLM_MAX_TOOL_ITERATIONS):
        response = provider.generate(messages, tools=_QUERY_DATAFRAME_TOOL)

        if not response.tool_calls:
            return {
                "text": response.text.strip() or baseline["text"],
                "table": baseline.get("table"),
                "figure": baseline.get("figure"),
                "sources": sources,
                "engine": f"llm:{provider.name}",
            }

        messages.append({"role": "assistant", "content": response.text, "tool_calls": response.tool_calls})
        for call in response.tool_calls:
            tool_result = _dispatch_query_dataframe(df, call.get("arguments") or {}, pii_columns)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": call.get("name", "query_dataframe"),
                "content": json.dumps(tool_result),
            })

    return {
        "text": baseline["text"],
        "table": baseline.get("table"),
        "figure": baseline.get("figure"),
        "sources": sources,
        "engine": "retrieval_only",
        "fallback_reason": "LLM did not produce a final answer within the tool-iteration budget.",
    }


def answer_with_rag(
    df: pd.DataFrame,
    query: str,
    target_col: str | None,
    rag_index: SimpleVectorIndex | None,
    llm_provider: Any = None,
    pii_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    RAG-grounded entry point for the chat copilot.

    Same return shape as execute_natural_language_query (text/table/figure),
    plus:
      sources : list[{"chunk_id","source_type","snippet","score"}] | None
      engine  : "deterministic" | "retrieval_only" | "llm:<provider_name>"
    """
    baseline = execute_natural_language_query(df, query, target_col)
    baseline.setdefault("sources", None)
    baseline.setdefault("engine", "deterministic")

    if rag_index is None or len(rag_index) == 0:
        return baseline

    try:
        hits = rag_index.search(query)
    except Exception:
        log.warning("Retrieval search failed; returning the deterministic answer.", exc_info=True)
        return baseline

    if not hits:
        return baseline

    sources = [
        {
            "chunk_id": h.chunk.chunk_id,
            "source_type": h.chunk.source_type,
            "snippet": h.chunk.text,
            "score": round(h.score, 4),
        }
        for h in hits
    ]

    provider = llm_provider
    if provider is None:
        from .llm import get_default_provider

        provider = get_default_provider()

    if provider is None:
        result = dict(baseline)
        result["sources"] = sources
        result["engine"] = "retrieval_only"
        return result

    pii_set = set(pii_columns) if pii_columns is not None else set(detect_pii_columns(list(df.columns)))

    try:
        return _run_llm_tool_loop(df, query, hits, sources, provider, pii_set, baseline)
    except Exception as exc:
        log.warning("LLM tool loop failed (%s); falling back to the retrieval-only answer.", exc, exc_info=True)
        result = dict(baseline)
        result["sources"] = sources
        result["engine"] = "retrieval_only"
        result["fallback_reason"] = str(exc)[:200]
        return result
