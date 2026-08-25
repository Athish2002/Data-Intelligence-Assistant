"""
dia/chat_analyst.py
───────────────────
Autonomous Conversational Data Copilot.
Parses natural language analytical queries, executes sandboxed calculations,
and generates rich text summaries, tables, and interactive Plotly visualizations.
Works completely standalone (offline deterministic mode) with optional LLM reasoning.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("dia.chat_analyst")


def _find_matching_column(query: str, columns: list[str]) -> str | None:
    """Find the best matching column name from query tokens."""
    q_clean = query.lower().replace("_", " ")
    for col in columns:
        col_clean = col.lower().replace("_", " ")
        if col.lower() in query.lower() or col_clean in q_clean:
            return col
    return None


def execute_natural_language_query(
    df: pd.DataFrame,
    query: str,
    target_col: str | None = None,
) -> dict[str, Any]:
    """
    Execute a natural language analytical query on the dataset.

    Returns
    -------
    dict with:
        text     : str - plain-English analytical answer
        table    : pd.DataFrame | None - tabular result if applicable
        figure   : plotly.graph_objects.Figure | None - chart if applicable
    """
    q = query.strip().lower()
    cols = list(df.columns)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    result: dict[str, Any] = {
        "text": "",
        "table": None,
        "figure": None,
    }

    # ── 1. Correlation Matrix or Pairwise Correlation ─────────────────────────
    if "correlation" in q or "correlate" in q:
        matched = [c for c in num_cols if c.lower() in q or c.lower().replace("_", " ") in q]
        if len(matched) >= 2:
            c1, c2 = matched[0], matched[1]
            corr = df[[c1, c2]].dropna().corr().iloc[0, 1]
            strength = "strong positive" if corr > 0.6 else ("moderate positive" if corr > 0.3 else ("strong negative" if corr < -0.6 else ("moderate negative" if corr < -0.3 else "weak/no linear")))
            result["text"] = f"The Pearson correlation between **`{c1}`** and **`{c2}`** is **{corr:.4f}** ({strength} relationship)."
            return result
        elif len(num_cols) >= 2:
            corr_df = df[num_cols].corr().round(3)
            result["text"] = f"Calculated the pairwise correlation matrix across {len(num_cols)} numeric features."
            result["table"] = corr_df
            try:
                import plotly.express as px
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Matrix",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=10, l=10, r=10))
                result["figure"] = fig
            except Exception:
                pass
            return result

    # ── 2. Distribution / Histogram ───────────────────────────────────────────
    if any(k in q for k in ["distribution", "histogram", "spread", "distribution of"]):
        target_c = _find_matching_column(q, cols) or (num_cols[0] if num_cols else None)
        if target_c and target_c in df.columns:
            s = df[target_c].dropna()
            if pd.api.types.is_numeric_dtype(s):
                result["text"] = (
                    f"**Distribution of `{target_c}`:** Mean = {s.mean():,.2f}, "
                    f"Median = {s.median():,.2f}, Std = {s.std():,.2f}, "
                    f"Min = {s.min():,.2f}, Max = {s.max():,.2f}."
                )
                try:
                    import plotly.express as px
                    fig = px.histogram(
                        df, x=target_c,
                        marginal="box",
                        title=f"Distribution of {target_c}",
                        color_discrete_sequence=["#6366f1"],
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
                    result["figure"] = fig
                except Exception:
                    pass
                return result
            else:
                counts = s.value_counts().reset_index()
                counts.columns = [target_c, "Count"]
                result["text"] = f"**Category breakdown for `{target_c}`:** Top category is `{counts.iloc[0, 0]}` with {counts.iloc[0, 1]:,} occurrences."
                result["table"] = counts.head(10)
                try:
                    import plotly.express as px
                    fig = px.bar(counts.head(10), x=target_c, y="Count", title=f"Top Categories in {target_c}")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
                    result["figure"] = fig
                except Exception:
                    pass
                return result

    # ── 3. Groupby & Segment Analysis ─────────────────────────────────────────
    if any(k in q for k in ["group by", "grouped by", "by ", "per ", "broken down by"]):
        # Find group column and metric column
        matched_group = _find_matching_column(q, cat_cols) or _find_matching_column(q, cols)
        matched_metric = _find_matching_column(q, num_cols) or (num_cols[0] if num_cols else None)

        if matched_group and matched_metric and matched_group != matched_metric:
            grouped = df.groupby(matched_group)[matched_metric].agg(["mean", "median", "count"]).round(2).reset_index()
            grouped.columns = [matched_group, f"Average {matched_metric}", f"Median {matched_metric}", "Count"]
            grouped = grouped.sort_values(by=f"Average {matched_metric}", ascending=False)
            
            top_group = grouped.iloc[0][matched_group]
            top_val = grouped.iloc[0][f"Average {matched_metric}"]
            result["text"] = (
                f"Grouped **`{matched_metric}`** by **`{matched_group}`** across {len(grouped)} segments. "
                f"Highest average is **{top_group}** with **{top_val:,.2f}**."
            )
            result["table"] = grouped.head(15)
            try:
                import plotly.express as px
                fig = px.bar(
                    grouped.head(10),
                    x=matched_group,
                    y=f"Average {matched_metric}",
                    color=f"Average {matched_metric}",
                    title=f"Average {matched_metric} by {matched_group}",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
                result["figure"] = fig
            except Exception:
                pass
            return result

    # ── 4. Average / Mean / Median / Max / Min ─────────────────────────────────
    if any(k in q for k in ["average", "mean", "median", "highest", "lowest", "max", "min"]):
        target_c = _find_matching_column(q, num_cols) or (num_cols[0] if num_cols else None)
        if target_c:
            s = df[target_c].dropna()
            avg_val = s.mean()
            med_val = s.median()
            max_val = s.max()
            min_val = s.min()
            result["text"] = (
                f"📊 Summary statistics for **`{target_c}`**:\n"
                f"- **Average (Mean):** {avg_val:,.2f}\n"
                f"- **Median:** {med_val:,.2f}\n"
                f"- **Minimum:** {min_val:,.2f}\n"
                f"- **Maximum:** {max_val:,.2f}\n"
                f"- **Total Rows:** {len(s):,}"
            )
            return result

    # ── 5. Scatter Plot ───────────────────────────────────────────────────────
    if "scatter" in q or "relationship between" in q or " vs " in q:
        matched = [c for c in num_cols if c.lower() in q or c.lower().replace("_", " ") in q]
        if len(matched) >= 2:
            c1, c2 = matched[0], matched[1]
            result["text"] = f"Plotting scatter comparison between **`{c1}`** and **`{c2}`**."
            try:
                import plotly.express as px
                fig = px.scatter(
                    df.sample(min(len(df), 500)),
                    x=c1, y=c2,
                    color=target_col if target_col and target_col in df.columns else None,
                    title=f"{c1} vs {c2}",
                    opacity=0.7,
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
                result["figure"] = fig
            except Exception:
                pass
            return result

    # ── 6. Fallback General Summary ───────────────────────────────────────────
    n_rows, n_cols = df.shape
    missing_sum = df.isnull().sum().sum()
    result["text"] = (
        f"I analyzed your query: *\"{query}\"*. Here is a quick snapshot of the dataset:\n"
        f"- **Rows & Columns:** {n_rows:,} rows, {n_cols} columns\n"
        f"- **Numeric features:** {len(num_cols)} ({', '.join(num_cols[:4])}{'...' if len(num_cols) > 4 else ''})\n"
        f"- **Categorical features:** {len(cat_cols)} ({', '.join(cat_cols[:4])}{'...' if len(cat_cols) > 4 else ''})\n"
        f"- **Total Missing Values:** {missing_sum:,}\n\n"
        f"💡 *Tip: Try asking queries like: 'Average {num_cols[0] if num_cols else 'charges'} by {cat_cols[0] if cat_cols else 'category'}', 'Distribution of {num_cols[0] if num_cols else 'age'}', or 'Correlation matrix'.*"
    )
    return result
