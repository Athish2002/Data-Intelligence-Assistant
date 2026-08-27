"""
dia/chat_analyst.py
───────────────────
Autonomous Conversational Data Copilot.
Parses natural language analytical queries, executes sandboxed calculations,
and generates rich text summaries, tables, and interactive Plotly visualizations.
Works completely standalone (offline deterministic mode) with optional LLM reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from .config import LLM_MAX_TOOL_ITERATIONS
from .retrieval import SimpleVectorIndex
from .validators import detect_pii_columns

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
                log.debug("Could not render the correlation heatmap chart.", exc_info=True)
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
                    log.debug("Could not render the distribution histogram chart.", exc_info=True)
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
                    log.debug("Could not render the top-categories bar chart.", exc_info=True)
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
                log.debug("Could not render the groupby bar chart.", exc_info=True)
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
                log.debug("Could not render the scatter plot chart.", exc_info=True)
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

    Never raises. Each stage strictly upgrades the previous one — a failure
    anywhere downgrades to the previous stage's result rather than erroring:
      1. execute_natural_language_query() always runs first; this is the floor.
      2. No/empty rag_index -> return the floor unchanged.
      3. Retrieval failure or no hits -> return the floor unchanged.
      4. No LLM provider available -> floor + cited sources ("retrieval_only").
      5. LLM tool loop -> floor + sources + LLM-composed answer ("llm:<name>"),
         falling back to stage 4's result if the loop raises for any reason.
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
