"""
dia/explainability.py
─────────────────────
Generates feature importance visualizations.

Priority chain:
  1. SHAP summary plot   (if shap installed and dataset not too wide)
  2. sklearn importances / coefficients  → Plotly bar chart
  3. sklearn importances / coefficients  → Seaborn bar chart (static fallback)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .utils import is_plotly_available, is_shap_available, is_seaborn_available

warnings.filterwarnings("ignore")


def generate_explanation(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    importance_series: pd.Series,
    top_n: int = 20,
) -> dict:
    """
    Generate a feature importance visualization.

    Returns
    -------
    dict with keys:
        method   : "shap" | "plotly_importance" | "seaborn_importance"
        figure   : plotly Figure OR matplotlib Figure
        shap_values : np.ndarray | None
    """
    top_importance = importance_series.head(top_n)

    # ── Attempt SHAP ──────────────────────────────────────────────────────────
    if is_shap_available() and len(feature_names) <= 200:
        shap_result = _try_shap(model, X_test, feature_names, top_n)
        if shap_result:
            return shap_result

    # ── Plotly bar chart ──────────────────────────────────────────────────────
    if is_plotly_available() and not top_importance.empty:
        fig = _plotly_importance(top_importance)
        return {"method": "plotly_importance", "figure": fig, "shap_values": None}

    # ── Seaborn fallback ──────────────────────────────────────────────────────
    if is_seaborn_available() and not top_importance.empty:
        fig = _seaborn_importance(top_importance)
        return {"method": "seaborn_importance", "figure": fig, "shap_values": None}

    return {"method": "none", "figure": None, "shap_values": None}


# ─── SHAP ────────────────────────────────────────────────────────────────────

def _try_shap(model, X_test: np.ndarray, feature_names: list[str], top_n: int) -> dict | None:
    try:
        import shap  # type: ignore

        # Use a small sample to keep it fast
        sample = X_test[: min(200, len(X_test))]

        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(sample)
        except Exception:
            try:
                explainer = shap.LinearExplainer(model, sample)
                shap_vals = explainer.shap_values(sample)
            except Exception:
                explainer = shap.KernelExplainer(
                    model.predict, shap.sample(sample, 50)
                )
                shap_vals = explainer.shap_values(sample, nsamples=50)

        # For multi-class, average absolute values across classes
        if isinstance(shap_vals, list):
            shap_arr = np.abs(np.array(shap_vals)).mean(axis=0)
        else:
            shap_arr = np.abs(shap_vals)

        mean_shap = pd.Series(
            shap_arr.mean(axis=0), index=feature_names
        ).sort_values(ascending=False).head(top_n)

        # Render as Plotly if available, else Seaborn
        if is_plotly_available():
            fig = _plotly_importance(mean_shap, title="SHAP Feature Importance (mean |SHAP|)")
        else:
            fig = _seaborn_importance(mean_shap, title="SHAP Feature Importance")

        return {"method": "shap", "figure": fig, "shap_values": shap_arr}
    except Exception:
        return None


# ─── Plotly bar chart ────────────────────────────────────────────────────────

def _plotly_importance(
    series: pd.Series,
    title: str = "Feature Importance",
) -> "plotly.graph_objects.Figure":
    import plotly.graph_objects as go  # type: ignore

    df = series.reset_index()
    df.columns = ["Feature", "Importance"]
    df = df.sort_values("Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["Importance"],
            y=df["Feature"],
            orientation="h",
            marker=dict(
                color=df["Importance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Importance Score",
        yaxis_title="",
        height=max(300, len(df) * 24),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ─── Seaborn bar chart (static fallback) ─────────────────────────────────────

def _seaborn_importance(
    series: pd.Series,
    title: str = "Feature Importance",
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    df = series.reset_index()
    df.columns = ["Feature", "Importance"]
    df = df.sort_values("Importance", ascending=True)

    height = max(4, len(df) * 0.35)
    fig, ax = plt.subplots(figsize=(9, height))

    sns.barplot(
        data=df,
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig
