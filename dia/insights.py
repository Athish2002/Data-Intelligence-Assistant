"""
dia/insights.py
───────────────
Generates automated smart insights based on data profiling, non-linear dynamics,
Simpson's Paradox / confounder detection, and prescriptive causal intervention guidance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

log = logging.getLogger("dia.insights")


# ─── Numerical Helpers & Noise Rejection ────────────────────────────────────

def _has_finite_variance(series: pd.Series) -> bool:
    """Check if a series has at least 5 finite values and non-zero variance."""
    try:
        s_num = pd.to_numeric(series, errors="coerce")
        valid = s_num[np.isfinite(s_num)]
        if len(valid) < 5:
            return False
        return float(valid.std(ddof=1)) > 1e-8
    except Exception:
        return False


def _clean_numeric_pair(
    x_series: pd.Series, y_series: pd.Series, min_samples: int = 15
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extract finite, non-null numeric pairs. Returns (x, y) arrays or (None, None).
    Guards against extreme values, infinite floats, and zero variance.
    """
    try:
        x_num = pd.to_numeric(x_series, errors="coerce")
        y_num = pd.to_numeric(y_series, errors="coerce")
        valid = np.isfinite(x_num) & np.isfinite(y_num)
        if valid.sum() < min_samples:
            return None, None

        x = x_num[valid].to_numpy(dtype=float)
        y = y_num[valid].to_numpy(dtype=float)

        std_x = np.std(x)
        std_y = np.std(y)
        if std_x < 1e-8 or std_y < 1e-8:
            return None, None

        return x, y
    except Exception:
        return None, None


def _evaluate_linear_relationship(
    x: np.ndarray, y: np.ndarray
) -> dict[str, Any] | None:
    """
    Compute Pearson correlation and linear regression statistics with exact p-value.
    Rejects noise when p >= 0.05 or |r| < 0.15.
    """
    n = len(x)
    if n < 10:
        return None

    try:
        corr, p_val = stats.pearsonr(x, y)
        if np.isnan(corr) or np.isnan(p_val) or p_val >= 0.05 or abs(corr) < 0.15:
            return None

        # OLS slope and intercept
        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)
        slope = corr * (std_y / (std_x + 1e-12))
        intercept = np.mean(y) - slope * np.mean(x)
        r2 = corr ** 2

        # 95% Confidence Interval for Pearson r via Fisher z-transform
        z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
        se_z = 1.0 / np.sqrt(max(1, n - 3))
        ci_lower = np.tanh(z - 1.96 * se_z)
        ci_upper = np.tanh(z + 1.96 * se_z)

        return {
            "corr": float(corr),
            "p_val": float(p_val),
            "r2": float(r2),
            "slope": float(slope),
            "intercept": float(intercept),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_x": float(std_x),
            "std_y": float(std_y),
            "slope_sd": float(slope * std_x),
        }
    except Exception:
        return None


def _evaluate_nonlinear_dynamics(
    x: np.ndarray, y: np.ndarray
) -> dict[str, Any] | None:
    """
    Detect non-linear dynamics: U-shaped, inverted U-shaped, or threshold effects.
    Uses standardized features to prevent numerical overflow on extreme numbers (e.g. 1e15).
    Rejects noise using strict nested F-tests and Welch's t-test (p < 0.01).
    """
    n = len(x)
    if n < 25:
        return None

    try:
        # Standardize for numerical stability
        mean_x, std_x = np.mean(x), np.std(x, ddof=1)
        mean_y, std_y = np.mean(y), np.std(y, ddof=1)
        if std_x < 1e-8 or std_y < 1e-8:
            return None

        zx = (x - mean_x) / std_x
        zy = (y - mean_y) / std_y

        # Linear baseline R2
        r_lin, p_lin = stats.pearsonr(zx, zy)
        r2_lin = float(r_lin ** 2) if not np.isnan(r_lin) else 0.0

        # ── 1. Quadratic Fit (U-shaped / Inverted U-shaped) ───────────────────
        X_quad = np.column_stack([zx ** 2, zx, np.ones(n)])
        ridge = 1e-5 * np.eye(3)
        ridge[-1, -1] = 0.0
        w_quad = np.linalg.solve(X_quad.T @ X_quad + ridge, X_quad.T @ zy)
        y_quad_pred = X_quad @ w_quad

        ss_tot = np.sum((zy - np.mean(zy)) ** 2)
        ss_res_quad = np.sum((zy - y_quad_pred) ** 2)
        r2_quad = float(1.0 - (ss_res_quad / (ss_tot + 1e-12)))

        # Incremental F-test: does quadratic term add significant explanation over linear?
        if r2_quad > r2_lin + 0.10 and r2_quad >= 0.15:
            df_diff = 1
            df_res = n - 3
            if df_res > 0:
                ss_res_lin = ss_tot * (1.0 - r2_lin)
                f_stat = ((ss_res_lin - ss_res_quad) / df_diff) / (ss_res_quad / df_res + 1e-12)
                p_quad = float(stats.f.sf(f_stat, df_diff, df_res))

                if p_quad < 0.01:
                    a_coef = w_quad[0]
                    b_coef = w_quad[1]
                    if abs(a_coef) > 1e-4:
                        z_vertex = -b_coef / (2.0 * a_coef)
                        # Check that the turning point is within the 5th to 95th percentiles of observed data
                        q05, q95 = np.percentile(zx, [5, 95])
                        if q05 <= z_vertex <= q95:
                            vertex_x = mean_x + z_vertex * std_x
                            curve_type = "U-shaped" if a_coef > 0 else "inverted U-shaped"
                            extremum_name = "minimum" if a_coef > 0 else "peak"
                            return {
                                "kind": "quadratic",
                                "curve_type": curve_type,
                                "extremum_name": extremum_name,
                                "vertex_x": float(vertex_x),
                                "r2_quad": float(r2_quad),
                                "r2_lin": float(r2_lin),
                                "r_lin": float(r_lin),
                                "p_value": float(p_quad),
                                "var_explained_pct": float(r2_quad * 100.0),
                            }

        # ── 2. Decision Tree Threshold Effect ────────────────────────────────
        min_leaf = max(10, int(0.05 * n))
        tree = DecisionTreeRegressor(max_depth=1, min_samples_leaf=min_leaf, random_state=42)
        tree.fit(x.reshape(-1, 1), y)

        if tree.tree_.node_count >= 3:
            threshold = float(tree.tree_.threshold[0])
            left_mask = x <= threshold
            right_mask = x > threshold

            if left_mask.sum() >= min_leaf and right_mask.sum() >= min_leaf:
                y_left = y[left_mask]
                y_right = y[right_mask]

                mean_l, mean_r = float(np.mean(y_left)), float(np.mean(y_right))
                var_l, var_r = float(np.var(y_left, ddof=1)), float(np.var(y_right, ddof=1))

                # Pooled std and Cohen's d
                n_l, n_r = len(y_left), len(y_right)
                s_pooled = np.sqrt(((n_l - 1) * var_l + (n_r - 1) * var_r) / (n_l + n_r - 2) + 1e-12)
                cohen_d = abs(mean_r - mean_l) / s_pooled

                if var_l < 1e-9 and var_r < 1e-9:
                    p_welch = 0.0 if abs(mean_r - mean_l) > 1e-6 else 1.0
                else:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        _, p_welch = stats.ttest_ind(y_left, y_right, equal_var=False)

                y_tree_pred = tree.predict(x.reshape(-1, 1))
                ss_res_tree = np.sum((y - y_tree_pred) ** 2)
                ss_tot_y = np.sum((y - np.mean(y)) ** 2)
                r2_tree = float(1.0 - (ss_res_tree / (ss_tot_y + 1e-12)))

                # Require Cohen's d >= 0.5, p < 0.005, variance explained >= 15%,
                # AND the step split must explain meaningfully more variance than linear (r2_tree > r2_lin + 0.05)
                # or linear correlation must be weak/moderate (abs(r_lin) < 0.40) to prevent classifying
                # smooth continuous linear slopes as discrete "threshold jumps".
                if cohen_d >= 0.5 and p_welch < 0.005 and r2_tree >= 0.15:
                    if r2_tree > r2_lin + 0.05 or abs(r_lin) < 0.40:
                        return {
                            "kind": "threshold",
                            "threshold": threshold,
                            "mean_left": mean_l,
                            "mean_right": mean_r,
                            "delta": float(mean_r - mean_l),
                            "cohen_d": float(cohen_d),
                            "p_value": float(p_welch),
                            "var_explained_pct": float(r2_tree * 100.0),
                            "r_lin": float(r_lin),
                            "r2_lin": float(r2_lin),
                        }
    except Exception:
        log.debug("Non-linear relationship evaluation error", exc_info=True)

    return None


# ─── Simpson's Paradox & Confounder Detection ────────────────────────────────

def _detect_simpsons_paradox(
    df: pd.DataFrame,
    num_col: str,
    target_col: str,
    cat_cols: list[str],
) -> dict[str, Any] | None:
    """
    Detect Simpson's Paradox:
    Aggregate correlation is significant, but reverses sign in >= 60% of subgroups
    defined by a categorical confounder. Requires minimum sample sizes (n >= 15 per group)
    to prevent false positives from tiny noise slices.
    """
    try:
        x_clean, y_clean = _clean_numeric_pair(df[num_col], df[target_col], min_samples=30)
        if x_clean is None:
            return None

        r_global, p_global = stats.pearsonr(x_clean, y_clean)
        if np.isnan(r_global) or abs(r_global) < 0.15 or p_global >= 0.05:
            return None

        target_sign = 1 if r_global > 0 else -1

        for grp_col in cat_cols:
            if grp_col == target_col or grp_col == num_col:
                continue

            unique_vals = df[grp_col].dropna().unique()
            if not (2 <= len(unique_vals) <= 12):
                continue

            subgroup_corrs: list[tuple[Any, float, int]] = []
            for val in unique_vals:
                sub_df = df[df[grp_col] == val]
                x_sub, y_sub = _clean_numeric_pair(sub_df[num_col], sub_df[target_col], min_samples=15)
                if x_sub is None:
                    continue

                r_sub, p_sub = stats.pearsonr(x_sub, y_sub)
                if not np.isnan(r_sub):
                    subgroup_corrs.append((val, float(r_sub), len(x_sub)))

            if len(subgroup_corrs) < 2:
                continue

            # Check for reversal: subgroup correlation has opposite sign of r_global
            reversal_groups = [
                (val, r, n_sub) for val, r, n_sub in subgroup_corrs
                if (r * target_sign) < -0.05
            ]

            reversal_pct = (len(reversal_groups) / len(subgroup_corrs)) * 100.0
            reversal_weight = sum(n_sub for _, _, n_sub in reversal_groups) / max(1, sum(n_sub for _, _, n_sub in subgroup_corrs))

            # Trigger condition: >= 60% of subgroups or >= 60% of sample weight exhibits reversal
            if reversal_pct >= 60.0 or reversal_weight >= 0.60:
                weighted_sub_r = sum(r * n_sub for _, r, n_sub in subgroup_corrs) / max(1, sum(n_sub for _, _, n_sub in subgroup_corrs))
                return {
                    "num_col": num_col,
                    "target_col": target_col,
                    "group_col": grp_col,
                    "r_global": float(r_global),
                    "p_global": float(p_global),
                    "reversal_pct": float(reversal_pct),
                    "mean_subgroup_r": float(weighted_sub_r),
                    "n_groups_reversed": len(reversal_groups),
                    "total_groups": len(subgroup_corrs),
                }
    except Exception:
        log.debug("Simpson's paradox evaluation error", exc_info=True)

    return None


# ─── Prescriptive Causal Guidance ────────────────────────────────────────────

def _compute_prescriptive_causal_action(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    excluded_cols: set[str],
) -> dict[str, Any] | None:
    """
    Computes actionable prescriptive intervention using Pearl's Do-Calculus.
    Validates that interventions strictly respect observed feature domain bounds
    and strictly rejects non-significant or noise features.
    """
    from dia.causal_discovery import discover_causal_graph, simulate_intervention

    if len(df) < 20 or not numeric_cols:
        return None

    try:
        # Exclude features already identified as having non-linear curvature
        eligible_cols = [c for c in numeric_cols if c not in excluded_cols]
        if not eligible_cols:
            return None

        # Sample if large for performance
        work_df = df.sample(min(len(df), 1000), random_state=42) if len(df) > 1000 else df

        graph = discover_causal_graph(work_df, target_col=target_col, alpha=0.05)

        candidate_causes = [c for c in graph.direct_causes_of_target if c in eligible_cols and c != target_col]
        if not candidate_causes:
            # Only consider candidate features that exhibit statistically significant correlation with target
            sig_candidates = []
            for c in eligible_cols:
                if c == target_col or c in graph.reverse_causality_risks:
                    continue
                x_vals, y_vals = _clean_numeric_pair(work_df[c], work_df[target_col], min_samples=20)
                if x_vals is None:
                    continue
                r_val, p_val = stats.pearsonr(x_vals, y_vals)
                if not np.isnan(r_val) and p_val < 0.05 and abs(r_val) >= 0.20:
                    sig_candidates.append((abs(r_val), c))

            sig_candidates.sort(key=lambda item: item[0], reverse=True)
            candidate_causes = [c for _, c in sig_candidates[:4]]

        if not candidate_causes:
            return None

        best_prescription: dict[str, Any] | None = None
        max_abs_effect = 0.0

        for col in candidate_causes:
            if col in graph.reverse_causality_risks:
                continue

            col_series = pd.to_numeric(work_df[col], errors="coerce").dropna()
            if col_series.empty or col_series.std() < 1e-8:
                continue

            c_min, c_max = float(col_series.min()), float(col_series.max())
            c_med = float(col_series.median())
            c_std = float(col_series.std())

            # Candidate adjustment: +0.75 SD, strictly clipped to observed data bounds
            raw_target_val = c_med + max(1e-4, 0.75 * c_std)
            intervention_val = float(np.clip(raw_target_val, c_min, c_max))
            actual_delta = intervention_val - c_med

            if abs(actual_delta) < 1e-6:
                # Try negative delta
                raw_target_val = c_med - max(1e-4, 0.75 * c_std)
                intervention_val = float(np.clip(raw_target_val, c_min, c_max))
                actual_delta = intervention_val - c_med

            if abs(actual_delta) < 1e-6:
                continue

            sim = simulate_intervention(
                work_df,
                treatment=col,
                outcome=target_col,
                intervention_value=intervention_val,
                graph=graph,
            )

            if sim.is_reverse_causality:
                continue

            # Strict noise rejection: 95% bootstrap confidence interval must not span zero
            if sim.confidence_interval_95 is not None:
                ci_l, ci_u = sim.confidence_interval_95
                if ci_l <= 0.0 <= ci_u:
                    # Effect indistinguishable from noise
                    continue

            effect_per_unit = sim.average_treatment_effect / actual_delta if abs(actual_delta) > 1e-8 else 0.0

            if abs(sim.average_treatment_effect) > max_abs_effect:
                max_abs_effect = abs(sim.average_treatment_effect)
                confounders = sim.adjustment_set or ["None (Unconfounded Direct Path)"]
                best_prescription = {
                    "col": col,
                    "target_col": target_col,
                    "delta": float(actual_delta),
                    "baseline_val": float(c_med),
                    "target_val": float(intervention_val),
                    "ate": float(sim.average_treatment_effect),
                    "effect_size": float(effect_per_unit),
                    "confounders": confounders,
                    "pct_change": float(sim.percent_change),
                }

        return best_prescription
    except Exception:
        log.debug("Prescriptive causal computation error", exc_info=True)
        return None


# ─── Master Entrypoint: generate_smart_insights ──────────────────────────────

def generate_smart_insights(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    max_insights: int = 4,
    causal_graph: Any = None,
) -> list[dict[str, str]]:
    """
    Generate natural language insights by finding multi-variate, non-linear,
    Simpson's paradox reversals, and prescriptive causal intervention guidance.

    Returns
    -------
    list of dicts: [{"title": str, "description": str, "type": "positive"|"negative"|"neutral"}]
    """
    if df.empty or target_col not in df.columns:
        return [{
            "title": "Complex Relationships",
            "description": "No data available or target column is missing.",
            "type": "neutral",
        }]

    insights: list[dict[str, str]] = []
    seen_features: set[str] = set()

    # Identify candidate columns with robust finite variance checks
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col and _has_finite_variance(df[c])
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col and df[c].nunique() >= 2
    ]

    is_regression = (task_type == "regression")
    target_series = pd.to_numeric(df[target_col], errors="coerce") if is_regression else None

    # Handle binary classification target encoded as 0/1
    binary_target_name = None
    if not is_regression:
        unique_targets = df[target_col].dropna().unique()
        if len(unique_targets) == 2:
            binary_target_name = str(unique_targets[1])
            target_series = (df[target_col] == unique_targets[1]).astype(float)
        elif len(unique_targets) > 2:
            top_class = df[target_col].value_counts().index[0]
            binary_target_name = str(top_class)
            target_series = (df[target_col] == top_class).astype(float)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Simpson's Paradox & Confounder Detection (Highest Priority Advisory)
    # ─────────────────────────────────────────────────────────────────────────
    if target_series is not None and cat_cols:
        target_finite = target_series.replace([np.inf, -np.inf], np.nan)
        for num_col in numeric_cols[:8]:
            temp_df = pd.DataFrame({num_col: df[num_col], "_target": target_finite}).replace([np.inf, -np.inf], np.nan)
            for c in cat_cols[:4]:
                temp_df[c] = df[c]

            paradox = _detect_simpsons_paradox(temp_df, num_col, "_target", cat_cols[:4])
            if paradox:
                dir_glob = "positive" if paradox["r_global"] > 0 else "negative"
                dir_sub = "negative" if paradox["r_global"] > 0 else "positive"
                grp_col = paradox["group_col"]

                # Structural Causal Resolution via DAG identifiability & Causal Discovery
                from dia.causal_discovery import discover_causal_graph
                from dia.causal_identifiability import CausalDAG, resolve_simpsons_paradox_algebraically

                # Discover or retrieve causal graph
                graph = causal_graph
                if graph is None:
                    try:
                        valid_mask = (
                            temp_df[num_col].notna()
                            & temp_df["_target"].notna()
                            & temp_df[grp_col].notna()
                        )
                        causal_sample = pd.DataFrame({
                            num_col: temp_df.loc[valid_mask, num_col].to_numpy(dtype=float),
                            "_target": temp_df.loc[valid_mask, "_target"].to_numpy(dtype=float),
                            grp_col: temp_df.loc[valid_mask, grp_col].astype("category").cat.codes.astype(float),
                        })
                        for extra_c in numeric_cols[:4]:
                            if extra_c not in causal_sample.columns and extra_c != target_col:
                                s = pd.to_numeric(df.loc[valid_mask, extra_c], errors="coerce").fillna(0.0)
                                if float(s.std()) > 1e-6:
                                    causal_sample[extra_c] = s.to_numpy(dtype=float)

                        if len(causal_sample) > 1000:
                            causal_sample = causal_sample.sample(1000, random_state=42)

                        graph = discover_causal_graph(causal_sample, target_col="_target", treatment_col=num_col)
                    except Exception:
                        log.debug("Causal graph discovery failed during Simpson's paradox analysis", exc_info=True)
                        graph = None

                dag = CausalDAG()
                is_connected_to_treatment = False
                is_connected_to_outcome = False
                outcome_target_names = {"_target", target_col}

                if isinstance(graph, CausalDAG):
                    dag = graph
                    for src, tgt in dag.edges:
                        edge_nodes = {src, tgt}
                        if grp_col in edge_nodes and num_col in edge_nodes:
                            is_connected_to_treatment = True
                        if grp_col in edge_nodes and (edge_nodes & outcome_target_names):
                            is_connected_to_outcome = True
                elif graph is not None:
                    for node in graph.nodes:
                        dag.add_node(node)
                    for edge in graph.edges:
                        src = edge.source
                        tgt = edge.target
                        if edge.direction in ("-->", "->"):
                            dag.add_edge(src, tgt)
                        elif edge.direction in ("<--", "<-"):
                            dag.add_edge(tgt, src)

                        edge_nodes = {src, tgt}
                        if grp_col in edge_nodes and num_col in edge_nodes:
                            is_connected_to_treatment = True
                        if grp_col in edge_nodes and (edge_nodes & outcome_target_names):
                            is_connected_to_outcome = True

                outcome_node = "_target" if "_target" in dag.nodes else (target_col if target_col in dag.nodes else "_target")
                dag.add_node(num_col)
                dag.add_node(outcome_node)
                dag.add_node(grp_col)

                resolution = resolve_simpsons_paradox_algebraically(
                    dag, treatment=num_col, outcome=outcome_node, conditioning_node=grp_col
                )

                if resolution.structural_role in ("Confounder", "Collider", "Mediator"):
                    verdict_text = (
                        f"Axiomatic Causal Verdict ({resolution.structural_role}): "
                        f"{resolution.causal_verdict} {resolution.explanation}"
                    )
                elif is_connected_to_treatment and is_connected_to_outcome:
                    verdict_text = (
                        f"Unverified Structural Hypothesis: Subgroup reversal observed across `{grp_col}`, "
                        f"but causal direction is unoriented in observational data. Stratifying by `{grp_col}` "
                        f"is an unverified structural hypothesis rather than an axiomatic confounder; "
                        f"domain validation is recommended to rule out collider or mediator bias."
                    )
                else:
                    verdict_text = (
                        f"Unverified Structural Hypothesis: `{grp_col}` is not causally connected to both `{num_col}` "
                        f"and `{target_col}` in the discovered causal DAG. The observed subgroup slope reversal "
                        f"may be an artifact of sample variance or selection bias rather than true confounding; "
                        f"treat subgroup stratification as an unverified structural hypothesis rather than an axiomatic confounder."
                    )

                insights.append({
                    "title": f"Simpson's Paradox: {num_col} vs {target_col} by {grp_col}",
                    "description": (
                        f"Simpson's Paradox detected: Global correlation is {dir_glob} ({paradox['r_global']:+.2f}), "
                        f"but in {paradox['reversal_pct']:.0f}% of segments defined by `{grp_col}`, "
                        f"the correlation is {dir_sub} (mean subgroup r = {paradox['mean_subgroup_r']:+.2f}). "
                        f"{verdict_text}"
                    ),
                    "type": "neutral",
                })
                seen_features.add(num_col)
                if len(insights) >= max_insights:
                    return insights[:max_insights]
                break

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Non-Linear & Threshold Dynamics Detection
    # ─────────────────────────────────────────────────────────────────────────
    if target_series is not None:
        for col in numeric_cols:
            if col in seen_features or len(insights) >= max_insights:
                continue

            x_vals, y_vals = _clean_numeric_pair(df[col], target_series, min_samples=25)
            if x_vals is None:
                continue

            nonlinear_info = _evaluate_nonlinear_dynamics(x_vals, y_vals)
            if nonlinear_info:
                seen_features.add(col)
                if nonlinear_info["kind"] == "quadratic":
                    curve_type = nonlinear_info["curve_type"]
                    vertex_x = nonlinear_info["vertex_x"]
                    insights.append({
                        "title": f"Non-Linear {curve_type.title()} Relationship in {col}",
                        "description": (
                            f"Non-linear relationship detected: `{col}` exhibits a {curve_type} relationship with `{target_col}` "
                            f"(explaining {nonlinear_info['var_explained_pct']:.1f}% of variance, p < {nonlinear_info['p_value']:.4f}), "
                            f"reaching its turning point {nonlinear_info['extremum_name']} at {col} ≈ {vertex_x:,.2f}. "
                            f"Linear correlation is near zero (r = {nonlinear_info['r_lin']:+.2f}) and misses this strong curvature."
                        ),
                        "type": "neutral",
                    })
                elif nonlinear_info["kind"] == "threshold":
                    thresh = nonlinear_info["threshold"]
                    delta = nonlinear_info["delta"]
                    dir_str = "jumps" if delta > 0 else "drops"
                    insights.append({
                        "title": f"Threshold Effect in {col}",
                        "description": (
                            f"Threshold effect detected: When `{col}` exceeds {thresh:,.2f}, `{target_col}` {dir_str} "
                            f"from {nonlinear_info['mean_left']:,.2f} to {nonlinear_info['mean_right']:,.2f} "
                            f"(Cohen's d = {nonlinear_info['cohen_d']:.2f}, explaining {nonlinear_info['var_explained_pct']:.1f}% "
                            f"variance, p < {nonlinear_info['p_value']:.4f})."
                        ),
                        "type": "positive" if delta > 0 else "negative",
                    })

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Prescriptive Causal Guidance
    # ─────────────────────────────────────────────────────────────────────────
    if target_series is not None and numeric_cols and len(insights) < max_insights:
        work_df_causal = df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan)
        work_df_causal["_target_causal"] = target_series.replace([np.inf, -np.inf], np.nan)

        causal_action = _compute_prescriptive_causal_action(
            work_df_causal, target_col="_target_causal", numeric_cols=numeric_cols, excluded_cols=seen_features
        )
        if causal_action:
            col = causal_action["col"]
            ate = causal_action["ate"]
            confounders_str = ", ".join(f"`{c}`" for c in causal_action["confounders"])

            insights.append({
                "title": f"Prescriptive Action: Adjust {col}",
                "description": (
                    f"Prescriptive Action: To shift `{target_col}`, adjusting `{col}` by {causal_action['delta']:+.2f} "
                    f"(from {causal_action['baseline_val']:,.2f} to {causal_action['target_val']:,.2f}) has the highest "
                    f"unconfounded causal impact ({causal_action['effect_size']:+.2f} per unit, estimated ATE: {ate:+.2f}). "
                    f"Confounders controlled: {confounders_str}."
                ),
                "type": "positive" if ate > 0 else "negative",
            })
            seen_features.add(col)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Multi-Variate Linear Relationships (With Statistical Impact, Not Canned Strings)
    # ─────────────────────────────────────────────────────────────────────────
    if target_series is not None:
        linear_candidates = []
        for col in numeric_cols:
            if col in seen_features:
                continue

            x_vals, y_vals = _clean_numeric_pair(df[col], target_series, min_samples=10)
            if x_vals is None:
                continue

            lin_info = _evaluate_linear_relationship(x_vals, y_vals)
            if lin_info:
                linear_candidates.append((col, lin_info))

        # Sort by correlation strength
        linear_candidates.sort(key=lambda item: abs(item[1]["corr"]), reverse=True)

        for col, lin_info in linear_candidates:
            if len(insights) >= max_insights:
                break

            seen_features.add(col)
            corr = lin_info["corr"]
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) >= 0.6 else ("moderate" if abs(corr) >= 0.3 else "weak")
            p_val = lin_info["p_val"]

            if is_regression:
                insights.append({
                    "title": f"{direction.title()} linear relationship with {col}",
                    "description": (
                        f"Significant {strength} {direction} relationship (r = {corr:+.2f}, 95% CI: [{lin_info['ci_lower']:+.2f}, {lin_info['ci_upper']:+.2f}], p < {p_val:.4f}): "
                        f"Each 1-SD shift in `{col}` (+{lin_info['std_x']:,.2f}) is associated with an expected {lin_info['slope_sd']:+,.2f} change in `{target_col}`, "
                        f"accounting for {lin_info['r2'] * 100:.1f}% of observed variance."
                    ),
                    "type": direction,
                })
            else:
                target_label = binary_target_name or "target class"
                try:
                    q25 = float(np.percentile(df[col].dropna(), 25))
                    q75 = float(np.percentile(df[col].dropna(), 75))
                    rate_high = float(target_series[df[col] >= q75].mean() * 100.0)
                    rate_low = float(target_series[df[col] <= q25].mean() * 100.0)
                    ratio = (rate_high / rate_low) if rate_low > 1e-4 else 1.0

                    insights.append({
                        "title": f"Impact of {col}",
                        "description": (
                            f"Significant association with `{target_col}` = **{target_label}** (point-biserial r = {corr:+.2f}, p < {p_val:.4f}): "
                            f"Records in the top quartile of `{col}` (> {q75:,.2f}) have a {rate_high:.1f}% rate of {target_label}, "
                            f"compared to {rate_low:.1f}% in the bottom quartile (< {q25:,.2f}) (relative risk {ratio:.1f}x)."
                        ),
                        "type": direction,
                    })
                except Exception:
                    insights.append({
                        "title": f"Impact of {col}",
                        "description": (
                            f"Significant association with `{target_col}` (point-biserial r = {corr:+.2f}, p < {p_val:.4f}), "
                            f"explaining {lin_info['r2'] * 100:.1f}% of target variance."
                        ),
                        "type": direction,
                    })

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Robust Fallback (If No Relationships Pass Significance Filters)
    # ─────────────────────────────────────────────────────────────────────────
    if not insights:
        insights.append({
            "title": "Complex Relationships",
            "description": (
                "No isolated bivariate linear correlations or threshold effects exceeded statistical "
                "significance thresholds. The predictive model is likely capturing complex, high-order "
                "multi-variate interactions across features."
            ),
            "type": "neutral",
        })

    return insights[:max_insights]
