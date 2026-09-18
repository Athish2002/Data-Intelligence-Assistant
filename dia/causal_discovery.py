"""
Causal Discovery & Pearl's Do-Calculus Policy Simulator.

Implements:
1. Constraint-based causal DAG induction via the PC algorithm:
   - Conditional independence testing with partial correlation & Fisher's z-transform.
   - Skeleton pruning with separating sets.
   - Collider (v-structure) orientation.
   - Meek's orientation rules propagation (R1-R4) to avoid cycles and new colliders.
2. Reverse causality risk detection.
3. Pearl's Do-Calculus policy simulator:
   - Back-door admissibility criterion identification.
   - Expected intervention outcome E[Y | do(X = x)].
   - Average Treatment Effect (ATE) and bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats
from scipy.special import erfc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models / Interface Contracts
# ---------------------------------------------------------------------------

class CausalEdge(BaseModel):
    """Represents a directed or undirected causal dependency between two variables."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    source: str
    target: str
    weight: float
    p_value: float = 0.0
    direction: str = "directed"  # "directed", "undirected", "-->", "<--", "---"
    is_direct_cause_of_target: bool = False


class CausalGraphResult(BaseModel):
    """Result of observational causal graph discovery via PC algorithm."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    nodes: List[str]
    edges: List[CausalEdge]
    adjacency_matrix: Dict[str, Dict[str, float]]
    reverse_causality_risks: List[str] = Field(default_factory=list)
    target_col: Optional[str] = None
    root_causes: List[str] = Field(default_factory=list)
    direct_causes_of_target: List[str] = Field(default_factory=list)
    indirect_causes: List[str] = Field(default_factory=list)
    confounders: List[str] = Field(default_factory=list)
    sink_nodes: List[str] = Field(default_factory=list)


class InterventionResult(BaseModel):
    """Result of Pearl's Do-Calculus intervention simulation E[Y | do(X = x)]."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    treatment: str
    outcome: str
    intervention_value: float
    baseline_expected_outcome: float
    intervened_expected_outcome: float
    average_treatment_effect: float
    adjustment_set: List[str]

    # Backward/alternate compatibility aliases
    treatment_col: Optional[str] = None
    treatment_value: Optional[float] = None
    outcome_col: Optional[str] = None
    counterfactual_expected_outcome: Optional[float] = None
    average_causal_effect: Optional[float] = None
    percent_change: float = 0.0
    backdoor_adjustment_set: List[str] = Field(default_factory=list)
    confidence_interval_95: Optional[Tuple[float, float]] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    is_reverse_causality: bool = False
    policy_interpretation: str = ""


# ---------------------------------------------------------------------------
# Core Statistical & Algorithmic Routines
# ---------------------------------------------------------------------------

def _preprocess_tabular_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    max_features: int = 15,
    treatment_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cleans, validates, and bounds features for causal discovery:
    - Filters to numeric / encodable columns
    - Drops zero-variance columns
    - Imputes NaNs with column medians
    - Bounds feature count to prevent combinatorial blow-up
    - Preserves both target_col and treatment_col without index offsets
    """
    if df.empty:
        return pd.DataFrame(), []

    work_df = df.copy()

    # Convert object/categorical columns to numeric codes
    for col in work_df.columns:
        if work_df[col].dtype == object or str(work_df[col].dtype) in ("category", "string"):
            try:
                work_df[col] = pd.to_numeric(work_df[col])
            except (ValueError, TypeError):
                work_df[col] = work_df[col].astype("category").cat.codes.astype(float)
        elif not np.issubdtype(work_df[col].dtype, np.number):
            work_df[col] = work_df[col].astype(float)

    # Fill NaNs with median or zero
    work_df = work_df.fillna(work_df.median(numeric_only=True).fillna(0.0))

    # Identify valid features with non-zero variance
    valid_cols: List[str] = []
    for col in work_df.columns:
        std_val = float(work_df[col].std(ddof=0))
        if not np.isnan(std_val) and std_val > 1e-8:
            valid_cols.append(col)

    if not valid_cols:
        return pd.DataFrame(), []

    # Prioritize target_col and treatment_col if specified
    selected_cols: List[str] = []
    if target_col and target_col in valid_cols:
        selected_cols.append(target_col)
    if treatment_col and treatment_col in valid_cols and treatment_col not in selected_cols:
        selected_cols.append(treatment_col)

    # Rank remaining columns by variance/correlation
    priority_set = set(selected_cols)
    remaining = [c for c in valid_cols if c not in priority_set]
    if target_col and target_col in work_df.columns:
        # Sort remaining features by absolute correlation with target
        corr_with_target = []
        target_series = work_df[target_col]
        for c in remaining:
            r = abs(float(work_df[c].corr(target_series)))
            corr_with_target.append((0.0 if np.isnan(r) else r, c))
        corr_with_target.sort(reverse=True)
        selected_cols.extend([c for _, c in corr_with_target[: max(0, max_features - len(selected_cols))]])
    else:
        # Sort by variance
        var_sorted = sorted(remaining, key=lambda c: float(work_df[c].var()), reverse=True)
        selected_cols.extend(var_sorted[: max(0, max_features - len(selected_cols))])

    return work_df[selected_cols], selected_cols


def _partial_correlation(
    data: np.ndarray,
    i: int,
    j: int,
    s_indices: Tuple[int, ...],
) -> float:
    """
    Computes the partial correlation between variable i and variable j conditioned on S.
    Uses linear regression residuals for numerical stability.
    """
    xi = data[:, i]
    xj = data[:, j]

    if len(s_indices) == 0:
        # Order-0 correlation
        std_i = np.std(xi)
        std_j = np.std(xj)
        if std_i < 1e-8 or std_j < 1e-8:
            return 0.0
        r = np.corrcoef(xi, xj)[0, 1]
        return 0.0 if np.isnan(r) else float(np.clip(r, -0.999999, 0.999999))

    # Regress xi on S and xj on S using regularized least squares
    xs = data[:, list(s_indices)]
    n = data.shape[0]
    xs_augmented = np.column_stack([xs, np.ones(n)])

    # Ridge regularizer for multicollinearity
    ridge = 1e-6 * np.eye(xs_augmented.shape[1])
    try:
        beta_i = np.linalg.solve(xs_augmented.T @ xs_augmented + ridge, xs_augmented.T @ xi)
        res_i = xi - xs_augmented @ beta_i

        beta_j = np.linalg.solve(xs_augmented.T @ xs_augmented + ridge, xs_augmented.T @ xj)
        res_j = xj - xs_augmented @ beta_j

        std_ri = np.std(res_i)
        std_rj = np.std(res_j)
        if std_ri < 1e-8 or std_rj < 1e-8:
            return 0.0

        r = np.corrcoef(res_i, res_j)[0, 1]
        return 0.0 if np.isnan(r) else float(np.clip(r, -0.999999, 0.999999))
    except np.linalg.LinAlgError:
        return 0.0


def _test_conditional_independence(
    data: np.ndarray,
    i: int,
    j: int,
    s_indices: Tuple[int, ...],
    alpha: float,
) -> Tuple[bool, float, float]:
    """
    Tests conditional independence of X_i and X_j given S using Fisher's z-transform:
    z = 0.5 * ln((1 + r) / (1 - r))
    test_statistic = sqrt(N - |S| - 3) * |z|
    Returns: (is_independent, p_value, partial_correlation)
    """
    n = data.shape[0]
    cond_size = len(s_indices)
    df = n - cond_size - 3

    r = _partial_correlation(data, i, j, s_indices)

    if df <= 0:
        # Not enough degrees of freedom to reject independence
        return (abs(r) < 0.1, 1.0, r)

    # Fisher's z-transform
    r_clipped = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * math.log((1.0 + r_clipped) / (1.0 - r_clipped))
    stat = math.sqrt(df) * abs(z)

    # Two-sided p-value: 2 * (1 - normal_cdf(stat)) = erfc(stat / sqrt(2))
    p_value = float(erfc(stat / math.sqrt(2.0)))
    is_independent = bool(p_value > alpha)

    return is_independent, p_value, r


# ---------------------------------------------------------------------------
# PC Algorithm (Causal Discovery Engine)
# ---------------------------------------------------------------------------

class CausalDiscoveryEngine:
    """PC-algorithm causal graph induction with partial-correlation conditional independence."""

    def __init__(self, alpha: float = 0.05, max_features: int = 15, max_cond_size: int = 3):
        self.alpha = alpha
        self.max_features = max_features
        self.max_cond_size = max_cond_size

    def discover_graph(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        treatment_col: Optional[str] = None,
    ) -> CausalGraphResult:
        """Executes PC algorithm causal graph discovery."""
        clean_df, feature_names = _preprocess_tabular_data(
            df=df,
            target_col=target_col,
            treatment_col=treatment_col,
            max_features=self.max_features,
        )

        p = len(feature_names)
        if p == 0:
            return CausalGraphResult(
                nodes=[],
                edges=[],
                adjacency_matrix={},
                reverse_causality_risks=[],
                target_col=target_col,
            )

        if p == 1:
            node = feature_names[0]
            return CausalGraphResult(
                nodes=[node],
                edges=[],
                adjacency_matrix={node: {node: 0.0}},
                reverse_causality_risks=[],
                target_col=target_col,
                root_causes=[node],
                sink_nodes=[node],
            )

        data = clean_df.to_numpy(dtype=float)
        n = data.shape[0]

        # ---------------------------------------------------------
        # Step 1: Skeleton Discovery via Conditional Independence
        # ---------------------------------------------------------
        # Complete undirected graph adjacency: adj[i] = set of neighbors
        adj: Dict[int, Set[int]] = {i: set(range(p)) - {i} for i in range(p)}
        sepset: Dict[Tuple[int, int], Set[int]] = {}
        edge_stats: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Initial pairwise correlations
        for i in range(p):
            for j in range(i + 1, p):
                r = _partial_correlation(data, i, j, ())
                edge_stats[(i, j)] = (r, 1.0)
                edge_stats[(j, i)] = (r, 1.0)

        # Iteratively prune edges with increasing conditioning set size k
        max_k = min(self.max_cond_size, p - 2)
        for k in range(max_k + 1):
            if n - k - 3 <= 0:
                break

            # Collect current edges
            edges_to_test: List[Tuple[int, int]] = []
            for i in range(p):
                for j in adj[i]:
                    if i < j:
                        edges_to_test.append((i, j))

            for i, j in edges_to_test:
                if j not in adj[i]:
                    continue

                # Possible conditioning subsets from adj[i] \ {j} and adj[j] \ {i}
                cand_i = adj[i] - {j}
                cand_j = adj[j] - {i}

                candidates_to_try = []
                if len(cand_i) >= k:
                    candidates_to_try.append(cand_i)
                if len(cand_j) >= k and cand_j != cand_i:
                    candidates_to_try.append(cand_j)

                if not candidates_to_try:
                    continue

                # Search through subsets of size k
                found_indep = False
                for cand in candidates_to_try:
                    for s_tuple in combinations(cand, k):
                        is_indep, p_val, p_corr = _test_conditional_independence(
                            data=data,
                            i=i,
                            j=j,
                            s_indices=s_tuple,
                            alpha=self.alpha,
                        )
                        if is_indep:
                            adj[i].remove(j)
                            adj[j].remove(i)
                            s_set = set(s_tuple)
                            sepset[(i, j)] = s_set
                            sepset[(j, i)] = s_set
                            edge_stats[(i, j)] = (p_corr, p_val)
                            edge_stats[(j, i)] = (p_corr, p_val)
                            found_indep = True
                            break
                    if found_indep:
                        break

                if found_indep:
                    continue

        # ---------------------------------------------------------
        # Step 2: Collider (V-Structure) Orientation
        # For unshielded triple i - k - j where i and j are non-adjacent:
        # If k not in sepset(i, j): orient i -> k <- j
        # ---------------------------------------------------------
        # Directed graph represented as:
        # dir_edges: set of (u, v) where u -> v
        # undir_edges: set of frozenset({u, v})
        dir_edges: Set[Tuple[int, int]] = set()
        undir_edges: Set[frozenset[int]] = set()

        for i in range(p):
            for j in adj[i]:
                if i < j:
                    undir_edges.add(frozenset({i, j}))

        # V-structure detection
        for k in range(p):
            # Neighbors of k in current undirected skeleton
            neighbors = [nb for nb in adj[k]]
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    i = neighbors[idx1]
                    j = neighbors[idx2]
                    # Check if i and j are non-adjacent
                    if j not in adj[i]:
                        sep = sepset.get((i, j), set())
                        if k not in sep:
                            # Orient i -> k and j -> k
                            dir_edges.add((i, k))
                            dir_edges.add((j, k))
                            undir_edges.discard(frozenset({i, k}))
                            undir_edges.discard(frozenset({j, k}))

        # ---------------------------------------------------------
        # Step 3: Meek's Rules Orientation Propagation
        # ---------------------------------------------------------
        def is_adjacent(a: int, b: int) -> bool:
            return b in adj[a]

        def has_dir_edge(a: int, b: int) -> bool:
            return (a, b) in dir_edges

        def has_undir_edge(a: int, b: int) -> bool:
            return frozenset({a, b}) in undir_edges

        changed = True
        iterations = 0
        max_iter = p * p

        while changed and iterations < max_iter:
            changed = False
            iterations += 1

            # Meek Rule 1:
            # If a -> b and b - c, and a and c not adjacent: orient b -> c
            for b in range(p):
                # nodes pointing to b
                parents_of_b = [a for a in range(p) if has_dir_edge(a, b)]
                # nodes with undirected edge to b
                undir_neighbors_of_b = [c for c in range(p) if has_undir_edge(b, c)]

                for a in parents_of_b:
                    for c in undir_neighbors_of_b:
                        if not is_adjacent(a, c):
                            # Orient b -> c
                            dir_edges.add((b, c))
                            undir_edges.discard(frozenset({b, c}))
                            changed = True

            # Meek Rule 2:
            # If a -> b -> c and a - c: orient a -> c (to avoid directed cycles)
            for a in range(p):
                for b in range(p):
                    if has_dir_edge(a, b):
                        for c in range(p):
                            if has_dir_edge(b, c) and has_undir_edge(a, c):
                                dir_edges.add((a, c))
                                undir_edges.discard(frozenset({a, c}))
                                changed = True

            # Meek Rule 3:
            # If a - b and there exist c, d such that a - c -> b and a - d -> b,
            # with c, d non-adjacent: orient a -> b
            for a in range(p):
                for b in range(p):
                    if has_undir_edge(a, b):
                        # Find c such that has_undir_edge(a, c) and has_dir_edge(c, b)
                        c_candidates = [
                            c for c in range(p)
                            if c != a and c != b and has_undir_edge(a, c) and has_dir_edge(c, b)
                        ]
                        if len(c_candidates) >= 2:
                            for idx1 in range(len(c_candidates)):
                                for idx2 in range(idx1 + 1, len(c_candidates)):
                                    c = c_candidates[idx1]
                                    d = c_candidates[idx2]
                                    if not is_adjacent(c, d):
                                        dir_edges.add((a, b))
                                        undir_edges.discard(frozenset({a, b}))
                                        changed = True
                                        break
                                if changed:
                                    break

        # ---------------------------------------------------------
        # Step 4: Construct CausalEdge list and Adjacency Matrix
        # ---------------------------------------------------------
        edges: List[CausalEdge] = []
        matrix: Dict[str, Dict[str, float]] = {u: {v: 0.0 for v in feature_names} for u in feature_names}

        for u_idx, v_idx in dir_edges:
            u_name = feature_names[u_idx]
            v_name = feature_names[v_idx]
            corr_val, p_val = edge_stats.get((u_idx, v_idx), (0.0, 0.0))
            is_direct = bool(target_col and v_name == target_col)
            edges.append(
                CausalEdge(
                    source=u_name,
                    target=v_name,
                    weight=float(round(corr_val, 4)),
                    p_value=float(round(p_val, 5)),
                    direction="-->",
                    is_direct_cause_of_target=is_direct,
                )
            )
            matrix[u_name][v_name] = float(round(corr_val, 4))

        for pair in undir_edges:
            u_idx, v_idx = tuple(pair)
            u_name = feature_names[u_idx]
            v_name = feature_names[v_idx]
            corr_val, p_val = edge_stats.get((u_idx, v_idx), (0.0, 0.0))
            edges.append(
                CausalEdge(
                    source=u_name,
                    target=v_name,
                    weight=float(round(corr_val, 4)),
                    p_value=float(round(p_val, 5)),
                    direction="---",
                    is_direct_cause_of_target=False,
                )
            )
            matrix[u_name][v_name] = float(round(corr_val, 4))
            matrix[v_name][u_name] = float(round(corr_val, 4))

        # ---------------------------------------------------------
        # Step 5: Classify Causal Roles and Reverse Causality
        # ---------------------------------------------------------
        # Build directed graph for ancestry / path queries
        in_degree: Dict[str, int] = {node: 0 for node in feature_names}
        out_degree: Dict[str, int] = {node: 0 for node in feature_names}
        parents_map: Dict[str, List[str]] = {node: [] for node in feature_names}
        children_map: Dict[str, List[str]] = {node: [] for node in feature_names}

        for e in edges:
            if e.direction == "-->":
                out_degree[e.source] += 1
                in_degree[e.target] += 1
                parents_map[e.target].append(e.source)
                children_map[e.source].append(e.target)

        root_causes = [node for node in feature_names if in_degree[node] == 0 and out_degree[node] > 0]
        sink_nodes = [node for node in feature_names if out_degree[node] == 0 and in_degree[node] > 0]

        direct_causes: List[str] = []
        indirect_causes: List[str] = []
        confounders: List[str] = []
        reverse_causality_risks: List[str] = []

        if target_col and target_col in feature_names:
            # Direct causes = parents of target OR undirected CPDAG neighbors adjacent to target
            direct_causes = list(parents_map.get(target_col, []))
            for e in edges:
                if e.direction == "---":
                    if e.source == target_col and e.target not in direct_causes:
                        direct_causes.append(e.target)
                        e.is_direct_cause_of_target = True
                    elif e.target == target_col and e.source not in direct_causes:
                        direct_causes.append(e.source)
                        e.is_direct_cause_of_target = True

            # Indirect causes = ancestors that are not direct parents
            ancestors = set()
            queue = list(direct_causes)
            while queue:
                curr = queue.pop(0)
                for p_node in parents_map.get(curr, []):
                    if p_node not in ancestors and p_node != target_col:
                        ancestors.add(p_node)
                        queue.append(p_node)
            indirect_causes = [a for a in ancestors if a not in direct_causes]

            # Reverse causality risks: features that are children/descendants of the target!
            # i.e., target -> feature
            descendants = set()
            d_queue = list(children_map.get(target_col, []))
            while d_queue:
                curr = d_queue.pop(0)
                if curr not in descendants:
                    descendants.add(curr)
                    d_queue.extend(children_map.get(curr, []))
            reverse_causality_risks = sorted(list(descendants))

            # Confounders: nodes with directed paths to both a feature and the target
            for node in feature_names:
                if node != target_col and out_degree[node] >= 2:
                    node_children = set(children_map.get(node, []))
                    if target_col in node_children:
                        other_children = node_children - {target_col}
                        if other_children:
                            confounders.append(node)

        return CausalGraphResult(
            nodes=feature_names,
            edges=edges,
            adjacency_matrix=matrix,
            reverse_causality_risks=reverse_causality_risks,
            target_col=target_col,
            root_causes=root_causes,
            direct_causes_of_target=direct_causes,
            indirect_causes=indirect_causes,
            confounders=confounders,
            sink_nodes=sink_nodes,
        )


# ---------------------------------------------------------------------------
# Pearl's Do-Calculus Policy Simulator
# ---------------------------------------------------------------------------

class DoCalculusSimulator:
    """Pearl's Do-Calculus policy simulator for estimating E[Y | do(X = x)]."""

    def __init__(self, graph_result: Optional[CausalGraphResult] = None):
        self.graph = graph_result

    def simulate_intervention(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        intervention_value: float,
        n_bootstrap: int = 50,
    ) -> InterventionResult:
        """
        Simulates Pearl's Do-Calculus policy intervention on observational data.
        E[Y | do(X = x)] = sum_z E[Y | X=x, Z=z] P(Z=z) via back-door adjustment.
        """
        if df.empty:
            raise ValueError("Input DataFrame for causal intervention simulation is empty.")

        if treatment not in df.columns or outcome not in df.columns:
            raise ValueError(
                f"Treatment '{treatment}' or outcome '{outcome}' not found in DataFrame columns."
            )

        # Baseline expected outcome
        y_raw = pd.to_numeric(df[outcome], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        baseline = float(np.mean(y_raw))

        if treatment == outcome:
            return InterventionResult(
                treatment=treatment,
                outcome=outcome,
                intervention_value=intervention_value,
                baseline_expected_outcome=baseline,
                intervened_expected_outcome=baseline,
                average_treatment_effect=0.0,
                adjustment_set=[],
                treatment_col=treatment,
                treatment_value=intervention_value,
                outcome_col=outcome,
                counterfactual_expected_outcome=baseline,
                average_causal_effect=0.0,
                percent_change=0.0,
                backdoor_adjustment_set=[],
                confidence_interval_95=(0.0, 0.0),
                confidence_interval_lower=0.0,
                confidence_interval_upper=0.0,
                is_reverse_causality=False,
                policy_interpretation="Treatment and outcome are identical; intervention has 0 effect.",
            )

        # Ensure we have a causal graph to determine back-door adjustment set
        graph = self.graph
        if graph is None:
            engine = CausalDiscoveryEngine(alpha=0.05)
            graph = engine.discover_graph(df=df, target_col=outcome, treatment_col=treatment)

        # ---------------------------------------------------------
        # Check Reverse Causality
        # ---------------------------------------------------------
        # Build children graph to verify if outcome is ancestor of treatment
        children_map: Dict[str, List[str]] = {n: [] for n in graph.nodes}
        for e in graph.edges:
            if e.direction == "-->":
                children_map.setdefault(e.source, []).append(e.target)

        # Check if outcome causes treatment: path from outcome to treatment
        is_reverse = False
        queue = list(children_map.get(outcome, []))
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr == treatment:
                is_reverse = True
                break
            if curr not in visited:
                visited.add(curr)
                queue.extend(children_map.get(curr, []))

        # ---------------------------------------------------------
        # Determine Back-Door Adjustment Set Z
        # ---------------------------------------------------------
        # In a causal DAG, Parents(treatment) blocks all back-door paths,
        # provided no parent is a descendant of treatment.
        # Check parents of treatment
        parents_of_x: List[str] = []
        for e in graph.edges:
            if e.direction == "-->" and e.target == treatment and e.source != outcome:
                parents_of_x.append(e.source)
            elif e.direction == "---" and (e.source == treatment or e.target == treatment):
                # For undirected edges, include the connected node if present in df
                other = e.target if e.source == treatment else e.source
                if other != outcome and other not in parents_of_x:
                    parents_of_x.append(other)

        # Filter adjustment set to columns present in df
        adjustment_set = [z for z in parents_of_x if z in df.columns and z != treatment and z != outcome]

        # ---------------------------------------------------------
        # Outcome Modeling & Counterfactual Prediction
        # ---------------------------------------------------------
        features_to_use = [treatment] + adjustment_set
        sub_df = df[features_to_use + [outcome]].copy()
        for col in sub_df.columns:
            sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
        sub_df = sub_df.fillna(sub_df.median(numeric_only=True).fillna(0.0))

        X_mat = sub_df[features_to_use].to_numpy(dtype=float)
        y_vec = sub_df[outcome].to_numpy(dtype=float)
        n = X_mat.shape[0]

        # Fit linear outcome model with ridge regularization: y = W @ X + b
        X_aug = np.column_stack([X_mat, np.ones(n)])
        p_dim = X_aug.shape[1]
        ridge_diag = 1e-5 * np.eye(p_dim)
        ridge_diag[-1, -1] = 0.0  # Do not regularize intercept

        try:
            weights = np.linalg.solve(X_aug.T @ X_aug + ridge_diag, X_aug.T @ y_vec)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(X_aug, y_vec, rcond=None)[0]

        # Generate counterfactual data where treatment is set to intervention_value
        X_cf = X_mat.copy()
        X_cf[:, 0] = float(intervention_value)
        X_cf_aug = np.column_stack([X_cf, np.ones(n)])

        # E[Y | do(X = x)] = (1/N) * sum_i f(x, z_i)
        y_cf_pred = X_cf_aug @ weights
        cf_expected = float(np.mean(y_cf_pred))

        if is_reverse:
            # If reverse causality, intervening on X has no causal impact on Y!
            ate = 0.0
            cf_expected = baseline
            pct_change = 0.0
            interpretation = (
                f"REVERSE CAUSALITY DETECTED: Outcome '{outcome}' causally influences '{treatment}'. "
                f"Policy intervention do({treatment} = {intervention_value:.2f}) provides zero causal leverage on '{outcome}'."
            )
            ci_lower = 0.0
            ci_upper = 0.0
        else:
            ate = float(cf_expected - baseline)
            pct_change = float((ate / (abs(baseline) + 1e-8)) * 100.0)

            direction_str = "increase" if ate > 0 else "decrease"
            interpretation = (
                f"Intervention do({treatment} = {intervention_value:.2f}) is estimated to {direction_str} "
                f"expected '{outcome}' from {baseline:.2f} to {cf_expected:.2f} "
                f"(ATE: {ate:+.3f}, {pct_change:+.1f}%), controlling for back-door confounders: {adjustment_set or 'none'}."
            )

            # Bootstrap 95% confidence intervals
            rng = np.random.default_rng(seed=42)
            boot_ates: List[float] = []
            for _ in range(max(10, n_bootstrap)):
                boot_idx = rng.choice(n, size=n, replace=True)
                X_b = X_aug[boot_idx]
                y_b = y_vec[boot_idx]
                try:
                    w_b = np.linalg.solve(X_b.T @ X_b + ridge_diag, X_b.T @ y_b)
                except np.linalg.LinAlgError:
                    w_b = weights
                y_pred_b = X_cf_aug[boot_idx] @ w_b
                boot_ates.append(float(np.mean(y_pred_b) - np.mean(y_b)))

            ci_lower = float(np.percentile(boot_ates, 2.5))
            ci_upper = float(np.percentile(boot_ates, 97.5))

        return InterventionResult(
            treatment=treatment,
            outcome=outcome,
            intervention_value=intervention_value,
            baseline_expected_outcome=round(baseline, 4),
            intervened_expected_outcome=round(cf_expected, 4),
            average_treatment_effect=round(ate, 4),
            adjustment_set=adjustment_set,
            treatment_col=treatment,
            treatment_value=intervention_value,
            outcome_col=outcome,
            counterfactual_expected_outcome=round(cf_expected, 4),
            average_causal_effect=round(ate, 4),
            percent_change=round(pct_change, 2),
            backdoor_adjustment_set=adjustment_set,
            confidence_interval_95=(round(ci_lower, 4), round(ci_upper, 4)),
            confidence_interval_lower=round(ci_lower, 4),
            confidence_interval_upper=round(ci_upper, 4),
            is_reverse_causality=is_reverse,
            policy_interpretation=interpretation,
        )


# ---------------------------------------------------------------------------
# High-Level Functional API
# ---------------------------------------------------------------------------

def discover_causal_graph(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    alpha: float = 0.05,
    treatment_col: Optional[str] = None,
) -> CausalGraphResult:
    """
    Discovers observational causal graph from tabular data using the PC algorithm.
    """
    engine = CausalDiscoveryEngine(alpha=alpha)
    return engine.discover_graph(df=df, target_col=target_col, treatment_col=treatment_col)


def simulate_intervention(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    intervention_value: float,
    graph: Optional[CausalGraphResult] = None,
) -> InterventionResult:
    """
    Simulates Pearl's Do-Calculus intervention E[Y | do(X = x)] using back-door adjustment.
    """
    simulator = DoCalculusSimulator(graph_result=graph)
    return simulator.simulate_intervention(
        df=df,
        treatment=treatment,
        outcome=outcome,
        intervention_value=intervention_value,
    )
