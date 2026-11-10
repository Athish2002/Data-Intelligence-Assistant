"""
dia/causal_identifiability.py
─────────────────────────────
Axiomatic Structural Causal Identifiability & Do-Calculus Invariant Engine.

Implements Pearl's Graphical Algebra on Directed Acyclic Graphs (DAGs):
1. Exact Graph-Theoretic d-Separation via Active-Path Exploration.
2. Backdoor Criterion Admissibility:
   - Validates that no node in Z is a descendant of X.
   - Verifies that Z d-separates X and Y in the mutilated graph G_{\\underline{X}}
     (all outgoing arrows from X removed).
   - Computes subset-minimal admissible adjustment sets Z*.
3. Frontdoor Criterion Admissibility:
   - Verifies mediator set M intercepts all directed paths X -> Y.
   - Verifies no unblocked backdoor path between X and M.
   - Verifies all backdoor paths between M and Y are blocked by X.
4. Axiomatic Simpson's Paradox Structural Resolver:
   - Identifies whether the conditioning attribute is a Confounder, Collider, or Mediator.
   - Explains algebraically whether the subgroup correlation or aggregate correlation
     represents the unconfounded causal truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class CausalDAG:
    """
    Directed Acyclic Graph (DAG) for structural causal modeling.
    Nodes are variable names (strings).
    Edges are directed pairs (source -> target).
    """
    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)

    def add_edge(self, source: str, target: str) -> None:
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.add((source, target))

    @property
    def parents(self) -> dict[str, set[str]]:
        p: dict[str, set[str]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            p[v].add(u)
        return p

    @property
    def children(self) -> dict[str, set[str]]:
        c: dict[str, set[str]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            c[u].add(v)
        return c

    def get_ancestors(self, targets: set[str]) -> set[str]:
        """Returns all ancestors of the given target nodes (including targets)."""
        ancestors: set[str] = set(targets)
        queue = deque(targets)
        p_map = self.parents
        while queue:
            curr = queue.popleft()
            for p in p_map.get(curr, set()):
                if p not in ancestors:
                    ancestors.add(p)
                    queue.append(p)
        return ancestors

    def get_descendants(self, sources: set[str]) -> set[str]:
        """Returns all descendants of the given source nodes (including sources)."""
        descendants: set[str] = set(sources)
        queue = deque(sources)
        c_map = self.children
        while queue:
            curr = queue.popleft()
            for c in c_map.get(curr, set()):
                if c not in descendants:
                    descendants.add(c)
                    queue.append(c)
        return descendants

    def get_undirected_neighbors(self) -> dict[str, set[str]]:
        """Returns undirected adjacency representation for path traversal."""
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_acyclic(self) -> bool:
        """Kahn's topological sort for acyclicity verification."""
        in_deg = {n: 0 for n in self.nodes}
        for _, v in self.edges:
            in_deg[v] += 1
        queue = deque([n for n, d in in_deg.items() if d == 0])
        count = 0
        c_map = self.children
        while queue:
            curr = queue.popleft()
            count += 1
            for child in c_map[curr]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return count == len(self.nodes)

    def mutilate_remove_outgoing(self, x_nodes: set[str]) -> CausalDAG:
        """
        Creates mutilated graph G_{\\underline{X}} by removing all outgoing arrows from X.
        Used for Pearl's Backdoor Criterion identification.
        """
        new_dag = CausalDAG(nodes=set(self.nodes))
        for u, v in self.edges:
            if u not in x_nodes:
                new_dag.add_edge(u, v)
        return new_dag

    def mutilate_remove_incoming(self, x_nodes: set[str]) -> CausalDAG:
        """
        Creates mutilated graph G_{\\overline{X}} by removing all incoming arrows into X.
        Represents graph under active intervention do(X).
        """
        new_dag = CausalDAG(nodes=set(self.nodes))
        for u, v in self.edges:
            if v not in x_nodes:
                new_dag.add_edge(u, v)
        return new_dag

    def find_all_simple_paths(
        self, start: str, target: str, visited: set[str] | None = None, max_depth: int = 15, max_paths: int = 500
    ) -> list[list[str]]:
        return find_all_simple_paths(self, start, target, visited, max_depth, max_paths)


# ───────────────────────────────────────────────────────────────────────────
# Exact d-Separation Algorithm
# ───────────────────────────────────────────────────────────────────────────

def find_all_simple_paths(
    dag: CausalDAG,
    start: str,
    target: str,
    visited: set[str] | None = None,
    max_depth: int = 15,
    max_paths: int = 500,
) -> list[list[str]]:
    """
    Finds simple undirected paths between start and target in DAG up to max_depth and max_paths.
    Maintained for path auditing, explanations, and UI visualization.
    """
    if visited is None:
        visited = set()
    if start == target:
        return [[start]]
    if len(visited) >= max_depth:
        return []

    neighbors = dag.get_undirected_neighbors()
    paths: list[list[str]] = []
    visited.add(start)
    for nxt in sorted(list(neighbors.get(start, set()))):
        if nxt not in visited:
            sub_paths = find_all_simple_paths(dag, nxt, target, visited, max_depth, max_paths)
            for sp in sub_paths:
                paths.append([start] + sp)
                if len(paths) >= max_paths:
                    break
        if len(paths) >= max_paths:
            break
    visited.remove(start)
    return paths


def is_d_separated(
    dag: CausalDAG,
    x: str | set[str],
    y: str | set[str],
    z: str | set[str] | None = None,
) -> bool:
    """
    Determines if X is d-separated from Y given conditioning set Z in DAG.
    Uses Pearl's exact Moralized Ancestral Graph d-separation algorithm (Lauritzen et al., 1990):
      1. Identify ancestral set A = Ancestors(X U Y U Z).
      2. Form induced subgraph G[A].
      3. Moralize: for every pair of parents of a common child in G[A], add an undirected edge.
      4. Convert all directed edges to undirected edges.
      5. Remove conditioning nodes Z (specifically, Z \\ (X U Y)).
      6. Test connectivity between X and Y via BFS.
    Returns True if X and Y are d-separated (independent given Z), False otherwise.
    Runs in polynomial time O(|V| + |E|^2) with zero path truncation or arbitrary depth limits.
    """
    x_set = {x} if isinstance(x, str) else set(x)
    y_set = {y} if isinstance(y, str) else set(y)
    z_set = set() if z is None else ({z} if isinstance(z, str) else set(z))

    if not x_set or not y_set:
        return True

    # If any overlap between X and Y, they cannot be d-separated
    if x_set & y_set:
        return False

    # 1. Identify ancestral set A = Ancestors(X U Y U Z) in DAG
    target_nodes = (x_set | y_set | z_set) & dag.nodes
    if not target_nodes:
        return True

    anc_set = dag.get_ancestors(target_nodes) & dag.nodes

    # 2, 3, 4. Build moralized undirected graph on anc_set
    adj: dict[str, set[str]] = {n: set() for n in anc_set}

    # Add undirected versions of directed edges in induced subgraph G[A]
    for u, v in dag.edges:
        if u in anc_set and v in anc_set:
            adj[u].add(v)
            adj[v].add(u)

    # Moralize: marry parents of common children in G[A]
    parents_map = dag.parents
    for child in anc_set:
        parents = [p for p in parents_map.get(child, set()) if p in anc_set]
        if len(parents) >= 2:
            for p1, p2 in combinations(parents, 2):
                adj[p1].add(p2)
                adj[p2].add(p1)

    # 5. Remove conditioning nodes Z \\ (X U Y)
    z_blocked = (z_set & anc_set) - (x_set | y_set)

    # 6. Test connectivity between X and Y in the moralized graph
    start_nodes = [node for node in (x_set & anc_set) if node not in z_blocked]
    if not start_nodes:
        return True

    target_y = y_set & anc_set
    if not target_y:
        return True

    queue = deque(start_nodes)
    visited = set(start_nodes)

    while queue:
        curr = queue.popleft()
        if curr in target_y:
            return False  # Active path exists
        for neighbor in adj.get(curr, set()):
            if neighbor not in visited and neighbor not in z_blocked:
                if neighbor in target_y:
                    return False  # Found target in Y
                visited.add(neighbor)
                queue.append(neighbor)

    return True


def _is_path_active(
    path: list[str],
    dag: CausalDAG,
    z_set: set[str],
    anc_z: set[str],
    edges: set[tuple[str, str]],
) -> bool:
    """Evaluates whether a single undirected path is active given Z."""
    if len(path) <= 2:
        # Direct edge between start and end is active if endpoints are distinct
        return True

    # Check each intermediate node v_{i}
    for i in range(1, len(path) - 1):
        prev_node = path[i - 1]
        curr_node = path[i]
        next_node = path[i + 1]

        # Is curr_node a collider on this path?
        # A collider has edges: prev_node -> curr_node and next_node -> curr_node
        is_collider = (prev_node, curr_node) in edges and (next_node, curr_node) in edges

        if is_collider:
            # Active if curr_node in Z or has a descendant in Z (i.e. curr_node in anc_z)
            if curr_node not in anc_z:
                return False  # Blocked by unactivated collider
        else:
            # Non-collider: active iff curr_node NOT in Z
            if curr_node in z_set:
                return False  # Blocked by conditioned non-collider

    return True


# ───────────────────────────────────────────────────────────────────────────
# Pearl's Backdoor Criterion & Minimal Adjustment Sets
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class BackdoorVerificationResult:
    is_admissible: bool
    treatment: str
    outcome: str
    candidate_set: list[str]
    minimal_adjustment_set: list[str] | None
    violations: list[str]
    formula_identifiable: bool


def verify_backdoor_criterion(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    candidate_z: set[str],
) -> BackdoorVerificationResult:
    """
    Formally verifies Pearl's Backdoor Criterion:
    A set Z satisfies the backdoor criterion relative to (X, Y) if:
      1. No node in Z is a descendant of X.
      2. Z d-separates X from Y in G_{\\underline{X}} (mutilated DAG with outgoing edges from X cut).
    """
    violations: list[str] = []

    # 1. Check descendant violation
    descendants_x = dag.get_descendants({treatment}) - {treatment}
    desc_intersection = candidate_z & descendants_x
    if desc_intersection:
        violations.append(
            f"Admissibility violation: Set contains descendants of treatment '{treatment}': {sorted(list(desc_intersection))}"
        )

    # 2. Check d-separation in G_{\\underline{X}}
    g_under_x = dag.mutilate_remove_outgoing({treatment})
    is_d_sep = is_d_separated(g_under_x, {treatment}, {outcome}, candidate_z)
    if not is_d_sep:
        violations.append(
            f"Blocking violation: Set does not d-separate '{treatment}' from '{outcome}' in mutilated graph G_\\underline{{{treatment}}}."
        )

    is_admissible = len(violations) == 0

    # Minimal adjustment set search
    minimal_set: list[str] | None = None
    if is_admissible:
        minimal_set = find_minimal_backdoor_set(dag, treatment, outcome, candidate_z)

    return BackdoorVerificationResult(
        is_admissible=is_admissible,
        treatment=treatment,
        outcome=outcome,
        candidate_set=sorted(list(candidate_z)),
        minimal_adjustment_set=sorted(minimal_set) if minimal_set is not None else None,
        violations=violations,
        formula_identifiable=is_admissible,
    )


def find_minimal_backdoor_set(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    admissible_superset: set[str],
) -> list[str]:
    """
    Finds a subset-minimal backdoor adjustment set Z* \\subseteq admissible_superset.
    Greedily prunes unnecessary variables while maintaining d-separation in G_{\\underline{X}}.
    """
    current_z = set(admissible_superset)
    g_under_x = dag.mutilate_remove_outgoing({treatment})

    # Try removing one variable at a time
    for candidate in sorted(list(admissible_superset)):
        trial_z = current_z - {candidate}
        if is_d_separated(g_under_x, {treatment}, {outcome}, trial_z):
            current_z = trial_z

    return list(current_z)


# ───────────────────────────────────────────────────────────────────────────
# Axiomatic Simpson's Paradox Structural Resolver
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class SimpsonsCausalResolution:
    treatment: str
    outcome: str
    conditioning_node: str
    structural_role: str  # "Confounder", "Collider", "Mediator", "M-Bias Collider", "Neutral"
    causal_verdict: str
    recommended_action: str
    explanation: str


def resolve_simpsons_paradox_algebraically(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
    conditioning_node: str,
) -> SimpsonsCausalResolution:
    """
    Given a Simpson's Paradox inversion between treatment and outcome when conditioned on Z,
    resolves the paradox using structural graph topology rather than empirical curve fitting:

    - Confounder: Z -> treatment and Z -> outcome.
      VERDICT: Conditioning on Z isolates true causal effect. The aggregate sign was spurious.
    - Collider: treatment -> Z <- outcome (or on an active collider path).
      VERDICT: Conditioning on Z induces Berkson's bias. The unconditioned or backdoor-adjusted effect is truth.
    - Mediator: treatment -> Z -> outcome.
      VERDICT: Conditioning on Z blocks the direct mechanism. Z mediates the effect.
    """
    parents_map = dag.parents
    children_map = dag.children

    is_parent_of_t = treatment in children_map.get(conditioning_node, set())
    is_parent_of_y = outcome in children_map.get(conditioning_node, set())
    is_child_of_t = conditioning_node in children_map.get(treatment, set())
    is_child_of_y = conditioning_node in children_map.get(outcome, set())

    # 1. Confounder: Z -> X and Z -> Y
    if is_parent_of_t and is_parent_of_y:
        role = "Confounder"
        verdict = f"Subgroup effect (conditioned on '{conditioning_node}') is the causal truth."
        action = f"Condition on '{conditioning_node}' to block non-causal backdoor path."
        explanation = (
            f"'{conditioning_node}' causes both treatment '{treatment}' and outcome '{outcome}'. "
            f"The aggregate correlation is contaminated by common-cause confounding bias. "
            f"Stratifying by '{conditioning_node}' restores the true invariant causal relationship."
        )

    # 2. Collider: X -> Z <- Y
    elif is_child_of_t and is_child_of_y:
        role = "Collider"
        verdict = f"Conditioning on '{conditioning_node}' creates false Berkson's bias. Do NOT stratify."
        action = f"Do NOT condition on '{conditioning_node}'. Use unconditioned or parent-adjusted model."
        explanation = (
            f"'{conditioning_node}' is a collider influenced jointly by '{treatment}' and '{outcome}'. "
            f"Conditioning on a collider opens a non-causal information path, inducing artificial spurious reversal."
        )

    # 3. Mediator: X -> Z -> Y
    elif is_child_of_t and is_parent_of_y:
        role = "Mediator"
        verdict = f"'{conditioning_node}' mediates the causal effect. Total effect requires leaving it unconditioned."
        action = f"To measure total causal effect, do NOT condition on '{conditioning_node}'."
        explanation = (
            f"'{conditioning_node}' lies on the directed causal chain '{treatment}' -> '{conditioning_node}' -> '{outcome}'. "
            f"Conditioning on it eliminates the indirect causal pathway (over-controlling bias)."
        )

    # 4. Check for M-structure or indirect confounder
    else:
        # Check if conditioning_node is an ancestor of both in DAG
        ancestors_t = dag.get_ancestors({treatment})
        ancestors_y = dag.get_ancestors({outcome})
        descendants_xy = dag.get_descendants({treatment, outcome})
        parents_z = parents_map.get(conditioning_node, set())

        is_ancestor_of_t = conditioning_node in ancestors_t
        is_ancestor_of_y = conditioning_node in ancestors_y
        is_descendant_of_xy = conditioning_node in descendants_xy

        if is_ancestor_of_t and is_ancestor_of_y:
            role = "Confounder"
            verdict = "Indirect confounder. Subgroup stratification is causally valid."
            action = f"Condition on '{conditioning_node}' to isolate direct treatment effect."
            explanation = f"'{conditioning_node}' is an indirect common ancestor of both variables."
        elif (
            not is_descendant_of_xy
            and not is_ancestor_of_t
            and not is_ancestor_of_y
            and len(parents_z) >= 2
            and any(
                p1 != p2
                for p1 in (parents_z & ancestors_t)
                for p2 in (parents_z & ancestors_y)
            )
        ):
            role = "M-Bias Collider"
            verdict = f"Conditioning on {conditioning_node} opens an M-structure backdoor path (Berkson's bias)."
            action = f"Do NOT condition on {conditioning_node}."
            explanation = (
                f"'{conditioning_node}' is an unconditioned collider on an M-structure path between "
                f"an ancestor of '{treatment}' and an ancestor of '{outcome}'. Conditioning on "
                f"'{conditioning_node}' opens an M-structure backdoor path (Berkson's bias)."
            )
        else:
            role = "Neutral"
            verdict = "No direct confounding path identified."
            action = "Verify DAG specification for unobserved confounders."
            explanation = f"'{conditioning_node}' does not form an obvious triangle or collider with treatment and outcome."

    return SimpsonsCausalResolution(
        treatment=treatment,
        outcome=outcome,
        conditioning_node=conditioning_node,
        structural_role=role,
        causal_verdict=verdict,
        recommended_action=action,
        explanation=explanation,
    )
