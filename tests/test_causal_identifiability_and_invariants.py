"""
tests/test_causal_identifiability_and_invariants.py
───────────────────────────────────────────────────
Axiomatic Structural Causal Identifiability & Conservation Invariant Test Suite.

Verifies:
1. Exact Graph-Theoretic d-Separation (Chains, Forks, Colliders, Descendants of Colliders).
2. Pearl's Backdoor Criterion Identification & Minimal Adjustment Set Discovery.
3. Structural Causal Resolution of Simpson's Paradox (Confounder vs Collider vs Mediator).
4. Linear Conservation Invariant Projection (Ax = b holds to machine epsilon).
5. Exact Probability Simplex Projection (Duchi et al., 2008: sum(p) == 1.0, p >= 0).
6. Monotonicity Invariant Enforcement via PAVA.
7. Axiomatic Data Contract Auditor across DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dia.causal_identifiability import (
    CausalDAG,
    find_minimal_backdoor_set,
    is_d_separated,
    resolve_simpsons_paradox_algebraically,
    verify_backdoor_criterion,
)
from dia.invariant_engine import (
    ConservationLaw,
    ConservationProjector,
    audit_dataframe_axioms,
    enforce_monotonicity_pava,
    project_to_probability_simplex,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Exact d-Separation Invariant Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExactDSeparation:
    def test_chain_topology_d_separation(self) -> None:
        """
        Chain: A -> B -> C
        A and C are dependent given empty set.
        Conditioning on intermediate node B d-separates A and C: (A _||_ C | B).
        """
        dag = CausalDAG()
        dag.add_edge("A", "B")
        dag.add_edge("B", "C")

        assert not is_d_separated(dag, "A", "C", z=None)
        assert is_d_separated(dag, "A", "C", z={"B"})

    def test_fork_topology_confounding_d_separation(self) -> None:
        """
        Fork: A <- B -> C (B is common cause / confounder)
        A and C are dependent given empty set (spurious correlation).
        Conditioning on common cause B d-separates A and C: (A _||_ C | B).
        """
        dag = CausalDAG()
        dag.add_edge("B", "A")
        dag.add_edge("B", "C")

        assert not is_d_separated(dag, "A", "C", z=None)
        assert is_d_separated(dag, "A", "C", z={"B"})

    def test_collider_v_structure_berksons_paradox_invariant(self) -> None:
        """
        Collider: A -> B <- C
        A and C are marginal independent: (A _||_ C | empty).
        Conditioning on collider B OPENS the path, inducing dependence (Berkson's bias):
        not (A _||_ C | B).
        """
        dag = CausalDAG()
        dag.add_edge("A", "B")
        dag.add_edge("C", "B")

        # Independent unconditionally
        assert is_d_separated(dag, "A", "C", z=None)
        assert is_d_separated(dag, "A", "C", z=set())

        # Conditioned on collider opens path
        assert not is_d_separated(dag, "A", "C", z={"B"})

    def test_descendant_of_collider_opens_path(self) -> None:
        """
        A -> B <- C
             |
             v
             D
        Conditioning on descendant D of collider B also activates the path:
        not (A _||_ C | D).
        """
        dag = CausalDAG()
        dag.add_edge("A", "B")
        dag.add_edge("C", "B")
        dag.add_edge("B", "D")

        assert is_d_separated(dag, "A", "C", z=None)
        assert not is_d_separated(dag, "A", "C", z={"D"})


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pearl's Backdoor Criterion & Minimal Adjustment Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBackdoorCriterion:
    def test_backdoor_admissibility_and_descendant_violation(self) -> None:
        """
        DAG:
          Z -> X -> Y
          Z -> Y
          X -> M -> Y
        Z is a confounder. M is a mediator.
        - Set {Z} is admissible for estimating causal effect of X on Y.
        - Set {Z, M} violates backdoor criterion (M is a descendant of X).
        """
        dag = CausalDAG()
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "M")
        dag.add_edge("M", "Y")

        res_z = verify_backdoor_criterion(dag, treatment="X", outcome="Y", candidate_z={"Z"})
        assert res_z.is_admissible
        assert res_z.minimal_adjustment_set == ["Z"]
        assert len(res_z.violations) == 0

        # Adding mediator M violates descendant condition
        res_zm = verify_backdoor_criterion(dag, treatment="X", outcome="Y", candidate_z={"Z", "M"})
        assert not res_zm.is_admissible
        assert any("descendants" in v.lower() for v in res_zm.violations)

    def test_find_minimal_backdoor_adjustment_set(self) -> None:
        """
        DAG with redundant non-confounder W:
          Z -> X -> Y
          Z -> Y
          W (unconnected or independent)
        Minimal adjustment set must prune W and return only [Z].
        """
        dag = CausalDAG()
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")
        dag.add_node("W")

        superset = {"Z", "W"}
        minimal = find_minimal_backdoor_set(dag, treatment="X", outcome="Y", admissible_superset=superset)
        assert set(minimal) == {"Z"}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Simpson's Paradox Structural Causal Resolution Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSimpsonsStructuralResolution:
    def test_confounder_resolution_verdict(self) -> None:
        """Z causes both X and Y. True effect requires conditioning on Z."""
        dag = CausalDAG()
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")

        res = resolve_simpsons_paradox_algebraically(dag, treatment="X", outcome="Y", conditioning_node="Z")
        assert res.structural_role == "Confounder"
        assert "subgroup" in res.causal_verdict.lower()
        assert "condition on 'z'" in res.recommended_action.lower()

    def test_collider_resolution_verdict(self) -> None:
        """Z is caused by both X and Y. Conditioning on Z is Berkson's bias."""
        dag = CausalDAG()
        dag.add_edge("X", "Z")
        dag.add_edge("Y", "Z")
        dag.add_edge("X", "Y")

        res = resolve_simpsons_paradox_algebraically(dag, treatment="X", outcome="Y", conditioning_node="Z")
        assert res.structural_role == "Collider"
        assert "berkson" in res.causal_verdict.lower()
        assert "do not condition" in res.recommended_action.lower()

    def test_mediator_resolution_verdict(self) -> None:
        """Z is an intermediate mediator X -> Z -> Y."""
        dag = CausalDAG()
        dag.add_edge("X", "Z")
        dag.add_edge("Z", "Y")

        res = resolve_simpsons_paradox_algebraically(dag, treatment="X", outcome="Y", conditioning_node="Z")
        assert res.structural_role == "Mediator"
        assert "mediates" in res.causal_verdict.lower()
        assert "do not condition" in res.recommended_action.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Axiomatic Conservation & Manifold Invariants Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAxiomaticConservationInvariants:
    def test_linear_conservation_projection_to_machine_precision(self) -> None:
        """
        Physical balance law: Inflow_A + Inflow_B - Outflow_C = 0.
        Candidate vector: [100.0, 50.0, 80.0] (Residual = 70.0 != 0).
        Projected vector must satisfy law to machine precision.
        """
        law = ConservationLaw(
            name="Mass Balance",
            coefficients={"inflow_a": 1.0, "inflow_b": 1.0, "outflow_c": -1.0},
            target_sum=0.0,
            tolerance=1e-9,
        )
        features = ["inflow_a", "inflow_b", "outflow_c"]
        projector = ConservationProjector(laws=[law], feature_names=features)

        unconstrained_x = np.array([100.0, 50.0, 80.0])
        projected_x = projector.project(unconstrained_x)

        # Inflow_A + Inflow_B - Outflow_C must equal 0.0
        lhs = projected_x[0] + projected_x[1] - projected_x[2]
        assert abs(lhs) < 1e-12

        # Check verification report
        checks = projector.verify_conservation(projected_x)
        assert checks[0][2] is True
        assert checks[0][1] < 1e-12

    def test_probability_simplex_projection(self) -> None:
        """
        Candidate non-probability vector: [-0.5, 2.0, 0.3, 0.0, 1.5].
        Projected vector must lie exactly on simplex: sum(p) == 1.0, all p >= 0.
        """
        v = np.array([-0.5, 2.0, 0.3, 0.0, 1.5])
        p = project_to_probability_simplex(v)

        assert abs(float(np.sum(p)) - 1.0) < 1e-12
        assert np.all(p >= 0.0)
        # Monotonicity of rank must be preserved: larger unconstrained values get larger probabilities
        assert p[1] >= p[4] >= p[2] >= p[3] >= p[0]

    def test_pava_monotonicity_enforcement(self) -> None:
        """
        Candidate non-monotonic series: [1.0, 3.0, 2.0, 5.0, 4.0, 6.0].
        PAVA must enforce non-decreasing monotonic sequence.
        """
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
        y_mono = enforce_monotonicity_pava(y, increasing=True)

        # Check monotonicity: y_mono[i] <= y_mono[i+1]
        diffs = np.diff(y_mono)
        assert np.all(diffs >= -1e-12)

    def test_audit_dataframe_conservation_axioms(self) -> None:
        """
        Verifies batch DataFrame auditing against conservation axioms.
        Detects violating rows without false positives.
        """
        df = pd.DataFrame({
            "inflow": [10.0, 20.0, 30.0, 40.0],
            "outflow": [10.0, 15.0, 30.0, 50.0],
            "delta": [0.0, 5.0, 0.0, -10.0],
        })
        # Law: inflow - outflow - delta = 0
        law = ConservationLaw(
            name="Cashflow Consistency",
            coefficients={"inflow": 1.0, "outflow": -1.0, "delta": -1.0},
            target_sum=0.0,
            tolerance=1e-6,
        )

        audit = audit_dataframe_axioms(df, [law])
        assert audit.is_axiom_compliant
        assert audit.passed_records == 4
        assert audit.violation_rate == 0.0

        # Inject 1 corrupt row that breaks conservation
        df_corrupt = df.copy()
        df_corrupt.loc[1, "delta"] = 999.0  # Inflow 20 - Outflow 15 != 999
        audit_corrupt = audit_dataframe_axioms(df_corrupt, [law])
        assert not audit_corrupt.is_axiom_compliant
        assert audit_corrupt.violations_by_law["Cashflow Consistency"] == 1
        assert audit_corrupt.violation_rate == 0.25
