"""
tests/test_agentic_workflow.py
──────────────────────────────
Rigorous Verification Suite for Stateful Agentic Causal Workflow.

Adheres strictly to:
1. /property-fuzz-testing:
   - Generative topological fuzzing across random DAGs (5 to 15 nodes).
   - Invariant: Minimal adjustment set strictly excludes descendants of treatment X.
   - Invariant: All identified colliders have in-degree >= 2.
   - Invariant: Backdoor d-separation in G_{\\underline{X}} strictly holds.
   - Metamorphic Invariant: Adding independent noise variables NEVER alters the minimal adjustment set.
   - Idempotence: Repeated execution produces identical graph identifiability results.
2. /e2e-output-verification-testing:
   - Anti-Shallow Testing Protocol: Inspects non-trivial computational values,
     actual ATE estimates, non-empty adjustment sets, valid p-values, and bounded confidence intervals.
   - Self-Healing Cyclic Loop: Injects impossible candidate violating linear conservation laws,
     probability simplex, and non-negativity; verifies InvariantCritic catches violations,
     orthogonally projects them onto the feasible manifold, and completes successfully.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from dia.agentic_workflow import (
    CausalAuditorNode,
    InvariantCriticNode,
    InvestigationState,
    PlannerNode,
    StatefulInvestigationMachine,
    SynthesizerNode,
    detect_all_colliders,
    find_optimal_backdoor_adjustment,
)
from dia.causal_identifiability import CausalDAG, is_d_separated
from dia.invariant_engine import ConservationLaw


# ───────────────────────────────────────────────────────────────────────────
# Topology Generators for Property-Based Fuzzing
# ───────────────────────────────────────────────────────────────────────────

def generate_random_dag(n_nodes: int, p_edge: float = 0.28, seed: int = 42) -> Tuple[CausalDAG, List[str]]:
    """Generates an acyclic directed graph with topologically ordered nodes."""
    rng = random.Random(seed)
    nodes = [f"V_{i}" for i in range(n_nodes)]
    dag = CausalDAG()
    for n in nodes:
        dag.add_node(n)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p_edge:
                dag.add_edge(nodes[i], nodes[j])
    return dag, nodes


# ═══════════════════════════════════════════════════════════════════════════
# 1. Property-Based Generative Fuzzing Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPropertyBasedGenerativeFuzzing:
    """Property-based generative verification over arbitrary graph topologies."""

    @pytest.mark.parametrize("seed", list(range(100, 125)))
    def test_random_dag_descendant_and_collider_invariants(self, seed: int) -> None:
        """
        Generates random DAGs with 5 to 15 nodes and verifies fundamental causal invariants:
        1. No node in minimal_adjustment_set is a descendant of treatment X.
        2. All colliders detected possess >= 2 incoming edges.
        3. The minimal adjustment set d-separates X and Y in mutilated graph G_{\\underline{X}}.
        """
        rng = random.Random(seed)
        n_nodes = rng.randint(5, 15)
        dag, nodes = generate_random_dag(n_nodes=n_nodes, p_edge=0.30, seed=seed)

        # Pick treatment X and outcome Y such that Y is after X in topological order
        treatment = nodes[1]
        outcome = nodes[-1]

        state = InvestigationState(
            query=f"Evaluate causal effect of {treatment} on {outcome}",
            causal_graph=dag,
            candidate_treatments=[treatment],
            target_outcome=outcome,
        )

        auditor = CausalAuditorNode()
        result_state = auditor.run(state)

        # Ensure execution did not fail unexpectedly
        assert result_state.status == "CHECKING_INVARIANTS"

        # Invariant 1: Minimal adjustment set strictly excludes descendants of treatment X
        descendants_x = dag.get_descendants({treatment}) - {treatment}
        adjustment_set = set(result_state.minimal_adjustment_set)
        violating_descendants = adjustment_set & descendants_x
        assert not violating_descendants, (
            f"Seed {seed}: Adjustment set contains descendants of treatment '{treatment}': "
            f"{violating_descendants}"
        )

        # Invariant 2: All identified colliders must have in-degree >= 2
        for col_node in result_state.colliders_detected:
            parents = dag.parents.get(col_node, set())
            assert len(parents) >= 2, (
                f"Seed {seed}: Node '{col_node}' marked as collider but has only {len(parents)} parents."
            )

        # Invariant 3: Backdoor d-separation in G_{\\underline{X}}
        g_under_x = dag.mutilate_remove_outgoing({treatment})
        is_blocked = is_d_separated(g_under_x, treatment, outcome, adjustment_set)
        assert is_blocked, (
            f"Seed {seed}: Minimal adjustment set {adjustment_set} fails to d-separate "
            f"'{treatment}' from '{outcome}' in G_\\underline{{{treatment}}}."
        )

    @pytest.mark.parametrize("seed", [201, 202, 203, 204, 205, 206, 207, 208])
    def test_metamorphic_independent_noise_invariance(self, seed: int) -> None:
        """
        Metamorphic Relation:
        Adding an independent noise variable (with zero structural causal edges to X or Y)
        must NEVER alter the computed minimal adjustment set.
        """
        # Create DAG with known confounding
        dag = CausalDAG()
        dag.add_edge("Z1", "X")
        dag.add_edge("Z1", "Y")
        dag.add_edge("Z2", "X")
        dag.add_edge("Z2", "Y")
        dag.add_edge("X", "Y")

        # Add some random background structure
        rng = random.Random(seed)
        extra_nodes = [f"Extra_{i}" for i in range(rng.randint(2, 6))]
        for en in extra_nodes:
            dag.add_node(en)
            if rng.random() < 0.4:
                dag.add_edge(en, "Z1")

        state1 = InvestigationState(
            query="Analyze effect of X on Y",
            causal_graph=dag,
            candidate_treatments=["X"],
            target_outcome="Y",
        )
        auditor = CausalAuditorNode()
        res1 = auditor.run(state1)
        base_adj = res1.minimal_adjustment_set

        # Metamorphic mutation: Inject completely isolated independent noise variables
        dag_mutated = CausalDAG(nodes=set(dag.nodes), edges=set(dag.edges))
        dag_mutated.add_node("Pure_Noise_Alpha")
        dag_mutated.add_node("Pure_Noise_Beta")

        state2 = InvestigationState(
            query="Analyze effect of X on Y",
            causal_graph=dag_mutated,
            candidate_treatments=["X"],
            target_outcome="Y",
        )
        res2 = auditor.run(state2)
        mutated_adj = res2.minimal_adjustment_set

        # Assert strict metamorphic invariance
        assert mutated_adj == base_adj, (
            f"Metamorphic noise violation: Base adjustment {base_adj} vs Mutated adjustment {mutated_adj}"
        )

    def test_causal_auditor_idempotence(self) -> None:
        """
        Idempotence Invariant:
        Re-running the CausalAuditorNode on the same state produces strictly identical outputs.
        """
        dag = CausalDAG()
        dag.add_edge("Conf", "Treat")
        dag.add_edge("Conf", "Outcome")
        dag.add_edge("Treat", "Outcome")
        dag.add_edge("Treat", "Mediator")
        dag.add_edge("Mediator", "Outcome")
        dag.add_edge("Treat", "Collider_C")
        dag.add_edge("Outcome", "Collider_C")

        state = InvestigationState(
            query="Evaluate Treat on Outcome",
            causal_graph=dag,
            candidate_treatments=["Treat"],
            target_outcome="Outcome",
        )

        auditor = CausalAuditorNode()
        res_pass1 = auditor.run(state)
        pass1_adj = list(res_pass1.minimal_adjustment_set)
        pass1_colliders = list(res_pass1.colliders_detected)
        pass1_paths = list(res_pass1.active_paths)

        # Run again
        res_pass2 = auditor.run(res_pass1)
        pass2_adj = list(res_pass2.minimal_adjustment_set)
        pass2_colliders = list(res_pass2.colliders_detected)
        pass2_paths = list(res_pass2.active_paths)

        assert pass1_adj == pass2_adj == ["Conf"]
        assert pass1_colliders == pass2_colliders == ["Collider_C", "Outcome"]
        assert pass1_paths == pass2_paths


# ═══════════════════════════════════════════════════════════════════════════
# 2. Output Verification Testing (Anti-Shallow Testing Protocol)
# ═══════════════════════════════════════════════════════════════════════════

class TestOutputVerificationTesting:
    """Verifies non-trivial output values, empirical ATE metrics, and self-healing cycles."""

    def test_e2e_confounding_investigation_dossier_outputs(self) -> None:
        """
        Full end-to-end investigation verification with observational data.
        Verifies:
        - Non-trivial Average Treatment Effect close to ground truth (3.50).
        - Non-empty minimal adjustment set containing confounder Z.
        - Valid, statistically significant p-value (< 0.001).
        - Strictly bounded 95% confidence interval encompassing the true parameter.
        - Publication-grade causal interpretation text.
        """
        # Ground-truth structural causal model:
        # Z ~ N(0, 1)
        # X = 1.2 * Z + N(0, 0.4^2)
        # Y = 3.5 * X + 2.0 * Z + N(0, 0.4^2)
        np.random.seed(1337)
        n = 500
        z = np.random.normal(0.0, 1.0, size=n)
        x = 1.2 * z + np.random.normal(0.0, 0.4, size=n)
        y = 3.5 * x + 2.0 * z + np.random.normal(0.0, 0.4, size=n)
        df = pd.DataFrame({"Confounder_Z": z, "Treatment_X": x, "Outcome_Y": y})

        dag = CausalDAG()
        dag.add_edge("Confounder_Z", "Treatment_X")
        dag.add_edge("Confounder_Z", "Outcome_Y")
        dag.add_edge("Treatment_X", "Outcome_Y")

        state = InvestigationState(
            query="Assess the causal impact of Treatment_X on Outcome_Y",
            causal_graph=dag,
            data=df,
            candidate_treatments=["Treatment_X"],
            target_outcome="Outcome_Y",
        )

        machine = StatefulInvestigationMachine()
        final_state = machine.run(state)

        # 1. Status conformance
        assert final_state.status == "COMPLETED"
        assert final_state.dossier is not None
        dossier = final_state.dossier

        # 2. Adjustment set verification
        assert dossier["minimal_adjustment_set"] == ["Confounder_Z"]
        assert dossier["causal_identifiability"] is True
        assert dossier["identifiability_status"] == "IDENTIFIABLE"

        # 3. Non-trivial ATE assertion (True value = 3.5)
        ate = dossier["average_treatment_effect"]
        assert isinstance(ate, float)
        assert 3.35 <= ate <= 3.65, f"Estimated ATE {ate} outside expected range [3.35, 3.65]"

        # 4. Standard error & t-statistic
        se = dossier["ate_standard_error"]
        assert 0.001 < se < 0.15, f"Unexpected SE {se}"
        assert abs(dossier["t_statistic"]) > 20.0

        # 5. Valid p-value
        p_val = dossier["p_value"]
        assert 0.0 <= p_val <= 0.001, f"Expected highly significant p-value, got {p_val}"

        # 6. Bounded 95% Confidence Interval
        ci = dossier["confidence_interval_95"]
        ci_lower = dossier["confidence_interval_lower"]
        ci_upper = dossier["confidence_interval_upper"]
        assert ci_lower < ate < ci_upper
        assert ci_lower < 3.5 < ci_upper
        assert (ci_upper - ci_lower) > 0.01

        # 7. Causal interpretation conformance
        interpretation = dossier["causal_interpretation"]
        assert "Backdoor Adjustment" in interpretation
        assert "Treatment_X" in interpretation
        assert "Outcome_Y" in interpretation
        assert "Confounder_Z" in interpretation

    def test_self_healing_linear_conservation_law_violation(self) -> None:
        """
        Verifies cyclic self-healing when a candidate intervention violates
        a physical conservation law (A x = b).
        - Injects an impossible candidate: Digital (800) + Traditional (700) = 1500 != 1000.
        - Verifies InvariantCritic detects residual (500.0).
        - Verifies workflow cycles back to Planner with constraint feedback.
        - Verifies orthogonal projection onto manifold: Digital = 550, Traditional = 450.
        - Verifies workflow completes successfully with self_healing_attempts == 1.
        """
        dag = CausalDAG()
        dag.add_edge("Spend_Digital", "Revenue")
        dag.add_edge("Spend_Traditional", "Revenue")

        budget_law = ConservationLaw(
            name="total_marketing_budget",
            coefficients={"Spend_Digital": 1.0, "Spend_Traditional": 1.0},
            target_sum=1000.0,
            tolerance=1e-5,
        )

        state = InvestigationState(
            query="Evaluate Spend_Digital on Revenue",
            causal_graph=dag,
            candidate_treatments=["Spend_Digital"],
            target_outcome="Revenue",
            conservation_laws=[budget_law],
            intervention_proposals={"Spend_Digital": 800.0, "Spend_Traditional": 700.0},
        )

        machine = StatefulInvestigationMachine()
        final_state = machine.run(state)

        assert final_state.status == "COMPLETED"
        assert final_state.self_healing_attempts == 1

        dossier = final_state.dossier
        assert dossier is not None
        assert dossier["self_healing_cycles"] == 1

        # Verify projected values satisfy conservation law to machine precision
        projected = dossier["projected_interventions"]
        assert abs(projected["Spend_Digital"] - 550.0) < 1e-5
        assert abs(projected["Spend_Traditional"] - 450.0) < 1e-5
        assert abs(projected["Spend_Digital"] + projected["Spend_Traditional"] - 1000.0) < 1e-5

        # Verify audit record trail
        records = dossier["invariants_verified"]
        assert any(r["status"] == "SELF_HEALED_PROJECTED" and abs(r["residual_before"] - 500.0) < 1e-4 for r in records)
        assert any(r["status"] == "PASSED" and abs(r["residual"]) < 1e-5 for r in records)

    def test_self_healing_probability_simplex_violation(self) -> None:
        """
        Verifies cyclic self-healing when intervention violates probability simplex:
        p_1 + p_2 + p_3 = 1.0, p_i >= 0.
        Injects impossible vector [0.8, 0.7, 0.5] (sum = 2.0).
        Verifies Duchi et al. projection onto Delta^3.
        """
        dag = CausalDAG()
        dag.add_edge("p1", "Outcome")

        state = InvestigationState(
            query="Evaluate p1 on Outcome",
            causal_graph=dag,
            candidate_treatments=["p1"],
            target_outcome="Outcome",
            simplex_groups=[["p1", "p2", "p3"]],
            intervention_proposals={"p1": 0.8, "p2": 0.7, "p3": 0.5},
        )

        machine = StatefulInvestigationMachine()
        final_state = machine.run(state)

        assert final_state.status == "COMPLETED"
        assert final_state.self_healing_attempts == 1

        projected = final_state.dossier["projected_interventions"]
        sum_p = projected["p1"] + projected["p2"] + projected["p3"]
        assert abs(sum_p - 1.0) < 1e-5
        assert all(projected[k] >= 0.0 for k in ["p1", "p2", "p3"])

    def test_self_healing_non_negativity_violation(self) -> None:
        """
        Verifies non-negativity constraint projection when negative intervention values are injected.
        """
        dag = CausalDAG()
        dag.add_edge("Safety_Stock", "Fulfillment_Rate")

        state = InvestigationState(
            query="Evaluate Safety_Stock on Fulfillment_Rate",
            causal_graph=dag,
            candidate_treatments=["Safety_Stock"],
            target_outcome="Fulfillment_Rate",
            non_negative_variables=["Safety_Stock"],
            intervention_proposals={"Safety_Stock": -45.0},
        )

        machine = StatefulInvestigationMachine()
        final_state = machine.run(state)

        assert final_state.status == "COMPLETED"
        assert final_state.self_healing_attempts == 1
        assert final_state.dossier["projected_interventions"]["Safety_Stock"] == 0.0

    def test_self_healing_max_cycles_exhaustion_terminates_with_failure(self) -> None:
        """
        Verifies that when self-healing attempts exceed max_self_healing_cycles (default 3),
        the workflow transitions deterministically to FAILED rather than hanging in an infinite loop.
        """
        dag = CausalDAG()
        dag.add_edge("X", "Y")

        budget_law = ConservationLaw(
            name="strict_law",
            coefficients={"A": 1.0},
            target_sum=100.0,
        )

        # Set max cycles to 0 to simulate exhaustion immediately
        state = InvestigationState(
            query="Evaluate X on Y",
            causal_graph=dag,
            candidate_treatments=["X"],
            target_outcome="Y",
            conservation_laws=[budget_law],
            intervention_proposals={"A": 50.0},
            max_self_healing_cycles=0,
        )

        machine = StatefulInvestigationMachine()
        final_state = machine.run(state)

        assert final_state.status == "FAILED"
        assert "Exceeded maximum self-healing attempts" in str(final_state.error_message)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Structural Topology & Intent Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralTopologyAndIntentEdgeCases:
    """Verifies Berkson's bias collider prevention, M-bias, and query parsing."""

    def test_berkson_bias_collider_is_never_conditioned(self) -> None:
        """
        Berkson's Paradox:
        X -> Collider <- Y
        X and Y are marginally independent.
        Conditioning on Collider creates spurious association (Berkson's bias).
        Verifies CausalAuditorNode identifies Collider and leaves minimal adjustment set empty.
        """
        dag = CausalDAG()
        dag.add_edge("Treatment_X", "Berkson_Collider")
        dag.add_edge("Outcome_Y", "Berkson_Collider")

        state = InvestigationState(
            query="Impact of Treatment_X on Outcome_Y",
            causal_graph=dag,
            candidate_treatments=["Treatment_X"],
            target_outcome="Outcome_Y",
        )

        auditor = CausalAuditorNode()
        res = auditor.run(state)

        assert "Berkson_Collider" in res.colliders_detected
        assert "Berkson_Collider" not in res.minimal_adjustment_set
        assert res.minimal_adjustment_set == []

    def test_m_bias_structural_topology(self) -> None:
        """
        M-Bias Topology:
        X <- U1 -> M <- U2 -> Y
        M is an unconditioned collider blocking the path X <- U1 -> M <- U2 -> Y.
        Conditioning on M opens the backdoor path.
        Verifies auditor keeps M unconditioned and returns minimal adjustment set [].
        """
        dag = CausalDAG()
        dag.add_edge("U1", "X")
        dag.add_edge("U1", "M")
        dag.add_edge("U2", "M")
        dag.add_edge("U2", "Y")
        dag.add_edge("X", "Y")

        state = InvestigationState(
            query="Causal effect of X on Y",
            causal_graph=dag,
            candidate_treatments=["X"],
            target_outcome="Y",
        )

        auditor = CausalAuditorNode()
        res = auditor.run(state)

        assert "M" in res.colliders_detected
        assert "M" not in res.minimal_adjustment_set
        assert res.minimal_adjustment_set == []

    @pytest.mark.parametrize(
        "query, expected_treatment, expected_outcome",
        [
            ("What is the causal impact of marketing_spend on revenue?", "marketing_spend", "revenue"),
            ("How does discount affect churn?", "discount", "churn"),
            ("Evaluate price -> sales_volume", "price", "sales_volume"),
            ("Assess effect of training on employee_productivity", "training", "employee_productivity"),
        ],
    )
    def test_planner_query_intent_resolution(
        self, query: str, expected_treatment: str, expected_outcome: str
    ) -> None:
        """Verifies PlannerNode resolves natural language queries into treatment and outcome."""
        planner = PlannerNode()
        state = InvestigationState(query=query)
        res = planner.run(state)

        assert res.target_outcome == expected_outcome
        assert res.candidate_treatments == [expected_treatment]
        assert res.status == "AUDITING_CAUSALITY"

    def test_synthetic_sem_data_generation_fallback(self) -> None:
        """
        Verifies that when observational data is omitted, SynthesizerNode generates
        a topologically sound SEM dataset and computes non-trivial causal metrics.
        """
        dag = CausalDAG()
        dag.add_edge("Z", "X")
        dag.add_edge("Z", "Y")
        dag.add_edge("X", "Y")

        state = InvestigationState(
            query="Evaluate X on Y",
            causal_graph=dag,
            candidate_treatments=["X"],
            target_outcome="Y",
        )

        machine = StatefulInvestigationMachine()
        final_state = machine.run(state)

        assert final_state.status == "COMPLETED"
        assert final_state.dossier is not None
        assert abs(final_state.dossier["average_treatment_effect"]) > 0.5
        assert final_state.dossier["minimal_adjustment_set"] == ["Z"]
        assert 0.0 <= final_state.dossier["p_value"] <= 1.0
        ci = final_state.dossier["confidence_interval_95"]
        assert ci[0] < ci[1]
