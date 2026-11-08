# dia package – Data Intelligence Assistant backend
from .data_loader import load_csv
from .data_profiler import detect_target_type, infer_column_roles, profile_dataframe
from .explainability import generate_explanation
from .goal_parser import parse_goal
from .model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_and_evaluate
from .utils import is_plotly_available, limit_models
from . import agentic_workflow
from . import causal_discovery
from . import causal_identifiability
from . import conformal_uncertainty
from . import drift_sentinel
from . import invariant_engine
from . import leakage_detector
from . import multimodal_fusion
from . import pareto_frontier
from . import recourse_engine
from . import stress_testing
from . import symbolic_features
from . import wasm_compiler
from .agentic_workflow import (
    AgenticInvestigationGraph,
    CausalAuditorNode,
    CausalInvestigationWorkflow,
    InvariantCriticNode,
    InvestigationState,
    PlannerNode,
    StatefulInvestigationMachine,
    SynthesizerNode,
)
from .causal_identifiability import (
    CausalDAG,
    is_d_separated,
    resolve_simpsons_paradox_algebraically,
    verify_backdoor_criterion,
)
from .causal_discovery import discover_causal_graph, simulate_intervention
from .conformal_uncertainty import evaluate_conformal_bounds
from .drift_sentinel import audit_drift_sentinel
from .invariant_engine import (
    ConservationLaw,
    ConservationProjector,
    audit_dataframe_axioms,
    enforce_monotonicity_pava,
    project_to_probability_simplex,
    scan_dataframe_conservation_laws,
)
from .leakage_detector import detect_leakage
from .multimodal_fusion import fuse_tabular_and_text
from .pareto_frontier import compute_pareto_frontier
from .recourse_engine import compute_recourse
from .stress_testing import run_stress_test
from .symbolic_features import discover_symbolic_features
from .wasm_compiler import transpile_to_edge_bundle

__all__ = [
    "load_csv",
    "profile_dataframe",
    "infer_column_roles",
    "detect_target_type",
    "parse_goal",
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
    "train_and_evaluate",
    "generate_explanation",
    "limit_models",
    "is_plotly_available",
    "causal_discovery",
    "conformal_uncertainty",
    "drift_sentinel",
    "invariant_engine",
    "leakage_detector",
    "multimodal_fusion",
    "pareto_frontier",
    "recourse_engine",
    "stress_testing",
    "symbolic_features",
    "wasm_compiler",
    "agentic_workflow",
    "causal_identifiability",
    "AgenticInvestigationGraph",
    "CausalAuditorNode",
    "CausalInvestigationWorkflow",
    "InvariantCriticNode",
    "InvestigationState",
    "PlannerNode",
    "StatefulInvestigationMachine",
    "SynthesizerNode",
    "CausalDAG",
    "is_d_separated",
    "resolve_simpsons_paradox_algebraically",
    "verify_backdoor_criterion",
    "audit_dataframe_axioms",
    "audit_drift_sentinel",
    "compute_pareto_frontier",
    "compute_recourse",
    "detect_leakage",
    "discover_causal_graph",
    "discover_symbolic_features",
    "enforce_monotonicity_pava",
    "evaluate_conformal_bounds",
    "fuse_tabular_and_text",
    "project_to_probability_simplex",
    "run_stress_test",
    "scan_dataframe_conservation_laws",
    "simulate_intervention",
    "transpile_to_edge_bundle",
    "ConservationLaw",
    "ConservationProjector",
]
