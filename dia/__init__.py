# dia package – Data Intelligence Assistant backend
from .data_loader import load_csv
from .data_profiler import detect_target_type, infer_column_roles, profile_dataframe
from .explainability import generate_explanation
from .goal_parser import parse_goal
from .model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_and_evaluate
from .utils import is_plotly_available, limit_models
from . import causal_discovery
from . import conformal_uncertainty
from . import drift_sentinel
from . import leakage_detector
from . import multimodal_fusion
from . import pareto_frontier
from . import recourse_engine
from . import stress_testing
from . import symbolic_features
from . import wasm_compiler
from .causal_discovery import discover_causal_graph, simulate_intervention
from .conformal_uncertainty import evaluate_conformal_bounds
from .drift_sentinel import audit_drift_sentinel
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
    "leakage_detector",
    "multimodal_fusion",
    "pareto_frontier",
    "recourse_engine",
    "stress_testing",
    "symbolic_features",
    "wasm_compiler",
    "audit_drift_sentinel",
    "compute_pareto_frontier",
    "compute_recourse",
    "detect_leakage",
    "discover_causal_graph",
    "discover_symbolic_features",
    "evaluate_conformal_bounds",
    "fuse_tabular_and_text",
    "run_stress_test",
    "simulate_intervention",
    "transpile_to_edge_bundle",
]

