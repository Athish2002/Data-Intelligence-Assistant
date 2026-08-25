# dia package – Data Intelligence Assistant backend
from .data_loader import load_csv
from .data_profiler import profile_dataframe, infer_column_roles, detect_target_type
from .goal_parser import parse_goal
from .model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_and_evaluate
from .explainability import generate_explanation
from .utils import limit_models, is_plotly_available

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
]
