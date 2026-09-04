"""
dia/model_trainer.py
────────────────────
Comprehensive ML Training Engine with 10+ Model Families, Automated Hyperparameter
Optimization (HPO), Cross-Validation, Class Imbalance weighting, Automated Feature
Engineering (AutoFE), Probability Calibration, and Blended Ensembles.

Supported models
────────────────
Classification:
  logreg   → LogisticRegression
  rf       → RandomForestClassifier
  et       → ExtraTreesClassifier
  gb       → GradientBoostingClassifier
  xgb      → XGBClassifier
  lgbm     → LGBMClassifier
  cat      → CatBoostClassifier
  svm      → Support Vector Classifier (SVC)
  knn      → KNeighborsClassifier
  mlp      → Multi-Layer Perceptron (Neural Net)

Regression:
  linreg   → LinearRegression
  rf       → RandomForestRegressor
  et       → ExtraTreesRegressor
  gb       → GradientBoostingRegressor
  xgb      → XGBRegressor
  lgbm     → LGBMRegressor
  cat      → CatBoostRegressor
  svm      → Support Vector Regressor (SVR)
  knn      → KNeighborsRegressor
  mlp      → Multi-Layer Perceptron Regressor
"""

import gc
import logging
import os
import warnings
from typing import Any


def safe_n_jobs(requested: int = -1) -> int:
    """
    Guarantees worker concurrency stays strictly bounded to prevent Loky/joblib
    process explosions and memory thrashing on Windows or high-core workstations.
    Capped at min(4, max(1, (os.cpu_count() or 2) // 4)).
    """
    cpu_total = os.cpu_count() or 2
    safe_limit = min(4, max(1, cpu_total // 4))
    if requested is None or requested <= 0:
        return safe_limit
    return min(requested, safe_limit)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight

log = logging.getLogger("dia.model_trainer")
warnings.filterwarnings("ignore")

# ─── HPO Parameter Search Spaces ─────────────────────────────────────────────

HPO_PARAM_GRIDS: dict[str, dict] = {
    "logreg": {
        "C": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        "penalty": ["l2"],
    },
    "rf": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "et": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
    },
    "gb": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "xgb": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "lgbm": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [-1, 5, 10, 20],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
        "subsample": [0.6, 0.8, 1.0],
    },
    "cat": {
        "iterations": [50, 100, 150],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0, 50.0],
        "gamma": ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9, 15],
        "weights": ["uniform", "distance"],
    },
    "mlp": {
        "hidden_layer_sizes": [(32,), (64, 32), (100,)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    },
    "linreg": {
        "fit_intercept": [True, False],
    },
}

# ─── Model Factories ─────────────────────────────────────────────────────────

def _make_xgb_clf(use_gpu: bool = False, n_jobs: int = -1):
    try:
        from xgboost import XGBClassifier
        params = {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss", "verbosity": 0, "n_jobs": safe_n_jobs(n_jobs)}
        if use_gpu:
            params.update({"tree_method": "hist", "device": "cuda"})
        return XGBClassifier(**params)
    except ImportError:
        return None


def _make_xgb_reg(use_gpu: bool = False, n_jobs: int = -1):
    try:
        from xgboost import XGBRegressor
        params = {"n_estimators": 100, "random_state": 42, "verbosity": 0, "n_jobs": safe_n_jobs(n_jobs)}
        if use_gpu:
            params.update({"tree_method": "hist", "device": "cuda"})
        return XGBRegressor(**params)
    except ImportError:
        return None


def _make_lgbm_clf(use_gpu: bool = False, n_jobs: int = -1):
    try:
        from lightgbm import LGBMClassifier
        params = {"n_estimators": 100, "random_state": 42, "verbose": -1, "n_jobs": safe_n_jobs(n_jobs)}
        if use_gpu:
            params.update({"device_type": "gpu"})
        return LGBMClassifier(**params)
    except ImportError:
        return None


def _make_lgbm_reg(use_gpu: bool = False, n_jobs: int = -1):
    try:
        from lightgbm import LGBMRegressor
        params = {"n_estimators": 100, "random_state": 42, "verbose": -1, "n_jobs": safe_n_jobs(n_jobs)}
        if use_gpu:
            params.update({"device_type": "gpu"})
        return LGBMRegressor(**params)
    except ImportError:
        return None


def _make_cat_clf(use_gpu: bool = False, n_jobs: int = -1):
    try:
        from catboost import CatBoostClassifier
        params = {"iterations": 100, "random_seed": 42, "verbose": 0, "thread_count": safe_n_jobs(n_jobs)}
        if use_gpu:
            params.update({"task_type": "GPU"})
        return CatBoostClassifier(**params)
    except ImportError:
        return None


def _make_cat_reg(use_gpu: bool = False, n_jobs: int = -1):
    try:
        from catboost import CatBoostRegressor
        params = {"iterations": 100, "random_seed": 42, "verbose": 0, "thread_count": safe_n_jobs(n_jobs)}
        if use_gpu:
            params.update({"task_type": "GPU"})
        return CatBoostRegressor(**params)
    except ImportError:
        return None


# ─── Model Registries ────────────────────────────────────────────────────────

CLASSIFICATION_MODELS: dict[str, dict] = {
    "logreg": {
        "label": "Logistic Regression",
        "description": "Fast, interpretable linear classifier.",
        "factory": lambda use_gpu, n_jobs: LogisticRegression(max_iter=1000, random_state=42),
    },
    "rf": {
        "label": "Random Forest",
        "description": "Robust bagging ensemble of decision trees.",
        "factory": lambda use_gpu, n_jobs: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs),
    },
    "et": {
        "label": "Extra Trees",
        "description": "Extremely randomized trees with lower variance.",
        "factory": lambda use_gpu, n_jobs: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs),
    },
    "gb": {
        "label": "Gradient Boosting",
        "description": "Sequential error-correcting boosted trees.",
        "factory": lambda use_gpu, n_jobs: GradientBoostingClassifier(n_estimators=100, random_state=42),
    },
    "xgb": {
        "label": "XGBoost",
        "description": "State-of-the-art exact & histogram gradient boosting.",
        "factory": _make_xgb_clf,
    },
    "lgbm": {
        "label": "LightGBM",
        "description": "Fast leaf-wise gradient boosting for large scale data.",
        "factory": _make_lgbm_clf,
    },
    "cat": {
        "label": "CatBoost",
        "description": "Oblivious decision trees optimized for categorical data.",
        "factory": _make_cat_clf,
    },
    "svm": {
        "label": "Support Vector Machine (SVC)",
        "description": "Max-margin hyperplane classifier with RBF kernel.",
        "factory": lambda use_gpu, n_jobs: SVC(probability=True, kernel="rbf", random_state=42),
    },
    "knn": {
        "label": "K-Nearest Neighbors (KNN)",
        "description": "Non-parametric instance-based similarity classifier.",
        "factory": lambda use_gpu, n_jobs: KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
    },
    "mlp": {
        "label": "Neural Network (MLP)",
        "description": "Multi-layer perceptron with backpropagation.",
        "factory": lambda use_gpu, n_jobs: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    },
}

REGRESSION_MODELS: dict[str, dict] = {
    "linreg": {
        "label": "Linear Regression (Ridge)",
        "description": "L2-regularized linear baseline.",
        "factory": lambda use_gpu, n_jobs: Ridge(random_state=42),
    },
    "rf": {
        "label": "Random Forest",
        "description": "Non-linear bagging ensemble for continuous targets.",
        "factory": lambda use_gpu, n_jobs: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs),
    },
    "et": {
        "label": "Extra Trees",
        "description": "Fast randomized ensemble with reduced variance.",
        "factory": lambda use_gpu, n_jobs: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs),
    },
    "gb": {
        "label": "Gradient Boosting",
        "description": "Sequential gradient boosted regression trees.",
        "factory": lambda use_gpu, n_jobs: GradientBoostingRegressor(n_estimators=100, random_state=42),
    },
    "xgb": {
        "label": "XGBoost",
        "description": "High-performance gradient boosting regressor.",
        "factory": _make_xgb_reg,
    },
    "lgbm": {
        "label": "LightGBM",
        "description": "Histogram-based gradient boosting regressor.",
        "factory": _make_lgbm_reg,
    },
    "cat": {
        "label": "CatBoost",
        "description": "Gradient boosting with symmetric decision trees.",
        "factory": _make_cat_reg,
    },
    "svm": {
        "label": "Support Vector Regressor (SVR)",
        "description": "Epsilon-support vector regression with RBF kernel.",
        "factory": lambda use_gpu, n_jobs: SVR(kernel="rbf"),
    },
    "knn": {
        "label": "K-Nearest Neighbors (KNN)",
        "description": "Local neighborhood continuous value averaging.",
        "factory": lambda use_gpu, n_jobs: KNeighborsRegressor(n_neighbors=5, n_jobs=n_jobs),
    },
    "mlp": {
        "label": "Neural Network (MLP)",
        "description": "Multi-layer perceptron regressor with Adam optimizer.",
        "factory": lambda use_gpu, n_jobs: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    },
}


# ─── Preprocessing helpers ────────────────────────────────────────────────────

class SafeCategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms any incoming categorical/object data into uniform clean strings,
    substituting nulls and NaNs with a deterministic sentinel string '__missing__'.
    Guarantees scikit-learn OneHotEncoder / OrdinalEncoder will NEVER encounter
    mixed types (e.g. ['int', 'str']) or unhashable objects.
    """
    def __init__(self, missing_sentinel: str = "__missing__"):
        self.missing_sentinel = missing_sentinel
        self.feature_names_in_ = None
        self.n_features_in_ = 0

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            self.n_features_in_ = len(X.columns)
        else:
            arr = np.asarray(X)
            self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
            self.feature_names_in_ = np.array([f"cat_{i}" for i in range(self.n_features_in_)], dtype=object)
        return self

    def transform(self, X):
        if hasattr(X, "to_numpy"):
            arr = X.to_numpy(dtype=object, copy=True)
        else:
            arr = np.array(X, dtype=object, copy=True)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        flat = arr.ravel()
        null_set = {"nan", "none", "null", "<na>", "", "n/a", "?", "--", "missing"}
        cleaned = [
            self.missing_sentinel
            if (pd.isna(v) or v is None or str(v).strip().lower() in null_set)
            else str(v).strip()
            for v in flat
        ]
        return np.array(cleaned, dtype=object).reshape(arr.shape)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        return np.array([f"cat_{i}" for i in range(self.n_features_in_)], dtype=object)


class SafeNumericTransformer(BaseEstimator, TransformerMixin):
    """
    Sanitizes numerical features:
    - Replaces inf, -inf with NaN.
    - Clips extreme values outside [-1e15, 1e15] to prevent float64 overflow.
    - Imputes NaN values with the median of training data, falling back to 0.0 if all-NaN.
    - Scales numerical features with RobustScaler.
    """
    def __init__(self):
        self.medians_: np.ndarray | None = None
        self.scaler = RobustScaler()
        self.feature_names_in_ = None
        self.n_features_in_ = 0

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            try:
                arr = X.to_numpy(dtype=float, copy=True)
            except (ValueError, TypeError):
                arr = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        else:
            try:
                arr = np.array(X, dtype=float, copy=True)
            except (ValueError, TypeError):
                arr = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        self.n_features_in_ = arr.shape[1]

        # Clean non-finite & clip
        arr[np.isinf(arr)] = np.nan
        arr = np.clip(arr, -1e15, 1e15)

        # Compute column-wise medians
        self.medians_ = np.zeros(self.n_features_in_, dtype=float)
        for j in range(self.n_features_in_):
            col_vals = arr[:, j]
            valid_vals = col_vals[~np.isnan(col_vals)]
            if len(valid_vals) > 0:
                self.medians_[j] = float(np.median(valid_vals))
            else:
                self.medians_[j] = 0.0

        imputed_arr = arr.copy()
        for j in range(self.n_features_in_):
            mask = np.isnan(imputed_arr[:, j])
            imputed_arr[mask, j] = self.medians_[j]

        self.scaler.fit(imputed_arr)
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            try:
                arr = X.to_numpy(dtype=float, copy=True)
            except (ValueError, TypeError):
                arr = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        elif hasattr(X, "to_numpy"):
            try:
                arr = X.to_numpy(dtype=float, copy=True)
            except (ValueError, TypeError):
                arr = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        else:
            try:
                arr = np.array(X, dtype=float, copy=True)
            except (ValueError, TypeError):
                arr = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        arr[np.isinf(arr)] = np.nan
        arr = np.clip(arr, -1e15, 1e15)

        for j in range(arr.shape[1]):
            med = self.medians_[j] if (self.medians_ is not None and j < len(self.medians_)) else 0.0
            mask = np.isnan(arr[:, j])
            arr[mask, j] = med

        return self.scaler.transform(arr)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        n_features = len(self.medians_) if self.medians_ is not None else 1
        return np.array([f"num_{i}" for i in range(n_features)], dtype=object)


def _build_preprocessor(X: pd.DataFrame):
    """Build a hardened ColumnTransformer with advanced encoding, bounded dimensions, and robust scaling."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols_all = X.select_dtypes(include=["object", "category"]).columns.tolist()

    low_card_cols = []
    high_card_cols = []
    for col in cat_cols_all:
        if X[col].nunique(dropna=True) <= 30:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)

    numeric_pipe = Pipeline([
        ("safe_num", SafeNumericTransformer()),
    ])

    low_card_pipe = Pipeline([
        ("safe_cat", SafeCategoricalTransformer()),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
    ])

    high_card_pipe = Pipeline([
        ("safe_cat", SafeCategoricalTransformer()),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if low_card_cols:
        transformers.append(("cat_low", low_card_pipe, low_card_cols))
    if high_card_cols:
        transformers.append(("cat_high", high_card_pipe, high_card_cols))

    if not transformers:
        transformers.append(("num_dummy", numeric_pipe, numeric_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_feature_names(preprocessor, X: pd.DataFrame) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return list(X.columns)


# ─── Metric helpers ───────────────────────────────────────────────────────────

def _classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            if avg == "binary" and y_proba.shape[1] >= 2:
                metrics["ROC-AUC"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
                metrics["PR-AUC"] = round(float(average_precision_score(y_true, y_proba[:, 1])), 4)
            elif avg == "macro" and y_proba.shape[1] == n_classes:
                metrics["ROC-AUC"] = round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 4)
        except Exception:
            log.debug("Could not compute ROC-AUC/PR-AUC for this model.", exc_info=True)
    return metrics


def _regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R²": round(float(r2_score(y_true, y_pred)), 4),
    }


# ─── Feature importance extractor ────────────────────────────────────────────

def _extract_importance(model, feature_names: list[str]) -> pd.Series:
    """Return a Series of feature importances sorted descending."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return pd.Series(dtype=float)

    s = pd.Series(importances, index=feature_names, name="importance")
    s = s.sort_values(ascending=False)
    total = s.sum()
    if total > 0:
        s = s / total
    return s.round(4)


# ─── Main training function ───────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    selected_model_keys: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
    use_gpu: bool = False,
    n_jobs: int = -1,
    handle_imbalance: bool = True,
    apply_cv: bool = False,
    enable_hpo: bool = False,
    hpo_iter: int = 10,
    enable_autofe: bool = False,
    calibrate_probs: bool = False,
    build_ensemble: bool = True,
) -> dict:
    """
    Train selected models and return comprehensive metrics, best model, and explanations.
    """
    n_jobs = safe_n_jobs(n_jobs)
    model_registry = (
        CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    )

    valid_keys = [k for k in selected_model_keys if k in model_registry]
    if not valid_keys:
        valid_keys = ["rf", "logreg"] if task_type == "classification" else ["rf", "linreg"]

    # ── 1. Prepare X and y ────────────────────────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in dataset columns: {list(df.columns)}")

    valid_target_rows = df[target_col].dropna()
    if len(valid_target_rows) < 2:
        raise ValueError(
            f"Target column '{target_col}' has fewer than 2 valid non-null rows ({len(valid_target_rows)} valid rows). "
            f"Please select a target column with sufficient data."
        )

    df_clean = df.copy().dropna(subset=[target_col])
    if len(df_clean) < 2:
        raise ValueError(
            f"Target column '{target_col}' has insufficient valid samples to train ({len(df_clean)} rows). "
            f"Please select a valid target column."
        )

    fe_columns = []
    if enable_autofe:
        from dia.feature_engineer import auto_engineer_features
        df_clean, fe_columns = auto_engineer_features(df_clean, target_col, task_type)

    feature_cols = [
        c for c in df_clean.columns
        if c != target_col
        and df_clean[c].nunique(dropna=True) > 1
        and not ((df_clean[c].dtype == object or pd.api.types.is_string_dtype(df_clean[c])) and df_clean[c].nunique() > 30 and df_clean[c].nunique() / len(df_clean) > 0.5)
    ]

    # Fallback: if all features got pruned, keep remaining columns
    if not feature_cols:
        feature_cols = [c for c in df_clean.columns if c != target_col]
        if not feature_cols:
            raise ValueError("No feature columns available to train on after excluding the target column.")

    X = df_clean[feature_cols]
    y_raw = df_clean[target_col]

    le = None
    if task_type == "classification":
        y_clean_str = y_raw.astype(str).str.strip()
        unique_classes = y_clean_str.unique()
        if len(unique_classes) < 2:
            raise ValueError(
                f"Target column '{target_col}' has only 1 distinct class ({list(unique_classes)}). "
                f"Classification models require at least 2 distinct classes to train."
            )
        le = LabelEncoder()
        y = le.fit_transform(y_clean_str)
    else:
        y_num = pd.to_numeric(
            y_raw.astype(str).str.replace(r"[^\d.\-+eE]", "", regex=True),
            errors="coerce"
        )
        valid_mask = ~pd.isna(y_num)
        # If fewer than 2 valid numeric samples OR > 50% became NaN when converting,
        # the target column is actually categorical/discrete! Auto-fallback to classification!
        if valid_mask.sum() < 2 or (valid_mask.sum() / len(y_raw) < 0.5):
            log.warning(
                "Task type was '%s' but target column '%s' cannot be parsed as numeric (valid numeric ratio: %.1f%%). "
                "Automatically switching task_type to 'classification'.",
                task_type, target_col, (valid_mask.sum() / len(y_raw)) * 100
            )
            task_type = "classification"
            model_registry = CLASSIFICATION_MODELS
            y_clean_str = y_raw.astype(str).str.strip()
            unique_classes = y_clean_str.unique()
            if len(unique_classes) < 2:
                raise ValueError(
                    f"Target column '{target_col}' has only 1 distinct class ({list(unique_classes)}). "
                    f"Classification models require at least 2 distinct classes to train."
                )
            le = LabelEncoder()
            y = le.fit_transform(y_clean_str)
            # update valid keys if needed
            valid_keys = [k for k in selected_model_keys if k in CLASSIFICATION_MODELS]
            if not valid_keys:
                valid_keys = ["rf", "logreg"]
        else:
            X = X[valid_mask]
            y = y_num[valid_mask].values

    if len(X) < 2 or len(y) < 2:
        raise ValueError(
            f"Target column '{target_col}' has insufficient valid samples to train ({len(X)} valid rows). "
            f"Please verify that the target column contains valid, non-null values."
        )

    # Dynamically adjust test_size if dataset is very small to avoid empty train/test split
    actual_test_size = test_size
    if len(X) * actual_test_size < 1:
        actual_test_size = 1 / len(X)
    if len(X) * (1 - actual_test_size) < 1:
        actual_test_size = 0.5

    stratify_target = None
    if task_type == "classification":
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) <= 20 and np.all(counts >= 2):
            stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, random_state=random_state,
        stratify=stratify_target,
    )

    preprocessor = _build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = _get_feature_names(preprocessor, X_train)

    sample_weight = None
    if handle_imbalance and task_type == "classification":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results = []
    best_score = -np.inf
    best_result = None
    fitted_estimators: list[tuple[str, Any]] = []

    for key in valid_keys:
        model_def = model_registry[key]
        estimator = model_def["factory"](use_gpu=use_gpu, n_jobs=n_jobs)

        if estimator is None:
            results.append({
                "model_key": key,
                "label": model_def["label"],
                "error": f"{model_def['label']} is not available.",
                "metrics": {},
                "importance": pd.Series(dtype=float),
                "best_params": {},
            })
            continue

        try:
            best_params = {}
            # ── HPO Tuning ────────────────────────────────────────────────────
            if enable_hpo and key in HPO_PARAM_GRIDS:
                param_dist = HPO_PARAM_GRIDS[key]
                if task_type == "classification":
                    unique_cls, cls_counts = np.unique(y_train, return_counts=True)
                    min_count = int(np.min(cls_counts)) if len(unique_cls) > 0 else 0
                    if min_count >= 3:
                        cv_hpo = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
                    elif min_count >= 2:
                        cv_hpo = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
                    else:
                        cv_hpo = KFold(n_splits=min(3, max(2, len(y_train))), shuffle=True, random_state=random_state)

                    if len(unique_cls) == 2 and min_count >= 2:
                        scoring_hpo = 'roc_auc'
                    elif len(unique_cls) > 2 and min_count >= 2:
                        scoring_hpo = 'roc_auc_ovr'
                    else:
                        scoring_hpo = 'accuracy'
                else:
                    cv_hpo = KFold(n_splits=min(3, max(2, len(y_train))), shuffle=True, random_state=random_state)
                    scoring_hpo = 'r2'

                try:
                    hpo_search = RandomizedSearchCV(
                        estimator=estimator,
                        param_distributions=param_dist,
                        n_iter=min(hpo_iter, 20),
                        scoring=scoring_hpo,
                        cv=cv_hpo,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        error_score=np.nan,
                    )

                    if sample_weight is not None:
                        try:
                            hpo_search.fit(X_train_proc, y_train, sample_weight=sample_weight)
                        except Exception:
                            hpo_search.fit(X_train_proc, y_train)
                    else:
                        hpo_search.fit(X_train_proc, y_train)

                    if hasattr(hpo_search, "best_estimator_") and hpo_search.best_estimator_ is not None:
                        estimator = hpo_search.best_estimator_
                        best_params = hpo_search.best_params_
                except Exception as hpo_err:
                    log.warning("HPO search encountered issue (%s), retaining base estimator", hpo_err)

            # ── Cross-Validation ──────────────────────────────────────────────
            cv_metrics = {}
            if apply_cv:
                if task_type == "classification":
                    unique_cls, cls_counts = np.unique(y_train, return_counts=True)
                    n_classes = len(unique_cls)
                    min_count = int(np.min(cls_counts)) if n_classes > 0 else 0
                    if min_count >= 5:
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                    elif min_count >= 2:
                        cv = StratifiedKFold(n_splits=min_count, shuffle=True, random_state=random_state)
                    else:
                        cv = KFold(n_splits=min(5, max(2, len(y_train))), shuffle=True, random_state=random_state)

                    if n_classes == 2 and min_count >= 2:
                        scoring = 'roc_auc'
                    elif n_classes > 2 and min_count >= 2:
                        scoring = 'roc_auc_ovr'
                    else:
                        scoring = 'accuracy'
                else:
                    cv = KFold(n_splits=min(5, max(2, len(y_train))), shuffle=True, random_state=random_state)
                    scoring = 'r2'

                cv_res = None
                for candidate_scoring in [scoring, 'accuracy' if task_type == "classification" else 'r2']:
                    try:
                        cv_res = cross_validate(
                            estimator, X_train_proc, y_train,
                            cv=cv, scoring=candidate_scoring, n_jobs=n_jobs,
                            params={'sample_weight': sample_weight} if sample_weight is not None else None,
                            error_score=np.nan
                        )
                        break
                    except TypeError:
                        try:
                            cv_res = cross_validate(
                                estimator, X_train_proc, y_train,
                                cv=cv, scoring=candidate_scoring, n_jobs=n_jobs,
                                fit_params={'sample_weight': sample_weight} if sample_weight is not None else None,
                                error_score=np.nan
                            )
                            break
                        except Exception:
                            pass
                    except Exception:
                        pass

                if cv_res is not None and 'test_score' in cv_res:
                    valid_scores = [s for s in cv_res['test_score'] if not np.isnan(s)]
                    if valid_scores:
                        cv_mean = float(np.mean(valid_scores))
                        cv_std = float(np.std(valid_scores))
                        cv_metrics = {"CV-Score": round(cv_mean, 4), "CV-Std": round(cv_std, 4)}

            # Final fit on full training split
            if not (enable_hpo and key in HPO_PARAM_GRIDS):
                if sample_weight is not None:
                    try:
                        estimator.fit(X_train_proc, y_train, sample_weight=sample_weight)
                    except Exception:
                        estimator.fit(X_train_proc, y_train)
                else:
                    estimator.fit(X_train_proc, y_train)

            # Optional Probability Calibration for classification
            if calibrate_probs and task_type == "classification" and hasattr(estimator, "predict_proba"):
                try:
                    calibrator = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv="prefit")
                    calibrator.fit(X_test_proc, y_test)
                    estimator = calibrator
                except Exception as e:
                    log.debug("Calibration skipped: %s", e)

            fitted_estimators.append((key, estimator))
            y_pred = estimator.predict(X_test_proc)

            if task_type == "classification":
                y_proba = (
                    estimator.predict_proba(X_test_proc)
                    if hasattr(estimator, "predict_proba") else None
                )
                metrics = _classification_metrics(y_test, y_pred, y_proba)
                primary_score = metrics.get("ROC-AUC", metrics.get("F1", 0))
            else:
                metrics = _regression_metrics(y_test, y_pred)
                primary_score = metrics.get("R²", -np.inf)

            metrics.update(cv_metrics)

            # Calculate Overfitting / Generalization Gap
            train_pred = estimator.predict(X_train_proc)
            train_score = (
                accuracy_score(y_train, train_pred)
                if task_type == "classification"
                else r2_score(y_train, train_pred)
            )
            test_eval_score = (
                metrics.get("Accuracy", 0) if task_type == "classification" else metrics.get("R²", 0)
            )
            gap = max(0.0, float(train_score - test_eval_score))
            metrics["Generalization Gap"] = round(gap, 3)

            importance = _extract_importance(estimator, feature_names)

            entry = {
                "model_key": key,
                "label": model_def["label"],
                "metrics": metrics,
                "importance": importance,
                "estimator": estimator,
                "error": None,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba if task_type == "classification" else None,
                "best_params": best_params,
            }
            results.append(entry)

            if primary_score > best_score:
                best_score = primary_score
                best_result = entry

        except Exception as exc:
            log.warning("Model %s failed: %s", key, exc)
            results.append({
                "model_key": key,
                "label": model_def["label"],
                "error": str(exc),
                "metrics": {},
                "importance": pd.Series(dtype=float),
                "best_params": {},
            })

    # ── 2. Automatic Blended Meta-Ensemble Strategy ───────────────────────────
    if build_ensemble and len(fitted_estimators) >= 2:
        try:
            ensemble_estimators = fitted_estimators[:3]
            if task_type == "classification":
                # Check all models support predict_proba
                proba_estimators = [(k, est) for k, est in ensemble_estimators if hasattr(est, "predict_proba")]
                if len(proba_estimators) >= 2:
                    ensemble = VotingClassifier(estimators=proba_estimators, voting="soft")
                    ensemble.fit(X_train_proc, y_train)
                    ens_pred = ensemble.predict(X_test_proc)
                    ens_proba = ensemble.predict_proba(X_test_proc)
                    ens_metrics = _classification_metrics(y_test, ens_pred, ens_proba)
                    ens_score = ens_metrics.get("ROC-AUC", ens_metrics.get("F1", 0))

                    ens_entry = {
                        "model_key": "voting_ensemble",
                        "label": "Soft Voting Ensemble (Top 3)",
                        "metrics": ens_metrics,
                        "importance": pd.Series(dtype=float),
                        "estimator": ensemble,
                        "error": None,
                        "y_true": y_test,
                        "y_pred": ens_pred,
                        "y_proba": ens_proba,
                        "best_params": {"voting": "soft", "base_models": [k for k, _ in proba_estimators]},
                    }
                    results.append(ens_entry)
                    if ens_score > best_score:
                        best_score = ens_score
                        best_result = ens_entry

            else:
                ensemble = VotingRegressor(estimators=ensemble_estimators)
                ensemble.fit(X_train_proc, y_train)
                ens_pred = ensemble.predict(X_test_proc)
                ens_metrics = _regression_metrics(y_test, ens_pred)
                ens_score = ens_metrics.get("R²", -np.inf)

                ens_entry = {
                    "model_key": "voting_ensemble",
                    "label": "Voting Ensemble (Top 3)",
                    "metrics": ens_metrics,
                    "importance": pd.Series(dtype=float),
                    "estimator": ensemble,
                    "error": None,
                    "y_true": y_test,
                    "y_pred": ens_pred,
                    "y_proba": None,
                    "best_params": {"base_models": [k for k, _ in ensemble_estimators]},
                }
                results.append(ens_entry)
                if ens_score > best_score:
                    best_score = ens_score
                    best_result = ens_entry

        except Exception as e:
            log.warning("Blended ensemble generation skipped: %s", e)

    if best_result is None:
        raise RuntimeError("All selected models failed. Check the error messages above.")

    # ── 3. Generate Justification Text ────────────────────────────────────────
    successful_models = [r for r in results if not r.get("error")]
    metric_name = "ROC-AUC" if task_type == "classification" and "ROC-AUC" in best_result["metrics"] else ("F1" if task_type == "classification" else "R²")

    if len(successful_models) > 1:
        runner_up = sorted(successful_models, key=lambda x: x["metrics"].get(metric_name, -np.inf), reverse=True)[1]
        runner_up_score = runner_up["metrics"].get(metric_name, 0)
        diff = best_score - runner_up_score

        if diff < 0.01:
            justification_text = f"**{best_result['label']}** was selected as the best model. It achieved an {metric_name} of **{best_score:.3f}**, in a statistical tie (< 1% difference) with {runner_up['label']} ({runner_up_score:.3f})."
        else:
            justification_text = f"**{best_result['label']}** was decisively selected as the best model, achieving an {metric_name} of **{best_score:.3f}**. This outperformed {runner_up['label']} (score: {runner_up_score:.3f}) by **+{diff:.3f} points**."
    else:
        justification_text = f"**{best_result['label']}** was selected. It achieved an {metric_name} of **{best_score:.3f}**."

    # Sweep temporary objects and caches
    try:
        del X_train_proc, y_train
    except Exception:
        pass
    gc.collect()

    return {
        "results": results,
        "best_model_key": best_result["model_key"],
        "best_model_label": best_result["label"],
        "best_metrics": best_result["metrics"],
        "best_params": best_result.get("best_params", {}),
        "X_test_processed": X_test_proc,
        "best_model": best_result["estimator"],
        "feature_names": feature_names,
        "task_type": task_type,
        "label_encoder": le,
        "preprocessor": preprocessor,
        "raw_feature_cols": feature_cols,
        "fe_columns": fe_columns,
        "best_importance": best_result.get("importance", pd.Series(dtype=float)),
        "y_true": best_result["y_true"],
        "y_pred": best_result["y_pred"],
        "y_proba": best_result["y_proba"],
        "justification": justification_text,
    }
