"""
dia/fairness.py
───────────────
Analyzes models for disparate impact and extracts segment rules (Ideal Customer Profiles).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


def extract_surrogate_rules(X: pd.DataFrame, y_pred: np.ndarray, feature_names: list[str]) -> str:
    """
    Trains a shallow decision tree on the model's predictions to extract
    plain-English rules describing the 'Ideal Profile' (the positive class).
    """
    if len(np.unique(y_pred)) != 2:
        return "Rule extraction requires a binary classification target."

    # Train a shallow tree on the predictions
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    # Ensure X is dense if it's a sparse matrix or DataFrame
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = X

    dt.fit(X_dense, y_pred)

    if feature_names and len(feature_names) == dt.n_features_in_:
        rules = export_text(dt, feature_names=feature_names)
    else:
        rules = export_text(dt)
    return rules

def check_disparate_impact(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categorical_cols: list[str]
) -> list[dict]:
    """
    Checks if the model performs significantly differently across different groups
    of categorical features (like Region, Department, etc.).
    """
    alerts = []

    # We need the original dataframe (test set indices) to map back the categories.
    # To keep this simple and decoupled, we assume df contains the test set slice
    # matching y_true and y_pred.

    if len(df) != len(y_true):
        # Fallback if dimensions don't match (e.g. dropped NA rows)
        return [{"feature": "N/A", "alert": "Could not align test set with original dataset for fairness check."}]

    overall_acc = accuracy_score(y_true, y_pred)

    for col in categorical_cols:
        if col not in df.columns:
            continue

        value_counts = df[col].value_counts()
        # Only check groups with a reasonable sample size
        valid_groups = value_counts[value_counts > max(10, len(df)*0.05)].index

        group_metrics = []
        for group in valid_groups:
            mask = (df[col] == group)
            if sum(mask) > 0:
                group_acc = accuracy_score(y_true[mask], y_pred[mask])
                group_metrics.append((group, group_acc))

        if len(group_metrics) > 1:
            # Check for a >15% drop in accuracy compared to the overall average
            for group, acc in group_metrics:
                if (overall_acc - acc) > 0.15:
                    alerts.append({
                        "feature": col,
                        "group": str(group),
                        "accuracy": acc,
                        "overall_accuracy": overall_acc,
                        "alert": f"The model is significantly less accurate for `{col}` = **{group}** ({acc:.0%} vs overall {overall_acc:.0%})."
                    })

    return alerts
