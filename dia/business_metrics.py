"""
dia/business_metrics.py
───────────────────────
Translates machine learning technical metrics into business impact and ROI.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_classification_roi(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    cost_fp: float, 
    cost_fn: float, 
    val_tp: float, 
    val_tn: float
) -> dict:
    """
    Calculates the financial ROI of a classification model based on a cost-benefit matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Binary classification expected
    if cm.shape != (2, 2):
        return {"error": "ROI calculation currently supports binary classification only."}
        
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    total_value = (tp * val_tp) + (tn * val_tn)
    net_roi = total_value - total_cost
    
    # Compare against a baseline of "guess everything is negative"
    baseline_fn = tp + fn
    baseline_tn = tn + fp
    baseline_cost = (baseline_fn * cost_fn)
    baseline_value = (baseline_tn * val_tn)
    baseline_roi = baseline_value - baseline_cost
    
    savings = net_roi - baseline_roi
    
    return {
        "net_roi": net_roi,
        "total_cost": total_cost,
        "total_value": total_value,
        "baseline_roi": baseline_roi,
        "model_savings": savings,
        "breakdown": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }
    }

def calculate_regression_impact(
    mae: float, 
    n_predictions: int,
    avg_target_value: float
) -> dict:
    """
    Translates MAE into business error costs.
    """
    total_error = mae * n_predictions
    error_margin_pct = (mae / avg_target_value) * 100 if avg_target_value else 0
    
    return {
        "total_error": total_error,
        "avg_error_per_prediction": mae,
        "error_margin_pct": error_margin_pct
    }

def generate_actionable_recommendations(
    feature_importances: pd.Series,
    insights: list[dict],
    archetype: str = "custom"
) -> list[str]:
    """
    Combines feature importance and correlation insights to generate 
    plain-English business actions tailored to the chosen business archetype.
    """
    actions = []
    
    # Only look at top 3 features
    top_features = feature_importances.head(3).index.tolist()
    
    # Define vocabulary based on archetype
    vocab = {
        "retention": {"metric": "churn risk", "increase": "improve retention", "decrease": "prevent churn"},
        "lead": {"metric": "conversion probability", "increase": "boost conversion", "decrease": "avoid wasted outreach"},
        "fraud": {"metric": "fraud likelihood", "increase": "flag for review", "decrease": "fast-track approval"},
        "custom": {"metric": "the target outcome", "increase": "increase this metric", "decrease": "decrease this metric"},
    }
    
    arch_vocab = vocab.get(archetype, vocab["custom"])
    
    for insight in insights:
        # Check if the insight is about a top feature
        for feature in top_features:
            if feature in insight["title"]:
                # If feature is positively correlated with the target
                if insight["type"] == "positive":
                    action = f"**Focus on `{feature}`:** This strongly increases {arch_vocab['metric']}. Strategies to maximize this will {arch_vocab['increase']}."
                else:
                    action = f"**Focus on `{feature}`:** This strongly decreases {arch_vocab['metric']}. Strategies to minimize this will {arch_vocab['decrease']}."
                    
                if action not in actions:
                    actions.append(action)
                    
    if not actions:
        actions.append("The model relies on complex interactions. Consider A/B testing broad strategies rather than focusing on single metrics.")
        
    return actions
