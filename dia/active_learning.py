"""
dia/active_learning.py
──────────────────────
Active Learning & Human-in-the-Loop (HITL) Uncertainty Sampling.
Identifies borderline predictions with highest entropy/uncertainty
and queues them for domain expert human review and relabeling.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("dia.active_learning")


def sample_uncertain_predictions(
    model: Any,
    df_raw: pd.DataFrame,
    X_processed: np.ndarray,
    n_samples: int = 10,
) -> dict[str, Any]:
    """
    Ranks predictions by uncertainty (margin / entropy) to find ambiguous cases.
    """
    if len(X_processed) == 0:
        return {"uncertain_samples": []}

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_processed)
        # Margin sampling: difference between top 2 highest probabilities (smaller = more uncertain)
        sorted_p = np.sort(probas, axis=1)
        if probas.shape[1] > 1:
            margins = sorted_p[:, -1] - sorted_p[:, -2]
            uncertainty_scores = 1.0 - margins
        else:
            uncertainty_scores = np.abs(probas[:, 0] - 0.5) * 2.0
    else:
        # Distance to decision function for SVM/Linear models
        if hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_processed)
            uncertainty_scores = 1.0 / (1.0 + np.abs(df_vals))
        else:
            uncertainty_scores = np.random.uniform(0.4, 0.6, size=len(X_processed))

    top_uncertain_indices = np.argsort(uncertainty_scores)[::-1][:n_samples]

    queue = []
    for idx in top_uncertain_indices:
        row_dict = df_raw.iloc[idx].to_dict()
        pred_val = model.predict(X_processed[idx : idx + 1])[0]
        unc_val = float(uncertainty_scores[idx])

        queue.append({
            "original_row_index": int(idx),
            "model_prediction": int(pred_val) if isinstance(pred_val, int | np.integer) else float(pred_val),
            "uncertainty_score": round(unc_val, 4),
            "status": "Awaiting Human Review",
            "features_preview": {k: v for k, v in list(row_dict.items())[:6]},
        })

    return {
        "total_uncertain_flagged": len(queue),
        "review_queue": queue,
    }
