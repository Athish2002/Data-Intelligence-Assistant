"""
dia/streaming_learner.py
────────────────────────
Streaming Ingestion & Online Incremental Learning Engine.
Simulates streaming data events, updates model weights incrementally
using partial_fit (SGD/Passive-Aggressive), and tracks online accuracy curves.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

log = logging.getLogger("dia.streaming")


def simulate_streaming_incremental_fit(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "classification",
    batch_size: int = 20,
    n_batches: int = 10,
) -> dict[str, Any]:
    """
    Simulates a live event stream where the model adapts weights online via partial_fit.
    """
    n_samples = len(X)
    actual_batches = min(n_batches, max(1, n_samples // batch_size))

    if task_type == "classification":
        classes = np.unique(y)
        model = SGDClassifier(loss="log_loss", random_state=42)
        # Initialize
        model.partial_fit(X[:batch_size], y[:batch_size], classes=classes)
    else:
        model = SGDRegressor(random_state=42)
        model.partial_fit(X[:batch_size], y[:batch_size])

    online_history = []
    total_processed = 0

    for b in range(actual_batches):
        start = b * batch_size
        end = min(n_samples, start + batch_size)
        if start >= n_samples:
            break

        X_b, y_b = X[start:end], y[start:end]
        
        # Online test-then-train (Prequential evaluation)
        t0 = time.perf_counter()
        preds = model.predict(X_b)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if task_type == "classification":
            score = accuracy_score(y_b, preds)
            metric_name = "accuracy"
        else:
            score = np.sqrt(mean_squared_error(y_b, preds))
            metric_name = "rmse"

        # Online weight update
        model.partial_fit(X_b, y_b)
        total_processed += len(X_b)

        online_history.append({
            "batch_number": b + 1,
            "samples_processed": total_processed,
            metric_name: round(float(score), 4),
            "step_latency_ms": round(float(latency_ms), 2),
        })

    return {
        "status": "success",
        "task_type": task_type,
        "total_streaming_batches": len(online_history),
        "total_records_ingested": total_processed,
        "streaming_learning_curve": online_history,
        "final_online_metric": online_history[-1][metric_name] if online_history else 0.0,
    }
