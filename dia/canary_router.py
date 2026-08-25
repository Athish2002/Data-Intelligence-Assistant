"""
dia/canary_router.py
────────────────────
Canary & Shadow Deployment Live Traffic Router.
Simulates production traffic routing between Champion (Primary) and Challenger (Canary)
models, tracking error divergence, latency distribution, and triggering automated rollback.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

log = logging.getLogger("dia.canary")


def simulate_canary_routing(
    champion_model: Any,
    challenger_model: Any,
    X_live: np.ndarray,
    canary_traffic_pct: float = 10.0,
    max_allowed_divergence_pct: float = 15.0,
    task_type: str = "classification",
) -> dict[str, Any]:
    """
    Simulates production inference across incoming live requests, routing
    (100 - canary_traffic_pct)% to Champion and canary_traffic_pct to Challenger.
    Computes statistical divergence and evaluates automated rollback circuit breaker.
    """
    n_requests = len(X_live)
    if n_requests < 5:
        return {
            "status": "insufficient_data",
            "message": "At least 5 live samples required for canary simulation.",
        }

    canary_ratio = canary_traffic_pct / 100.0
    is_clf = task_type == "classification"

    champion_predictions = []
    challenger_predictions = []
    routed_destination = []
    latencies_champ = []
    latencies_chall = []

    for i in range(n_requests):
        sample = X_live[i : i + 1]

        # Champion prediction
        t0 = time.perf_counter()
        champ_pred = champion_model.predict(sample)[0]
        t_champ = (time.perf_counter() - t0) * 1000
        champion_predictions.append(champ_pred)
        latencies_champ.append(t_champ)

        # Challenger prediction (Shadow or Routed)
        t1 = time.perf_counter()
        chall_pred = challenger_model.predict(sample)[0]
        t_chall = (time.perf_counter() - t1) * 1000
        challenger_predictions.append(chall_pred)
        latencies_chall.append(t_chall)

        # Route decision
        if np.random.rand() < canary_ratio:
            routed_destination.append("Challenger (Canary)")
        else:
            routed_destination.append("Champion (Baseline)")

    champ_arr = np.array(champion_predictions)
    chall_arr = np.array(challenger_predictions)

    # Calculate prediction divergence between Champion and Challenger
    if is_clf:
        disagreements = int(np.sum(champ_arr != chall_arr))
        divergence_pct = round(float((disagreements / n_requests) * 100), 2)
    else:
        abs_diff = np.abs(champ_arr - chall_arr)
        divergence_pct = round(float(np.mean(abs_diff > np.std(champ_arr) * 0.5) * 100), 2)

    # Circuit Breaker / Automated Rollback Status
    rollback_triggered = divergence_pct > max_allowed_divergence_pct
    status_verdict = (
        "🚨 CIRCUIT BREAKER TRIGGERED: Automated Rollback to Champion"
        if rollback_triggered
        else "✅ CANARY STABLE: Challenger operating within safe bounds"
    )

    canary_routed_count = routed_destination.count("Challenger (Canary)")
    champ_routed_count = routed_destination.count("Champion (Baseline)")

    return {
        "status": "success",
        "total_requests_processed": n_requests,
        "canary_traffic_target_pct": canary_traffic_pct,
        "champion_requests_served": champ_routed_count,
        "canary_requests_served": canary_routed_count,
        "prediction_divergence_pct": divergence_pct,
        "max_allowed_divergence_pct": max_allowed_divergence_pct,
        "circuit_breaker_rollback_triggered": rollback_triggered,
        "status_verdict": status_verdict,
        "champion_latency_p99_ms": round(float(np.percentile(latencies_champ, 99)), 3),
        "challenger_latency_p99_ms": round(float(np.percentile(latencies_chall, 99)), 3),
    }
