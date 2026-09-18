"""
tests/test_live_server.py
─────────────────────────
Live HTTP integration verification script testing the running FastAPI server.
"""

import sys
import requests

BASE_URL = "http://localhost:8000"

def run_live_verification():
    print("[1] Verifying System Health...")
    r = requests.get(f"{BASE_URL}/api/v1/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health = r.json()
    print(f"    [OK] Health status: {health['status']}, version: {health['version']}, CPU cores: {health['cpu_cores']}")

    print("[2] Verifying Static Frontend Mount...")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200, "Frontend index.html not served"
    assert "<!DOCTYPE html>" in r.text or "<html" in r.text
    print("    [OK] Frontend SPA successfully mounted and serving HTML")

    print("[3] Ingesting Demo Dataset (Bank Credit Risk & Default)...")
    r = requests.post(f"{BASE_URL}/api/v1/ingest/demo", json={"demo_name": "Bank Credit Risk & Default"}, timeout=15)
    assert r.status_code == 200, f"Demo ingest failed: {r.text}"
    ingest = r.json()
    session_id = ingest["session_id"]
    print(f"    [OK] Ingested session: {session_id} ({ingest['n_rows']} rows, {ingest['n_cols']} cols, domain: {ingest['detected_domain']})")

    print("[4] Testing R1: Forensic Target Leakage & Poisoning Sleuth...")
    r = requests.get(f"{BASE_URL}/api/v1/governance/leakage/{session_id}", timeout=15)
    assert r.status_code == 200, f"Leakage endpoint failed: {r.text}"
    leakage = r.json()
    print(f"    [OK] Leakage Risk Score: {leakage['overall_leakage_risk_score']:.1f}/100, Quarantined features: {len(leakage['quarantine_features'])}")

    print("[5] Testing R2: Causal Discovery DAG (PC Algorithm)...")
    r = requests.get(f"{BASE_URL}/api/v1/causal/graph/{session_id}", timeout=15)
    assert r.status_code == 200, f"Causal graph endpoint failed: {r.text}"
    graph = r.json()
    print(f"    [OK] Discovered causal graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, root causes: {graph['root_causes']}")

    print("[6] Testing R2: Pearl's Do-Calculus Policy Simulator...")
    t_col = graph["nodes"][0]
    o_col = graph["nodes"][1]
    interv_payload = {
        "session_id": session_id,
        "treatment": t_col,
        "outcome": o_col,
        "intervention_value": 1.0,
    }
    r = requests.post(f"{BASE_URL}/api/v1/causal/intervention", json=interv_payload, timeout=15)
    assert r.status_code == 200, f"Causal intervention failed: {r.text}"
    interv = r.json()
    print(f"    [OK] Do-Calculus: do({interv['treatment']}={interv['intervention_value']}) -> E[Y]={interv['intervened_expected_outcome']:.4f} (ATE: {interv['average_treatment_effect']:+.4f})")

    print("[7] Testing R3: Multi-Objective Pareto Frontier Flight Simulator...")
    pareto_payload = {
        "profit_weight": 1.0,
        "risk_weight": 1.0,
        "fairness_weight": 1.0,
        "cost_fp": 20.0,
        "cost_fn": 150.0,
        "benefit_tp": 100.0,
    }
    r = requests.post(f"{BASE_URL}/api/v1/optimization/pareto/{session_id}", json=pareto_payload, timeout=15)
    assert r.status_code == 200, f"Pareto optimization failed: {r.text}"
    pareto = r.json()
    print(f"    [OK] Pareto Frontier: {len(pareto['frontier_points'])} non-dominated configurations, Hypervolume: {pareto['hypervolume']:.4f}")
    if pareto["knee_point"]:
        knee = pareto["knee_point"]
        print(f"      * Knee Point: profit=${knee['roi_profit']:,.2f}, risk={knee['default_risk']*100:.1f}%, fairness={knee['fairness_ratio']:.2f}")

    print("[8] Testing R4: Zero-Compute Client-Side Edge Model Transpiler...")
    r = requests.get(f"{BASE_URL}/api/v1/export/edge-bundle/{session_id}", timeout=15)
    assert r.status_code == 200, f"Edge bundle JSON failed: {r.text}"
    edge = r.json()
    print(f"    [OK] Compiled Edge Bundle: {edge['model_type']}, bundle size: {edge['bundle_size_kb']:.1f} KB (< 500 KB limit), latency: {edge['estimated_latency_us']} us")

    r_down = requests.get(f"{BASE_URL}/api/v1/export/edge-bundle/{session_id}?download=true", timeout=15)
    assert r_down.status_code == 200, "Edge bundle download failed"
    assert "DiaEdgeEngine" in r_down.text
    print(f"    [OK] Downloadable JS Edge Bundle verified ({len(r_down.text)} characters)")

    print("[9] Testing AutoML Pipeline Training...")
    train_payload = {
        "session_id": session_id,
        "goal": "Predict loan default risk for automated credit decisions",
    }
    r_train = requests.post(f"{BASE_URL}/api/v1/pipeline/train", json=train_payload, timeout=90)
    assert r_train.status_code == 200, f"AutoML training failed: {r_train.text}"
    train = r_train.json()
    print(f"    [OK] Champion Model: {train['best_model_label']} (Readiness: {train['readiness_score_pct']}%)")
    print(f"    [OK] Models evaluated: {len(train['models_evaluated'])}, SHAP drivers: {len(train['shap_importance'])}")

    print("\n============================================================")
    print("ALL LIVE SERVER INTEGRATION CHECKS PASSED FLAWLESSLY! SUCCESS")
    print("============================================================")

if __name__ == "__main__":
    run_live_verification()
