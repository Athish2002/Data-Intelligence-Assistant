"""
tests/test_innovations_suite.py
───────────────────────────────
Comprehensive test suite verifying the 5 Strategic Innovations:
R1. Automated Target Leakage & Data Poisoning Sleuth (dia/leakage_detector.py)
R2. Causal Discovery & Pearl's Do-Calculus Policy Simulator (dia/causal_discovery.py)
R3. Multi-Objective Pareto Frontier "Flight Simulator" (dia/pareto_frontier.py)
R4. Zero-Compute Client-Side Edge Model Transpiler (dia/wasm_compiler.py)
R5. Multi-Modal Tabular-Text Semantic Fusion Engine (dia/multimodal_fusion.py)
And their integrated FastAPI endpoints in api/server.py.
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from dia.leakage_detector import (
    detect_leakage,
    quarantine_features,
    TargetLeakageSleuth,
    LeakageReport,
    PoisoningReport,
    FeatureAssessment,
    LeakageSeverity,
)
from dia.causal_discovery import (
    discover_causal_graph,
    simulate_intervention,
    CausalGraphResult,
    InterventionResult,
)
from dia.pareto_frontier import (
    compute_pareto_frontier,
    ParetoFrontierSimulator,
    ParetoFrontierResult,
    ParetoPoint,
)
from dia.wasm_compiler import (
    transpile_to_edge_bundle,
    EdgeBundle,
)
from dia.multimodal_fusion import (
    MultiModalFusionEngine,
    detect_text_columns,
    fuse_tabular_and_text,
)
from api.server import (
    ingest_demo_dataset,
    get_target_leakage_report,
    get_causal_graph,
    simulate_causal_intervention,
    optimize_pareto_frontier,
    export_edge_bundle,
    InterventionRequest,
    ParetoOptimizationRequest,
)
from api.schemas import DemoIngestRequest


# ─── R1: Leakage & Poisoning Sleuth Tests ────────────────────────────────────

def test_leakage_detector_synthetic_proxy_and_id():
    """Verify target leakage detection with a perfect proxy and high-cardinality ID."""
    np.random.seed(42)
    n = 200
    target = np.random.binomial(1, 0.5, n)
    clean_feature = np.random.normal(0, 1, n)
    proxy_leakage = target * 10.0 + np.random.normal(0, 0.01, n)  # 0.999 correlation
    id_feature = [f"ID_{i:06d}" for i in range(n)]

    df = pd.DataFrame({
        "clean_feat": clean_feature,
        "proxy_leak": proxy_leakage,
        "customer_uuid": id_feature,
        "target": target,
    })

    report = detect_leakage(df, target_col="target")
    assert isinstance(report, LeakageReport)
    assert report.has_leakage is True
    assert "proxy_leak" in report.quarantine_features
    assert "customer_uuid" in report.id_memorization_features
    assert "clean_feat" in report.clean_features
    assert report.overall_leakage_risk_score > 40.0

    # Test quarantine DataFrame sanitization
    clean_df = quarantine_features(df, report)
    assert "proxy_leak" not in clean_df.columns
    assert "clean_feat" in clean_df.columns


def test_leakage_detector_temporal_inversion():
    """Verify detection of temporal violations where feature timestamp occurs after target."""
    n = 100
    dates_target = pd.date_range("2024-01-01", periods=n, freq="D")
    dates_feature_future = pd.date_range("2024-02-01", periods=n, freq="D")  # In future!

    df = pd.DataFrame({
        "target_date": dates_target,
        "future_event_date": dates_feature_future,
        "metric": np.random.randn(n),
        "label": np.random.choice([0, 1], n),
    })

    sleuth = TargetLeakageSleuth()
    report = sleuth.analyze(df, target_col="label", time_col="target_date")
    assert report.temporal_leakage_detected is True


def test_leakage_detector_conflicting_label_poisoning():
    """Verify detection of conflicting duplicate label poisoning."""
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 1.0, 3.0],
        "feature_b": [10.0, 20.0, 10.0, 30.0],
        "target": [0, 1, 1, 0],  # Row 0 and Row 2 have identical features but conflicting labels!
    })

    sleuth = TargetLeakageSleuth()
    report = sleuth.analyze(df, target_col="target")
    assert report.poisoning_detected is True
    assert report.poisoning_report.conflict_count >= 1


# ─── R2: Causal Discovery & Do-Calculus Simulator Tests ───────────────────────

def test_causal_discovery_graph_induction():
    """Verify PC constraint-based causal DAG discovery."""
    np.random.seed(42)
    n = 250
    # True causal structure: Z -> X -> Y
    z = np.random.normal(0, 1, n)
    x = 0.8 * z + np.random.normal(0, 0.5, n)
    y = 1.2 * x + np.random.normal(0, 0.5, n)
    w = np.random.normal(0, 1, n)  # Independent noise

    df = pd.DataFrame({"Z": z, "X": x, "Y": y, "W": w})
    graph_res = discover_causal_graph(df, target_col="Y", alpha=0.05)

    assert isinstance(graph_res, CausalGraphResult)
    assert "X" in graph_res.nodes
    assert "Y" in graph_res.nodes
    assert len(graph_res.edges) > 0
    assert any((e.source == "X" and e.target == "Y") or (e.source == "Y" and e.target == "X") for e in graph_res.edges)


def test_causal_intervention_do_calculus():
    """Verify Pearl's Do-Calculus intervention E[Y | do(X = x)]."""
    np.random.seed(42)
    n = 200
    x = np.random.uniform(10, 50, n)
    y = 2.5 * x + np.random.normal(0, 2, n)
    df = pd.DataFrame({"Income": x, "Spending": y})

    res = simulate_intervention(
        df=df,
        treatment="Income",
        outcome="Spending",
        intervention_value=40.0,
    )
    assert isinstance(res, InterventionResult)
    assert res.baseline_expected_outcome > 0
    assert res.intervened_expected_outcome > 0
    assert res.policy_interpretation != ""
    assert res.confidence_interval_95 is not None


# ─── R3: Pareto Frontier Flight Simulator Tests ──────────────────────────────

def test_pareto_frontier_optimizer():
    """Verify multi-objective non-dominated Pareto sorting and knee-point detection."""
    models_data = [
        {"model_name": "Model_HighProfit_HighRisk", "roi_profit": 50000.0, "default_risk": 0.25, "fairness_ratio": 0.70},
        {"model_name": "Model_LowProfit_LowRisk", "roi_profit": 20000.0, "default_risk": 0.05, "fairness_ratio": 0.85},
        {"model_name": "Model_Balanced_Knee", "roi_profit": 42000.0, "default_risk": 0.10, "fairness_ratio": 0.90},
        {"model_name": "Model_Dominated_Bad", "roi_profit": 15000.0, "default_risk": 0.30, "fairness_ratio": 0.50},
        {"model_name": "Model_MaxFairness", "roi_profit": 30000.0, "default_risk": 0.12, "fairness_ratio": 0.98},
    ]

    res = compute_pareto_frontier(
        models_data=models_data,
        profit_weight=1.0,
        risk_weight=1.0,
        fairness_weight=1.0,
    )

    assert isinstance(res, ParetoFrontierResult)
    assert len(res.frontier_points) >= 3
    # Model_Dominated_Bad must NOT be in the Pareto frontier
    frontier_names = [p.model_name for p in res.frontier_points]
    assert "Model_Dominated_Bad" not in frontier_names
    assert res.knee_point is not None
    assert res.hypervolume > 0.0
    assert len(res.flight_recommendations) > 0


def test_pareto_simulator_threshold_sweeper():
    """Verify threshold sweeper creates non-dominated points."""
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], n, p=[0.7, 0.3])
    y_prob = np.random.uniform(0.0, 1.0, n)
    protected = pd.Series(np.random.choice(["GroupA", "GroupB"], n))

    sim = ParetoFrontierSimulator(steps=20)
    sim_res = sim.compute_frontier(
        y_true=y_true,
        y_prob=y_prob,
        protected_series=protected,
        cost_fp=20.0,
        cost_fn=100.0,
        val_tp=50.0,
        val_tn=0.0,
    )
    assert isinstance(sim_res, ParetoFrontierResult)
    assert len(sim_res.all_evaluated_points) == 20
    assert len(sim_res.frontier_points) > 0
    assert sim_res.knee_point is not None


# ─── R4: Edge JS/WASM Transpiler Tests ───────────────────────────────────────

def test_wasm_transpiler_logistic_regression():
    """Verify LogisticRegression transpilation to JS and Node.js inference."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]

    clf = LogisticRegression()
    clf.fit(X, y)

    bundle = transpile_to_edge_bundle(clf, feature_names)
    assert isinstance(bundle, EdgeBundle)
    assert bundle.bundle_size_kb < 500.0
    assert "DiaEdgeEngine" in bundle.js_code
    assert bundle.wat_code is not None

    # Test Node.js execution of bundle
    test_record = {"feat_0": 1.2, "feat_1": -0.5, "feat_2": 0.0, "feat_3": 0.8}
    js_test_script = f"""
    {bundle.js_code}
    const rec = {test_record};
    const proba = DiaEdgeEngine.predictProba(rec);
    console.log(JSON.stringify(proba));
    """

    with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w") as f:
        f.write(js_test_script)
        temp_path = f.name

    try:
        proc = subprocess.run(["node", temp_path], capture_output=True, text=True, timeout=5)
        assert proc.returncode == 0
        output_str = proc.stdout.strip()
        assert "0" in output_str and "1" in output_str
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_wasm_transpiler_decision_tree():
    """Verify DecisionTreeClassifier transpilation to JS and Node.js inference."""
    np.random.seed(42)
    X = np.random.randn(60, 3)
    y = (X[:, 0] > 0).astype(int)
    feature_names = ["x0", "x1", "x2"]

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    bundle = transpile_to_edge_bundle(dt, feature_names)
    assert bundle.model_type == "DecisionTreeClassifier"
    assert bundle.bundle_size_kb < 100.0

    js_test_script = f"""
    {bundle.js_code}
    const proba = DiaEdgeEngine.predictProba({{ "x0": 1.5, "x1": 0.0, "x2": -1.0 }});
    console.log(JSON.stringify(proba));
    """
    with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w") as f:
        f.write(js_test_script)
        temp_path = f.name

    try:
        proc = subprocess.run(["node", temp_path], capture_output=True, text=True, timeout=5)
        assert proc.returncode == 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_wasm_transpiler_random_forest():
    """Verify RandomForestClassifier transpilation to standalone JS ensemble."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 1] > 0).astype(int)
    feature_names = ["f0", "f1", "f2"]

    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    rf.fit(X, y)

    bundle = transpile_to_edge_bundle(rf, feature_names)
    assert bundle.model_type == "RandomForestClassifier"
    assert "trees" in bundle.js_code
    assert bundle.bundle_size_kb < 500.0


# ─── R5: Multi-Modal Tabular-Text Semantic Fusion Tests ───────────────────────

def test_multimodal_text_detection_and_fusion():
    """Verify unstructured free-text detection and dense semantic embedding fusion."""
    df = pd.DataFrame({
        "age": [25, 45, 32, 58, 29, 61],
        "income": [50000.0, 95000.0, 62000.0, 110000.0, 54000.0, 125000.0],
        "notes": [
            "Customer requested loan approval for purchasing commercial enterprise assets in urban center.",
            "Applicant reports stellar financial background with no delinquencies on revolving credit lines.",
            "Short term borrower seeking immediate debt consolidation loan with steady income stream.",
            "Enterprise executive investing in real estate commercial mortgage with pristine credit history.",
            "New account holder applying for small unsecured personal credit line with co-signer.",
            "High net worth account seeking commercial expansion funding with extensive collateral assets.",
        ],
        "approved": [1, 1, 0, 1, 0, 1],
    })

    text_cols = detect_text_columns(df)
    assert "notes" in text_cols
    assert "age" not in text_cols

    engine = MultiModalFusionEngine(n_components=4, drop_raw_text=True)
    fusion_result = engine.fit_transform(df, target_col="approved")
    fused_df = fusion_result.fused_df

    assert "notes" not in fused_df.columns  # Original text replaced by embeddings
    assert any("text_notes_svd" in c for c in fused_df.columns)
    assert "age" in fused_df.columns
    assert "income" in fused_df.columns
    assert fused_df.shape[0] == 6
    assert not fused_df.isnull().values.any()
    assert len(fusion_result.new_semantic_features) == 4

    # Also test functional API
    fused_func_df, meta = fuse_tabular_and_text(df, n_components=4, drop_raw_text=True)
    assert fused_func_df.shape[0] == 6
    assert len(meta["dense_feature_names"]) == 4


# ─── Integration REST API Endpoints Tests ────────────────────────────────────

def test_api_innovations_integrated_endpoints():
    """Verify live endpoint calls for R1-R4 innovations through FastAPI server."""
    # 1. Ingest demo dataset
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Telecom Customer Churn"))
    session_id = ingest_res.session_id
    assert session_id is not None

    # 2. Target Leakage Sleuth endpoint
    leakage_resp = get_target_leakage_report(session_id=session_id)
    assert leakage_resp.session_id == session_id
    assert isinstance(leakage_resp.overall_leakage_risk_score, float)
    assert isinstance(leakage_resp.quarantine_features, list)
    assert isinstance(leakage_resp.clean_features, list)

    # 3. Causal Graph endpoint
    causal_resp = get_causal_graph(session_id=session_id)
    assert causal_resp.session_id == session_id
    assert len(causal_resp.nodes) > 0
    assert isinstance(causal_resp.edges, list)

    # 4. Causal Intervention simulation endpoint
    # Use first two columns from nodes
    t_col = causal_resp.nodes[0]
    o_col = causal_resp.nodes[1]
    interv_req = InterventionRequest(
        session_id=session_id,
        treatment=t_col,
        outcome=o_col,
        intervention_value=1.0,
    )
    interv_resp = simulate_causal_intervention(interv_req)
    assert interv_resp.status == "success"
    assert isinstance(interv_resp.average_treatment_effect, float)

    # 5. Pareto Optimization endpoint
    pareto_req = ParetoOptimizationRequest(
        profit_weight=1.0,
        risk_weight=1.0,
        fairness_weight=1.0,
        cost_fp=20.0,
        cost_fn=150.0,
        benefit_tp=100.0,
    )
    pareto_resp = optimize_pareto_frontier(session_id=session_id, payload=pareto_req)
    assert pareto_resp.status == "success"
    assert len(pareto_resp.frontier_points) > 0
    assert pareto_resp.knee_point is not None

    # 6. Edge Bundle Transpiler endpoint
    edge_resp = export_edge_bundle(session_id=session_id, download=False)
    assert edge_resp.status == "success"
    assert "DiaEdgeEngine" in edge_resp.js_code
    assert edge_resp.bundle_size_kb < 500.0
    assert len(edge_resp.feature_names) > 0

    # 7. Edge Bundle attachment download endpoint
    download_resp = export_edge_bundle(session_id=session_id, download=True)
    assert download_resp.media_type == "application/javascript"
    assert "attachment" in download_resp.headers["Content-Disposition"]
