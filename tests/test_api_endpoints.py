"""
tests/test_api_endpoints.py
────────────────────────────
Comprehensive integration test suite for FastAPI REST API endpoints.
Verifies all 4 workspaces: Core, AutoML, Adaptive AI Engines, and Governance.
"""

import pytest
from fastapi import HTTPException

from dia.utils import is_torch_available
from api.server import (
    chat_with_data,
    get_health,
    list_demo_datasets,
    ingest_demo_dataset,
    get_data_profile,
    get_readiness_audit,
    list_llm_providers,
    run_automl_pipeline,
    simulate_whatif,
    get_counterfactual,
    optimize_business_roi,
    get_active_learning_queue,
    get_autoencoder_analysis,
    simulate_bandits,
    generate_synthetic,
    get_data_contract_suite,
    get_gdpr_audit,
    get_all_artifacts,
    export_artifact,
)
from api.schemas import (
    ChatRequest,
    DemoIngestRequest,
    TrainPipelineRequest,
    SimulateRequest,
    CounterfactualRequest,
    RoiOptimizeRequest,
    BanditSimRequest,
    SyntheticGenerateRequest,
)


def test_health_endpoint():
    res = get_health()
    assert res.status == "healthy"
    assert res.cpu_cores > 0
    assert res.version == "13.6.0"


def test_list_demos_endpoint():
    demos = list_demo_datasets()
    assert len(demos) >= 4
    names = [d.name for d in demos]
    assert "Telecom Customer Churn" in names
    assert "Bank Credit Risk & Default" in names


def test_ingest_demo_dataset():
    res = ingest_demo_dataset(DemoIngestRequest(demo_name="Telecom Customer Churn"))
    assert res.session_id is not None
    assert res.n_rows > 0
    assert res.n_cols > 0
    assert len(res.columns) > 0
    assert res.capabilities["can_causal"] is True


def test_core_intelligence_endpoints():
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Telecom Customer Churn"))
    session_id = ingest_res.session_id

    # 1. Profile
    prof_res = get_data_profile(session_id)
    assert prof_res.shape[0] > 0
    assert len(prof_res.columns_info) > 0

    # 2. Readiness Audit
    read_res = get_readiness_audit(session_id)
    assert read_res.overall_readiness_score > 0
    assert len(read_res.checks) >= 4


def test_automl_training_and_inference_flow():
    # 1. Ingest
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Bank Credit Risk & Default"))
    session_id = ingest_res.session_id

    # 2. Train AutoML pipeline
    train_res = run_automl_pipeline(
        TrainPipelineRequest(
            session_id=session_id,
            goal="Predict loan default likelihood for underwriting",
        )
    )
    assert train_res.status == "success"
    assert train_res.best_model_label is not None
    assert len(train_res.models_evaluated) > 0
    assert len(train_res.shap_importance) > 0

    # 3. Simulate What-If scenario
    sample_features = {
        "annual_income": 85000.0,
        "credit_score": 720.0,
        "debt_to_income_ratio": 0.22,
        "loan_amount": 25000.0,
    }
    sim_res = simulate_whatif(
        SimulateRequest(session_id=session_id, feature_overrides=sample_features)
    )
    assert sim_res.status == "success"
    assert sim_res.prediction is not None

    # 4. Counterfactual search
    cf_res = get_counterfactual(
        CounterfactualRequest(session_id=session_id, row_index=0, desired_outcome=0)
    )
    assert cf_res.status in ("success", "already_desired", "approximate")

    # 5. Business ROI Optimizer
    roi_res = optimize_business_roi(
        RoiOptimizeRequest(session_id=session_id, benefit_tp=100.0, cost_fp=20.0, cost_fn=150.0)
    )
    assert roi_res.status in ("success", "not_applicable")

    # 6. Active Learning Queue
    al_res = get_active_learning_queue(session_id)
    assert al_res.status == "success"

    # 7. Deep Autoencoder (optional — needs torch, which is intentionally not a hard dependency)
    if is_torch_available():
        ae_res = get_autoencoder_analysis(session_id)
        assert ae_res.status == "success"
        assert ae_res.reconstruction_mae >= 0
    else:
        with pytest.raises(HTTPException) as exc_info:
            get_autoencoder_analysis(session_id)
        assert exc_info.value.status_code == 501

    # 8. Contextual Bandits
    ban_res = simulate_bandits(BanditSimRequest(session_id=session_id, n_steps=20))
    assert ban_res.status == "success"

    # 9. Synthetic Generation
    syn_res = generate_synthetic(SyntheticGenerateRequest(session_id=session_id, n_samples=50))
    assert syn_res.status == "success"
    assert syn_res.n_generated == 50

    # 10. Governance Data Contract
    contract_res = get_data_contract_suite(session_id)
    assert contract_res.status == "success"
    assert contract_res.n_expectations > 0

    # 11. GDPR Audit
    gdpr_res = get_gdpr_audit(session_id)
    assert gdpr_res.status == "success"

    # 12. Artifacts & Exports
    arts_res = get_all_artifacts(session_id)
    assert len(arts_res.python_script) > 0

    exp_res = export_artifact(session_id=session_id, format_type="python")
    assert exp_res.status_code == 200
    assert len(exp_res.body) > 0


def test_chat_endpoint_includes_sources():
    ingest_res = ingest_demo_dataset(DemoIngestRequest(demo_name="Telecom Customer Churn"))
    session_id = ingest_res.session_id

    # Phrased to lexically overlap with the always-present dataset-overview
    # chunk, so retrieval finds a hit even without sentence-transformers
    # installed (this environment runs the zero-dependency lexical fallback).
    chat_res = chat_with_data(ChatRequest(session_id=session_id, query="How many rows and columns are in this dataset?"))
    assert chat_res.status == "success"
    assert chat_res.content
    assert chat_res.sources


def test_llm_providers_endpoint_lists_all_backends():
    res = list_llm_providers()
    keys = {p.key for p in res.providers}
    assert keys == {"ollama", "groq", "gemini"}
