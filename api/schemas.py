"""
api/schemas.py
──────────────
Complete Pydantic V2 schemas for Data Intelligence Assistant REST API.
Covers Data Profiling, 5-Pillar Readiness Audit, AutoML, SHAP, Business ROI,
What-If Simulators, Causal Uplift, Autoencoders, Bandits, Time-Series, NLP,
Graph Intelligence, and Governance MLOps suite.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ─── System & Health ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "13.6.0"
    cpu_cores: int
    gpu_available: bool
    platform: str = "Windows"
    active_sessions_count: int


class SessionDetailItem(BaseModel):
    session_id: str
    goal: str = ""
    n_rows: int = 0
    n_cols: int = 0
    memory_mb: float = 0.0
    has_pipeline: bool = False
    best_model: str | None = None
    age_seconds: float = 0.0
    idle_seconds: float = 0.0
    access_count: int = 1


class SystemMetricsResponse(BaseModel):
    status: str = "ok"
    process_memory_rss_mb: float
    process_memory_vms_mb: float
    system_memory_total_gb: float
    system_memory_used_gb: float
    system_memory_available_gb: float
    system_memory_percent: float
    cpu_percent: float
    cpu_cores_logical: int
    active_sessions_count: int
    max_sessions_capacity: int
    session_ttl_minutes: int
    python_version: str
    platform_name: str
    gpu_available: bool
    thread_count: int
    uptime_seconds: float
    sessions_detail: list[SessionDetailItem] = []


class SystemGcResponse(BaseModel):
    status: str = "ok"
    reclaimed_mb: float
    unreachable_objects_collected: int
    current_rss_mb: float
    active_sessions_remaining: int


# ─── Demo & Ingestion ────────────────────────────────────────────────────────

class DemoDatasetItem(BaseModel):
    id: str
    name: str
    domain: str
    default_goal: str
    default_target: str
    description: str


class IngestResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    columns: list[str]
    sample_data: list[dict[str, Any]]
    sanitize_report: dict[str, Any]
    detected_domain: str
    suggested_objectives: list[str]
    capabilities: dict[str, bool]


class DemoIngestRequest(BaseModel):
    demo_name: str = Field(..., description="Name of the demo benchmark dataset")


# ─── Auto-Detect Objectives ──────────────────────────────────────────────────

class AutoDetectRequest(BaseModel):
    session_id: str


class AutoDetectResponse(BaseModel):
    domain: str
    objectives: list[str]
    confidence_matches: int
    engine: str


# ─── Workspace 1: Data Profiling & Readiness Audit ───────────────────────────

class ColumnRoleItem(BaseModel):
    column: str
    dtype: str
    role: str  # numeric, categorical_low, categorical_high, datetime, text, id_entity
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]


class ProfileResponse(BaseModel):
    session_id: str
    shape: list[int]
    memory_mb: float
    total_cells_repaired: int
    columns_info: list[ColumnRoleItem]
    correlations: list[dict[str, Any]]
    summary_stats: dict[str, dict[str, Any]]


class AuditCheckItem(BaseModel):
    category: str
    title: str
    verdict: str  # PASS, WARN, FAIL
    score: int
    details: str
    action_item: str | None = None


class ReadinessResponse(BaseModel):
    session_id: str
    overall_readiness_score: int
    production_verdict: str
    checks: list[AuditCheckItem]
    key_recommendations: list[str]


# ─── Workspace 2: AutoML Training, Curves & Explainability ────────────────────

class TrainPipelineRequest(BaseModel):
    session_id: str
    goal: str = Field(..., min_length=3, max_length=1000)
    user_target_col: str | None = None
    selected_models: list[str] | None = None


class ModelMetricItem(BaseModel):
    label: str
    key: str
    metrics: dict[str, Any]
    is_best: bool = False


class TrainPipelineResponse(BaseModel):
    session_id: str
    status: str
    target_col: str
    final_task_type: str
    best_model_label: str
    best_model_key: str
    models_evaluated: list[dict[str, Any]]
    evaluation_metrics: dict[str, Any]
    shap_importance: list[dict[str, Any]]
    readiness_score_pct: int
    readiness_verdict: str
    privacy_risk_score: int
    latency_stats: dict[str, Any]
    smart_insights: list[str]
    capabilities: dict[str, bool]
    roc_curve: dict[str, Any] | None = None
    confusion_matrix: list[list[int]] | None = None


class RoiOptimizeRequest(BaseModel):
    session_id: str
    benefit_tp: float = 100.0  # Value of true positive
    cost_fp: float = 20.0      # Cost of false positive
    cost_fn: float = 150.0     # Cost of false negative
    benefit_tn: float = 0.0    # Benefit of true negative


class RoiOptimizeResponse(BaseModel):
    status: str
    optimal_threshold: float
    max_expected_profit: float
    default_profit_at_50: float
    net_profit_gain: float
    threshold_curve: list[dict[str, float]]


class ActiveLearningResponse(BaseModel):
    status: str
    n_uncertain: int
    uncertain_samples: list[dict[str, Any]]
    selection_strategy: str


# ─── Inference & Simulation ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id: str
    features: dict[str, Any]


class PredictResponse(BaseModel):
    status: str
    prediction: Any
    probabilities: dict[str, float] | None = None
    task_type: str
    latency_ms: float


class SimulateRequest(BaseModel):
    session_id: str
    feature_overrides: dict[str, Any]


class SimulateResponse(BaseModel):
    status: str
    prediction: Any
    probability: float | None = None
    task_type: str


# ─── Workspace 3: Adaptive AI Engines ────────────────────────────────────────

class CounterfactualRequest(BaseModel):
    session_id: str
    row_index: int = 0
    desired_outcome: Any = 1


class CounterfactualResponse(BaseModel):
    status: str
    original_prediction: Any
    counterfactual_prediction: Any
    perturbations: list[dict[str, Any]]
    message: str | None = None


class UpliftRequest(BaseModel):
    session_id: str
    treatment_column: str = "(Auto-Synthesize Action)"


class UpliftResponse(BaseModel):
    status: str
    average_treatment_effect_ate: float
    recommended_action: str
    uplift_quadrant_distribution: dict[str, Any]


class AutoencoderResponse(BaseModel):
    status: str
    reconstruction_mae: float
    anomaly_threshold: float
    anomalous_samples_count: int
    top_anomalous_samples: list[dict[str, Any]]
    loss_history: list[float]


class BanditSimRequest(BaseModel):
    session_id: str
    n_steps: int = 100
    alpha: float = 1.0


class BanditSimResponse(BaseModel):
    status: str
    cumulative_reward: float
    total_steps: int
    arm_selection_counts: dict[str, int]
    reward_trajectory: list[float]


class OnlineLearningResponse(BaseModel):
    status: str
    n_batches: int
    final_loss: float
    batch_loss_history: list[float]


class TimeSeriesForecastRequest(BaseModel):
    session_id: str
    date_col: str | None = None
    forecast_horizon: int = 14


class TimeSeriesForecastResponse(BaseModel):
    status: str
    evaluation: dict[str, float]
    historical_points: list[dict[str, Any]]
    future_projections: list[dict[str, Any]]
    message: str | None = None


class SyntheticGenerateRequest(BaseModel):
    session_id: str
    n_samples: int = 100
    apply_dp_noise: bool = False
    epsilon: float = 1.0


class SyntheticGenerateResponse(BaseModel):
    status: str
    n_generated: int
    fidelity_score_pct: float
    sample_records: list[dict[str, Any]]
    download_filename: str


class NlpAnalysisResponse(BaseModel):
    status: str
    analyzed_text_column: str
    sentiment_distribution: dict[str, float]
    top_keywords: list[dict[str, Any]]


class GraphAnalysisResponse(BaseModel):
    status: str
    n_nodes: int
    n_edges: int
    top_central_entities: list[dict[str, Any]]
    bipartite_edges: list[dict[str, Any]]


# ─── Workspace 4: Governance & MLOps ─────────────────────────────────────────

class DataContractResponse(BaseModel):
    status: str
    contract_yaml: str
    great_expectations_json: str
    n_expectations: int
    expectations: list[dict[str, Any]] = []


class GdprAuditResponse(BaseModel):
    status: str
    privacy_risk_score: int
    pii_entities_detected: list[dict[str, Any]]
    ropa_markdown: str
    frameworks: dict[str, Any] = {}
    ropa_details: dict[str, Any] = {}


class DriftMonitorResponse(BaseModel):
    status: str
    psi_score: float
    drift_status: str  # NO DRIFT, MODERATE DRIFT, SEVERE DRIFT
    feature_drift_breakdown: list[dict[str, Any]]


class ArtifactsResponse(BaseModel):
    session_id: str
    python_script: str
    fastapi_code: str
    airflow_dag: str
    dockerfile: str
    docker_compose: str
    k8s_manifest: str
    sql_query: str
    model_card_md: str


# ─── Copilot ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    status: str
    role: str = "assistant"
    content: str
    table_data: list[dict[str, Any]] | None = None
    chart_json: str | None = None
    sources: list[dict[str, Any]] | None = None


# ─── Data Dictionary ─────────────────────────────────────────────────────────

class DictionaryUploadResponse(BaseModel):
    status: str  # "success" | "rejected"
    n_terms_parsed: int
    n_terms_saved: int
    safe_to_persist: bool
    rejected_reason: str | None = None


class DictionaryEntry(BaseModel):
    term: str
    definition: str
    source_label: str | None = None


class DictionaryListResponse(BaseModel):
    status: str
    entries: list[DictionaryEntry]
    count: int


# ─── LLM Providers ───────────────────────────────────────────────────────────

class LLMProviderStatus(BaseModel):
    key: str
    label: str
    available: bool
    requires: str | None = None


class LLMProvidersResponse(BaseModel):
    providers: list[LLMProviderStatus]
    default_provider: str | None = None
