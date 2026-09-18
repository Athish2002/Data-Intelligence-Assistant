"""
api/schemas.py
──────────────
Complete Pydantic V2 schemas for Data Intelligence Assistant REST API.
Covers Data Profiling, 5-Pillar Readiness Audit, AutoML, SHAP, Business ROI,
What-If Simulators, Causal Uplift, Autoencoders, Bandits, Time-Series, NLP,
Graph Intelligence, and Governance MLOps suite.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ─── System & Health ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "13.6.0"
    cpu_cores: int
    gpu_available: bool
    platform: str = "Windows"
    active_sessions_count: int
    storage_status: str = "healthy"
    storage_type: str = "local"
    storage_details: dict[str, Any] = Field(default_factory=dict)
    auth_enabled: bool = False
    tenants_count: int = 1
    ram_percent: Optional[float] = Field(default=None, description="RAM utilization percentage")
    ram_used_gb: Optional[float] = Field(default=None, description="Used RAM in GB")
    ram_total_gb: Optional[float] = Field(default=None, description="Total system RAM in GB")
    ram_available_gb: Optional[float] = Field(default=None, description="Available RAM in GB")


class SystemReadinessResponse(BaseModel):
    status: str = "ready"
    storage_healthy: bool = True
    storage_type: str = "local"
    memory_available_gb: float = 0.0
    memory_healthy: bool = True
    active_sessions_count: int = 0
    subsystems: dict[str, str] = Field(default_factory=dict)


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
    tenant_id: str = "default"
    org_id: str = "default"


# ─── Auth, RBAC & Storage Schemas ──────────────────────────────────────────────

class AuthTokenRequest(BaseModel):
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    role: str = "DataScientist"
    tenant_id: str = "default"
    org_id: str = "default"


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant_id: str
    org_id: str
    expires_in: int


class UserContextResponse(BaseModel):
    user_id: str
    role: str
    tenant_id: str
    org_id: str
    permissions: list[str] = []


class AuditEventItem(BaseModel):
    event_id: str
    timestamp: float
    event_type: str
    user_id: str
    role: str
    tenant_id: str
    org_id: str
    resource: str
    status: str
    ip_address: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class AuditLogsResponse(BaseModel):
    total_events: int
    events: list[AuditEventItem]


class StorageStatusResponse(BaseModel):
    status: str
    backend: str
    writable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class StorageFileListResponse(BaseModel):
    tenant_id: str
    prefix: str
    files: list[str] = []
    total_count: int = 0


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
    suggested_target: str | None = None
    goal: str | None = None


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
    brier_score: float | None = None
    calibration_curve: dict[str, Any] | None = None


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


# ─── Next-Gen Innovations (L1-L5) Schemas ────────────────────────────────────

class RecourseActionItem(BaseModel):
    feature: str
    original_value: Any
    proposed_value: Any
    delta: float
    delta_display: str
    direction: str
    normalized_cost: float
    relative_difficulty: str


class RecourseRequest(BaseModel):
    session_id: str
    row_index: int = 0
    desired_class: int = 1
    target_probability_threshold: float = 0.55
    immutable_features: list[str] = []
    custom_feature_overrides: dict[str, float] | None = None


class RecourseResponse(BaseModel):
    session_id: str
    original_prediction: int
    target_prediction: int
    original_probability: float
    target_probability: float
    total_recourse_cost: float
    feasibility: str
    actions: list[RecourseActionItem] = []
    counterfactual_vector: dict[str, Any] = {}
    executive_guidance: list[str] = []
    immutable_features_locked: list[str] = []
    status: str = "success"


class StressScenarioItem(BaseModel):
    scenario_name: str
    baseline_adverse_rate: float
    stressed_adverse_rate: float
    rate_delta: float
    var_95: float
    var_99: float
    cvar_95: float
    survival_probability: float
    resilience_rating: str
    top_risk_drivers: list[dict[str, Any]] = []
    tail_distribution: list[float] = []


class StressTestRequest(BaseModel):
    session_id: str
    custom_shocks: dict[str, float] | None = None
    n_simulations: int = 250


class StressTestResponse(BaseModel):
    session_id: str
    n_simulations: int
    n_evaluated_rows: int
    overall_resilience_grade: str
    baseline_loss_or_default_rate: float
    scenarios: list[StressScenarioItem] = []
    executive_recommendations: list[str] = []
    status: str = "success"


class ConformalInstanceItem(BaseModel):
    predicted_label: float | int
    conformal_set: list[int] = []
    conformal_interval: list[float] | None = None
    aleatoric_entropy: float
    epistemic_distance: float
    uncertainty_classification: str
    requires_human_review: bool
    coverage_guarantee_pct: float


class ConformalBoundsRequest(BaseModel):
    session_id: str
    alpha: float = Field(0.10, ge=0.01, le=0.50)


class ConformalBoundsResponse(BaseModel):
    session_id: str
    alpha_error_rate: float
    coverage_guarantee_pct: float
    task_type: str
    quantile_threshold: float
    empirical_coverage: float
    average_set_size_or_width: float
    ood_flagged_count: int
    ood_flagged_pct: float
    sample_evaluations: list[ConformalInstanceItem] = []
    executive_verdict: str
    status: str = "success"


class DiscoveredFormulaItem(BaseModel):
    feature_name: str
    formula_latex: str
    sql_expression: str
    python_expression: str
    base_features: list[str]
    correlation_with_target: float
    correlation_lift: float
    mutual_info_score: float
    description: str


class SymbolicDiscoveryRequest(BaseModel):
    session_id: str
    max_candidates: int = 120
    top_k: int = 5


class SymbolicDiscoveryResponse(BaseModel):
    session_id: str
    target_column: str
    n_evaluated_expressions: int
    n_discovered_formulas: int
    formulas: list[DiscoveredFormulaItem] = []
    consolidated_sql_view: str
    consolidated_python_transform: str
    status: str = "success"


class FeatureDriftItem(BaseModel):
    feature_name: str
    feature_type: str
    ks_statistic: float | None = None
    p_value: float | None = None
    psi_score: float
    drift_status: str
    baseline_mean: float | None = None
    current_mean: float | None = None
    mean_shift_pct: float | None = None


class DriftSentinelRequest(BaseModel):
    session_id: str
    sample_fraction: float = 0.4
    synthetic_shift_strength: float = 0.0


class DriftSentinelResponse(BaseModel):
    session_id: str
    total_features_monitored: int
    n_reference_samples: int
    n_current_samples: int
    drifting_feature_count: int
    drifting_feature_ratio: float
    multivariate_mmd_score: float
    overall_sentinel_status: str
    governance_action: str
    feature_drift_breakdown: list[FeatureDriftItem] = []
    top_drifting_features: list[str] = []
    actionable_recommendations: list[str] = []
    status: str = "success"
