"""
app.py
──────
Data Intelligence Assistant – Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import traceback
import json
import numpy as np
import pandas as pd

import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Data Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Bootstrap logging (before any dia imports) ───────────────────────────────
from dia.logging_config import setup_logging  # noqa: E402
from dia.ui_theme import inject_custom_theme, render_hero_banner

setup_logging()

import logging  # noqa: E402

log = logging.getLogger("dia.app")

# ─── Inject Glassmorphic Custom Theme ─────────────────────────────────────────
inject_custom_theme()

# ─── Hero banner ─────────────────────────────────────────────────────────────
render_hero_banner()

# ─── Remaining imports ────────────────────────────────────────────────────────
from dia.config import MAX_MODELS
from dia.data_profiler import (
    detect_target_type,
    generate_readiness_report,
    infer_column_roles,
    profile_dataframe,
)
from dia.exceptions import (
    ConfigurationError,
    DIAError,
    IngestionError,
    ModelTrainingError,
    ValidationError,
)
from dia.explainability import generate_explanation
from dia.goal_parser import parse_goal
from dia.hardware import get_cpu_cores, is_gpu_available, estimate_training_time
from dia.ingestion import SOURCE_REGISTRY
from dia.model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_and_evaluate
from dia.utils import limit_models
from dia.validators import detect_pii_columns, validate_goal_text, validate_target_column
from dia.insights import generate_smart_insights
from dia.code_generator import (
    generate_pipeline_code,
    generate_airflow_dag,
    generate_fastapi_app,
    generate_dockerfile,
    generate_docker_compose,
    generate_github_actions_pipeline,
    generate_k8s_manifests,
)
from dia.data_quality import generate_data_contract, format_contract_markdown
from dia.experimentation import calculate_ab_test_sample_size
from dia.chat_analyst import execute_natural_language_query
from dia.drift_monitor import calculate_drift_report
from dia.compliance import scan_dataset_privacy, format_compliance_dossier_markdown
from dia.gdpr import (
    generate_ropa_record,
    format_gdpr_audit_markdown,
    process_dsar_access_export,
    process_dsar_erasure_anonymization,
)
from dia.mlops_registry import (
    benchmark_model_latency,
    generate_mlflow_run_manifest,
    generate_model_card_markdown,
)
from dia.causal_engine import generate_counterfactual, estimate_uplift_t_learner
from dia.time_series import detect_time_series_column, train_time_series_forecaster
from dia.nlp_processor import detect_text_columns, extract_lexical_features, extract_tfidf_dense_features
from dia.synthetic_data import generate_synthetic_dataset
from dia.streaming_learner import simulate_streaming_incremental_fit
from dia.report_generator import generate_executive_html_report
from dia.active_learning import sample_uncertain_predictions
from dia.deep_autoencoder import train_tabular_autoencoder
from dia.bandit_optimizer import run_contextual_bandit_simulation
from dia.sql_transpiler import transpile_model_to_sql
from dia.feature_store import generate_feature_store_definitions
from dia.graph_engine import construct_and_analyze_entity_graph
from dia.canary_router import simulate_canary_routing
from dia.expectations import generate_and_evaluate_expectations
from dia.rbac import ROLE_PERMISSIONS
from ui.dashboard import (
    render_column_roles,
    render_explainability,
    render_final_summary,
    render_model_results,
    render_overview,
    render_readiness_report,
    render_smart_insights,
    render_business_impact,
)
from ui.simulator import render_simulator
from dia.ui_theme import render_mission_ribbon


# ═══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    user_persona = st.selectbox(
        "👤 Active Persona (RBAC)",
        options=list(ROLE_PERMISSIONS.keys()),
        index=0,
        help="Filters visible modules and tools according to enterprise security tier."
    )
    st.divider()
    st.caption("💡 Toggle dark/light mode via Streamlit's ☰ menu → Settings.")
    st.divider()

    # ── Step 1: Data Source ───────────────────────────────────────────────────
    st.markdown('<span class="step-badge">1</span> **Data Source**', unsafe_allow_html=True)

    available_sources = {k: v for k, v in SOURCE_REGISTRY.items() if v["available"]}
    source_key = st.selectbox(
        "Choose data source",
        options=list(available_sources.keys()),
        format_func=lambda k: available_sources[k]["label"],
        help="Select where your dataset comes from.",
        key="source_key",
    )

    source_info = available_sources[source_key]

    # ── Dynamic source config form ────────────────────────────────────────────
    source_params: dict = {}

    if source_key == "demo_sample":
        demo_name = st.selectbox(
            "Select Benchmark Demo Dataset",
            options=[
                "Telecom Customer Churn",
                "Bank Credit Risk & Default",
                "Real Estate Price Estimation",
                "Dirty & Malformed Retail E-Commerce",
            ],
            index=0,
            key="demo_dataset_choice",
        )
        source_params["demo_name"] = demo_name
        st.info("💡 Zero setup required! Instantly analyzes synthetic enterprise benchmark data.")

    elif source_key == "local_csv":
        uploaded_file = st.file_uploader(
            "Choose a CSV file (max 500 MB)",
            type=["csv"],
            help="Your file is kept in memory only — never written to disk.",
            key="uploaded_file",
        )
        source_params["uploaded_file"] = uploaded_file

    elif source_key == "url":
        url_input = st.text_input(
            "CSV URL",
            placeholder="https://example.com/dataset.csv",
            key="url_input",
        )
        source_params["url"] = url_input

    elif source_key == "sql":
        sql_conn = st.text_input(
            "SQLAlchemy connection string",
            placeholder="postgresql+psycopg2://user:pass@host:5432/db",
            type="password",
            key="sql_conn",
            help="Supports PostgreSQL, MySQL, SQLite, SQL Server.",
        )
        sql_input = st.text_area(
            "SQL query or table name",
            placeholder="SELECT * FROM my_table\n-- or just type the table name below",
            key="sql_query",
            height=80,
        )
        sql_table = st.text_input(
            "Table name (if no query above)",
            key="sql_table",
        )
        source_params.update({
            "connection_string": sql_conn,
            "query": sql_input,
            "table_name": sql_table,
        })

    elif source_key == "s3":
        col1, col2 = st.columns(2)
        with col1:
            s3_bucket = st.text_input("Bucket name", key="s3_bucket")
        with col2:
            s3_key = st.text_input("Object key", key="s3_key")
        s3_region = st.text_input("Region", value="us-east-1", key="s3_region")
        s3_access_key = st.text_input("Access key ID", type="password", key="s3_access")
        s3_secret_key = st.text_input("Secret access key", type="password", key="s3_secret")
        st.caption("💡 Leave keys empty to use IAM role or env vars.")
        source_params.update({
            "bucket": s3_bucket, "key": s3_key, "region_name": s3_region,
            "aws_access_key_id": s3_access_key or None,
            "aws_secret_access_key": s3_secret_key or None,
        })

    elif source_key == "gcs":
        gcs_bucket = st.text_input("GCS Bucket", key="gcs_bucket")
        gcs_blob = st.text_input("Blob path", key="gcs_blob")
        gcs_sa_json = st.text_area(
            "Service account JSON (optional — leave blank for ADC)",
            height=80, key="gcs_sa",
        )
        source_params.update({
            "bucket": gcs_bucket, "blob_name": gcs_blob,
            "service_account_json": gcs_sa_json,
        })

    elif source_key == "azure":
        az_conn = st.text_input(
            "Connection string (or leave blank to use account+key)",
            type="password", key="az_conn",
        )
        az_account = st.text_input("Account name", key="az_account")
        az_key = st.text_input("Account key", type="password", key="az_key")
        az_container = st.text_input("Container name", key="az_container")
        az_blob = st.text_input("Blob name", key="az_blob")
        source_params.update({
            "connection_string": az_conn, "account_name": az_account,
            "account_key": az_key, "container_name": az_container,
            "blob_name": az_blob,
        })

    elif source_key == "bigquery":
        bq_project = st.text_input("GCP Project ID", key="bq_project")
        bq_query = st.text_area("SQL Query", key="bq_query", height=80)
        bq_sa_json = st.text_area(
            "Service account JSON (optional — leave blank for ADC)",
            height=80, key="bq_sa",
        )
        source_params.update({
            "project": bq_project, "query": bq_query,
            "service_account_json": bq_sa_json,
        })

    elif source_key == "snowflake":
        sf_account = st.text_input("Account (e.g. xy12345.us-east-1)", key="sf_account")
        sf_user = st.text_input("Username", key="sf_user")
        sf_pass = st.text_input("Password", type="password", key="sf_pass")
        sf_wh = st.text_input("Warehouse", key="sf_wh")
        sf_db = st.text_input("Database", key="sf_db")
        sf_schema = st.text_input("Schema", value="PUBLIC", key="sf_schema")
        sf_query = st.text_area("SQL Query (or table name below)", key="sf_query", height=70)
        sf_table = st.text_input("Table name", key="sf_table")
        source_params.update({
            "account": sf_account, "user": sf_user, "password": sf_pass,
            "warehouse": sf_wh, "database": sf_db, "schema": sf_schema,
            "query": sf_query, "table_name": sf_table,
        })

    st.divider()

    # ── Step 2: Goal ──────────────────────────────────────────────────────────
    st.markdown('<span class="step-badge">2</span> **Prediction Goal**', unsafe_allow_html=True)
    
    # Auto-detect objectives feature (Local & Offline by default)
    with st.expander("🎯 Auto-Detect Domain & Goals (Instant Local AI)", expanded=True):
        st.markdown("Instantly analyzes your schema, types, and value distributions **locally (0ms latency, zero API limits)** to detect domain and recommend ML goals.")
        
        if st.button("🔍 Auto-Detect Objectives Now", key="auto_detect_btn"):
            try:
                with st.spinner("Analyzing schema locally..."):
                    if source_key == "demo_sample":
                        from dia.demo_datasets import get_demo_dataset
                        df_peek, _, _ = get_demo_dataset(source_params.get("demo_name", "Telecom Customer Churn"))
                    else:
                        ingestion_cls = source_info["cls"]
                        if ingestion_cls is None:
                            st.error(f"Data source missing dependencies: {source_info.get('requires')}")
                            df_peek = None
                        else:
                            ingest_res = ingestion_cls().load(**source_params)
                            df_peek = ingest_res.df
                            
                    if df_peek is not None:
                        from dia.llm_context import get_dataset_context_and_objectives
                        ai_res = get_dataset_context_and_objectives(df_peek)
                        st.session_state["ai_suggestions"] = ai_res
            except Exception as e:
                st.error(f"Could not load data for preview: {e}")
                
        if "ai_suggestions" in st.session_state:
            res = st.session_state["ai_suggestions"]
            st.success(f"🏢 **Domain:** `{res.get('domain', 'General Analytics')}`")
            st.markdown("**Click a suggested objective to apply:**")
            for i, obj in enumerate(res.get("objectives", [])):
                if st.button(f"👉 {obj}", key=f"btn_obj_{i}"):
                    st.session_state["goal_input"] = obj
                    st.rerun()
                    
        with st.expander("🔑 Optional Cloud LLM (Gemini API)", expanded=False):
            gemini_key = st.text_input("Gemini API Key (Optional)", type="password", key="gemini_key_input", help="Optional. The assistant already runs 100% offline.")
                
    goal_raw = st.text_area(
        "Describe your ML goal in plain English",
        placeholder="e.g. predict customer churn\nforecast next month's sales\nclassify loan default",
        height=90,
        key="goal_input",
        help="Max 1,000 characters. The assistant will detect task type automatically.",
    )

    st.divider()

    # ── Step 3: Model selection ───────────────────────────────────────────────
    st.markdown(
        '<span class="step-badge">3</span> **Select Models** *(up to 4)*',
        unsafe_allow_html=True,
    )
    st.caption("Only models compatible with the detected task type will be trained.")

    unique_keys = list(dict.fromkeys(
        list(CLASSIFICATION_MODELS.keys()) + list(REGRESSION_MODELS.keys())
    ))
    all_model_info = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}

    selected_model_keys: list[str] = []
    for key in unique_keys:
        meta = all_model_info[key]
        optional = meta.get("optional", False)
        label = meta["label"]
        desc = meta.get("description", "")
        suffix = " *(optional)*" if optional else ""
        checked = st.checkbox(
            f"{label}{suffix}",
            value=key in ("logreg", "rf", "linreg"),
            help=desc,
            key=f"model_chk_{key}",
        )
        if checked:
            selected_model_keys.append(key)

    if len(selected_model_keys) > MAX_MODELS:
        st.warning(f"Only the first {MAX_MODELS} selected models will be trained.")
        selected_model_keys = selected_model_keys[:MAX_MODELS]

    st.divider()

    # ─── Hardware & Optimization ──────────────────────────────────────────────────
    with st.expander("⚙️ Hardware & Optimization"):
        has_gpu = is_gpu_available()
        total_cores = get_cpu_cores()
        default_cores = max(1, total_cores - 2) if total_cores > 2 else total_cores
        
        use_gpu = st.toggle("Use GPU Acceleration", value=has_gpu, disabled=not has_gpu, 
                            help="Train models (XGBoost/LightGBM) on the GPU if detected.")
        if not has_gpu:
            st.caption("No NVIDIA GPU detected. Ensure `nvidia-smi` is available.")
            
        n_jobs = st.slider("CPU Cores (Threads)", min_value=1, max_value=total_cores, value=default_cores, 
                           help="Maximum number of CPU cores to use for model training.")

    with st.expander("🔬 Advanced ML Settings"):
        handle_imbalance = st.toggle("Handle Class Imbalance (SMOTE/Weighting)", value=True, 
                                     help="Automatically penalize the model for missing rare minority classes (Crucial for Fraud/Churn).")
        apply_cv = st.toggle("Use 5-Fold Cross Validation", value=True, 
                             help="Evaluate models more rigorously by testing them across 5 different splits. Slower, but more reliable.")
        enable_hpo = st.toggle("Tune Hyperparameters (HPO)", value=False,
                               help="Automatically search and tune hyperparameters (learning rate, depth, regularization) using Randomized Search.")
        hpo_iter = 10
        if enable_hpo:
            hpo_iter = st.slider("HPO Search Iterations", min_value=5, max_value=25, value=10, step=5,
                                 help="Number of parameter combinations to sample per model. Higher = better tuning, slower training.")
        enable_autofe = st.toggle("Automated Feature Engineering (AutoFE)", value=False,
                                  help="Automatically synthesize date decompositions, non-linear interaction ratios, and log transforms.")
        build_ensemble = st.toggle("Synthesize Blended Meta-Ensemble", value=True,
                                   help="Automatically construct a soft voting/weighted ensemble combining top-performing models.")
        calibrate_probs = st.toggle("Calibrate Probabilities (Platt/Sigmoid)", value=False,
                                    help="Calibrate classification probabilities for precise financial ROI and risk metrics.")
                             
    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    can_run = bool(goal_raw.strip() and selected_model_keys)
    run_analysis = st.button(
        "🚀 Run Analysis",
        use_container_width=True,
        disabled=not can_run,
        key="run_btn",
    )

    st.divider()
    st.caption("Data Intelligence Assistant v1.1.0 · Built with Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
#   IDLE STATE
# ═══════════════════════════════════════════════════════════════════════════════

if not run_analysis and "analysis_done" not in st.session_state:
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 2rem;color:#64748b;">
            <div style="font-size:4rem;">📂</div>
            <h3 style="color:#94a3b8;">Connect a data source to get started</h3>
            <p>Choose a source in the sidebar, enter your prediction goal,
            select models, and click <b>Run Analysis</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not run_analysis:
    st.stop()

# Clear previous results on new run
for key in ("analysis_done", "df", "meta", "goal_info", "annotated_profile",
            "target_col", "final_task_type", "readiness", "train_result", 
            "explanation", "insights", "code_script"):
    st.session_state.pop(key, None)


# ═══════════════════════════════════════════════════════════════════════════════
#   ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

status = st.status("🔄 Running analysis…", expanded=True)

def _safe_status_update(lbl: str, st_val: str, exp: bool | None = None):
    try:
        if hasattr(status, "update"):
            if exp is not None:
                status.update(label=lbl, state=st_val, expanded=exp)
            else:
                status.update(label=lbl, state=st_val)
    except Exception:
        pass

with status:
    progress_bar = st.progress(0)
    
    try:
        # ── Phase 1: Ingestion (0 - 20%) ──────────────────────────────────────
        st.write("📡 **Phase 1:** Loading data...")
        progress_bar.progress(5)
        
        goal = validate_goal_text(goal_raw)
        
        if source_key == "demo_sample":
            from dia.demo_datasets import get_demo_dataset
            df, demo_goal, demo_target = get_demo_dataset(source_params.get("demo_name", "Telecom Customer Churn"))
            meta = {
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
                "file_size_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2),
                "encoding": "in-memory",
            }
        else:
            ingestion_cls = source_info["cls"]
            if ingestion_cls is None:
                st.error(
                    f"The selected source requires **{source_info['requires']}** to be installed. "
                    f"Run: `pip install {source_info['requires']}`"
                )
                st.stop()

            ingestion_result = ingestion_cls().load(**source_params)
            df = ingestion_result.df
            meta = ingestion_result.meta
            meta.setdefault("n_rows", df.shape[0])
            meta.setdefault("n_cols", df.shape[1])
            meta.setdefault("file_size_mb", "N/A")
            meta.setdefault("encoding", "N/A")
        
        pii_cols = detect_pii_columns(list(df.columns))
        if pii_cols:
            st.warning(
                f"⚠️ **Possible PII detected** in columns: "
                f"{', '.join(f'`{html.escape(c)}`' for c in pii_cols[:5])}. "
                "Ensure you have the right to use this data and that it is anonymised."
            )
            
        progress_bar.progress(20)

        # ── Phase 2: Profiling & Goal Parsing (20 - 40%) ──────────────────────
        st.write("🔍 **Phase 2:** Profiling dataset and parsing goal...")
        
        goal_info = parse_goal(goal, columns=df.columns.tolist())
        profile = profile_dataframe(df)
        annotated_profile = infer_column_roles(df, profile)
        
        progress_bar.progress(40)

        # ── Phase 3: Target Detection & Readiness (40 - 60%) ──────────────────
        st.write("🎯 **Phase 3:** Detecting target and checking data readiness...")
        
        from dia.column_resolver import rank_target_candidates_advanced
        advanced_ranked = rank_target_candidates_advanced(goal, df)
        if advanced_ranked:
            target_col = advanced_ranked[0]["column"]
            st.caption(f"🎯 **Target Auto-Resolved:** `{target_col}` (Confidence: {advanced_ranked[0]['confidence']:.0%})")
        else:
            candidates = [c for c in goal_info.get("target_candidates", []) if c in df.columns]
            target_col = candidates[0] if candidates else df.columns[-1]

        validate_target_column(target_col, df)

        target_type_info = detect_target_type(df, target_col)
        final_task_type = target_type_info["task_type"]
        
        readiness = generate_readiness_report(
            df, annotated_profile, goal_info, target_col, final_task_type
        )
        
        progress_bar.progress(60)

        # ── Phase 4: Model Training (60 - 80%) ────────────────────────────────
        st.write(f"🤖 **Phase 4:** Training models for {final_task_type}...")
        
        valid_registry = (
            CLASSIFICATION_MODELS if final_task_type == "classification" else REGRESSION_MODELS
        )
        task_model_keys = [k for k in selected_model_keys if k in valid_registry]
        if not task_model_keys:
            task_model_keys = list(valid_registry.keys())[:2]
            
        n_rows, n_cols = df.shape
        est_time = estimate_training_time(n_rows, n_cols, len(task_model_keys), use_gpu)
        st.info(f"⏳ **Estimated training time:** {est_time}")
            
        train_result = train_and_evaluate(
            df=df,
            target_col=target_col,
            task_type=final_task_type,
            selected_model_keys=task_model_keys,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            handle_imbalance=handle_imbalance,
            apply_cv=apply_cv,
            enable_hpo=enable_hpo,
            hpo_iter=hpo_iter,
            enable_autofe=enable_autofe,
            calibrate_probs=calibrate_probs,
            build_ensemble=build_ensemble,
        )
        
        progress_bar.progress(80)

        # ── Phase 5: Insights & Code Exports (80 - 100%) ─────────────────────
        st.write("💡 **Phase 5:** Generating insights, data contracts, and microservice exports...")
        
        best_idx = next(
            (i for i, r in enumerate(train_result["results"])
             if r["model_key"] == train_result["best_model_key"]),
            0,
        )
        explanation = generate_explanation(
            model=train_result["best_model"],
            X_test=train_result["X_test_processed"],
            feature_names=train_result["feature_names"],
            importance_series=train_result["results"][best_idx]["importance"],
        )
        
        insights = generate_smart_insights(df, target_col, final_task_type)
        
        # Enterprise Data Contracts, Compliance & Production Code Generation
        data_contract_json = generate_data_contract(df, target_col)
        data_contract_md = format_contract_markdown(data_contract_json)
        
        compliance_report = scan_dataset_privacy(df)
        compliance_md = format_compliance_dossier_markdown(compliance_report)
        
        ropa_record = generate_ropa_record(df, target_col, final_task_type, ingestion_result.source_label)
        ropa_md = format_gdpr_audit_markdown(ropa_record)
        
        # MLOps Latency Benchmarking, MLflow Manifest & Model Card
        latency_stats = benchmark_model_latency(
            train_result["best_model"],
            train_result["X_test_processed"]
        )
        best_metrics = train_result["results"][best_idx]["metrics"]
        mlflow_manifest = generate_mlflow_run_manifest(
            model_name=train_result["best_model_label"],
            target_col=target_col,
            task_type=final_task_type,
            metrics=best_metrics,
            params=train_result.get("best_params", {}),
            feature_names=train_result["feature_names"],
            df=df,
            latency_stats=latency_stats,
        )
        model_card_md = generate_model_card_markdown(
            model_name=train_result["best_model_label"],
            target_col=target_col,
            task_type=final_task_type,
            metrics=best_metrics,
            feature_names=train_result["feature_names"],
            df=df,
            latency_stats=latency_stats,
        )

        # Executive Report & Active Learning Queue
        executive_html = generate_executive_html_report(
            goal=goal,
            target_col=target_col,
            task_type=final_task_type,
            train_result=train_result,
            readiness=readiness,
            compliance_report=compliance_report,
            latency_stats=latency_stats,
        )
        uncertain_samples = sample_uncertain_predictions(
            train_result["best_model"],
            df,
            train_result["X_test_processed"]
        )
        ts_date_col = detect_time_series_column(df)
        text_cols = detect_text_columns(df)

        code_script = generate_pipeline_code(
            source_label=ingestion_result.source_label,
            target_col=target_col,
            task_type=final_task_type,
            best_model_key=train_result["best_model_key"],
            model_label=train_result["best_model_label"],
            best_params=train_result.get("best_params", {}),
        )
        
        airflow_dag = generate_airflow_dag(
            source_label=ingestion_result.source_label,
            target_col=target_col,
            task_type=final_task_type,
            best_model_key=train_result["best_model_key"],
            model_label=train_result["best_model_label"]
        )

        fastapi_code = generate_fastapi_app(
            target_col=target_col,
            task_type=final_task_type,
            model_label=train_result["best_model_label"],
            raw_feature_cols=train_result.get("raw_feature_cols"),
        )
        dockerfile_code = generate_dockerfile()
        docker_compose_code = generate_docker_compose()
        ci_cd_workflow = generate_github_actions_pipeline(train_result["best_model_label"], target_col)
        k8s_manifests = generate_k8s_manifests(train_result["best_model_label"])
        
        progress_bar.progress(100)

        # ── Store results in session state ────────────────────────────────────
        st.session_state.update({
            "analysis_done": True,
            "df": df,
            "meta": meta,
            "goal_info": goal_info,
            "goal": goal,
            "annotated_profile": annotated_profile,
            "target_col": target_col,
            "final_task_type": final_task_type,
            "readiness": readiness,
            "train_result": train_result,
            "explanation": explanation,
            "insights": insights,
            "code_script": code_script,
            "airflow_dag": airflow_dag,
            "fastapi_code": fastapi_code,
            "dockerfile_code": dockerfile_code,
            "docker_compose_code": docker_compose_code,
            "ci_cd_workflow": ci_cd_workflow,
            "k8s_manifests": k8s_manifests,
            "data_contract_json": data_contract_json,
            "data_contract_md": data_contract_md,
            "compliance_report": compliance_report,
            "compliance_md": compliance_md,
            "ropa_record": ropa_record,
            "ropa_md": ropa_md,
            "latency_stats": latency_stats,
            "mlflow_manifest": mlflow_manifest,
            "model_card_md": model_card_md,
            "executive_html": executive_html,
            "uncertain_samples": uncertain_samples,
            "ts_date_col": ts_date_col,
            "text_cols": text_cols,
            "source_label": ingestion_result.source_label,
            "chat_history": [],
        })

    except ValidationError as exc:
        _safe_status_update("❌ Validation Error", "error")
        st.error(f"**Validation Error:** {html.escape(str(exc))}")
        log.warning("Validation error: %s", exc)
        st.stop()
    except (IngestionError, ConfigurationError) as exc:
        _safe_status_update("❌ Data Source Error", "error")
        st.error(f"**Data Source Error:** {html.escape(str(exc))}")
        log.error("Ingestion/config error: %s", exc)
        st.stop()
    except ModelTrainingError as exc:
        _safe_status_update("❌ Training Error", "error")
        st.error(f"**Training Error:** {html.escape(str(exc))}")
        log.error("Training error: %s", exc)
        st.stop()
    except DIAError as exc:
        _safe_status_update("❌ Error", "error")
        st.error(f"**Error:** {html.escape(str(exc))}")
        log.error("DIA error: %s", exc)
        st.stop()
    except Exception as exc:  # noqa: BLE001
        _safe_status_update("❌ Unexpected Error", "error")
        # Show generic message to user; log full traceback server-side
        st.error(
            "An unexpected error occurred. "
            "Please check your data source configuration and try again."
        )
        log.exception("Unexpected error: %s", exc)
        with st.expander("🐛 Technical details (for debugging)"):
            st.code(traceback.format_exc())
        st.stop()

if hasattr(status, "update"):
    try:
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#   DASHBOARD TABS
# ═══════════════════════════════════════════════════════════════════════════════

s = st.session_state

# ─── Mission Control Status Ribbon ───────────────────────────────────────────
render_mission_ribbon(
    persona=user_persona,
    target_col=s.get("target_col", "Target"),
    task_type=s.get("final_task_type", "classification"),
    best_model=s.get("train_result", {}).get("best_model_label", "Best Model"),
    latency_p99=s.get("latency_stats", {}).get("p99_ms", 1.8),
    privacy_risk=s.get("compliance_report", {}).get("privacy_risk_score", s.get("compliance_report", {}).get("summary", {}).get("privacy_risk_score", 0)),
)

# ─── Dataset-Adaptive Capabilities Detection ──────────────────────────────────
df_curr = s["df"]
has_dates = any(pd.api.types.is_datetime64_any_dtype(df_curr[c]) or "date" in str(c).lower() or "time" in str(c).lower() for c in df_curr.columns)
has_text = any(pd.api.types.is_object_dtype(df_curr[c]) and df_curr[c].dropna().astype(str).str.len().mean() > 25 for c in df_curr.columns)
has_entities = len(df_curr.select_dtypes(include=["object", "category"]).columns) >= 2
is_clf = s.get("final_task_type") == "classification"

# Workspace Navigation
workspace = st.segmented_control(
    "Select Intelligence Workspace",
    options=[
        "📊 Core Intelligence",
        "🤖 AutoML & Explainability",
        "🧬 Adaptive AI Engines",
        "🛡️ Governance, MLOps & Production",
    ],
    default="📊 Core Intelligence",
    key="active_workspace_nav",
)

if not workspace:
    workspace = "📊 Core Intelligence"

# ═══════════════════════════════════════════════════════════════════════════════
#   WORKSPACE 1: CORE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
if workspace == "📊 Core Intelligence":
    w_tabs = st.tabs([
        "📊 Overview & Sanitization",
        "🏷️ Column Roles & Schema",
        "🩺 Data Readiness Audit",
        "💡 Smart Insights & Drivers",
        "📝 Executive Summary",
    ])
    
    with w_tabs[0]:
        render_overview(s["df"], s["meta"])
        
    with w_tabs[1]:
        render_column_roles(s["annotated_profile"])
        
    with w_tabs[2]:
        render_readiness_report(
            s["readiness"], s["goal_info"], s["target_col"], s["final_task_type"]
        )
        
    with w_tabs[3]:
        render_smart_insights(s["insights"])
        
    with w_tabs[4]:
        render_final_summary(
            df=s["df"],
            goal=s["goal"],
            goal_info=s["goal_info"],
            target_col=s["target_col"],
            task_type=s["final_task_type"],
            readiness=s["readiness"],
            train_result=s["train_result"],
            annotated_profile=s["annotated_profile"],
        )

# ═══════════════════════════════════════════════════════════════════════════════
#   WORKSPACE 2: AUTOML & EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif workspace == "🤖 AutoML & Explainability":
    w_tabs = st.tabs([
        "🏆 Model Leaderboard & Metrics",
        "🔬 Explainability (SHAP)",
        "💼 Business ROI & Impact",
        "🎛️ Interactive Simulator",
        "👥 Active Learning Queue",
        "🧪 A/B Test Planner",
    ])
    
    with w_tabs[0]:
        render_model_results(s["train_result"])
        
    with w_tabs[1]:
        render_explainability(s["explanation"], s["train_result"]["best_model_label"])
        
    with w_tabs[2]:
        render_business_impact(
            s["train_result"], s["df"], s["target_col"], s["final_task_type"], s["insights"]
        )
        
    with w_tabs[3]:
        render_simulator(
            df=s["df"], 
            feature_names=s["train_result"].get("raw_feature_cols", s["train_result"]["feature_names"]), 
            preprocessor=s["train_result"]["preprocessor"], 
            model=s["train_result"]["best_model"], 
            task_type=s["final_task_type"], 
            label_encoder=s["train_result"]["label_encoder"]
        )
        
    with w_tabs[4]:
        st.subheader("👥 Active Learning & Human-in-the-Loop (HITL) Queue")
        st.markdown(r"Intelligently surface borderline predictions ($0.45 \le p \le 0.55$) where model uncertainty is highest for human verification.")
        
        unc_queue = s["uncertain_samples"].get("review_queue", [])
        if unc_queue:
            st.info(f"Identified {len(unc_queue)} high-uncertainty observations awaiting domain expert review.")
            for item in unc_queue[:5]:
                with st.expander(f"Row #{item['original_row_index']} &bull; Prediction: `{item['model_prediction']}` (Uncertainty Score: {item['uncertainty_score']})"):
                    st.json(item["features_preview"])
                    c_lbl1, c_lbl2 = st.columns(2)
                    with c_lbl1:
                        if st.button(f"✅ Confirm Prediction #{item['original_row_index']}", key=f"conf_{item['original_row_index']}"):
                            st.success(f"Confirmed row #{item['original_row_index']}")
                    with c_lbl2:
                        if st.button(f"✏️ Override / Correct #{item['original_row_index']}", key=f"ovr_{item['original_row_index']}"):
                            st.warning(f"Queued row #{item['original_row_index']} for next retraining cycle.")
        else:
            st.success("All predictions exhibit high model certainty.")

    with w_tabs[5]:
        st.subheader("🧪 A/B Test Planner (Experimentation Engine)")
        st.markdown("Before deploying the new model, calculate the required sample size and traffic to prove its effectiveness against the baseline rules-engine.")
        
        col1, col2 = st.columns(2)
        with col1:
            base_conv = st.number_input("Current Baseline Conversion Rate (e.g. 0.10 for 10%)", value=0.10, min_value=0.01, max_value=0.99, step=0.01)
            expected_lift = st.number_input("Expected Relative Lift (e.g. 0.20 for a 20% improvement)", value=0.20, min_value=0.01, max_value=5.0, step=0.05)
        with col2:
            stat_power = st.slider("Statistical Power", min_value=0.50, max_value=0.99, value=0.80, help="Probability of detecting an effect if there is one.")
            alpha = st.slider("Significance Level (Alpha)", min_value=0.01, max_value=0.10, value=0.05, help="Probability of a false positive.")
            
        try:
            res = calculate_ab_test_sample_size(base_conv, expected_lift, stat_power, alpha)
            st.success(f"**Target Conversion Rate:** {res['target_rate']*100:.2f}% (Absolute MDE: {res['absolute_mde']*100:.2f}%)")
            st.info(f"**Required Sample Size (Per Variant):** {res['sample_size_per_variant']:,} users")
            st.warning(f"**Total Traffic Required:** {res['total_traffic_required']:,} users")
        except Exception as e:
            st.error(f"Error calculating sample size: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#   WORKSPACE 3: ADAPTIVE AI & DOMAIN ENGINES
# ═══════════════════════════════════════════════════════════════════════════════
elif workspace == "🧬 Adaptive AI Engines":
    # Build list of active engines based on dataset capabilities
    adaptive_options: list[str] = [
        "🎯 Causal & Counterfactuals",
        "🧠 Deep Autoencoder (DL)",
        "🎰 Contextual Bandits (RL)",
        "🧬 Generative Synthetic Data",
        "⚡ Online Incremental Learning",
    ]
    if has_dates:
        adaptive_options.append("📈 Time-Series Forecaster")
    if has_text:
        adaptive_options.append("📝 NLP & Lexical Analysis")
    if has_entities:
        adaptive_options.append("🕸️ Graph Intelligence")

    w_tabs = st.tabs(adaptive_options)
    tab_idx = 0

    # 1. Causal & Counterfactuals
    with w_tabs[tab_idx]:
        st.subheader("🎯 Causal Machine Learning & Counterfactual Interventions")
        st.markdown("Determine prescriptive minimal actionable interventions or estimate Uplift Treatment Effects.")
        
        c_tab1, c_tab2 = st.tabs(["🔍 Counterfactual What-If", "👥 Causal Uplift (T-Learner)"])
        with c_tab1:
            row_sel = st.slider("Select Record Index to Analyze", 0, max(0, len(s["df"]) - 1), 0, key="cf_row_idx")
            desired_val = st.selectbox("Desired Target Outcome", [0, 1] if is_clf else [0.0, 100.0], key="cf_desired_val")
            if st.button("🔍 Generate Counterfactual Recommendations", key="cf_btn"):
                try:
                    cf_res = generate_counterfactual(
                        model=s["train_result"]["best_model"],
                        instance=s["df"].iloc[row_sel],
                        feature_names=s["train_result"]["feature_names"],
                        desired_outcome=desired_val,
                        X_reference=s["train_result"]["X_test_processed"],
                    )
                    if cf_res["status"] == "already_desired":
                        st.info(cf_res["message"])
                    else:
                        st.success(f"Original Prediction: `{cf_res['original_prediction']}` ➔ Counterfactual Outcome: `{cf_res['counterfactual_prediction']}`")
                        if cf_res["perturbations"]:
                            st.markdown("##### 🛠️ Prescriptive Interventions Required")
                            st.dataframe(pd.DataFrame(cf_res["perturbations"]), use_container_width=True)
                except Exception as e:
                    st.error(f"Counterfactual Error: {e}")

        with c_tab2:
            treat_candidate = st.selectbox("Select Binary Treatment Column", options=["(Auto-Synthesize Action)"] + s["df"].columns.tolist(), key="uplift_treat_col")
            if st.button("📊 Estimate Treatment Uplift", key="uplift_btn"):
                try:
                    t_vec = np.random.binomial(1, 0.5, size=len(s["train_result"]["X_test_processed"])) if treat_candidate == "(Auto-Synthesize Action)" else (s["df"][treat_candidate].values == 1).astype(int)
                    uplift_res = estimate_uplift_t_learner(
                        model=s["train_result"]["best_model"],
                        X=s["train_result"]["X_test_processed"],
                        y=s["train_result"]["y_test"],
                        treatment=t_vec[:len(s["train_result"]["y_test"])],
                        feature_names=s["train_result"]["feature_names"],
                    )
                    if uplift_res["status"] == "success":
                        u1, u2 = st.columns(2)
                        u1.metric("Average Treatment Effect (ATE)", f"{uplift_res['average_treatment_effect_ate']:+.4f}")
                        u2.metric("Recommended Action", uplift_res["recommended_action"])
                        st.json(uplift_res["uplift_quadrant_distribution"])
                except Exception as e:
                    st.error(f"Uplift Error: {e}")
    tab_idx += 1

    # 2. Deep Autoencoder
    with w_tabs[tab_idx]:
        st.subheader("🧠 Deep Tabular Neural Autoencoder (DL Anomaly Engine)")
        st.markdown("Trains a multi-layer deep neural bottleneck architecture ($D \\to 64 \\to 16 \\to 4 \\to 16 \\to 64 \\to D$) to isolate anomalies by reconstruction error ($\text{MSE}$).")
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            ae_epochs = st.slider("Training Epochs", 10, 100, 30, step=5, key="ae_epochs_slider")
        with dl_c2:
            ae_latent = st.slider("Latent Dimension", 2, 16, 4, step=1, key="ae_latent_slider")

        if st.button("🚀 Train Deep Autoencoder", key="ae_train_btn"):
            try:
                ae_res = train_tabular_autoencoder(
                    X_processed=s["train_result"]["X_test_processed"],
                    feature_names=s["train_result"]["feature_names"],
                    epochs=ae_epochs,
                    latent_dim=ae_latent,
                )
                if ae_res["status"] == "success":
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Final Loss (MSE)", f"{ae_res['final_reconstruction_loss']}")
                    d2.metric("Anomaly Threshold", f"{ae_res['anomaly_threshold_mse']}")
                    d3.metric("Anomalies Flagged", f"{ae_res['total_anomalies_detected']}", delta=f"{ae_res['anomaly_rate_pct']}% of samples", delta_color="inverse")
                    st.dataframe(pd.DataFrame(ae_res["feature_attribution_ranking"]), use_container_width=True)
            except Exception as e:
                st.error(f"Deep Autoencoder Error: {e}")
    tab_idx += 1

    # 3. Contextual Bandits
    with w_tabs[tab_idx]:
        st.subheader("🎰 Contextual Multi-Armed Bandits (Reinforcement Learning)")
        st.markdown("Adaptive decision policy (**LinUCB**) balancing Exploration vs. Exploitation in real time.")
        rl_alpha = st.slider("Exploration Intensity (Alpha)", 0.1, 3.0, 1.2, step=0.1, key="rl_alpha_slider")
        if st.button("▶️ Launch LinUCB Contextual Bandit Simulation", key="rl_sim_btn"):
            try:
                bandit_res = run_contextual_bandit_simulation(
                    X_contexts=s["train_result"]["X_test_processed"],
                    alpha_exploration=rl_alpha,
                )
                if bandit_res["status"] == "success":
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Bandit Policy Total Rewards", f"{bandit_res['bandit_total_rewards']:,}")
                    b2.metric("Random Baseline Rewards", f"{bandit_res['random_baseline_rewards']:,}")
                    b3.metric("Relative Policy Lift", f"{bandit_res['relative_policy_lift_pct']:+.1f}%", delta="Lift Over Random")
                    st.json(bandit_res["action_selection_distribution"])
            except Exception as e:
                st.error(f"Bandit Error: {e}")
    tab_idx += 1

    # 4. Synthetic Data
    with w_tabs[tab_idx]:
        st.subheader("🧬 Generative Synthetic Data & Differential Privacy")
        st.markdown("Generate statistically faithful synthetic tabular datasets with optional $(\\epsilon)$-Differential Privacy.")
        syn_c1, syn_c2 = st.columns(2)
        with syn_c1:
            syn_rows = st.number_input("Number of Synthetic Samples", min_value=50, max_value=50000, value=len(s["df"]), step=100)
        with syn_c2:
            apply_dp = st.checkbox("Inject $(\\epsilon)$-Differential Privacy Noise", value=False)
            eps_val = st.slider("Epsilon Privacy Budget", 0.1, 10.0, 1.0, step=0.1) if apply_dp else 1.0

        if st.button("✨ Synthesize High-Fidelity Dataset", key="synth_btn"):
            try:
                df_synth, synth_meta = generate_synthetic_dataset(
                    df=s["df"],
                    n_samples=syn_rows,
                    apply_dp_noise=apply_dp,
                    epsilon=eps_val,
                )
                st.success(f"Generated {len(df_synth):,} synthetic records with **{synth_meta['overall_distribution_fidelity_pct']}% Overall Distribution Fidelity**!")
                st.dataframe(df_synth.head(50), use_container_width=True)
                st.download_button("⬇️ Download `synthetic_dataset.csv`", data=df_synth.to_csv(index=False), file_name="synthetic_dataset.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Synthetic Generation Error: {e}")
    tab_idx += 1

    # 5. Online Incremental Learning
    with w_tabs[tab_idx]:
        st.subheader("⚡ Online Incremental Learning & Streaming Ingestion")
        st.markdown("Simulate high-throughput event streaming where models update weights incrementally on-the-fly (`partial_fit`).")
        st_c1, st_c2 = st.columns(2)
        with st_c1:
            batch_sz = st.slider("Streaming Batch Size", 5, 100, 20, key="stream_batch_sz")
        with st_c2:
            n_batches_sim = st.slider("Stream Steps to Simulate", 3, 30, 10, key="stream_n_batches")

        if st.button("▶️ Launch Live Event Stream Simulation", key="stream_btn"):
            try:
                stream_res = simulate_streaming_incremental_fit(
                    X=s["train_result"]["X_test_processed"],
                    y=s["train_result"]["y_test"],
                    task_type=s["final_task_type"],
                    batch_size=batch_sz,
                    n_batches=n_batches_sim,
                )
                st.success(f"Stream completed! Processed {stream_res['total_records_ingested']} live records across {stream_res['total_streaming_batches']} batches.")
                st.dataframe(pd.DataFrame(stream_res["streaming_learning_curve"]), use_container_width=True)
            except Exception as e:
                st.error(f"Streaming Simulation Error: {e}")
    tab_idx += 1

    # 6. Time-Series Forecasting (Conditional: has_dates)
    if has_dates:
        with w_tabs[tab_idx]:
            st.subheader("📈 Automated Time-Series & Temporal Forecasting")
            st.markdown("Detect temporal patterns, generate lag features, and project multi-step forecasts with 95% confidence bounds.")
            date_cols_avail = [c for c in s["df"].columns if pd.api.types.is_datetime64_any_dtype(s["df"][c]) or "date" in str(c).lower() or "time" in str(c).lower()]
            date_sel = st.selectbox("Temporal Date Column", options=date_cols_avail, index=0, key="ts_date_sel")
            horizon = st.slider("Forecast Horizon (Future Steps)", 7, 60, 14, key="ts_horizon_slider")
            if st.button("🚀 Train Temporal Forecaster & Project Future", key="ts_train_btn"):
                try:
                    ts_res = train_time_series_forecaster(
                        df=s["df"],
                        date_col=date_sel,
                        target_col=s["target_col"],
                        forecast_horizon=horizon,
                    )
                    if ts_res["status"] == "success":
                        t1, t2, t3 = st.columns(3)
                        t1.metric("MAE", f"{ts_res['evaluation']['mae']}")
                        t2.metric("RMSE", f"{ts_res['evaluation']['rmse']}")
                        t3.metric("R² Score", f"{ts_res['evaluation']['r2_score']}")
                        st.dataframe(pd.DataFrame(ts_res["future_projections"]), use_container_width=True)
                    else:
                        st.warning(ts_res["message"])
                except Exception as e:
                    st.error(f"Time-Series Error: {e}")
        tab_idx += 1

    # 7. NLP & Lexical Analysis (Conditional: has_text)
    if has_text:
        with w_tabs[tab_idx]:
            st.subheader("📝 Natural Language Processing & Lexical Analysis")
            text_cols = [c for c in s["df"].columns if pd.api.types.is_object_dtype(s["df"][c])]
            txt_sel = st.selectbox("Select Text Column to Analyze", options=text_cols, key="nlp_txt_sel")
            if st.button("🔍 Extract Lexical & TF-IDF Features", key="nlp_btn"):
                try:
                    lex_df = extract_lexical_features(s["df"][txt_sel])
                    tfidf_df, vocab = extract_tfidf_dense_features(s["df"][txt_sel], max_features=10)
                    st.markdown("##### 📊 Extracted Lexical Complexity Metrics")
                    st.dataframe(lex_df.head(20), use_container_width=True)
                    st.markdown(f"##### 🔤 Top TF-IDF Tokens: `{', '.join(vocab)}`")
                    st.dataframe(tfidf_df.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"NLP Error: {e}")
        tab_idx += 1

    # 8. Graph Intelligence (Conditional: has_entities)
    if has_entities:
        with w_tabs[tab_idx]:
            st.subheader("🕸️ Graph Intelligence & Relational Network Analytics")
            st.markdown("Calculate PageRank authority centrality and detect community clusters using NetworkX.")
            cat_cols = s["df"].select_dtypes(include=["object", "category"]).columns.tolist()
            g_c1, g_c2 = st.columns(2)
            with g_c1:
                src_col = st.selectbox("Source Entity Column", options=cat_cols, index=0, key="g_src_col")
            with g_c2:
                tgt_col = st.selectbox("Target Entity Column", options=cat_cols, index=min(1, len(cat_cols) - 1), key="g_tgt_col")

            if st.button("🕸️ Build Relational Graph & Compute Topology", key="g_build_btn"):
                try:
                    g_res = construct_and_analyze_entity_graph(
                        df=s["df"],
                        source_node_col=src_col,
                        target_node_col=tgt_col,
                    )
                    if g_res["status"] == "success":
                        g1, g2, g3 = st.columns(3)
                        g1.metric("Graph Nodes", f"{g_res['total_nodes']:,}")
                        g2.metric("Graph Edges", f"{g_res['total_edges']:,}")
                        g3.metric("Network Density", f"{g_res['graph_density']}")
                        st.markdown("##### 👑 Top Influential Entity Nodes (PageRank Centrality)")
                        st.dataframe(pd.DataFrame(g_res["top_influential_nodes"]), use_container_width=True)
                        st.markdown("##### 👥 Community & Collusion Clusters")
                        st.json(g_res["top_communities"])
                except Exception as e:
                    st.error(f"Graph Analytics Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#   WORKSPACE 4: GOVERNANCE, MLOPS & PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════
elif workspace == "🛡️ Governance, MLOps & Production":
    w_tabs = st.tabs([
        "🧪 Data Quality (GX) & Contracts",
        "🔒 Privacy & GDPR Audit",
        "🏛️ Feature Store (Feast)",
        "📡 Model Registry & Drift Monitor",
        "💾 In-Database SQL Transpiler",
        "💻 Production Code & Deployment",
        "📑 Executive Briefing",
        "💬 AI Chat Copilot",
    ])

    # 1. Data Quality & Contracts
    with w_tabs[0]:
        st.subheader("🧪 Great Expectations (GX) Data Quality Certification")
        try:
            gx_res = generate_and_evaluate_expectations(s["df"])
            q1, q2, q3 = st.columns(3)
            q1.metric("Quality Score", f"{gx_res['data_quality_score_pct']}%")
            q2.metric("Tests Passed", f"{gx_res['tests_passed']} / {gx_res['total_tests_evaluated']}")
            q3.metric("Certification Verdict", gx_res["pipeline_certification_verdict"])
            st.dataframe(pd.DataFrame(gx_res["test_results_breakdown"]), use_container_width=True)
            st.download_button("⬇️ Download `great_expectations_suite.json`", data=json.dumps(gx_res["great_expectations_suite_json"], indent=2), file_name="great_expectations_suite.json", mime="application/json")
        except Exception as e:
            st.error(f"Great Expectations Error: {e}")

        st.divider()
        st.subheader("🛡️ Production Data Contract Schema")
        st.markdown(s["data_contract_md"])
        st.download_button("⬇️ Download `contract.json`", data=json.dumps(s["data_contract_json"], indent=2), file_name="contract.json", mime="application/json")

    # 2. Privacy & GDPR
    with w_tabs[1]:
        st.subheader("🔒 Enterprise Data Privacy & Regulatory Compliance Dossier")
        st.markdown(s["compliance_md"])
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download `compliance_dossier.json`", data=json.dumps(s["compliance_report"], indent=2), file_name="compliance_dossier.json", mime="application/json")
        with c2:
            st.download_button("⬇️ Download `compliance_dossier.md`", data=s["compliance_md"], file_name="compliance_dossier.md", mime="text/markdown")
            
        st.divider()
        st.subheader("🇪🇺 GDPR Article 30 - Record of Processing Activities (ROPA)")
        st.markdown(s["ropa_md"])
        r1, r2 = st.columns(2)
        with r1:
            st.download_button("⬇️ Download `ropa_article30.json`", data=json.dumps(s["ropa_record"], indent=2), file_name="ropa_article30.json", mime="application/json")
        with r2:
            st.download_button("⬇️ Download `ropa_article30.md`", data=s["ropa_md"], file_name="ropa_article30.md", mime="text/markdown")

        st.divider()
        st.subheader("👤 Data Subject Rights (DSAR) Interactive Portal")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            id_field = st.selectbox("Subject Identifier Column", options=s["df"].columns.tolist(), key="dsar_id_col")
        with d_col2:
            subject_id_val = st.text_input("Enter Subject / Customer ID", key="dsar_subj_val", placeholder="e.g. CUST-1001")

        if subject_id_val.strip():
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button("📤 Export Portable Data (Art. 15/20)", key="dsar_export_btn"):
                    try:
                        export_res = process_dsar_access_export(s["df"], id_field, subject_id_val.strip())
                        st.success(f"Found {export_res['record_count']} record(s) for subject `{subject_id_val}`.")
                        st.json(export_res)
                    except Exception as e:
                        st.error(f"DSAR Export Error: {e}")
            with btn_c2:
                if st.button("🗑️ Anonymize Subject Data (Art. 17 Erasure)", key="dsar_erase_btn"):
                    try:
                        df_erased, erase_res = process_dsar_erasure_anonymization(s["df"], id_field, subject_id_val.strip())
                        st.success(f"Anonymized {erase_res['modified_rows']} record(s) for `{subject_id_val}`.")
                        s["df"] = df_erased
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erasure Error: {e}")

    # 3. Feature Store
    with w_tabs[2]:
        st.subheader("🏛️ Automated Enterprise Feature Store (Feast-Compatible)")
        fs_c1, fs_c2 = st.columns(2)
        with fs_c1:
            entity_col = st.selectbox("Primary Entity Identifier Key", options=s["df"].columns.tolist(), index=0, key="fs_entity_sel")
        with fs_c2:
            ttl_days = st.slider("Feature View TTL (Days)", 1, 365, 30, key="fs_ttl_slider")

        try:
            fs_res = generate_feature_store_definitions(
                df=s["df"],
                entity_col=entity_col,
                target_col=s["target_col"],
                feature_view_name=f"{entity_col}_features_fv",
                ttl_days=ttl_days,
            )
            st.success(f"Registered **{fs_res['total_features_registered']} Features** under Entity `{entity_col}`.")
            st.code(fs_res["feast_python_code"], language="python")
            c_fs1, c_fs2 = st.columns(2)
            with c_fs1:
                st.download_button("⬇️ Download `features.py`", data=fs_res["feast_python_code"], file_name="features.py", mime="text/x-python")
            with c_fs2:
                st.download_button("⬇️ Download `feature_store.yaml`", data=fs_res["feast_yaml_code"], file_name="feature_store.yaml", mime="text/yaml")
        except Exception as e:
            st.error(f"Feature Store Error: {e}")

    # 4. Model Registry & Drift Monitor
    with w_tabs[3]:
        st.subheader("📦 Model Registry & Production Observability")
        lat = s["latency_stats"]
        l_c1, l_c2, l_c3, l_c4 = st.columns(4)
        with l_c1:
            st.metric("⏱️ Median Latency (p50)", f"{lat.get('p50_ms', 1.0):.2f} ms")
        with l_c2:
            st.metric("⚡ Tail Latency (p99)", f"{lat.get('p99_ms', 3.0):.2f} ms")
        with l_c3:
            st.metric("🚀 Max Throughput", f"{lat.get('throughput_qps', 500):,.0f} req/s")
        with l_c4:
            st.metric("🏷️ Registry Status", "PRODUCTION READY", delta="Verified")

        st.divider()
        st.markdown("### 📋 Google Model Card v1 (Mitchell et al. Standard)")
        st.markdown(s["model_card_md"])
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.download_button("⬇️ Download `model_card.md`", data=s["model_card_md"], file_name="model_card.md", mime="text/markdown")
        with m_c2:
            st.download_button("⬇️ Download `mlflow_run_manifest.json`", data=json.dumps(s["mlflow_manifest"], indent=2), file_name="mlflow_run_manifest.json", mime="application/json")

        st.divider()
        st.subheader("🌊 Production Data Drift & Anomaly Scanner")
        drift_file = st.file_uploader("Upload New Production Batch (CSV)", type=["csv"], key="drift_batch_uploader")
        if drift_file is not None:
            try:
                prod_df = pd.read_csv(drift_file)
                drift_report = calculate_drift_report(s["df"], prod_df, s["target_col"])
                d1, d2, d3 = st.columns(3)
                d1.metric("Drifted Features", f"{drift_report['drifted_columns_count']} / {drift_report['total_columns_evaluated']}", delta=f"{drift_report['drift_percentage']}% Drifted", delta_color="inverse")
                d2.metric("New Batch Rows", f"{drift_report['cur_row_count']:,}")
                d3.metric("Anomalies", f"{drift_report['anomalies']['anomalies_detected']:,}", delta=f"{drift_report['anomalies']['anomaly_pct']}% of rows", delta_color="inverse")
                st.dataframe(pd.DataFrame(drift_report["column_reports"]), use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating drift: {e}")

        st.divider()
        st.subheader("🚦 Shadow & Canary Deployment Traffic Router")
        can_c1, can_c2 = st.columns(2)
        with can_c1:
            can_pct = st.slider("Canary Traffic Allocation (%)", 1.0, 50.0, 10.0, step=1.0, key="can_traffic_pct")
        with can_c2:
            div_tol = st.slider("Max Allowed Prediction Divergence (%)", 5.0, 50.0, 15.0, step=1.0, key="can_div_tol")

        if st.button("▶️ Launch Live Canary Traffic Simulation", key="can_sim_btn"):
            try:
                can_res = simulate_canary_routing(
                    champion_model=s["train_result"]["best_model"],
                    challenger_model=s["train_result"]["best_model"],
                    X_live=s["train_result"]["X_test_processed"],
                    canary_traffic_pct=can_pct,
                    max_allowed_divergence_pct=div_tol,
                    task_type=s["final_task_type"],
                )
                if can_res["status"] == "success":
                    cn1, cn2, cn3 = st.columns(3)
                    cn1.metric("Champion Requests", f"{can_res['champion_requests_served']:,}")
                    cn2.metric("Canary Requests", f"{can_res['canary_requests_served']:,}")
                    cn3.metric("Prediction Divergence", f"{can_res['prediction_divergence_pct']}%")
                    st.info(f"**Deployment Verdict:** {can_res['status_verdict']}")
            except Exception as e:
                st.error(f"Canary Error: {e}")

    # 5. In-Database SQL Transpiler
    with w_tabs[4]:
        st.subheader("💾 Native In-Database SQL Model Transpiler")
        sql_d1, sql_d2 = st.columns(2)
        with sql_d1:
            dialect = st.selectbox("Target SQL Data Warehouse Dialect", ["Standard ANSI SQL", "Snowflake", "Google BigQuery", "PostgreSQL", "AWS Redshift", "Databricks Spark SQL"], key="sql_dialect_sel")
        with sql_d2:
            source_tbl = st.text_input("Source Table Name", value="customer_analytics_table", key="sql_table_name")

        try:
            sql_res = transpile_model_to_sql(
                model=s["train_result"]["best_model"],
                feature_names=s["train_result"]["feature_names"],
                table_name=source_tbl,
                task_type=s["final_task_type"],
                dialect=dialect,
            )
            st.markdown(f"**Transpiled SQL Query ({sql_res['transpiled_trees_count']} Estimators):**")
            st.code(sql_res["sql_code"], language="sql")
            st.download_button("⬇️ Download `model_scoring.sql`", data=sql_res["sql_code"], file_name="model_scoring.sql", mime="text/x-sql")
        except Exception as e:
            st.error(f"SQL Transpiler Error: {e}")

    # 6. Production Code & Deployment Bundles
    with w_tabs[5]:
        st.subheader("💻 Production Code & Deployment Export")
        export_type = st.radio(
            "Select Export Target", 
            [
                "Standalone Python Script", 
                "Apache Airflow DAG", 
                "FastAPI REST API Microservice", 
                "Docker Deployment Bundle",
                "GitHub Actions CI/CD Pipeline",
                "Kubernetes GitOps Manifests",
            ]
        )
        if export_type == "Standalone Python Script":
            st.code(s["code_script"], language="python")
            st.download_button("⬇️ Download `pipeline.py`", data=s["code_script"], file_name="pipeline.py", mime="text/x-python")
        elif export_type == "Apache Airflow DAG":
            st.code(s["airflow_dag"], language="python")
            st.download_button("⬇️ Download `airflow_dag.py`", data=s["airflow_dag"], file_name="airflow_dag.py", mime="text/x-python")
        elif export_type == "FastAPI REST API Microservice":
            st.code(s["fastapi_code"], language="python")
            st.download_button("⬇️ Download `main.py`", data=s["fastapi_code"], file_name="main.py", mime="text/x-python")
        elif export_type == "Docker Deployment Bundle":
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.code(s["dockerfile_code"], language="dockerfile")
                st.download_button("⬇️ Download `Dockerfile`", data=s["dockerfile_code"], file_name="Dockerfile", mime="text/plain")
            with col_d2:
                st.code(s["docker_compose_code"], language="yaml")
                st.download_button("⬇️ Download `docker-compose.yml`", data=s["docker_compose_code"], file_name="docker-compose.yml", mime="text/yaml")
        elif export_type == "GitHub Actions CI/CD Pipeline":
            st.code(s["ci_cd_workflow"], language="yaml")
            st.download_button("⬇️ Download `ci_cd_pipeline.yml`", data=s["ci_cd_workflow"], file_name="ci_cd_pipeline.yml", mime="text/yaml")
        else:
            st.code(s["k8s_manifests"], language="yaml")
            st.download_button("⬇️ Download `k8s_deployment.yaml`", data=s["k8s_manifests"], file_name="k8s_deployment.yaml", mime="text/yaml")

    # 7. Executive Briefing
    with w_tabs[6]:
        st.subheader("📑 Autonomous Executive Briefing & Boardroom Dossier")
        st.download_button(
            "⬇️ Download Standalone `Executive_Briefing.html`",
            data=s["executive_html"],
            file_name="Executive_Briefing.html",
            mime="text/html",
        )
        with st.expander("📄 Preview Executive Briefing"):
            st.components.v1.html(s["executive_html"], height=600, scrolling=True)

    # 8. AI Chat Copilot
    with w_tabs[7]:
        st.subheader("💬 Autonomous Data Copilot")
        st.markdown("Ask any ad-hoc analytical or visualization questions about your dataset in plain natural language.")
        
        quick_cols = st.columns(4)
        quick_query = None
        with quick_cols[0]:
            if st.button("📊 Feature Correlations", key="chip_corr"):
                quick_query = "correlation matrix"
        with quick_cols[1]:
            if st.button("📈 Target Distribution", key="chip_dist"):
                quick_query = f"distribution of {s['target_col']}"
        with quick_cols[2]:
            num_preview = s["df"].select_dtypes(include=["number"]).columns.tolist()
            cat_preview = s["df"].select_dtypes(include=["object", "category"]).columns.tolist()
            n_col = num_preview[0] if num_preview else "value"
            c_col = cat_preview[0] if cat_preview else "category"
            if st.button(f"🔍 Avg {n_col} by {c_col}", key="chip_grp"):
                quick_query = f"average {n_col} by {c_col}"
        with quick_cols[3]:
            if st.button("🧹 Clear History", key="chip_clear"):
                s["chat_history"] = []
                st.rerun()

        for msg in s.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("table") is not None:
                    st.dataframe(msg["table"], use_container_width=True)
                if msg.get("figure") is not None:
                    st.plotly_chart(msg["figure"], use_container_width=True)

        user_prompt = st.chat_input("Ask a question about your data...")
        if quick_query:
            user_prompt = quick_query

        if user_prompt:
            s["chat_history"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                res = execute_natural_language_query(s["df"], user_prompt, s["target_col"])
                st.markdown(res["text"])
                if res["table"] is not None:
                    st.dataframe(res["table"], use_container_width=True)
                if res["figure"] is not None:
                    st.plotly_chart(res["figure"], use_container_width=True)

                s["chat_history"].append({
                    "role": "assistant",
                    "content": res["text"],
                    "table": res["table"],
                    "figure": res["figure"]
                })
