"""
dia/pipeline_coordinator.py
───────────────────────────
Unified Pipeline Coordinator & Execution Orchestrator.
Provides clean service orchestration, cross-engine state encapsulation,
and memoized caching for end-to-end data intelligence workflows.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from dia.active_learning import sample_uncertain_predictions
from dia.code_generator import (
    generate_airflow_dag,
    generate_docker_compose,
    generate_dockerfile,
    generate_fastapi_app,
    generate_github_actions_pipeline,
    generate_k8s_manifests,
    generate_pipeline_code,
)
from dia.compliance import format_compliance_dossier_markdown, scan_dataset_privacy
from dia.data_profiler import (
    detect_target_type,
    generate_readiness_report,
    infer_column_roles,
    profile_dataframe,
)
from dia.data_quality import format_contract_markdown, generate_data_contract
from dia.explainability import generate_explanation
from dia.gdpr import format_gdpr_audit_markdown, generate_ropa_record
from dia.goal_parser import parse_goal
from dia.insights import generate_smart_insights
from dia.mlops_registry import (
    benchmark_model_latency,
    generate_mlflow_run_manifest,
    generate_model_card_markdown,
)
from dia.model_trainer import train_and_evaluate
from dia.report_generator import generate_executive_html_report

log = logging.getLogger("dia.coordinator")


class PipelineCoordinator:
    """Central orchestrator managing multi-stage execution and state propagation."""

    @staticmethod
    def execute_full_pipeline(
        df: pd.DataFrame,
        goal_text: str,
        user_target_col: str | None = None,
        user_task_type: str | None = None,
        selected_models: list[str] | None = None,
        use_hpo: bool = False,
        use_voting: bool = True,
        hpo_trials: int = 10,
        test_size: float = 0.20,
    ) -> dict[str, Any]:
        """
        Executes all 5 core stages synchronously and produces the complete session state dictionary.
        """
        # Stage 0: Autonomous Malformed Data Repair & Sanitization
        from dia.data_sanitizer import format_sanitization_report_markdown, sanitize_dataframe
        clean_df, sanitize_report = sanitize_dataframe(df)
        sanitize_md = format_sanitization_report_markdown(sanitize_report)

        # Stage 1: Data Profiling
        meta = {
            "n_rows": int(clean_df.shape[0]),
            "n_cols": int(clean_df.shape[1]),
            "file_size_mb": round(float(clean_df.memory_usage(deep=True).sum() / (1024 * 1024)), 2),
            "encoding": "utf-8",
            "sanitize_report": sanitize_report,
        }
        profile_df = profile_dataframe(clean_df)

        # Stage 2: Goal Parsing & Column Roles
        from dia.column_resolver import resolve_column
        goal_info = parse_goal(goal_text, clean_df.columns.tolist())

        # Resolve target column safely (handles typos, synonyms, and value matches)
        target_col = None
        if user_target_col:
            resolved_col, conf, _ = resolve_column(user_target_col, clean_df)
            if resolved_col:
                target_col = resolved_col

        if not target_col and goal_info.get("target_candidates"):
            for cand in goal_info["target_candidates"]:
                resolved_col, conf, _ = resolve_column(cand, clean_df)
                if resolved_col:
                    target_col = resolved_col
                    break

        if not target_col:
            target_col = clean_df.columns[-1]

        annotated_profile = infer_column_roles(clean_df, profile_df)

        detected_type_dict = detect_target_type(clean_df, target_col)
        final_task_type = user_task_type or goal_info.get("task_type") or detected_type_dict.get("task_type", "classification")

        # Stage 3: Data Readiness Audit & Smart Insights
        readiness = generate_readiness_report(
            df=clean_df,
            annotated_profile=annotated_profile,
            goal_info=goal_info,
            target_col=target_col,
            task_type=final_task_type,
        )
        insights = generate_smart_insights(
            df=clean_df,
            target_col=target_col,
            task_type=final_task_type,
        )

        # Stage 4: AutoML Model Training
        model_keys = selected_models or (["rf", "logreg"] if final_task_type == "classification" else ["rf", "linreg"])
        train_result = train_and_evaluate(
            df=clean_df,
            target_col=target_col,
            task_type=final_task_type,
            selected_model_keys=model_keys,
            test_size=test_size,
            enable_hpo=use_hpo,
            hpo_iter=hpo_trials,
            build_ensemble=use_voting,
        )

        # Stage 5: Explainability & Governance Precomputations
        best_idx = next(
            (i for i, r in enumerate(train_result["results"])
             if r["model_key"] == train_result["best_model_key"]),
            0,
        )
        explanation = generate_explanation(
            model=train_result["best_model"],
            X_test=train_result["X_test_processed"],
            feature_names=train_result.get("feature_names", []),
            importance_series=train_result["results"][best_idx].get("importance", pd.Series(dtype=float)),
        )

        # Privacy & Compliance
        compliance_report = scan_dataset_privacy(clean_df)
        compliance_md = format_compliance_dossier_markdown(compliance_report)
        ropa_record = generate_ropa_record(clean_df, target_col, final_task_type)
        ropa_md = format_gdpr_audit_markdown(ropa_record)

        # MLOps Benchmarking & Lineage
        latency_stats = benchmark_model_latency(train_result["best_model"], train_result["X_test_processed"])
        mlflow_manifest = generate_mlflow_run_manifest(
            model_name=train_result["best_model_label"],
            target_col=target_col,
            task_type=final_task_type,
            metrics=train_result["best_metrics"],
            params=train_result.get("best_params", {}),
            feature_names=train_result.get("feature_names", []),
            df=clean_df,
            latency_stats=latency_stats,
        )
        model_card_md = generate_model_card_markdown(
            model_name=train_result["best_model_label"],
            target_col=target_col,
            task_type=final_task_type,
            metrics=train_result["best_metrics"],
            feature_names=train_result.get("feature_names", []),
            df=clean_df,
            latency_stats=latency_stats,
        )

        # Contracts & Active Learning
        data_contract_json = generate_data_contract(clean_df, target_col)
        data_contract_md = format_contract_markdown(data_contract_json)
        uncertain_samples = sample_uncertain_predictions(
            train_result["best_model"], clean_df, train_result["X_test_processed"]
        )

        # Executive HTML Report
        executive_html = generate_executive_html_report(
            goal=goal_text,
            target_col=target_col,
            task_type=final_task_type,
            train_result=train_result,
            readiness=readiness,
            compliance_report=compliance_report,
            latency_stats=latency_stats,
        )

        # Code Exports
        code_script = generate_pipeline_code(
            source_label="In-Memory Dataset",
            target_col=target_col,
            task_type=final_task_type,
            best_model_key=train_result.get("best_model_key", "rf"),
            model_label=train_result.get("best_model_label", "Random Forest"),
            best_params=train_result.get("best_params", {}),
        )
        airflow_dag = generate_airflow_dag(
            source_label="In-Memory Dataset",
            target_col=target_col,
            task_type=final_task_type,
            best_model_key=train_result.get("best_model_key", "rf"),
            model_label=train_result.get("best_model_label", "Random Forest"),
        )
        fastapi_code = generate_fastapi_app(
            target_col=target_col,
            task_type=final_task_type,
            model_label=train_result.get("best_model_label", "Random Forest"),
            raw_feature_cols=train_result.get("raw_feature_cols", []),
        )
        dockerfile_code = generate_dockerfile()
        docker_compose_code = generate_docker_compose()
        ci_cd_workflow = generate_github_actions_pipeline(
            model_label=train_result.get("best_model_label", "Random Forest"),
            target_col=target_col,
        )
        k8s_manifests = generate_k8s_manifests(
            model_label=train_result.get("best_model_label", "Random Forest"),
        )

        return {
            "df": clean_df,
            "meta": meta,
            "profile_df": profile_df,
            "sanitize_report": sanitize_report,
            "sanitize_md": sanitize_md,
            "goal": goal_text,
            "goal_info": goal_info,
            "target_col": target_col,
            "final_task_type": final_task_type,
            "annotated_profile": annotated_profile,
            "readiness": readiness,
            "insights": insights,
            "train_result": train_result,
            "explanation": explanation,
            "compliance_report": compliance_report,
            "compliance_md": compliance_md,
            "ropa_record": ropa_record,
            "ropa_md": ropa_md,
            "latency_stats": latency_stats,
            "mlflow_manifest": mlflow_manifest,
            "model_card_md": model_card_md,
            "data_contract_json": data_contract_json,
            "data_contract_md": data_contract_md,
            "uncertain_samples": uncertain_samples,
            "executive_html": executive_html,
            "code_script": code_script,
            "airflow_dag": airflow_dag,
            "fastapi_code": fastapi_code,
            "dockerfile_code": dockerfile_code,
            "docker_compose_code": docker_compose_code,
            "ci_cd_workflow": ci_cd_workflow,
            "k8s_manifests": k8s_manifests,
            "pipeline_executed": True,
        }
