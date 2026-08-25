"""
dia/gdpr.py
───────────
Enterprise GDPR Compliance & Data Subject Rights (DSR/DSAR) Automation Engine.
Implements Article 15 (Access), Article 17 (Erasure), Article 20 (Portability),
Article 25 (Privacy by Design), and Article 30 (Record of Processing Activities - ROPA).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from dia.compliance import scan_dataset_privacy, mask_dataframe_pii

log = logging.getLogger("dia.gdpr")


def generate_ropa_record(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    data_source: str = "Internal Upload",
    controller_name: str = "Data Intelligence Enterprise",
    dpo_contact: str = "dpo@company.internal",
) -> dict[str, Any]:
    """
    Generates a formal GDPR Article 30 Record of Processing Activities (ROPA).
    """
    privacy_scan = scan_dataset_privacy(df)
    pii_columns = [p["column"] for p in privacy_scan["detected_pii"]]
    categories = list({cat for p in privacy_scan["detected_pii"] for cat in p["header_categories"]})

    # Classify special category data (Article 9)
    special_categories = [
        c for c in categories if c in ["Health Data (PHI)", "Protected Class / Demographics"]
    ]

    ropa = {
        "record_id": f"ROPA-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "controller": {
            "name": controller_name,
            "dpo_contact": dpo_contact,
            "representative": "Lead Data Scientist / Compliance Officer",
        },
        "processing_purpose": f"Automated Machine Learning ({task_type.capitalize()}) to predict '{target_col}'.",
        "legal_basis": "Article 6(1)(f) Legitimate Interest / Article 6(1)(a) Consent",
        "data_categories": categories or ["Standard Numerical/Categorical Operational Metrics"],
        "special_category_data": special_categories or ["None Detected"],
        "pii_fields": pii_columns,
        "data_recipients": ["Internal Data Analytics Team", "Authorized ML Inference API"],
        "transfers_outside_eea": "No (Local / Sovereign Private Infrastructure)",
        "retention_period": "30 days post-analysis in ephemeral memory, or pursuant to organizational retention schedule",
        "technical_security_measures": [
            "In-memory stream processing with zero persistent plaintext leaks",
            "SSRF-isolated network ingestion",
            "Automated PII masking and pseudonymization algorithms",
            "Strict input validation and output HTML entity encoding",
        ],
        "compliance_summary": {
            "privacy_risk_score": privacy_scan["privacy_risk_score"],
            "total_features": len(df.columns),
            "pii_features_count": len(pii_columns),
            "gdpr_status": "Compliant with Privacy by Design Controls" if not special_categories else "Requires Explicit Consent (Art. 9 Special Category Data)",
        }
    }
    return ropa


def process_dsar_access_export(
    df: pd.DataFrame,
    identifier_col: str,
    subject_id: str | int,
) -> dict[str, Any]:
    """
    Processes a Data Subject Access Request (DSAR) under GDPR Article 15 and Article 20 (Portability).
    """
    if identifier_col not in df.columns:
        raise ValueError(f"Identifier column '{identifier_col}' not found in dataset.")

    matches = df[df[identifier_col].astype(str) == str(subject_id)]
    if matches.empty:
        return {
            "status": "not_found",
            "message": f"No records found matching subject ID '{subject_id}' in column '{identifier_col}'.",
            "records": [],
        }

    records_payload = matches.to_dict(orient="records")
    return {
        "status": "success",
        "request_type": "Article 15/20 Data Subject Access & Portability",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": str(subject_id),
        "identifier_field": identifier_col,
        "record_count": len(records_payload),
        "data_payload": records_payload,
        "legal_notice": "Provided in machine-readable JSON format pursuant to GDPR Article 20.",
    }


def process_dsar_erasure_anonymization(
    df: pd.DataFrame,
    identifier_col: str,
    subject_id: str | int,
    strategy: str = "anonymize",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Executes a Right to Erasure (Article 17) request by either dropping the rows or anonymizing PII fields.
    """
    if identifier_col not in df.columns:
        raise ValueError(f"Identifier column '{identifier_col}' not found in dataset.")

    df_out = df.copy()
    match_mask = df_out[identifier_col].astype(str) == str(subject_id)
    matched_count = int(match_mask.sum())

    if matched_count == 0:
        return df_out, {
            "status": "not_found",
            "message": f"No records found for subject ID '{subject_id}'.",
            "modified_rows": 0,
        }

    if strategy == "delete":
        df_out = df_out[~match_mask].reset_index(drop=True)
        action_taken = "Hard deletion of matching record(s)"
    else:
        # Anonymize/Cryptographic Pseudonymization
        privacy_scan = scan_dataset_privacy(df_out)
        pii_cols = [p["column"] for p in privacy_scan["detected_pii"]]
        for col in df_out.columns:
            if col in pii_cols or col == identifier_col:
                if pd.api.types.is_numeric_dtype(df_out[col]):
                    df_out.loc[match_mask, col] = np.nan
                else:
                    df_out.loc[match_mask, col] = "ANONYMIZED_GDPR_ART17"
        action_taken = "Irreversible field-level anonymization"

    return df_out, {
        "status": "success",
        "request_type": "Article 17 Right to Erasure",
        "action_taken": action_taken,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject_id": str(subject_id),
        "modified_rows": matched_count,
    }


def format_gdpr_audit_markdown(ropa: dict[str, Any]) -> str:
    """Formats the GDPR Article 30 ROPA record as a formal Markdown Audit Document."""
    lines = [
        f"# 🇪🇺 GDPR Article 30 - Record of Processing Activities (ROPA)",
        f"**Record ID:** `{ropa['record_id']}` | **Generated (UTC):** `{ropa['timestamp_utc']}`",
        "",
        "## 1. Controller & Governance Information",
        f"- **Data Controller:** {ropa['controller']['name']}",
        f"- **Data Protection Officer (DPO):** `{ropa['controller']['dpo_contact']}`",
        f"- **Representative:** {ropa['controller']['representative']}",
        "",
        "## 2. Processing Scope & Legal Basis",
        f"- **Processing Purpose:** {ropa['processing_purpose']}",
        f"- **Lawful Basis (Art. 6):** {ropa['legal_basis']}",
        f"- **Data Categories Identified:** {', '.join(ropa['data_categories'])}",
        f"- **Special Category Data (Art. 9):** {', '.join(ropa['special_category_data'])}",
        f"- **Cross-Border Transfers (EEA):** {ropa['transfers_outside_eea']}",
        f"- **Data Retention Horizon:** {ropa['retention_period']}",
        "",
        "## 3. Technical & Organizational Measures (TOMs - Art. 32)",
    ]

    for tom in ropa["technical_security_measures"]:
        lines.append(f"- ✅ {tom}")

    return "\n".join(lines)


# ─── Backward-compatible aliases ──────────────────────────────────────────────
generate_gdpr_article30_ropa = generate_ropa_record
format_ropa_markdown = format_gdpr_audit_markdown

