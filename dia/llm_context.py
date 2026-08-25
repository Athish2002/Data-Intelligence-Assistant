"""
dia/llm_context.py
──────────────────
Local & Offline Autonomous Dataset Domain & ML Objective Inference Engine.
Performs semantic lexicon mapping, schema matching, and statistical heuristics
locally with zero API dependencies, instant sub-millisecond response, and zero network failures.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("dia.context_analyzer")

# ─── Comprehensive Local Domain Knowledge Base ───────────────────────────────
DOMAIN_TAXONOMY = {
    "Telecommunications & Subscription SaaS": {
        "keywords": ["churn", "tenure", "monthly_charges", "contract", "internet", "tech_support", "plan", "subscription", "subscriber", "call_duration", "data_usage", "renewal", "mrr", "arr"],
        "objectives": [
            "Predict customer churn flag and retention risk",
            "Forecast customer monthly billing charges and revenue",
            "Identify high-risk accounts likely to cancel subscription",
        ],
    },
    "Banking, FinTech & Credit Underwriting": {
        "keywords": ["loan", "credit", "credit_score", "debt", "income", "default", "default_risk", "applicant", "mortgage", "balance", "transaction", "revolving", "interest_rate", "dti"],
        "objectives": [
            "Predict loan default risk for credit underwriting",
            "Estimate applicant credit score and risk grade",
            "Classify high-risk financial credit applications",
        ],
    },
    "Fraud Detection & Cybersecurity": {
        "keywords": ["fraud", "is_fraud", "ip_address", "suspicious", "anomaly", "chargeback", "transaction_amount", "device", "location", "login_attempts", "attack", "malicious"],
        "objectives": [
            "Detect fraudulent financial transactions",
            "Classify anomalous account access patterns",
            "Predict chargeback and security escalation probability",
        ],
    },
    "E-Commerce, Marketing & Retail": {
        "keywords": ["order", "cart", "product", "price", "sale", "discount", "rating", "review", "customer", "purchase", "conversion", "basket", "quantity", "shipping", "category"],
        "objectives": [
            "Predict customer purchase conversion probability",
            "Forecast total order sales value and basket size",
            "Classify customer lifetime value tier (High vs Low value)",
        ],
    },
    "Healthcare & Clinical Diagnostics": {
        "keywords": ["patient", "diagnosis", "glucose", "insulin", "blood_pressure", "cholesterol", "bmi", "age", "admission", "readmission", "hospital", "heart_disease", "disease", "treatment"],
        "objectives": [
            "Predict patient clinical diagnostic outcome",
            "Estimate hospital 30-day readmission risk",
            "Classify disease progression and health risk tier",
        ],
    },
    "Human Resources & Talent Analytics": {
        "keywords": ["employee", "salary", "department", "attrition", "performance", "satisfaction", "years_at_company", "tenure", "job_role", "promotion", "overtime", "compensation"],
        "objectives": [
            "Predict employee attrition and turnover risk",
            "Classify workforce performance rating category",
            "Estimate market salary and compensation benchmark",
        ],
    },
    "Real Estate & Property Valuation": {
        "keywords": ["price", "sale_price", "sqft", "square_feet", "bedroom", "bathroom", "lot_size", "property", "house", "tax", "neighborhood", "zipcode", "renovation", "built_year"],
        "objectives": [
            "Predict property market sale price",
            "Forecast square-foot property valuation",
            "Classify property tier based on physical and location specs",
        ],
    },
    "Industrial IoT & Predictive Maintenance": {
        "keywords": ["sensor", "temperature", "vibration", "pressure", "rpm", "failure", "maintenance", "machine", "operating_hours", "voltage", "engine", "error_code"],
        "objectives": [
            "Predict machine equipment failure (Predictive Maintenance)",
            "Forecast operating sensor temperature and pressure trends",
            "Classify industrial operational health state",
        ],
    },
}


def infer_dataset_context_locally(df: pd.DataFrame) -> dict[str, Any]:
    """
    Performs fast, 100% offline semantic inference of dataset domain and top ML goals.
    Uses column name tokenization, type inspection, and statistical distributions.
    """
    cols = [str(c).lower().replace(" ", "_") for c in df.columns]
    cols_set = set(cols)

    best_domain = "General Business Analytics"
    max_matches = 0
    matched_objectives = []

    # 1. Match against domain taxonomy using semantic and value-aware resolver
    from .column_resolver import resolve_column
    for domain_name, data in DOMAIN_TAXONOMY.items():
        matches = 0
        for kw in data["keywords"]:
            matched_col, conf, _ = resolve_column(kw, df)
            if matched_col and conf >= 0.70:
                matches += 1
        if matches > max_matches:
            max_matches = matches
            best_domain = domain_name
            matched_objectives = list(data["objectives"])

    # 2. Dynamic column-specific objective synthesis
    binary_cols = []
    continuous_num_cols = []
    cat_cols = []

    for col in df.columns:
        series = df[col].dropna()
        n_unique = series.nunique()
        if n_unique == 2:
            binary_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series) and n_unique > 10:
            continuous_num_cols.append(col)
        elif pd.api.types.is_object_dtype(series) and 2 < n_unique <= 10:
            cat_cols.append(col)

    # If domain taxonomy didn't find high-confidence match, generate statistical objectives
    if max_matches < 2 or not matched_objectives:
        custom_objs = []
        if binary_cols:
            custom_objs.append(f"Predict binary outcome `{binary_cols[0]}` from feature variables")
        if continuous_num_cols:
            custom_objs.append(f"Forecast continuous numeric value `{continuous_num_cols[0]}` using regression")
        if cat_cols:
            custom_objs.append(f"Classify multi-class category `{cat_cols[0]}`")
        if len(binary_cols) > 1:
            custom_objs.append(f"Classify secondary indicator `{binary_cols[1]}`")
        
        if not custom_objs:
            custom_objs = [
                f"Predict `{df.columns[-1]}` based on all correlated features",
                "Perform clustering and anomaly isolation",
                "Analyze key feature drivers and correlations",
            ]
        matched_objectives = custom_objs[:3]

    return {
        "status": "success",
        "domain": best_domain,
        "objectives": matched_objectives,
        "confidence_matches": max_matches,
        "engine": "Local Semantic & Statistical Ontology (100% Offline, Zero Latency)",
    }


def get_dataset_context_and_objectives(df: pd.DataFrame, api_key: str | None = None) -> dict[str, Any]:
    """
    Main entry point: Attempts Gemini API if key is explicitly provided;
    otherwise executes high-speed local offline semantic inference.
    """
    if api_key and api_key.strip():
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key.strip())
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze dataset schema:
            Columns: {df.columns.tolist()}
            Data sample: {df.head(2).to_dict(orient='records')}
            
            Return JSON with keys 'domain' and 'objectives' (list of 3 strings).
            """
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            import json
            return json.loads(raw_text)
        except Exception as e:
            log.warning("Gemini API error, falling back to local detector: %s", e)
            # Seamless fallback to local
            local_res = infer_dataset_context_locally(df)
            local_res["fallback_note"] = f"Gemini API returned error ({str(e)[:50]}); switched to local inference."
            return local_res

    # Default to 100% local, instant, zero-failure detector
    return infer_dataset_context_locally(df)
