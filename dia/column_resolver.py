"""
dia/column_resolver.py
──────────────────────
Autonomous Semantic & Value-Aware Column Resolver Engine.
Resolves columns when:
1. Exact name is misspelled or has typographical differences (fuzzy matching).
2. Different naming conventions/casing are used (camelCase vs snake_case).
3. Semantic domain synonyms are used (e.g. 'churn' -> 'exited', 'salary' -> 'annual_income', 'price' -> 'sale_price').
4. The column name is generic (e.g. 'class', 'target', 'status'), but the column values contain the concept (e.g. ['fraud', 'legit']).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

# Comprehensive semantic synonym dictionary for tabular ML
_SYNONYM_CLUSTERS = {
    "churn": [
        "churn", "churn_flag", "is_churn", "has_churned", "exited", "exit",
        "attrition", "leave", "left", "cancelled", "canceled", "terminated",
        "unsubscribed", "retention", "retained", "churn_risk", "churn_label"
    ],
    "default": [
        "default", "default_risk", "default_flag", "is_default", "bad_loan",
        "delinquency", "delinquent", "chargeoff", "charge_off", "credit_risk",
        "non_payment", "overdue", "loan_status", "risk_flag"
    ],
    "fraud": [
        "fraud", "is_fraud", "fraud_flag", "fraudulent", "anomaly", "suspicious",
        "scam", "chargeback", "bad_actor", "illicit", "class", "target_class"
    ],
    "price": [
        "price", "sale_price", "saleprice", "cost", "amount", "value",
        "valuation", "fare", "fee", "charge", "rent", "listing_price",
        "market_price", "total_price", "unit_price"
    ],
    "revenue": [
        "revenue", "sales", "gross_sales", "income", "turnover", "total_revenue",
        "arr", "mrr", "spend", "total_spend", "lifetime_spend", "clv", "ltv"
    ],
    "income": [
        "income", "annual_income", "salary", "wage", "earnings", "compensation",
        "gross_income", "net_income", "pay", "monthly_income"
    ],
    "customer": [
        "customer", "customer_id", "client", "client_id", "user", "user_id",
        "member", "member_id", "account", "account_id", "subscriber", "subscriber_id",
        "cust_id", "cust_no", "applicant_id"
    ],
    "target": [
        "target", "label", "class", "y", "output", "result", "outcome",
        "response", "ground_truth", "status", "flag"
    ],
    "survival": [
        "survived", "survival", "died", "mortality", "vital_status", "is_alive"
    ],
    "satisfaction": [
        "satisfaction", "rating", "score", "nps", "sentiment", "feedback_score",
        "review_score", "csat"
    ],
    "conversion": [
        "conversion", "converted", "is_converted", "purchase", "purchased",
        "is_purchased", "bought", "buyer", "ordered", "is_ordered", "buy",
        "conversion_flag", "purchase_flag", "buy_flag", "order_flag", "target_flag", "success"
    ],
}


def normalize_string(s: str) -> str:
    """Normalizes string by converting camelCase/spaces to snake_case and lowercasing."""
    s_clean = str(s).strip()
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s_clean)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    s3 = re.sub(r"[^\w\s]", "", s2)
    return re.sub(r"[\s_]+", "_", s3).strip("_")


def resolve_column(
    query: str,
    df_or_columns: pd.DataFrame | Sequence[str],
    similarity_threshold: float = 0.60,
) -> tuple[str | None, float, str]:
    """
    Finds the best matching column in a DataFrame or column list.

    Returns
    -------
    (matched_column_name, confidence_score, resolution_reason)
    """
    if isinstance(df_or_columns, pd.DataFrame):
        df = df_or_columns
        columns = list(df.columns)
    else:
        df = None
        columns = list(df_or_columns)

    if not columns:
        return None, 0.0, "No columns available"

    query_str = str(query).strip()
    query_norm = normalize_string(query_str)

    # ── 1. Exact Match ────────────────────────────────────────────────────────
    for col in columns:
        if col == query_str:
            return col, 1.0, "Exact match"

    # ── 2. Case-Insensitive & Normalized Match ─────────────────────────────────
    for col in columns:
        if normalize_string(col) == query_norm:
            return col, 0.98, f"Normalized casing match ('{query_str}' ➔ '{col}')"

    # ── 3. Substring / Token Inclusion Match ──────────────────────────────────
    for col in columns:
        col_norm = normalize_string(col)
        if query_norm in col_norm or col_norm in query_norm:
            return col, 0.90, f"Substring match ('{query_str}' in '{col}')"

    # ── 4. Domain Synonym Matching ────────────────────────────────────────────
    # Find which synonym cluster the query belongs to
    matched_cluster = None
    for cluster_name, synonyms in _SYNONYM_CLUSTERS.items():
        norm_synonyms = [normalize_string(s) for s in synonyms]
        if query_norm in norm_synonyms or cluster_name == query_norm:
            matched_cluster = synonyms
            break

    if matched_cluster:
        for col in columns:
            col_norm = normalize_string(col)
            for syn in matched_cluster:
                if normalize_string(syn) in col_norm or col_norm in normalize_string(syn):
                    return col, 0.88, f"Semantic domain synonym ('{query_str}' ➔ '{col}')"

    # ── 5. Value-Based Content Inspection ─────────────────────────────────────
    if df is not None:
        val_col, val_score, val_reason = _resolve_by_column_values(query_str, df)
        if val_col and val_score >= 0.80:
            return val_col, val_score, val_reason

    # ── 6. Fuzzy String Similarity Match ──────────────────────────────────────
    best_col = None
    best_ratio = 0.0
    for col in columns:
        col_norm = normalize_string(col)
        ratio = difflib.SequenceMatcher(None, query_norm, col_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    if best_ratio >= similarity_threshold and best_col:
        return best_col, round(best_ratio, 2), f"Fuzzy typographical match (similarity: {best_ratio:.0%})"

    return None, 0.0, f"No matching column found for '{query_str}'"


def _resolve_by_column_values(query: str, df: pd.DataFrame) -> tuple[str | None, float, str]:
    """
    Inspects categorical column values to see if they contain the query concept.
    For example: query='fraud' and column 'class' has values ['fraud', 'legit'].
    """
    query_lower = query.lower().strip()
    query_tokens = set(re.findall(r"[a-z]+", query_lower))

    for col in df.columns:
        series = df[col]
        # Only inspect categorical/low-cardinality columns
        if series.nunique() <= 50:
            val_samples = [str(v).lower().strip() for v in series.dropna().unique()]

            # Check if query tokens appear inside the column values
            for token in query_tokens:
                if len(token) >= 3 and any(token in val for val in val_samples):
                    return col, 0.85, f"Value content match (column '{col}' contains '{token}')"

    return None, 0.0, ""


def rank_target_candidates_advanced(
    goal: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Ranks all columns in the DataFrame as potential prediction targets using
    hybrid NLP tokens, semantic synonyms, fuzzy matching, and column value inspection.
    """
    goal_lower = goal.lower()
    goal_tokens = re.findall(r"[a-z]+", goal_lower)

    candidates: list[dict[str, Any]] = []

    for col in df.columns:
        score = 0.0
        reasons: list[str] = []
        col_norm = normalize_string(col)

        # 1. Check direct query match against goal tokens
        matching_token_count = 0
        for token in goal_tokens:
            if len(token) < 3:
                continue
            matched_col, conf, reason = resolve_column(token, [col])
            if matched_col and conf > 0.6:
                score = max(score, conf)
                matching_token_count += 1
                reasons.append(reason)

        if matching_token_count > 1:
            score += 0.05 * (matching_token_count - 1)
            reasons.append(f"Multiple goal keywords match column ({matching_token_count} matches)")

        # 2. Check if column is a known target indicator name
        if any(term in col_norm for term in ["target", "label", "class", "output", "flag", "status", "risk", "outcome", "churn", "default"]):
            score = max(score, 0.70)
            score += 0.06
            reasons.append(f"Target indicator column name ('{col}')")

        # 3. Check value-based match and binary/categorical preference for outcome goals
        if df[col].nunique() <= 30:
            val_samples = [str(v).lower() for v in df[col].dropna().unique()[:10]]
            for token in goal_tokens:
                if len(token) >= 3 and any(token in v for v in val_samples):
                    score = max(score, 0.85)
                    reasons.append(f"Contains target values matching '{token}'")

        # Prioritize binary flags (e.g. default_risk, churn_flag, is_fraud) when goal is outcome prediction
        if df[col].nunique() == 2 and any(term in goal_lower for term in ["predict", "churn", "default", "risk", "fraud", "conversion", "outcome"]):
            score += 0.08
            reasons.append("Binary outcome variable matching predictive goal")

        # 4. Heavy penalty for Primary Key / ID columns
        if col_norm.endswith("_id") or col_norm in ["id", "client_id", "user_id", "customer_id", "account_id", "cust_id", "cust_no"]:
            score *= 0.20

        if score > 0:
            candidates.append({
                "column": col,
                "confidence": round(score, 2),
                "reasons": reasons,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
            })

    # Sort descending by confidence score
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Fallback to last column if no matches
    if not candidates and len(df.columns) > 0:
        candidates.append({
            "column": df.columns[-1],
            "confidence": 0.50,
            "reasons": ["Default fallback to last dataset column"],
            "dtype": str(df[df.columns[-1]].dtype),
            "n_unique": int(df[df.columns[-1]].nunique()),
        })

    return candidates
