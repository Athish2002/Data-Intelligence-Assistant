"""
tests/test_column_resolver.py
─────────────────────────────
Unit tests for the Semantic & Value-Aware Column Resolver Engine.
"""

import pandas as pd
from dia.column_resolver import resolve_column, rank_target_candidates_advanced, normalize_string


def test_normalize_string():
    assert normalize_string("SalePrice") == "sale_price"
    assert normalize_string("annual_salary ($)") == "annual_salary"
    assert normalize_string("  Customer ID  ") == "customer_id"


def test_resolve_casing_and_fuzzy_typos():
    columns = ["customer_id", "sale_price", "annual_income", "default_risk"]
    
    # CamelCase query
    col, conf, reason = resolve_column("SalePrice", columns)
    assert col == "sale_price"
    assert conf >= 0.90
    
    # Typo query
    col, conf, reason = resolve_column("defualt", columns)
    assert col == "default_risk"
    assert conf >= 0.60
    
    # Suffix query
    col, conf, reason = resolve_column("income", columns)
    assert col == "annual_income"
    assert conf >= 0.85


def test_resolve_semantic_synonyms():
    columns = ["client_id", "tenure", "exited", "gross_earnings"]
    
    # 'churn' should resolve to 'exited'
    col, conf, reason = resolve_column("churn", columns)
    assert col == "exited"
    assert conf >= 0.80
    assert "synonym" in reason.lower()
    
    # 'salary' should resolve to 'gross_earnings'
    col, conf, reason = resolve_column("salary", ["user_id", "gross_earnings"])
    assert col == "gross_earnings" or "gross_earnings" in str(col)


def test_resolve_by_column_values():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "amount": [10.5, 20.0, 50.0, 100.0],
        "outcome_type": ["legitimate", "fraud", "legitimate", "fraud"],
    })
    
    # Query 'fraud' should find column 'outcome_type' by inspecting cell values
    col, conf, reason = resolve_column("fraud", df)
    assert col == "outcome_type"
    assert conf >= 0.80
    assert "value" in reason.lower() or "synonym" in reason.lower()


def test_rank_target_candidates_advanced():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "age": [25, 30, 45, 50],
        "is_churn": [0, 1, 0, 1],
    })
    
    ranked = rank_target_candidates_advanced("Predict customer churn", df)
    assert len(ranked) > 0
    assert ranked[0]["column"] == "is_churn"
    assert ranked[0]["confidence"] >= 0.80
