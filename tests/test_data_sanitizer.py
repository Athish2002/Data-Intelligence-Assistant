"""
tests/test_data_sanitizer.py
────────────────────────────
Unit tests for the Automated Data Sanitizer & Malformed Data Repair Engine.
"""

import numpy as np
import pandas as pd

from dia.data_sanitizer import (
    format_sanitization_report_markdown,
    robust_parse_csv_bytes,
    sanitize_dataframe,
)


def test_sanitize_dirty_numerics_and_currencies():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "price_usd": ["$1,200.50", "$3,400.00", "$500.25", "$10k"],
        "discount_pct": ["10%", "15.5%", "0%", "25%"],
        "revenue_mil": ["1.2M", "0.5M", "3.0M", "10M"],
    })

    clean_df, report = sanitize_dataframe(df)

    assert pd.api.types.is_numeric_dtype(clean_df["price_usd"])
    assert clean_df["price_usd"].iloc[0] == 1200.50
    assert clean_df["price_usd"].iloc[3] == 10000.0

    assert pd.api.types.is_numeric_dtype(clean_df["discount_pct"])
    assert clean_df["discount_pct"].iloc[1] == 15.5

    assert pd.api.types.is_numeric_dtype(clean_df["revenue_mil"])
    assert clean_df["revenue_mil"].iloc[0] == 1200000.0
    assert clean_df["revenue_mil"].iloc[3] == 10000000.0
    assert report["total_cells_repaired"] > 0


def test_sanitize_dirty_nulls_and_booleans():
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "status": ["active", "inactive", "active", "N/A", "--"],
        "opt_in": ["yes", "no", "yes", "no", "yes"],
        "churn": ["true", "false", "true", "false", "true"],
        "score": [10.5, np.inf, 20.0, -np.inf, 15.0],
    })

    clean_df, report = sanitize_dataframe(df)

    # Check null standardization
    assert clean_df["status"].isna().sum() == 2

    # Check boolean normalization
    assert set(clean_df["opt_in"].unique()) == {0, 1}
    assert set(clean_df["churn"].unique()) == {0, 1}

    # Check Inf replacement
    assert clean_df["score"].isna().sum() == 2
    assert report["inf_replaced_count"] == 2


def test_robust_parse_csv_bytes_semicolon_and_special_chars():
    # Semicolon-delimited, special chars in column headers, dirty numbers
    raw_csv = """  Customer ID  ; Annual Salary ($) ; Active Flag ; Missing Col 
CUST-001 ; $85,000.00 ; yes ; normal
CUST-002 ; $120,500.50 ; no ; ?
CUST-003 ; $45,000 ; yes ; N/A
""".encode("latin-1")

    clean_df, report = robust_parse_csv_bytes(raw_csv)

    assert clean_df.shape[0] == 3
    assert "Customer_ID" in clean_df.columns or "Annual_Salary" in clean_df.columns[1]
    assert report["detected_delimiter"] == ";"
    assert report["total_cells_repaired"] > 0


def test_format_sanitization_report_markdown():
    report = {
        "total_cells_repaired": 1240,
        "detected_encoding": "utf-8",
        "detected_delimiter": ",",
        "renamed_columns": {" Old Header ($) ": "Old_Header"},
        "coerced_numeric_columns": ["salary", "discount"],
        "boolean_converted_columns": ["is_active"],
        "imputed_null_counts": {"status": 15},
        "inf_replaced_count": 2,
    }
    md = format_sanitization_report_markdown(report)
    assert "Automated Data Sanitization" in md
    assert "1,240" in md
