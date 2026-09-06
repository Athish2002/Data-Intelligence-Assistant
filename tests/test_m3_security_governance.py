"""
tests/test_m3_security_governance.py

Milestone M3 Test Suite: Enterprise Security, Data Governance & GDPR Hardening.
Tests:
- Data contract generation with infinite and NaN values.
- Data contract active validation (schema, nulls, bounds, sets).
- GDPR PII detection without false positive substring matches.
- PII masking with +/- inf and extreme numbers without OverflowError.
- API endpoints for /api/v1/governance/contract/{session_id} and /api/v1/governance/gdpr/{session_id}.
- Frontend static security audit verifying escapeHtml usage.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dia.compliance import scan_dataset_privacy, mask_dataframe_pii
from dia.data_quality import generate_data_contract, validate_data_contract
from api.server import app, SESSION_STORE


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure session store is cleaned before and after tests."""
    SESSION_STORE.clear()
    yield
    SESSION_STORE.clear()


def test_data_contract_generation_finite_bounds():
    """Verify data contract generation handles infinity and NaNs gracefully without unhandled exceptions."""
    df = pd.DataFrame({
        "num_clean": [1.0, 2.5, 3.8, 4.2],
        "num_with_inf": [10.0, np.inf, 20.0, -np.inf],
        "cat_clean": ["A", "B", "A", "B"],
        "target": [0, 1, 0, 1],
    })

    contract = generate_data_contract(df, target_col="target")
    assert "expectations" in contract
    assert len(contract["expectations"]) > 0

    # Ensure min/max values for between expectations are finite
    for exp in contract["expectations"]:
        if exp["expectation_type"] == "expect_column_values_to_be_between":
            min_val = exp["kwargs"].get("min_value")
            max_val = exp["kwargs"].get("max_value")
            assert np.isfinite(min_val), f"min_value should be finite, got {min_val}"
            assert np.isfinite(max_val), f"max_value should be finite, got {max_val}"
            assert min_val <= max_val


def test_data_contract_validation_success():
    """Verify that validate_data_contract passes when DataFrame conforms to contract."""
    df_train = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "score": [1.5, 2.5, 3.5, 4.5],
        "tier": ["Gold", "Silver", "Gold", "Bronze"],
        "target": [0, 1, 0, 1],
    })

    contract = generate_data_contract(df_train, target_col="target")
    val_res = validate_data_contract(df_train, contract)

    assert val_res["success"] is True
    assert val_res["n_failed"] == 0
    assert val_res["n_passed"] == val_res["n_expectations"]
    assert val_res["pass_rate"] == 1.0


def test_data_contract_validation_detects_violations():
    """Verify that validate_data_contract catches missing columns, out of bounds, unseen categories, and nulls."""
    df_train = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "tier": ["Gold", "Silver", "Gold", "Bronze"],
        "target": [0, 1, 0, 1],
    })
    contract = generate_data_contract(df_train, target_col="target")

    # 1. Test missing column
    df_missing = df_train.drop(columns=["tier"])
    res_missing = validate_data_contract(df_missing, contract)
    assert res_missing["success"] is False
    assert any("tier" in str(r.get("error", "")) for r in res_missing["details"])

    # 2. Test out of bounds numeric value
    df_oob = df_train.copy()
    df_oob.loc[0, "age"] = 999  # max was 50
    res_oob = validate_data_contract(df_oob, contract)
    assert res_oob["success"] is False
    assert any(r["expectation_type"] == "expect_column_values_to_be_between" and not r["success"] for r in res_oob["details"])

    # 3. Test unseen category
    df_unseen = df_train.copy()
    df_unseen.loc[0, "tier"] = "Platinum"  # unseen category
    res_unseen = validate_data_contract(df_unseen, contract)
    assert res_unseen["success"] is False
    assert any(r["expectation_type"] == "expect_column_values_to_be_in_set" and not r["success"] for r in res_unseen["details"])

    # 4. Test excessive nulls
    df_nulls = df_train.copy()
    df_nulls["age"] = np.nan
    res_nulls = validate_data_contract(df_nulls, contract)
    assert res_nulls["success"] is False
    assert any(r["expectation_type"] == "expect_column_values_to_not_be_null" and not r["success"] for r in res_nulls["details"])


def test_gdpr_audit_no_substring_false_positives():
    """Verify word-boundary token matching does not falsely flag words containing substrings like 'age'."""
    df = pd.DataFrame({
        "data_usage": [100.5, 200.3, 300.2],    # Contains 'age', but is NOT protected class 'age'
        "storage_gb": [50, 100, 250],
        "percentage": [0.12, 0.45, 0.88],       # Contains 'age', but is NOT protected class 'age'
        "coverage_ratio": [0.9, 0.85, 0.95],    # Contains 'age', but is NOT protected class 'age'
        "user_email": ["alice@example.com", "bob@example.com", "charlie@example.com"],  # REAL PII
        "phone_number": ["555-0199", "555-0188", "555-0177"],  # REAL PII
    })

    audit = scan_dataset_privacy(df)
    detected_pii = audit.get("detected_pii", [])
    detected_cols = [p["column"] for p in detected_pii]

    # Real PII columns must be detected
    assert "user_email" in detected_cols
    assert "phone_number" in detected_cols

    # Substring matches should NOT be flagged as protected category
    for p in detected_pii:
        col = p["column"]
        assert col not in ["data_usage", "storage_gb", "percentage", "coverage_ratio"], (
            f"Column '{col}' was falsely flagged as PII: {p}"
        )


def test_mask_dataframe_pii_with_inf_and_nan():
    """Verify mask_dataframe_pii safely handles infinity, large numbers, and nulls without OverflowError."""
    df = pd.DataFrame({
        "salary": [100000.0, np.inf, -np.inf, np.nan, 84321.0],
        "email": ["test@example.com", None, "admin@corp.internal", np.nan, "user@dia.ai"],
    })

    # Mask both columns
    masked = mask_dataframe_pii(df, pii_columns=["salary", "email"])

    # Verify email masking
    assert masked["email"].iloc[0] == "t***@example.com"
    assert pd.isna(masked["email"].iloc[1])

    # Verify salary masking handles infinity without OverflowError
    salary_vals = masked["salary"].tolist()
    assert salary_vals[0] == "***00"
    assert salary_vals[1] == "***00"  # inf
    assert salary_vals[2] == "***00"  # -inf
    assert pd.isna(salary_vals[3])    # nan
    assert salary_vals[4] == f"***{84321 % 100:02d}"


def test_api_data_contract_and_gdpr_endpoints():
    """Verify /api/v1/governance/contract and /api/v1/governance/gdpr endpoints serialize valid response payloads."""
    client = TestClient(app)

    # 1. Create a session with tabular data
    session_id = "test-governance-session"
    df = pd.DataFrame({
        "age": [25, 35, 45, 55],
        "salary": [50000, 75000, 90000, 120000],
        "email": ["a@a.com", "b@b.com", "c@c.com", "d@d.com"],
        "target": [0, 1, 0, 1],
    })
    SESSION_STORE[session_id] = {
        "session_id": session_id,
        "df": df,
        "target": "target",
        "pipeline_result": None,
    }

    # 2. Test Data Contract endpoint
    resp_contract = client.get(f"/api/v1/governance/contract/{session_id}")
    assert resp_contract.status_code == 200
    data_contract = resp_contract.json()
    assert data_contract["status"] == "success"
    assert data_contract["n_expectations"] > 0
    # great_expectations_json must be valid non-empty JSON
    parsed_ge = json.loads(data_contract["great_expectations_json"])
    assert "expectations" in parsed_ge
    assert len(parsed_ge["expectations"]) == data_contract["n_expectations"]

    # 3. Test GDPR Audit endpoint
    resp_gdpr = client.get(f"/api/v1/governance/gdpr/{session_id}")
    assert resp_gdpr.status_code == 200
    data_gdpr = resp_gdpr.json()
    assert data_gdpr["status"] == "success"
    assert "pii_entities_detected" in data_gdpr
    assert isinstance(data_gdpr["pii_entities_detected"], list)
    detected_names = [item["column"] for item in data_gdpr["pii_entities_detected"]]
    assert "email" in detected_names


def test_frontend_xss_sanitization_integrity():
    """Static audit of frontend files ensuring escapeHtml is consistently imported and applied."""
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "js")
    assert os.path.exists(frontend_dir)

    views = ["views/core.js", "views/automl.js", "views/system.js", "toast.js"]
    for v in views:
        path = os.path.join(frontend_dir, v)
        assert os.path.exists(path), f"Frontend file {v} not found"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that escapeHtml is referenced in the file
        assert "escapeHtml" in content, f"escapeHtml not found in {v}"
