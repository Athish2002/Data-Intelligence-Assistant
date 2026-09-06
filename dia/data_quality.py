"""
dia/data_quality.py
───────────────────
Generates a 'Data Contract' (Expectation Suite) based on the training dataframe.
This mimics what Great Expectations does to prevent data drift in production.
"""
import json
from typing import Any

import numpy as np
import pandas as pd


def generate_data_contract(df: pd.DataFrame, target_col: str) -> dict:
    """
    Generate a JSON-serializable dictionary of data quality expectations
    based on the observed properties of the training dataframe.
    """
    expectations = []

    # 1. Table-level expectations
    expectations.append({
        "expectation_type": "expect_table_columns_to_match_set",
        "kwargs": {
            "column_set": list(df.columns)
        }
    })

    # 2. Target expectations
    expectations.append({
        "expectation_type": "expect_column_to_exist",
        "kwargs": {"column": target_col}
    })
    expectations.append({
        "expectation_type": "expect_column_values_to_not_be_null",
        "kwargs": {"column": target_col}
    })

    # 3. Feature-level expectations
    for col in df.columns:
        if col == target_col:
            continue

        dtype_str = str(df[col].dtype)

        # Null expectations
        null_pct = df[col].isnull().sum() / len(df)
        if null_pct == 0:
            expectations.append({
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": col}
            })
        elif null_pct < 1.0:
            # Great Expectations 'mostly' parameter specifies minimum proportion of non-null values
            non_null_rate = max(0.01, round(1.0 - null_pct - 0.05, 2))
            expectations.append({
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": col, "mostly": non_null_rate}
            })

        # Numeric bounds
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_series = df[col].dropna()
            if not valid_series.empty:
                finite_series = valid_series[np.isfinite(valid_series)]
                if not finite_series.empty:
                    min_val = float(finite_series.min())
                    max_val = float(finite_series.max())
                    expectations.append({
                        "expectation_type": "expect_column_values_to_be_between",
                        "kwargs": {
                            "column": col,
                            "min_value": min_val,
                            "max_value": max_val
                        }
                    })
            expectations.append({
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": col, "type_": dtype_str}
            })

        # Categorical sets (if low cardinality)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            n_unique = df[col].nunique()
            if n_unique < 20:  # low cardinality
                unique_vals = [str(x) for x in df[col].dropna().unique()]
                expectations.append({
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "column": col,
                        "value_set": unique_vals
                    }
                })
            expectations.append({
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": col, "type_": dtype_str}
            })

    contract = {
        "meta": {
            "great_expectations_version": "0.15.0",
            "notes": "Auto-generated Data Contract for Inference Pipeline",
        },
        "expectations": expectations
    }
    return contract

def format_contract_markdown(contract: dict) -> str:
    """Format the contract dictionary as readable markdown."""
    md = "### 🛡️ Generated Data Contract (Expectations)\n\n"
    md += "This contract defines the strict rules that production data must pass before model inference. If these rules are violated, an alert should be fired to prevent silent failures.\n\n"

    table_rules = [e for e in contract["expectations"] if e["expectation_type"].startswith("expect_table")]
    col_rules = [e for e in contract["expectations"] if e["expectation_type"].startswith("expect_column")]

    md += "#### Table-Level Constraints\n"
    for r in table_rules:
        if r["expectation_type"] == "expect_table_columns_to_match_set":
            md += f"- **Required Schema:** Must contain exactly {len(r['kwargs']['column_set'])} columns.\n"

    md += "\n#### Column-Level Constraints\n"

    # Group by column
    col_map = {}
    for r in col_rules:
        c = r["kwargs"].get("column")
        if c:
            if c not in col_map:
                col_map[c] = []
            col_map[c].append(r)

    for col, rules in col_map.items():
        md += f"- **`{col}`**:\n"
        for r in rules:
            rtype = r["expectation_type"]
            if rtype == "expect_column_to_exist":
                pass # Redundant with table
            elif rtype == "expect_column_values_to_not_be_null":
                md += "  - 🚫 Must not contain Nulls.\n"
            elif rtype == "expect_column_values_to_be_between":
                md += f"  - 📏 Values must be between **{r['kwargs']['min_value']:.2f}** and **{r['kwargs']['max_value']:.2f}**.\n"
            elif rtype == "expect_column_values_to_be_in_set":
                vals = r['kwargs']['value_set']
                show_vals = ", ".join(vals[:5]) + ("..." if len(vals) > 5 else "")
                md += f"  - 🗂️ Must be one of: `[{show_vals}]`\n"
            elif rtype == "expect_column_values_to_be_of_type":
                md += f"  - 🔠 Type must be `{r['kwargs']['type_']}`.\n"

    return md


def validate_data_contract(df: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    """
    Actively validates an incoming DataFrame against a Great Expectations-style contract dictionary.
    Returns validation summary and detailed rule-by-rule results.
    """
    expectations = contract.get("expectations", [])
    results: list[dict[str, Any]] = []
    n_passed = 0

    for exp in expectations:
        etype = exp.get("expectation_type", "")
        kwargs = exp.get("kwargs", {})
        col = kwargs.get("column")
        passed = True
        error_msg = None

        try:
            if etype == "expect_table_columns_to_match_set":
                expected_set = set(kwargs.get("column_set", []))
                actual_set = set(df.columns)
                if expected_set != actual_set:
                    passed = False
                    missing = expected_set - actual_set
                    extra = actual_set - expected_set
                    error_msg = f"Schema mismatch. Missing: {list(missing)[:5]}, Extra: {list(extra)[:5]}"

            elif etype == "expect_column_to_exist":
                if col not in df.columns:
                    passed = False
                    error_msg = f"Column '{col}' does not exist in DataFrame"

            elif etype == "expect_column_values_to_not_be_null":
                if col not in df.columns:
                    passed = False
                    error_msg = f"Column '{col}' missing"
                else:
                    mostly = float(kwargs.get("mostly", 1.0))
                    non_null_rate = float(df[col].notna().mean()) if len(df) > 0 else 1.0
                    if non_null_rate < mostly - 1e-5:
                        passed = False
                        error_msg = f"Observed non-null rate {non_null_rate:.3f} < required mostly {mostly:.3f}"

            elif etype == "expect_column_values_to_be_between":
                if col not in df.columns:
                    passed = False
                    error_msg = f"Column '{col}' missing"
                else:
                    valid = df[col].dropna()
                    if not valid.empty:
                        min_req = float(kwargs.get("min_value", -float("inf")))
                        max_req = float(kwargs.get("max_value", float("inf")))
                        finite_valid = valid[np.isfinite(valid)]
                        if not finite_valid.empty:
                            obs_min = float(finite_valid.min())
                            obs_max = float(finite_valid.max())
                            if obs_min < min_req - 1e-4 or obs_max > max_req + 1e-4:
                                passed = False
                                error_msg = f"Observed range [{obs_min:.2f}, {obs_max:.2f}] outside [{min_req:.2f}, {max_req:.2f}]"

            elif etype == "expect_column_values_to_be_in_set":
                if col not in df.columns:
                    passed = False
                    error_msg = f"Column '{col}' missing"
                else:
                    valid = df[col].dropna().astype(str)
                    allowed = set(kwargs.get("value_set", []))
                    unseen = set(valid.unique()) - allowed
                    if unseen:
                        passed = False
                        error_msg = f"Encountered unexpected categories: {list(unseen)[:5]}"

            elif etype == "expect_column_values_to_be_of_type":
                if col not in df.columns:
                    passed = False
                    error_msg = f"Column '{col}' missing"

        except Exception as err:
            passed = False
            error_msg = f"Evaluation exception: {str(err)}"

        if passed:
            n_passed += 1

        results.append({
            "expectation_type": etype,
            "column": col or "table",
            "success": passed,
            "error": error_msg,
        })

    n_total = len(expectations)
    n_failed = n_total - n_passed
    all_success = (n_failed == 0)

    return {
        "success": all_success,
        "n_expectations": n_total,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "pass_rate": round(n_passed / max(1, n_total), 4),
        "details": results,
    }
