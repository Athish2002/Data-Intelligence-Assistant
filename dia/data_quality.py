"""
dia/data_quality.py
───────────────────
Generates a 'Data Contract' (Expectation Suite) based on the training dataframe.
This mimics what Great Expectations does to prevent data drift in production.
"""
import pandas as pd
import json

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
                min_val = float(valid_series.min())
                max_val = float(valid_series.max())
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
                "kwargs": {"column": col, "type_": "number"}
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
                "kwargs": {"column": col, "type_": "string"}
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
                md += f"  - 🚫 Must not contain Nulls.\n"
            elif rtype == "expect_column_values_to_be_between":
                md += f"  - 📏 Values must be between **{r['kwargs']['min_value']:.2f}** and **{r['kwargs']['max_value']:.2f}**.\n"
            elif rtype == "expect_column_values_to_be_in_set":
                vals = r['kwargs']['value_set']
                show_vals = ", ".join(vals[:5]) + ("..." if len(vals) > 5 else "")
                md += f"  - 🗂️ Must be one of: `[{show_vals}]`\n"
            elif rtype == "expect_column_values_to_be_of_type":
                md += f"  - 🔠 Type must be `{r['kwargs']['type_']}`.\n"
                
    md += "\n\n```json\n" + json.dumps(contract, indent=2) + "\n```"
    return md
