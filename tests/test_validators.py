"""
tests/test_validators.py
────────────────────────
Unit tests for dia/validators.py — the input validation layer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dia.exceptions import ValidationError
from dia.validators import (
    detect_pii_columns,
    sanitise_column_names,
    validate_dataframe_shape,
    validate_goal_text,
    validate_magic_bytes,
    validate_model_selection,
    validate_sql_identifier,
    validate_target_column,
    validate_uploaded_file,
)
from tests.conftest import MockUploadedFile

# ─── validate_uploaded_file ───────────────────────────────────────────────────

class TestValidateUploadedFile:
    def test_none_raises(self) -> None:
        with pytest.raises(ValidationError, match="No file"):
            validate_uploaded_file(None)

    def test_wrong_extension_raises(self) -> None:
        f = MockUploadedFile(b"data", name="data.xlsx", mime="application/vnd.ms-excel")
        with pytest.raises(ValidationError, match=".csv"):
            validate_uploaded_file(f)

    def test_disallowed_mime_raises(self) -> None:
        f = MockUploadedFile(b"data", name="data.csv", mime="application/zip")
        with pytest.raises(ValidationError, match="file type"):
            validate_uploaded_file(f)

    def test_valid_csv_passes(self) -> None:
        f = MockUploadedFile(b"a,b\n1,2", name="ok.csv", mime="text/csv")
        validate_uploaded_file(f)  # should not raise


# ─── validate_magic_bytes ─────────────────────────────────────────────────────

class TestValidateMagicBytes:
    def test_zip_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ZIP"):
            validate_magic_bytes(b"PK\x03\x04rest of zip")

    def test_pdf_rejected(self) -> None:
        with pytest.raises(ValidationError, match="PDF"):
            validate_magic_bytes(b"%PDFblah")

    def test_elf_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ELF"):
            validate_magic_bytes(b"\x7fELFblah")

    def test_plain_csv_passes(self) -> None:
        validate_magic_bytes(b"col1,col2\nval1,val2")

    def test_utf8_bom_csv_passes(self) -> None:
        validate_magic_bytes(b"\xef\xbb\xbfcol1,col2\n")


# ─── validate_goal_text ───────────────────────────────────────────────────────

class TestValidateGoalText:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_goal_text("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValidationError):
            validate_goal_text("   ")

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValidationError, match="character limit"):
            validate_goal_text("x" * 1001)

    def test_control_chars_stripped(self) -> None:
        cleaned = validate_goal_text("predict\x00churn")
        assert "\x00" not in cleaned

    def test_valid_goal_returned(self) -> None:
        cleaned = validate_goal_text("predict customer churn")
        assert cleaned == "predict customer churn"

    def test_goal_at_max_length_passes(self) -> None:
        validate_goal_text("a" * 1000)  # should not raise


# ─── validate_target_column ───────────────────────────────────────────────────

class TestValidateTargetColumn:
    def test_missing_column_raises(self, churn_df: pd.DataFrame) -> None:
        with pytest.raises(ValidationError, match="not found"):
            validate_target_column("nonexistent", churn_df)

    def test_too_many_nulls_raises(self) -> None:
        df = pd.DataFrame({"target": [None] * 5, "feature": [1, 2, 3, 4, 5]})
        with pytest.raises(ValidationError, match="non-null"):
            validate_target_column("target", df, min_non_null=10)

    def test_valid_column_passes(self, churn_df: pd.DataFrame) -> None:
        validate_target_column("churn_flag", churn_df)


# ─── validate_dataframe_shape ─────────────────────────────────────────────────

class TestValidateDataframeShape:
    def test_empty_df_raises(self) -> None:
        with pytest.raises(ValidationError, match="no rows"):
            validate_dataframe_shape(pd.DataFrame())

    def test_single_column_raises(self) -> None:
        df = pd.DataFrame({"only": [1, 2, 3]})
        with pytest.raises(ValidationError, match="2 column"):
            validate_dataframe_shape(df)

    def test_valid_shape_passes(self, churn_df: pd.DataFrame) -> None:
        validate_dataframe_shape(churn_df)

    def test_too_many_rows_raises(self) -> None:
        from dia.config import MAX_ROWS

        df = pd.DataFrame({"a": range(MAX_ROWS + 1), "b": range(MAX_ROWS + 1)})
        with pytest.raises(ValidationError, match="row limit"):
            validate_dataframe_shape(df)

    def test_too_many_columns_raises(self) -> None:
        from dia.config import MAX_COLS

        df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(MAX_COLS + 1)})
        with pytest.raises(ValidationError, match="column limit"):
            validate_dataframe_shape(df)


# ─── validate_sql_identifier ───────────────────────────────────────────────────

class TestValidateSqlIdentifier:
    def test_bare_identifier_passes(self) -> None:
        assert validate_sql_identifier("customers") == "customers"

    def test_schema_qualified_identifier_passes(self) -> None:
        assert validate_sql_identifier("public.customers") == "public.customers"

    def test_strips_whitespace(self) -> None:
        assert validate_sql_identifier("  customers  ") == "customers"

    def test_rejects_injection_attempt(self) -> None:
        with pytest.raises(ValidationError, match="Invalid table name"):
            validate_sql_identifier("customers; DROP TABLE users;--")

    def test_rejects_quoted_identifier(self) -> None:
        with pytest.raises(ValidationError):
            validate_sql_identifier('"customers"')

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            validate_sql_identifier("")

    def test_custom_field_label_appears_in_message(self) -> None:
        with pytest.raises(ValidationError, match="Invalid schema name"):
            validate_sql_identifier("bad;name", field_label="schema name")


# ─── validate_model_selection ─────────────────────────────────────────────────

class TestValidateModelSelection:
    def test_no_valid_keys_raises(self) -> None:
        with pytest.raises(ValidationError, match="No valid"):
            validate_model_selection(["bad_key"], {"logreg": {}})

    def test_caps_at_max_models(self) -> None:
        registry = {str(i): {} for i in range(10)}
        result = validate_model_selection([str(i) for i in range(10)], registry)
        assert len(result) <= 4

    def test_filters_invalid_keys(self) -> None:
        registry = {"logreg": {}, "rf": {}}
        result = validate_model_selection(["logreg", "invalid", "rf"], registry)
        assert result == ["logreg", "rf"]


# ─── sanitise_column_names ────────────────────────────────────────────────────

class TestSanitiseColumnNames:
    def test_strips_null_bytes(self) -> None:
        result = sanitise_column_names(["col\x00name"])
        assert result == ["colname"]

    def test_replaces_empty_with_unnamed(self) -> None:
        result = sanitise_column_names(["", "  "])
        assert result[0].startswith("unnamed_")
        assert result[1].startswith("unnamed_")

    def test_valid_names_unchanged(self) -> None:
        names = ["customer_id", "tenure_months", "churn_flag"]
        assert sanitise_column_names(names) == names


# ─── detect_pii_columns ───────────────────────────────────────────────────────

class TestDetectPIIColumns:
    def test_detects_email(self) -> None:
        assert "email_address" in detect_pii_columns(["email_address", "revenue"])

    def test_detects_ssn(self) -> None:
        assert "ssn" in detect_pii_columns(["ssn", "age"])

    def test_clean_columns_not_flagged(self) -> None:
        result = detect_pii_columns(["tenure_months", "monthly_charges", "churn_flag"])
        assert result == []
