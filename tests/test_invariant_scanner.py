"""
tests/test_invariant_scanner.py
───────────────────────────────
Rigorous Generative Property Fuzzing and End-to-End API Output Verification
for the Algebraic "Suggest & Lock" Invariant Scanner.

Enforces:
1. Generative Property Fuzzing (/property-fuzz-testing):
   - Recovers exact linear dependencies (X3 = 2.0*X1 - 1.5*X2) across 10 random seeds with zero error.
   - 100% false-positive rejection on pure Gaussian noise (500x6 independent random normals).
   - Disentangles multiple simultaneous linear balance laws without cross-contamination.
   - Prunes single-variable, constant, and all-zero trivial candidates.
2. End-to-End API Output Verification (/e2e-output-verification-testing):
   - Ingest demo dataset -> POST /api/v1/invariants/scan/{session_id} -> validates Pydantic V2 schema.
   - Locks candidate conservation laws into session store via POST /api/v1/invariants/lock/{session_id}.
   - Queries GET /api/v1/invariants/{session_id} verifying locked state and axiomatic audit compliance.
   - Direct assertion of exact state persistence in SESSION_STORE.
"""

from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.schemas import (
    AxiomAuditSummaryResponse,
    ConservationLawItem,
    InvariantLockRequest,
    InvariantLockResponse,
    InvariantScanRequest,
    InvariantScanResponse,
    InvariantsStatusResponse,
)
from api.server import SESSION_STORE, app
from dia.invariant_engine import (
    ConservationLaw,
    audit_dataframe_axioms,
    scan_dataframe_conservation_laws,
)


# ───────────────────────────────────────────────────────────────────────────
# 1. Generative Property Fuzzing Suite
# ───────────────────────────────────────────────────────────────────────────

class TestGenerativePropertyFuzzing:
    """Property-based generative fuzz testing for algebraic invariant detection."""

    def test_linear_conservation_recovery_across_10_seeds(self):
        """
        Synthesizes random datasets with 500 rows and 5 features where
        feature 3 is constructed as X3 = 2.0*X1 - 1.5*X2.
        Fuzzes across 10 random seeds.
        Asserts the scanner recovers the exact linear dependency with zero error.
        """
        for seed in range(10):
            rng = np.random.default_rng(seed)
            x1 = rng.standard_normal(500)
            x2 = rng.standard_normal(500)
            x3 = 2.0 * x1 - 1.5 * x2
            x4 = rng.standard_normal(500)
            x5 = rng.standard_normal(500)

            df = pd.DataFrame({
                "X1": x1,
                "X2": x2,
                "X3": x3,
                "X4": x4,
                "X5": x5,
            })

            laws = scan_dataframe_conservation_laws(
                df=df,
                max_features=12,
                r2_threshold=0.9999,
                residual_var_tol=1e-8,
            )

            # Invariant: exactly one closed linear conservation law must be discovered
            assert len(laws) == 1, f"Seed {seed}: Expected 1 law, discovered {len(laws)}: {laws}"
            law = laws[0]

            # Invariant: non-zero coefficients must strictly involve X1, X2, X3 and exclude independent X4, X5
            coeffs = law.coefficients
            assert "X1" in coeffs, f"Seed {seed}: X1 missing from coefficients {coeffs}"
            assert "X2" in coeffs, f"Seed {seed}: X2 missing from coefficients {coeffs}"
            assert "X3" in coeffs, f"Seed {seed}: X3 missing from coefficients {coeffs}"
            assert "X4" not in coeffs, f"Seed {seed}: X4 spuriously included in coefficients {coeffs}"
            assert "X5" not in coeffs, f"Seed {seed}: X5 spuriously included in coefficients {coeffs}"

            # Invariant: exact normalized coefficients: 2.0*X1 - 1.5*X2 - 1.0*X3 = 0
            assert coeffs["X1"] == pytest.approx(2.0, abs=1e-5), f"Seed {seed}: X1 coefficient != 2.0"
            assert coeffs["X2"] == pytest.approx(-1.5, abs=1e-5), f"Seed {seed}: X2 coefficient != -1.5"
            assert coeffs["X3"] == pytest.approx(-1.0, abs=1e-5), f"Seed {seed}: X3 coefficient != -1.0"

            # Invariant: perfect variance explained (R^2 = 1.0) and zero residual variance (< 1e-8)
            assert law.variance_explained == pytest.approx(1.0, abs=1e-6)
            assert law.residual_variance <= 1e-8
            assert law.target_sum == 0.0

            # Invariant: formula text generation
            assert "2.0 X1 - 1.5 X2 - X3 = 0" in law.formula or "2.0*X1 - 1.5*X2 - X3 = 0" in law.formula or "X3" in law.formula

    def test_pure_gaussian_noise_false_positive_rejection(self):
        """
        Fuzzes with pure Gaussian noise (500 rows x 6 independent random normals).
        Asserts scanner discovers 0 spurious conservation laws across 10 random seeds (100% false positive rejection).
        """
        for seed in range(10):
            rng = np.random.default_rng(seed + 500)
            noise_matrix = rng.standard_normal((500, 6))
            df_noise = pd.DataFrame(
                noise_matrix,
                columns=[f"noise_feat_{i}" for i in range(6)],
            )

            laws = scan_dataframe_conservation_laws(
                df=df_noise,
                max_features=12,
                r2_threshold=0.9999,
                residual_var_tol=1e-8,
            )

            # Invariant: 100% false-positive rejection
            assert len(laws) == 0, f"Seed {seed}: Spurious conservation law discovered in pure noise: {laws}"

    def test_multiple_simultaneous_disjoint_conservation_laws(self):
        """
        Verifies that SVD null-space analysis with Row Reduced Echelon Form (RREF)
        correctly separates multiple simultaneous conservation laws without cross-contamination.
        """
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = x1 + x2  # Law 1: X1 + X2 - X3 = 0

        x4 = rng.standard_normal(n)
        x5 = rng.standard_normal(n)
        x6 = 2.0 * x4 - x5  # Law 2: 2.0*X4 - X5 - X6 = 0

        x7 = rng.standard_normal(n)
        x8 = rng.standard_normal(n)

        df = pd.DataFrame({
            "X1": x1, "X2": x2, "X3": x3,
            "X4": x4, "X5": x5, "X6": x6,
            "X7": x7, "X8": x8,
        })

        laws = scan_dataframe_conservation_laws(df, max_features=12)
        assert len(laws) == 2, f"Expected 2 disjoint laws, discovered {len(laws)}"

        # Verify separation of support sets
        law1_feats = set(laws[0].coefficients.keys())
        law2_feats = set(laws[1].coefficients.keys())

        assert law1_feats == {"X1", "X2", "X3"}
        assert law2_feats == {"X4", "X5", "X6"}

        # Disjoint support invariant
        assert len(law1_feats.intersection(law2_feats)) == 0

    def test_pruning_constant_and_single_variable_columns(self):
        """
        Verifies that trivial all-zero or single-variable columns (zero variance)
        are cleanly pruned and do not generate vacuous single-variable conservation laws.
        """
        rng = np.random.default_rng(99)
        n = 500
        df = pd.DataFrame({
            "const_zero": np.zeros(n),
            "const_pi": np.full(n, 3.14159),
            "random_a": rng.standard_normal(n),
            "random_b": rng.standard_normal(n),
        })

        laws = scan_dataframe_conservation_laws(df)
        assert len(laws) == 0, f"Expected 0 laws for independent/constant data, found: {laws}"


# ───────────────────────────────────────────────────────────────────────────
# 2. End-to-End API Output Verification Suite
# ───────────────────────────────────────────────────────────────────────────

class TestAPIOutputVerification:
    """Rigorous end-to-end API lifecycle verification for /api/v1/invariants endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_demo_dataset_scan_lock_and_get_lifecycle(self, client):
        """
        Tests the complete Suggest & Lock lifecycle on an ingested demo dataset:
        1. Ingest demo dataset -> session_id
        2. POST /api/v1/invariants/scan/{session_id} -> verify schema and candidate laws
        3. POST /api/v1/invariants/lock/{session_id} -> lock candidate laws
        4. GET /api/v1/invariants/{session_id} -> verify state persistence and compliance audit
        5. Direct assertion on SESSION_STORE[session_id]["locked_invariants"]
        """
        # 1. Ingest demo benchmark
        res_ingest = client.post("/api/v1/ingest/demo", json={"demo_name": "Telecom Customer Churn"})
        assert res_ingest.status_code == 200
        ingest_data = res_ingest.json()
        session_id = ingest_data["session_id"]
        assert session_id is not None
        assert session_id in SESSION_STORE

        # 2. Scan invariants
        res_scan = client.post(f"/api/v1/invariants/scan/{session_id}")
        assert res_scan.status_code == 200
        scan_data = res_scan.json()

        # Validate Pydantic V2 schema compliance
        scan_model = InvariantScanResponse.model_validate(scan_data)
        assert scan_model.session_id == session_id
        assert scan_model.status == "success"
        assert scan_model.laws_found >= 1
        assert len(scan_model.candidates) >= 1

        first_law = scan_model.candidates[0]
        assert len(first_law.name) > 0
        assert len(first_law.coefficients) >= 2
        assert first_law.target_sum == 0.0
        assert first_law.variance_explained > 0.0

        # 3. Lock invariant into session manifold
        lock_payload = InvariantLockRequest(laws=[first_law])
        res_lock = client.post(
            f"/api/v1/invariants/lock/{session_id}",
            json=lock_payload.model_dump(),
        )
        assert res_lock.status_code == 200
        lock_data = res_lock.json()

        # Validate Lock Response schema
        lock_model = InvariantLockResponse.model_validate(lock_data)
        assert lock_model.session_id == session_id
        assert lock_model.status == "locked"
        assert lock_model.locked_count == 1
        assert len(lock_model.locked_invariants) == 1
        assert lock_model.locked_invariants[0].name == first_law.name

        # 4. Query GET /api/v1/invariants/{session_id}
        res_get = client.get(f"/api/v1/invariants/{session_id}")
        assert res_get.status_code == 200
        get_data = res_get.json()

        # Validate Status Response schema
        status_model = InvariantsStatusResponse.model_validate(get_data)
        assert status_model.session_id == session_id
        assert status_model.status == "success"
        assert status_model.locked_count == 1
        assert len(status_model.locked_invariants) == 1
        assert status_model.locked_invariants[0].name == first_law.name

        # Verify audit report output
        assert status_model.audit_report is not None
        assert status_model.audit_report.total_records == ingest_data["n_rows"]
        assert status_model.audit_report.passed_records >= 0
        assert 0.0 <= status_model.audit_report.violation_rate <= 1.0

        # 5. Direct State Persistence Assertion in SESSION_STORE
        stored_session = SESSION_STORE[session_id]
        assert "locked_invariants" in stored_session
        assert len(stored_session["locked_invariants"]) == 1
        stored_law = stored_session["locked_invariants"][0]
        assert stored_law.name == first_law.name
        assert stored_law.coefficients == first_law.coefficients

    def test_synthetic_exact_conservation_api_lifecycle(self, client):
        """
        Creates a custom session with an exact synthetic linear balance law
        (X3 = 2.0*X1 - 1.5*X2), scans it via API, locks it, and verifies 100% compliance.
        """
        rng = np.random.default_rng(123)
        n = 500
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = 2.0 * x1 - 1.5 * x2

        df_synthetic = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
        session_id = str(uuid.uuid4())

        # Seed custom session in session store
        SESSION_STORE[session_id] = {
            "df": df_synthetic,
            "goal": "Verify conservation law API",
            "target": "X3",
            "tenant_id": "default",
            "org_id": "default",
            "locked_invariants": [],
        }

        # 1. Scan via API
        res_scan = client.post(f"/api/v1/invariants/scan/{session_id}")
        assert res_scan.status_code == 200
        scan_body = res_scan.json()
        assert scan_body["laws_found"] == 1
        cand = scan_body["candidates"][0]

        assert cand["coefficients"]["X1"] == pytest.approx(2.0, abs=1e-5)
        assert cand["coefficients"]["X2"] == pytest.approx(-1.5, abs=1e-5)
        assert cand["coefficients"]["X3"] == pytest.approx(-1.0, abs=1e-5)
        assert cand["variance_explained"] == pytest.approx(1.0, abs=1e-6)

        # 2. Lock via API
        res_lock = client.post(
            f"/api/v1/invariants/lock/{session_id}",
            json={"laws": [cand]},
        )
        assert res_lock.status_code == 200
        assert res_lock.json()["locked_count"] == 1

        # 3. Retrieve status via API and verify 100% compliance
        res_get = client.get(f"/api/v1/invariants/{session_id}")
        assert res_get.status_code == 200
        status_data = res_get.json()
        audit = status_data["audit_report"]

        assert audit["is_axiom_compliant"] is True
        assert audit["violation_rate"] == 0.0
        assert audit["total_records"] == 500
        assert audit["passed_records"] == 500
        assert len(audit["details"]) >= 1

    def test_direct_list_lock_payload(self, client):
        """Verifies API accepts direct list of laws in POST /api/v1/invariants/lock."""
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = {
            "df": pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 2.0]}),
            "tenant_id": "default",
            "locked_invariants": [],
        }

        law_payload = [
            {
                "name": "Equality Axiom: A - B = 0",
                "formula": "A - B = 0",
                "coefficients": {"A": 1.0, "B": -1.0},
                "target_sum": 0.0,
                "tolerance": 1e-6,
            }
        ]

        res_lock = client.post(f"/api/v1/invariants/lock/{session_id}", json=law_payload)
        assert res_lock.status_code == 200
        assert res_lock.json()["locked_count"] == 1
        assert len(SESSION_STORE[session_id]["locked_invariants"]) == 1

    def test_session_not_found_boundary(self, client):
        """Verifies 404 responses for non-existent session IDs."""
        missing_id = "non-existent-session-00000"
        r1 = client.post(f"/api/v1/invariants/scan/{missing_id}")
        assert r1.status_code == 404

        r2 = client.post(f"/api/v1/invariants/lock/{missing_id}", json={"laws": []})
        assert r2.status_code == 404

        r3 = client.get(f"/api/v1/invariants/{missing_id}")
        assert r3.status_code == 404
