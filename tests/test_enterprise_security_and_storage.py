"""
tests/test_enterprise_security_and_storage.py
─────────────────────────────────────────────
Comprehensive Enterprise Test Suite verifying:
1. Pluggable Role-Based Access Control (RBAC): Admin, DataScientist, Auditor, Viewer
2. Pure-Python RFC 7519 HMAC-SHA256 JWT engine (sign, verify, tamper-detect, expire)
3. API Key resolution with multi-tenant context injection
4. Enterprise Security Headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options)
5. Audit event logging and forensic trail queryability
6. Multi-tenant session and storage isolation (tenant_id / org_id partitioning)
7. Pluggable Storage Abstraction (LocalStorageBackend & S3CompatibleStorageBackend)
8. Path traversal attack defenses (directory escaping prevention)
9. System readiness (/api/v1/ready) and health (/api/v1/health) storage reporting
10. Kubernetes manifests syntax and structure validation
"""

import json
import os
import shutil
import tempfile
import time
import pytest
import yaml
from fastapi.testclient import TestClient

from api.schemas import (
    AuthTokenRequest,
    DemoIngestRequest,
    HealthResponse,
    SystemReadinessResponse,
)
from api.server import (
    SESSION_STORE,
    app,
    get_health,
    get_readiness,
    get_system_metrics,
    ingest_demo_dataset,
)
import dia.config as dia_config
from dia.rbac import (
    Permission,
    UserContext,
    UserRole,
    audit_logger,
    build_user_context,
    create_access_token,
    decode_access_token,
    resolve_api_key,
)
from dia.storage_provider import (
    LocalStorageBackend,
    S3CompatibleStorageBackend,
    get_storage_backend,
    reset_storage_backend,
    set_storage_backend,
)


# ─── 1. RBAC & User Context Tests ─────────────────────────────────────────────

def test_rbac_roles_and_permissions():
    """Verify standard permissions matrix for Admin, DataScientist, Auditor, and Viewer."""
    admin = build_user_context(user_id="admin-1", role=UserRole.ADMIN.value)
    assert admin.is_admin()
    assert admin.has_permission(Permission.SYSTEM_ADMIN)
    assert admin.has_permission(Permission.DATA_INGEST)
    assert admin.has_permission(Permission.MODEL_TRAIN)
    assert admin.has_permission(Permission.AUDIT_READ)

    ds = build_user_context(user_id="ds-1", role=UserRole.DATA_SCIENTIST.value)
    assert not ds.is_admin()
    assert ds.has_permission(Permission.DATA_INGEST)
    assert ds.has_permission(Permission.MODEL_TRAIN)
    assert ds.has_permission(Permission.MODEL_EXPORT)
    assert not ds.has_permission(Permission.SYSTEM_ADMIN)
    assert not ds.has_permission(Permission.AUDIT_READ)

    auditor = build_user_context(user_id="aud-1", role=UserRole.AUDITOR.value)
    assert not auditor.is_admin()
    assert auditor.has_permission(Permission.GOVERNANCE_READ)
    assert auditor.has_permission(Permission.AUDIT_READ)
    assert not auditor.has_permission(Permission.MODEL_TRAIN)
    assert not auditor.has_permission(Permission.DATA_INGEST)

    viewer = build_user_context(user_id="view-1", role=UserRole.VIEWER.value)
    assert not viewer.is_admin()
    assert viewer.has_permission(Permission.SYSTEM_READ)
    assert viewer.has_permission(Permission.DATA_READ)
    assert not viewer.has_permission(Permission.DATA_INGEST)
    assert not viewer.has_permission(Permission.MODEL_TRAIN)
    assert not viewer.has_permission(Permission.AUDIT_READ)


def test_tenant_access_control():
    """Verify tenant isolation rules on UserContext."""
    admin = build_user_context(user_id="adm", role=UserRole.ADMIN.value, tenant_id="tenant-alpha")
    user_alpha = build_user_context(user_id="u1", role=UserRole.DATA_SCIENTIST.value, tenant_id="tenant-alpha")
    user_beta = build_user_context(user_id="u2", role=UserRole.DATA_SCIENTIST.value, tenant_id="tenant-beta")

    # Admin can access any tenant
    assert admin.can_access_tenant("tenant-alpha")
    assert admin.can_access_tenant("tenant-beta")
    assert admin.can_access_tenant("tenant-gamma")

    # Non-admin can only access own tenant
    assert user_alpha.can_access_tenant("tenant-alpha")
    assert not user_alpha.can_access_tenant("tenant-beta")
    assert user_beta.can_access_tenant("tenant-beta")
    assert not user_beta.can_access_tenant("tenant-alpha")


# ─── 2. Pure-Python RFC 7519 HMAC-SHA256 JWT Engine ───────────────────────────

def test_jwt_token_lifecycle():
    """Verify sign, verify, tamper-detection, and expiration in pure Python JWT engine."""
    secret = "test-secret-key-32-characters-minimum!"
    token = create_access_token(
        user_id="alice",
        role=UserRole.DATA_SCIENTIST.value,
        tenant_id="tenant-finance",
        org_id="risk-division",
        expires_minutes=60,
        secret=secret,
    )
    assert isinstance(token, str)
    assert token.count(".") == 2

    # 1. Valid token decode
    payload = decode_access_token(token, secret=secret)
    assert payload["sub"] == "alice"
    assert payload["role"] == UserRole.DATA_SCIENTIST.value
    assert payload["tenant_id"] == "tenant-finance"
    assert payload["org_id"] == "risk-division"
    assert payload["exp"] > time.time()

    # 2. Tampered signature fails
    with pytest.raises(ValueError, match="signature verification failed"):
        decode_access_token(token, secret="wrong-secret-key")

    # 3. Tampered payload fails
    parts = token.split(".")
    tampered = f"{parts[0]}.eyJob21lIjoidHJ1ZSJ9.{parts[2]}"
    with pytest.raises(ValueError, match="signature verification failed"):
        decode_access_token(tampered, secret=secret)

    # 4. Expired token fails
    expired_token = create_access_token(
        user_id="bob",
        expires_minutes=-10,  # in the past
        secret=secret,
    )
    with pytest.raises(ValueError, match="expired"):
        decode_access_token(expired_token, secret=secret)

    # 5. Malformed format fails
    with pytest.raises(ValueError, match="Invalid JWT format"):
        decode_access_token("not.a.valid.jwt.string", secret=secret)


# ─── 3. API Key Resolution Tests ──────────────────────────────────────────────

def test_api_key_resolution():
    """Verify built-in enterprise API keys and custom configuration."""
    admin_ctx = resolve_api_key("dia-admin-key")
    assert admin_ctx is not None
    assert admin_ctx.role == UserRole.ADMIN.value
    assert admin_ctx.is_admin()

    ds_ctx = resolve_api_key("dia-ds-key")
    assert ds_ctx is not None
    assert ds_ctx.role == UserRole.DATA_SCIENTIST.value

    aud_ctx = resolve_api_key("dia-auditor-key")
    assert aud_ctx is not None
    assert aud_ctx.role == UserRole.AUDITOR.value

    view_ctx = resolve_api_key("dia-viewer-key")
    assert view_ctx is not None
    assert view_ctx.role == UserRole.VIEWER.value

    # Unknown key returns None
    assert resolve_api_key("unknown-bogus-key") is None
    assert resolve_api_key("") is None


# ─── 4. Audit Event Logging Tests ─────────────────────────────────────────────

def test_audit_event_logging():
    """Verify thread-safe audit logging buffer and query filtering."""
    ctx_corp = build_user_context(user_id="auditor-1", role=UserRole.AUDITOR.value, tenant_id="corp")
    ctx_retail = build_user_context(user_id="ds-1", role=UserRole.DATA_SCIENTIST.value, tenant_id="retail")

    ev1 = audit_logger.record_event("DATA_INGEST", ctx_corp, "/api/v1/ingest", "SUCCESS", {"rows": 100})
    ev2 = audit_logger.record_event("MODEL_TRAIN", ctx_retail, "/api/v1/train", "SUCCESS", {"model": "rf"})
    ev3 = audit_logger.record_event("AUTH_FAILURE", None, "/api/v1/login", "FAILED", {"ip": "1.2.3.4"})

    assert ev1.event_id is not None
    assert ev1.tenant_id == "corp"
    assert ev2.tenant_id == "retail"

    # Query all events
    all_events = audit_logger.query_events(limit=50)
    assert len(all_events) >= 3

    # Filter by tenant
    corp_events = audit_logger.query_events(tenant_id="corp", limit=50)
    assert any(e.event_id == ev1.event_id for e in corp_events)
    assert not any(e.event_id == ev2.event_id for e in corp_events)

    # Filter by event type
    train_events = audit_logger.query_events(event_type="MODEL_TRAIN", limit=50)
    assert any(e.event_id == ev2.event_id for e in train_events)
    assert not any(e.event_id == ev1.event_id for e in train_events)


# ─── 5. Pluggable LocalStorageBackend Tests ────────────────────────────────────

def test_local_storage_backend_operations():
    """Verify LocalStorageBackend CRUD, atomic writes, and health checks."""
    temp_dir = tempfile.mkdtemp(prefix="dia_test_storage_")
    try:
        backend = LocalStorageBackend(root_path=temp_dir)
        tenant = "acme-corp"

        # 1. Save and Read
        uri = backend.save("models/model_v1.pkl", b"\x80\x04\x95test_bytes", tenant_id=tenant)
        assert os.path.exists(uri)
        data = backend.read("models/model_v1.pkl", tenant_id=tenant)
        assert data == b"\x80\x04\x95test_bytes"

        # 2. Exists
        assert backend.exists("models/model_v1.pkl", tenant_id=tenant)
        assert not backend.exists("models/missing.pkl", tenant_id=tenant)

        # 3. List files
        backend.save("datasets/raw.csv", b"a,b,c\n1,2,3\n", tenant_id=tenant)
        files = backend.list_files(tenant_id=tenant)
        assert "models/model_v1.pkl" in files
        assert "datasets/raw.csv" in files

        # 4. Get URL
        url = backend.get_url("models/model_v1.pkl", tenant_id=tenant)
        assert url.startswith("file://")

        # 5. Delete
        assert backend.delete("models/model_v1.pkl", tenant_id=tenant)
        assert not backend.exists("models/model_v1.pkl", tenant_id=tenant)
        assert not backend.delete("models/model_v1.pkl", tenant_id=tenant)

        # 6. Health probe
        h = backend.health()
        assert h["status"] == "healthy"
        assert h["writable"] is True
        assert h["backend"] == "local"
        assert h["latency_ms"] >= 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_local_storage_path_traversal_defense():
    """Verify LocalStorageBackend prevents directory escaping attacks."""
    temp_dir = tempfile.mkdtemp(prefix="dia_test_traversal_")
    try:
        backend = LocalStorageBackend(root_path=temp_dir)
        tenant = "tenant-safe"

        # Path traversal with ../ should be blocked
        with pytest.raises(ValueError, match="Directory traversal prohibited"):
            backend.save("../../etc/shadow", b"malicious", tenant_id=tenant)

        with pytest.raises(ValueError, match="Directory traversal prohibited"):
            backend.read("../../../windows/system.ini", tenant_id=tenant)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─── 6. Pluggable S3CompatibleStorageBackend Tests ─────────────────────────────

def test_s3_compatible_storage_backend_mock_mode():
    """Verify S3CompatibleStorageBackend CRUD, namespacing, and health checks in mock mode."""
    backend = S3CompatibleStorageBackend(bucket="enterprise-vault", prefix="data_mesh", mock=True)
    tenant = "fintech-org"

    # 1. Save and Read
    s3_uri = backend.save("checkpoints/cp_01.bin", b"checkpoint_binary_content", tenant_id=tenant)
    assert s3_uri == f"s3://enterprise-vault/data_mesh/{tenant}/checkpoints/cp_01.bin"

    assert backend.exists("checkpoints/cp_01.bin", tenant_id=tenant)
    assert not backend.exists("checkpoints/non_existent.bin", tenant_id=tenant)

    read_data = backend.read("checkpoints/cp_01.bin", tenant_id=tenant)
    assert read_data == b"checkpoint_binary_content"

    # 2. List
    backend.save("exports/report.pdf", b"pdf_data", tenant_id=tenant)
    files = backend.list_files(tenant_id=tenant)
    assert "checkpoints/cp_01.bin" in files
    assert "exports/report.pdf" in files

    # 3. Get URL
    url = backend.get_url("exports/report.pdf", tenant_id=tenant)
    assert "enterprise-vault" in url

    # 4. Delete
    assert backend.delete("checkpoints/cp_01.bin", tenant_id=tenant)
    assert not backend.exists("checkpoints/cp_01.bin", tenant_id=tenant)

    # 5. Health probe
    h = backend.health()
    assert h["status"] == "healthy"
    assert h["backend"] == "s3"
    assert h["writable"] is True
    assert h["bucket"] == "enterprise-vault"


def test_storage_backend_factory_singleton():
    """Verify get_storage_backend and set_storage_backend factory."""
    orig = get_storage_backend()
    assert orig is not None

    custom = S3CompatibleStorageBackend(bucket="custom-test", mock=True)
    set_storage_backend(custom)
    assert get_storage_backend() is custom
    assert get_storage_backend().health()["bucket"] == "custom-test"

    # Reset back
    set_storage_backend(orig)


# ─── 7. HTTP API Middleware & Security Headers Tests ──────────────────────────

def test_enterprise_security_headers():
    """Verify CSP, HSTS, X-Frame-Options, and X-Content-Type-Options on HTTP responses."""
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200

    headers = resp.headers
    assert "Content-Security-Policy" in headers
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-XSS-Protection") == "1; mode=block"


def test_health_and_readiness_endpoints():
    """Verify /api/v1/health and /api/v1/ready reporting storage status and hardware telemetry."""
    client = TestClient(app)

    # Health
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert h_data["status"] == "healthy"
    assert "storage_status" in h_data
    assert "storage_type" in h_data
    assert h_data["storage_status"] in ("healthy", "degraded")
    assert h_data["cpu_cores"] > 0
    assert h_data["tenants_count"] >= 1

    # Readiness
    r_ready = client.get("/api/v1/ready")
    assert r_ready.status_code == 200
    ready_data = r_ready.json()
    assert ready_data["status"] == "ready"
    assert ready_data["storage_healthy"] is True
    assert ready_data["memory_healthy"] is True
    assert "subsystems" in ready_data
    assert ready_data["subsystems"]["storage"] == "healthy"


def test_auth_token_issuance_and_profile():
    """Verify /api/v1/auth/token generates valid JWT and /api/v1/auth/me verifies profile."""
    client = TestClient(app)

    # 1. Issue token via API Key
    resp = client.post(
        "/api/v1/auth/token",
        json={"api_key": "dia-ds-key"},
    )
    assert resp.status_code == 200
    token_data = resp.json()
    assert token_data["token_type"] == "bearer"
    assert token_data["role"] == UserRole.DATA_SCIENTIST.value
    jwt_token = token_data["access_token"]

    # 2. Access /api/v1/auth/me using Bearer JWT
    resp_me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {jwt_token}"},
    )
    assert resp_me.status_code == 200
    me_data = resp_me.json()
    assert me_data["role"] == UserRole.DATA_SCIENTIST.value
    assert "data:ingest" in me_data["permissions"]


def test_storage_api_endpoints():
    """Verify /api/v1/storage/status and /api/v1/storage/files."""
    client = TestClient(app)

    # Status
    resp = client.get("/api/v1/storage/status")
    assert resp.status_code == 200
    st_data = resp.json()
    assert st_data["status"] in ("healthy", "degraded")
    assert st_data["writable"] is True

    # Files
    resp_files = client.get("/api/v1/storage/files")
    assert resp_files.status_code == 200
    assert "files" in resp_files.json()


def test_multi_tenant_session_isolation_via_http():
    """Verify that tenant A's session is isolated and protected from tenant B."""
    client = TestClient(app)

    # 1. Ingest session under tenant-alpha
    token_alpha = create_access_token(
        user_id="user-alpha",
        role=UserRole.DATA_SCIENTIST.value,
        tenant_id="tenant-alpha",
    )
    ingest_resp = client.post(
        "/api/v1/ingest/demo",
        json={"demo_name": "Telecom Customer Churn"},
        headers={"Authorization": f"Bearer {token_alpha}", "X-Tenant-ID": "tenant-alpha"},
    )
    assert ingest_resp.status_code == 200
    session_id = ingest_resp.json()["session_id"]

    # 2. Access session with tenant-alpha token -> SUCCESS
    prof_alpha = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"Authorization": f"Bearer {token_alpha}", "X-Tenant-ID": "tenant-alpha"},
    )
    assert prof_alpha.status_code == 200

    # 3. Access session with tenant-beta token -> 403 FORBIDDEN
    token_beta = create_access_token(
        user_id="user-beta",
        role=UserRole.DATA_SCIENTIST.value,
        tenant_id="tenant-beta",
    )
    prof_beta = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"Authorization": f"Bearer {token_beta}", "X-Tenant-ID": "tenant-beta"},
    )
    assert prof_beta.status_code == 403
    assert "belongs to tenant" in prof_beta.json()["detail"]

    # 4. Access session with admin token -> SUCCESS (Admin cross-tenant access)
    token_admin = create_access_token(
        user_id="admin-master",
        role=UserRole.ADMIN.value,
        tenant_id="enterprise",
    )
    prof_admin = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"Authorization": f"Bearer {token_admin}"},
    )
    assert prof_admin.status_code == 200


def test_rbac_method_restrictions_via_http():
    """Verify that Viewer and Auditor roles are restricted from mutating actions."""
    client = TestClient(app)

    # Viewer cannot POST
    token_viewer = create_access_token(
        user_id="viewer-1",
        role=UserRole.VIEWER.value,
        tenant_id="t-view",
    )
    post_resp = client.post(
        "/api/v1/ingest/demo",
        json={"demo_name": "Telecom Customer Churn"},
        headers={"Authorization": f"Bearer {token_viewer}"},
    )
    assert post_resp.status_code == 403
    assert "read-only access" in post_resp.json()["detail"]

    # Auditor cannot train models
    token_auditor = create_access_token(
        user_id="auditor-1",
        role=UserRole.AUDITOR.value,
        tenant_id="t-audit",
    )
    train_resp = client.post(
        "/api/v1/pipeline/train",
        json={"session_id": "dummy", "goal": "predict"},
        headers={"Authorization": f"Bearer {token_auditor}"},
    )
    assert train_resp.status_code == 403
    assert "Auditor" in train_resp.json()["detail"]


# ─── 8. Kubernetes Manifests Validation ───────────────────────────────────────

def test_kubernetes_manifests_valid_yaml():
    """Verify that all Kubernetes manifests in deploy/k8s/ are syntactically valid YAML."""
    k8s_dir = os.path.join(os.path.dirname(__file__), "..", "deploy", "k8s")
    assert os.path.isdir(k8s_dir), f"deploy/k8s directory must exist: {k8s_dir}"

    expected_files = [
        "namespace.yaml",
        "configmap.yaml",
        "secret.yaml",
        "pvc.yaml",
        "deployment.yaml",
        "service.yaml",
        "ingress.yaml",
    ]

    for fname in expected_files:
        fpath = os.path.join(k8s_dir, fname)
        assert os.path.exists(fpath), f"Missing Kubernetes manifest: {fname}"
        with open(fpath, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            assert len(docs) >= 1, f"Manifest {fname} is empty"
            for doc in docs:
                assert "apiVersion" in doc, f"Manifest {fname} missing apiVersion"
                assert "kind" in doc, f"Manifest {fname} missing kind"
                assert "metadata" in doc, f"Manifest {fname} missing metadata"

    # Specific checks on deployment.yaml
    deploy_path = os.path.join(k8s_dir, "deployment.yaml")
    with open(deploy_path, "r", encoding="utf-8") as f:
        deploy_doc = yaml.safe_load(f)
        spec = deploy_doc["spec"]["template"]["spec"]
        assert spec["securityContext"]["runAsNonRoot"] is True
        assert spec["securityContext"]["runAsUser"] == 10001
        container = spec["containers"][0]
        assert container["livenessProbe"]["httpGet"]["path"] == "/api/v1/health"
        assert container["readinessProbe"]["httpGet"]["path"] == "/api/v1/ready"
        assert any(v["name"] == "storage-volume" for v in spec["volumes"])
        assert any(m["name"] == "storage-volume" for m in container["volumeMounts"])

    # Check kustomization.yaml exists and is valid
    kustomize_path = os.path.join(k8s_dir, "kustomization.yaml")
    assert os.path.exists(kustomize_path), "kustomization.yaml must exist in deploy/k8s"
    with open(kustomize_path, "r", encoding="utf-8") as f:
        k_doc = yaml.safe_load(f)
        assert k_doc.get("kind") == "Kustomization"
        for ef in expected_files:
            assert ef in k_doc.get("resources", []), f"{ef} must be declared in kustomization.yaml"


# ─── 9. Adversarial Security & Multi-Tenancy Boundary Tests ───────────────────

def test_tenant_spoofing_defense_on_jwt_tokens():
    """Verify that non-admin Bearer token holders cannot spoof or switch tenants via X-Tenant-ID header."""
    client = TestClient(app)

    # 1. Ingest session under tenant-victim
    token_victim = create_access_token(user_id="victim", role=UserRole.DATA_SCIENTIST.value, tenant_id="tenant-victim")
    ingest_res = client.post(
        "/api/v1/ingest/demo",
        json={"demo_name": "Telecom Customer Churn"},
        headers={"Authorization": f"Bearer {token_victim}"},
    )
    assert ingest_res.status_code == 200
    session_id = ingest_res.json()["session_id"]

    # 2. Attacker belongs to tenant-attacker, tries to spoof X-Tenant-ID: tenant-victim
    token_attacker = create_access_token(user_id="attacker", role=UserRole.DATA_SCIENTIST.value, tenant_id="tenant-attacker")
    spoof_resp = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"Authorization": f"Bearer {token_attacker}", "X-Tenant-ID": "tenant-victim"},
    )
    assert spoof_resp.status_code == 403
    assert "Forbidden: non-admin user cannot switch to tenant" in spoof_resp.json()["detail"]

    # 3. Attacker accesses without X-Tenant-ID header -> properly rejected with tenant isolation
    clean_resp = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"Authorization": f"Bearer {token_attacker}"},
    )
    assert clean_resp.status_code == 403
    assert "belongs to tenant 'tenant-victim'" in clean_resp.json()["detail"]

    # 4. Legitimate Admin CAN switch or impersonate tenants via X-Tenant-ID
    token_admin = create_access_token(user_id="admin", role=UserRole.ADMIN.value, tenant_id="enterprise")
    admin_resp = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"Authorization": f"Bearer {token_admin}", "X-Tenant-ID": "tenant-victim"},
    )
    assert admin_resp.status_code == 200


def test_tenant_spoofing_defense_on_api_keys():
    """Verify that non-admin API key holders cannot spoof tenants via X-Tenant-ID header."""
    client = TestClient(app)

    # Ingest session under tenant-alpha
    token_alpha = create_access_token(user_id="u1", role=UserRole.ADMIN.value, tenant_id="tenant-alpha")
    ingest_res = client.post(
        "/api/v1/ingest/demo",
        json={"demo_name": "Telecom Customer Churn"},
        headers={"Authorization": f"Bearer {token_alpha}", "X-Tenant-ID": "tenant-alpha"},
    )
    session_id = ingest_res.json()["session_id"]

    # Non-admin API key (dia-ds-key has default tenant 'default') tries to switch to tenant-alpha
    spoof_resp = client.get(
        f"/api/v1/profile/{session_id}",
        headers={"X-API-Key": "dia-ds-key", "X-Tenant-ID": "tenant-alpha"},
    )
    assert spoof_resp.status_code == 403
    assert "Forbidden: non-admin API key cannot switch to tenant" in spoof_resp.json()["detail"]


def test_auth_enabled_strict_token_endpoint_enforcement():
    """Verify that when AUTH_ENABLED=True, token issuance requires valid API keys and prevents unauthenticated Admin minting."""
    client = TestClient(app)
    orig_auth = dia_config.AUTH_ENABLED
    try:
        dia_config.AUTH_ENABLED = True

        # 1. Unauthenticated token request without API key is rejected
        anon_resp = client.post(
            "/api/v1/auth/token",
            json={"role": "Admin", "tenant_id": "corp", "username": "attacker"},
        )
        assert anon_resp.status_code == 401
        assert "Authentication required: an API key must be provided" in anon_resp.json()["detail"]

        # 2. Invalid API key is rejected
        invalid_resp = client.post(
            "/api/v1/auth/token",
            json={"api_key": "invalid-bogus-key", "role": "Admin"},
        )
        assert invalid_resp.status_code == 401

        # 3. Non-admin API key cannot elevate to Admin
        ds_resp = client.post(
            "/api/v1/auth/token",
            json={"api_key": "dia-ds-key", "role": "Admin"},
        )
        assert ds_resp.status_code == 200
        # Role must be locked to DataScientist, ignoring the requested Admin escalation
        assert ds_resp.json()["role"] == UserRole.DATA_SCIENTIST.value

        # 4. Admin API key CAN specify role and tenant
        admin_resp = client.post(
            "/api/v1/auth/token",
            json={"api_key": "dia-admin-key", "role": "Viewer", "tenant_id": "partner-xyz"},
        )
        assert admin_resp.status_code == 200
        assert admin_resp.json()["role"] == "Viewer"
        assert admin_resp.json()["tenant_id"] == "partner-xyz"
    finally:
        dia_config.AUTH_ENABLED = orig_auth


def test_custom_api_keys_disables_builtin_backdoors():
    """Verify that setting custom DIA_API_KEYS disables the built-in default keys."""
    orig_keys = dia_config.API_KEYS_CONFIG
    try:
        dia_config.API_KEYS_CONFIG = json.dumps({
            "secret-prod-1": {"role": "Viewer", "tenant_id": "prod-tenant", "org_id": "hq", "user_id": "v1"}
        })

        # Built-in dia-admin-key should no longer resolve
        assert resolve_api_key("dia-admin-key") is None
        assert resolve_api_key("dia-ds-key") is None

        # Custom configured key resolves
        custom_ctx = resolve_api_key("secret-prod-1")
        assert custom_ctx is not None
        assert custom_ctx.role == UserRole.VIEWER.value
        assert custom_ctx.tenant_id == "prod-tenant"
    finally:
        dia_config.API_KEYS_CONFIG = orig_keys
        # Restored
        assert resolve_api_key("dia-admin-key") is not None


def test_local_storage_sibling_and_empty_path_traversal_defense():
    """Verify LocalStorageBackend rejects sibling directory prefixes and empty paths."""
    temp_dir = tempfile.mkdtemp(prefix="dia_test_sibling_")
    try:
        backend = LocalStorageBackend(root_path=temp_dir)
        tenant = "corp"

        # Sibling folder prefix traversal (e.g. corp_victim)
        with pytest.raises(ValueError, match="Directory traversal prohibited"):
            backend.save("../corp_victim/secret.txt", b"exploit", tenant_id=tenant)

        with pytest.raises(ValueError, match="Directory traversal prohibited"):
            backend.read("../corp_victim/secret.txt", tenant_id=tenant)

        # Empty path
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.save("", b"data", tenant_id=tenant)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_s3_storage_path_traversal_and_empty_path_defense():
    """Verify S3CompatibleStorageBackend rejects path traversal and empty paths."""
    backend = S3CompatibleStorageBackend(bucket="secure-vault", mock=True)
    tenant = "tenant_a"

    # Relative ../ traversal escaping tenant
    with pytest.raises(ValueError, match="Directory traversal prohibited"):
        backend.save("../../other_tenant/secret.txt", b"hack", tenant_id=tenant)

    with pytest.raises(ValueError, match="Directory traversal prohibited"):
        backend.read("../../other_tenant/secret.txt", tenant_id=tenant)

    with pytest.raises(ValueError, match="Directory traversal prohibited"):
        backend.delete("../../other_tenant/secret.txt", tenant_id=tenant)

    # Empty path
    with pytest.raises(ValueError, match="cannot be empty"):
        backend.save("", b"hack", tenant_id=tenant)


def test_bounded_session_store_cross_tenant_overwrite_prevention():
    """Verify BoundedSessionStore prevents a different tenant from hijacking an existing session ID."""
    from dia.session_manager import BoundedSessionStore
    store = BoundedSessionStore(max_sessions=5)

    store["session_100"] = {"df": None, "tenant_id": "tenant-owner", "goal": "initial"}
    assert store._meta["session_100"]["tenant_id"] == "tenant-owner"

    # Different tenant tries to overwrite
    with pytest.raises(PermissionError, match="not permitted to overwrite session"):
        store["session_100"] = {"df": None, "tenant_id": "tenant-intruder", "goal": "hacked"}

    # Owner can update their session
    store["session_100"] = {"df": None, "tenant_id": "tenant-owner", "goal": "updated"}
    assert store._meta["session_100"]["goal"] == "updated"


def test_auth_enabled_blocks_unauthenticated_session_access():
    """Verify that when AUTH_ENABLED=True, unauthenticated requests to session endpoints are blocked."""
    client = TestClient(app)
    orig_auth = dia_config.AUTH_ENABLED

    # Ingest session with admin token while auth is disabled
    token = create_access_token(user_id="admin", role=UserRole.ADMIN.value, tenant_id="corp")
    res = client.post("/api/v1/ingest/demo", json={"demo_name": "Telecom Customer Churn"}, headers={"Authorization": f"Bearer {token}"})
    session_id = res.json()["session_id"]

    try:
        dia_config.AUTH_ENABLED = True
        # Try unauthenticated access
        unauth_res = client.get(f"/api/v1/profile/{session_id}")
        assert unauth_res.status_code == 401
    finally:
        dia_config.AUTH_ENABLED = orig_auth

