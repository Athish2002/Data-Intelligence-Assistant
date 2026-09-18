"""
dia/rbac.py
───────────
Enterprise Role-Based Access Control (RBAC), Multi-Tenancy & Audit Engine.
Enforces security contexts, persona permissions, tenant isolation,
zero-dependency RFC 7519 HMAC-SHA256 JWT authentication, and forensic audit logging.
"""

from __future__ import annotations

import base64
import collections
import enum
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dia.config import (
    API_KEYS_CONFIG,
    DEFAULT_ORG_ID,
    DEFAULT_ROLE,
    DEFAULT_TENANT_ID,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET,
)

log = logging.getLogger("dia.rbac")


# ─── RBAC Roles & Permissions ──────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    ADMIN = "Admin"
    DATA_SCIENTIST = "DataScientist"
    AUDITOR = "Auditor"
    VIEWER = "Viewer"


class Permission(str, enum.Enum):
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_READ = "system:read"
    DATA_INGEST = "data:ingest"
    DATA_READ = "data:read"
    DATA_DELETE = "data:delete"
    MODEL_TRAIN = "model:train"
    MODEL_PREDICT = "model:predict"
    MODEL_EXPORT = "model:export"
    GOVERNANCE_READ = "governance:read"
    GOVERNANCE_WRITE = "governance:write"
    STORAGE_READ = "storage:read"
    STORAGE_WRITE = "storage:write"
    AUDIT_READ = "audit:read"


ROLE_ACTION_PERMISSIONS: dict[str, set[str]] = {
    UserRole.ADMIN.value: {
        Permission.SYSTEM_ADMIN.value,
        Permission.SYSTEM_READ.value,
        Permission.DATA_INGEST.value,
        Permission.DATA_READ.value,
        Permission.DATA_DELETE.value,
        Permission.MODEL_TRAIN.value,
        Permission.MODEL_PREDICT.value,
        Permission.MODEL_EXPORT.value,
        Permission.GOVERNANCE_READ.value,
        Permission.GOVERNANCE_WRITE.value,
        Permission.STORAGE_READ.value,
        Permission.STORAGE_WRITE.value,
        Permission.AUDIT_READ.value,
        "*",
    },
    UserRole.DATA_SCIENTIST.value: {
        Permission.SYSTEM_READ.value,
        Permission.DATA_INGEST.value,
        Permission.DATA_READ.value,
        Permission.DATA_DELETE.value,
        Permission.MODEL_TRAIN.value,
        Permission.MODEL_PREDICT.value,
        Permission.MODEL_EXPORT.value,
        Permission.GOVERNANCE_READ.value,
        Permission.GOVERNANCE_WRITE.value,
        Permission.STORAGE_READ.value,
        Permission.STORAGE_WRITE.value,
    },
    UserRole.AUDITOR.value: {
        Permission.SYSTEM_READ.value,
        Permission.DATA_READ.value,
        Permission.GOVERNANCE_READ.value,
        Permission.STORAGE_READ.value,
        Permission.AUDIT_READ.value,
    },
    UserRole.VIEWER.value: {
        Permission.SYSTEM_READ.value,
        Permission.DATA_READ.value,
        Permission.GOVERNANCE_READ.value,
    },
}

# Aliases for backwards-compatibility with Streamlit UI persona strings
ROLE_ACTION_PERMISSIONS["Admin (Full Access)"] = ROLE_ACTION_PERMISSIONS[UserRole.ADMIN.value]
ROLE_ACTION_PERMISSIONS["Lead Data Scientist"] = ROLE_ACTION_PERMISSIONS[UserRole.DATA_SCIENTIST.value]
ROLE_ACTION_PERMISSIONS["Business Analyst / Executive"] = ROLE_ACTION_PERMISSIONS[UserRole.VIEWER.value]
ROLE_ACTION_PERMISSIONS["Compliance & Privacy Officer"] = ROLE_ACTION_PERMISSIONS[UserRole.AUDITOR.value]


# ─── Streamlit UI Dashboard Tab Role Permissions ──────────────────────────────

_ADMIN_TABS = [
    "📊 Overview", "🔍 Column Roles", "🩺 Readiness", "💡 Smart Insights",
    "🤖 Model Results", "🔬 Explainability", "💼 Business Impact", "🎛️ Simulator",
    "🎯 Causal & Counterfactuals", "📈 Time-Series Forecast", "🧬 Synthetic Data",
    "⚡ Online Learning", "💬 Chat Copilot", "🌊 Drift Monitor",
    "🔒 Privacy & Compliance", "📦 Model Registry & MLOps", "🛡️ Data Contract",
    "🧪 A/B Test Planner", "👥 Active Learning", "📑 Executive Briefing", "💻 Code Export", "📝 Summary"
]

_DS_TABS = [
    "📊 Overview", "🔍 Column Roles", "🩺 Readiness", "💡 Smart Insights",
    "🤖 Model Results", "🔬 Explainability", "🎛️ Simulator", "🎯 Causal & Counterfactuals",
    "📈 Time-Series Forecast", "🧬 Synthetic Data", "⚡ Online Learning", "💬 Chat Copilot",
    "🌊 Drift Monitor", "📦 Model Registry & MLOps", "🛡️ Data Contract", "👥 Active Learning",
    "💻 Code Export", "📝 Summary"
]

_AUDITOR_TABS = [
    "📊 Overview", "🔒 Privacy & Compliance", "🛡️ Data Contract",
    "📑 Executive Briefing", "📝 Summary"
]

_VIEWER_TABS = [
    "📊 Overview", "💡 Smart Insights", "💼 Business Impact", "🎛️ Simulator",
    "🎯 Causal & Counterfactuals", "📈 Time-Series Forecast", "💬 Chat Copilot",
    "🧪 A/B Test Planner", "📑 Executive Briefing", "📝 Summary"
]

ROLE_PERMISSIONS: dict[str, list[str]] = {
    # Canonical Enterprise Roles
    UserRole.ADMIN.value: _ADMIN_TABS,
    UserRole.DATA_SCIENTIST.value: _DS_TABS,
    UserRole.AUDITOR.value: _AUDITOR_TABS,
    UserRole.VIEWER.value: _VIEWER_TABS,
    # Legacy Persona Strings (maintained for compatibility)
    "Admin (Full Access)": _ADMIN_TABS,
    "Lead Data Scientist": _DS_TABS,
    "Business Analyst / Executive": _VIEWER_TABS,
    "Compliance & Privacy Officer": _AUDITOR_TABS,
}


def filter_tabs_for_user_role(role: str, all_tab_names: list[str]) -> list[str]:
    """Returns only the subset of dashboard tabs permitted for the given user role."""
    allowed = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS[UserRole.ADMIN.value])
    return [t for t in all_tab_names if t in allowed]


# ─── User & Security Context ──────────────────────────────────────────────────

class UserContext(BaseModel):
    """
    Authenticated User Context encapsulating Identity, Role, Tenant, and Permissions.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = "anonymous"
    role: str = UserRole.ADMIN.value
    tenant_id: str = DEFAULT_TENANT_ID
    org_id: str = DEFAULT_ORG_ID
    permissions: set[str] = Field(default_factory=set)

    def has_permission(self, permission: str | Permission) -> bool:
        perm_val = permission.value if isinstance(permission, Permission) else str(permission)
        if Permission.SYSTEM_ADMIN.value in self.permissions or "*" in self.permissions:
            return True
        return perm_val in self.permissions

    def can_access_tenant(self, target_tenant_id: str) -> bool:
        if self.is_admin() or target_tenant_id in ("*", "_all"):
            return True
        return self.tenant_id == target_tenant_id

    def is_admin(self) -> bool:
        return (
            self.role in (UserRole.ADMIN.value, "Admin (Full Access)")
            or Permission.SYSTEM_ADMIN.value in self.permissions
            or "*" in self.permissions
        )


def build_user_context(
    user_id: str = "anonymous",
    role: str = DEFAULT_ROLE,
    tenant_id: str = DEFAULT_TENANT_ID,
    org_id: str = DEFAULT_ORG_ID,
    extra_permissions: list[str] | None = None,
) -> UserContext:
    """Builds a validated UserContext with permissions inferred from the role."""
    # Normalize role
    normalized_role = role
    for canonical in UserRole:
        if role.lower().replace("_", "").replace(" ", "") == canonical.value.lower():
            normalized_role = canonical.value
            break

    perms = set(ROLE_ACTION_PERMISSIONS.get(normalized_role, ROLE_ACTION_PERMISSIONS[UserRole.VIEWER.value]))
    if extra_permissions:
        perms.update(extra_permissions)

    return UserContext(
        user_id=user_id,
        role=normalized_role,
        tenant_id=tenant_id,
        org_id=org_id,
        permissions=perms,
    )


# ─── Pure-Python Zero-Dependency RFC 7519 HMAC-SHA256 JWT Engine ──────────────

def _b64url_encode(data: bytes) -> str:
    """Base64 URL-safe encoding without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64 URL-safe decoding with auto-padded alignment."""
    rem = len(s) % 4
    if rem > 0:
        s += "=" * (4 - rem)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def create_access_token(
    user_id: str,
    role: str = UserRole.DATA_SCIENTIST.value,
    tenant_id: str = "default",
    org_id: str = "default",
    expires_minutes: int | None = None,
    secret: str | None = None,
) -> str:
    """
    Creates a signed RFC 7519 HMAC-SHA256 JSON Web Token (JWT).
    Pure Python standard library implementation with zero external dependencies.
    """
    signing_secret = (secret or JWT_SECRET).encode("utf-8")
    ttl = (expires_minutes if expires_minutes is not None else JWT_EXPIRE_MINUTES) * 60
    now = int(time.time())

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "role": role,
        "tenant_id": tenant_id,
        "org_id": org_id,
        "iat": now,
        "exp": now + int(ttl),
        "jti": uuid.uuid4().hex,
    }

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(signing_secret, signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str, secret: str | None = None) -> dict[str, Any]:
    """
    Decodes and cryptographically verifies an RFC 7519 HMAC-SHA256 JWT.
    Throws ValueError on malformed token, signature mismatch, or token expiration.
    """
    signing_secret = (secret or JWT_SECRET).encode("utf-8")
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format: token must contain exactly 3 dot-separated segments.")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(signing_secret, signing_input, hashlib.sha256).digest()
    expected_sig_b64 = _b64url_encode(expected_sig)

    # Constant-time comparison prevents timing attack vulnerabilities
    if not hmac.compare_digest(sig_b64, expected_sig_b64):
        raise ValueError("JWT signature verification failed: invalid signature.")

    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed JWT payload: {exc}") from exc

    # Validate expiration
    exp = payload.get("exp")
    if exp is not None and time.time() > float(exp):
        raise ValueError("JWT token has expired.")

    return payload


# ─── API Key Resolution ────────────────────────────────────────────────────────

_BUILTIN_API_KEYS: dict[str, dict[str, str]] = {
    "dia-admin-key": {"role": UserRole.ADMIN.value, "tenant_id": "default", "org_id": "default", "user_id": "system-admin"},
    "dia-ds-key": {"role": UserRole.DATA_SCIENTIST.value, "tenant_id": "default", "org_id": "default", "user_id": "team-datascientist"},
    "dia-auditor-key": {"role": UserRole.AUDITOR.value, "tenant_id": "default", "org_id": "default", "user_id": "audit-officer"},
    "dia-viewer-key": {"role": UserRole.VIEWER.value, "tenant_id": "default", "org_id": "default", "user_id": "read-viewer"},
}


def resolve_api_key(api_key: str) -> UserContext | None:
    """
    Resolves an API Key against DIA_API_KEYS configuration or default enterprise keys.
    Returns UserContext on match, None if unauthorized.
    When custom API keys are configured, fallback to built-in default keys is strictly disabled.
    """
    if not api_key:
        return None

    # Dynamically check dia.config if available
    try:
        import dia.config as dia_config
        active_config = getattr(dia_config, "API_KEYS_CONFIG", API_KEYS_CONFIG)
    except Exception:
        active_config = API_KEYS_CONFIG

    # 1. Check environment JSON configuration
    if active_config:
        try:
            keys_dict = json.loads(active_config)
            if isinstance(keys_dict, dict):
                if api_key in keys_dict:
                    entry = keys_dict[api_key]
                    return build_user_context(
                        user_id=entry.get("user_id", f"key-{api_key[:6]}"),
                        role=entry.get("role", UserRole.VIEWER.value),
                        tenant_id=entry.get("tenant_id", DEFAULT_TENANT_ID),
                        org_id=entry.get("org_id", DEFAULT_ORG_ID),
                    )
                # If custom keys are configured, do NOT fall through to built-in test keys
                return None
        except Exception:
            # Check comma-separated format: key:role:tenant:org
            matched_any_format = False
            for item in active_config.split(","):
                tokens = item.strip().split(":")
                if len(tokens) >= 2:
                    matched_any_format = True
                    if tokens[0] == api_key:
                        k_role = tokens[1]
                        k_tenant = tokens[2] if len(tokens) > 2 else DEFAULT_TENANT_ID
                        k_org = tokens[3] if len(tokens) > 3 else DEFAULT_ORG_ID
                        return build_user_context(
                            user_id=f"key-{api_key[:6]}",
                            role=k_role,
                            tenant_id=k_tenant,
                            org_id=k_org,
                        )
            if matched_any_format:
                return None

    # 2. Check built-in standard enterprise keys (fallback only when DIA_API_KEYS is unset)
    if api_key in _BUILTIN_API_KEYS:
        info = _BUILTIN_API_KEYS[api_key]
        return build_user_context(
            user_id=info["user_id"],
            role=info["role"],
            tenant_id=info["tenant_id"],
            org_id=info["org_id"],
        )

    return None


# ─── Enterprise Audit Event Logging ───────────────────────────────────────────

class AuditEvent(BaseModel):
    """Structured audit trail record for compliance and SOC2/GDPR governance."""
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    event_type: str
    user_id: str
    role: str
    tenant_id: str
    org_id: str
    resource: str
    status: str
    ip_address: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """
    Thread-safe in-memory circular buffer of compliance audit events.
    Enables instant queryability for security audits without external database dependencies.
    """

    def __init__(self, max_events: int = 2000):
        self._events: collections.deque[AuditEvent] = collections.deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record_event(
        self,
        event_type: str,
        user_context: UserContext | None,
        resource: str,
        status: str,
        details: dict[str, Any] | None = None,
        ip_address: str = "",
    ) -> AuditEvent:
        ctx = user_context or build_user_context()
        event = AuditEvent(
            event_type=event_type,
            user_id=ctx.user_id,
            role=ctx.role,
            tenant_id=ctx.tenant_id,
            org_id=ctx.org_id,
            resource=resource,
            status=status,
            ip_address=ip_address,
            details=details or {},
        )
        with self._lock:
            self._events.append(event)
        log.info(
            "AUDIT [%s] user=%s role=%s tenant=%s resource=%s status=%s",
            event.event_type, event.user_id, event.role, event.tenant_id, event.resource, event.status,
        )
        return event

    def query_events(
        self,
        tenant_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        with self._lock:
            events = list(self._events)
        if tenant_id and tenant_id not in ("*", "_all"):
            events = [e for e in events if e.tenant_id == tenant_id]
        if event_type:
            events = [e for e in events if e.event_type.lower() == event_type.lower()]
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]


# Global audit logger singleton
audit_logger = AuditLogger()
