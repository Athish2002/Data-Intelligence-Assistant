"""
dia/storage_provider.py
───────────────────────
Pluggable Storage Abstraction Layer for Data Intelligence Assistant.
Decouples all session data, raw uploads, model artifacts, and caching
from hardcoded directories into an extensible storage interface.

Supports:
1. LocalStorageBackend: Local disks, physical volume mounts, NFS/Ceph, and Kubernetes PVCs.
2. S3CompatibleStorageBackend: AWS S3, MinIO, GCP Cloud Storage (via S3 API), and Azure Blob (via S3 gateway).
"""

from __future__ import annotations

import abc
import io
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from dia.config import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_PREFIX,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
    STORAGE_PATH,
    STORAGE_TYPE,
)

log = logging.getLogger("dia.storage")


class StorageBackend(abc.ABC):
    """
    Abstract Protocol defining required storage operations across all storage providers.
    All operations are strictly tenant-namespaced to preserve multi-tenant isolation.
    """

    @abc.abstractmethod
    def save(
        self,
        path: str,
        data: bytes | str | BinaryIO,
        tenant_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Saves binary or text data at the specified relative path for a tenant.
        Returns a persistent storage URI or path identifier.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, path: str, tenant_id: str = "default") -> bytes:
        """
        Reads and returns the raw bytes for the specified path and tenant.
        Raises FileNotFoundError if the file does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, path: str, tenant_id: str = "default") -> bool:
        """
        Deletes the file at the specified path and tenant.
        Returns True if deleted, False if file did not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, path: str, tenant_id: str = "default") -> bool:
        """Returns True if the file exists under the tenant's namespace."""
        raise NotImplementedError

    @abc.abstractmethod
    def list_files(self, prefix: str = "", tenant_id: str = "default") -> list[str]:
        """
        Returns a list of relative file paths matching the prefix within the tenant's namespace.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_url(self, path: str, tenant_id: str = "default") -> str:
        """Returns a canonical URI or locator string for the stored resource."""
        raise NotImplementedError

    @abc.abstractmethod
    def health(self) -> dict[str, Any]:
        """
        Performs an active read/write liveness probe and returns storage health status.
        """
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """
    High-performance, path-traversal-hardened storage backend for local filesystems,
    separate physical disks, host mounts, and Kubernetes PersistentVolumeClaims (PVC).
    """

    def __init__(self, root_path: str | Path | None = None):
        self.root_path = Path(root_path or STORAGE_PATH).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _resolve_tenant_path(self, path: str, tenant_id: str) -> Path:
        """
        Sanitizes and resolves a relative path strictly within the tenant's subdirectory.
        Throws ValueError if path is empty or directory traversal (e.g. '../') is attempted.
        """
        clean_tenant = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_")).strip() or "default"
        clean_path = path.replace("\\", "/").strip().lstrip("/")
        if not clean_path:
            raise ValueError("Storage path cannot be empty.")
        tenant_root = (self.root_path / clean_tenant).resolve()
        target_path = (tenant_root / clean_path).resolve()

        # Strict sandbox invariant: target must be inside tenant_root and not tenant_root itself
        if target_path == tenant_root or not target_path.is_relative_to(tenant_root):
            raise ValueError(f"Directory traversal prohibited: '{path}' escapes tenant root '{clean_tenant}'.")
        return target_path

    def save(
        self,
        path: str,
        data: bytes | str | BinaryIO,
        tenant_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        target = self._resolve_tenant_path(path, tenant_id)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Convert input data to raw bytes
        if isinstance(data, str):
            content = data.encode("utf-8")
        elif isinstance(data, (io.IOBase, BinaryIO)):
            content = data.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
        else:
            content = bytes(data)

        # Atomic write pattern: write to sibling temp file then rename
        temp_file = target.parent / f".tmp_{uuid.uuid4().hex}"
        try:
            with open(temp_file, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, target)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise

        return str(target)

    def read(self, path: str, tenant_id: str = "default") -> bytes:
        target = self._resolve_tenant_path(path, tenant_id)
        if not target.is_file():
            raise FileNotFoundError(f"Storage path '{path}' not found for tenant '{tenant_id}'.")
        with open(target, "rb") as f:
            return f.read()

    def delete(self, path: str, tenant_id: str = "default") -> bool:
        clean_tenant = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_")).strip() or "default"
        tenant_root = (self.root_path / clean_tenant).resolve()
        target = self._resolve_tenant_path(path, tenant_id)
        if target.is_file():
            target.unlink()
            # Clean up empty parent directories up to tenant root
            parent = target.parent
            while parent != tenant_root and parent != self.root_path and parent.is_relative_to(tenant_root) and parent.exists():
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break
            return True
        return False

    def exists(self, path: str, tenant_id: str = "default") -> bool:
        try:
            target = self._resolve_tenant_path(path, tenant_id)
            return target.is_file()
        except ValueError:
            return False

    def list_files(self, prefix: str = "", tenant_id: str = "default") -> list[str]:
        clean_tenant = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_")).strip() or "default"
        tenant_root = (self.root_path / clean_tenant).resolve()
        if not tenant_root.is_dir():
            return []

        clean_prefix = prefix.replace("\\", "/").lstrip("/")
        matches: list[str] = []
        for root, _, files in os.walk(tenant_root):
            for file_name in files:
                full_path = Path(root) / file_name
                rel_path = full_path.relative_to(tenant_root).as_posix()
                if rel_path.startswith(clean_prefix):
                    matches.append(rel_path)
        return sorted(matches)

    def get_url(self, path: str, tenant_id: str = "default") -> str:
        target = self._resolve_tenant_path(path, tenant_id)
        return target.as_uri()

    def health(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        probe_id = uuid.uuid4().hex[:8]
        probe_path = f".probe/liveness_{probe_id}.tmp"
        try:
            saved_loc = self.save(probe_path, b"probe_ok", tenant_id="_system")
            read_back = self.read(probe_path, tenant_id="_system")
            self.delete(probe_path, tenant_id="_system")
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            free_gb = 0.0
            total_gb = 0.0
            try:
                usage = shutil.disk_usage(str(self.root_path))
                free_gb = round(usage.free / (1024 ** 3), 2)
                total_gb = round(usage.total / (1024 ** 3), 2)
            except Exception:
                pass

            is_ok = read_back == b"probe_ok"
            return {
                "status": "healthy" if is_ok else "unhealthy",
                "backend": "local",
                "path": str(self.root_path),
                "writable": is_ok,
                "latency_ms": latency_ms,
                "free_disk_gb": free_gb,
                "total_disk_gb": total_gb,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "backend": "local",
                "path": str(self.root_path),
                "writable": False,
                "error": str(exc),
            }


class S3CompatibleStorageBackend(StorageBackend):
    """
    S3-Compatible cloud storage backend supporting AWS S3, MinIO, GCP Cloud Storage (S3 API),
    and Azure Blob (via S3 gateway). Includes built-in mock/emulation mode for zero-dependency testing.
    """

    def __init__(
        self,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        prefix: str | None = None,
        s3_client: Any | None = None,
        mock: bool = False,
    ):
        self.bucket = bucket or S3_BUCKET
        self.endpoint_url = endpoint_url or S3_ENDPOINT_URL
        self.region = region or S3_REGION
        self.access_key_id = access_key_id or S3_ACCESS_KEY_ID
        self.secret_access_key = secret_access_key or S3_SECRET_ACCESS_KEY
        self.prefix = (prefix if prefix is not None else S3_PREFIX).strip("/ ")
        self._lock = threading.Lock()

        self.is_mock = mock or (os.getenv("DIA_STORAGE_MOCK_S3", "false").lower() in ("true", "1", "yes"))
        self._mock_objects: dict[str, bytes] = {}

        if s3_client is not None:
            self.client = s3_client
            self.is_mock = False
        elif not self.is_mock:
            try:
                import boto3  # type: ignore
                client_kwargs: dict[str, Any] = {
                    "region_name": self.region,
                }
                if self.endpoint_url:
                    client_kwargs["endpoint_url"] = self.endpoint_url
                if self.access_key_id and self.secret_access_key:
                    client_kwargs["aws_access_key_id"] = self.access_key_id
                    client_kwargs["aws_secret_access_key"] = self.secret_access_key
                self.client = boto3.client("s3", **client_kwargs)
            except (ImportError, Exception) as exc:
                log.warning(
                    "boto3 initialization not possible (%s). S3CompatibleStorageBackend falling back to emulated in-memory mock mode.",
                    exc,
                )
                self.is_mock = True
                self.client = None
        else:
            self.client = None

    def _make_key(self, path: str, tenant_id: str) -> str:
        clean_tenant = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_")).strip() or "default"
        clean_path = path.replace("\\", "/").strip().lstrip("/")
        if not clean_path:
            raise ValueError("Storage path cannot be empty.")

        # Disallow directory traversal escaping tenant namespace
        parts = [p for p in clean_path.split("/") if p and p != "."]
        depth = 0
        for p in parts:
            if p == "..":
                depth -= 1
                if depth < 0:
                    raise ValueError(f"Directory traversal prohibited: '{path}' escapes tenant namespace '{clean_tenant}'.")
            else:
                depth += 1

        import posixpath
        normalized = posixpath.normpath(clean_path)
        if normalized == ".." or normalized.startswith("../"):
            raise ValueError(f"Directory traversal prohibited: '{path}' escapes tenant namespace '{clean_tenant}'.")

        if self.prefix:
            return f"{self.prefix}/{clean_tenant}/{normalized}"
        return f"{clean_tenant}/{normalized}"

    def save(
        self,
        path: str,
        data: bytes | str | BinaryIO,
        tenant_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        key = self._make_key(path, tenant_id)
        if isinstance(data, str):
            content = data.encode("utf-8")
        elif isinstance(data, (io.IOBase, BinaryIO)):
            content = data.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
        else:
            content = bytes(data)

        if self.is_mock or self.client is None:
            with self._lock:
                self._mock_objects[key] = content
            return f"s3://{self.bucket}/{key}"

        extra_args: dict[str, Any] = {}
        if metadata:
            extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            **extra_args,
        )
        return f"s3://{self.bucket}/{key}"

    def read(self, path: str, tenant_id: str = "default") -> bytes:
        key = self._make_key(path, tenant_id)
        if self.is_mock or self.client is None:
            with self._lock:
                if key not in self._mock_objects:
                    raise FileNotFoundError(f"S3 object '{key}' not found in bucket '{self.bucket}'.")
                return self._mock_objects[key]

        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            raise FileNotFoundError(f"S3 object '{key}' read error: {e}") from e

    def delete(self, path: str, tenant_id: str = "default") -> bool:
        key = self._make_key(path, tenant_id)
        if self.is_mock or self.client is None:
            with self._lock:
                if key in self._mock_objects:
                    del self._mock_objects[key]
                    return True
                return False

        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def exists(self, path: str, tenant_id: str = "default") -> bool:
        key = self._make_key(path, tenant_id)
        if self.is_mock or self.client is None:
            with self._lock:
                return key in self._mock_objects

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "", tenant_id: str = "default") -> list[str]:
        clean_tenant = "".join(c for c in tenant_id if c.isalnum() or c in ("-", "_")).strip() or "default"
        clean_prefix = prefix.replace("\\", "/").strip().lstrip("/")
        tenant_prefix = f"{self.prefix}/{clean_tenant}/" if self.prefix else f"{clean_tenant}/"
        target_prefix = f"{tenant_prefix}{clean_prefix}"

        if self.is_mock or self.client is None:
            with self._lock:
                results: list[str] = []
                for k in self._mock_objects.keys():
                    if k.startswith(target_prefix):
                        rel = k[len(tenant_prefix):].lstrip("/")
                        results.append(rel)
                return sorted(results)

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            results = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=target_prefix):
                for obj in page.get("Contents", []):
                    k = obj["Key"]
                    rel = k[len(tenant_prefix):].lstrip("/")
                    results.append(rel)
            return sorted(results)
        except Exception as e:
            log.warning("S3 list_objects error: %s", e)
            return []

    def get_url(self, path: str, tenant_id: str = "default") -> str:
        key = self._make_key(path, tenant_id)
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    def health(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        probe_id = uuid.uuid4().hex[:8]
        probe_path = f".probe/health_{probe_id}.tmp"
        try:
            self.save(probe_path, b"s3_probe_ok", tenant_id="_system")
            read_back = self.read(probe_path, tenant_id="_system")
            self.delete(probe_path, tenant_id="_system")
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            is_ok = read_back == b"s3_probe_ok"
            return {
                "status": "healthy" if is_ok else "unhealthy",
                "backend": "s3",
                "bucket": self.bucket,
                "region": self.region,
                "endpoint": self.endpoint_url or "aws-standard",
                "writable": is_ok,
                "latency_ms": latency_ms,
                "is_mock": self.is_mock,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "backend": "s3",
                "bucket": self.bucket,
                "region": self.region,
                "writable": False,
                "error": str(exc),
                "is_mock": self.is_mock,
            }


# ─── Singleton Storage Management & Factory ───────────────────────────────────

_STORAGE_INSTANCE: StorageBackend | None = None
_STORAGE_LOCK = threading.Lock()


def get_storage_backend() -> StorageBackend:
    """
    Returns the configured StorageBackend singleton.
    Instantiates LocalStorageBackend or S3CompatibleStorageBackend based on DIA_STORAGE_TYPE.
    """
    global _STORAGE_INSTANCE
    if _STORAGE_INSTANCE is None:
        with _STORAGE_LOCK:
            if _STORAGE_INSTANCE is None:
                backend_type = STORAGE_TYPE.lower().strip()
                if backend_type in ("s3", "s3compatible", "minio", "cloud"):
                    _STORAGE_INSTANCE = S3CompatibleStorageBackend()
                else:
                    _STORAGE_INSTANCE = LocalStorageBackend()
    return _STORAGE_INSTANCE


def set_storage_backend(backend: StorageBackend) -> None:
    """Explicitly sets or overrides the global storage backend singleton (useful for testing)."""
    global _STORAGE_INSTANCE
    with _STORAGE_LOCK:
        _STORAGE_INSTANCE = backend


def reset_storage_backend() -> None:
    """Resets the singleton storage backend."""
    global _STORAGE_INSTANCE
    with _STORAGE_LOCK:
        _STORAGE_INSTANCE = None
