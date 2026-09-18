"""
dia/session_manager.py
──────────────────────
High-Performance, Thread-Safe Bounded LRU Session Store with Monotonic TTL Expiration
and Resource Deallocation for the Data Intelligence Assistant backend.

Prevents unbounded memory growth by:
1. Enforcing an LRU bulkhead capacity limit (default: 5 concurrent sessions).
2. Evicting sessions exceeding a time-to-live threshold (default: 30 minutes).
3. Explicitly dereferencing DataFrames, ML models, preprocessors, and RAG indices on eviction.
4. Invoking Python's garbage collector (`gc.collect()`) upon session eviction.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger("dia.session_manager")


class BoundedSessionStore:
    """
    Thread-safe LRU in-memory store for user analytical sessions with TTL pruning.
    Emulates standard dictionary access while enforcing memory constraints.
    """

    def __init__(self, max_sessions: int = 5, ttl_seconds: int = 1800):
        self.max_sessions = max(1, max_sessions)
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _cleanup_expired_locked(self) -> int:
        """Purges sessions older than ttl_seconds. Assumes lock is held."""
        now = time.time()
        expired_keys = [
            sid for sid, meta in self._meta.items()
            if (now - meta.get("last_accessed", now)) > self.ttl_seconds
        ]
        for sid in expired_keys:
            log.info("Session %s expired (idle for >%ds). Evicting from memory.", sid, self.ttl_seconds)
            self._evict_locked(sid)
        return len(expired_keys)

    def _evict_locked(self, session_id: str) -> None:
        """Explicitly deallocates memory associated with session_id. Assumes lock is held."""
        sess = self._store.pop(session_id, None)
        self._meta.pop(session_id, None)

        if sess is not None:
            try:
                # Break circular references and clear heavyweight buffers
                if "df" in sess:
                    del sess["df"]
                if "rag_index" in sess:
                    del sess["rag_index"]
                if "pipeline_result" in sess:
                    pipe = sess.pop("pipeline_result", None)
                    if pipe and isinstance(pipe, dict):
                        pipe.clear()
                sess.clear()
            except Exception as e:
                log.warning("Exception while clearing session %s: %s", session_id, e)
            finally:
                gc.collect()

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            self._cleanup_expired_locked()
            return session_id in self._store

    def __len__(self) -> int:
        with self._lock:
            self._cleanup_expired_locked()
            return len(self._store)

    def __getitem__(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._cleanup_expired_locked()
            if session_id not in self._store:
                raise KeyError(f"Session '{session_id}' not found or expired.")
            # Move to end to mark as recently used (LRU policy)
            self._store.move_to_end(session_id)
            if session_id in self._meta:
                self._meta[session_id]["last_accessed"] = time.time()
                self._meta[session_id]["access_count"] += 1
            return self._store[session_id]

    def __setitem__(self, session_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._cleanup_expired_locked()

            incoming_tenant = str(value.get("tenant_id") or "default")
            incoming_org = str(value.get("org_id") or "default")

            # If key exists, ensure tenant isolation before updating
            if session_id in self._store:
                existing_tenant = self._meta.get(session_id, {}).get("tenant_id", "default")
                if existing_tenant != incoming_tenant and incoming_tenant not in ("*", "_all"):
                    raise PermissionError(
                        f"Tenant '{incoming_tenant}' is not permitted to overwrite session '{session_id}' owned by '{existing_tenant}'."
                    )
                self._store[session_id] = value
                self._store.move_to_end(session_id)
                now = time.time()
                self._meta[session_id]["last_accessed"] = now
                df = value.get("df")
                if df is not None and hasattr(df, "shape"):
                    self._meta[session_id]["n_rows"] = int(df.shape[0])
                    self._meta[session_id]["n_cols"] = int(df.shape[1])
                if "goal" in value:
                    self._meta[session_id]["goal"] = str(value.get("goal", "Predict target"))
                return

            # If at capacity, evict least-recently-used session (first item)
            while len(self._store) >= self.max_sessions:
                oldest_sid, _ = next(iter(self._store.items()))
                log.info("Session capacity reached (%d). Evicting LRU session %s.", self.max_sessions, oldest_sid)
                self._evict_locked(oldest_sid)

            now = time.time()
            df = value.get("df")
            n_rows = int(df.shape[0]) if df is not None and hasattr(df, "shape") else 0
            n_cols = int(df.shape[1]) if df is not None and hasattr(df, "shape") else 0
            tenant_id = str(value.get("tenant_id") or "default")
            org_id = str(value.get("org_id") or "default")

            self._store[session_id] = value
            self._meta[session_id] = {
                "created_at": now,
                "last_accessed": now,
                "access_count": 1,
                "goal": str(value.get("goal", "Predict target")),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "tenant_id": tenant_id,
                "org_id": org_id,
            }

    def get(self, session_id: str, default: Any = None) -> Any:
        with self._lock:
            try:
                return self[session_id]
            except KeyError:
                return default

    def pop(self, session_id: str, default: Any = None) -> Any:
        with self._lock:
            self._cleanup_expired_locked()
            if session_id not in self._store:
                return default
            self._meta.pop(session_id, None)
            return self._store.pop(session_id)

    def can_access_session(self, session_id: str, tenant_id: str = "default", is_admin: bool = False) -> bool:
        """Verifies if tenant is authorized to access the given session."""
        with self._lock:
            self._cleanup_expired_locked()
            if session_id not in self._store:
                return False
            if is_admin or tenant_id in ("*", "_all"):
                return True
            return self._meta.get(session_id, {}).get("tenant_id", "default") == tenant_id

    def get_for_tenant(self, session_id: str, tenant_id: str = "default", is_admin: bool = False) -> dict[str, Any]:
        """Retrieves session ensuring tenant isolation. Raises PermissionError on tenant breach."""
        with self._lock:
            sess = self[session_id]
            sess_tenant = self._meta.get(session_id, {}).get("tenant_id", "default")
            if not is_admin and tenant_id not in ("*", "_all") and sess_tenant != tenant_id:
                raise PermissionError(f"Tenant '{tenant_id}' is not authorized to access session '{session_id}' (owner: '{sess_tenant}').")
            return sess

    def pop_for_tenant(self, session_id: str, tenant_id: str = "default", is_admin: bool = False) -> dict[str, Any]:
        """Evicts session ensuring tenant isolation. Raises PermissionError on tenant breach."""
        with self._lock:
            self._cleanup_expired_locked()
            if session_id not in self._store:
                raise KeyError(f"Session '{session_id}' not found.")
            sess_tenant = self._meta.get(session_id, {}).get("tenant_id", "default")
            if not is_admin and tenant_id not in ("*", "_all") and sess_tenant != tenant_id:
                raise PermissionError(f"Tenant '{tenant_id}' is not authorized to delete session '{session_id}' (owner: '{sess_tenant}').")
            return self.pop(session_id)

    def get_active_tenants(self) -> set[str]:
        """Returns the set of all unique tenant IDs with active in-memory sessions."""
        with self._lock:
            self._cleanup_expired_locked()
            return {meta.get("tenant_id", "default") for meta in self._meta.values()}

    def clear(self) -> None:
        """Purges all sessions and sweeps memory."""
        with self._lock:
            keys = list(self._store.keys())
            for sid in keys:
                self._evict_locked(sid)
            self._store.clear()
            self._meta.clear()
            try:
                from dia.retrieval import clear_retrieval_model_cache
                clear_retrieval_model_cache()
            except Exception:
                pass
            gc.collect()

    def get_sessions_summary(self, tenant_id: str | None = None, is_admin: bool = False) -> list[dict[str, Any]]:
        """Returns structured metadata for active sessions, optionally filtered by tenant."""
        with self._lock:
            self._cleanup_expired_locked()
            now = time.time()
            summaries = []
            for sid, meta in self._meta.items():
                sess_tenant = meta.get("tenant_id", "default")
                if tenant_id and not is_admin and tenant_id not in ("*", "_all"):
                    if sess_tenant != tenant_id:
                        continue

                sess = self._store.get(sid, {})
                df = sess.get("df")
                est_bytes = 0
                if df is not None and hasattr(df, "memory_usage"):
                    try:
                        est_bytes = int(df.memory_usage(deep=True).sum())
                    except Exception:
                        est_bytes = sys.getsizeof(df)

                has_pipeline = sess.get("pipeline_result") is not None
                best_model = None
                if has_pipeline and isinstance(sess.get("pipeline_result"), dict):
                    best_model = sess["pipeline_result"].get("best_model_label")

                summaries.append({
                    "session_id": sid,
                    "goal": meta.get("goal", ""),
                    "n_rows": meta.get("n_rows", 0),
                    "n_cols": meta.get("n_cols", 0),
                    "memory_mb": round(est_bytes / (1024 * 1024), 2),
                    "has_pipeline": has_pipeline,
                    "best_model": best_model,
                    "age_seconds": round(now - meta.get("created_at", now), 1),
                    "idle_seconds": round(now - meta.get("last_accessed", now), 1),
                    "access_count": meta.get("access_count", 1),
                    "tenant_id": sess_tenant,
                    "org_id": meta.get("org_id", "default"),
                })
            return summaries


def trigger_system_garbage_collection() -> None:
    """Purges cached retrieval models and triggers a garbage collection cycle."""
    try:
        from dia.retrieval import clear_retrieval_model_cache
        clear_retrieval_model_cache()
    except Exception:
        pass
    gc.collect()
