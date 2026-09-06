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

            # If key exists, update and move to end
            if session_id in self._store:
                self._store[session_id] = value
                self._store.move_to_end(session_id)
                self._meta[session_id]["last_accessed"] = time.time()
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

            self._store[session_id] = value
            self._meta[session_id] = {
                "created_at": now,
                "last_accessed": now,
                "access_count": 1,
                "goal": str(value.get("goal", "Predict target")),
                "n_rows": n_rows,
                "n_cols": n_cols,
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

    def get_sessions_summary(self) -> list[dict[str, Any]]:
        """Returns structured metadata for all active sessions for telemetry reporting."""
        with self._lock:
            self._cleanup_expired_locked()
            now = time.time()
            summaries = []
            for sid, meta in self._meta.items():
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
