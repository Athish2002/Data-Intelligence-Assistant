"""
dia/retrieval.py
─────────────────
RAG retrieval layer: turns dataset profiling output and an optional data
dictionary into small text chunks, embeds them, and answers similarity
search. Built once per session at ingest time.

Gate-and-fallback, matching goal_parser.py's pattern: cosine similarity
over sentence-transformers embeddings when available, degrading to a
zero-dependency lexical token-overlap scorer when it isn't. RAG without
embeddings still beats no RAG at all.

Corpus scale justifies brute-force NumPy over FAISS/Chroma: a session's
chunk count is one per column (tens) + a few readiness/target chunks +
dictionary terms (tens to low hundreds) — at most a few hundred rows. An
exhaustive cosine matmul over that is low-single-digit milliseconds, and
the index is rebuilt from scratch every ingest anyway, so there's no
index-reuse benefit an ANN structure would normally amortize.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import EMBEDDING_MODEL_NAME, RAG_TOP_K
from .data_profiler import infer_column_roles, profile_dataframe
from .utils import is_sentence_transformers_available
from .validators import detect_pii_columns

__all__ = [
    "RetrievalChunk",
    "ScoredChunk",
    "SimpleVectorIndex",
    "embed_texts",
    "build_session_index",
]


@dataclass
class RetrievalChunk:
    chunk_id: str
    source_type: str
    """One of: "column_profile" | "dataset_overview" | "readiness" | "target" | "dictionary"."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: RetrievalChunk
    score: float


# Process-lifetime model cache — a query embedding happens on every chat
# turn, so reloading the ~80MB model per message would make chat feel broken.
_MODEL_CACHE: dict[str, Any] = {}


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """L2-normalized (n, d) embedding matrix, or None if sentence-transformers is absent."""
    if not texts or not is_sentence_transformers_available():
        return None
    from sentence_transformers import SentenceTransformer

    model = _MODEL_CACHE.get(EMBEDDING_MODEL_NAME)
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _MODEL_CACHE[EMBEDDING_MODEL_NAME] = model
    return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


class SimpleVectorIndex:
    """Brute-force cosine-similarity index with a lexical fallback. See module docstring."""

    def __init__(self, chunks: Sequence[RetrievalChunk]) -> None:
        self.chunks: list[RetrievalChunk] = list(chunks)
        self._embeddings: np.ndarray | None = (
            embed_texts([c.text for c in self.chunks]) if self.chunks else None
        )

    @property
    def is_semantic(self) -> bool:
        return self._embeddings is not None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[ScoredChunk]:
        if not self.chunks or not query.strip():
            return []
        if self._embeddings is not None:
            q_vec = embed_texts([query])
            if q_vec is not None:
                sims = self._embeddings @ q_vec[0]
                order = np.argsort(-sims)[:top_k]
                return [ScoredChunk(self.chunks[i], float(sims[i])) for i in order]
        return self._search_lexical(query, top_k)

    def _search_lexical(self, query: str, top_k: int) -> list[ScoredChunk]:
        q_tok = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not q_tok:
            return []
        scored = []
        for c in self.chunks:
            c_tok = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            overlap = len(q_tok & c_tok)
            if overlap:
                scored.append(ScoredChunk(c, overlap / max(1, len(q_tok | c_tok))))
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]


# ─── Chunk builders ────────────────────────────────────────────────────────────

def _dataset_overview_chunk(df: pd.DataFrame, annotated_profile: pd.DataFrame) -> RetrievalChunk:
    n_rows, n_cols = df.shape
    role_counts = annotated_profile["inferred_role"].value_counts().to_dict()
    role_summary = ", ".join(f"{count} {role}" for role, count in role_counts.items())
    text = (
        f"Dataset overview: {n_rows:,} rows, {n_cols} columns. "
        f"Column roles: {role_summary}."
    )
    return RetrievalChunk(chunk_id="overview", source_type="dataset_overview", text=text)


def _chunks_from_profile(
    annotated_profile: pd.DataFrame, pii_columns: set[str]
) -> list[RetrievalChunk]:
    chunks = []
    for _, row in annotated_profile.iterrows():
        col = str(row["column"])
        redacted = col in pii_columns
        sample = "[redacted — possible PII]" if redacted else (row["sample_values"] or "(none)")
        text = (
            f"Column '{col}' (dtype {row['dtype']}). "
            f"Role: {row['inferred_role']} ({row['confidence_label']} confidence) — {row['explanation']} "
            f"Missing: {row['null_pct']}%. Unique: {row['unique_count']} ({row['unique_ratio']:.0%}). "
            f"Sample values: {sample}."
        )
        chunks.append(
            RetrievalChunk(
                chunk_id=f"column:{col}",
                source_type="column_profile",
                text=text,
                metadata={"column": col, "redacted": redacted},
            )
        )
    return chunks


def _chunks_from_readiness(readiness: dict[str, Any]) -> list[RetrievalChunk]:
    chunks = []
    verdict = readiness.get("verdict", "unknown")
    score = readiness.get("score", "unknown")
    summary_lines = readiness.get("summary_lines") or []
    overview_text = f"Data readiness verdict: {verdict} (score {score}/100). " + " ".join(summary_lines)
    chunks.append(RetrievalChunk(chunk_id="readiness:overview", source_type="readiness", text=overview_text))

    for key, label in (
        ("leakage_risk", "Columns flagged as possible target leakage risk"),
        ("missing_signals", "Missing-signal warnings"),
        ("useful_features", "Columns identified as useful features"),
    ):
        values = readiness.get(key) or []
        if values:
            chunks.append(
                RetrievalChunk(
                    chunk_id=f"readiness:{key}",
                    source_type="readiness",
                    text=f"{label}: {', '.join(str(v) for v in values)}.",
                )
            )
    return chunks


def _chunk_from_target(target_col: str, target_type_info: dict[str, Any]) -> RetrievalChunk:
    task_type = target_type_info.get("task_type", "unknown")
    reason = target_type_info.get("reason", "")
    n_classes = target_type_info.get("n_classes")
    text = f"Target column '{target_col}': task type is {task_type}. {reason}"
    if n_classes is not None:
        text += f" ({n_classes} classes.)"
    return RetrievalChunk(chunk_id="target", source_type="target", text=text)


def _chunks_from_dictionary(entries: list[dict[str, str]]) -> list[RetrievalChunk]:
    # No redaction here — entries are already PII-gated before reaching the
    # dictionary store (dia.ingestion.data_dictionary + dia.dictionary_store).
    chunks = []
    for e in entries:
        term = str(e.get("term", "")).strip()
        definition = str(e.get("definition", "")).strip()
        if not term:
            continue
        chunks.append(
            RetrievalChunk(
                chunk_id=f"dict:{term.lower()}",
                source_type="dictionary",
                text=f"Business term '{term}': {definition}",
                metadata={"term": term},
            )
        )
    return chunks


def build_session_index(
    df: pd.DataFrame,
    *,
    annotated_profile: pd.DataFrame | None = None,
    readiness: dict[str, Any] | None = None,
    target_col: str | None = None,
    target_type_info: dict[str, Any] | None = None,
    pii_columns: list[str] | None = None,
    dictionary_entries: list[dict[str, str]] | None = None,
) -> SimpleVectorIndex:
    """
    Build a session's retrieval index from whatever pipeline artifacts the
    caller already has.

    - annotated_profile omitted -> computed fresh via profile_dataframe +
      infer_column_roles (cheap, safe to call at raw-ingest time).
    - readiness / target_type_info omitted -> those chunk types are simply
      skipped (true before a goal/target is known); call again later once
      they're available to upgrade an already-cached index in place.
    - pii_columns omitted -> falls back to the cheap header-only
      dia.validators.detect_pii_columns() as a redaction floor. A caller
      that already ran dia.compliance.scan_dataset_privacy() should pass
      its detected columns instead for stronger coverage.
    - dictionary_entries -> list of {"term", "definition"} dicts, e.g. from
      dia.dictionary_store.load_entries().

    Never raises: any missing optional input degrades gracefully rather
    than blocking index construction.
    """
    if annotated_profile is None:
        annotated_profile = infer_column_roles(df, profile_dataframe(df))

    pii_set = (
        set(pii_columns) if pii_columns is not None else set(detect_pii_columns(list(df.columns)))
    )

    chunks: list[RetrievalChunk] = [_dataset_overview_chunk(df, annotated_profile)]
    chunks.extend(_chunks_from_profile(annotated_profile, pii_set))
    if readiness:
        chunks.extend(_chunks_from_readiness(readiness))
    if target_col and target_type_info:
        chunks.append(_chunk_from_target(target_col, target_type_info))
    if dictionary_entries:
        chunks.extend(_chunks_from_dictionary(dictionary_entries))

    return SimpleVectorIndex(chunks)
