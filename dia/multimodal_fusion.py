"""
dia/multimodal_fusion.py
────────────────────────
Multi-Modal Tabular-Text Semantic Fusion Engine.

Enterprise tabular datasets frequently contain rich unstructured free-text columns
(e.g., customer comments, support ticket logs, transaction descriptions, clinical notes).
This module provides an end-to-end multi-modal pipeline:
1. Heuristic Free-Text Column Detection: Distinguishes natural free-text fields from
   low-cardinality categoricals, fixed-format codes, and UUIDs via word count, string
   length variance, token distribution, and uniqueness ratio.
2. Dense Semantic Vector Extraction: Generates compact, noise-reduced dense semantic
   embeddings via Scikit-Learn TfidfVectorizer + TruncatedSVD (Latent Semantic Analysis).
3. Interpretable Keyword Attribution: Extracts top n-gram keywords per latent SVD component
   to maintain human explainability and enable downstream TreeSHAP attribution.
4. Clean Tabular Fusion: Horizontally concatenates dense semantic representations
   with structured numerical and categorical features for unified AutoML training.

Defensive safeguards:
- Handles empty strings, nulls/NaNs, mixed data types, zero text columns, and single-row tables.
- Resource hygiene: Deallocates intermediate sparse matrices and triggers explicit gc.collect().
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Data Contracts & Models ──────────────────────────────────────────────────

class MultimodalFusionResult(BaseModel):
    """
    Standard result schema for multi-modal tabular-text semantic fusion.
    Compatible with Pydantic v2 and FastAPI serialization.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fused_df: Any  # pd.DataFrame
    text_columns_detected: List[str] = Field(default_factory=list)
    dense_feature_names: List[str] = Field(default_factory=list)
    top_keywords_per_component: Dict[str, List[str]] = Field(default_factory=dict)
    explained_variance_ratio: List[float] = Field(default_factory=list)
    fusion_manifest: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class TextModalityMeta:
    """Per-column metadata and explainability attribution."""
    column_name: str
    sample_count: int
    avg_words_per_sample: float
    n_components: int
    semantic_feature_names: list[str]
    component_keyword_attributions: dict[str, list[str]]


@dataclass
class FusionResult:
    """Dataclass representation of the multi-modal fusion output."""
    fused_df: pd.DataFrame
    detected_text_columns: list[str]
    new_semantic_features: list[str]
    modality_meta: list[TextModalityMeta]
    fusion_manifest: dict[str, Any]


# ─── Text Column Detection Heuristics ─────────────────────────────────────────

def detect_text_columns(
    df: pd.DataFrame,
    min_avg_words: float = 3.0,
    min_unique_ratio: float = 0.3,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Heuristically identifies unstructured natural language free-text columns in a DataFrame.

    Criteria evaluated per candidate column:
    1. Object, string, or category dtype.
    2. Average token/word count >= min_avg_words (e.g. >= 3.0 words per non-empty record).
    3. Uniqueness ratio (nunique / non-null rows) >= min_unique_ratio (free-text exhibits high entropy).
    4. String character length variance >= 15.0 (natural text has variable sentence lengths, unlike fixed-length codes).
    5. Not parseable as pure timestamps, numeric values, or booleans.

    Args:
        df: Input pandas DataFrame.
        min_avg_words: Minimum average whitespace/word token count per row (default 3.0).
        min_unique_ratio: Minimum ratio of distinct text values to total valid rows (default 0.3).
        exclude_cols: Optional list of column names to bypass (e.g., target or ID).

    Returns:
        List of identified free-text column names in df.
    """
    if df is None or df.empty or len(df) == 0:
        return []

    excluded = set(exclude_cols or [])
    text_cols: List[str] = []

    for col in df.columns:
        if col in excluded:
            continue

        series = df[col]

        # Only evaluate string/object/categorical columns
        is_str_like = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        )
        if not is_str_like:
            continue

        # Filter non-null strings
        valid_entries = series.dropna().astype(str).str.strip()
        valid_entries = valid_entries[valid_entries != ""]
        n_valid = len(valid_entries)

        if n_valid < 3:
            continue

        # 1. Uniqueness ratio check
        n_unique = valid_entries.nunique()
        u_ratio = n_unique / float(n_valid)
        if u_ratio < min_unique_ratio:
            continue

        # 2. Token / word count distribution
        # Sample up to 500 rows for high performance on large datasets
        sample = valid_entries.iloc[:500]
        word_counts = [len(re.findall(r"\b\w+\b", s)) for s in sample]
        avg_words = float(np.mean(word_counts)) if word_counts else 0.0

        if avg_words < min_avg_words:
            continue

        # 3. String character length & variance check
        char_lengths = [len(s) for s in sample]
        avg_length = float(np.mean(char_lengths)) if char_lengths else 0.0
        length_variance = float(np.var(char_lengths)) if char_lengths else 0.0

        # Natural text usually averages >= 15 characters with non-trivial length variance
        if avg_length < 12.0 or length_variance < 15.0:
            continue

        # 4. Check that column is not a pure ISO timestamp or numeric string
        try:
            head_sample = sample.head(10)
            parsed_dates = pd.to_datetime(head_sample, errors="coerce")
            if parsed_dates.notna().sum() >= len(head_sample) * 0.8:
                # Column is primarily timestamps/dates
                continue
        except Exception:
            pass

        text_cols.append(col)

    return text_cols


# ─── Multi-Modal Tabular-Text Semantic Fusion Engine ──────────────────────────

class MultiModalFusionEngine:
    """
    Production-grade Multi-Modal Tabular-Text Semantic Fusion Engine.

    Transforms unstructured text columns into dense semantic vector representations
    using Scikit-Learn TfidfVectorizer + TruncatedSVD and fuses them with structured data.
    Supports both offline training (fit_transform) and low-latency inference (transform).
    """

    def __init__(
        self,
        n_components: int = 8,
        max_features: int = 5000,
        drop_raw_text: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the multi-modal fusion engine.

        Args:
            n_components: Desired number of dense latent semantic dimensions per text column (default 8).
            max_features: Maximum vocabulary size for TF-IDF tokenization (default 5000).
            drop_raw_text: Whether to drop the raw text columns from the fused DataFrame (default False).
            random_state: Random state for deterministic TruncatedSVD decomposition.
        """
        self.n_components = int(n_components)
        self.max_features = int(max_features)
        self.drop_raw_text = bool(drop_raw_text)
        self.random_state = int(random_state)

        # State stored during fit
        self.text_columns: List[str] = []
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.svd_models: Dict[str, TruncatedSVD] = {}
        self.dense_feature_names: List[str] = []
        self.component_keywords: Dict[str, List[str]] = {}
        self.explained_variance: Dict[str, List[float]] = {}
        self.modality_meta: List[TextModalityMeta] = []

    def detect_text_columns(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """Detects unstructured free-text columns in df."""
        return detect_text_columns(df, exclude_cols=exclude_cols)

    def fit(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ) -> MultiModalFusionEngine:
        """
        Fits TF-IDF vectorizers and TruncatedSVD transformers on text columns.

        Args:
            df: Input DataFrame.
            text_columns: Optional explicit list of text columns. If None, auto-detected.
            target_col: Optional target column to exclude from detection.

        Returns:
            self
        """
        if df is None or df.empty:
            return self

        # Determine target text columns
        if text_columns is None:
            exclude = [target_col] if target_col else []
            self.text_columns = detect_text_columns(df, exclude_cols=exclude)
        else:
            self.text_columns = [c for c in text_columns if c in df.columns]

        self.vectorizers.clear()
        self.svd_models.clear()
        self.dense_feature_names.clear()
        self.component_keywords.clear()
        self.explained_variance.clear()
        self.modality_meta.clear()

        n_rows = len(df)

        for col in self.text_columns:
            # Defensive fillna and string conversion
            text_series = df[col].fillna("").astype(str).str.strip()

            # Initialize TF-IDF Vectorizer
            # Use English stopwords, unigram + bigram, defensive token pattern
            tfidf = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            )

            try:
                tfidf_matrix = tfidf.fit_transform(text_series)
            except ValueError:
                # If stop_words removes all tokens or vocabulary empty, retry without stop_words
                tfidf = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words=None,
                    ngram_range=(1, 1),
                    min_df=1,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                )
                try:
                    tfidf_matrix = tfidf.fit_transform(text_series)
                except Exception:
                    tfidf_matrix = None

            n_vocab = tfidf_matrix.shape[1] if tfidf_matrix is not None else 0

            if tfidf_matrix is None or n_vocab == 0:
                # Skip or create dummy zero transformer
                self.vectorizers[col] = tfidf
                continue

            # Determine number of SVD components bounded by samples and vocabulary
            k_eff = min(self.n_components, n_vocab, max(1, n_rows - 1))
            if k_eff < 1:
                k_eff = 1

            svd = TruncatedSVD(n_components=k_eff, random_state=self.random_state)
            svd.fit(tfidf_matrix)

            self.vectorizers[col] = tfidf
            self.svd_models[col] = svd

            # Extract feature names & keywords per latent component
            feature_names = tfidf.get_feature_names_out()
            col_feature_names: List[str] = []
            col_keywords: Dict[str, List[str]] = {}

            for comp_idx in range(k_eff):
                comp_name = f"text_{col}_svd_{comp_idx}"
                col_feature_names.append(comp_name)
                self.dense_feature_names.append(comp_name)

                # Weight vector across vocabulary
                weights = svd.components_[comp_idx]
                abs_weights = np.abs(weights)
                top_indices = np.argsort(abs_weights)[::-1][:min(8, len(feature_names))]
                top_terms = [
                    str(feature_names[idx])
                    for idx in top_indices
                    if abs_weights[idx] > 1e-6
                ]
                col_keywords[comp_name] = top_terms
                self.component_keywords[comp_name] = top_terms

            var_ratios = [float(v) for v in svd.explained_variance_ratio_.tolist()]
            self.explained_variance[col] = var_ratios

            # Calculate average word count
            word_counts = [len(re.findall(r"\b\w+\b", s)) for s in text_series.head(200)]
            avg_words = float(np.mean(word_counts)) if word_counts else 0.0

            meta = TextModalityMeta(
                column_name=col,
                sample_count=n_rows,
                avg_words_per_sample=round(avg_words, 2),
                n_components=k_eff,
                semantic_feature_names=col_feature_names,
                component_keyword_attributions=col_keywords,
            )
            self.modality_meta.append(meta)

            # Proactive resource cleanup
            del tfidf_matrix
            gc.collect()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame by generating dense semantic embeddings and fusing them.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Fused pandas DataFrame.
        """
        if df is None or df.empty or not self.text_columns or not self.svd_models:
            return df.copy(deep=False) if df is not None else pd.DataFrame()

        dense_dfs: List[pd.DataFrame] = []

        for col in self.text_columns:
            if col not in self.svd_models or col not in self.vectorizers:
                continue

            tfidf = self.vectorizers[col]
            svd = self.svd_models[col]

            # Defensive text extraction
            if col in df.columns:
                series = df[col].fillna("").astype(str).str.strip()
            else:
                series = pd.Series([""] * len(df), index=df.index)

            try:
                tfidf_mat = tfidf.transform(series)
                dense_arr = svd.transform(tfidf_mat)
                del tfidf_mat
            except Exception:
                dense_arr = np.zeros((len(df), svd.n_components), dtype=float)

            col_feature_names = [f"text_{col}_svd_{i}" for i in range(dense_arr.shape[1])]
            dense_col_df = pd.DataFrame(dense_arr, columns=col_feature_names, index=df.index)
            dense_dfs.append(dense_col_df)

        if not dense_dfs:
            return df.copy(deep=False)

        # Concatenate dense features horizontally
        all_dense = pd.concat(dense_dfs, axis=1)

        # Retain or drop raw text columns
        if self.drop_raw_text:
            base_df = df.drop(columns=[c for c in self.text_columns if c in df.columns])
        else:
            base_df = df

        fused_df = pd.concat([base_df, all_dense], axis=1)

        # Clean intermediate memory
        del dense_dfs, all_dense
        gc.collect()

        return fused_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        drop_raw_text: Optional[bool] = None,
    ) -> FusionResult:
        """
        Fits semantic models and returns the FusionResult dataclass.

        Args:
            df: Input DataFrame.
            text_columns: Optional explicit list of text columns.
            target_col: Optional target column to exclude.
            drop_raw_text: Optional override for drop_raw_text.

        Returns:
            FusionResult containing fused DataFrame and detailed metadata.
        """
        if drop_raw_text is not None:
            self.drop_raw_text = bool(drop_raw_text)

        self.fit(df=df, text_columns=text_columns, target_col=target_col)
        fused_df = self.transform(df)

        manifest = {
            "num_text_columns": len(self.text_columns),
            "text_columns": self.text_columns,
            "total_semantic_features": len(self.dense_feature_names),
            "n_components_per_column": self.n_components,
            "drop_raw_text": self.drop_raw_text,
            "explained_variance": self.explained_variance,
        }

        return FusionResult(
            fused_df=fused_df,
            detected_text_columns=list(self.text_columns),
            new_semantic_features=list(self.dense_feature_names),
            modality_meta=list(self.modality_meta),
            fusion_manifest=manifest,
        )


# ─── Functional API ───────────────────────────────────────────────────────────

def fuse_tabular_and_text(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    n_components: int = 8,
    drop_raw_text: bool = False,
    max_features: int = 5000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Primary functional interface for multi-modal tabular and text semantic fusion.

    Identifies unstructured free-text fields (or uses explicitly specified ones),
    extracts dense semantic vector components via TF-IDF + TruncatedSVD, preserves
    keyword explainability, and returns the fused DataFrame alongside metadata.

    Args:
        df: Input pandas DataFrame containing tabular and optional free-text columns.
        text_columns: Optional list of specific text columns to fuse. If None, auto-detected.
        n_components: Number of latent semantic dimensions per text column (default 8).
        drop_raw_text: If True, removes raw text columns from the output DataFrame (default False).
        max_features: Maximum TF-IDF n-gram vocabulary size (default 5000).

    Returns:
        Tuple containing:
        1. fused_df: pandas DataFrame with horizontal dense semantic embeddings concatenated.
        2. meta_dict: Dictionary with keys:
           - text_columns_detected: List of processed text column names.
           - dense_feature_names: List of newly generated feature names (e.g. text_{col}_svd_{i}).
           - top_keywords_per_component: Dict mapping each component to its top indicative keywords.
           - explained_variance_ratio: List of explained variance ratios across components.
           - fusion_manifest: Full transformation manifest.
    """
    if df is None or df.empty or len(df) == 0:
        empty_meta = {
            "text_columns_detected": [],
            "dense_feature_names": [],
            "top_keywords_per_component": {},
            "explained_variance_ratio": [],
            "fusion_manifest": {"status": "EMPTY_INPUT"},
        }
        return (pd.DataFrame() if df is None else df.copy(deep=False), empty_meta)

    engine = MultiModalFusionEngine(
        n_components=n_components,
        max_features=max_features,
        drop_raw_text=drop_raw_text,
    )

    result = engine.fit_transform(df=df, text_columns=text_columns)

    # Flatten explained variance across all columns
    flat_explained_variance: List[float] = []
    for var_list in engine.explained_variance.values():
        flat_explained_variance.extend(var_list)

    meta_dict: Dict[str, Any] = {
        "text_columns_detected": result.detected_text_columns,
        "dense_feature_names": result.new_semantic_features,
        "top_keywords_per_component": engine.component_keywords,
        "explained_variance_ratio": flat_explained_variance,
        "fusion_manifest": result.fusion_manifest,
    }

    return result.fused_df, meta_dict
