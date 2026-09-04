"""
dia/nlp_processor.py
────────────────────
Multimodal NLP & Free-Text Intelligence Engine.
Detects unstructured text columns, performs lexical analysis,
and extracts TF-IDF dense embeddings.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger("dia.nlp")


def detect_text_columns(df: pd.DataFrame, min_avg_words: float = 3.0) -> list[str]:
    """
    Detects string columns containing unstructured free text (sentences/paragraphs).
    """
    text_cols = []
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(50)
            if not sample.empty:
                avg_words = sample.apply(lambda x: len(x.split())).mean()
                if avg_words >= min_avg_words:
                    text_cols.append(col)
    return text_cols


def extract_lexical_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Extracts lexical, readability, and structural features from a text column.
    """
    s_clean = series.fillna("").astype(str)

    char_len = s_clean.apply(len)
    word_count = s_clean.apply(lambda x: len(x.split()))
    avg_word_len = char_len / (word_count.replace(0, 1))
    uppercase_ratio = s_clean.apply(lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x)))
    digit_ratio = s_clean.apply(lambda x: sum(1 for c in x if c.isdigit()) / max(1, len(x)))
    exclamation_count = s_clean.apply(lambda x: x.count("!"))
    question_count = s_clean.apply(lambda x: x.count("?"))

    # Basic Sentiment Rule Engine (Positive vs Negative vocabulary)
    pos_words = {"good", "great", "excellent", "amazing", "love", "best", "fast", "happy", "reliable", "perfect"}
    neg_words = {"bad", "poor", "terrible", "worst", "hate", "slow", "error", "fail", "broken", "issue", "bug"}

    def _sentiment_score(text: str) -> float:
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos = len(words & pos_words)
        neg = len(words & neg_words)
        total = pos + neg
        return float((pos - neg) / total) if total > 0 else 0.0

    sentiment = s_clean.apply(_sentiment_score)

    return pd.DataFrame({
        f"{prefix}_char_count": char_len,
        f"{prefix}_word_count": word_count,
        f"{prefix}_avg_word_len": avg_word_len,
        f"{prefix}_uppercase_ratio": uppercase_ratio,
        f"{prefix}_digit_ratio": digit_ratio,
        f"{prefix}_exclamations": exclamation_count,
        f"{prefix}_questions": question_count,
        f"{prefix}_sentiment_score": sentiment,
    })


def extract_tfidf_dense_features(
    series: pd.Series,
    prefix: str,
    n_components: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extracts TF-IDF n-grams and compresses into dense latent semantic components.
    """
    s_clean = series.fillna("").astype(str)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
    try:
        X_tfidf = tfidf.fit_transform(s_clean)

        # SVD Compression
        n_comp = min(n_components, X_tfidf.shape[1], max(1, len(s_clean) - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_dense = svd.fit_transform(X_tfidf)

        feature_names = [f"{prefix}_semantic_comp_{i+1}" for i in range(n_comp)]
        df_dense = pd.DataFrame(X_dense, columns=feature_names, index=series.index)
        return df_dense, feature_names
    except Exception:
        return pd.DataFrame(index=series.index), []
