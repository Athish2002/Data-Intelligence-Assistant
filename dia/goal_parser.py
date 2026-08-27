"""
dia/goal_parser.py
──────────────────
Maps the user's free‑text prediction goal to:
  • task_type          – "classification" | "regression"
  • target_candidates  – list of candidate column names from the dataframe
  • confidence         – 0.0 – 1.0 how sure we are about the task type

Two strategies are implemented:
  1. Keyword matching  (always available, zero extra dependencies)
  2. Semantic matching (requires sentence-transformers; graceful fallback)
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .utils import is_sentence_transformers_available

# ─── Keyword rules ────────────────────────────────────────────────────────────

_CLASSIFICATION_KEYWORDS = [
    "classif", "churn", "fraud", "spam", "cancer", "disease", "diagnos",
    "sentiment", "category", "categori", "label", "survive", "survival",
    "default", "risk", "detect", "predict whether", "predict if",
    "binary", "multi-class", "anomaly",
]

_REGRESSION_KEYWORDS = [
    "price", "cost", "amount", "revenue", "sales", "salary", "wage",
    "forecast", "predict how much", "predict the value", "estimate",
    "regression", "continuous", "score", "rating", "duration", "age",
    "temperature", "demand", "quantity", "rate",
]

# Seed phrases used for semantic intent matching
_CLASSIFICATION_SEEDS = [
    "predict whether a customer will churn",
    "classify whether a patient has a disease",
    "detect fraud in transactions",
    "spam detection",
    "binary classification task",
]

_REGRESSION_SEEDS = [
    "predict house price",
    "forecast sales revenue",
    "estimate salary based on experience",
    "regression prediction of a numeric value",
    "forecast demand or quantity",
]


def _keyword_match(goal: str) -> tuple[str | None, float]:
    """Return (task_type, confidence) based on keyword scanning."""
    goal_lower = goal.lower()

    cls_hits = sum(kw in goal_lower for kw in _CLASSIFICATION_KEYWORDS)
    reg_hits = sum(kw in goal_lower for kw in _REGRESSION_KEYWORDS)

    total = cls_hits + reg_hits
    if total == 0:
        return None, 0.0

    if cls_hits > reg_hits:
        conf = min(0.5 + 0.1 * cls_hits, 0.90)
        return "classification", round(conf, 2)
    elif reg_hits > cls_hits:
        conf = min(0.5 + 0.1 * reg_hits, 0.90)
        return "regression", round(conf, 2)
    else:
        # tie – slight lean toward classification
        return "classification", 0.50


def _semantic_match(goal: str) -> tuple[str | None, float]:
    """Return (task_type, confidence) using sentence-transformers cosine similarity."""
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        goal_emb = model.encode(goal, convert_to_tensor=True)

        cls_embs = model.encode(_CLASSIFICATION_SEEDS, convert_to_tensor=True)
        reg_embs = model.encode(_REGRESSION_SEEDS, convert_to_tensor=True)

        cls_score = float(util.cos_sim(goal_emb, cls_embs).max())
        reg_score = float(util.cos_sim(goal_emb, reg_embs).max())

        if cls_score > reg_score:
            return "classification", round(min(cls_score, 0.99), 2)
        elif reg_score > cls_score:
            return "regression", round(min(reg_score, 0.99), 2)
        else:
            return "classification", 0.50
    except Exception:
        return None, 0.0


def _infer_target_candidates(goal: str, columns: Sequence[str]) -> list[str]:
    """
    Heuristically rank columns by how likely they are the prediction target.
    Leverages synonym clusters, normalized sub-token matching, and fuzzy resolution.
    """
    from .column_resolver import normalize_string, resolve_column

    goal_tokens = re.findall(r"[a-z]+", goal.lower())
    ranked: list[tuple[str, float]] = []

    for col in columns:
        score = 0.0
        col_norm = normalize_string(col)

        # 1. Match each goal token against column
        for token in goal_tokens:
            if len(token) < 3:
                continue
            matched_col, conf, _ = resolve_column(token, [col])
            if matched_col and conf > 0.6:
                score = max(score, conf)

        # 2. Check if column matches generic target indicator terms
        if any(term in col_norm for term in ["target", "label", "class", "output", "flag", "status"]):
            score = max(score, 0.65)

        # 3. Downrank Primary Key / ID columns
        if col_norm.endswith("_id") or col_norm in ["id", "client_id", "user_id", "customer_id", "account_id", "cust_id", "cust_no", "applicant_id"]:
            score *= 0.20

        ranked.append((col, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    candidates = [c for c, s in ranked if s > 0.5]
    return candidates[:5] if candidates else [columns[-1]]


def parse_goal(
    goal: str,
    columns: Sequence[str] | None = None,
    use_semantic: bool = True,
) -> dict:
    """
    Parse a free-text ML goal.

    Parameters
    ----------
    goal        : user's typed goal string
    columns     : list of dataframe column names (for target candidate ranking)
    use_semantic: attempt sentence-transformers if available

    Returns
    -------
    dict with keys:
        task_type          : "classification" | "regression" | "unknown"
        confidence         : float 0–1
        method             : "semantic" | "keyword" | "hybrid"
        target_candidates  : list[str]  (empty if no columns provided)
        explanation        : human-readable explanation of the inference
    """
    goal = goal.strip()
    if not goal:
        return {
            "task_type": "unknown",
            "confidence": 0.0,
            "method": "none",
            "target_candidates": [],
            "explanation": "No goal provided.",
        }

    # Run both matchers
    kw_type, kw_conf = _keyword_match(goal)

    sem_type, sem_conf = (None, 0.0)
    method = "keyword"
    if use_semantic and is_sentence_transformers_available():
        sem_type, sem_conf = _semantic_match(goal)
        method = "semantic"

    # Combine: semantic wins if available and confident, else fallback to keyword
    if sem_type and sem_conf >= kw_conf:
        task_type = sem_type
        confidence = sem_conf
        method = "semantic" if kw_type == sem_type else "hybrid"
    elif kw_type:
        task_type = kw_type
        confidence = kw_conf
        method = "keyword"
    else:
        task_type = "unknown"
        confidence = 0.0
        method = "none"

    # Build explanation
    if task_type == "classification":
        explanation = (
            f"The goal '{goal}' was interpreted as a **classification** task "
            f"(confidence: {confidence:.0%}).  "
            "The app will predict a discrete category or binary outcome."
        )
    elif task_type == "regression":
        explanation = (
            f"The goal '{goal}' was interpreted as a **regression** task "
            f"(confidence: {confidence:.0%}).  "
            "The app will predict a continuous numeric value."
        )
    else:
        explanation = (
            f"Could not determine task type from '{goal}'.  "
            "Please include words like 'predict', 'classify', 'price', or 'churn'."
        )

    target_candidates = (
        _infer_target_candidates(goal, list(columns)) if columns else []
    )

    return {
        "task_type": task_type,
        "confidence": confidence,
        "method": method,
        "target_candidates": target_candidates,
        "explanation": explanation,
    }
