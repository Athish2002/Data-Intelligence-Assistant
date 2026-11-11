"""
dia/invariant_engine.py
───────────────────────
Axiomatic Conservation Laws & Closed Manifold Projection Engine.

Guarantees that machine learning predictions, recourse recommendations,
and what-if interventions strictly satisfy closed physical, financial,
and operational invariants without empirical leakage or drift:

1. Linear Conservation Invariant Projection (Kirchhoff / Balance Laws):
   - Projects unconstrained vectors x onto affine subspace {v | A v = b}
     via orthogonal projector: x* = x - A^T (A A^T)^(-1) (A x - b).
   - Satisfies A x* = b to machine epsilon precision.
2. Exact Probability Simplex Projection:
   - Duchi et al. (2008) O(K log K) projection onto Delta^K = {p | sum(p) = 1, p >= 0}.
3. Monotonicity Invariant Enforcement:
   - Isotonic Pool Adjacent Violators Algorithm (PAVA) enforcing directional axioms.
4. Axiomatic Data Contract Auditor:
   - Detects conservation and domain violations before and after model inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────
# 1. Linear Conservation Invariant Projection (A x = b)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ConservationLaw:
    """
    Defines a linear conservation constraint across features:
    sum_i (coefficients[feature_i] * x[feature_i]) = target_sum
    e.g., Inflow - Outflow = Delta_Balance
    """
    name: str
    coefficients: dict[str, float]
    target_sum: float = 0.0
    tolerance: float = 1e-6
    formula: str = ""
    variance_explained: float = 1.0
    residual_variance: float = 0.0


class ConservationProjector:
    """
    Orthogonally projects candidate vectors onto the linear manifold defined
    by one or more conservation laws {x in R^d | A x = b}.
    """

    def __init__(self, laws: list[ConservationLaw], feature_names: list[str]):
        self.laws = laws
        self.feature_names = feature_names
        self.feat_to_idx = {f: i for i, f in enumerate(feature_names)}
        self._build_matrices()

    def _build_matrices(self) -> None:
        m = len(self.laws)
        d = len(self.feature_names)
        self.A = np.zeros((m, d), dtype=float)
        self.b = np.zeros(m, dtype=float)

        for row_idx, law in enumerate(self.laws):
            self.b[row_idx] = law.target_sum
            for feat, coef in law.coefficients.items():
                if feat in self.feat_to_idx:
                    self.A[row_idx, self.feat_to_idx[feat]] = coef

        # Exact pseudoinverse projector: A^T (A A^T)^(-1)
        A_AT = self.A @ self.A.T
        try:
            inv_A_AT = np.linalg.pinv(A_AT)
            self.projector_matrix = self.A.T @ inv_A_AT
        except Exception:
            self.projector_matrix = np.linalg.pinv(self.A)
        self.projector = self.projector_matrix

    def project(
        self,
        x: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Orthogonally projects vector x (1D) or batch matrix x (2D) onto A x = b.
        If bounds (lower, upper) are provided, clips iteratively.
        """
        x_arr = np.asarray(x, dtype=float).copy()
        if x_arr.ndim == 1:
            residual = self.A @ x_arr - self.b
            x_proj = x_arr - (self.projector @ residual)

            if bounds is not None:
                lower, upper = bounds
                x_proj = np.clip(x_proj, lower, upper)
                # Second projection pass after clipping
                residual = self.A @ x_proj - self.b
                x_proj = x_proj - (self.projector @ residual)

            return x_proj
        elif x_arr.ndim == 2:
            # 2D batch vectorized projection: x_proj = x - residuals @ projector^T
            residuals = (x_arr @ self.A.T) - self.b  # shape (N, m)
            x_proj = x_arr - (residuals @ self.projector.T)  # shape (N, d)

            if bounds is not None:
                lower, upper = bounds
                x_proj = np.clip(x_proj, lower, upper)
                # Second projection pass after clipping
                residuals = (x_proj @ self.A.T) - self.b
                x_proj = x_proj - (residuals @ self.projector.T)

            return x_proj
        else:
            raise ValueError(
                f"Expected 1D vector (d,) or 2D batch matrix (N, d), got shape {x_arr.shape}"
            )

    def verify_conservation(self, x: np.ndarray) -> list[tuple[str, float, bool]]:
        """Verifies if vector x satisfies all conservation laws within tolerance."""
        results = []
        for row_idx, law in enumerate(self.laws):
            lhs = float(self.A[row_idx] @ x)
            rhs = float(self.b[row_idx])
            error = abs(lhs - rhs)
            passed = error <= law.tolerance
            results.append((law.name, error, passed))
        return results


# ───────────────────────────────────────────────────────────────────────────
# 2. Probability Simplex Projection (Duchi et al., 2008)
# ───────────────────────────────────────────────────────────────────────────

def project_to_probability_simplex(v: np.ndarray) -> np.ndarray:
    """
    Exact Euclidean projection of vector v onto the probability simplex:
    Delta^K = {p in R^K | sum(p) = 1, p >= 0}.
    Operates in O(K log K) time using sort-and-threshold.
    """
    v_arr = np.asarray(v, dtype=float).flatten()
    k = len(v_arr)
    if k == 0:
        return np.array([], dtype=float)
    if k == 1:
        return np.array([1.0], dtype=float)

    # Sort descending
    u = np.sort(v_arr)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(1, k + 1)
    cond = u - (cssv - 1.0) / indices > 0
    if not np.any(cond):
        rho = 1
    else:
        rho = int(indices[cond][-1])

    theta = (cssv[rho - 1] - 1.0) / float(rho)
    p = np.maximum(v_arr - theta, 0.0)

    # Final normalization safeguard against IEEE floating point truncation
    total = np.sum(p)
    if total > 0:
        p /= total
    else:
        p = np.full(k, 1.0 / k)

    return p


# ───────────────────────────────────────────────────────────────────────────
# 3. Monotonicity Invariant Enforcement (PAVA Algorithm)
# ───────────────────────────────────────────────────────────────────────────

def enforce_monotonicity_pava(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    increasing: bool = True,
) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
    Guarantees strict monotonic progression: y_0 <= y_1 <= ... <= y_{n-1}
    (or decreasing if increasing=False).
    """
    y_arr = np.asarray(y, dtype=float).copy()
    if not increasing:
        y_arr = -y_arr

    n = len(y_arr)
    if n <= 1:
        return y_arr if increasing else -y_arr

    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)

    # Blocks: (weight, sum_val, start_idx, end_idx)
    stack: list[list[float]] = []

    for i in range(n):
        cur_weight = float(w[i])
        cur_sum = float(w[i] * y_arr[i])
        block = [cur_weight, cur_sum, float(i), float(i)]

        while stack:
            prev_block = stack[-1]
            prev_mean = prev_block[1] / prev_block[0]
            cur_mean = block[1] / block[0]

            if prev_mean > cur_mean:
                # Merge violating adjacent blocks
                popped = stack.pop()
                block[0] += popped[0]
                block[1] += popped[1]
                block[2] = popped[2]  # update start
            else:
                break

        stack.append(block)

    # Reconstruct monotonic array
    result = np.zeros(n, dtype=float)
    for block in stack:
        mean_val = block[1] / block[0]
        start_idx = int(block[2])
        end_idx = int(block[3])
        result[start_idx : end_idx + 1] = mean_val

    return result if increasing else -result


# ───────────────────────────────────────────────────────────────────────────
# 4. Axiomatic Data Contract Auditor
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class AxiomAuditSummary:
    total_records: int
    passed_records: int
    violation_rate: float
    violations_by_law: dict[str, int]
    is_axiom_compliant: bool
    details: list[str]


def audit_dataframe_axioms(
    df: pd.DataFrame,
    laws: list[ConservationLaw],
) -> AxiomAuditSummary:
    """
    Verifies closed conservation laws across rows in a pandas DataFrame.
    Flags impossible rows where physical balance laws are violated.
    """
    if df.empty or not laws:
        return AxiomAuditSummary(
            total_records=len(df),
            passed_records=len(df),
            violation_rate=0.0,
            violations_by_law={law.name: 0 for law in laws},
            is_axiom_compliant=True,
            details=["No conservation laws specified or DataFrame is empty."],
        )

    cols = list(df.columns)
    projector = ConservationProjector(laws=laws, feature_names=cols)

    # Convert dataframe columns to numeric 2D array
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Shape: (N, d), A shape: (m, d), b shape: (m,)
    # Residuals matrix: |X @ A.T - b| -> shape (N, m)
    residuals = np.abs(X @ projector.A.T - projector.b)
    tolerances = np.array([law.tolerance for law in laws], dtype=float)

    violated = residuals > tolerances  # (N, m) boolean mask
    violations_count: dict[str, int] = {}
    for i, law in enumerate(laws):
        violations_count[law.name] = int(np.sum(violated[:, i]))

    failed_mask = np.any(violated, axis=1)
    total = len(df)
    n_failed = int(np.sum(failed_mask))
    n_passed = total - n_failed
    rate = (n_failed / total) if total > 0 else 0.0

    details = []
    for l_name, cnt in violations_count.items():
        if cnt > 0:
            details.append(f"Law '{l_name}' violated by {cnt} records ({cnt/total*100:.1f}%).")

    return AxiomAuditSummary(
        total_records=total,
        passed_records=n_passed,
        violation_rate=float(rate),
        violations_by_law=violations_count,
        is_axiom_compliant=n_failed == 0,
        details=details if details else ["All records strictly satisfy closed conservation axioms."],
    )


# ───────────────────────────────────────────────────────────────────────────
# 5. Algebraic Invariant Scanner (Suggest & Lock Engine)
# ───────────────────────────────────────────────────────────────────────────

def _format_law_equation(coeffs: dict[str, float], target_sum: float = 0.0) -> str:
    """Formats linear coefficients into clean human-readable formula: sum c_i X_i = target_sum."""
    terms: list[str] = []
    for feat, coef in coeffs.items():
        if abs(coef) < 1e-5:
            continue
        c_round = round(coef, 4)
        is_one = abs(abs(c_round) - 1.0) < 1e-5
        c_str = f"{c_round:.1f}" if c_round == int(c_round) else f"{c_round}"
        c_abs_str = f"{abs(c_round):.1f}" if c_round == int(c_round) else f"{abs(c_round)}"

        if not terms:
            if is_one and c_round > 0:
                terms.append(f"{feat}")
            elif is_one and c_round < 0:
                terms.append(f"-{feat}")
            elif c_round < 0:
                terms.append(f"-{c_abs_str} {feat}")
            else:
                terms.append(f"{c_str} {feat}")
        else:
            if is_one and c_round > 0:
                terms.append(f"+ {feat}")
            elif is_one and c_round < 0:
                terms.append(f"- {feat}")
            elif c_round > 0:
                terms.append(f"+ {c_abs_str} {feat}")
            else:
                terms.append(f"- {c_abs_str} {feat}")

    if target_sum == 0.0 or abs(target_sum) < 1e-5:
        t_str = "0"
    else:
        b_val = round(target_sum, 2)
        t_str = f"{int(b_val)}" if abs(b_val - int(b_val)) < 1e-5 else f"{b_val}"
    return " ".join(terms) + f" = {t_str}"


def _infer_balance_category(features: list[str]) -> str:
    """Infers domain balance law archetype from feature nomenclature."""
    feats_lower = [f.lower() for f in features]
    if any(k in f for f in feats_lower for k in ["inflow", "outflow", "delta", "drain"]):
        return "Flow Balance"
    if any(k in f for f in feats_lower for k in ["mass", "kg", "inlet", "outlet"]):
        return "Mass Balance"
    if any(k in f for f in feats_lower for k in ["revenue", "cost", "profit", "margin", "price", "fee"]):
        return "Financial Balance"
    if any(k in f for f in feats_lower for k in ["asset", "liability", "equity"]):
        return "Accounting Balance"
    if any(k in f for f in feats_lower for k in ["energy", "power", "watt", "joule"]):
        return "Energy Balance"
    return "Mass Balance" if any("x" in f for f in feats_lower) else "Conservation Law"


def _rref(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Computes Row Reduced Echelon Form (RREF) using Gauss-Jordan elimination.
    Disentangles mixed orthogonal null space basis vectors into sparse linear relations.
    """
    B = A.astype(float).copy()
    rows, cols = B.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = int(np.argmax(np.abs(B[r:, c])) + r)
        if np.abs(B[pivot, c]) < tol:
            continue
        B[[r, pivot]] = B[[pivot, r]]
        B[r] /= B[r, c]
        for i in range(rows):
            if i != r:
                B[i] -= B[i, c] * B[r]
        r += 1
    return B


def scan_dataframe_conservation_laws(
    df: pd.DataFrame,
    max_features: int = 12,
    r2_threshold: float = 0.9999,
    residual_var_tol: float = 1e-8,
) -> list[ConservationLaw]:
    """
    Uses Singular Value Decomposition (SVD) and Null-Space analysis on normalized
    continuous numerical features to discover exact linear combinations:
    sum_{i=1}^k c_i X_i approx 0

    Normalizes candidate coefficient vectors so the smallest non-zero coefficient is +-1.0.
    Prunes trivial all-zero or single-variable solutions.
    Automatically generates clear human-readable names and equations.
    """
    if df is None or df.empty:
        return []

    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        return []

    # Drop zero-variance/constant columns
    valid_cols: list[str] = []
    for c in num_df.columns:
        s = num_df[c].dropna()
        if len(s) > 2 and float(s.std()) > 1e-12:
            valid_cols.append(c)

    if len(valid_cols) < 2:
        return []

    # Select top features by variance if exceeding max_features
    if len(valid_cols) > max_features:
        sorted_by_var = sorted(valid_cols, key=lambda c: float(num_df[c].dropna().var()), reverse=True)[:max_features]
        feature_cols = [c for c in num_df.columns if c in set(sorted_by_var)]
    else:
        feature_cols = valid_cols

    # Clean rows with complete values in selected features
    sub_df = num_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    N, d = sub_df.shape
    if N < 3 or N < d + 1 or d < 2:
        return []

    M = sub_df.to_numpy(dtype=float)
    scales = np.linalg.norm(M, axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    Z = M / scales
    # Augment normalized matrix with intercept column 1_N / sqrt(N) to discover affine laws
    Z_aug = np.hstack([Z, np.ones((N, 1), dtype=float) / np.sqrt(N)])

    try:
        _U, S, Vt = np.linalg.svd(Z_aug, full_matrices=False)
    except Exception:
        return []

    # Null space singular values check
    null_mask = S < 1e-4
    if not np.any(null_mask):
        return []

    # Convert null vectors back to original feature scale: [w, -\beta]^T
    aug_scales = np.append(scales, np.sqrt(N))
    V_null = Vt[null_mask] / aug_scales
    V_basis = _rref(V_null)
    candidates: list[ConservationLaw] = []

    for row in V_basis:
        max_abs = np.max(np.abs(row[:d]))
        if max_abs < 1e-12:
            continue
        row_clean = row[:d].copy()
        row_clean[np.abs(row_clean) < 1e-5 * max_abs] = 0.0

        non_zero_indices = np.where(np.abs(row_clean) > 1e-5)[0]
        if len(non_zero_indices) < 2:
            continue

        nz_magnitudes = np.abs(row_clean[non_zero_indices])
        min_nz = float(np.min(nz_magnitudes))
        if min_nz < 1e-12:
            continue

        # Normalizes candidate coefficient vectors so the smallest non-zero coefficient is +-1.0
        normalized_row = row_clean / min_nz

        # Ensure first non-zero coefficient is positive
        first_nz_idx = non_zero_indices[0]
        if normalized_row[first_nz_idx] < 0:
            normalized_row = -normalized_row

        cleaned_coeffs: dict[str, float] = {}
        for idx in non_zero_indices:
            feat_name = feature_cols[idx]
            val = float(normalized_row[idx])
            val_rounded = round(val, 6)
            if abs(val_rounded - round(val_rounded)) < 1e-4:
                cleaned_coeffs[feat_name] = float(round(val_rounded))
            elif abs(val_rounded - round(val_rounded, 2)) < 1e-4:
                cleaned_coeffs[feat_name] = round(val_rounded, 2)
            elif abs(val_rounded - round(val_rounded, 4)) < 1e-5:
                cleaned_coeffs[feat_name] = round(val_rounded, 4)
            else:
                cleaned_coeffs[feat_name] = val_rounded

        active_coeffs = {f: c for f, c in cleaned_coeffs.items() if abs(c) > 1e-5}
        if len(active_coeffs) < 2:
            continue

        cand_vec = np.zeros(d, dtype=float)
        for idx in range(d):
            cand_vec[idx] = active_coeffs.get(feature_cols[idx], 0.0)

        y = M @ cand_vec
        target_sum = float(np.mean(y))
        std_y = float(np.std(y))
        if abs(target_sum) < 1e-4 * std_y or abs(target_sum) < 1e-5:
            target_sum = 0.0
        elif abs(target_sum - round(target_sum)) < 1e-4:
            target_sum = float(round(target_sum))
        else:
            target_sum = round(target_sum, 4)

        var_y = float(np.var(y))

        primary_idx = max(
            non_zero_indices,
            key=lambda idx: abs(active_coeffs.get(feature_cols[idx], 0.0)) * float(np.std(M[:, idx]))
        )
        primary_col = feature_cols[primary_idx]
        c_primary = active_coeffs[primary_col]
        var_primary = float(np.var(M[:, primary_idx]))

        residual_variance = var_y / (c_primary ** 2)
        variance_explained = 1.0 - (residual_variance / var_primary) if var_primary > 1e-12 else 1.0

        # Scale-invariant residual variance normalized relative to primary feature variance
        norm_res_var = residual_variance / (var_primary + 1e-12)
        if variance_explained < r2_threshold or norm_res_var > (1.0 - r2_threshold + 1e-7):
            continue

        eq_str = _format_law_equation(active_coeffs, target_sum=target_sum)
        category = _infer_balance_category(list(active_coeffs.keys()))
        law_name = f"{category}: {eq_str}"

        law = ConservationLaw(
            name=law_name,
            coefficients=active_coeffs,
            target_sum=target_sum,
            tolerance=1e-6,
            formula=eq_str,
            variance_explained=float(variance_explained),
            residual_variance=float(residual_variance),
        )
        candidates.append(law)

    return candidates


# Canonical export alias for discovery scanner
scan_conservation_laws = scan_dataframe_conservation_laws
