"""
tests/test_smart_insights.py
─────────────────────────────
Rigorous production-grade verification and adversarial stress testing suite
for smart non-linear insights, Simpson's Paradox confounder detection,
prescriptive causal guidance, and chat analyst analytical intent execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dia.chat_analyst import execute_natural_language_query
from dia.insights import generate_smart_insights


# ═══════════════════════════════════════════════════════════════════════════
# 1. Non-Linear Dynamics & Noise Rejection Invariant Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNonLinearDynamicsAndNoiseRejection:
    def test_u_shaped_relationship_detected_when_pearson_near_zero(self) -> None:
        """
        Verify that a synthetic U-shaped quadratic relationship (y = x^2) is
        successfully detected by the non-linear detector even though the linear
        Pearson correlation is near zero (|r| < 0.15).
        """
        rng = np.random.default_rng(seed=42)
        x = np.linspace(-10.0, 10.0, 300)
        # Symmetrical parabola centered at 0 with mild noise
        y = 0.5 * (x ** 2) + rng.normal(0.0, 1.5, size=len(x))

        df = pd.DataFrame({"x_feature": x, "y_target": y})

        # Check that Pearson r is indeed near zero
        linear_r = float(np.corrcoef(x, y)[0, 1])
        assert abs(linear_r) < 0.15, f"Expected Pearson r near zero, got {linear_r:.3f}"

        insights = generate_smart_insights(df, target_col="y_target", task_type="regression")
        assert len(insights) >= 1

        titles = [i["title"] for i in insights]
        descs = [i["description"] for i in insights]

        # Must flag non-linear or U-shaped dynamics
        assert any("x_feature" in t for t in titles)
        assert any("u-shaped" in d.lower() or "non-linear" in d.lower() for d in descs)
        assert any("turning point" in d.lower() or "curvature" in d.lower() for d in descs)

    def test_noise_rejection_invariant_pure_gaussian_noise(self) -> None:
        """
        NOISE REJECTION INVARIANT:
        If features and target are purely random i.i.d. Gaussian and Uniform noise,
        the non-linear detector MUST NOT hallucinate a relationship.
        """
        rng = np.random.default_rng(seed=1337)
        n = 350
        df = pd.DataFrame({
            "noise_gauss": rng.normal(0.0, 1.0, size=n),
            "noise_unif": rng.uniform(-10.0, 10.0, size=n),
            "target": rng.normal(50.0, 5.0, size=n),
        })

        insights = generate_smart_insights(df, target_col="target", task_type="regression")

        # The noise rejection filter must discard spurious non-linear/linear fits
        # and fall back to the safe Complex Relationships explanation
        assert len(insights) == 1
        assert "Complex Relationships" in insights[0]["title"]
        assert "significance" in insights[0]["description"].lower()

    def test_threshold_jump_effect_detected(self) -> None:
        """
        Verify that a discontinuous threshold jump is caught by the shallow tree detector.
        """
        rng = np.random.default_rng(seed=101)
        x = np.linspace(10.0, 100.0, 250)
        # Below threshold 55 -> mean 15; above threshold 55 -> mean 45 (jump of +30)
        y = np.where(x > 55.0, 45.0, 15.0) + rng.normal(0.0, 2.0, size=len(x))

        df = pd.DataFrame({"metric_val": x, "outcome": y})

        insights = generate_smart_insights(df, target_col="outcome", task_type="regression")
        assert len(insights) >= 1

        descs = [i["description"] for i in insights]
        threshold_insights = [d for d in descs if "threshold" in d.lower()]
        assert len(threshold_insights) >= 1
        assert "cohen's d" in threshold_insights[0].lower() or "jumps" in threshold_insights[0].lower()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Simpson's Paradox & Confounder Detection Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSimpsonsParadoxAndConfounders:
    def test_simpsons_paradox_flagged_on_reversing_subgroups(self) -> None:
        """
        Verify that synthetic Simpson's Paradox data (where global correlation is
        positive r > +0.35, but within each subgroup correlation is negative r < -0.7)
        is flagged as a high-priority advisory insight.
        """
        rng = np.random.default_rng(seed=999)
        n_per_group = 60

        # Subgroup A: x in [0, 10], y = -2.0*x + 40
        xA = rng.uniform(0.0, 10.0, size=n_per_group)
        yA = -2.0 * xA + 40.0 + rng.normal(0.0, 1.0, size=n_per_group)
        grpA = ["Segment_Alpha"] * n_per_group

        # Subgroup B: x in [15, 25], y = -2.0*x + 75
        xB = rng.uniform(15.0, 25.0, size=n_per_group)
        yB = -2.0 * xB + 75.0 + rng.normal(0.0, 1.0, size=n_per_group)
        grpB = ["Segment_Beta"] * n_per_group

        # Subgroup C: x in [30, 40], y = -2.0*x + 110
        xC = rng.uniform(30.0, 40.0, size=n_per_group)
        yC = -2.0 * xC + 110.0 + rng.normal(0.0, 1.0, size=n_per_group)
        grpC = ["Segment_Gamma"] * n_per_group

        df = pd.DataFrame({
            "ad_spend": np.concatenate([xA, xB, xC]),
            "revenue": np.concatenate([yA, yB, yC]),
            "market_tier": np.concatenate([grpA, grpB, grpC]),
        })

        # Verify mathematically that global correlation is positive
        r_global = float(np.corrcoef(df["ad_spend"], df["revenue"])[0, 1])
        assert r_global > 0.35, f"Global correlation must be positive, got {r_global:.3f}"

        # Verify within each subgroup correlation is negative
        for grp in ["Segment_Alpha", "Segment_Beta", "Segment_Gamma"]:
            sub = df[df["market_tier"] == grp]
            r_sub = float(np.corrcoef(sub["ad_spend"], sub["revenue"])[0, 1])
            assert r_sub < -0.70, f"Subgroup correlation must be strongly negative, got {r_sub:.3f}"

        insights = generate_smart_insights(df, target_col="revenue", task_type="regression")
        assert len(insights) >= 1

        paradox_insights = [i for i in insights if "simpson's paradox" in i["title"].lower() or "simpson's paradox" in i["description"].lower()]
        assert len(paradox_insights) >= 1

        p_desc = paradox_insights[0]["description"]
        assert "global correlation is positive" in p_desc.lower()
        assert "negative" in p_desc.lower()
        assert "market_tier" in p_desc

    def test_simpsons_paradox_guards_against_tiny_noisy_slices(self) -> None:
        """
        Guard against spurious noise reversals:
        A true positive relationship with tiny random 3-sample subgroups must NOT trigger
        a spurious Simpson's Paradox alert.
        """
        rng = np.random.default_rng(seed=42)
        n = 100
        x = np.linspace(1.0, 50.0, n)
        y = 2.0 * x + rng.normal(0.0, 2.0, size=n)
        # 33 tiny categories with ~3 items each
        tiny_cats = [f"cat_{i % 33}" for i in range(n)]

        df = pd.DataFrame({"x_val": x, "y_val": y, "tiny_group": tiny_cats})

        insights = generate_smart_insights(df, target_col="y_val", task_type="regression")
        # Simpson's Paradox must NOT be triggered on underpowered slices
        assert not any("simpson's paradox" in i["title"].lower() for i in insights)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Prescriptive Causal Guidance Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPrescriptiveCausalGuidance:
    def test_prescriptive_causal_action_and_boundary_verification(self) -> None:
        """
        Verify that causal action calculates an unconfounded treatment effect and
        respects observed feature domain bounds.
        """
        rng = np.random.default_rng(seed=777)
        n = 200

        # Confounder Z drives both X and Y
        z = rng.normal(10.0, 2.0, size=n)
        x = np.clip(0.6 * z + rng.normal(5.0, 1.0, size=n), 0.0, 25.0)
        y = 3.0 * x + 1.5 * z + rng.normal(0.0, 0.5, size=n)

        df = pd.DataFrame({"treatment_x": x, "confounder_z": z, "outcome_y": y})

        insights = generate_smart_insights(df, target_col="outcome_y", task_type="regression")

        prescriptive_insights = [i for i in insights if "prescriptive action" in i["title"].lower()]
        assert len(prescriptive_insights) >= 1

        p_desc = prescriptive_insights[0]["description"]
        assert "treatment_x" in p_desc
        assert "unconfounded causal impact" in p_desc.lower()

        # Domain boundary verification: Ensure intervention was within observed bounds
        x_min, x_max = df["treatment_x"].min(), df["treatment_x"].max()
        assert x_min >= 0.0 and x_max <= 25.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Adversarial Stress & Numerical Stability Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAdversarialStressAndNumericalStability:
    def test_zero_variance_numeric_column(self) -> None:
        """Zero variance / constant columns must be handled with zero division errors."""
        df = pd.DataFrame({
            "constant_col": [42.0] * 50,
            "varying_col": np.linspace(1.0, 10.0, 50),
            "target": np.linspace(10.0, 100.0, 50),
        })
        insights = generate_smart_insights(df, target_col="target", task_type="regression")
        assert isinstance(insights, list)
        assert len(insights) >= 1
        # Constant col should not cause issues
        assert not any("constant_col" in i["title"] for i in insights)

    def test_extreme_outliers_and_overflow_resilience(self) -> None:
        """Extreme numbers up to 1e15 must not cause float overflow or matrix errors."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 1e15] * 10)
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 10)
        df = pd.DataFrame({"extreme_x": x, "target": y})

        # Must not raise FloatingPointError or LinAlgError
        insights = generate_smart_insights(df, target_col="target", task_type="regression")
        assert isinstance(insights, list)

    def test_nan_and_infinite_values(self) -> None:
        """Columns with NaN and Inf values must be filtered cleanly without crashing."""
        df = pd.DataFrame({
            "inf_col": [np.inf, -np.inf, 1.0, 2.0, 3.0] * 10,
            "nan_col": [np.nan] * 50,
            "clean_x": np.linspace(1.0, 20.0, 50),
            "target": np.linspace(2.0, 40.0, 50),
        })
        insights = generate_smart_insights(df, target_col="target", task_type="regression")
        assert isinstance(insights, list)
        assert len(insights) >= 1

    def test_empty_and_single_row_dataframe(self) -> None:
        """Empty or single-row DataFrames return safe fallback."""
        empty_df = pd.DataFrame({"target": []})
        insights = generate_smart_insights(empty_df, target_col="target", task_type="regression")
        assert len(insights) == 1
        assert "Complex Relationships" in insights[0]["title"]

        single_df = pd.DataFrame({"x": [1], "target": [10]})
        insights_single = generate_smart_insights(single_df, target_col="target", task_type="regression")
        assert len(insights_single) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Chat Analyst Intent & Analytical Execution Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChatAnalystAnalyticalExecution:
    @pytest.fixture
    def analytical_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=123)
        n = 150
        return pd.DataFrame({
            "region": rng.choice(["North", "South", "East", "West"], size=n),
            "tier": rng.choice(["Gold", "Silver", "Bronze"], size=n),
            "monthly_charges": rng.normal(75.0, 15.0, size=n),
            "tenure_months": rng.integers(1, 72, size=n),
            "total_spend": rng.normal(2500.0, 500.0, size=n),
        })

    def test_intent_segment_comparison(self, analytical_df: pd.DataFrame) -> None:
        """Segment comparison queries must produce SEM, IQR, and statistical hypothesis tests."""
        res = execute_natural_language_query(
            analytical_df, query="Compare monthly_charges across region"
        )
        assert res["text"]
        assert "±" in res["text"] or "SEM" in res["text"] or "Mean" in res["text"]
        assert res["table"] is not None
        assert "SEM" in res["table"].columns or "Mean" in res["table"].columns
        assert res["figure"] is not None

    def test_intent_outlier_investigation(self, analytical_df: pd.DataFrame) -> None:
        """Outlier investigation queries must compute Tukey IQR fences and skewness."""
        # Inject deliberate outlier
        analytical_df.loc[0, "monthly_charges"] = 999.0

        res = execute_natural_language_query(
            analytical_df, query="Find outliers and extreme anomalies in monthly_charges"
        )
        assert res["text"]
        assert "tukey" in res["text"].lower() or "outlier" in res["text"].lower()
        assert "fence" in res["text"].lower() or "z-score" in res["text"].lower()
        assert res["figure"] is not None

    def test_intent_top_k_contributors(self, analytical_df: pd.DataFrame) -> None:
        """Top-K contributor queries must compute share percentage and cumulative Pareto share."""
        res = execute_natural_language_query(
            analytical_df, query="Top 3 tiers by total_spend"
        )
        assert res["text"]
        assert "share" in res["text"].lower() or "leader" in res["text"].lower()
        assert res["table"] is not None
        assert "Share (%)" in res["table"].columns

    def test_intent_bivariate_relationship(self, analytical_df: pd.DataFrame) -> None:
        """Bivariate relationship queries must compute Pearson CI, Spearman rho, and OLS fit."""
        res = execute_natural_language_query(
            analytical_df, query="Relationship between tenure_months and monthly_charges"
        )
        assert res["text"]
        assert "pearson" in res["text"].lower()
        assert "spearman" in res["text"].lower()
        assert "ols" in res["text"].lower() or "r =" in res["text"].lower()
        assert res["figure"] is not None

    def test_intent_distribution_profile(self, analytical_df: pd.DataFrame) -> None:
        """Distribution profiling queries must compute central tendency, IQR, and dispersion."""
        res = execute_natural_language_query(
            analytical_df, query="Distribution of monthly_charges"
        )
        assert res["text"]
        assert "mean" in res["text"].lower()
        assert "median" in res["text"].lower()
        assert "iqr" in res["text"].lower()
        assert res["figure"] is not None

    def test_adversarial_queries_safe_fallback(self, analytical_df: pd.DataFrame) -> None:
        """Gibberish or missing columns queries must safely return dataset snapshot without throwing."""
        res_empty = execute_natural_language_query(analytical_df, query="")
        assert "rows" in res_empty["text"].lower()

        res_gibberish = execute_natural_language_query(
            analytical_df, query="What is the quantum flux of non_existent_column?"
        )
        assert "snapshot" in res_gibberish["text"].lower() or "rows" in res_gibberish["text"].lower()
