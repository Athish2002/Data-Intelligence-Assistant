"""
dia/experimentation.py
──────────────────────
Statistical A/B Testing Planner.
Calculates required sample sizes and minimum detectable effects for deploying models.
"""
import math

import scipy.stats as stats


def calculate_ab_test_sample_size(
    baseline_conversion: float,
    expected_lift: float,
    power: float = 0.8,
    alpha: float = 0.05
) -> dict:
    """
    Calculate the required sample size per variant for a two-tailed A/B test.

    baseline_conversion: Current baseline rate (e.g., 0.10 for 10% conversion)
    expected_lift: Expected relative improvement (e.g., 0.20 for 20% relative lift -> new rate 12%)
    power: Statistical power (1 - Type II error)
    alpha: Significance level (Type I error)
    """
    if baseline_conversion <= 0 or baseline_conversion >= 1:
        raise ValueError("Baseline conversion must be strictly between 0 and 1.")

    p1 = baseline_conversion
    p2 = baseline_conversion * (1 + expected_lift)

    # Cap target probabilities safely within (0.0001, 0.9999)
    p2 = max(0.0001, min(0.9999, p2))

    if abs(p1 - p2) < 1e-5:
        raise ValueError("Expected lift is too close to zero to calculate sample size.")

    # Standard normal deviates
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Pooled variance
    p_pool = (p1 + p2) / 2

    # Required sample size formula (Fleiss / Lehr formula for two-sample proportion test)
    numerator = ((z_alpha * math.sqrt(2 * p_pool * (1 - p_pool))) +
                 (z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))) ** 2
    denominator = (p1 - p2) ** 2

    sample_size = int(math.ceil(numerator / denominator))

    return {
        "baseline_rate": round(p1, 4),
        "target_rate": round(p2, 4),
        "absolute_mde": round(p2 - p1, 4),
        "relative_mde": round(expected_lift, 4),
        "sample_size_per_variant": sample_size,
        "total_traffic_required": sample_size * 2,
        "statistical_power": power,
        "significance_level": alpha
    }
