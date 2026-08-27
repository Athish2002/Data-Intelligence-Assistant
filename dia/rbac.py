"""
dia/rbac.py
───────────
Enterprise Role-Based Access Control (RBAC) & Multi-Tenancy Engine.
Enforces security context and feature permissions across user personas.
"""

from __future__ import annotations


# ─── RBAC Personas & Tab Access Permissions ───────────────────────────────────

ROLE_PERMISSIONS: dict[str, list[str]] = {
    "Admin (Full Access)": [
        "📊 Overview", "🔍 Column Roles", "🩺 Readiness", "💡 Smart Insights",
        "🤖 Model Results", "🔬 Explainability", "💼 Business Impact", "🎛️ Simulator",
        "🎯 Causal & Counterfactuals", "📈 Time-Series Forecast", "🧬 Synthetic Data",
        "⚡ Online Learning", "💬 Chat Copilot", "🌊 Drift Monitor",
        "🔒 Privacy & Compliance", "📦 Model Registry & MLOps", "🛡️ Data Contract",
        "🧪 A/B Test Planner", "👥 Active Learning", "📑 Executive Briefing", "💻 Code Export", "📝 Summary"
    ],
    "Lead Data Scientist": [
        "📊 Overview", "🔍 Column Roles", "🩺 Readiness", "💡 Smart Insights",
        "🤖 Model Results", "🔬 Explainability", "🎛️ Simulator", "🎯 Causal & Counterfactuals",
        "📈 Time-Series Forecast", "🧬 Synthetic Data", "⚡ Online Learning", "💬 Chat Copilot",
        "🌊 Drift Monitor", "📦 Model Registry & MLOps", "🛡️ Data Contract", "👥 Active Learning",
        "💻 Code Export", "📝 Summary"
    ],
    "Business Analyst / Executive": [
        "📊 Overview", "💡 Smart Insights", "💼 Business Impact", "🎛️ Simulator",
        "🎯 Causal & Counterfactuals", "📈 Time-Series Forecast", "💬 Chat Copilot",
        "🧪 A/B Test Planner", "📑 Executive Briefing", "📝 Summary"
    ],
    "Compliance & Privacy Officer": [
        "📊 Overview", "🔒 Privacy & Compliance", "🛡️ Data Contract",
        "📑 Executive Briefing", "📝 Summary"
    ],
}


def filter_tabs_for_user_role(role: str, all_tab_names: list[str]) -> list[str]:
    """
    Returns only the subset of dashboard tabs permitted for the given user role.
    """
    allowed = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["Admin (Full Access)"])
    return [t for t in all_tab_names if t in allowed]
