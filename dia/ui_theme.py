"""
dia/ui_theme.py
───────────────
Unified Design System, Custom CSS Tokens, and Glassmorphic UX Components
for the Data Intelligence Assistant (DIA) enterprise interface.
"""

from __future__ import annotations

import streamlit as st


def inject_custom_theme() -> None:
    """Injects high-performance, responsive Glassmorphic CSS into the Streamlit app."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --primary: #00e575;
    --primary-glow: rgba(0, 229, 117, 0.25);
    --secondary: #ded7cb;
    --accent: #f59e0b;
    --cyan: #00e575;
    --emerald: #00e575;
    --amber: #f59e0b;
    --rose: #f43f5e;

    --bg-dark: #050505;
    --card-bg: #0e0e0e;
    --card-border: #1f1f1f;
    --card-hover-border: #333333;
    --radius: 8px;
    --radius-sm: 6px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.01em;
    background-color: #050505 !important;
    color: #ded7cb !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar Flat Architectural Black */
section[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid var(--card-border);
}
section[data-testid="stSidebar"] * {
    color: #ded7cb !important;
}

/* App Main Container */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1350px;
    background-color: #050505;
}

/* Flat Metric Cards */
[data-testid="metric-container"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
    transition: border-color 0.15s ease;
}

[data-testid="metric-container"]:hover {
    border-color: var(--card-hover-border);
}

/* Flat Phosphor Green Primary Buttons */
.stButton > button {
    background: #00e575 !important;
    color: #050505 !important;
    border: 1px solid #00e575 !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    transition: background-color 0.15s ease !important;
}

.stButton > button:hover {
    background: #00ff82 !important;
    border-color: #00ff82 !important;
    color: #000000 !important;
}

.stDownloadButton > button {
    background: #141414 !important;
    color: #f5f0e6 !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid #282828 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
}

.stDownloadButton > button:hover {
    background: #1c1c1c !important;
    border-color: #383838 !important;
}

/* Tabs Navigation Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(18, 20, 29, 0.4);
    padding: 6px;
    border-radius: var(--radius);
    border: 1px solid var(--card-border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 6px 14px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #94a3b8;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%) !important;
    color: #c7d2fe !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
}

/* Hero Banner */
.dia-hero-container {
    background: linear-gradient(135deg, rgba(15, 17, 26, 0.9) 0%, rgba(24, 27, 44, 0.9) 50%, rgba(20, 14, 38, 0.9) 100%);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}

.dia-hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
    pointer-events: none;
}

.dia-hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.35rem;
}

.dia-hero-subtitle {
    color: #94a3b8;
    font-size: 0.98rem;
    line-height: 1.5;
    margin: 0;
}

/* Mission Control Ribbon */
.mission-ribbon {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    background: rgba(18, 20, 29, 0.6);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.25rem;
    align-items: center;
    justify-content: space-between;
}

.ribbon-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #cbd5e1;
}

.dot-online {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero_banner() -> None:
    """Renders the styled hero header banner."""
    st.markdown(
        """
<div class="dia-hero-container">
  <div class="dia-hero-title">🧠 Data Intelligence Assistant (DIA Enterprise v11.0)</div>
  <p class="dia-hero-subtitle">
    Autonomous Machine Learning, Causal Inference, Time-Series Forecasting, Deep Learning Anomaly Manifolds,
    Contextual Bandits, Regulatory Data Governance (GDPR/HIPAA/SOC2), and Native In-Database SQL Transpilation.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_mission_ribbon(
    persona: str,
    target_col: str,
    task_type: str,
    best_model: str,
    latency_p99: float,
    privacy_risk: int,
) -> None:
    """Renders real-time mission control status ribbon at the top of the analytics center."""
    privacy_color = "#10b981" if privacy_risk == 0 else "#f59e0b"
    privacy_status = "PASSED & SECURED" if privacy_risk == 0 else f"{privacy_risk} Risks Detected"

    st.markdown(
        f"""
<div class="mission-ribbon">
  <div class="ribbon-pill">
    <div class="dot-online"></div>
    <span>Status: <strong>PIPELINE CERTIFIED</strong></span>
  </div>
  <div class="ribbon-pill">
    <span>👤 Persona: <strong style="color:#818cf8;">{persona}</strong></span>
  </div>
  <div class="ribbon-pill">
    <span>🎯 Target: <strong>{target_col}</strong> ({task_type})</span>
  </div>
  <div class="ribbon-pill">
    <span>🏆 Champion: <strong style="color:#c084fc;">{best_model}</strong></span>
  </div>
  <div class="ribbon-pill">
    <span>⚡ SLA p99: <strong>{latency_p99:.2f} ms</strong></span>
  </div>
  <div class="ribbon-pill">
    <span>🔒 Compliance: <strong style="color:{privacy_color};">{privacy_status}</strong></span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
