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
    --primary: #6366f1;
    --primary-glow: rgba(99, 102, 241, 0.35);
    --secondary: #a855f7;
    --accent: #ec4899;
    --cyan: #06b6d4;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;

    --bg-dark: #090a0f;
    --card-bg: rgba(18, 20, 29, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-hover-border: rgba(99, 102, 241, 0.4);
    --radius: 14px;
    --radius-sm: 8px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.01em;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar Glassmorphism */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090a10 0%, #11131f 100%);
    border-right: 1px solid var(--card-border);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* App Main Container */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1350px;
}

/* Glassmorphic Metric Cards */
[data-testid="metric-container"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(12px);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="metric-container"]:hover {
    border-color: var(--card-hover-border);
    transform: translateY(-2px);
    box-shadow: 0 12px 30px -6px var(--primary-glow);
}

/* Buttons with Gradient Glow */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
}

.stButton > button:hover {
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #059669 100%) !important;
    color: #ffffff !important;
    border-radius: var(--radius-sm) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px 0 rgba(13, 148, 136, 0.35) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px 0 rgba(5, 150, 105, 0.5) !important;
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
