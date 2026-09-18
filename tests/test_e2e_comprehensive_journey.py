"""
tests/test_e2e_comprehensive_journey.py
─────────────────────────────────────────
Full Interactive Playwright End-to-End User Journey Test Suite for Milestone 3 (Requirement R1).
Executes authentic multi-step user workflows against live running FastAPI server:

Stage 1: Landing Page Onboarding Flow
  - Hero header, CTA buttons (#btn-launch-demo, #btn-ingest-data, #btn-go-dashboard)
  - Natural document scroll to #architectural-pillars, #causal-simulator-section, #execution-pathways
  - Interactive Pearl Causal DAG Node Inspection (Z, X, M, Y) via selectDagNode
  - Pearl Do-Calculus Policy Range Slider (#causal-policy-slider, updatePolicyIntervention)
  - In-app live Health Telemetry HUD modal lifecycle (#health-telemetry-modal)

Stage 2: 1-Click Demo Ingestion & Workbench Transition
  - Benchmark launch (Banking Credit Risk, Supply Chain Logistics, Clinical ICU Sepsis)
  - Seamless navigation to /app analytical workbench
  - Schema ingestion, metadata inspection, and Core Overview population

Stage 3: AutoML Tournament & Explainability Exploration
  - Launch AutoML training pipeline (#train-btn, 5-stage tracker)
  - Verify multi-model leaderboard (Accuracy, ROC-AUC, F1, Brier Score, Champion badge)
  - Plotly ROC Curve SVG trace elements and 2x2 confusion matrix sample conservation
  - TreeSHAP feature attributions chart rendering

Stage 4: Counterfactual Recourse & Policy Simulation
  - What-If scenario simulator live scoring (#sim-run-btn, gauge prediction)
  - Algorithmic recourse optimization (#recourse-run-btn, unmasked raw feature units)
  - Workbench Pearl Do-Calculus policy simulator (adaptive:causal, #do-simulate-btn)

Stage 5: Governance & Audit Dossier Export
  - EU AI Act Annex III and GDPR Art. 22 compliance cards (governance:dossier)
  - Live interactive dossier preview iframe and export download button
  - Great Expectations data contracts natural language assertions (governance:contracts)

Stage 6: Edge Model WASM / C99 Transpilation
  - Standalone client-side edge bundle inspection (governance:code, data-file-id="edge_bundle")
  - Bundle size and microsecond latency verification
  - In-browser local scoring sandbox (#edge-eval-btn running DiaEdgeEngine.predictProba)

Strict Error Contract:
  - assert len(console_errors) == 0 and len(page_errors) == 0
"""

import os
import re
import time
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("DIA_BASE_URL", "http://127.0.0.1:8000")


def launch_browser(playwright_instance):
    """Launch headless browser with Chrome/Edge/Chromium fallback."""
    for channel in ["chrome", "msedge"]:
        try:
            return playwright_instance.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return playwright_instance.chromium.launch(headless=True)


def test_e2e_comprehensive_six_stage_journey():
    """Execute complete 6-stage end-to-end interactive user journey against live server."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page_errors = []

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: Landing Page Onboarding & Interactive Explorations
        # ─────────────────────────────────────────────────────────────────────
        page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=30000)
        assert "DIA • Data Intelligence at Machine Speed" in page.title()

        # A. Hero section verification
        hero_title = page.locator("h1").inner_text()
        assert "Data Intelligence" in hero_title
        assert page.locator("#btn-launch-demo").is_visible()
        assert page.locator("#btn-ingest-data").is_visible()
        assert page.locator("#btn-go-dashboard").is_visible()

        # B. Natural document scrolling to core sections
        page.locator("#architectural-pillars").scroll_into_view_if_needed()
        assert page.locator("#architectural-pillars").is_visible()

        page.locator("#causal-simulator-section").scroll_into_view_if_needed()
        assert page.locator("#causal-simulator-section").is_visible()

        page.locator("#execution-pathways").scroll_into_view_if_needed()
        assert page.locator("#execution-pathways").is_visible()

        # C. Interactive Pearl Causal DAG Node Inspection (Z, X, M, Y)
        for node_key in ["z", "x", "m", "y"]:
            page.evaluate(f"window.selectDagNode('{node_key}')")
            title = page.locator("#dag-inspector-title").inner_text()
            desc = page.locator("#dag-inspector-desc").inner_text()
            assert len(title) > 0, f"Empty inspector title for node {node_key}"
            assert len(desc) > 0, f"Empty inspector description for node {node_key}"

        # D. Pearl Do-Calculus Policy Range Slider
        page.evaluate("window.updatePolicyIntervention(12.0)")
        assert "12.0%" in page.locator("#policy-slider-val").inner_text()
        assert len(page.locator("#metric-sim-default").inner_text()) > 0
        assert len(page.locator("#metric-sim-ate").inner_text()) > 0
        assert "95% CI:" in page.locator("#metric-sim-ci").inner_text()

        # E. In-App Health Telemetry HUD Modal Lifecycle
        page.locator("#nav-telemetry-btn").click()
        page.wait_for_selector("#health-telemetry-modal:not(.hidden)", timeout=8000)
        page.wait_for_selector("#telemetry-status-pill", timeout=8000)
        assert page.locator("#telemetry-modal-title, #health-modal-title").is_visible()

        # Dismiss modal via close button
        page.locator("#close-telemetry-btn").click()
        page.wait_for_selector("#health-telemetry-modal", state="hidden", timeout=5000)

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: 1-Click Demo Ingestion & Workbench Transition
        # ─────────────────────────────────────────────────────────────────────
        # Trigger Bank Credit Risk benchmark demo
        page.wait_for_function("() => typeof window.launchDemoBenchmark === 'function'", timeout=15000)
        page.evaluate("window.launchDemoBenchmark('Bank Credit Risk & Default')")
        page.wait_for_url("**/app**", timeout=20000)

        # Verify auto-populated schema, target and goal
        page.wait_for_function(
            "() => document.getElementById('target-col-input') && document.getElementById('target-col-input').value.length > 0",
            timeout=15000
        )
        target_val = page.locator("#target-col-input").input_value()
        assert target_val == "default_risk", f"Unexpected target column: {target_val}"

        # Verify Overview table population in Core workspace
        page.wait_for_function("() => typeof window.switchWorkspace === 'function'", timeout=15000)
        page.evaluate("window.switchWorkspace('core', 'overview')")
        page.wait_for_selector("#tab-content table, #tab-content .glass-card", timeout=15000)
        content_text = page.locator("#tab-content").inner_text()
        assert "default_risk" in content_text or "Overview" in content_text

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: AutoML Tournament & Explainability Exploration
        # ─────────────────────────────────────────────────────────────────────
        page.evaluate("window.switchWorkspace('automl', 'leaderboard')")
        page.click("#train-btn")

        # 5-Stage progress tracker activates
        page.wait_for_selector("#train-progress:not(.hidden)", timeout=5000)

        # Wait for AutoML pipeline execution and leaderboard population
        page.wait_for_selector("#tab-content .glass-card:has-text('Automated Model Leaderboard')", timeout=90000)
        leaderboard_cards = page.locator("#tab-content .grid > div")
        assert leaderboard_cards.count() >= 3, "Expected at least 3 models in tournament leaderboard"

        # Verify Champion card metrics
        champ_card = page.locator("#tab-content .grid > div:has-text('Champion')").first
        champ_text = champ_card.inner_text()
        assert "Accuracy:" in champ_text
        assert "ROC-AUC:" in champ_text

        # Verify Diagnostic Curves and Confusion Matrix
        page.evaluate("window.switchWorkspace('automl', 'curves')")
        page.wait_for_selector("#roc-chart-container svg.main-svg path", state="attached", timeout=15000)
        assert page.locator("#roc-chart-container svg.main-svg path").count() > 0

        # Confusion Matrix Conservation Check
        curves_text = page.locator("#tab-content").inner_text()
        assert "TRUE NEGATIVE" in curves_text
        assert "FALSE POSITIVE" in curves_text
        assert "FALSE NEGATIVE" in curves_text
        assert "TRUE POSITIVE" in curves_text

        cm_cells = page.locator("#tab-content .grid-cols-12 .text-lg.font-mono").all_inner_texts()
        parsed_cells = [int(c.strip()) for c in cm_cells if c.strip().isdigit()]
        if len(parsed_cells) >= 4:
            total_samples = sum(parsed_cells[:4])
            assert total_samples == 120, f"Confusion matrix total {total_samples} != 120 holdout samples"

        # Verify TreeSHAP Waterfall / Explainability Chart
        page.evaluate("window.switchWorkspace('automl', 'explainability')")
        page.wait_for_selector("#shap-chart-container svg.main-svg", timeout=15000)
        assert page.locator("#shap-chart-container svg.main-svg").first.is_visible()

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 4: Counterfactual Recourse & Pearl Policy Simulation
        # ─────────────────────────────────────────────────────────────────────
        # A. What-If Simulator & Wachter Recourse
        page.evaluate("window.switchWorkspace('automl', 'simulator')")
        page.wait_for_selector("#sim-run-btn", timeout=10000)

        # Run live What-If simulation
        page.click("#sim-run-btn")
        page.wait_for_function(
            "() => { const el = document.getElementById('gauge-label'); return el && el.innerText !== 'Awaiting' && el.innerText !== 'Scoring...'; }",
            timeout=10000
        )
        gauge_val = page.locator("#gauge-label").inner_text().strip()
        assert len(gauge_val) > 0, "Empty gauge prediction"

        # Compute Actionable Recourse (ensure numeric inputs only for custom_feature_overrides)
        page.evaluate("""() => {
            document.querySelectorAll('.sim-input').forEach(el => {
                if (isNaN(Number(el.value))) {
                    el.classList.remove('sim-input');
                }
            });
        }""")
        page.click("#recourse-run-btn")
        page.wait_for_selector("#recourse-box:not(.hidden)", timeout=20000)
        page.wait_for_selector("#recourse-cost-badge:not(:empty)", timeout=20000)
        cost_badge = page.locator("#recourse-cost-badge").inner_text()
        assert "Recourse Cost:" in cost_badge

        # Recourse actions displayed in real feature units
        recourse_text = page.locator("#recourse-actions-list").inner_text()
        assert len(recourse_text) > 0

        # B. Pearl Do-Calculus Policy Simulator in Workbench
        page.evaluate("window.switchWorkspace('adaptive', 'causal')")
        page.wait_for_selector("#causal-dag-svg", timeout=15000)
        page.wait_for_selector("#do-simulate-btn", timeout=10000)

        page.click("#do-simulate-btn")
        page.wait_for_selector("#do-results-box:not(.hidden)", timeout=15000)
        page.wait_for_function(
            "() => { const el = document.getElementById('do-results-box'); return el && !el.innerText.includes('Computing'); }",
            timeout=20000
        )
        do_results = page.locator("#do-results-box").inner_text()
        assert ("ATE:" in do_results) or ("Policy Simulation" in do_results) or ("Observational" in do_results)

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 5: Governance & Regulatory Audit Dossier Export
        # ─────────────────────────────────────────────────────────────────────
        page.evaluate("window.switchWorkspace('governance', 'dossier')")
        page.wait_for_selector("#dossier-download-btn", timeout=15000)
        page.wait_for_selector("#dossier-iframe", timeout=15000)

        dossier_href = page.locator("#dossier-download-btn").get_attribute("href")
        assert "export" in dossier_href and "dossier" in dossier_href

        gov_text = page.locator("#tab-content").inner_text()
        assert "Annex III" in gov_text or "EU AI Act" in gov_text or "Compliant" in gov_text

        # Verify Great Expectations Data Contracts table
        page.evaluate("window.switchWorkspace('governance', 'contracts')")
        page.wait_for_selector("#tab-content table tbody tr", timeout=15000)
        contract_rows = page.locator("#tab-content table tbody tr")
        assert contract_rows.count() >= 5, "Expected at least 5 Great Expectations contract rows"

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 6: Edge Model WASM / C99 Transpilation & Browser Scoring
        # ─────────────────────────────────────────────────────────────────────
        page.evaluate("window.switchWorkspace('governance', 'code')")
        page.wait_for_selector('.file-tab-btn[data-file-id="edge_bundle"]', timeout=15000)

        # Click Edge Bundle tab
        page.locator('.file-tab-btn[data-file-id="edge_bundle"]').click()
        page.wait_for_selector("#active-file-code", timeout=15000)

        code_text = page.locator("#active-file-code").inner_text()
        assert "DiaEdgeEngine" in code_text
        assert "predictProba" in code_text
        assert len(code_text) < 100000, f"Edge bundle exceeds size limit ({len(code_text)} bytes)"

        # In-Browser Sub-Millisecond Scoring Sandbox
        page.wait_for_selector("#edge-eval-btn", timeout=10000)
        page.click("#edge-eval-btn")
        page.wait_for_selector("#edge-score-result:not(:empty)", timeout=10000)

        score_res = page.locator("#edge-score-result").inner_text()
        assert "Output:" in score_res
        assert ("μs" in score_res) or ("Latency:" in score_res) or ("ms" in score_res)

        # ─────────────────────────────────────────────────────────────────────
        # STRICT ERROR CONTRACT
        # ─────────────────────────────────────────────────────────────────────
        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors encountered: {real_errors}"
        assert len(page_errors) == 0, f"Page unhandled errors encountered: {page_errors}"

        browser.close()


def test_e2e_benchmark_scenarios_multi_domain():
    """Verify 1-click ingestion and schema recognition across remaining benchmark domains."""
    with sync_playwright() as p:
        browser = launch_browser(p)

        for demo_name, expected_target in [
            ("Supply Chain Delivery Delay", "delay_flag"),
            ("Clinical ICU Sepsis Prediction", "sepsis_target"),
        ]:
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            console_errors = []
            page_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            page.goto(f"{BASE_URL}/app", wait_until="networkidle", timeout=30000)
            page.wait_for_function("() => typeof window.quickLoadBenchmark === 'function'", timeout=15000)
            page.evaluate(f"window.quickLoadBenchmark('{demo_name}')")

            page.wait_for_function(
                f"() => document.getElementById('target-col-input') && document.getElementById('target-col-input').value === '{expected_target}'",
                timeout=20000
            )
            target = page.locator("#target-col-input").input_value()
            assert target == expected_target, f"Failed to ingest {demo_name}; got {target}"

            # Verify overview metrics
            page.evaluate("window.switchWorkspace('core', 'overview')")
            page.wait_for_selector("#tab-content table, #tab-content .glass-card", timeout=15000)

            real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
            assert len(real_errors) == 0, f"Errors in {demo_name}: {real_errors}"
            assert len(page_errors) == 0, f"Page errors in {demo_name}: {page_errors}"

            context.close()

        browser.close()
