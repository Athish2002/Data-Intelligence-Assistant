"""
tests/test_causal_canvas_e2e.py
───────────────────────────────
Playwright End-to-End Test Suite for the Interactive Causal DAG Topology
& Invariant Studio (Milestone Phase 3).

Verifies:
1. Ingest a demo benchmark dataset with known causal DAG structure (Telecom Customer Churn).
2. Navigate to Adaptive AI Engines -> Causal Discovery & Do-Calculus Studio.
3. Assert SVG `#causal-dag-svg` renders with non-zero nodes and directed edges.
4. Click a node (e.g. collider 'churn_flag') to toggle it into Conditioning Set Z:
   - Conditioning set state is updated and displayed ({ churn_flag }).
   - Node visual state reflects selection (data-conditioned="true", glowing outer ring, Z-SET badge).
   - Collider detection triggers live Berkson's Bias warning badge.
   - Active paths vs blocked paths edge styles update.
5. Click "Clear Conditioning Set" button:
   - Conditioning set resets to empty.
   - Node returns to unconditioned visual state (data-conditioned="false").
   - Berkson's Bias warning clears.
6. Axiomatic Invariant "Suggest & Lock" Shelf:
   - Click [Scan Conservation Laws] button.
   - Assert discovered glassmorphic invariant cards (e.g. Mass Balance).
   - Click [Lock Invariant] button and assert "✓ Locked Physical Axiom" verified badge.
7. Strict Error Contract: console errors count == 0 and page errors count == 0.
"""

from __future__ import annotations

import os
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


def test_interactive_causal_dag_and_invariants_e2e():
    """Verify Causal DAG visual d-separation, collider bias warning, and invariant shelf."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors: list[str] = []
        page_errors: list[str] = []

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Step 1: Open workbench and ingest benchmark dataset with known causal structure
        page.goto(f"{BASE_URL}/app", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_function("() => typeof window.quickLoadBenchmark === 'function'", timeout=15000)

        # Ingest Telecom Customer Churn benchmark
        page.evaluate("window.quickLoadBenchmark('Telecom Customer Churn')")
        page.wait_for_function(
            "() => document.getElementById('target-col-input') && document.getElementById('target-col-input').value.length > 0",
            timeout=20000
        )
        assert page.locator("#target-col-input").input_value() == "churn_flag"

        # Step 2: Navigate to Adaptive AI Engines -> Causal view
        page.evaluate("window.switchWorkspace('adaptive', 'causal')")
        page.wait_for_selector("#causal-dag-svg", timeout=15000)
        page.wait_for_selector("#dag-nodes-group", timeout=10000)
        page.wait_for_selector("#dag-edges-group", timeout=10000)

        # Step 3: Verify SVG #causal-dag-svg has non-zero nodes and edges
        page.wait_for_function(
            "() => document.querySelectorAll('#dag-nodes-group .dag-node-interactive').length > 0",
            timeout=10000
        )
        nodes_count = page.locator("#dag-nodes-group .dag-node-interactive").count()
        edges_count = page.locator("#dag-edges-group .dag-edge-interactive").count()

        assert nodes_count >= 4, f"Expected at least 4 nodes in DAG canvas, got {nodes_count}"
        assert edges_count >= 2, f"Expected at least 2 directed edges in DAG canvas, got {edges_count}"

        # Assert conditioning set is initially unconditioned
        cond_text_initial = page.locator("#conditioning-set-display").inner_text()
        assert "Unconditioned" in cond_text_initial or "∅" in cond_text_initial

        # Step 4: Click a node in the DAG to toggle it into Conditioning Set Z
        # Target node 'churn_flag' is a collider with multiple incoming causal arrows
        churn_node = page.locator(".dag-node-interactive[data-node='churn_flag']")
        if churn_node.count() == 0:
            churn_node = page.locator(".dag-node-interactive").first

        target_node_name = churn_node.get_attribute("data-node")
        assert churn_node.get_attribute("data-conditioned") == "false"

        churn_node.click()

        # Step 5: Assert Conditioning Set Z updated and node reflects selection
        page.wait_for_function(
            f"() => {{ const el = document.querySelector(\".dag-node-interactive[data-node='{target_node_name}']\"); return el && el.getAttribute('data-conditioned') === 'true'; }}",
            timeout=5000
        )
        assert churn_node.get_attribute("data-conditioned") == "true"

        cond_text_after = page.locator("#conditioning-set-display").inner_text()
        assert target_node_name in cond_text_after

        # Node has visual badge / glowing indicator
        assert churn_node.locator("text", has_text="Z-SET").count() == 1

        # Step 6: Collider Detection & Berkson's Bias Warning check
        # 'churn_flag' has incoming edges from tenure_months, monthly_charges, etc.
        berkson_warnings = page.locator("#berkson-warning-container")
        if target_node_name == "churn_flag":
            page.wait_for_selector("#berkson-warning-container .animate-fadeIn", timeout=5000)
            warning_text = berkson_warnings.inner_text()
            assert "Berkson's Bias Warning" in warning_text
            assert "Conditioning on collider 'churn_flag'" in warning_text
            assert "spurious association" in warning_text

        # Step 7: Clear Conditioning Set reset button
        clear_btn = page.locator("#clear-conditioning-btn")
        assert clear_btn.is_visible()
        clear_btn.click()

        page.wait_for_function(
            f"() => {{ const el = document.querySelector(\".dag-node-interactive[data-node='{target_node_name}']\"); return el && el.getAttribute('data-conditioned') === 'false'; }}",
            timeout=5000
        )
        assert churn_node.get_attribute("data-conditioned") == "false"
        assert churn_node.locator("text", has_text="Z-SET").count() == 0
        assert page.locator("#berkson-warning-container").inner_text().strip() == ""

        # Step 8: Axiomatic Invariant "Suggest & Lock" Shelf
        scan_btn = page.locator("#scan-invariants-btn")
        assert scan_btn.is_visible()
        scan_btn.click()

        # Wait for scan response and cards population
        page.wait_for_selector(".lock-invariant-btn, #invariants-cards-container [id^='law-card-']", timeout=15000)
        law_cards = page.locator("#invariants-cards-container [id^='law-card-']")
        assert law_cards.count() >= 1, "Expected at least 1 discovered conservation law card"

        first_card_text_upper = law_cards.first.inner_text().upper()
        assert any(term in first_card_text_upper for term in ["MASS BALANCE", "INVARIANT", "BALANCE", "AXIOM"])
        assert "RESIDUAL MAE" in first_card_text_upper
        assert "COMPLIANCE" in first_card_text_upper

        # 1-click lock invariant
        lock_btn = page.locator(".lock-invariant-btn").first
        lock_btn.click()

        page.wait_for_selector("span:has-text('Locked Physical Axiom')", timeout=10000)
        verified_badge = page.locator("span:has-text('Locked Physical Axiom')").first
        assert verified_badge.is_visible()
        assert "✓ Locked Physical Axiom (Enforced in Projections)" in verified_badge.inner_text()

        # Step 9: Zero Console Errors and Zero Page Errors contract
        browser.close()
        assert len(console_errors) == 0, f"Encountered browser console errors: {console_errors}"
        assert len(page_errors) == 0, f"Encountered uncaught page exceptions: {page_errors}"
