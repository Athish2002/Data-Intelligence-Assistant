"""
tests/test_interactive_landing_features.py
──────────────────────────────────────────
Targeted Playwright test verifying:
1. Top bar authentic reframing (Capabilities, Architecture, Docs/API, Health Telemetry, Launch Workbench).
2. Elimination of uncontrolled polling loops (/api/v1/system/metrics and /api/v1/causal/graph/...).
3. Elimination of custom cursor left-edge sticking on move, click, and idle.
4. Telemetry modal open/close lifecycle, data rendering, and auto-refresh termination.
5. Mobile navigation drawer reframing.
6. Zero console errors and zero page errors across all interactions.
"""

import time
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"


def test_landing_page_interactive_full():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page_errors = []
        network_requests = []

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("request", lambda req: network_requests.append(req.url))

        # ─── 1. Load Landing Page ──────────────────────────────────────────────
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
        time.sleep(1)

        # ─── 2. Top Bar Reframing Verification ─────────────────────────────────
        # A. Brand Logo & Data Intelligence Badge
        brand_logo = page.locator("header a[title='Data Intelligence Assistant']")
        assert brand_logo.count() == 1, "Brand logo link not found"
        badge = page.locator("header span:has-text('Data Intelligence')")
        assert badge.count() >= 1, "Data Intelligence badge missing"

        # B. Capabilities Dropdown
        cap_group = page.locator("#nav-capabilities-group")
        assert cap_group.count() == 1, "#nav-capabilities-group missing"
        assert "Capabilities" in cap_group.inner_text()
        # Verify capabilities items
        assert page.locator("header a:has-text('Autonomous AutoML Ensembles')").count() >= 1
        assert page.locator("header a:has-text('Pearl Observational Causal DAG')").count() >= 1
        assert page.locator("header a:has-text('Conformal Coverage Sets')").count() >= 1
        assert page.locator("header a:has-text('Pre-computed C99 & WASM Scoring')").count() >= 1
        assert page.locator("header a:has-text('Governance & Audit Dossiers')").count() >= 1

        # C. Architecture Link
        arch_link = page.locator("nav a[href='#architectural-pillars']")
        assert arch_link.count() == 1, "Architecture link to #architectural-pillars missing"

        # D. Docs / API Dropdown
        docs_group = page.locator("#nav-docs-group")
        assert docs_group.count() == 1, "#nav-docs-group missing"
        assert page.locator("header a[href='/docs']").count() >= 1, "Swagger /docs link missing"
        assert page.locator("header a[href='/redoc']").count() >= 1, "ReDoc /redoc link missing"

        # E. Health Telemetry Direct Trigger
        telemetry_btn = page.locator("#nav-telemetry-btn")
        assert telemetry_btn.count() == 1, "#nav-telemetry-btn trigger missing"
        assert telemetry_btn.is_visible(), "#nav-telemetry-btn should be visible"

        # F. Launch Workbench CTA
        workbench_btn = page.locator("#nav-launch-workbench-btn")
        assert workbench_btn.count() == 1, "#nav-launch-workbench-btn missing"
        assert "Launch Workbench" in workbench_btn.inner_text()

        # ─── 3. Custom Cursor Physics & Left-Edge Verification ─────────────────
        dot = page.locator("#cursor-dot")
        ring = page.locator("#cursor-ring")
        aura = page.locator("#cursor-aura")

        # Before any mousemove, verify elements exist and do not cause document horizontal overflow
        has_h_overflow = page.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth")
        assert not has_h_overflow, "Custom cursor or layout caused horizontal document overflow"

        # Move mouse to (500, 350)
        page.mouse.move(500, 350)
        time.sleep(0.2)

        dot_transform = page.evaluate("() => document.getElementById('cursor-dot').style.transform")
        assert "translate3d(500px, 350px, 0px)" in dot_transform, f"Dot did not move to 500,350: {dot_transform}"

        cx_val = page.evaluate("() => document.documentElement.style.getPropertyValue('--cx')")
        cy_val = page.evaluate("() => document.documentElement.style.getPropertyValue('--cy')")
        assert cx_val == "500px"
        assert cy_val == "350px"

        # Click and verify ring does NOT jump to 0,0 or left edge
        page.mouse.down()
        time.sleep(0.05)
        ring_transform_down = page.evaluate("() => document.getElementById('cursor-ring').style.transform")
        assert "translate3d(0px, 0px" not in ring_transform_down, f"Ring jumped to 0,0 on click: {ring_transform_down}"
        assert "scale" in ring_transform_down

        page.mouse.up()
        time.sleep(0.1)

        # ─── 4. Health Telemetry Modal & Polling Loop Verification ─────────────
        # Record request counts before opening modal
        initial_health_reqs = sum(1 for url in network_requests if "/api/v1/health" in url)
        initial_metrics_reqs = sum(1 for url in network_requests if "/api/v1/system/metrics" in url)
        initial_causal_reqs = sum(1 for url in network_requests if "/api/v1/causal" in url)

        # Ensure no causal network requests are happening on landing page
        assert initial_causal_reqs == 0, f"Unexpected causal requests on landing page: {initial_causal_reqs}"

        # Click Health Telemetry trigger button
        telemetry_btn.click()
        time.sleep(0.8)

        modal = page.locator("#health-telemetry-modal")
        assert modal.is_visible(), "Health telemetry modal failed to open"

        # Verify rendered metrics
        cpu_val = page.locator("#telemetry-cpu-val")
        ram_val = page.locator("#telemetry-ram-val")
        storage_val = page.locator("#telemetry-storage-val")
        assert cpu_val.is_visible(), "CPU core metric card not rendered"
        assert ram_val.is_visible(), "RAM metric card not rendered"
        assert storage_val.is_visible(), "Storage metric card not rendered"

        # Verify health probe was fetched
        post_open_health_reqs = sum(1 for url in network_requests if "/api/v1/health" in url)
        assert post_open_health_reqs > initial_health_reqs, "Expected /api/v1/health fetch on modal open"

        # Strictly ZERO calls to /api/v1/system/metrics on landing page
        post_open_metrics_reqs = sum(1 for url in network_requests if "/api/v1/system/metrics" in url)
        assert post_open_metrics_reqs == 0, f"Expected 0 /api/v1/system/metrics requests on landing page, got {post_open_metrics_reqs}"

        # Test visibilitychange: hiding document halts polling
        page.evaluate("() => { Object.defineProperty(document, 'visibilityState', { value: 'hidden', writable: true }); document.dispatchEvent(new Event('visibilitychange')); }")
        time.sleep(0.3)
        health_reqs_hidden = sum(1 for url in network_requests if "/api/v1/health" in url)
        time.sleep(3.5)
        assert sum(1 for url in network_requests if "/api/v1/health" in url) == health_reqs_hidden, "Polling continued while tab was hidden!"

        # Restore visibility
        page.evaluate("() => { Object.defineProperty(document, 'visibilityState', { value: 'visible', writable: true }); document.dispatchEvent(new Event('visibilitychange')); }")
        time.sleep(0.5)

        # Close modal with Escape key
        page.keyboard.press("Escape")
        time.sleep(0.4)
        assert not modal.is_visible(), "Modal should be hidden after pressing Escape"

        # Wait 4 seconds to verify polling STOPPED when modal closed
        health_reqs_at_close = sum(1 for url in network_requests if "/api/v1/health" in url)
        time.sleep(4.0)
        health_reqs_after_wait = sum(1 for url in network_requests if "/api/v1/health" in url)

        assert health_reqs_after_wait == health_reqs_at_close, (
            f"Polling did NOT stop after modal was closed! "
            f"Requests before: {health_reqs_at_close}, after: {health_reqs_after_wait}"
        )

        # ─── 5. Mobile Navigation Drawer Reframing ─────────────────────────────
        page.set_viewport_size({"width": 375, "height": 667})
        time.sleep(0.4)

        mobile_menu_btn = page.locator("#mobile-menu-btn")
        assert mobile_menu_btn.is_visible()
        mobile_menu_btn.click()
        time.sleep(0.3)

        mobile_menu = page.locator("#mobile-menu")
        assert mobile_menu.is_visible()

        # Check authentic DIA mobile menu contents
        assert page.locator("#mobile-menu a:has-text('Autonomous AutoML Ensembles')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('Causal Discovery & Policy Simulator')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('Conformal Uncertainty Sets')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('Pre-computed C99 & WASM Scoring')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('Governance & Audit Dossiers')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('System Architecture')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('Docs / OpenAPI Spec')").count() >= 1
        assert page.locator("#mobile-menu button:has-text('Health Telemetry HUD')").count() >= 1
        assert page.locator("#mobile-menu a:has-text('Launch Workbench')").count() >= 1

        # Test mobile drawer auto-dismiss when clicking an in-page anchor
        page.locator("#mobile-menu a[href='#causal-simulator-section']").click()
        time.sleep(0.4)
        assert not mobile_menu.is_visible(), "Mobile drawer should auto-dismiss after link selection"

        # Reopen mobile menu to test Health Telemetry trigger from mobile
        mobile_menu_btn.click()
        time.sleep(0.3)
        assert mobile_menu.is_visible()

        # Click Health Telemetry HUD from mobile menu
        page.locator("#mobile-menu button:has-text('Health Telemetry HUD')").click()
        time.sleep(0.6)
        assert modal.is_visible(), "Modal failed to open from mobile menu"
        assert not mobile_menu.is_visible(), "Mobile menu should auto-dismiss when telemetry trigger is clicked"

        # Close with explicit close button
        page.locator("#close-telemetry-btn").click()
        time.sleep(0.4)
        assert not modal.is_visible(), "Modal failed to close from close button"

        # Test Touch Emulation hides custom cursor
        page.evaluate("() => { window.dispatchEvent(new Event('touchstart')); }")
        time.sleep(0.2)
        has_touch_class = page.evaluate("() => document.body.classList.contains('touch-device-active')")
        assert has_touch_class, "Body should have touch-device-active class on touchstart"
        cursor_display = page.evaluate("() => window.getComputedStyle(document.getElementById('cursor-ring')).display")
        assert cursor_display == "none", f"Cursor ring should have display:none on touch device, got: {cursor_display}"

        # ─── 6. Workbench CTA Navigation ───────────────────────────────────────
        page.set_viewport_size({"width": 1440, "height": 900})
        time.sleep(0.3)

        workbench_btn.click()
        page.wait_for_url("**/app*", timeout=10000)
        time.sleep(0.8)
        assert "Data Intelligence" in page.title()

        # ─── 7. Check for Zero Console / Page Errors ───────────────────────────
        assert len(console_errors) == 0, f"Found console errors: {console_errors}"
        assert len(page_errors) == 0, f"Found page unhandled errors: {page_errors}"

        browser.close()


if __name__ == "__main__":
    test_landing_page_interactive_full()
    print("ALL TESTS IN test_interactive_landing_features.py PASSED!")
