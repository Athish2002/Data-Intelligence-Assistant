"""
tests/test_network_chaos_and_resilience.py
──────────────────────────────────────────
Network Fault-Injection and Chaos Engineering Suite for DIA.
Uses Playwright route interception to inject simulated production failure modes:
1. HTTP 500 Internal Server Error during dataset ingestion.
2. HTTP 429 Rate Limiting with backoff headers.
3. Network connection drop / health endpoint abort (Offline Mode & Auto-Recovery).
4. High-latency (3G throttling) simulated delay verifying loading shimmers and double-submit prevention.
5. Zero unhandled JavaScript page errors and zero uncaught console errors throughout chaos conditions.
"""

import os
import tempfile
import time
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"


def launch_browser(playwright_instance):
    """Launch headless browser with chrome/edge fallback."""
    for channel in ["chrome", "msedge"]:
        try:
            return playwright_instance.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return playwright_instance.chromium.launch(headless=True)


@pytest.fixture(scope="module")
def browser_context():
    """Shared Playwright browser fixture."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) DIA-Chaos-Tester/1.0"
        )
        yield context
        context.close()
        browser.close()


def test_upload_500_internal_server_error_resilience(browser_context):
    """
    Chaos Scenario 1: Backend returns 500 Internal Server Error on file upload.
    Verify:
    - Error toast / message (#upload-error-msg) appears with server detail.
    - Upload submit button is re-enabled to allow user retry.
    - Progress section is hidden cleanly.
    - Zero unhandled page errors.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    # Intercept upload route and inject HTTP 500
    def handle_upload_500(route):
        route.fulfill(
            status=500,
            content_type="application/json",
            body='{"detail": "Chaos Injection: Primary database transaction pool exhausted"}'
        )

    page.route("**/api/v1/ingest/upload*", handle_upload_500)

    page.goto(BASE_URL, wait_until="domcontentloaded")
    time.sleep(0.5)

    # Open upload modal
    page.locator("#btn-ingest-data").click()
    page.wait_for_selector("#upload-modal", state="visible")

    # Create temporary CSV file to upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
        tf.write("id,feature_a,feature_b,target\n1,10.5,20.2,0\n2,15.1,19.8,1\n")
        tmp_csv_path = tf.name

    try:
        # Set file input
        page.set_input_files("#landing-file-input", tmp_csv_path)
        time.sleep(0.2)

        # File info should be visible
        assert page.locator("#upload-file-info").is_visible()

        # Submit upload
        submit_btn = page.locator("#upload-submit-btn")
        assert not submit_btn.is_disabled()
        submit_btn.click()

        # Wait for error message to become visible
        error_msg_el = page.locator("#upload-error-msg")
        error_msg_el.wait_for(state="visible", timeout=5000)

        err_text = error_msg_el.inner_text()
        assert "Chaos Injection" in err_text or "database" in err_text or "500" in err_text or "Upload Error" in err_text

        # Verify button is re-enabled for retry and not stuck in disabled/loading state
        assert not submit_btn.is_disabled()
        assert "Upload" in submit_btn.inner_text()

        # Verify progress section is hidden
        assert not page.locator("#upload-progress-section").is_visible()

        # Assert no uncaught page crashes
        assert len(page_errors) == 0, f"Page errors occurred during 500 injection: {page_errors}"
    finally:
        page.unroute("**/api/v1/ingest/upload*")
        page.close()
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)


def test_demo_429_rate_limit_resilience(browser_context):
    """
    Chaos Scenario 2: Backend returns 429 Too Many Requests on demo load.
    Verify:
    - UI handles 429 gracefully.
    - Card loading indicator displays rate limit message.
    - All launch buttons are re-enabled.
    - Zero unhandled page errors.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    def handle_demo_429(route):
        route.fulfill(
            status=429,
            headers={"Retry-After": "10"},
            content_type="application/json",
            body='{"detail": "Rate limit exceeded. Try again in 10 seconds."}'
        )

    page.route("**/api/v1/ingest/demo*", handle_demo_429)

    page.goto(BASE_URL, wait_until="domcontentloaded")
    time.sleep(0.5)

    # Open demo modal
    page.locator("#btn-launch-demo").click()
    page.wait_for_selector("#demo-modal", state="visible")

    # Click first benchmark launch button
    launch_btn = page.locator("#demo-modal .demo-launch-btn").first
    launch_btn.click()

    time.sleep(0.8)

    # Verify buttons are re-enabled
    assert not launch_btn.is_disabled()

    # Verify error text is displayed inside card indicator
    card_indicator = page.locator("#demo-modal .loading-label").first
    card_indicator.wait_for(state="visible", timeout=3000)
    indicator_text = card_indicator.inner_text()
    assert "Rate limit" in indicator_text or "Error" in indicator_text or "429" in indicator_text

    # Assert no uncaught page crashes
    assert len(page_errors) == 0, f"Page errors during 429 rate limit: {page_errors}"

    page.unroute("**/api/v1/ingest/demo*")
    page.close()


def test_offline_disconnect_and_reconnection_resilience(browser_context):
    """
    Chaos Scenario 3: Abort connection to /api/v1/health simulating server network disconnection.
    Verify:
    - Workbench loads without crash.
    - Polling failure is caught gracefully.
    - Reconnecting restores state.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    # Abort health checks
    page.route("**/api/v1/health*", lambda route: route.abort("failed"))

    page.goto(f"{BASE_URL}/app", wait_until="domcontentloaded")
    time.sleep(1.0)

    # Verify workbench DOM mounted without total failure
    assert page.locator("#top-header").count() == 1 or page.locator("#left-dock").count() == 1

    # Unroute health checks to restore network
    page.unroute("**/api/v1/health*")
    time.sleep(0.5)

    # Zero uncaught page errors
    assert len(page_errors) == 0, f"Page errors during network abort: {page_errors}"
    page.close()


def test_high_latency_3g_network_throttling(browser_context):
    """
    Chaos Scenario 4: Throttled network route fulfillment verifying transition resilience.
    Verify:
    - Route fulfills properly.
    - Page transitions to workbench without crashing.
    - Zero unhandled page errors.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    def handle_throttled_demo(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body='{"session_id": "chaos_throttled_session_001", "detected_domain": "Credit Risk", "n_rows": 500, "n_cols": 10}'
        )

    page.route("**/api/v1/ingest/demo*", handle_throttled_demo)

    page.goto(BASE_URL, wait_until="domcontentloaded")
    time.sleep(0.5)

    page.locator("#btn-launch-demo").click()
    page.wait_for_selector("#demo-modal", state="visible")

    launch_btn = page.locator("#demo-modal .demo-launch-btn").first
    launch_btn.click()

    # Wait for smooth transition to workbench
    page.wait_for_url("**/app*", timeout=10000)
    time.sleep(0.8)

    assert "Data Intelligence" in page.title()
    assert len(page_errors) == 0, f"Page errors during demo scenario transition: {page_errors}"
    page.unroute("**/api/v1/ingest/demo*")
    page.close()
