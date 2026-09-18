"""
tests/test_edge_cases_and_fuzzing.py
───────────────────────────────────
Adversarial Boundary, Edge Case, and Fuzzing Suite for DIA.
Validates:
1. High-frequency slider boundary scrubbing (0%, 100%, negative, NaN, rapid events).
2. Malicious filename uploads (path traversal, Windows device names, 0-byte files, emojis).
3. Malicious URL query parameter injection on /app (XSS vectors, SQLi fragments, ultra-long strings).
4. Corrupted sessionStorage JSON parsing and graceful fallback.
5. Zero unhandled JavaScript page errors and 100% clean recovery.
"""

import os
import io
import tempfile
import time
import pytest
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from api.server import app

BASE_URL = "http://127.0.0.1:8000"
test_client = TestClient(app)


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
    """Playwright browser fixture for edge case testing."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) DIA-Edge-Case-Fuzzer/1.0"
        )
        yield context
        context.close()
        browser.close()


def test_slider_boundary_scrubbing_and_rapid_fuzzing(browser_context):
    """
    Edge Case 1: Extreme boundary and rapid input event fuzzing on policy slider.
    Verify:
    - Zero, negative, extreme high values do not result in NaN% or undefined%.
    - Rapid sequence of 50 input events does not crash DOM or desync values.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    page.goto(BASE_URL, wait_until="domcontentloaded")
    time.sleep(0.5)

    # Test extreme boundaries via evaluation of updatePolicyIntervention
    extreme_values = [0.0, -10.0, 100.0, 9999.0, 0.0001, 25.0]
    for val in extreme_values:
        page.evaluate(f"window.updatePolicyIntervention({val})")
        time.sleep(0.05)

        sim_risk = page.locator("#metric-sim-default").inner_text()
        ate = page.locator("#metric-sim-ate").inner_text()

        assert "NaN" not in sim_risk, f"NaN found in sim risk for input {val}"
        assert "undefined" not in sim_risk, f"undefined found in sim risk for input {val}"
        assert "%" in sim_risk, f"sim risk missing % formatting for input {val}"
        assert "NaN" not in ate, f"NaN found in ATE for input {val}"

    # Rapid high-frequency scrubbing: 50 events in quick succession
    page.evaluate("""
        () => {
            const slider = document.getElementById('causal-policy-slider');
            for (let i = 1; i <= 50; i++) {
                const simulatedVal = (i % 16) + 1;
                slider.value = simulatedVal;
                slider.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    """)
    time.sleep(0.3)

    final_risk = page.locator("#metric-sim-default").inner_text()
    assert "NaN" not in final_risk and "%" in final_risk
    assert len(page_errors) == 0, f"Page errors during rapid slider scrubbing: {page_errors}"

    page.close()


def test_adversarial_filename_upload_validation():
    """
    Edge Case 2: Adversarial filename upload testing via API client.
    Tests:
    - Directory traversal filenames (../../../../etc/passwd.csv)
    - Windows reserved device names (CON.csv, NUL.csv, PRN.csv)
    - 0-byte empty file upload
    - Unicode and emojis in filename
    """
    valid_csv_content = b"id,val,target\n1,10,0\n2,20,1\n3,30,0\n"

    # 1. Directory traversal filename: should be sanitized or processed safely without filesystem escape
    traversal_name = "../../../../etc/passwd.csv"
    resp = test_client.post(
        "/api/v1/ingest/upload",
        files={"file": (traversal_name, valid_csv_content, "text/csv")}
    )
    # The API should either succeed with sanitized filename or reject with 400, never 500
    assert resp.status_code in [200, 400], f"Unexpected status {resp.status_code} for path traversal"

    # 2. Windows reserved device names
    for device_name in ["CON.csv", "NUL.csv", "PRN.csv", "AUX.csv"]:
        resp = test_client.post(
            "/api/v1/ingest/upload",
            files={"file": (device_name, valid_csv_content, "text/csv")}
        )
        assert resp.status_code in [200, 400], f"Crash on reserved device name {device_name}"

    # 3. 0-byte empty file
    resp_empty = test_client.post(
        "/api/v1/ingest/upload",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    # Should reject empty files with client error (400)
    assert resp_empty.status_code == 400, f"Expected 400 for empty file, got {resp_empty.status_code}"

    # 4. Unicode / emoji filename
    emoji_name = "dataset_🔥_metric_📈.csv"
    resp_emoji = test_client.post(
        "/api/v1/ingest/upload",
        files={"file": (emoji_name, valid_csv_content, "text/csv")}
    )
    assert resp_emoji.status_code in [200, 400], f"Crash on unicode filename {emoji_name}"


def test_url_query_parameter_injection_resilience(browser_context):
    """
    Edge Case 3: Malicious URL query parameter injection into /app.
    Tests:
    - Cross-site scripting (XSS) in query params.
    - SQL injection fragments in query params.
    - Ultra-long parameter strings (4000+ chars).
    - Verifies zero script execution, zero unhandled errors.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    xss_payload = "<script>window.__xss_fired=true;</script><img src=x onerror=window.__xss_fired=true;>"
    sqli_payload = "' OR '1'='1' UNION SELECT NULL, NULL, NULL;--"
    ultra_long = "A" * 4000

    # 1. Test XSS payload in ?demo=
    page.goto(f"{BASE_URL}/app?demo={xss_payload}", wait_until="domcontentloaded")
    time.sleep(0.8)

    xss_fired = page.evaluate("() => Boolean(window.__xss_fired)")
    assert not xss_fired, "XSS executed from query parameter!"
    assert len(page_errors) == 0, f"Page errors with XSS query param: {page_errors}"

    # 2. Test SQLi payload in ?session_id=
    page.goto(f"{BASE_URL}/app?session_id={sqli_payload}", wait_until="domcontentloaded")
    time.sleep(0.5)
    assert len(page_errors) == 0, f"Page errors with SQLi query param: {page_errors}"

    # 3. Test Ultra-long parameter in ?tab=
    page.goto(f"{BASE_URL}/app?tab={ultra_long}", wait_until="domcontentloaded")
    time.sleep(0.5)
    assert len(page_errors) == 0, f"Page errors with ultra-long query param: {page_errors}"

    page.close()


def test_corrupted_session_storage_recovery(browser_context):
    """
    Edge Case 4: Pre-seeding corrupted, invalid JSON and null bytes in sessionStorage.
    Verify:
    - Workbench (/app) initializes without uncaught SyntaxError.
    - Gracefully falls back to empty state.
    """
    page = browser_context.new_page()
    page_errors = []
    page.on("pageerror", lambda err: page_errors.append(str(err)))

    # Navigate to base first to access sessionStorage domain
    page.goto(BASE_URL, wait_until="domcontentloaded")
    time.sleep(0.3)

    # Invert/corrupt sessionStorage values
    page.evaluate("""
        () => {
            sessionStorage.setItem('dia_pending_session', '{corrupted: JSON without quotes, :;;');
            sessionStorage.setItem('dia_pending_autotrain', 'undefined');
            sessionStorage.setItem('dia_auth_token', '\\x00\\x01\\x02\\x03');
        }
    """)

    # Navigate to workbench
    page.goto(f"{BASE_URL}/app", wait_until="domcontentloaded")
    time.sleep(1.0)

    # Verify workbench still rendered its structure cleanly
    assert page.locator("#top-header").count() == 1 or page.locator("#left-dock").count() == 1

    # Assert zero unhandled page crashes
    assert len(page_errors) == 0, f"Workbench crashed on corrupted sessionStorage: {page_errors}"

    page.close()
