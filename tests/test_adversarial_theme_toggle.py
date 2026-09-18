"""
tests/test_adversarial_theme_toggle.py
──────────────────────────────────────
Adversarial Stress Test Suite for Milestone M1 Theme Engine & Cross-Page Persistence.
Designed by Challenger M1_1 (EMPIRICAL CHALLENGER / critic, specialist).

Focus Areas:
1. High-frequency rapid toggling (15-20 rapid clicks in 0-20ms bursts) on #theme-toggle-btn.
2. Zero console errors and zero page errors across all stress sequences.
3. Class and attribute synchronization: data-theme, data-theme-mode, light/dark classes.
4. Clean transition teardown: .theme-in-transition removed without leaking.
5. Cross-page persistence between Landing (/) and Workbench (/app) in both directions.
6. Mobile drawer toggle (#theme-toggle-btn-mobile) bidirectional sync with desktop toggle.
7. Resilience against corrupted localStorage states and legacy key migration.
8. CSS layout stability: Header, dock, and container bounding boxes invariant across rapid toggles.
"""

import os
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"


def launch_browser(playwright_instance):
    """Launch headless browser with Chrome/Edge fallback."""
    for channel in ["chrome", "msedge"]:
        try:
            return playwright_instance.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return playwright_instance.chromium.launch(headless=True)


# ─── 1. Rapid Toggle Stress on Workbench (/app) ───────────────────────────────

def test_adversarial_rapid_toggle_stress_workbench():
    """Adversarially hammer #theme-toggle-btn with 15 rapid clicks.
    Assert zero console errors, zero page errors, exact state sync, and clean transition removal.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Load workbench
        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # Initial baseline measurement of layout bounding boxes
        header_box_before = page.locator("#sticky-workspace-header").bounding_box()
        dock_box_before = page.locator("#left-dock").bounding_box()
        assert header_box_before is not None, "Sticky workspace header missing"
        assert dock_box_before is not None, "Left dock missing"

        initial_theme = page.evaluate("() => localStorage.getItem('dia_theme_mode') || 'night'")
        expected_toggle_count = 15

        # Execute 15 rapid clicks with minimal delay
        toggle_btn = page.locator("#theme-toggle-btn")
        assert toggle_btn.is_visible(), "#theme-toggle-btn must be visible on workbench"

        for _ in range(expected_toggle_count):
            toggle_btn.click()

        # Determine expected final theme after 15 toggles (odd count flips state)
        expected_final_theme = "day" if initial_theme == "night" else "night"
        expected_data_theme = "tokyo-sand" if expected_final_theme == "day" else "cyber-aurora"

        # Check DOM synchronization
        actual_data_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        actual_theme_mode = page.evaluate("() => document.documentElement.getAttribute('data-theme-mode')")
        stored_mode = page.evaluate("() => localStorage.getItem('dia_theme_mode')")

        assert actual_data_theme == expected_data_theme, (
            f"data-theme mismatch after 15 toggles: expected {expected_data_theme}, got {actual_data_theme}"
        )
        assert actual_theme_mode == expected_final_theme, (
            f"data-theme-mode mismatch: expected {expected_final_theme}, got {actual_theme_mode}"
        )
        assert stored_mode == expected_final_theme, (
            f"localStorage dia_theme_mode mismatch: expected {expected_final_theme}, got {stored_mode}"
        )

        # Class presence check
        has_light = page.evaluate("() => document.documentElement.classList.contains('light')")
        has_dark = page.evaluate("() => document.documentElement.classList.contains('dark')")
        if expected_final_theme == "day":
            assert has_light and not has_dark, "Expected class 'light' and NOT 'dark' in Day mode"
        else:
            assert has_dark and not has_light, "Expected class 'dark' and NOT 'light' in Night mode"

        # Wait 350ms for transition timeout to expire
        page.wait_for_timeout(350)

        # Verify transition class is completely removed
        in_transition = page.evaluate("() => document.documentElement.classList.contains('theme-in-transition')")
        assert not in_transition, ".theme-in-transition was not cleaned up after transition period"

        # Layout shift verification: Verify layout didn't collapse or wildly shift
        header_box_after = page.locator("#sticky-workspace-header").bounding_box()
        dock_box_after = page.locator("#left-dock").bounding_box()
        assert header_box_after["width"] == header_box_before["width"], "Header width shifted after rapid toggles"
        assert dock_box_after["width"] == dock_box_before["width"], "Left dock width shifted after rapid toggles"

        # Assert zero console or page errors
        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors detected during rapid toggling: {real_errors}"
        assert len(page_errors) == 0, f"Page errors detected during rapid toggling: {page_errors}"

        browser.close()


# ─── 2. Rapid Toggle Stress on Landing Page (/) ───────────────────────────────

def test_adversarial_rapid_toggle_stress_landing():
    """Adversarially hammer #theme-toggle-btn on landing page with 16 rapid clicks.
    Assert zero console errors, zero page errors, and mobile drawer sync.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/", wait_until="networkidle")

        initial_theme = page.evaluate("() => localStorage.getItem('dia_theme_mode') || 'night'")

        # 16 rapid toggles (even count -> returns to initial theme)
        toggle_btn = page.locator("#theme-toggle-btn")
        assert toggle_btn.is_visible(), "#theme-toggle-btn must be visible on landing page desktop nav"

        for _ in range(16):
            toggle_btn.click()

        page.wait_for_timeout(350)

        final_theme = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        assert final_theme == initial_theme, f"Theme did not return to initial state {initial_theme} after 16 clicks"

        # Check mobile button in drawer has the same data-current-theme attribute
        mobile_btn = page.locator("#theme-toggle-btn-mobile")
        if mobile_btn.count() > 0:
            mobile_current = mobile_btn.get_attribute("data-current-theme")
            assert mobile_current == final_theme, (
                f"Mobile toggle data-current-theme ({mobile_current}) desynchronized from final theme ({final_theme})"
            )

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Landing console errors during rapid toggling: {real_errors}"
        assert len(page_errors) == 0, f"Landing page errors during rapid toggling: {page_errors}"

        browser.close()


# ─── 3. Cross-Page Persistence: Landing (/) -> Workbench (/app) ───────────────

def test_adversarial_cross_page_persistence_landing_to_workbench():
    """Set Tokyo Sand on landing page (/), navigate to workbench (/app),
    and assert /app immediately mounts in Tokyo Sand without FOUC or state reset.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Open Landing Page
        page.goto(f"{BASE_URL}/", wait_until="networkidle")

        # Explicitly apply Tokyo Sand (Day Mode)
        page.evaluate("() => window.applyTheme && window.applyTheme('tokyo-sand', false)")
        page.wait_for_timeout(100)

        # Confirm landing page is in tokyo-sand
        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "tokyo-sand"
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "day"

        # 2. Navigate to Workbench (/app)
        page.goto(f"{BASE_URL}/app", wait_until="domcontentloaded")

        # Check immediate DOM attributes upon mount (before waiting for full networkidle)
        mount_data_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        mount_theme_mode = page.evaluate("() => document.documentElement.getAttribute('data-theme-mode')")
        mount_has_light = page.evaluate("() => document.documentElement.classList.contains('light')")
        mount_has_dark = page.evaluate("() => document.documentElement.classList.contains('dark')")
        stored_mode = page.evaluate("() => localStorage.getItem('dia_theme_mode')")

        assert mount_data_theme == "tokyo-sand", (
            f"Preloader FOUC failure: /app mounted with data-theme='{mount_data_theme}', expected 'tokyo-sand'"
        )
        assert mount_theme_mode == "day", f"/app mounted with data-theme-mode='{mount_theme_mode}', expected 'day'"
        assert mount_has_light is True, "/app mounted without 'light' class"
        assert mount_has_dark is False, "/app mounted with unwanted 'dark' class"
        assert stored_mode == "day", f"/app localStorage lost 'day' mode, got: {stored_mode}"

        # Wait for full scripts and verify theme engine still agrees
        page.wait_for_load_state("networkidle")
        engine_theme = page.evaluate("() => window.getCurrentTheme && window.getCurrentTheme()")
        assert engine_theme == "day", f"theme.js getCurrentTheme() returned '{engine_theme}', expected 'day'"

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors during cross-page navigation: {real_errors}"
        assert len(page_errors) == 0, f"Page errors during cross-page navigation: {page_errors}"

        browser.close()


# ─── 4. Cross-Page Persistence: Workbench (/app) -> Landing (/) ───────────────

def test_adversarial_cross_page_persistence_workbench_to_landing():
    """Set Night mode on workbench (/app), navigate to landing page (/),
    and assert landing page mounts in Cyber Aurora.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Open Workbench
        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # Explicitly apply Night mode
        page.evaluate("() => window.applyTheme && window.applyTheme('cyber-aurora', false)")
        page.wait_for_timeout(100)

        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "cyber-aurora"
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "night"

        # 2. Click Home button to return to Landing Page
        home_btn = page.locator("#nav-landing-home-btn")
        if home_btn.is_visible():
            home_btn.click()
            page.wait_for_load_state("domcontentloaded")
        else:
            page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")

        mount_data_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        mount_theme_mode = page.evaluate("() => document.documentElement.getAttribute('data-theme-mode')")
        mount_has_dark = page.evaluate("() => document.documentElement.classList.contains('dark')")
        mount_has_light = page.evaluate("() => document.documentElement.classList.contains('light')")

        assert mount_data_theme == "cyber-aurora", (
            f"Landing mounted with data-theme='{mount_data_theme}', expected 'cyber-aurora'"
        )
        assert mount_theme_mode == "night", f"Landing mounted with data-theme-mode='{mount_theme_mode}', expected 'night'"
        assert mount_has_dark is True, "Landing mounted without 'dark' class"
        assert mount_has_light is False, "Landing mounted with unwanted 'light' class"

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors on return to landing: {real_errors}"
        assert len(page_errors) == 0, f"Page errors on return to landing: {page_errors}"

        browser.close()


# ─── 5. Corrupted Storage Recovery and Legacy Migration ───────────────────────

def test_adversarial_legacy_key_migration_and_corrupted_storage():
    """Verify theme engine gracefully handles legacy 'dia-theme' key and recovers from corrupted values."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Sub-test A: Legacy migration from 'dia-theme' = 'tokyo-sand'
        page.goto(f"{BASE_URL}/app", wait_until="domcontentloaded")
        page.evaluate("""() => {
            localStorage.removeItem('dia_theme_mode');
            localStorage.setItem('dia-theme', 'tokyo-sand');
        }""")
        page.reload(wait_until="networkidle")

        migrated_mode = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        assert migrated_mode == "day", f"Failed to migrate legacy key 'tokyo-sand' to 'day': got {migrated_mode}"
        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "tokyo-sand"

        # Sub-test B: Corrupted value in localStorage
        page.evaluate("""() => {
            localStorage.setItem('dia_theme_mode', 'CORRUPT_UNKNOWN_MODE_xyz');
            localStorage.setItem('dia-theme', 'INVALID_THEME');
        }""")
        page.reload(wait_until="networkidle")

        # Must safely fall back to night without throwing exceptions
        fallback_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        assert fallback_theme == "cyber-aurora", f"Corrupted storage didn't fall back to cyber-aurora: {fallback_theme}"

        # Clicking toggle should still work smoothly from fallback state
        page.click("#theme-toggle-btn")
        page.wait_for_timeout(350)
        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "tokyo-sand"
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "day"

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors during corruption testing: {real_errors}"
        assert len(page_errors) == 0, f"Page errors during corruption testing: {page_errors}"

        browser.close()


# ─── 6. Custom Event Dispatch Contract ───────────────────────────────────────

def test_adversarial_custom_event_theme_changed_dispatch():
    """Verify 'dia-theme-changed' custom event is reliably emitted on every theme toggle with correct detail payload."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # Register event tracker in page context
        page.evaluate("""() => {
            window.__theme_change_events = [];
            window.addEventListener('dia-theme-changed', (e) => {
                window.__theme_change_events.push(e.detail);
            });
        }""")

        # Trigger 4 toggles
        for _ in range(4):
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(50)

        events = page.evaluate("() => window.__theme_change_events")
        assert len(events) == 4, f"Expected exactly 4 'dia-theme-changed' events, received: {len(events)}"

        for ev in events:
            assert "theme" in ev, f"Event detail missing 'theme': {ev}"
            assert "dataTheme" in ev, f"Event detail missing 'dataTheme': {ev}"
            assert ev["theme"] in ["night", "day"], f"Invalid theme mode in event: {ev['theme']}"
            assert ev["dataTheme"] in ["cyber-aurora", "tokyo-sand"], f"Invalid dataTheme in event: {ev['dataTheme']}"

        browser.close()
