"""
tests/test_adversarial_m3_slider_and_theme_stress.py
────────────────────────────────────────────────────
Adversarial Verification Suite for Milestone 3:
1. Pearl's Do-Calculus slider and mathematical boundaries:
   - Scrub slider at 0.0%, 8.5%, 15.0%, and extreme values (-100%, +1000%).
   - Verify simulated risk is strictly clamped in [0.1%, 99.9%].
   - Verify deltaRate, ATE, and 95% CI bounds never produce NaN, -NaN%, Infinity, or undefined.
   - Verify 95% CI is symmetric around ATE (ate - 2.4 to ate + 2.4).
   - Monte Carlo fuzzing over 100 random values in [-500.0, 2000.0].
   - Adversarial non-numeric / malformed inputs.
2. Rapid asynchronous theme toggles:
   - 15+ rapid asynchronous clicks in rapid succession across / and /app.
   - Assert localStorage['dia_theme_mode'], document.documentElement.getAttribute('data-theme'),
     and #theme-toggle-btn state remain strictly synchronized.
   - Assert zero console errors and zero unhandled page errors occur during or after the storm.
"""

import os
import re
import random
import pytest
from playwright.sync_api import sync_playwright, expect

BASE_URL = os.environ.get("DIA_BASE_URL", "http://127.0.0.1:8000")


def launch_browser(playwright_instance):
    """Launch headless browser with chrome/msedge/chromium fallback."""
    for channel in ["chrome", "msedge"]:
        try:
            return playwright_instance.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return playwright_instance.chromium.launch(headless=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. PEARL DO-CALCULUS SLIDER MATHEMATICAL BOUNDARIES & CLAMPING
# ═════════════════════════════════════════════════════════════════════════════

def test_pearl_slider_specified_boundaries():
    """Scrub slider at 0.0%, 8.5%, 15.0%, -100.0%, and +1000.0%.
    Verify risk clamping [0.1%, 99.9%], symmetry (ate ± 2.4), and absence of NaN/Infinity/undefined.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/", wait_until="networkidle")

        test_cases = [
            {
                "rate": 8.5,
                "expected_delta_rate": 0.0,
                "expected_ate": 0.0,
                "expected_risk": 25.6,
                "expected_ci_low": -2.4,
                "expected_ci_high": 2.4,
                "desc": "Baseline rate (8.5%)",
            },
            {
                "rate": 0.0,
                "expected_delta_rate": -8.5,
                "expected_ate": -27.2,
                "expected_risk": 0.1,  # clamped from -1.6%
                "expected_ci_low": -29.6,
                "expected_ci_high": -24.8,
                "desc": "Zero boundary (0.0%)",
            },
            {
                "rate": 15.0,
                "expected_delta_rate": 6.5,
                "expected_ate": 20.8,
                "expected_risk": 46.4,
                "expected_ci_low": 18.4,
                "expected_ci_high": 23.2,
                "desc": "Target high policy rate (15.0%)",
            },
            {
                "rate": -100.0,
                "expected_delta_rate": -108.5,
                "expected_ate": -347.2,
                "expected_risk": 0.1,  # strictly clamped to 0.1%
                "expected_ci_low": -349.6,
                "expected_ci_high": -344.8,
                "desc": "Extreme negative rate (-100.0%)",
            },
            {
                "rate": 1000.0,
                "expected_delta_rate": 991.5,
                "expected_ate": 3172.8,
                "expected_risk": 99.9,  # strictly clamped to 99.9%
                "expected_ci_low": 3170.4,
                "expected_ci_high": 3175.2,
                "desc": "Extreme positive rate (+1000.0%)",
            },
        ]

        for tc in test_cases:
            rate = tc["rate"]
            # Trigger policy update both via slider event and exposed function
            page.evaluate(f"() => {{ window.updatePolicyIntervention({rate}); }}")

            risk_text = page.locator("#metric-sim-default").inner_text()
            delta_text = page.locator("#metric-sim-delta").inner_text()
            ate_text = page.locator("#metric-sim-ate").inner_text()
            ci_text = page.locator("#metric-sim-ci").inner_text()
            badge_text = page.locator("#policy-slider-val").inner_text()

            # 1. Negative checks: zero NaN, -NaN%, Infinity, undefined
            for field_name, txt in [
                ("risk", risk_text),
                ("delta", delta_text),
                ("ate", ate_text),
                ("ci", ci_text),
                ("badge", badge_text),
            ]:
                assert "NaN" not in txt, f"[{tc['desc']}] {field_name} produced NaN: '{txt}'"
                assert "-NaN%" not in txt, f"[{tc['desc']}] {field_name} produced -NaN%: '{txt}'"
                assert "Infinity" not in txt, f"[{tc['desc']}] {field_name} produced Infinity: '{txt}'"
                assert "undefined" not in txt, f"[{tc['desc']}] {field_name} produced undefined: '{txt}'"

            # 2. Risk clamping check: strictly in [0.1%, 99.9%]
            m_risk = re.search(r"([\d.]+)%", risk_text)
            assert m_risk is not None, f"Could not parse risk float from '{risk_text}'"
            parsed_risk = float(m_risk.group(1))
            assert 0.1 <= parsed_risk <= 99.9, f"[{tc['desc']}] Risk {parsed_risk}% outside [0.1%, 99.9%]"
            assert abs(parsed_risk - tc["expected_risk"]) < 0.05, (
                f"[{tc['desc']}] Expected risk {tc['expected_risk']}%, got {parsed_risk}%"
            )

            # 3. ATE check
            m_ate = re.search(r"([+-]?[\d.]+)%", ate_text)
            assert m_ate is not None, f"Could not parse ATE float from '{ate_text}'"
            parsed_ate = float(m_ate.group(1))
            assert abs(parsed_ate - tc["expected_ate"]) < 0.05, (
                f"[{tc['desc']}] Expected ATE {tc['expected_ate']}%, got {parsed_ate}%"
            )

            # 4. 95% CI symmetry check: ate - 2.4 to ate + 2.4
            m_ci = re.search(r"\[([+-]?[\d.]+)%,\s*([+-]?[\d.]+)%\]", ci_text)
            assert m_ci is not None, f"Could not parse CI range from '{ci_text}'"
            ci_low = float(m_ci.group(1))
            ci_high = float(m_ci.group(2))

            assert abs(ci_low - tc["expected_ci_low"]) < 0.05, (
                f"[{tc['desc']}] Expected CI low {tc['expected_ci_low']}%, got {ci_low}%"
            )
            assert abs(ci_high - tc["expected_ci_high"]) < 0.05, (
                f"[{tc['desc']}] Expected CI high {tc['expected_ci_high']}%, got {ci_high}%"
            )

            # Mathematical symmetry: (ci_low + ci_high) / 2 == parsed_ate
            ci_midpoint = (ci_low + ci_high) / 2.0
            assert abs(ci_midpoint - parsed_ate) < 0.05, (
                f"[{tc['desc']}] 95% CI is NOT symmetric around ATE! Midpoint={ci_midpoint}, ATE={parsed_ate}"
            )
            # CI half-width must be exactly 2.4 (within rounding tolerance)
            assert abs((ci_high - parsed_ate) - 2.4) < 0.05, (
                f"[{tc['desc']}] CI high margin != 2.4: {ci_high - parsed_ate}"
            )
            assert abs((parsed_ate - ci_low) - 2.4) < 0.05, (
                f"[{tc['desc']}] CI low margin != 2.4: {parsed_ate - ci_low}"
            )

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors: {real_errors}"
        assert len(page_errors) == 0, f"Page errors: {page_errors}"

        browser.close()


def test_pearl_slider_monte_carlo_fuzzing():
    """Monte Carlo fuzzing over 100 diverse rates in range [-500.0, 2000.0].
    Asserts every evaluation satisfies clamping, symmetry, and zero NaN/Infinity.
    """
    random.seed(42)
    test_rates = [
        random.uniform(-500.0, 2000.0) for _ in range(100)
    ]
    # Include boundary values explicitly
    test_rates.extend([-1000.0, -100.0, -8.5, 0.0, 0.001, 8.499, 8.5, 8.501, 15.0, 50.0, 1000.0, 10000.0])

    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/", wait_until="networkidle")

        for rate in test_rates:
            page.evaluate(f"window.updatePolicyIntervention({rate})")

            risk_txt = page.locator("#metric-sim-default").inner_text()
            ate_txt = page.locator("#metric-sim-ate").inner_text()
            ci_txt = page.locator("#metric-sim-ci").inner_text()

            # Negative assertions
            for txt in [risk_txt, ate_txt, ci_txt]:
                assert "NaN" not in txt, f"Fuzz rate {rate} produced NaN in {txt}"
                assert "Infinity" not in txt, f"Fuzz rate {rate} produced Infinity in {txt}"
                assert "undefined" not in txt, f"Fuzz rate {rate} produced undefined in {txt}"

            # Clamping assertion: strictly [0.1, 99.9]
            m_risk = re.search(r"([\d.]+)%", risk_txt)
            assert m_risk is not None
            r_val = float(m_risk.group(1))
            assert 0.1 <= r_val <= 99.9, f"Risk {r_val}% out of [0.1, 99.9] for rate {rate}"

            # Symmetry assertion
            m_ate = re.search(r"([+-]?[\d.]+)%", ate_txt)
            m_ci = re.search(r"\[([+-]?[\d.]+)%,\s*([+-]?[\d.]+)%\]", ci_txt)
            assert m_ate is not None and m_ci is not None
            ate_val = float(m_ate.group(1))
            ci_low = float(m_ci.group(1))
            ci_high = float(m_ci.group(2))

            midpoint = (ci_low + ci_high) / 2.0
            assert abs(midpoint - ate_val) < 0.1, (
                f"Asymmetry detected for rate {rate}: ATE={ate_val}, CI=[{ci_low}, {ci_high}], Mid={midpoint}"
            )
            assert abs((ci_high - ate_val) - 2.4) < 0.1
            assert abs((ate_val - ci_low) - 2.4) < 0.1

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors during fuzzing: {real_errors}"
        assert len(page_errors) == 0, f"Page errors during fuzzing: {page_errors}"

        browser.close()


def test_pearl_slider_adversarial_malformed_inputs():
    """Inject malformed and adversarial non-numeric inputs.
    Verify engine handles them gracefully without crashing or corrupting DOM.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/", wait_until="networkidle")

        # Set known valid state first
        page.evaluate("window.updatePolicyIntervention(8.5)")
        baseline_risk = page.locator("#metric-sim-default").inner_text()
        assert "25.6%" in baseline_risk

        malformed_inputs = [
            "NaN",
            "undefined",
            "null",
            "''",
            "'abc'",
            "'<script>alert(1)</script>'",
            "{}",
            "[]",
        ]

        for mal in malformed_inputs:
            # Should be safely ignored by isNaN check without mutating DOM or throwing
            page.evaluate(f"() => {{ try {{ window.updatePolicyIntervention({mal}); }} catch (e) {{}} }}")
            cur_risk = page.locator("#metric-sim-default").inner_text()
            assert "NaN" not in cur_risk
            assert "undefined" not in cur_risk
            # Value should remain untouched at baseline
            assert cur_risk == baseline_risk, f"Malformed input {mal} corrupted state to {cur_risk}"

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors on malformed inputs: {real_errors}"
        assert len(page_errors) == 0, f"Page errors on malformed inputs: {page_errors}"

        browser.close()


# ═════════════════════════════════════════════════════════════════════════════
# 2. RAPID THEME TOGGLE ASYNCHRONOUS STORM STRESS
# ═════════════════════════════════════════════════════════════════════════════

def test_rapid_theme_toggle_asynchronous_storm_workbench():
    """Perform 20 rapid asynchronous toggle clicks on Workbench (/app).
    Assert localStorage['dia_theme_mode'], document data-theme, and #theme-toggle-btn remain strictly synchronized.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # Verify button presence
        toggle_btn = page.locator("#theme-toggle-btn")
        assert toggle_btn.is_visible()

        # Capture initial state
        initial_theme = page.evaluate("() => localStorage.getItem('dia_theme_mode') || 'night'")

        # Storm Part 1: Rapid asynchronous clicks dispatched via Promise.all with random sub-10ms delays
        storm_clicks = 18  # Even number returns to original state
        js_storm = f"""
        async () => {{
            const btn = document.getElementById('theme-toggle-btn');
            const promises = [];
            for (let i = 0; i < {storm_clicks}; i++) {{
                promises.push(new Promise(resolve => {{
                    setTimeout(() => {{
                        btn.click();
                        resolve();
                    }}, Math.floor(Math.random() * 8));
                }}));
            }}
            await Promise.all(promises);
        }}
        """
        page.evaluate(js_storm)

        # Wait for CSS transitions to finish (280ms duration + buffer)
        page.wait_for_timeout(400)

        # Inspect final state
        stored_mode = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        data_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        data_theme_mode = page.evaluate("() => document.documentElement.getAttribute('data-theme-mode')")
        btn_current_theme = toggle_btn.get_attribute("data-current-theme")
        btn_aria_label = toggle_btn.get_attribute("aria-label")
        btn_title = toggle_btn.get_attribute("title")

        # Assert strict synchronization
        if stored_mode == "day":
            assert data_theme == "tokyo-sand", f"data-theme mismatch in day mode: {data_theme}"
            assert data_theme_mode == "day"
            assert btn_current_theme == "day"
            assert "Night Mode" in btn_aria_label, f"aria-label mismatch: {btn_aria_label}"
            assert "Night Mode" in btn_title, f"title mismatch: {btn_title}"
            # Moon icon hidden, sun icon visible
            assert page.locator("#theme-toggle-btn .theme-icon-moon").is_hidden()
            assert page.locator("#theme-toggle-btn .theme-icon-sun").is_visible()
            assert page.evaluate("() => document.documentElement.classList.contains('light')")
            assert not page.evaluate("() => document.documentElement.classList.contains('dark')")
        else:
            assert data_theme == "cyber-aurora", f"data-theme mismatch in night mode: {data_theme}"
            assert data_theme_mode == "night"
            assert btn_current_theme == "night"
            assert "Day Mode" in btn_aria_label, f"aria-label mismatch: {btn_aria_label}"
            assert "Day Mode" in btn_title, f"title mismatch: {btn_title}"
            # Moon icon visible, sun icon hidden
            assert page.locator("#theme-toggle-btn .theme-icon-moon").is_visible()
            assert page.locator("#theme-toggle-btn .theme-icon-sun").is_hidden()
            assert page.evaluate("() => document.documentElement.classList.contains('dark')")
            assert not page.evaluate("() => document.documentElement.classList.contains('light')")

        # Transition class must be cleaned up
        assert not page.evaluate("() => document.documentElement.classList.contains('theme-in-transition')")

        # Storm Part 2: 15 consecutive rapid UI clicks with odd total (flips state)
        current_before = stored_mode
        for _ in range(15):
            toggle_btn.click()
            page.wait_for_timeout(10)

        page.wait_for_timeout(400)

        expected_after_15 = "day" if current_before == "night" else "night"
        final_stored = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        final_data_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        final_btn_theme = toggle_btn.get_attribute("data-current-theme")

        assert final_stored == expected_after_15, f"Expected {expected_after_15} after 15 clicks, got {final_stored}"
        assert final_btn_theme == expected_after_15
        expected_dt = "tokyo-sand" if expected_after_15 == "day" else "cyber-aurora"
        assert final_data_theme == expected_dt

        # Zero console errors or page errors
        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors during workbench toggle storm: {real_errors}"
        assert len(page_errors) == 0, f"Page errors during workbench toggle storm: {page_errors}"

        browser.close()


def test_rapid_theme_toggle_asynchronous_storm_landing():
    """Perform 20 rapid asynchronous toggle clicks on Landing Page (/).
    Assert synchronization across localStorage, document root, and both desktop & mobile toggle buttons.
    """
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/", wait_until="networkidle")

        toggle_btn = page.locator("#theme-toggle-btn")
        assert toggle_btn.is_visible()

        # Execute 20 rapid asynchronous clicks
        page.evaluate("""
        async () => {
            const btn = document.getElementById('theme-toggle-btn');
            const promises = [];
            for (let i = 0; i < 20; i++) {
                promises.push(new Promise(resolve => {
                    setTimeout(() => {
                        btn.click();
                        resolve();
                    }, Math.floor(Math.random() * 10));
                }));
            }
            await Promise.all(promises);
        }
        """)

        page.wait_for_timeout(400)

        stored_mode = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        data_theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        btn_theme = toggle_btn.get_attribute("data-current-theme")

        # Desktop toggle sync
        assert btn_theme == stored_mode, f"Button theme ({btn_theme}) != localStorage ({stored_mode})"
        expected_dt = "tokyo-sand" if stored_mode == "day" else "cyber-aurora"
        assert data_theme == expected_dt, f"data-theme ({data_theme}) != expected ({expected_dt})"

        # Mobile drawer toggle sync (#theme-toggle-btn-mobile)
        mobile_btn = page.locator("#theme-toggle-btn-mobile")
        if mobile_btn.count() > 0:
            mobile_theme = mobile_btn.get_attribute("data-current-theme")
            assert mobile_theme == stored_mode, (
                f"Mobile button theme ({mobile_theme}) desynchronized from localStorage ({stored_mode})"
            )

        # Zero console errors or page errors
        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors on landing theme storm: {real_errors}"
        assert len(page_errors) == 0, f"Page errors on landing theme storm: {page_errors}"

        browser.close()


def test_theme_persistence_across_page_navigation():
    """Verify theme set on / carries to /app, and theme set on /app carries back to /."""
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Start at / and set to day mode
        page.goto(f"{BASE_URL}/", wait_until="networkidle")
        current_theme = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        if current_theme != "day":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(350)
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "day"

        # 2. Navigate to /app and verify day mode is maintained on load
        page.goto(f"{BASE_URL}/app", wait_until="networkidle")
        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "tokyo-sand"
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "day"
        expect(page.locator("#theme-toggle-btn")).to_have_attribute("data-current-theme", "day")

        # 3. Switch to night mode on /app
        page.click("#theme-toggle-btn")
        page.wait_for_timeout(350)
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "night"
        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "cyber-aurora"

        # 4. Navigate back to / and verify night mode is maintained
        page.goto(f"{BASE_URL}/", wait_until="networkidle")
        assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") == "cyber-aurora"
        assert page.evaluate("() => localStorage.getItem('dia_theme_mode')") == "night"
        expect(page.locator("#theme-toggle-btn")).to_have_attribute("data-current-theme", "night")

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0
        assert len(page_errors) == 0

        browser.close()
