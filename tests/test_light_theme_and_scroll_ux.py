"""
tests/test_light_theme_and_scroll_ux.py
───────────────────────────────────────
Comprehensive test suite verifying:
1. Tokyo Sand light theme text readability & WCAG 2.2 AA contrast ratios (>= 4.5:1).
2. In-app Health Telemetry HUD modal behavior (no raw JSON in a new browser tab).
3. Sticky workspace navigation header (#sticky-workspace-header) on scroll.
4. Persistent Section Outline & Quick Navigation dock (#dock-section-outline) with active scroll spy.
5. Collapsed dock icon rail mode (#dock-icon-rail).
6. Zero console errors and zero unhandled page errors across Desktop and Mobile viewports.
"""

import os
import math
import pytest
from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)
BASE_URL = "http://127.0.0.1:8000"


def relative_luminance(r: int, g: int, b: int) -> float:
    """Calculates WCAG 2.2 relative luminance for an sRGB color."""
    def channel_linear(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = channel_linear(r)
    g_lin = channel_linear(g)
    b_lin = channel_linear(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def contrast_ratio(rgb1: tuple, rgb2: tuple) -> float:
    """Computes the WCAG contrast ratio between two (r, g, b) colors."""
    lum1 = relative_luminance(rgb1[0], rgb1[1], rgb1[2])
    lum2 = relative_luminance(rgb2[0], rgb2[1], rgb2[2])
    l_max = max(lum1, lum2)
    l_min = min(lum1, lum2)
    return (l_max + 0.05) / (l_min + 0.05)


def parse_rgb(rgb_str: str) -> tuple:
    """Parses 'rgb(r, g, b)' or 'rgba(r, g, b, a)' into (r, g, b)."""
    clean = rgb_str.replace("rgba(", "").replace("rgb(", "").replace(")", "").strip()
    parts = [p.strip() for p in clean.split(",")][:3]
    return tuple(int(float(p)) for p in parts)


# ─── Static Template & CSS Contract Tests ────────────────────────────────────

def test_template_contracts_sticky_header_and_dock():
    """Verify HTML template has sticky navigation header and section outline containers."""
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="sticky-workspace-header"' in html, "Missing #sticky-workspace-header sticky container"
    assert 'id="workspace-nav"' in html, "Missing #workspace-nav"
    assert 'id="subtab-bar"' in html, "Missing #subtab-bar"
    assert 'id="dock-section-outline"' in html, "Missing #dock-section-outline in left dock"
    assert 'id="dock-icon-rail"' in html, "Missing #dock-icon-rail in left dock"
    assert 'id="dock-telemetry-widget"' in html, "Missing #dock-telemetry-widget in left dock"
    assert 'id="nav-landing-home-btn"' in html, "Missing #nav-landing-home-btn"
    assert 'id="dock-landing-home-btn"' in html, "Missing #dock-landing-home-btn"
    assert 'id="hud-landing-home-btn"' in html, "Missing #hud-landing-home-btn"
    assert 'id="health-telemetry-modal"' in html, "Missing #health-telemetry-modal"
    assert 'id="theme-toggle-btn"' in html, "Missing #theme-toggle-btn"


def test_css_contracts_tokyo_sand_overrides():
    """Verify CSS has high contrast definitions for Tokyo Sand and sticky header."""
    css_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "css", "style.css"))
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    assert 'html[data-theme="tokyo-sand"]' in css
    assert 'html[data-theme="tokyo-sand"] .workspace-btn.active' in css
    assert 'html[data-theme="tokyo-sand"] #nav-landing-home-btn' in css
    assert 'html[data-theme="tokyo-sand"] #dock-landing-home-btn' in css
    assert '#sticky-workspace-header' in css
    assert '#left-dock.dock-collapsed' in css
    assert '.dock-outline-btn' in css


# ─── Live Playwright Interactive & Visual Tests ──────────────────────────────

def launch_browser(playwright_instance):
    """Launch headless browser with chrome/edge fallback."""
    for channel in ["chrome", "msedge"]:
        try:
            return playwright_instance.chromium.launch(channel=channel, headless=True)
        except Exception:
            pass
    return playwright_instance.chromium.launch(headless=True)


def test_playwright_tokyo_sand_contrast_and_interactions():
    """End-to-end interactive verification of Tokyo Sand light theme readability."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to workbench
        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # 2. Switch theme to Tokyo Sand via 1-click #theme-toggle-btn
        curr_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if curr_theme != "tokyo-sand":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(350)

        theme_attr = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if theme_attr != "tokyo-sand":
            page.evaluate("window.applyTheme && window.applyTheme('tokyo-sand')")
            page.wait_for_timeout(300)
            theme_attr = page.evaluate("document.documentElement.getAttribute('data-theme')")
        assert theme_attr == "tokyo-sand", f"Theme attribute not set to tokyo-sand: {theme_attr}"

        # 3. Assert active workspace button contrast (Executive Dashboard)
        ws_active = page.locator(".workspace-btn.active")
        assert ws_active.count() > 0
        ws_styles = page.evaluate("""
            () => {
                const el = document.querySelector('.workspace-btn.active');
                if (!el) return null;
                const comp = window.getComputedStyle(el);
                return {
                    color: comp.color,
                    backgroundColor: comp.backgroundColor
                };
            }
        """)
        assert ws_styles is not None
        fg = parse_rgb(ws_styles["color"])
        bg = parse_rgb(ws_styles["backgroundColor"])
        ws_contrast = contrast_ratio(fg, bg)
        assert ws_contrast >= 4.5, f"Active workspace button contrast {ws_contrast:.2f}:1 fails WCAG AA (< 4.5:1)"

        # 4. Assert Home buttons readability
        home_btn_styles = page.evaluate("""
            () => {
                const nav = document.querySelector('#nav-landing-home-btn');
                const dock = document.querySelector('#dock-landing-home-btn');
                const hud = document.querySelector('#hud-landing-home-btn');
                const navComp = nav ? window.getComputedStyle(nav) : null;
                const dockComp = dock ? window.getComputedStyle(dock) : null;
                const hudComp = hud ? window.getComputedStyle(hud) : null;
                return {
                    navColor: navComp?.color,
                    navBg: navComp?.backgroundColor,
                    dockColor: dockComp?.color,
                    dockBg: dockComp?.backgroundColor,
                    hudColor: hudComp?.color,
                    hudBg: hudComp?.backgroundColor
                };
            }
        """)
        assert home_btn_styles["navColor"] is not None
        assert home_btn_styles["dockColor"] is not None

        # Verify dark text on light backgrounds for Tokyo Sand
        dock_fg = parse_rgb(home_btn_styles["dockColor"])
        # In Tokyo Sand, text must be dark (r, g, b < 100)
        assert dock_fg[0] < 100 and dock_fg[1] < 100 and dock_fg[2] < 100, f"Dock Home button text {dock_fg} is too light in Tokyo Sand"

        # 5. Assert Empty State Return to Landing Page button readability
        empty_return_btn = page.locator('#tab-content a[title="Return to Landing Page"]')
        if empty_return_btn.count() > 0:
            btn_styles = page.evaluate("""
                () => {
                    const el = document.querySelector('#tab-content a[title="Return to Landing Page"]');
                    if (!el) return null;
                    const comp = window.getComputedStyle(el);
                    return { color: comp.color, bg: comp.backgroundColor };
                }
            """)
            if btn_styles and btn_styles["color"]:
                btn_fg = parse_rgb(btn_styles["color"])
                assert btn_fg[0] < 100 and btn_fg[1] < 100 and btn_fg[2] < 100, f"Empty return button text {btn_fg} is unreadable"

        # 6. Capture screenshot artifact for verification
        os.makedirs("frontend/img", exist_ok=True)
        page.screenshot(path="frontend/img/test_tokyo_sand_verified.png")

        browser.close()

        # Check console / page errors
        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Console errors found: {real_errors}"
        assert len(page_errors) == 0, f"Page errors found: {page_errors}"


def test_playwright_health_telemetry_modal():
    """Verify clicking Health Telemetry opens in-app HUD modal (never raw JSON in a new tab)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # Check modal is initially hidden
        modal = page.locator("#health-telemetry-modal")
        assert not modal.is_visible()

        # Click HUD Health Telemetry trigger
        page.click("#hud-health-telemetry-btn")
        page.wait_for_timeout(350)

        # Assert modal is now visible in-app
        assert modal.is_visible(), "Telemetry modal failed to open upon clicking HUD trigger"

        # Assert metric cards are populated
        cpu_val = page.locator("#telemetry-cpu-val").inner_text()
        ram_val = page.locator("#telemetry-ram-val").inner_text()
        storage_val = page.locator("#telemetry-storage-val").inner_text()
        assert len(cpu_val) > 0, "CPU metric not rendered in telemetry modal"
        assert len(ram_val) > 0, "RAM metric not rendered in telemetry modal"
        assert len(storage_val) > 0, "Storage metric not rendered in telemetry modal"

        # Assert dismissible via Escape
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        assert not modal.is_visible(), "Telemetry modal failed to close upon pressing Escape"

        browser.close()


def test_playwright_sticky_navigation_and_sidebar_outline_on_scroll():
    """Verify sticky navigation header stays pinned and sidebar section outline updates on scroll."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # 1. Assert Section Outline is populated in left dock on initial view
        page.wait_for_selector("#dock-outline-list .dock-outline-btn", timeout=5000)
        outline_btns = page.locator("#dock-outline-list .dock-outline-btn")
        assert outline_btns.count() >= 3, f"Expected >= 3 outline anchors, found {outline_btns.count()}"

        # 2. Ingest benchmark so canvas has full dataset content
        page.evaluate("window.quickLoadBenchmark && window.quickLoadBenchmark('Bank Credit Risk & Default')")
        page.wait_for_timeout(2500)

        # 3. Scroll canvas down by 600px
        page.evaluate("document.getElementById('center-canvas').scrollTop = 600")
        page.wait_for_timeout(300)

        # 4. Assert sticky workspace header remains visible in viewport and flush with canvas top
        header_pos = page.evaluate("""
            () => {
                const el = document.getElementById('sticky-workspace-header');
                const canvas = document.getElementById('center-canvas');
                if (!el || !canvas) return null;
                const rect = el.getBoundingClientRect();
                const cRect = canvas.getBoundingClientRect();
                return { top: rect.top, bottom: rect.bottom, height: rect.height, canvasTop: cRect.top };
            }
        """)
        assert header_pos is not None
        assert abs(header_pos["top"] - header_pos["canvasTop"]) < 2, f"Sticky header top ({header_pos['top']}) does not align flush with canvas top ({header_pos['canvasTop']})!"
        assert header_pos["height"] > 0

        # 5. Assert sidebar outline is still visible (never empty void)
        dock_visible = page.locator("#dock-section-outline").is_visible()
        assert dock_visible, "Section outline dock disappeared on scroll"

        # 6. Reset scroll to top, then click Benchmarks anchor to verify programmatic scroll
        page.evaluate("document.getElementById('center-canvas').scrollTop = 0")
        page.wait_for_timeout(200)

        outline_btns = page.locator("#dock-outline-list .dock-outline-btn")
        last_outline_btn = outline_btns.nth(outline_btns.count() - 1)
        last_outline_btn.click()
        page.wait_for_timeout(600)

        scroll_pos = page.evaluate("document.getElementById('center-canvas').scrollTop")
        assert scroll_pos > 100, f"Canvas failed to scroll on outline click: scrollTop = {scroll_pos}"

        # 6. Test Dock collapse into Icon Rail
        page.evaluate("window.toggleDock && window.toggleDock(false)")
        page.wait_for_timeout(300)
        rail_visible = page.locator("#dock-icon-rail").is_visible()
        assert rail_visible, "Icon rail failed to show when dock is collapsed"

        # 7. Take verification screenshot
        page.screenshot(path="frontend/img/test_sticky_scroll_dock_verified.png")

        browser.close()


def test_playwright_mobile_responsiveness():
    """Verify mobile viewport (375x667) maintains full canvas usability, responsive dock drawer, and zero errors."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 375, "height": 667})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{BASE_URL}/app", wait_until="networkidle")
        page.wait_for_timeout(500)

        # 1. Assert center canvas has full viewport width on mobile (not squished to 75px)
        canvas_box = page.locator("#center-canvas").bounding_box()
        assert canvas_box is not None
        assert canvas_box["width"] >= 350, f"Mobile center canvas width ({canvas_box['width']}px) is crushed (< 350px)!"
        assert canvas_box["x"] == 0, f"Mobile center canvas x ({canvas_box['x']}) is pushed off-screen!"

        # 2. Assert left dock is collapsed by default on mobile
        dock = page.locator("#left-dock")
        assert not dock.is_visible(), "Left dock should be collapsed by default on mobile viewports"

        # 3. Test tapping dock toggle button opens drawer overlay
        page.click("#toggle-dock-btn")
        page.wait_for_timeout(350)
        assert dock.is_visible(), "Left dock drawer failed to open upon clicking toggle button on mobile"
        open_dock_box = dock.bounding_box()
        assert open_dock_box["width"] >= 300, f"Opened dock width {open_dock_box['width']} is too small"

        # 4. Tap toggle button again to collapse dock
        page.click("#toggle-dock-btn")
        page.wait_for_timeout(350)
        assert not dock.is_visible(), "Left dock drawer failed to collapse back on mobile"

        # 5. Switch to Tokyo Sand on mobile and verify contrast
        curr_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if curr_theme != "tokyo-sand":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(350)
        theme_attr = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if theme_attr != "tokyo-sand":
            page.evaluate("window.applyTheme && window.applyTheme('tokyo-sand')")
            page.wait_for_timeout(300)

        # Verify active button contrast on mobile
        ws_styles = page.evaluate("""
            () => {
                const el = document.querySelector('.workspace-btn.active');
                if (!el) return null;
                const comp = window.getComputedStyle(el);
                return { color: comp.color, bg: comp.backgroundColor };
            }
        """)
        assert ws_styles is not None
        fg = parse_rgb(ws_styles["color"])
        bg = parse_rgb(ws_styles["bg"])
        ratio = contrast_ratio(fg, bg)
        assert ratio >= 4.5, f"Mobile active workspace button contrast {ratio:.2f}:1 fails WCAG AA"

        page.screenshot(path="frontend/img/test_mobile_tokyo_sand_verified.png")

        browser.close()

        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(real_errors) == 0, f"Mobile console errors: {real_errors}"
        assert len(page_errors) == 0, f"Mobile page errors: {page_errors}"


def test_playwright_1click_night_day_toggle_and_persistence():
    """Verify 1-click #theme-toggle-btn toggles between night and day and persists in dia_theme_mode."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(f"{BASE_URL}/app", wait_until="networkidle")

        # Initial theme defaults to night
        mode_0 = page.evaluate("() => localStorage.getItem('dia_theme_mode') || 'night'")
        assert mode_0 in ["night", "day"]

        # Click toggle button
        page.click("#theme-toggle-btn")
        page.wait_for_timeout(350)

        theme_attr = page.evaluate("document.documentElement.getAttribute('data-theme')")
        theme_mode = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        if mode_0 == "night":
            assert theme_attr == "tokyo-sand"
            assert theme_mode == "day"
        else:
            assert theme_attr == "cyber-aurora"
            assert theme_mode == "night"

        # Click toggle button again
        page.click("#theme-toggle-btn")
        page.wait_for_timeout(350)
        theme_attr_2 = page.evaluate("document.documentElement.getAttribute('data-theme')")
        theme_mode_2 = page.evaluate("() => localStorage.getItem('dia_theme_mode')")
        assert theme_mode_2 == mode_0

        browser.close()

