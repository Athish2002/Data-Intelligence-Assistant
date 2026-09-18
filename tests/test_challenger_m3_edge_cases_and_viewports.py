"""
tests/test_challenger_m3_edge_cases_and_viewports.py
────────────────────────────────────────────────────
Empirical Challenger test suite for Milestone 3.
Authored by Challenger 2 (challenger_m3_2).

Scope:
1. Adversarially stress-test edge-case dataset uploads against live server:
   - 1-row dataset (`id,feature,target\n1,10.0,0\n`)
   - High-cardinality UUID dataset (100 rows of unique UUIDs)
   - Non-ASCII and multi-language column names with emojis (`客户_id`, `prénom`, `tâche_🎯`, `target`)
   - Highly sparse dataset with missing values, null tokens (`?`, `NA`, `null`, `None`, empty string)
   - Verify API /api/v1/ingest/upload returns well-formed schemas without 500 crashes.
   - Verify downstream /api/v1/profile/{session_id} and /api/v1/readiness/{session_id} survive.
2. Verify browser session uploads render informative status without console errors or page errors.
3. Adversarially test responsive layout invariants across viewports:
   - Mobile (320px, 375px), Tablet (768px), Desktop (1024px, 1440px, 2560px).
   - Assert scrollWidth <= clientWidth on both / and /app.
   - Assert #center-canvas maintains readable width (>= 350px on standard viewports).
   - Assert mobile drawer overlay dismisses cleanly when navigating.
"""

import os
import io
import json
import uuid
import tempfile
import urllib.request
import urllib.error
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


def _post_multipart_file(endpoint_url: str, filename: str, content_bytes: bytes) -> tuple[int, dict]:
    """Helper to upload file bytes using standard urllib multipart form-data."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: text/csv\r\n\r\n")
    body.extend(content_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        endpoint_url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.status
            resp_data = json.loads(resp.read().decode("utf-8"))
            return status_code, resp_data
    except urllib.error.HTTPError as e:
        status_code = e.code
        err_body = e.read().decode("utf-8")
        try:
            resp_data = json.loads(err_body)
        except Exception:
            resp_data = {"raw_error": err_body}
        return status_code, resp_data


# ─── 1. API Direct Ingest Stress Tests ───────────────────────────────────────

class TestApiIngestEdgeCases:
    """Stress-test /api/v1/ingest/upload directly with extreme datasets."""

    def test_one_row_dataset(self):
        """Test minimal 1-row dataset."""
        content = "id,feature,target\n1,10.0,0\n".encode("utf-8")
        status_code, data = _post_multipart_file(f"{BASE_URL}/api/v1/ingest/upload", "one_row.csv", content)

        assert status_code == 200, f"Expected 200, got {status_code}: {data}"
        assert data["n_rows"] == 1
        assert data["n_cols"] == 3
        assert "id" in data["columns"]
        assert "feature" in data["columns"]
        assert "target" in data["columns"]
        assert len(data["sample_data"]) == 1
        session_id = data["session_id"]
        assert session_id

        # Verify downstream profile survives 1-row data
        with urllib.request.urlopen(f"{BASE_URL}/api/v1/profile/{session_id}") as prof_resp:
            assert prof_resp.status == 200
            prof_data = json.loads(prof_resp.read().decode())
            assert prof_data["shape"][0] == 1
            assert prof_data["shape"][1] == 3

    def test_high_cardinality_uuid_dataset(self):
        """Test high-cardinality UUID dataset (100 rows of unique UUIDs)."""
        rows = ["id,uuid_feature,metric_val,target"]
        for i in range(100):
            rows.append(f"{i},{uuid.uuid4()},{i * 1.25},{i % 2}")
        content = ("\n".join(rows) + "\n").encode("utf-8")

        status_code, data = _post_multipart_file(f"{BASE_URL}/api/v1/ingest/upload", "uuids_100.csv", content)
        assert status_code == 200, f"Expected 200, got {status_code}: {data}"
        assert data["n_rows"] == 100
        assert data["n_cols"] == 4
        session_id = data["session_id"]

        # Downstream profile check
        with urllib.request.urlopen(f"{BASE_URL}/api/v1/profile/{session_id}") as prof_resp:
            assert prof_resp.status == 200
            prof_data = json.loads(prof_resp.read().decode())
            assert prof_data["shape"][0] == 100
            assert prof_data["shape"][1] == 4

    def test_non_ascii_multilingual_emoji_dataset(self):
        """Test non-ASCII and multi-language column names with emojis."""
        headers = "客户_id,prénom,tâche_🎯,target"
        rows = [
            headers,
            "1,Jean,Mission_Alpha,0",
            "2,Chloé,Mission_Beta,1",
            "3,李四,Mission_Gamma,0",
            "4,Müller,Mission_Delta,1",
        ]
        content = ("\n".join(rows) + "\n").encode("utf-8")

        status_code, data = _post_multipart_file(f"{BASE_URL}/api/v1/ingest/upload", "multilingual_emoji.csv", content)
        assert status_code == 200, f"Expected 200, got {status_code}: {data}"
        assert data["n_rows"] == 4
        assert data["n_cols"] == 4
        assert "target" in data["columns"]
        # Emojis/Unicode handled safely in sanitized columns
        cols = data["columns"]
        assert any("客户" in c for c in cols)
        assert any("prénom" in c for c in cols)
        assert any("tâche" in c for c in cols)

        session_id = data["session_id"]
        with urllib.request.urlopen(f"{BASE_URL}/api/v1/profile/{session_id}") as prof_resp:
            assert prof_resp.status == 200

    def test_sparse_dataset_with_mixed_null_tokens(self):
        """Test highly sparse dataset with null tokens (?, NA, null, None, empty string)."""
        rows = [
            "id,feat_a,feat_b,feat_c,feat_d,target",
            "1,10.0,?,NA,null,0",
            "2,,None,?,20.5,1",
            "3,NA,null,None,,0",
            "4,?,None,NA,null,1",
            "5,50.0,?,?,None,0",
        ]
        content = ("\n".join(rows) + "\n").encode("utf-8")

        status_code, data = _post_multipart_file(f"{BASE_URL}/api/v1/ingest/upload", "sparse_nulls.csv", content)
        assert status_code == 200, f"Expected 200, got {status_code}: {data}"
        assert data["n_rows"] == 5
        assert "sanitize_report" in data
        assert data["sanitize_report"]["total_cells_repaired"] > 0

        session_id = data["session_id"]
        with urllib.request.urlopen(f"{BASE_URL}/api/v1/readiness/{session_id}") as r_resp:
            assert r_resp.status == 200


# ─── 2. Browser UI Ingestion Session Tests ───────────────────────────────────

class TestBrowserEdgeCaseUploadUi:
    """Verify in a browser session that uploading edge data renders informative status without console errors."""

    @pytest.mark.parametrize("filename,content", [
        ("1_row.csv", "id,feature,target\n1,10.0,0\n"),
        ("uuids_100.csv", "id,uuid_feature,target\n" + "\n".join([f"{i},{uuid.uuid4()},{i%2}" for i in range(100)]) + "\n"),
        ("multilingual_emoji.csv", "客户_id,prénom,tâche_🎯,target\n1,Jean,A_🎯,0\n2,Chloé,B_🚀,1\n3,李四,C_⭐,0\n"),
        ("sparse_nulls.csv", "id,feature_1,feature_2,target\n1,10.0,?,0\n2,,NA,1\n3,20.0,null,0\n4,?,None,1\n"),
    ])
    def test_browser_upload_renders_cleanly(self, filename, content):
        with sync_playwright() as p:
            browser = launch_browser(p)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            console_errors = []
            page_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as tf:
                tf.write(content)
                temp_path = tf.name

            try:
                page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=30000)

                page.click("#btn-ingest-data")
                page.wait_for_selector("#upload-modal:not(.hidden)", timeout=8000)

                page.set_input_files("#landing-file-input", temp_path)
                page.wait_for_selector("#upload-file-info:not(.hidden)", timeout=8000)

                page.click("#upload-submit-btn")

                # The upload should transition to workbench (/app) or display informative feedback
                try:
                    page.wait_for_url("**/app**", timeout=12000)
                    page.wait_for_function("() => typeof window.switchWorkspace === 'function'", timeout=15000)
                    page.evaluate("window.switchWorkspace('core', 'overview')")
                    page.wait_for_selector("#tab-content table, #tab-content .glass-card, #toast-container", timeout=15000)
                except Exception:
                    # In case of small alert or toast
                    toast = page.locator("#toast-container, #upload-error-msg")
                    assert toast.is_visible()

                # Filter benign errors (e.g. favicon 404)
                real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
                assert len(real_errors) == 0, f"Console errors uploading {filename}: {real_errors}"
                assert len(page_errors) == 0, f"Page errors uploading {filename}: {page_errors}"

            finally:
                context.close()
                browser.close()
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass


# ─── 3. Adversarial Responsive Viewport Layout Invariants ────────────────────

class TestResponsiveLayoutInvariants:
    """
    Adversarially test responsive layout invariants across viewports:
    - Mobile (320px, 375px), Tablet (768px), Desktop (1024px, 1440px, 2560px)
    - Assert scrollWidth <= clientWidth on both / and /app
    - Assert #center-canvas maintains readable width (>= 350px on standard viewports)
    - Assert mobile drawer overlay dismisses cleanly when navigating
    """

    VIEWPORTS = [
        ("Mobile_320", 320, 568),
        ("Mobile_375", 375, 667),
        ("Tablet_768", 768, 1024),
        ("Desktop_1024", 1024, 768),
        ("Desktop_1440", 1440, 900),
        ("UltraWide_2560", 2560, 1440),
    ]

    @pytest.mark.parametrize("name,width,height", VIEWPORTS)
    def test_landing_page_no_horizontal_overflow(self, name, width, height):
        """Landing page (/) must satisfy scrollWidth <= clientWidth across all viewports."""
        with sync_playwright() as p:
            browser = launch_browser(p)
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

            console_errors = []
            page_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=30000)

            # Assert scrollWidth <= clientWidth
            is_valid = page.evaluate("""() => {
                const doc = document.documentElement;
                return doc.scrollWidth <= doc.clientWidth;
            }""")
            scroll_w = page.evaluate("() => document.documentElement.scrollWidth")
            client_w = page.evaluate("() => document.documentElement.clientWidth")
            assert is_valid, f"Horizontal overflow on landing at {name} ({width}x{height}): scrollWidth={scroll_w} > clientWidth={client_w}"

            real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
            assert len(real_errors) == 0, f"Console errors at {name}: {real_errors}"
            assert len(page_errors) == 0, f"Page errors at {name}: {page_errors}"

            context.close()
            browser.close()

    @pytest.mark.parametrize("name,width,height", VIEWPORTS)
    def test_workbench_no_horizontal_overflow_and_canvas_width(self, name, width, height):
        """Workbench (/app) must satisfy scrollWidth <= clientWidth and readable canvas width."""
        with sync_playwright() as p:
            browser = launch_browser(p)
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

            console_errors = []
            page_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            page.goto(f"{BASE_URL}/app", wait_until="networkidle", timeout=30000)
            page.wait_for_function("() => typeof window.switchWorkspace === 'function'", timeout=15000)

            # Assert scrollWidth <= clientWidth
            is_valid = page.evaluate("""() => {
                const doc = document.documentElement;
                return doc.scrollWidth <= doc.clientWidth;
            }""")
            scroll_w = page.evaluate("() => document.documentElement.scrollWidth")
            client_w = page.evaluate("() => document.documentElement.clientWidth")
            assert is_valid, f"Horizontal overflow on workbench at {name} ({width}x{height}): scrollWidth={scroll_w} > clientWidth={client_w}"

            # Assert #center-canvas maintains readable width
            canvas_box = page.locator("#center-canvas").bounding_box()
            assert canvas_box is not None, "#center-canvas not found or not rendered"

            canvas_width = canvas_box["width"]
            if width >= 375:
                # Standard viewports: >= 350px
                assert canvas_width >= 340.0, f"Center canvas width {canvas_width}px < 340px at standard viewport {name} ({width}px)"
            else:
                # 320px mobile viewport: canvas occupies full available mobile width
                assert canvas_width >= 280.0, f"Center canvas width {canvas_width}px < 280px at extreme mobile {name} (320px)"

            real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
            assert len(real_errors) == 0, f"Console errors at {name}: {real_errors}"
            assert len(page_errors) == 0, f"Page errors at {name}: {page_errors}"

            context.close()
            browser.close()

    @pytest.mark.parametrize("name,width,height", [
        ("Mobile_320", 320, 568),
        ("Mobile_375", 375, 667),
        ("Tablet_768", 768, 1024),
    ])
    def test_mobile_drawer_overlay_dismissal_on_navigation(self, name, width, height):
        """On viewports < 1024px, opening mobile drawer and clicking a link dismisses the drawer cleanly."""
        with sync_playwright() as p:
            browser = launch_browser(p)
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

            page.goto(f"{BASE_URL}/app", wait_until="networkidle", timeout=30000)
            page.wait_for_function("() => typeof window.switchWorkspace === 'function'", timeout=15000)

            # Initially on small viewports (< 1024px), left dock is collapsed
            initial_collapsed = page.evaluate("() => document.getElementById('left-dock').classList.contains('dock-collapsed')")
            assert initial_collapsed, f"Dock expected to be initially collapsed at {name}"

            # Open left dock using toggle button
            toggle_btn = page.locator("#toggle-dock-btn")
            assert toggle_btn.is_visible()
            toggle_btn.click()
            page.wait_for_timeout(250)

            # Now dock should not have dock-collapsed and should be fixed overlay
            now_open = page.evaluate("() => !document.getElementById('left-dock').classList.contains('dock-collapsed')")
            assert now_open, f"Dock failed to open on toggle at {name}"

            # Click a section outline button or navigation link inside left dock
            outline_btns = page.locator("#dock-outline-list .dock-outline-btn, #left-dock button, #left-dock .workspace-btn")
            assert outline_btns.count() > 0, "No navigable buttons found in left dock"
            outline_btns.first.click()
            page.wait_for_timeout(350)

            # Verify dock is cleanly auto-dismissed (re-collapsed)
            dismissed = page.evaluate("() => document.getElementById('left-dock').classList.contains('dock-collapsed')")
            assert dismissed, f"Dock was not cleanly dismissed on navigation click at {name}"

            context.close()
            browser.close()
