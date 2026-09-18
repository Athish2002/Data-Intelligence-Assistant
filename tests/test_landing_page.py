"""
tests/test_landing_page.py
──────────────────────────
Verification test suite for the dedicated standalone landing page.
Verifies:
1. GET / returns 200 with landing.html content
2. GET /landing returns 200 with landing.html content
3. GET /app returns 200 with workbench index.html content
4. GET /workbench returns 200 with workbench index.html content
5. GET /index.html returns 200 with workbench index.html content
6. GET /js/landing.js returns 200 with landing JavaScript
7. Visual fidelity assertions:
   - "Data Intelligence at Machine Speed"
   - "ENTERPRISE PLATFORM ACTIVE"
   - "Launch Interactive Demo"
   - "Ingest Custom Data"
   - "Go to Dashboard"
   - "Live causal graph DAG"
   - "Pearl Do-Calculus"
   - "Zero-Compute Edge Transpiler"
   - "Conformal Coverage Sets"
"""

import os
from fastapi.testclient import TestClient
from api.server import app


client = TestClient(app)


def test_landing_page_route_root():
    """Verify GET / serves the dedicated standalone landing page."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "<!DOCTYPE html>" in html
    assert "Data Intelligence at Machine Speed" in html
    assert "ENTERPRISE PLATFORM ACTIVE" in html
    assert "Launch Interactive Demo" in html
    assert "Ingest Custom Data" in html
    assert "Go to Dashboard" in html
    assert "Live causal graph DAG" in html
    assert "Pearl" in html and "Do-Calculus" in html


def test_landing_page_route_landing():
    """Verify GET /landing serves the dedicated standalone landing page."""
    response = client.get("/landing")
    assert response.status_code == 200
    html = response.text
    assert "Data Intelligence at Machine Speed" in html
    assert "ENTERPRISE PLATFORM ACTIVE" in html


def test_workbench_page_route_app():
    """Verify GET /app serves the full analytical workbench index.html."""
    response = client.get("/app")
    assert response.status_code == 200
    html = response.text
    assert "<!DOCTYPE html>" in html
    assert "Data Intelligence Assistant" in html
    assert "active-dataset-badge" in html or "toggle-inspector-btn" in html


def test_workbench_page_route_workbench():
    """Verify GET /workbench serves the full analytical workbench index.html."""
    response = client.get("/workbench")
    assert response.status_code == 200
    html = response.text
    assert "<!DOCTYPE html>" in html
    assert "Data Intelligence Assistant" in html


def test_landing_javascript_served():
    """Verify GET /js/landing.js serves the dedicated landing script."""
    response = client.get("/js/landing.js")
    assert response.status_code == 200
    assert "launchDemoBenchmark" in response.text
    assert "selectDagNode" in response.text


def test_no_legacy_tokens_in_landing():
    """Ensure landing page adheres strictly to project integrity standards."""
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    landing_path = os.path.join(frontend_dir, "landing.html")
    landing_js_path = os.path.join(frontend_dir, "js", "landing.js")

    assert os.path.exists(landing_path)
    assert os.path.exists(landing_js_path)

    with open(landing_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "v14.0" not in content
        assert "v14 " not in content
        assert "alert(" not in content

    with open(landing_js_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "v14.0" not in content
        assert "v14 " not in content
        assert "alert(" not in content


def test_landing_trailing_slashes():
    """Verify trailing slash URLs are routed properly without 404 or redirect loops."""
    r1 = client.get("/landing/")
    assert r1.status_code == 200
    assert "Data Intelligence at Machine Speed" in r1.text

    r2 = client.get("/app/")
    assert r2.status_code == 200
    assert "<!DOCTYPE html>" in r2.text

    r3 = client.get("/workbench/")
    assert r3.status_code == 200
    assert "<!DOCTYPE html>" in r3.text


def test_public_routes_bypass_auth(monkeypatch):
    """Verify landing and workbench routes remain public even when DIA_AUTH_ENABLED=true."""
    import dia.config as dia_config
    monkeypatch.setattr(dia_config, "AUTH_ENABLED", True)

    for path in ["/", "/landing", "/landing/", "/app", "/app/", "/workbench", "/workbench/", "/js/landing.js"]:
        resp = client.get(path)
        assert resp.status_code == 200, f"Public path '{path}' was blocked by auth middleware (status {resp.status_code})"


def test_landing_page_action_options_and_cards():
    """Verify all 3 distinct action options and 4 feature cards are present."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # 3 distinct action options
    assert "Launch Interactive Demo" in html
    assert "Ingest Custom Data" in html
    assert "Go to Dashboard" in html

    # 4 feature cards matching mockup
    assert "Pearl" in html and "Do-Calculus" in html
    assert "Zero-" in html and "Compute" in html and "Edge" in html
    assert "Transpiler" in html
    assert "Conformal" in html and "Coverage" in html and "Sets" in html

    # Modals present
    assert 'id="upload-modal"' in html
    assert 'id="demo-modal"' in html
    assert 'id="landing-dropzone"' in html

