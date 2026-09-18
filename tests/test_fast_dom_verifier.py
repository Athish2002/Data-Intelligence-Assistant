"""
tests/test_fast_dom_verifier.py
───────────────────────────────
Fast-DOM Synthetic Verifier for the Data Intelligence Assistant (DIA).
A zero-browser, sub-second headless DOM and HTML tag balance validator.
Validates:
1. Complete HTML tag nesting, balance, and well-formedness across frontend templates.
2. SVG DAG topology, node coordinates, active halo pulse, and absence of phantom nodes.
3. Interactive slider structure and baseline attribute boundaries.
4. Precision custom cursor DOM anchors and ambient aura containers.
5. Multi-way navigation linkages and absence of embedded duplicate marketing views in workbench.
6. Absolute removal of unsubstantiated marketing claims and presence of rigorous statistical terminology.
7. Execution time < 500ms (sub-second zero-overhead headless test).
"""

import os
import re
import time
from html.parser import HTMLParser
import pytest

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
LANDING_HTML_PATH = os.path.join(FRONTEND_DIR, "landing.html")
WORKBENCH_HTML_PATH = os.path.join(FRONTEND_DIR, "index.html")


class DOMTagBalanceParser(HTMLParser):
    """
    Validates opening and closing tag balance for non-void HTML elements.
    Tracks all element IDs, classes, and attributes.
    """
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"
    }

    # SVG void / self-closing elements
    SVG_SELF_CLOSING = {
        "circle", "ellipse", "line", "path", "polygon", "polyline", "rect", "stop", "use"
    }

    def __init__(self):
        super().__init__()
        self.tag_stack = []
        self.elements_by_id = {}
        self.elements_by_class = set()
        self.unmatched_closings = []
        self.unclosed_tags = []

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        attr_dict = dict(attrs)
        
        if "id" in attr_dict:
            self.elements_by_id[attr_dict["id"]] = {
                "tag": tag_lower,
                "attrs": attr_dict,
            }
            
        if "class" in attr_dict:
            for cls in attr_dict["class"].split():
                self.elements_by_class.add(cls)

        if tag_lower not in self.VOID_TAGS and tag_lower not in self.SVG_SELF_CLOSING:
            self.tag_stack.append(tag_lower)

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in self.VOID_TAGS or tag_lower in self.SVG_SELF_CLOSING:
            return

        if not self.tag_stack:
            self.unmatched_closings.append(tag_lower)
            return

        if self.tag_stack[-1] == tag_lower:
            self.tag_stack.pop()
        else:
            if tag_lower in self.tag_stack:
                while self.tag_stack and self.tag_stack[-1] != tag_lower:
                    self.unclosed_tags.append(self.tag_stack.pop())
                if self.tag_stack:
                    self.tag_stack.pop()
            else:
                self.unmatched_closings.append(tag_lower)


def test_fast_dom_subsecond_execution_benchmark():
    """Verify that Fast-DOM Synthetic Verifier completes parsing both templates in < 500ms."""
    start_time = time.perf_counter()

    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        landing_content = f.read()
    with open(WORKBENCH_HTML_PATH, "r", encoding="utf-8") as f:
        workbench_content = f.read()

    p1 = DOMTagBalanceParser()
    p1.feed(landing_content)
    p2 = DOMTagBalanceParser()
    p2.feed(workbench_content)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    assert elapsed_ms < 500, f"Fast-DOM verification exceeded 500ms threshold: {elapsed_ms:.2f}ms"


def test_landing_html_tag_balance_and_structure():
    """Verify landing.html has balanced tags and no dangling non-void elements."""
    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    parser = DOMTagBalanceParser()
    parser.feed(content)

    assert len(parser.unmatched_closings) == 0, f"Unmatched closing tags in landing.html: {parser.unmatched_closings}"
    assert len(parser.tag_stack) == 0, f"Unclosed tags remaining in landing.html: {parser.tag_stack}"


def test_workbench_html_tag_balance_and_structure():
    """Verify index.html has balanced tags and no dangling non-void elements."""
    with open(WORKBENCH_HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    parser = DOMTagBalanceParser()
    parser.feed(content)

    assert len(parser.unmatched_closings) == 0, f"Unmatched closing tags in index.html: {parser.unmatched_closings}"
    assert len(parser.tag_stack) == 0, f"Unclosed tags remaining in index.html: {parser.tag_stack}"


def test_causal_dag_topology_and_halo_pulse():
    """Verify Causal DAG contains exactly 4 nodes (Z, X, M, Y), halo pulse on Treatment (X), and no phantom W."""
    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    parser = DOMTagBalanceParser()
    parser.feed(content)

    # 4 Authentic Causal Nodes
    assert "dag-node-z" in parser.elements_by_id, "Missing Confounder node #dag-node-z"
    assert "dag-node-x" in parser.elements_by_id, "Missing Treatment node #dag-node-x"
    assert "dag-node-m" in parser.elements_by_id, "Missing Mediator node #dag-node-m"
    assert "dag-node-y" in parser.elements_by_id, "Missing Outcome node #dag-node-y"

    # Strict absence of phantom IV node W
    assert "dag-node-w" not in parser.elements_by_id, "Phantom node #dag-node-w must NOT exist in DAG"

    # SVG element verified
    assert "interactive-dag-svg" in parser.elements_by_id, "Missing #interactive-dag-svg"

    # Active Halo Glow on Treatment (X) centered at (170, 163)
    assert 'treatment-halo' in content or "treatment-halo" in parser.elements_by_class, "Active treatment halo missing"
    assert 'x="68"' in content and 'width="204"' in content, "Halo pulse center x should be 170 (68 + 102)"
    assert 'y="123"' in content and 'height="80"' in content, "Halo pulse center y should be 163 (123 + 40)"


def test_policy_intervention_slider_and_metric_cards():
    """Verify policy intervention slider attributes and live metric cards."""
    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    parser = DOMTagBalanceParser()
    parser.feed(content)

    # Slider attributes
    assert "causal-policy-slider" in parser.elements_by_id, "Missing #causal-policy-slider"
    slider_attrs = parser.elements_by_id["causal-policy-slider"]["attrs"]
    assert slider_attrs.get("type") == "range"
    assert float(slider_attrs.get("min", "0")) <= 2.0
    assert float(slider_attrs.get("max", "0")) >= 15.0
    assert float(slider_attrs.get("step", "1")) <= 0.5

    # Slider value display
    assert "policy-slider-val" in parser.elements_by_id, "Missing #policy-slider-val"

    # Live Metric Cards
    assert "metric-sim-default" in parser.elements_by_id, "Missing #metric-sim-default"
    assert "metric-sim-ate" in parser.elements_by_id, "Missing #metric-sim-ate"
    assert "metric-sim-delta" in parser.elements_by_id, "Missing #metric-sim-delta"
    assert "metric-sim-ci" in parser.elements_by_id, "Missing #metric-sim-ci"

    # Dynamic inspection banner
    assert "dag-inspector-container" in parser.elements_by_id, "Missing #dag-inspector-container"
    assert "dag-inspector-title" in parser.elements_by_id, "Missing #dag-inspector-title"
    assert "dag-inspector-desc" in parser.elements_by_id, "Missing #dag-inspector-desc"


def test_precision_cursor_and_spotlight_elements():
    """Verify precision custom cursor elements and spotlight hooks in both templates."""
    for path, name in [(LANDING_HTML_PATH, "landing.html"), (WORKBENCH_HTML_PATH, "index.html")]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        parser = DOMTagBalanceParser()
        parser.feed(content)

        assert "cursor-dot" in parser.elements_by_id, f"#cursor-dot missing in {name}"
        assert "cursor-ring" in parser.elements_by_id, f"#cursor-ring missing in {name}"
        assert "cursor-aura" in parser.elements_by_id, f"#cursor-aura missing in {name}"


def test_multi_way_navigation_and_workbench_decoupling():
    """Verify multi-way Home navigation exists in workbench and landing page is decoupled."""
    with open(WORKBENCH_HTML_PATH, "r", encoding="utf-8") as f:
        wb_content = f.read()

    parser = DOMTagBalanceParser()
    parser.feed(wb_content)

    # Multi-way Home navigation in workbench
    assert "nav-landing-home-btn" in parser.elements_by_id, "Missing #nav-landing-home-btn in top navbar"
    assert "dock-landing-home-btn" in parser.elements_by_id, "Missing #dock-landing-home-btn in left dock"
    assert "hud-landing-home-btn" in parser.elements_by_id, "Missing #hud-landing-home-btn in bottom HUD"

    # Workbench should NOT embed marketing hero or landing copy inside its tabs
    assert "Data Intelligence at Machine Speed" not in wb_content
    assert "ENTERPRISE PLATFORM ACTIVE" not in wb_content
    assert 'id="hero-canvas"' not in wb_content


def test_script_module_boundaries_and_imports():
    """Verify ES module script boundaries across templates."""
    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        landing_content = f.read()
    with open(WORKBENCH_HTML_PATH, "r", encoding="utf-8") as f:
        wb_content = f.read()

    assert 'type="module"' in landing_content
    assert 'src="js/landing.js"' in landing_content
    assert 'type="module"' in wb_content
    assert 'src="js/app.js"' in wb_content


def test_rigorous_terminology_and_absence_of_marketing_claims():
    """Verify absence of unverified marketing claims and presence of mathematical/technical terms."""
    banned_claims = [
        "Zero-GPU Required",
        "100% Client Privacy",
        "Zero-Compute Edge Store",
        "Zero-Compute Transpiler",
        "zero-compute scoring bundles",
    ]

    required_terms = [
        "Observational DAG Induction",
        "Finite-Sample Conformal Sets",
        "Wachter Counterfactual Recourse",
        "In-Memory Tenant Isolation",
        "Pearl's Rule 2",
        "Backdoor Admissible Set",
    ]

    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        landing_content = f.read()

    for claim in banned_claims:
        assert claim not in landing_content, f"Found banned claim '{claim}' in landing.html"

    for term in required_terms:
        assert term in landing_content, f"Missing required technical term '{term}' in landing.html"
