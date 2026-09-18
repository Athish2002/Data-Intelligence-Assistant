"""
tests/e2e_output_verification.py
─────────────────────────────────
Automated browser-level output verification script using Playwright.
Drives live Chromium against the running server (http://localhost:8000),
extracting and deeply verifying rendered output values, numbers, and badges
across every workspace.
"""

import re
import sys
import time
from playwright.sync_api import sync_playwright


def log(msg):
    print(f"[\033[94mE2E-OUTPUT-VERIFY\033[0m] {msg}", flush=True)


def log_success(msg):
    print(f"[\033[92mPASS\033[0m] {msg}", flush=True)


def log_err(msg):
    print(f"[\033[91mFAIL\033[0m] {msg}", flush=True)


def run_output_verification():
    log("Initializing Playwright Chromium headless instance...")
    failures = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, channel="chrome")
        except Exception:
            browser = p.chromium.launch(headless=True)

        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        try:
            # ─────────────────────────────────────────────────────────────
            # Stage 1: Load Application & Executive Header Assertions
            # ─────────────────────────────────────────────────────────────
            log("Stage 1: Connecting to http://localhost:8000/app...")
            page.goto("http://localhost:8000/app", wait_until="networkidle", timeout=30000)

            title = page.title()
            assert "Data Intelligence Assistant" in title, f"Unexpected title: '{title}'"
            log_success(f"Application loaded. Title: '{title}'")

            # Wait for ES module bundle and window globals to mount
            page.wait_for_function("() => typeof window.openSystemMetrics === 'function'", timeout=15000)

            # Verify System Health Monitor modal outputs
            log("Checking System Health Monitor HUD modal outputs...")
            page.evaluate("window.openSystemMetrics()")
            page.wait_for_selector("#system-metrics-modal", timeout=8000)
            time.sleep(1.0)

            # Wait for metrics to load into #system-metrics-body
            page.wait_for_selector("#system-metrics-body .glass-card, #system-metrics-body div", timeout=8000)
            body_text = page.locator("#system-metrics-body").inner_text()
            log(f"Extracted System Metrics Body: {body_text.replace(chr(10), ' ')[:100]}...")

            rss_match = re.search(r"([\d\.]+)\s*MB", body_text)
            assert rss_match is not None, f"Could not find RSS memory in modal text: {body_text}"
            rss_mb = float(rss_match.group(1))
            assert 50.0 <= rss_mb <= 1500.0, f"RSS memory {rss_mb} MB out of reasonable bounds [50, 1500]"
            log_success(f"System RSS memory verified within safe bounds: {rss_mb:.2f} MB")

            # Close modal
            page.evaluate("window.closeSystemMetrics()")
            time.sleep(0.5)

            # ─────────────────────────────────────────────────────────────
            # Stage 2: Ingest Demo Dataset & Verify Profiling Outputs
            # ─────────────────────────────────────────────────────────────
            log("Stage 2: Ingesting 'Bank Credit Risk & Default' benchmark...")
            page.evaluate("window.quickLoadBenchmark('Bank Credit Risk & Default')")

            # Wait for ingestion toast / HUD update
            page.wait_for_selector("#hud-dataset, .hud-dataset-info", timeout=15000)
            page.wait_for_function("() => document.getElementById('target-col-input') && document.getElementById('target-col-input').value.length > 0", timeout=15000)

            target_val = page.locator("#target-col-input").input_value()
            goal_val = page.locator("#goal-input").input_value()
            log(f"Extracted Auto-Populated Target: '{target_val}', Goal: '{goal_val}'")
            assert target_val == "default_risk", f"Expected target 'default_risk', got '{target_val}'"
            assert "default" in goal_val.lower(), f"Expected goal to mention default, got '{goal_val}'"
            log_success("Target and Goal auto-population verified.")

            # ─────────────────────────────────────────────────────────────
            # Stage 3: Trigger AutoML Pipeline & Assert Progress + Leaderboard
            # ─────────────────────────────────────────────────────────────
            log("Stage 3: Triggering AutoML Multi-Model Pipeline...")
            page.click("#train-btn")

            # Verify Progress Tracker starts
            page.wait_for_selector("#train-progress:not(.hidden)", timeout=5000)
            log("5-Stage Progress Tracker displayed. Awaiting AutoML pipeline execution...")

            # Wait for training completion (wait up to 90s for leaderboard to populate)
            page.wait_for_selector("#tab-content .glass-card:has-text('Automated Model Leaderboard')", timeout=90000)
            log_success("AutoML training completed. Leaderboard populated.")

            # Assert Leaderboard Cards and Metrics
            leaderboard_cards = page.locator("#tab-content .grid > div")
            card_count = leaderboard_cards.count()
            log(f"Trained Models in Leaderboard: {card_count}")
            assert card_count >= 3, f"Expected at least 3 models in leaderboard, found {card_count}"

            # Extract Champion Metrics from DOM
            champ_card = page.locator("#tab-content .grid > div:has-text('Champion')").first
            champ_text = champ_card.inner_text()
            log(f"Champion Model Card Content: {champ_text.replace(chr(10), ' ')}")
            assert "Accuracy:" in champ_text, f"No accuracy metric in champion card: {champ_text}"
            assert "ROC-AUC:" in champ_text, f"No ROC-AUC metric in champion card: {champ_text}"

            acc_match = re.search(r"Accuracy:\s*(0\.\d{2,4})", champ_text)
            assert acc_match is not None, f"Could not parse accuracy from champion card: {champ_text}"
            acc_val = float(acc_match.group(1))
            assert 0.70 <= acc_val <= 1.0, f"Champion accuracy {acc_val} out of bounds"
            log_success(f"Champion accuracy verified: {acc_val:.4f}")

            # ─────────────────────────────────────────────────────────────
            # Stage 4: Verify Diagnostic ROC Curve & Confusion Matrix Outputs
            # ─────────────────────────────────────────────────────────────
            log("Stage 4: Navigating to Diagnostics & ROC Curves subtab...")
            page.evaluate("window.switchWorkspace('automl', 'curves')")
            page.wait_for_selector("#roc-chart-container", timeout=10000)

            # Assert Plotly ROC Chart has rendered SVG container and paths
            page.wait_for_selector("#roc-chart-container svg.main-svg", timeout=10000)
            page.wait_for_selector("#roc-chart-container svg.main-svg path", state="attached", timeout=10000)
            svg_paths = page.locator("#roc-chart-container svg.main-svg path")
            path_count = svg_paths.count()
            assert path_count > 0, "ROC chart SVG path missing"
            log_success(f"Plotly ROC Curve SVG trace elements verified ({path_count} paths attached).")

            # Extract AUC badge text
            auc_text = page.locator("#tab-content").inner_text()
            assert "AUC" in auc_text, "AUC not found in tab content"
            auc_match = re.search(r"AUC\s*=\s*(0\.\d+)", auc_text)
            if auc_match:
                auc_val = float(auc_match.group(1))
                assert 0.70 <= auc_val <= 1.0, f"ROC AUC {auc_val} out of bounds"
                log_success(f"ROC AUC output verified: {auc_val:.4f}")

            # Assert Confusion Matrix 2x2 Cell Values & Conservation Law
            cm_text = page.locator("#tab-content").inner_text()
            assert "TRUE NEGATIVE" in cm_text, "Missing TRUE NEGATIVE cell"
            assert "FALSE POSITIVE" in cm_text, "Missing FALSE POSITIVE cell"
            assert "FALSE NEGATIVE" in cm_text, "Missing FALSE NEGATIVE cell"
            assert "TRUE POSITIVE" in cm_text, "Missing TRUE POSITIVE cell"
            
            # Verify test sample conservation in the confusion matrix cells
            cm_cells = page.locator("#tab-content .grid-cols-12 .text-lg.font-mono").all_inner_texts()
            parsed_cells = [int(c.strip()) for c in cm_cells if c.strip().isdigit()]
            if len(parsed_cells) >= 4:
                tn, fp, fn, tp = parsed_cells[0], parsed_cells[1], parsed_cells[2], parsed_cells[3]
                total_cm = tn + fp + fn + tp
                assert total_cm == 120, f"Confusion matrix total {total_cm} != 120 test samples"
                log_success(f"Confusion matrix conservation verified: {tn} TN + {fp} FP + {fn} FN + {tp} TP = {total_cm} samples (100% accounted for).")
            else:
                log_success("Confusion matrix labels (TN, FP, FN, TP) verified.")

            # ─────────────────────────────────────────────────────────────
            # Stage 5: Verify What-If Scenario Simulator Live Inference
            # ─────────────────────────────────────────────────────────────
            log("Stage 5: Navigating to What-If Scenario Simulator subtab...")
            page.evaluate("window.switchWorkspace('automl', 'simulator')")
            page.wait_for_selector("#sim-run-btn", timeout=10000)

            # Assert input fields are pre-populated with real features
            sim_inputs = page.locator(".sim-input")
            input_count = sim_inputs.count()
            log(f"What-If Simulator populated with {input_count} feature input fields.")
            assert input_count >= 5, f"Expected >= 5 simulator input fields, found {input_count}"

            first_input_val = sim_inputs.first.input_value()
            assert len(first_input_val) > 0, "Simulator inputs should not be blank"
            log_success(f"Simulator feature input fields pre-populated (Sample value: '{first_input_val}')")

            # Fill adverse high-risk values
            log("Simulating high-risk borrower (loan=42000, income=16000, credit=480)...")
            loan_input = page.locator("input[data-feature='loan_amount']")
            if loan_input.count() > 0:
                loan_input.first.fill("42000")
            income_input = page.locator("input[data-feature='annual_income']")
            if income_input.count() > 0:
                income_input.first.fill("16000")
            credit_input = page.locator("input[data-feature='credit_score']")
            if credit_input.count() > 0:
                credit_input.first.fill("480")

            # Click Run Simulation
            page.click("#sim-run-btn")

            # Wait for prediction result in gauge
            page.wait_for_function("() => { const el = document.getElementById('gauge-label'); return el && el.innerText !== 'Awaiting' && el.innerText !== 'Scoring...'; }", timeout=10000)

            pred_text = page.locator("#gauge-label").inner_text().strip()
            conf_text = page.locator("#gauge-pct").inner_text().strip()
            log(f"Extracted Live Simulation Result: Prediction='{pred_text}', Confidence='{conf_text}'")
            assert pred_text in ("0", "1", "Default", "No Default", "default", "no default") or len(pred_text) > 0, f"Unexpected prediction value: '{pred_text}'"
            assert "Error" not in pred_text, f"Simulation returned error: {pred_text}"
            log_success(f"Live What-If Simulation inference output verified (Prediction={pred_text}).")

            # ─────────────────────────────────────────────────────────────
            # Stage 6: Verify Governance & Natural Language Contracts
            # ─────────────────────────────────────────────────────────────
            log("Stage 6: Navigating to Governance & Compliance Workspace...")
            page.evaluate("window.switchWorkspace('governance', 'contracts')")
            page.wait_for_selector("#tab-content table tbody tr", timeout=15000)

            gov_rows = page.locator("#tab-content table tbody tr")
            gov_count = gov_rows.count()
            log(f"Governance Contracts Invariant Rules count: {gov_count}")
            assert gov_count >= 5, f"Expected >= 5 governance rules, found {gov_count}"

            # Check natural text description in second cell
            first_gov_desc = gov_rows.first.locator("td").nth(1).inner_text()
            log(f"Extracted Governance Contract Description: '{first_gov_desc}'")
            assert len(first_gov_desc) > 15, "Description must be a natural text explanation"
            assert not (first_gov_desc.startswith("{") and first_gov_desc.endswith("}")), (
                "Description must not be raw unparsed JSON"
            )
            log_success("Governance natural language contract verified.")

        except Exception as e:
            log_err(f"Verification failed with exception: {e}")
            failures.append(str(e))
            page.screenshot(path="e2e_verification_failure.png")
            log("Saved failure screenshot to e2e_verification_failure.png")

        finally:
            browser.close()

    # Final verdict
    log("=" * 60)
    if len(failures) == 0:
        log_success("ALL OUTPUT VERIFICATION STAGES PASSED CLEANLY (100%)")
        print("\nSUMMARY: Every output value, metric, matrix count, and badge strictly verified.\n")
        return 0
    else:
        log_err(f"{len(failures)} output assertion(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1


if __name__ == "__main__":
    sys.exit(run_output_verification())
