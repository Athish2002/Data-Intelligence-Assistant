---
name: e2e-output-verification-testing
description: Master rigorous end-to-end output verification, mathematical invariance testing, and deep assertion protocols with Playwright and pytest. Use PROACTIVELY to ensure automated tests validate real output values, metrics, and contracts rather than superficial status codes.
---

# End-to-End Output Verification & Invariant Testing (Pro)

Master of rigorous, high-fidelity automated testing that eliminates shallow, cosmetic assertions and mandates deep, deterministic verification of actual computational outputs, mathematical invariants, domain behaviors, and visual DOM values.

## Core Philosophy: Anti-Shallow Testing Protocol

A test that only asserts `response.status_code == 200` or `expect(element).toBeVisible()` does NOT prove correctness. A system can return HTTP 200 with an empty JSON object, a NaN prediction, a corrupted matrix, or a blank graph canvas.

### The 5 Laws of Output Verification

1. **Law of Non-Trivial Value Assertion**:
   - Every automated test MUST inspect the content, type, range, and value of the output.
   - Forbid asserting merely that a key exists; assert that `result["value"]` falls within the mathematically required bounds.
2. **Law of Mathematical Conservation**:
   - For classification contingency matrices: $TN + FP + FN + TP = N_{\text{test}}$.
   - For probability outputs: $0.0 \le P(y=1) \le 1.0$ and $\sum_c P(y=c) = 1.0 \pm 10^{-6}$.
   - For normalized feature importances: $\sum_i I(f_i) \approx 1.0$ and $I(f_i) \ge 0$.
   - For ROC curves: FPR and TPR arrays must be monotonically non-decreasing, non-empty, and start at $(0, 0)$ and end at $(1, 1)$.
3. **Law of Behavioral Monotonicity & Sensitivity**:
   - What-If simulations and counterfactuals MUST be tested with contrasting extreme inputs.
   - A prime low-risk profile MUST yield lower risk probability than an adverse high-risk profile ($P_{\text{low}} < P_{\text{high}}$).
4. **Law of DOM Text & State Conformance**:
   - In browser automation, assert exact or regex-matched rendered text in the DOM (`expect(el).toHaveText(...)`), not just locator presence.
   - Verify that badges, metrics cards, table rows, and charts reflect the exact calculated numerical values from the backend response.
5. **Law of No-Ghost Fallbacks**:
   - Automated tests must verify fallback states explicitly (e.g. continuous targets on ROC charts display an informative diagnostic card rather than an empty element).

---

## Capabilities & Implementation Patterns

### 1. Backend Invariant Testing with `pytest`
- **Output Schema & Strict Typings**: Pydantic model validation with strict type assertions.
- **Metric Verification Gates**: Ensure machine learning metrics (Accuracy, ROC-AUC, F1, Precision, Recall) exceed minimum acceptable domain thresholds ($> 0.70$) and are not hardcoded constants.
- **Data Quality Invariants**: Test data readiness scores, null fraction bounds, and column resolution confidence scores.

### 2. Live Browser Output Verification with Playwright
- **Automated User Journey Driven Testing**:
  - Ingest real data fixtures $\to$ Profile $\to$ Trigger AutoML $\to$ Wait for progress tracker completion $\to$ Assert leaderboard rows.
- **DOM Value Extraction**:
  ```python
  # Extract and parse numerical output from the DOM
  metric_text = page.locator("#champion-accuracy").inner_text()
  accuracy = float(metric_text.replace("%", "")) / 100.0
  assert 0.75 <= accuracy <= 1.0, f"Unexpected champion accuracy: {accuracy}"
  ```
- **Contingency Matrix Cell Verification**:
  ```python
  tn = int(page.locator("#cm-tn").inner_text())
  fp = int(page.locator("#cm-fp").inner_text())
  fn = int(page.locator("#cm-fn").inner_text())
  tp = int(page.locator("#cm-tp").inner_text())
  assert tn + fp + fn + tp == expected_test_count
  ```
- **Live What-If Simulation Form Verification**:
  ```python
  # Fill form with extreme perturbation and assert real-time prediction output
  page.fill("#input-loan_amnt", "50000")
  page.fill("#input-person_income", "12000")
  page.click("#run-simulation-btn")
  page.wait_for_selector("#sim-result-badge", state="visible")
  badge_text = page.locator("#sim-result-badge").inner_text()
  assert "Default" in badge_text
  ```

---

## Production Verification Checklist

- [ ] Every API test asserts payload data types, expected value ranges, and non-null guarantees.
- [ ] Classification tests assert confusion matrix cell conservation ($TN + FP + FN + TP = N$).
- [ ] Probability predictions are asserted within $[0.0, 1.0]$ with monotonic domain sensitivity.
- [ ] Playwright E2E tests assert rendered numerical metrics and status text in the DOM.
- [ ] No test passes on HTTP 200 alone without inspecting the response body.
