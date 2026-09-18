# Project: Data Intelligence Assistant — Platform Architecture & Verification

## Architecture
Data Intelligence Assistant (DIA) is an enterprise-grade AI, AutoML, Causal Discovery, and Governance platform.
- **Frontend SPA**: `frontend/landing.html` (Landing Page) and `frontend/index.html` (Analytical Workbench).
  - Pure JavaScript modules in `frontend/js/` (`app.js`, `theme.js`, `landing.js`, `cursor.js`, and `views/*.js`).
  - Vanilla CSS in `frontend/css/style.css` utilizing design tokens for Cyber Aurora / Obsidian (Night Mode) and Tokyo Sand (Day Mode).
- **Backend API**: FastAPI application in `api/server.py` with Pydantic V2 schemas in `api/schemas.py`.
- **Core ML & Reasoning Engines (`dia/`)**:
  - `dia/data_profiler.py`: Column role and type inference.
  - `dia/pipeline_coordinator.py`: End-to-end AutoML pipeline coordination.
  - `dia/model_trainer.py`: Scikit-learn, XGBoost, LightGBM model training and scoring.
  - `dia/recourse_engine.py`: Wachter counterfactual recourse in normalized L1/MAD space.
  - `dia/causal_discovery.py`: PC algorithm CPDAG causal discovery and Pearl's Do-Calculus.
  - `dia/wasm_compiler.py`: Edge model transpiler generating standalone JS and WAT scoring bundles.
  - `dia/leakage_detector.py`: Information-theoretic target leakage and data poisoning detection.
  - `dia/pareto_frontier.py`: Multi-objective Pareto frontier optimizer.
  - `dia/deep_autoencoder.py`: Anomaly detection autoencoder with strict memory caps.
  - `dia/demo_datasets.py`: Authentic benchmark scenario generators.
- **Test Infrastructure (`tests/`)**:
  - 334+ backend pytest unit, integration, and property tests.
  - Playwright end-to-end browser test suites.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 1-Click Night/Day Toggle | Replace `<select id="theme-selector">` with 1-click `#theme-toggle-btn` (🌙 / ☀️) across `/` and `/app` | M1 | Survey / R1 |
| 2 | 2 Polished Themes | Strict 2-theme system: Night Mode (Cyber Aurora / Obsidian `#040608`) and Day Mode (Tokyo Sand `#f5f2eb`) | M1 | Survey / R1 |
| 3 | WCAG 2.2 AA Contrast | Text and UI contrast >= 4.5:1 across both themes; upgrade green accent to `#166534` for Tokyo Sand | M1 | Survey / R1 |
| 4 | Theme Persistence & Debounce | Store in `localStorage` (`dia_theme_mode`) with rapid-toggle debounce lock preventing layout/CSS stutter | M1 | Survey / R1 |
| 5 | Pinned 3-Pane Workbench Layout | `#left-dock`, `#center-canvas`, `#inspector-drawer` starting flush at y=45px, `overflow: hidden` on body, scroll confined to center canvas | M1 | Survey / R2 |
| 6 | Workbench Sticky Header | `#sticky-workspace-header` pinned on scroll with 100% opaque `--bg-dock` shelf background and 1px border divider | M1 | Survey / R2 |
| 7 | Landing Page Natural Scroll & Sticky Header | Natural vertical document scrolling (`overflow-y: auto`), sticky header at `y=0`, section anchors | M1 | Survey / R2 |
| 8 | Responsive Viewports & Mobile Drawer | Responsive viewports (375px mobile, 768px tablet, 1440px+ desktop); auto-dismiss left dock on mobile link click | M1 | Survey / R2 |
| 9 | Custom Cursor Physics & Touch Suppression | Offscreen init (`translate3d(-999px, -999px, 0)`), spring lerp (0.20), suppression on `body.touch-device-active` | M1 | Survey / R2 |
| 10 | Boolean Column Preservation | Include `bool` features in `ColumnTransformer` numeric/categorical pipelines without silent drops | M2 | Survey / R3 |
| 11 | Binary Target Encoding & Directionality | Prevent lexicographical inversion of binary labels (`Default` vs `Non-Default`) in `model_trainer.py` | M2 | Survey / R3 |
| 12 | 1-to-1 Feature & Matrix Alignment | Raw DataFrame -> Preprocessed Matrix -> Models -> SHAP Values -> UI Importance charts without column drops | M2 | Survey / R3 |
| 13 | Voting Ensemble Genuine Feature Importance | Compute genuine importance for Voting Ensemble averaging base estimator importances (eliminate 0.05 mock) | M2 | Survey / R3 |
| 14 | Wachter Recourse Invariant Inversion | Invert normalized L1/MAD perturbations back to raw feature units and labels; wrap model with preprocessor | M2 | Survey / R3 |
| 15 | Causal DAG PC Adjacency Mapping | PC algorithm CPDAG adjacency mapping to feature names, edges, and conditioning sets without index offsets | M2 | Survey / R3 |
| 16 | Decision Tree Support in Model Registries | Register `dt` (DecisionTreeClassifier / DecisionTreeRegressor) in `CLASSIFICATION_MODELS` and `REGRESSION_MODELS` | M2 | Survey / R4 |
| 17 | Authentic Metrics: Brier Score & Calibration | Compute Brier score loss and calibration curves across classification models; expose in `TrainPipelineResponse` | M2 | Survey / R4 |
| 18 | Authentic Benchmark Scenarios | Supply Chain Logistics and Clinical ICU Sepsis generators in `dia/demo_datasets.py` with 1-click UI launch | M2 | Survey / R4 |
| 19 | Memory Bounds & Regularization Guards | Autoencoder capped at 50k rows, batch eval at 512 samples; guard against constant features ($s_x < 1e-9$) | M2 | Survey / R4 |
| 20 | Subtab State Sync & Contextual Empty States | Maintain session state in `sessionStorage` and backend store; provide 1-click launchers on empty states | M2 | Survey / R4 |
| 21 | Edge Model Transpiler JS/WAT Bundles | Transpile trained models to standalone JS and WAT bundles (< 500 KB, < 50ms batch inference) | M2 | Survey / R4 |
| 22 | 10-Stage Interactive User Journey Playwright Test | Live user journey across landing, benchmark demo, workbench, profiler, AutoML, SHAP, What-If, Causal, Export, Edge | M3 | Survey / R5 |
| 23 | Rapid Theme Toggle Stress Test | 10+ rapid Night/Day toggles asserting 0 token desync, 0 misaligned elements, 0 contrast degradation | M3 | Survey / R5 |
| 24 | Pearl's Do-Calculus Boundary Scrubbing Test | Scrub slider at 0%, 8.5%, 15%, negative and extreme values asserting safe metric clamping and no NaNs | M3 | Survey / R5 |
| 25 | Edge-Case Data Ingestion Tests | Ingestion of 1-row, high-cardinality IDs, non-ASCII/emoji column names (`用户_id`, `prénom`, `âge`), missing values | M3 | Survey / R5 |
| 26 | Responsive Sweeps & Zero Overflow Tests | Assert `scrollWidth <= clientWidth` across 320px, 768px, 1024px, 1440px, 2560px viewports | M3 | Survey / R5 |
| 27 | Zero Console & Page Errors Invariant | Assert `console.error == 0` and `pageerror == 0` across all interactive Playwright flows | M3 | Survey / R5 |
| 28 | Test Suite Repair & 100% Pass Rate | Repair `test_light_theme_and_scroll_ux.py` and `e2e_output_verification.py`; 100% pass across 334+ pytest suite | M3 | Survey / R5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Frontend Theme & UI/UX Layout Alignment | Features 1–9 (R1 + R2): 1-click Night/Day toggle, 2 themes, WCAG AA, localStorage, 3-pane flex, sticky headers, landing scroll, cursor | none | IN_PROGRESS |
| M2 | Backend Pipeline, Feature Mapping & Training Integrity | Features 10–21 (R3 + R4): bool columns, target encoding, feature alignment, ensemble importance, recourse inversion, Causal DAG, Decision Trees, Brier/calibration, benchmarks, memory bounds, WASM | none | PLANNED |
| M3 | Comprehensive Interactive E2E Simulation & Verification | Features 22–28 (R5): 10-stage Playwright journey, toggle stress, slider scrubbing, edge ingestion, responsive sweeps, 100% pytest (334+) and Playwright suites | M1, M2 | PLANNED |

## Interface Contracts

### Theme System Interface Contract (`frontend/js/theme.js`)
- `window.initTheme()`: Reads `dia_theme_mode` (or migrates `dia-theme`) from `localStorage`. Defaults to `'night'` (`cyber-aurora`). Applies class `dark` or removes it, sets `data-theme="cyber-aurora"` (night) or `data-theme="tokyo-sand"` (day).
- `window.toggleTheme()`: Debounced toggle between `'night'` and `'day'`. Updates DOM root attributes, updates `#theme-toggle-btn` icon (🌙 / ☀️), and stores `'night'` or `'day'` in `localStorage.setItem('dia_theme_mode', theme)`.
- `window.getCurrentTheme()`: Returns `'night'` or `'day'`.

### Data Preprocessing & Column Alignment Contract (`dia/pipeline_coordinator.py` & `dia/model_trainer.py`)
- `ColumnTransformer`: Must explicitly include `bool` dtype in numeric or categorical transformers to prevent silent column dropping when `remainder="drop"`.
- `train_pipeline(...)`: Returns `TrainPipelineResponse` with `feature_names` strictly matching the columns of the preprocessed matrix $X_{\text{proc}}$.
- `VotingClassifier` / `VotingRegressor`: Computes feature importances as the arithmetic mean of base estimators with `feature_importances_`, normalized to sum to 1.0. Eliminates all dummy mock assignments (`0.05`).
- `recourse_engine.explain(...)`: Operates on candidate vector transformed through the pipeline preprocessor so that model evaluations match feature matrix dimensionality. Perturbations are projected back to raw column units and category labels.

### Benchmark Scenarios Contract (`dia/demo_datasets.py`)
- `DEMO_BENCHMARKS`: Registers keys:
  - `bank_credit`: Credit Risk Assessment (binary classification)
  - `supply_chain`: Supply Chain Delivery Delay (binary/multiclass classification)
  - `clinical_sepsis`: Clinical ICU Sepsis Prediction (binary classification with severe class imbalance)
  - `real_estate`: Real Estate Housing Valuation (continuous regression)
  - `telecom`: Telecom Customer Churn (binary classification)

## Code Layout
- `frontend/`:
  - `landing.html`: Public landing page.
  - `index.html`: Analytical workbench.
  - `css/style.css`: Core design tokens, theme overrides, layout rules.
  - `js/theme.js`: Dedicated theme manager (1-click toggle, localStorage, debounce).
  - `js/app.js`: Workbench application controller.
  - `js/landing.js`: Landing page interactive controller.
  - `js/cursor.js`: Custom cursor physics and touchscreen suppression.
  - `js/views/*.js`: Subtab views (Overview, Data Intelligence, AutoML, Reasoning, Governance, Architecture).
- `dia/`:
  - `data_profiler.py`: Data ingestion, summary, and role inference.
  - `pipeline_coordinator.py`: AutoML pipeline coordinator.
  - `model_trainer.py`: Model training, hyperparameter tuning, metrics, and SHAP.
  - `recourse_engine.py`: Counterfactual recourse in normalized L1/MAD space.
  - `causal_discovery.py`: PC algorithm and Pearl's Do-Calculus.
  - `wasm_compiler.py`: Client-side edge model transpiler (JS/WAT).
  - `demo_datasets.py`: Benchmark dataset generators.
  - `deep_autoencoder.py`: Anomaly detection deep autoencoder.
- `api/`:
  - `server.py`: FastAPI application endpoints.
  - `schemas.py`: Pydantic request and response schemas.
- `tests/`:
  - Pytest unit and integration test modules (`test_data_profiler.py`, `test_model_robustness.py`, `test_innovations_suite.py`, etc.).
  - Playwright E2E test suites (`test_interactive_landing_features.py`, `test_light_theme_and_scroll_ux.py`, `test_fast_dom_verifier.py`, `test_landing_page.py`, `test_accessibility_and_keyboard.py`, `test_network_chaos_and_resilience.py`, `test_edge_cases_and_fuzzing.py`, `test_visual_regression.py`, `test_e2e_comprehensive_journey.py`).
