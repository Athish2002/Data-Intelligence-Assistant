/**
 * frontend/js/app.js
 * ──────────────────
 * Master Fullstack SPA Orchestrator.
 * Connects reactive state, workbench dock, command palette, and workspace views.
 */

import { ApiClient } from './api.js';
import { state, WORKSPACE_SUBTABS, el, escapeHtml } from './state.js';
import {
  showNotification,
  showSuccess,
  showError,
  showLoading,
  hideLoading,
  showTrainingProgress,
  hideTrainingProgress,
} from './toast.js';
import { setupCustomCursor } from './cursor.js';
import { setupWorkbenchFeatures, inspectModel, renderInspectorContent, toggleInspector, closeInspector, toggleDock } from './inspector.js';
import { setupCommandPalette } from './palette.js';
import { initTheme, toggleTheme, applyTheme, getCurrentTheme, onThemeChange } from './theme.js';

// Import Views
import { renderExecutiveDashboard } from './views/dashboard.js';
import {
  renderCoreOverview,
  renderCoreSchema,
  renderCoreReadiness,
  renderCoreInsights,
  renderCoreExecutive,
  renderCoreSymbolic,
} from './views/core.js';
import {
  renderModelRequiredState,
  renderAutoMLLeaderboard,
  renderAutoMLCurves,
  renderAutoMLExplainability,
  renderAutoMLRoiOptimizer,
  renderAutoMLPareto,
  renderAutoMLSimulator,
  renderAutoMLCockpit,
  renderAutoMLStress,
  renderAutoMLConformal,
  renderAutoMLActiveLearning,
} from './views/automl.js';
import {
  renderAdaptiveCausal,
  renderAdaptiveAutoencoder,
  renderAdaptiveBandits,
  renderAdaptiveSynthetic,
  renderAdaptiveOnline,
  renderAdaptiveTimeSeries,
  renderAdaptiveNlp,
  renderAdaptiveGraph,
} from './views/adaptive.js';
import {
  renderGovernanceContracts,
  renderGovernanceGdpr,
  renderGovernanceDrift,
  renderGovernanceSql,
  renderGovernanceCode,
  renderGovernanceCopilot,
  renderGovernanceDossier,
} from './views/governance.js';
import { openSystemMetricsModal, closeSystemMetricsModal } from './views/system.js';

// Expose global system monitor functions
window.openSystemMetrics = openSystemMetricsModal;
window.closeSystemMetrics = closeSystemMetricsModal;

// Expose global functions required by inline HTML onclick handlers
export async function loadDemoFromCard(demoId) {
  if (!demoId) return;
  if (el.demoSelect) el.demoSelect.value = demoId;
  showLoading(`Loading benchmark: ${demoId}...`);
  try {
    const res = await ApiClient.ingestDemo(demoId);
    handleIngestSuccess(res);
    showSuccess(`Loaded benchmark: ${res.detected_domain} (${res.n_rows} rows)`);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
}

export async function launchScreenWithDemo(workspace, subtab, demoName = 'Bank Credit Risk & Default', autoTrain = false) {
  try {
    // 1. Ingest benchmark if not already ingested or no active session
    if (!state.datasetMeta || !state.sessionId) {
      showLoading(`Ingesting benchmark: ${demoName}...`);
      if (el.demoSelect) el.demoSelect.value = demoName;
      const res = await ApiClient.ingestDemo(demoName);
      handleIngestSuccess(res);
    }

    // 2. If autoTrain is requested and no model has been trained yet
    if (autoTrain && !state.pipelineResult) {
      const domainName = state.datasetMeta ? state.datasetMeta.detected_domain : demoName;
      showLoading(`Training AutoML pipeline for ${domainName}...`);
      const goal = (el.goalInput && el.goalInput.value.trim()) || 'Predict default risk';
      const targetCol = (el.targetColInput && el.targetColInput.value.trim()) || null;
      const res = await ApiClient.trainPipeline(state.sessionId, goal, targetCol);
      state.pipelineResult = res;
      handleTrainSuccess(res, false);
    }

    // 3. Navigate directly to target workspace and subtab
    switchWorkspace(workspace, subtab);
    showSuccess(`Launched ${workspace.toUpperCase()} → ${subtab.toUpperCase()} with active scenario!`);
  } catch (err) {
    showError(`Direct launch failed: ${err.message}`);
  } finally {
    hideLoading();
  }
}

export function toggleAurora(forcedState = null) {
  const auroraEl = document.getElementById('ambient-aurora');
  if (!auroraEl) return;
  const isCurrentlyHidden = auroraEl.classList.contains('aurora-hidden');
  const shouldHide = forcedState !== null ? !forcedState : !isCurrentlyHidden;
  auroraEl.classList.toggle('aurora-hidden', shouldHide);
  localStorage.setItem('dia-aurora-enabled', shouldHide ? 'false' : 'true');
  showSuccess(shouldHide ? 'Cyber Aurora ambient background paused' : 'Cyber Aurora ambient background active');
}

window.state = state;
window.inspectModel = inspectModel;
window.applyTheme = applyTheme;
window.toggleAurora = toggleAurora;
window.loadDemoFromCard = loadDemoFromCard;
window.renderCurrentView = renderCurrentView;
window.launchScreenWithDemo = launchScreenWithDemo;

// ─── Theme Manager Integration ──────────────────────────────────────────────
window.applyTheme = applyTheme;
window.toggleTheme = toggleTheme;
window.initTheme = initTheme;
window.getCurrentTheme = getCurrentTheme;

// Register theme change listener to reflow responsive charts & views
onThemeChange(() => {
  window.dispatchEvent(new Event('resize'));
  if (state.datasetMeta) {
    renderCurrentView();
  }
});

// ─── Workbench Data Ingestion & Router ───────────────────────────────────────

function switchWorkspace(wsId, subtabId = null) {
  state.activeWorkspace = wsId;
  if (subtabId) {
    state.activeSubtab = subtabId;
  } else {
    const list = WORKSPACE_SUBTABS[wsId];
    if (list && list.length) state.activeSubtab = list[0].id;
  }
  document.querySelectorAll('.workspace-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.workspace === wsId);
  });
  renderSubtabBar();
  renderCurrentView();
}

window.switchWorkspace = switchWorkspace;

window.quickLoadBenchmark = async function (demoName) {
  if (el.demoSelect) {
    el.demoSelect.value = demoName;
  }
  showLoading(`Ingesting benchmark: ${demoName}...`);
  try {
    const res = await ApiClient.ingestDemo(demoName);
    handleIngestSuccess(res);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
};

async function loadSystemHealth() {
  try {
    const health = await ApiClient.getHealth();
    const badgeText = document.getElementById('system-health-text');
    if (badgeText) {
      badgeText.textContent = `CPU ${health.cpu_cores}C • ${health.gpu_available ? 'GPU ACCELERATED' : 'CPU MODE'} • ONLINE`;
    } else if (el.systemHealthBadge) {
      el.systemHealthBadge.innerHTML = `
        <span class="w-2 h-2 rounded-full bg-[#00e575] animate-pulse mr-1.5"></span>
        <span id="system-health-text" class="font-mono text-[10px] font-semibold text-[#4ade80]">CPU ${health.cpu_cores}C &bull; ${health.gpu_available ? 'GPU ACCELERATED' : 'CPU MODE'} &bull; ONLINE</span>
      `;
    }
    const dockPlatform = document.getElementById('dock-platform-status');
    if (dockPlatform) {
      dockPlatform.textContent = `${health.cpu_cores}C • ${health.gpu_available ? 'GPU' : 'CPU'}`;
    }
    const dockStorage = document.getElementById('dock-storage-status');
    if (dockStorage) {
      dockStorage.textContent = health.storage_type ? `${health.storage_type.toUpperCase()} • OK` : 'NVMe • OK';
    }
  } catch (err) {
    const badgeText = document.getElementById('system-health-text');
    if (badgeText) {
      badgeText.textContent = 'API Offline';
    } else if (el.systemHealthBadge) {
      el.systemHealthBadge.innerHTML = `<span id="system-health-text" class="text-[10px] text-rose-400">API Offline</span>`;
    }
  }
}

async function loadDemoList() {
  try {
    const demos = await ApiClient.getDemos();
    el.demoSelect.innerHTML = demos
      .map(
        (d) =>
          `<option value="${d.id}">📊 ${d.name} (${d.domain})</option>`
      )
      .join('');
  } catch (err) {
    console.error('Failed to load demos:', err);
  }
}

// ─── Event Handlers ──────────────────────────────────────────────────────────

function setupEventListeners() {
  // Load Demo Benchmark
  el.loadDemoBtn.addEventListener('click', async () => {
    const demoName = el.demoSelect.value;
    showLoading('Loading benchmark dataset...');
    try {
      const res = await ApiClient.ingestDemo(demoName);
      handleIngestSuccess(res);
    } catch (err) {
      showError(err.message);
    } finally {
      hideLoading();
    }
  });

  // CSV File Upload
  el.csvFileInput.addEventListener('change', async (e) => {
    if (!e.target.files.length) return;
    const file = e.target.files[0];
    showLoading(`Ingesting & sanitizing ${file.name}...`);
    try {
      const res = await ApiClient.uploadCSV(file);
      handleIngestSuccess(res);
    } catch (err) {
      showError(err.message);
    } finally {
      hideLoading();
    }
  });

  // Drag & Drop
  el.uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.uploadDropzone.classList.add('border-cyan-400');
  });
  el.uploadDropzone.addEventListener('dragleave', () => {
    el.uploadDropzone.classList.remove('border-cyan-400');
  });
  el.uploadDropzone.addEventListener('drop', async (e) => {
    e.preventDefault();
    el.uploadDropzone.classList.remove('border-cyan-400');
    if (e.dataTransfer.files.length) {
      const file = e.dataTransfer.files[0];
      showLoading(`Ingesting ${file.name}...`);
      try {
        const res = await ApiClient.uploadCSV(file);
        handleIngestSuccess(res);
      } catch (err) {
        showError(err.message);
      } finally {
        hideLoading();
      }
    }
  });

  // Auto-Detect Objectives
  el.autodetectBtn.addEventListener('click', async () => {
    if (!state.sessionId) {
      showError('Please load or upload a dataset first.');
      return;
    }
    showLoading('Analyzing schema & distributions locally...');
    try {
      const res = await ApiClient.autoDetectObjectives(state.sessionId);
      if (res.objectives && res.objectives.length > 0) {
        el.goalInput.value = res.objectives[0];
        showSuccess(`Auto-detected domain: ${res.domain}`);
      }
    } catch (err) {
      showError(err.message);
    } finally {
      hideLoading();
    }
  });

  // Train AutoML Pipeline
  el.trainBtn.addEventListener('click', async () => {
    if (!state.sessionId) {
      showError('Please connect a data source first.');
      return;
    }
    const goal = el.goalInput.value.trim();
    if (!goal) {
      showError('Please describe your prediction goal in plain English.');
      return;
    }
    const targetCol = el.targetColInput.value.trim() || null;

    showTrainingProgress();
    try {
      const res = await ApiClient.trainPipeline(state.sessionId, goal, targetCol);
      state.pipelineResult = res;
      handleTrainSuccess(res);
    } catch (err) {
      showError(err.message);
    } finally {
      hideTrainingProgress();
    }
  });

  // Workspace Switching
  document.querySelectorAll('.workspace-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.workspace-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeWorkspace = btn.dataset.workspace;
      state.activeSubtab = WORKSPACE_SUBTABS[state.activeWorkspace][0].id;
      renderSubtabBar();
      renderCurrentView();

      // Auto-dismiss left dock on mobile link click if inside dock
      if (window.innerWidth < 1024 && btn.closest('#left-dock')) {
        toggleDock(false);
      }
    });
  });

  // HUD and Global Keyboard Shortcuts (⌘B / Ctrl+B for Dock, ⌘I / Ctrl+I for Inspector)
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      toggleDock();
    } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'i') {
      e.preventDefault();
      toggleInspector();
    }
  });

  document.getElementById('hud-dock-shortcut')?.addEventListener('click', () => toggleDock());
  document.getElementById('hud-inspector-shortcut')?.addEventListener('click', () => toggleInspector());
}

function handleIngestSuccess(res) {
  state.sessionId = res.session_id;
  state.datasetMeta = res;
  state.pipelineResult = null;

  // Add conditional adaptive tabs based on capabilities
  WORKSPACE_SUBTABS.adaptive = [
    { id: 'causal', label: 'Causal & Counterfactuals', icon: 'target' },
    { id: 'autoencoder', label: 'Deep Autoencoder', icon: 'cpu' },
    { id: 'bandits', label: 'Contextual Bandits', icon: 'play-circle' },
    { id: 'synthetic', label: 'Synthetic Data (DP)', icon: 'dna' },
    { id: 'online', label: 'Online Streaming Fit', icon: 'zap' },
  ];
  if (res.capabilities.has_dates) {
    WORKSPACE_SUBTABS.adaptive.push({ id: 'timeseries', label: 'Time-Series Forecast', icon: 'trending-up' });
  }
  if (res.capabilities.has_text) {
    WORKSPACE_SUBTABS.adaptive.push({ id: 'nlp', label: 'NLP & Lexical Analysis', icon: 'file-text' });
  }
  if (res.capabilities.has_entities) {
    WORKSPACE_SUBTABS.adaptive.push({ id: 'graph', label: 'Graph Intelligence', icon: 'share-2' });
  }

  if (el.activeDatasetBadge) {
    el.activeDatasetBadge.innerHTML = `
      <span class="text-xs px-2.5 py-1 rounded-md font-mono font-medium border flex items-center space-x-1.5 whitespace-nowrap truncate" style="background-color: var(--tag-bg); color: var(--tag-text); border-color: var(--card-border);" title="${res.detected_domain} (${res.n_rows.toLocaleString()} rows, ${res.n_cols} cols)">
        <span>📊</span>
        <span class="truncate">${res.detected_domain || 'Active Dataset'}</span>
      </span>
    `;
  }
  if (el.activeModelBadge) {
    el.activeModelBadge.innerHTML = '';
  }

  if (el.hudDataset) {
    el.hudDataset.innerHTML = `DATASET: <strong style="color: var(--text-primary);">${res.detected_domain || 'Active'}</strong> (${res.n_rows} rows)`;
  }

  if (res.goal && el.goalInput) {
    el.goalInput.value = res.goal;
  } else if (res.suggested_objectives && res.suggested_objectives.length > 0 && el.goalInput) {
    el.goalInput.value = res.suggested_objectives[0];
  }

  if (res.suggested_target && el.targetColInput) {
    el.targetColInput.value = res.suggested_target;
  }

  el.trainBtn.disabled = false;
  renderSubtabBar();
  renderCurrentView();
}

function handleTrainSuccess(res, autoSwitch = true) {
  if (el.activeModelBadge) {
    el.activeModelBadge.innerHTML = `
      <span class="text-xs px-2.5 py-1 rounded-md font-mono font-semibold border flex items-center space-x-1.5 whitespace-nowrap truncate" style="background-color: #092314; color: #4ade80; border-color: #134e2c;" title="Champion Model: ${res.best_model_label}">
        <span>🏆</span>
        <span class="truncate">${res.best_model_label}</span>
      </span>
    `;
  }

  if (el.hudModel) {
    el.hudModel.innerHTML = `CHAMPION: <strong style="color: #00e575;">${res.best_model_label}</strong>`;
  }

  // Update & prime Inspector Drawer
  renderInspectorContent();
  if (window.innerWidth >= 1280) {
    toggleInspector(true);
  }

  if (autoSwitch) {
    state.activeWorkspace = 'automl';
    state.activeSubtab = 'leaderboard';
    document.querySelectorAll('.workspace-btn').forEach((b) => b.classList.remove('active'));
    document.querySelector('.workspace-btn[data-workspace="automl"]')?.classList.add('active');
    renderSubtabBar();
    renderCurrentView();
  }
}

// ─── Sub-Tab Rendering ───────────────────────────────────────────────────────

function renderSubtabBar() {
  const subtabs = WORKSPACE_SUBTABS[state.activeWorkspace] || [];
  el.subtabBar.innerHTML = subtabs
    .map(
      (tab) => `
      <button class="subtab-btn flex items-center space-x-1.5 ${state.activeSubtab === tab.id ? 'active' : ''}" data-subtab="${tab.id}">
        <i data-lucide="${tab.icon}" class="w-3.5 h-3.5"></i>
        <span>${tab.label}</span>
      </button>
    `
    )
    .join('');

  el.subtabBar.querySelectorAll('.subtab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      el.subtabBar.querySelectorAll('.subtab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeSubtab = btn.dataset.subtab;
      renderCurrentView();
    });
  });

  lucide.createIcons();
}

// ─── Screen Catalog for Context-Aware Empty States ───────────────────────────

const SCREEN_CATALOG = {
  // Governance
  'governance:dossier': {
    title: 'Executive Regulatory Audit Dossier',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Comprehensive regulatory audit compiling model cards, GDPR Article 9/30 ROPA, Great Expectations data contracts, target leakage detection, and conformal uncertainty sets into an exportable HTML dossier.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'file-check',
  },
  'governance:contracts': {
    title: 'Data Quality Contracts (Great Expectations)',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Automated finite-bounds schema verification, null limits, and active dataset constraint evaluation suite.',
    recommendedDemo: 'Hospital Patient Readmission',
    requiresModel: false,
    icon: 'check-square',
  },
  'governance:gdpr': {
    title: 'GDPR & Privacy Compliance Audit',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Word-boundary token scans for PII entities, Article 9 special categories, and automated masking.',
    recommendedDemo: 'Hospital Patient Readmission',
    requiresModel: false,
    icon: 'lock',
  },
  'governance:drift': {
    title: 'Sentinel Drift Monitor & Multivariate MMD',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Two-sample Kolmogorov-Smirnov distribution checks, PSI calculations, and multivariate RBF-kernel MMD triage.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'git-commit',
  },
  'governance:code': {
    title: 'Production Code & Edge C99 / WASM Transpiler',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Transpiles trained models to standalone JS/WASM scoring engines (< 500 KB) with microsecond client evaluation.',
    recommendedDemo: 'Telecom Customer Churn',
    requiresModel: true,
    icon: 'code',
  },
  'governance:sql': {
    title: 'In-Database SQL Transpiler',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Compiles trained ML trees into pure ANSI-SQL queries that execute inside PostgreSQL, Snowflake, or BigQuery.',
    recommendedDemo: 'Housing Price Valuation',
    requiresModel: true,
    icon: 'database',
  },
  'governance:copilot': {
    title: 'AI Analytical Chat Copilot',
    badge: 'GOVERNANCE & MLOPS',
    description: 'Context-grounded assistant for interactive dataset queries, SQL synthesis, and diagnostic guidance.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'message-square',
  },

  // Adaptive AI
  'adaptive:causal': {
    title: 'Interactive Causal DAG Studio & Do-Calculus',
    badge: 'ADAPTIVE AI',
    description: 'Constraint-based PC algorithm uncovering causal graphs with clickable domain prior edge toggles and Pearl\'s Do-Calculus policy simulation.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'target',
  },
  'adaptive:autoencoder': {
    title: 'Deep PyTorch Autoencoder & Anomaly Detection',
    badge: 'ADAPTIVE AI',
    description: 'Reconstruction error profiling identifying high-dimensional outliers and data corruptions.',
    recommendedDemo: 'Housing Price Valuation',
    requiresModel: false,
    icon: 'cpu',
  },
  'adaptive:bandits': {
    title: 'Contextual Multi-Armed Bandits',
    badge: 'ADAPTIVE AI',
    description: 'Upper Confidence Bound (UCB) policy optimization for real-time exploratory decision making.',
    recommendedDemo: 'Telecom Customer Churn',
    requiresModel: false,
    icon: 'play-circle',
  },
  'adaptive:synthetic': {
    title: 'Differentially Private Synthetic Data Generator',
    badge: 'ADAPTIVE AI',
    description: 'Generates privacy-preserving tabular twins with calibrated Laplacian noise and statistical fidelity evaluation.',
    recommendedDemo: 'Hospital Patient Readmission',
    requiresModel: false,
    icon: 'dna',
  },
  'adaptive:online': {
    title: 'Online Streaming Model Fit (SGD / River)',
    badge: 'ADAPTIVE AI',
    description: 'Continuous incremental model adaptation for high-velocity streaming data pipelines.',
    recommendedDemo: 'Telecom Customer Churn',
    requiresModel: false,
    icon: 'zap',
  },

  // Autonomous ML
  'automl:cockpit': {
    title: 'Champion vs. Challenger Split Cockpit',
    badge: 'AUTONOMOUS ML',
    description: 'Side-by-side production model comparison with delta metrics, traffic allocation slider, and 1-click model promotion.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: true,
    icon: 'columns',
  },
  'automl:simulator': {
    title: 'What-If & Wachter Recourse Playground',
    badge: 'AUTONOMOUS ML',
    description: 'Interactive counterfactual recourse optimizer computing minimum actionable feature modifications with MAD scaling and immutable masks.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: true,
    icon: 'sliders',
  },
  'automl:stress': {
    title: 'Macroeconomic Stress Testing & Copula Shocks',
    badge: 'AUTONOMOUS ML',
    description: 'Cholesky Gaussian copula decomposition with macroeconomic shock presets computing VaR95/99 and CVaR95.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'alert-triangle',
  },
  'automl:conformal': {
    title: 'Split Conformal Prediction & Epistemic Sets',
    badge: 'AUTONOMOUS ML',
    description: 'Finite-sample coverage guarantees separating aleatoric Shannon entropy from epistemic Mahalanobis OOD distance.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'check-circle',
  },
  'automl:pareto': {
    title: 'Multi-Objective Pareto Flight Simulator',
    badge: 'AUTONOMOUS ML',
    description: 'Resolves business trade-offs across Expected Net Profit, Default Risk, and Demographic Parity Fairness.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: true,
    icon: 'crosshair',
  },
  'automl:leaderboard': {
    title: 'AutoML Model Leaderboard & Evaluation',
    badge: 'AUTONOMOUS ML',
    description: 'Leaderboard ranking 6 model architectures with 5-Fold Stratified Cross Validation and metric breakdowns.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: true,
    icon: 'trophy',
  },

  // Core Intelligence
  'core:symbolic': {
    title: 'Symbolic Feature Discovery & SQL Invariants',
    badge: 'DATA INTELLIGENCE',
    description: 'Discovers non-linear algebraic invariants across features, ranked by Mutual Information lift, with ANSI-SQL export.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'binary',
  },
  'core:readiness': {
    title: 'Data Readiness & Target Leakage Sleuth',
    badge: 'DATA INTELLIGENCE',
    description: 'Mutual information ratios, quasi-target correlations, row ID memorization, and automated feature quarantine.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'shield-check',
  },
  'core:overview': {
    title: 'Dataset Overview & Automated Sanitization',
    badge: 'DATA INTELLIGENCE',
    description: 'Delimiter detection, malformed row repairs, type inference, and distribution profiling.',
    recommendedDemo: 'Telecom Customer Churn',
    requiresModel: false,
    icon: 'table',
  },
  'dashboard:system_arch': {
    title: 'Full-Stack System Architecture Blueprint',
    badge: 'SYSTEM ARCHITECTURE',
    description: 'Interactive C4 Container topologies, asynchronous dataflow lifecycles, Pro Max UI/UX design token matrices, and OpenAPI contract SLA guarantees.',
    recommendedDemo: 'Bank Credit Risk & Default',
    requiresModel: false,
    icon: 'layers',
  },
};

// ─── Section Outline & Scroll-Spy Manager (Never Empty Sidebar) ──────────────

export function updateDockSectionOutline() {
  const outlineList = document.getElementById('dock-outline-list');
  const outlineCount = document.getElementById('dock-outline-count');
  if (!outlineList) return;

  const { activeWorkspace: w, activeSubtab: s } = state;
  let items = [];

  // Inspect tab-content for elements marked with data-section-anchor or IDs starting with section-
  const sectionEls = document.querySelectorAll('#tab-content [data-section-anchor], #tab-content [id^="section-"]');
  if (sectionEls && sectionEls.length > 0) {
    sectionEls.forEach((section, idx) => {
      const id = section.id || `section-anchor-${idx}`;
      if (!section.id) section.id = id;
      const label = section.getAttribute('data-section-anchor') || section.querySelector('h2, h3, h4')?.textContent?.trim() || `Section ${idx + 1}`;
      items.push({ id, label });
    });
  } else {
    // Curated contextual anchors based on workspace
    const catalog = {
      dashboard: [
        { id: 'section-mission-overview', label: 'Mission Overview' },
        { id: 'section-executive-kpis', label: 'Executive KPIs' },
        { id: 'section-quick-pillars', label: 'Core Pillars' },
        { id: 'section-benchmarks', label: 'Benchmark Datasets' },
      ],
      core: [
        { id: 'sticky-workspace-header', label: 'Data Ingestion & Profiling' },
        { id: 'tab-content', label: 'Feature Distributions & Sanitize' },
        { id: 'tab-content', label: 'Target Leakage & Quarantine' },
      ],
      automl: [
        { id: 'sticky-workspace-header', label: 'AutoML Leaderboard' },
        { id: 'tab-content', label: 'Split Cockpit Evaluation' },
        { id: 'tab-content', label: 'TreeSHAP Attribution' },
      ],
      adaptive: [
        { id: 'sticky-workspace-header', label: 'Causal DAG Induction' },
        { id: 'tab-content', label: "Pearl's Do-Calculus Simulator" },
        { id: 'tab-content', label: 'Pareto Multi-Objective Frontier' },
      ],
      governance: [
        { id: 'sticky-workspace-header', label: 'Executive Regulatory Dossier' },
        { id: 'tab-content', label: 'Sentinel Drift Monitor' },
        { id: 'tab-content', label: 'Edge C99 & WASM Transpiler' },
      ],
    };
    items = catalog[w] || [
      { id: 'sticky-workspace-header', label: 'Workspace Controls' },
      { id: 'tab-content', label: 'Studio View' },
    ];
  }

  if (outlineCount) {
    outlineCount.textContent = `${items.length} ${items.length === 1 ? 'Anchor' : 'Anchors'}`;
  }

  outlineList.innerHTML = items.map((it, idx) => `
    <button class="dock-outline-btn w-full text-left px-2.5 py-1.5 rounded flex items-center justify-between transition-all cursor-pointer group ${idx === 0 ? 'active' : ''}" data-target-id="${escapeHtml(it.id)}" title="Scroll to ${escapeHtml(it.label)}">
      <span class="flex items-center space-x-2 truncate">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500/40 group-hover:bg-[#00e575] transition-colors outline-dot flex-shrink-0"></span>
        <span class="truncate outline-label">${escapeHtml(it.label)}</span>
      </span>
      <i data-lucide="chevron-right" class="w-3 h-3 text-stone-500 group-hover:translate-x-0.5 transition-transform flex-shrink-0"></i>
    </button>
  `).join('');

  // Attach smooth scrolling click handlers
  outlineList.querySelectorAll('.dock-outline-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = btn.getAttribute('data-target-id');
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        const canvas = document.getElementById('center-canvas');
        if (canvas) {
          const stickyHeader = document.getElementById('sticky-workspace-header');
          const headerHeight = stickyHeader ? stickyHeader.offsetHeight : 0;
          const targetRect = targetEl.getBoundingClientRect();
          const canvasRect = canvas.getBoundingClientRect();
          const scrollOffset = targetRect.top - canvasRect.top + canvas.scrollTop - headerHeight - 12;
          canvas.scrollTo({ top: Math.max(0, scrollOffset), behavior: 'smooth' });
        } else {
          targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        outlineList.querySelectorAll('.dock-outline-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        // Auto-dismiss left dock on mobile link click
        if (window.innerWidth < 1024) {
          toggleDock(false);
        }
      }
    });
  });

  if (window.lucide) lucide.createIcons();
}

function setupCanvasScrollSpy() {
  const canvas = document.getElementById('center-canvas');
  if (!canvas) return;

  let ticking = false;
  canvas.addEventListener('scroll', () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        const anchors = document.querySelectorAll('#tab-content [data-section-anchor], #tab-content [id^="section-"]');
        if (!anchors || anchors.length === 0) {
          ticking = false;
          return;
        }
        const canvasRect = canvas.getBoundingClientRect();
        let currentId = null;

        anchors.forEach((el) => {
          const rect = el.getBoundingClientRect();
          if (rect.top - canvasRect.top <= 140) {
            currentId = el.id;
          }
        });

        if (currentId) {
          document.querySelectorAll('.dock-outline-btn').forEach((btn) => {
            const matches = btn.getAttribute('data-target-id') === currentId;
            btn.classList.toggle('active', matches);
          });
        }
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
}

window.updateDockSectionOutline = updateDockSectionOutline;

// ─── Main View Dispatcher ────────────────────────────────────────────────────

function renderCurrentView() {
  const { activeWorkspace: w, activeSubtab: s } = state;

  // Apply smooth GPU hardware-accelerated view cross-fade
  if (el.tabContent) {
    el.tabContent.classList.remove('view-transition');
    void el.tabContent.offsetWidth; // Force DOM reflow to replay CSS keyframes
    el.tabContent.classList.add('view-transition');
  }

  // Workspace 0: Executive Dashboard (Always accessible)
  if (w === 'dashboard') {
    renderExecutiveDashboard();
    updateDockSectionOutline();
    lucide.createIcons();
    return;
  }

  // If no dataset loaded yet and user visits other workspace, show context-aware launcher
  if (!state.datasetMeta) {
    const screenKey = `${w}:${s}`;
    const screenInfo = SCREEN_CATALOG[screenKey] || {
      title: `${w.toUpperCase()} Analytical Studio`,
      badge: `${w.toUpperCase()} WORKSPACE`,
      description: `This workspace requires an active dataset connection to perform automated machine learning, causal inference, and regulatory auditing.`,
      recommendedDemo: 'Bank Credit Risk & Default',
      requiresModel: false,
      icon: 'database',
    };

    el.tabContent.innerHTML = `
      <div class="glass-card p-6 sm:p-8 text-center max-w-2xl mx-auto my-8 space-y-5 border-t-4 border-t-[#00e575]">
        <div class="w-14 h-14 rounded-2xl mx-auto flex items-center justify-center border shadow-lg" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
          <i data-lucide="${screenInfo.icon}" class="w-7 h-7 text-[#00e575]"></i>
        </div>

        <div class="space-y-1.5">
          <div class="flex items-center justify-center space-x-2">
            <span class="text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
              ${escapeHtml(screenInfo.badge)} &bull; SUBTAB: ${escapeHtml(s.toUpperCase())}
            </span>
          </div>
          <h3 class="text-xl sm:text-2xl font-black font-mono tracking-tight" style="color: var(--text-primary);">
            ${escapeHtml(screenInfo.title)}
          </h3>
          <p class="text-xs sm:text-sm leading-relaxed max-w-lg mx-auto" style="color: var(--text-secondary);">
            ${escapeHtml(screenInfo.description)}
          </p>
        </div>

        <!-- 1-Click Launch Button for This Specific Screen -->
        <div class="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
          <button onclick="window.launchScreenWithDemo('${w}', '${s}', '${screenInfo.recommendedDemo}', ${screenInfo.requiresModel})" class="w-full sm:w-auto px-6 py-3 rounded-lg font-bold text-xs uppercase tracking-wider cursor-pointer shadow-lg active:scale-95 flex items-center justify-center space-x-2 bg-[#00e575] hover:bg-[#00ff82] text-black">
            <i data-lucide="play" class="w-4 h-4 fill-current"></i>
            <span>Launch ${escapeHtml(screenInfo.title)} with Demo (1-Click)</span>
          </button>
          <a href="/" class="w-full sm:w-auto px-4 py-3 rounded-lg font-bold text-xs border hover:bg-white/5 cursor-pointer flex items-center justify-center space-x-2 no-underline" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-primary);" title="Return to Landing Page">
            <i data-lucide="home" class="w-4 h-4 text-[#00e575]"></i>
            <span>Return to Landing Page</span>
          </a>
        </div>

        <!-- Alternative Scenario Quick-Pick -->
        <div class="pt-3 border-t border-white/10 space-y-2">
          <div class="text-[11px] font-mono uppercase tracking-wider" style="color: var(--text-muted);">
            Or Launch This Screen with an Industry Domain:
          </div>
          <div class="flex flex-wrap items-center justify-center gap-2">
            <button onclick="window.launchScreenWithDemo('${w}', '${s}', 'Bank Credit Risk & Default', ${screenInfo.requiresModel})" class="px-3 py-1.5 rounded border text-xs font-mono hover:border-[#00e575] hover:text-[#00e575] transition-colors cursor-pointer" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-secondary);">
              💳 Banking Default
            </button>
            <button onclick="window.launchScreenWithDemo('${w}', '${s}', 'Telecom Customer Churn', ${screenInfo.requiresModel})" class="px-3 py-1.5 rounded border text-xs font-mono hover:border-[#00e575] hover:text-[#00e575] transition-colors cursor-pointer" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-secondary);">
              📡 Telecom Churn
            </button>
            <button onclick="window.launchScreenWithDemo('${w}', '${s}', 'Hospital Patient Readmission', ${screenInfo.requiresModel})" class="px-3 py-1.5 rounded border text-xs font-mono hover:border-[#00e575] hover:text-[#00e575] transition-colors cursor-pointer" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-secondary);">
              🏥 Hospital Patient
            </button>
            <button onclick="window.launchScreenWithDemo('${w}', '${s}', 'Housing Price Valuation', ${screenInfo.requiresModel})" class="px-3 py-1.5 rounded border text-xs font-mono hover:border-[#00e575] hover:text-[#00e575] transition-colors cursor-pointer" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-secondary);">
              🏡 Housing Price
            </button>
          </div>
        </div>

      </div>
    `;
    lucide.createIcons();
    return;
  }

  // Workspace 1: Core Intelligence
  if (w === 'core') {
    if (s === 'overview') renderCoreOverview();
    else if (s === 'schema') renderCoreSchema();
    else if (s === 'readiness') renderCoreReadiness();
    else if (s === 'insights') renderCoreInsights();
    else if (s === 'symbolic') renderCoreSymbolic();
    else if (s === 'executive') renderCoreExecutive();
  }
  // Workspace 2: AutoML
  else if (w === 'automl') {
    if (s === 'leaderboard') renderAutoMLLeaderboard();
    else if (s === 'cockpit') renderAutoMLCockpit();
    else if (s === 'curves') renderAutoMLCurves();
    else if (s === 'explainability') renderAutoMLExplainability();
    else if (s === 'roi') renderAutoMLRoiOptimizer();
    else if (s === 'pareto') renderAutoMLPareto();
    else if (s === 'simulator') renderAutoMLSimulator();
    else if (s === 'stress') renderAutoMLStress();
    else if (s === 'conformal') renderAutoMLConformal();
    else if (s === 'active_learning') renderAutoMLActiveLearning();
  }
  // Workspace 3: Adaptive AI Engines
  else if (w === 'adaptive') {
    if (s === 'causal') renderAdaptiveCausal();
    else if (s === 'autoencoder') renderAdaptiveAutoencoder();
    else if (s === 'bandits') renderAdaptiveBandits();
    else if (s === 'synthetic') renderAdaptiveSynthetic();
    else if (s === 'online') renderAdaptiveOnline();
    else if (s === 'timeseries') renderAdaptiveTimeSeries();
    else if (s === 'nlp') renderAdaptiveNlp();
    else if (s === 'graph') renderAdaptiveGraph();
  }
  // Workspace 4: Governance & Production
  else if (w === 'governance') {
    if (s === 'contracts') renderGovernanceContracts();
    else if (s === 'gdpr') renderGovernanceGdpr();
    else if (s === 'drift') renderGovernanceDrift();
    else if (s === 'dossier') renderGovernanceDossier();
    else if (s === 'sql') renderGovernanceSql();
    else if (s === 'code') renderGovernanceCode();
    else if (s === 'copilot') renderGovernanceCopilot();
  }

  updateDockSectionOutline();
  lucide.createIcons();
}

// ─── Executive Mission Control Dashboard ─────────────────────────────────────



async function startSystemHealthBadgePolling() {
  const badgeText = document.getElementById('system-health-text');
  if (!badgeText) return;

  const updateBadge = async () => {
    try {
      const res = await fetch('/api/v1/system/metrics');
      if (res.ok) {
        const d = await res.json();
        badgeText.textContent = `RAM: ${d.process_memory_rss_mb} MB | CPU: ${d.cpu_percent}%`;
      }
    } catch (e) {}
  };

  await updateBadge();
  setInterval(updateBadge, 5000);
}

// ─── Initialization ──────────────────────────────────────────────────────────

async function init() {
  initTheme();
  setupCustomCursor();
  setupWorkbenchFeatures();
  setupCommandPalette();
  setupCanvasScrollSpy();
  updateDockSectionOutline();
  lucide.createIcons();
  setupEventListeners();
  await loadSystemHealth();
  await loadDemoList();

  // Check for pending session passed from standalone landing page
  const pendingSessionStr = sessionStorage.getItem('dia_pending_session');
  const shouldAutoTrain = sessionStorage.getItem('dia_pending_autotrain') === 'true';

  if (pendingSessionStr) {
    sessionStorage.removeItem('dia_pending_session');
    sessionStorage.removeItem('dia_pending_autotrain');
    try {
      const parsedRes = JSON.parse(pendingSessionStr);
      if (parsedRes && parsedRes.session_id) {
        handleIngestSuccess(parsedRes);
        if (shouldAutoTrain) {
          launchScreenWithDemo('dashboard', 'mission_control', parsedRes.detected_domain || 'Bank Credit Risk & Default', true);
        } else {
          switchWorkspace('core', 'overview');
          showSuccess(`Dataset loaded: ${parsedRes.detected_domain || 'Custom Dataset'} (${parsedRes.n_rows.toLocaleString()} rows)`);
        }
      }
    } catch (err) {
      console.error('Failed to parse pending session from landing page:', err);
      renderSubtabBar();
      renderCurrentView();
    }
  } else {
    // Check URL parameters: ?demo=... or ?session_id=...
    const urlParams = new URLSearchParams(window.location.search);
    const demoParam = urlParams.get('demo');
    const sessionIdParam = urlParams.get('session_id');

    if (demoParam) {
      window.history.replaceState({}, document.title, window.location.pathname);
      launchScreenWithDemo('dashboard', 'mission_control', demoParam, true);
    } else if (sessionIdParam) {
      window.history.replaceState({}, document.title, window.location.pathname);
      showLoading('Loading active session...');
      ApiClient.getProfile(sessionIdParam)
        .then((profile) => {
          state.sessionId = sessionIdParam;
          state.datasetMeta = profile;
          switchWorkspace('core', 'overview');
          showSuccess(`Session active: ${sessionIdParam.slice(0, 8)}...`);
        })
        .catch((err) => {
          showError(`Failed to load session: ${err.message}`);
          renderSubtabBar();
          renderCurrentView();
        })
        .finally(() => hideLoading());
    } else {
      renderSubtabBar();
      renderCurrentView();
    }
  }

  startSystemHealthBadgePolling();
}

window.addEventListener('DOMContentLoaded', init);
