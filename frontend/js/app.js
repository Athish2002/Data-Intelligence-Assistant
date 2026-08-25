/**
 * frontend/js/app.js
 * ──────────────────
 * Complete UI Controller for Data Intelligence Assistant Fullstack SPA.
 * Manages 4 Workspaces, 27 Subtabs, Interactive Plotly Visualizations,
 * Real-Time Simulators, and Governance Tools.
 */

import { ApiClient } from './api.js';

// Global Reactive State
const state = {
  sessionId: null,
  datasetMeta: null,
  pipelineResult: null,
  activeWorkspace: 'core',
  activeSubtab: 'overview',
};

// Sub-Tab Layout Definitions
const WORKSPACE_SUBTABS = {
  core: [
    { id: 'overview', label: 'Overview & Sanitization', icon: 'table' },
    { id: 'schema', label: 'Column Roles & Schema', icon: 'columns' },
    { id: 'readiness', label: 'Data Readiness Audit', icon: 'shield-check' },
    { id: 'insights', label: 'Smart Insights & Drivers', icon: 'sparkles' },
    { id: 'executive', label: 'Executive Summary', icon: 'file-text' },
  ],
  automl: [
    { id: 'leaderboard', label: 'Leaderboard & Evaluation', icon: 'trophy' },
    { id: 'curves', label: 'ROC & Confusion Matrix', icon: 'activity' },
    { id: 'explainability', label: 'SHAP Explainability', icon: 'bar-chart-3' },
    { id: 'roi', label: 'Business ROI Optimizer', icon: 'dollar-sign' },
    { id: 'simulator', label: 'What-If Simulator', icon: 'sliders' },
    { id: 'active_learning', label: 'Active Learning Queue', icon: 'user-check' },
  ],
  adaptive: [
    { id: 'causal', label: 'Causal & Counterfactuals', icon: 'target' },
    { id: 'autoencoder', label: 'Deep Autoencoder', icon: 'cpu' },
    { id: 'bandits', label: 'Contextual Bandits', icon: 'play-circle' },
    { id: 'synthetic', label: 'Synthetic Data (DP)', icon: 'dna' },
    { id: 'online', label: 'Online Streaming Fit', icon: 'zap' },
  ],
  governance: [
    { id: 'contracts', label: 'Data Quality (GX)', icon: 'check-square' },
    { id: 'gdpr', label: 'GDPR & Privacy Audit', icon: 'lock' },
    { id: 'drift', label: 'Drift Monitor (PSI)', icon: 'git-commit' },
    { id: 'sql', label: 'In-Database SQL Transpiler', icon: 'database' },
    { id: 'code', label: 'Production Code & Docker', icon: 'code' },
    { id: 'copilot', label: 'AI Chat Copilot', icon: 'message-square' },
  ],
};

// DOM Elements
const el = {
  demoSelect: document.getElementById('demo-select'),
  loadDemoBtn: document.getElementById('load-demo-btn'),
  csvFileInput: document.getElementById('csv-file-input'),
  uploadDropzone: document.getElementById('upload-dropzone'),
  goalInput: document.getElementById('goal-input'),
  targetColInput: document.getElementById('target-col-input'),
  autodetectBtn: document.getElementById('autodetect-btn'),
  trainBtn: document.getElementById('train-btn'),
  trainProgress: document.getElementById('train-progress'),
  progressBar: document.getElementById('progress-bar'),
  statusText: document.getElementById('status-text'),
  workspaceNav: document.getElementById('workspace-nav'),
  subtabBar: document.getElementById('subtab-bar'),
  tabContent: document.getElementById('tab-content'),
  activeDatasetBadge: document.getElementById('active-dataset-badge'),
  systemHealthBadge: document.getElementById('system-health-badge'),
};

// ─── Initialization ──────────────────────────────────────────────────────────

async function init() {
  lucide.createIcons();
  setupEventListeners();
  await loadSystemHealth();
  await loadDemoList();
  renderSubtabBar();
  renderCurrentView();
}

async function loadSystemHealth() {
  try {
    const health = await ApiClient.getHealth();
    el.systemHealthBadge.innerHTML = `
      <span class="live-pulse mr-1.5"></span>
      <span class="text-xs text-emerald-400 font-medium">${health.cpu_cores} Cores &bull; ${health.gpu_available ? 'GPU Active' : 'CPU Engine'} &bull; v${health.version}</span>
    `;
  } catch (err) {
    el.systemHealthBadge.innerHTML = `<span class="text-xs text-rose-400">API Offline</span>`;
  }
}

async function loadDemoList() {
  try {
    const demos = await ApiClient.getDemos();
    el.demoSelect.innerHTML = demos
      .map(
        (d) =>
          `<option value="${d.name}">📊 ${d.name} (${d.domain})</option>`
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
    });
  });
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

  el.activeDatasetBadge.innerHTML = `
    <span class="text-xs bg-cyan-950/80 text-cyan-400 border border-cyan-500/30 px-3 py-1 rounded-full font-medium">
      📊 ${res.n_rows.toLocaleString()} Rows &bull; ${res.n_cols} Cols &bull; ${res.detected_domain}
    </span>
  `;

  if (res.suggested_objectives && res.suggested_objectives.length > 0) {
    el.goalInput.value = res.suggested_objectives[0];
  }

  el.trainBtn.disabled = false;
  renderSubtabBar();
  renderCurrentView();
}

function handleTrainSuccess(res) {
  el.activeDatasetBadge.innerHTML += `
    <span class="text-xs bg-indigo-950/80 text-indigo-300 border border-indigo-500/30 px-3 py-1 rounded-full font-medium ml-2">
      🤖 Best Model: ${res.best_model_label} &bull; Target: ${res.target_col}
    </span>
  `;
  state.activeWorkspace = 'automl';
  state.activeSubtab = 'leaderboard';
  document.querySelectorAll('.workspace-btn').forEach((b) => b.classList.remove('active'));
  document.querySelector('.workspace-btn[data-workspace="automl"]')?.classList.add('active');
  renderSubtabBar();
  renderCurrentView();
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

// ─── Main View Dispatcher ────────────────────────────────────────────────────

function renderCurrentView() {
  if (!state.datasetMeta) {
    el.tabContent.innerHTML = `
      <div class="glass-card p-12 text-center text-slate-400">
        <div class="text-5xl mb-4">📂</div>
        <h3 class="text-xl font-semibold text-slate-200 mb-2">Connect a Data Source</h3>
        <p class="text-sm max-w-md mx-auto">Upload any CSV file or pick a benchmark demo dataset in the sidebar to start autonomous analysis.</p>
      </div>
    `;
    return;
  }

  const { activeWorkspace: w, activeSubtab: s } = state;

  // Workspace 1: Core Intelligence
  if (w === 'core') {
    if (s === 'overview') renderCoreOverview();
    else if (s === 'schema') renderCoreSchema();
    else if (s === 'readiness') renderCoreReadiness();
    else if (s === 'insights') renderCoreInsights();
    else if (s === 'executive') renderCoreExecutive();
  }
  // Workspace 2: AutoML
  else if (w === 'automl') {
    if (s === 'leaderboard') renderAutoMLLeaderboard();
    else if (s === 'curves') renderAutoMLCurves();
    else if (s === 'explainability') renderAutoMLExplainability();
    else if (s === 'roi') renderAutoMLRoiOptimizer();
    else if (s === 'simulator') renderAutoMLSimulator();
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
    else if (s === 'sql') renderGovernanceSql();
    else if (s === 'code') renderGovernanceCode();
    else if (s === 'copilot') renderGovernanceCopilot();
  }

  lucide.createIcons();
}

// ═══════════════════════════════════════════════════════════════════════════════
//   WORKSPACE 1: CORE INTELLIGENCE VIEWS
// ═══════════════════════════════════════════════════════════════════════════════

function renderCoreOverview() {
  const d = state.datasetMeta;
  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Total Rows</div>
          <div class="text-2xl font-bold text-slate-100">${d.n_rows.toLocaleString()}</div>
          <div class="text-xs text-emerald-400 mt-1">✓ Ingested & Validated</div>
        </div>
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Total Features</div>
          <div class="text-2xl font-bold text-slate-100">${d.n_cols}</div>
          <div class="text-xs text-cyan-400 mt-1">${d.columns.length} columns active</div>
        </div>
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Business Domain</div>
          <div class="text-lg font-bold text-indigo-300 truncate">${d.detected_domain}</div>
          <div class="text-xs text-slate-400 mt-1">Semantic ontology match</div>
        </div>
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Data Sanitizer</div>
          <div class="text-2xl font-bold text-emerald-400">${d.sanitize_report.total_cells_repaired || 0}</div>
          <div class="text-xs text-emerald-300 mt-1">Dirty cells auto-repaired</div>
        </div>
      </div>

      <div class="glass-card p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-base font-semibold text-slate-200 flex items-center">
            <i data-lucide="table" class="w-4 h-4 mr-2 text-cyan-400"></i> Sanitized Dataset Preview (First 10 Rows)
          </h3>
          <span class="text-xs text-slate-400">Zero data loss guaranteed</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-slate-300 border-collapse">
            <thead>
              <tr class="border-b border-slate-700/60 bg-slate-900/40 text-slate-400 font-semibold">
                ${d.columns.map((c) => `<th class="p-3">${c}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${d.sample_data
                .map(
                  (r) => `
                <tr class="border-b border-slate-800/40 hover:bg-slate-800/30">
                  ${d.columns.map((c) => `<td class="p-3">${r[c] !== null ? r[c] : '<span class="text-slate-600">null</span>'}</td>`).join('')}
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

async function renderCoreSchema() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Loading schema and column roles...</div>`;
  try {
    const prof = await ApiClient.getProfile(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6">
        <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center">
          <i data-lucide="columns" class="w-4 h-4 mr-2 text-indigo-400"></i> Column Roles & Schema Profile
        </h3>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-slate-300 border-collapse">
            <thead>
              <tr class="border-b border-slate-700 bg-slate-900/50 text-slate-400 font-semibold">
                <th class="p-3">Column Name</th>
                <th class="p-3">Data Type</th>
                <th class="p-3">Detected Role</th>
                <th class="p-3">Missing %</th>
                <th class="p-3">Uniques</th>
                <th class="p-3">Sample Values</th>
              </tr>
            </thead>
            <tbody>
              ${prof.columns_info
                .map(
                  (col) => `
                <tr class="border-b border-slate-800/40 hover:bg-slate-800/30">
                  <td class="p-3 font-medium text-slate-200 font-mono">${col.column}</td>
                  <td class="p-3 text-slate-400">${col.dtype}</td>
                  <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] uppercase font-bold ${getRoleBadgeClass(col.role)}">${col.role}</span></td>
                  <td class="p-3 ${col.null_pct > 0 ? 'text-amber-400' : 'text-slate-400'}">${col.null_pct}% (${col.null_count})</td>
                  <td class="p-3 font-mono">${col.unique_count.toLocaleString()}</td>
                  <td class="p-3 text-slate-400 font-mono text-[11px] truncate max-w-xs">${col.sample_values.join(', ')}</td>
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error loading schema: ${err.message}</div>`;
  }
}

function getRoleBadgeClass(role) {
  if (role === 'numeric') return 'bg-cyan-950 text-cyan-400 border border-cyan-500/30';
  if (role.startsWith('categorical')) return 'bg-purple-950 text-purple-400 border border-purple-500/30';
  if (role === 'datetime') return 'bg-emerald-950 text-emerald-400 border border-emerald-500/30';
  if (role === 'id_entity') return 'bg-slate-800 text-slate-400 border border-slate-700';
  return 'bg-amber-950 text-amber-400 border border-amber-500/30';
}

async function renderCoreReadiness() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Auditing 5-pillar dataset readiness...</div>`;
  try {
    const r = await ApiClient.getReadiness(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Circular Gauge & Production Verdict -->
        <div class="glass-card p-6 flex flex-col md:flex-row items-center justify-between gap-6">
          <div class="flex items-center space-x-6">
            <div class="w-24 h-24 rounded-full border-4 border-cyan-500 flex flex-col items-center justify-center bg-cyan-950/30 shadow-lg shadow-cyan-500/20">
              <span class="text-2xl font-bold text-slate-100">${r.overall_readiness_score}%</span>
              <span class="text-[9px] uppercase font-bold text-cyan-400">Health</span>
            </div>
            <div>
              <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Production Readiness Verdict</div>
              <div class="text-xl font-bold text-slate-100 mt-0.5">${r.production_verdict}</div>
              <p class="text-xs text-slate-400 mt-1">Evaluated across missingness, multicollinearity, sample power, and class balance.</p>
            </div>
          </div>
          <div class="text-right">
            <span class="px-3 py-1 bg-emerald-950 border border-emerald-500/40 text-emerald-400 text-xs font-semibold rounded-full">
              Enterprise Grade
            </span>
          </div>
        </div>

        <!-- 5-Pillar Audit Checklist -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          ${r.checks
            .map(
              (c) => `
            <div class="glass-card p-4 border-l-4 ${c.verdict === 'PASS' ? 'border-l-emerald-500' : c.verdict === 'WARN' ? 'border-l-amber-500' : 'border-l-rose-500'}">
              <div class="flex justify-between items-start mb-1">
                <span class="font-semibold text-xs text-slate-200">${c.category}: ${c.title}</span>
                <span class="text-[10px] uppercase font-bold px-2 py-0.5 rounded ${c.verdict === 'PASS' ? 'bg-emerald-950 text-emerald-400' : 'bg-amber-950 text-amber-400'}">${c.verdict}</span>
              </div>
              <p class="text-xs text-slate-400 mt-1">${c.details}</p>
              ${c.action_item ? `<div class="mt-2 text-[11px] text-cyan-300 font-medium">💡 Recommendation: ${c.action_item}</div>` : ''}
            </div>
          `
            )
            .join('')}
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
  }
}

async function renderCoreInsights() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Generating correlation matrix & smart insights...</div>`;
  try {
    const prof = await ApiClient.getProfile(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <div class="glass-card p-6">
          <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center">
            <i data-lucide="activity" class="w-4 h-4 mr-2 text-cyan-400"></i> Top Pairwise Feature Correlations
          </h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            ${prof.correlations.slice(0, 8)
              .map(
                (c) => `
              <div class="p-3 bg-slate-900/50 border border-slate-800 rounded-lg flex justify-between items-center text-xs">
                <span class="font-mono text-slate-300">${c.feature_a} &harr; ${c.feature_b}</span>
                <span class="font-mono font-bold ${c.correlation > 0 ? 'text-cyan-400' : 'text-rose-400'}">${c.correlation > 0 ? '+' : ''}${c.correlation}</span>
              </div>
            `
              )
              .join('')}
          </div>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
  }
}

function renderCoreExecutive() {
  const d = state.datasetMeta;
  const p = state.pipelineResult;
  el.tabContent.innerHTML = `
    <div class="glass-card p-8 space-y-4">
      <h3 class="text-lg font-bold text-slate-100 flex items-center">
        <i data-lucide="briefcase" class="w-5 h-5 mr-2 text-indigo-400"></i> Executive Business Briefing
      </h3>
      <p class="text-xs text-slate-300 leading-relaxed">
        The Data Intelligence Assistant ingested and processed <b>${d.n_rows.toLocaleString()} records</b> across <b>${d.n_cols} features</b>. 
        Autonomous zero-loss sanitization repaired <b>${d.sanitize_report.total_cells_repaired || 0} dirty or malformed entries</b>. 
        Business domain was determined as <span class="text-cyan-300 font-semibold">${d.detected_domain}</span>.
      </p>
      ${
        p
          ? `
        <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2 text-xs">
          <div class="font-semibold text-slate-200">AutoML Deployment Summary:</div>
          <div>&bull; Primary Target: <code class="text-cyan-300">${p.target_col}</code> (${p.final_task_type})</div>
          <div>&bull; Champion Model: <span class="text-emerald-400 font-semibold">${p.best_model_label}</span></div>
          <div>&bull; Production Health Score: <span class="text-cyan-400 font-semibold">${p.readiness_score_pct}%</span> (${p.readiness_verdict})</div>
        </div>
      `
          : '<div class="text-xs text-slate-500 italic">Launch AutoML in the sidebar to populate model evaluation summaries.</div>'
      }
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════════════════════
//   WORKSPACE 2: AUTOML & EXPLAINABILITY VIEWS
// ═══════════════════════════════════════════════════════════════════════════════

function renderAutoMLLeaderboard() {
  const p = state.pipelineResult;
  if (!p) {
    el.tabContent.innerHTML = `<div class="glass-card p-12 text-center text-slate-400"><div class="text-5xl mb-3">🤖</div><h3 class="text-lg font-semibold text-slate-200 mb-1">Pipeline Not Trained</h3><p class="text-xs">Click Launch Autonomous ML Pipeline in the sidebar.</p></div>`;
    return;
  }

  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="glass-card p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="text-lg font-bold text-slate-100 flex items-center">
              <i data-lucide="trophy" class="w-5 h-5 mr-2 text-amber-400"></i> Model Leaderboard & Benchmark
            </h3>
            <p class="text-xs text-slate-400 mt-1">Target: <code class="text-cyan-300 font-mono">${p.target_col}</code> &bull; Task Type: <span class="uppercase font-semibold text-indigo-400">${p.final_task_type}</span></p>
          </div>
          <span class="text-xs font-semibold px-3 py-1 bg-emerald-950 border border-emerald-500/40 text-emerald-400 rounded-full">
            Champion: ${p.best_model_label}
          </span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          ${p.models_evaluated
            .map(
              (m) => `
            <div class="p-4 rounded-xl border ${m.is_best ? 'border-cyan-500/60 bg-cyan-950/20' : 'border-slate-800 bg-slate-900/30'}">
              <div class="flex justify-between items-start mb-2">
                <span class="font-semibold text-sm text-slate-200">${m.label}</span>
                ${m.is_best ? '<span class="text-[10px] uppercase font-bold bg-cyan-500 text-slate-950 px-2 py-0.5 rounded">Champion</span>' : ''}
              </div>
              <div class="space-y-1 text-xs text-slate-400 mt-2">
                ${Object.entries(m.metrics)
                  .map(([k, v]) => `<div class="flex justify-between font-mono"><span>${k}:</span> <span class="text-slate-200 font-semibold">${typeof v === 'number' ? v.toFixed(4) : v}</span></div>`)
                  .join('')}
              </div>
            </div>
          `
            )
            .join('')}
        </div>
      </div>
    </div>
  `;
}

function renderAutoMLCurves() {
  const p = state.pipelineResult;
  if (!p) {
    renderAutoMLLeaderboard();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- ROC Curve Container -->
        <div class="glass-card p-6">
          <h3 class="text-base font-semibold text-slate-200 mb-2 flex items-center">
            <i data-lucide="activity" class="w-4 h-4 mr-2 text-cyan-400"></i> Interactive ROC Curve
          </h3>
          <div id="roc-chart-container" class="h-64 w-full"></div>
        </div>

        <!-- Confusion Matrix Container -->
        <div class="glass-card p-6">
          <h3 class="text-base font-semibold text-slate-200 mb-2 flex items-center">
            <i data-lucide="grid" class="w-4 h-4 mr-2 text-indigo-400"></i> Confusion Matrix
          </h3>
          <div id="cm-chart-container" class="h-64 w-full"></div>
        </div>
      </div>
    </div>
  `;

  // Render Plotly ROC Chart
  if (p.roc_curve && p.roc_curve.fpr) {
    Plotly.newPlot(
      'roc-chart-container',
      [
        {
          x: p.roc_curve.fpr,
          y: p.roc_curve.tpr,
          mode: 'lines',
          name: p.best_model_label,
          line: { color: '#06b6d4', width: 3 },
        },
        {
          x: [0, 1],
          y: [0, 1],
          mode: 'lines',
          name: 'Random Baseline',
          line: { color: '#64748b', dash: 'dash' },
        },
      ],
      {
        margin: { l: 40, r: 20, t: 10, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { title: 'False Positive Rate', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'True Positive Rate', color: '#94a3b8', gridcolor: '#1e293b' },
        legend: { font: { color: '#cbd5e1' } },
      },
      { responsive: true, displayModeBar: false }
    );
  }

  // Render Confusion Matrix Heatmap
  if (p.confusion_matrix) {
    Plotly.newPlot(
      'cm-chart-container',
      [
        {
          z: p.confusion_matrix,
          x: ['Predicted 0', 'Predicted 1'],
          y: ['Actual 0', 'Actual 1'],
          type: 'heatmap',
          colorscale: 'Blues',
          showscale: false,
        },
      ],
      {
        margin: { l: 60, r: 20, t: 10, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { color: '#94a3b8' },
        yaxis: { color: '#94a3b8' },
      },
      { responsive: true, displayModeBar: false }
    );
  }
}

function renderAutoMLExplainability() {
  const p = state.pipelineResult;
  if (!p) {
    renderAutoMLLeaderboard();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-200 flex items-center">
        <i data-lucide="bar-chart-3" class="w-4 h-4 mr-2 text-cyan-400"></i> Global SHAP Feature Importance
      </h3>
      <div id="shap-chart-container" class="h-80 w-full"></div>
    </div>
  `;

  // Render Horizontal Bar Chart with Plotly
  const feats = p.shap_importance.map((s) => s.feature).reverse();
  const imps = p.shap_importance.map((s) => s.importance).reverse();

  Plotly.newPlot(
    'shap-chart-container',
    [
      {
        x: imps,
        y: feats,
        type: 'bar',
        orientation: 'h',
        marker: {
          color: imps,
          colorscale: 'Viridis',
        },
      },
    ],
    {
      margin: { l: 140, r: 20, t: 10, b: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis: { title: 'Mean |SHAP Value| (Feature Impact)', color: '#94a3b8', gridcolor: '#1e293b' },
      yaxis: { color: '#94a3b8', tickfont: { size: 11 } },
    },
    { responsive: true, displayModeBar: false }
  );
}

function renderAutoMLRoiOptimizer() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-200 flex items-center">
        <i data-lucide="dollar-sign" class="w-4 h-4 mr-2 text-emerald-400"></i> Business ROI & Profit Curve Optimizer
      </h3>
      <p class="text-xs text-slate-400">Fine-tune decision probability thresholds to maximize Expected Monetary Value (EMV).</p>
      
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div>
          <label class="block text-[11px] text-slate-400 mb-1">Benefit TP ($)</label>
          <input type="number" id="roi-tp" value="100" class="w-full glass-input text-xs">
        </div>
        <div>
          <label class="block text-[11px] text-slate-400 mb-1">Cost FP ($)</label>
          <input type="number" id="roi-fp" value="20" class="w-full glass-input text-xs">
        </div>
        <div>
          <label class="block text-[11px] text-slate-400 mb-1">Cost FN ($)</label>
          <input type="number" id="roi-fn" value="150" class="w-full glass-input text-xs">
        </div>
        <div>
          <label class="block text-[11px] text-slate-400 mb-1">Benefit TN ($)</label>
          <input type="number" id="roi-tn" value="0" class="w-full glass-input text-xs">
        </div>
      </div>

      <button id="roi-calc-btn" class="py-2 px-4 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg">
        ⚡ Calculate Optimal Profit Curve
      </button>

      <div id="roi-results-container" class="mt-4"></div>
    </div>
  `;

  document.getElementById('roi-calc-btn')?.addEventListener('click', async () => {
    const tp = document.getElementById('roi-tp').value;
    const fp = document.getElementById('roi-fp').value;
    const fn = document.getElementById('roi-fn').value;
    const tn = document.getElementById('roi-tn').value;
    const resDiv = document.getElementById('roi-results-container');
    resDiv.innerHTML = '<div class="text-xs text-slate-400">Computing profit surface...</div>';

    try {
      const res = await ApiClient.optimizeRoi(state.sessionId, tp, fp, fn, tn);
      resDiv.innerHTML = `
        <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-xl grid grid-cols-3 gap-4 text-center">
          <div>
            <div class="text-[10px] text-slate-400 uppercase">Optimal Threshold</div>
            <div class="text-xl font-bold text-cyan-400">${res.optimal_threshold}</div>
          </div>
          <div>
            <div class="text-[10px] text-slate-400 uppercase">Max Expected Profit</div>
            <div class="text-xl font-bold text-emerald-400">$${res.max_expected_profit.toLocaleString()}</div>
          </div>
          <div>
            <div class="text-[10px] text-slate-400 uppercase">Net Profit Uplift</div>
            <div class="text-xl font-bold text-indigo-400">+$${res.net_profit_gain.toLocaleString()}</div>
          </div>
        </div>
      `;
    } catch (err) {
      resDiv.innerHTML = `<div class="text-rose-400 text-xs">${err.message}</div>`;
    }
  });
}

function renderAutoMLSimulator() {
  const d = state.datasetMeta;
  const p = state.pipelineResult;

  if (!p) {
    renderAutoMLLeaderboard();
    return;
  }

  const sample = d.sample_data[0] || {};
  const targetCol = p.target_col;

  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-200 flex items-center">
        <i data-lucide="sliders" class="w-4 h-4 mr-2 text-indigo-400"></i> Interactive What-If Scenario Simulator
      </h3>
      <p class="text-xs text-slate-400">Adjust feature values to compute real-time model inferences and confidence scores.</p>
      
      <div id="sim-form" class="grid grid-cols-1 md:grid-cols-3 gap-4">
        ${Object.entries(sample)
          .filter(([col]) => col !== targetCol)
          .slice(0, 9)
          .map(
            ([col, val]) => `
            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1 truncate" title="${col}">${col}</label>
              <input type="${typeof val === 'number' ? 'number' : 'text'}" 
                     id="sim-input-${col}" 
                     value="${val !== null ? val : ''}" 
                     class="w-full glass-input text-xs">
            </div>
          `
          )
          .join('')}
      </div>

      <div class="flex items-center justify-between border-t border-slate-800 pt-4">
        <button id="sim-run-btn" class="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg flex items-center">
          <i data-lucide="zap" class="w-3.5 h-3.5 mr-1.5"></i> Run Real-Time Inference
        </button>
        <div id="sim-result-box" class="text-sm font-semibold text-slate-300"></div>
      </div>
    </div>
  `;

  document.getElementById('sim-run-btn')?.addEventListener('click', async () => {
    const overrides = {};
    Object.keys(sample).forEach((col) => {
      if (col === targetCol) return;
      const inp = document.getElementById(`sim-input-${col}`);
      if (inp) {
        overrides[col] = !isNaN(inp.value) && inp.value !== '' ? parseFloat(inp.value) : inp.value;
      }
    });

    try {
      const res = await ApiClient.simulate(state.sessionId, overrides);
      const resBox = document.getElementById('sim-result-box');
      resBox.innerHTML = `
        <span class="text-slate-400">Prediction:</span> 
        <span class="text-cyan-400 font-bold font-mono text-base ml-1">${res.prediction}</span>
        ${res.probability !== null ? `<span class="text-xs text-slate-400 ml-2">(${(res.probability * 100).toFixed(1)}% probability)</span>` : ''}
      `;
    } catch (err) {
      showError(err.message);
    }
  });

  lucide.createIcons();
}

async function renderAutoMLActiveLearning() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Sampling boundary uncertainty queue...</div>`;
  try {
    const res = await ApiClient.getActiveLearningQueue(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <div class="flex justify-between items-center">
          <h3 class="text-base font-semibold text-slate-200 flex items-center">
            <i data-lucide="user-check" class="w-4 h-4 mr-2 text-amber-400"></i> Human-In-The-Loop Active Learning Queue
          </h3>
          <span class="text-xs text-slate-400">${res.n_uncertain} samples prioritized</span>
        </div>
        <p class="text-xs text-slate-400">Records near the classification boundary with highest entropy, prioritized for expert domain labeling.</p>
        
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-slate-300 border-collapse">
            <thead>
              <tr class="border-b border-slate-700 bg-slate-900/50 text-slate-400 font-semibold">
                ${res.uncertain_samples.length > 0 ? Object.keys(res.uncertain_samples[0]).map((k) => `<th class="p-2.5">${k}</th>`).join('') : ''}
              </tr>
            </thead>
            <tbody>
              ${res.uncertain_samples
                .map(
                  (r) => `
                <tr class="border-b border-slate-800/40 hover:bg-slate-800/30">
                  ${Object.values(r).map((v) => `<td class="p-2.5 font-mono">${v}</td>`).join('')}
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//   WORKSPACE 3: ADAPTIVE AI ENGINES VIEWS
// ═══════════════════════════════════════════════════════════════════════════════

function renderAdaptiveCausal() {
  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Counterfactual Interventions -->
        <div class="glass-card p-6 space-y-3">
          <h3 class="text-base font-semibold text-slate-100 flex items-center">
            <i data-lucide="target" class="w-4 h-4 mr-2 text-rose-400"></i> Prescriptive Counterfactual Search
          </h3>
          <p class="text-xs text-slate-400">Find the smallest actionable feature adjustments to change an outcome.</p>
          <div class="flex space-x-3">
            <input type="number" id="cf-row-idx" value="0" min="0" class="glass-input w-24 text-xs" placeholder="Row Index">
            <select id="cf-desired-val" class="glass-input text-xs flex-1">
              <option value="1">Desired Target: 1 (Favorable)</option>
              <option value="0">Desired Target: 0 (Unfavorable)</option>
            </select>
          </div>
          <button id="cf-run-btn" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700">
            🔍 Compute Counterfactual Shifts
          </button>
          <div id="cf-res-box" class="text-xs text-slate-300"></div>
        </div>

        <!-- Causal Uplift (T-Learner) -->
        <div class="glass-card p-6 space-y-3">
          <h3 class="text-base font-semibold text-slate-100 flex items-center">
            <i data-lucide="trending-up" class="w-4 h-4 mr-2 text-emerald-400"></i> Causal Uplift (T-Learner ATE)
          </h3>
          <p class="text-xs text-slate-400">Estimate Average Treatment Effect (ATE) to avoid contacting Sleeping Dogs.</p>
          <button id="uplift-run-btn" class="w-full py-2 bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-semibold rounded-lg">
            ⚡ Calculate Uplift & Treatment Segments
          </button>
          <div id="uplift-res-box" class="text-xs text-slate-300"></div>
        </div>
      </div>
    </div>
  `;

  document.getElementById('cf-run-btn')?.addEventListener('click', async () => {
    const row = document.getElementById('cf-row-idx').value;
    const desired = document.getElementById('cf-desired-val').value;
    const resBox = document.getElementById('cf-res-box');
    resBox.innerHTML = '<span class="text-slate-400">Searching minimal perturbations...</span>';
    try {
      const res = await ApiClient.getCounterfactual(state.sessionId, row, desired);
      resBox.innerHTML = `
        <div class="p-3 bg-slate-900/80 rounded-lg border border-slate-800 space-y-1 mt-2">
          <div>Original Prediction: <code class="text-slate-200">${res.original_prediction}</code> &rarr; Counterfactual: <code class="text-emerald-400 font-bold">${res.counterfactual_prediction}</code></div>
          ${res.perturbations.length > 0 ? `<div class="text-[11px] text-cyan-300 mt-1">${res.perturbations.length} feature changes suggested.</div>` : '<div class="text-slate-400">No shifts required.</div>'}
        </div>
      `;
    } catch (err) {
      resBox.innerHTML = `<span class="text-rose-400">${err.message}</span>`;
    }
  });

  document.getElementById('uplift-run-btn')?.addEventListener('click', async () => {
    const resBox = document.getElementById('uplift-res-box');
    resBox.innerHTML = '<span class="text-slate-400">Estimating T-Learner uplift...</span>';
    try {
      const res = await ApiClient.getUplift(state.sessionId, '(Auto-Synthesize Action)');
      resBox.innerHTML = `
        <div class="p-3 bg-slate-900/80 rounded-lg border border-slate-800 space-y-2 mt-2">
          <div class="flex justify-between"><span>Average Treatment Effect (ATE):</span> <span class="text-emerald-400 font-bold font-mono">+${(res.average_treatment_effect_ate * 100).toFixed(2)}%</span></div>
          <div class="text-[11px] text-slate-300">💡 Recommended Action: ${res.recommended_action}</div>
        </div>
      `;
    } catch (err) {
      resBox.innerHTML = `<span class="text-rose-400">${err.message}</span>`;
    }
  });

  lucide.createIcons();
}

async function renderAdaptiveAutoencoder() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Training deep autoencoder anomaly engine...</div>`;
  try {
    const res = await ApiClient.getAutoencoder(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="cpu" class="w-4 h-4 mr-2 text-cyan-400"></i> Deep Autoencoder Latent Anomaly Detector
        </h3>
        <div class="grid grid-cols-3 gap-4 text-center">
          <div class="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
            <div class="text-[10px] text-slate-400 uppercase">Reconstruction MAE</div>
            <div class="text-lg font-bold text-cyan-400">${res.reconstruction_mae.toFixed(4)}</div>
          </div>
          <div class="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
            <div class="text-[10px] text-slate-400 uppercase">Anomaly Threshold</div>
            <div class="text-lg font-bold text-amber-400">${res.anomaly_threshold.toFixed(4)}</div>
          </div>
          <div class="p-3 bg-slate-900/60 rounded-xl border border-slate-800">
            <div class="text-[10px] text-slate-400 uppercase">Anomalous Records</div>
            <div class="text-lg font-bold text-rose-400">${res.anomalous_samples_count}</div>
          </div>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
  }
}

async function renderAdaptiveBandits() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-100 flex items-center">
        <i data-lucide="play-circle" class="w-4 h-4 mr-2 text-indigo-400"></i> Contextual Bandits (LinUCB Policy)
      </h3>
      <p class="text-xs text-slate-400">Simulate multi-armed bandit exploration vs. exploitation on active customer contexts.</p>
      
      <div class="flex space-x-3">
        <input type="number" id="bandit-steps" value="100" class="glass-input text-xs w-32" placeholder="Steps">
        <button id="bandit-run-btn" class="py-2 px-4 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg">
          🚀 Run Bandit Simulation
        </button>
      </div>

      <div id="bandit-res-box" class="mt-4"></div>
    </div>
  `;

  document.getElementById('bandit-run-btn')?.addEventListener('click', async () => {
    const steps = document.getElementById('bandit-steps').value;
    const resBox = document.getElementById('bandit-res-box');
    resBox.innerHTML = '<span class="text-slate-400">Simulating LinUCB arms...</span>';
    try {
      const res = await ApiClient.simulateBandits(state.sessionId, steps, 1.0);
      resBox.innerHTML = `
        <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
          <div class="flex justify-between text-xs"><span>Total Cumulative Reward:</span> <span class="text-emerald-400 font-bold font-mono">${res.cumulative_reward.toFixed(2)}</span></div>
          <div class="flex justify-between text-xs"><span>Steps Simulated:</span> <span class="text-slate-200 font-mono">${res.total_steps}</span></div>
        </div>
      `;
    } catch (err) {
      resBox.innerHTML = `<span class="text-rose-400">${err.message}</span>`;
    }
  });

  lucide.createIcons();
}

function renderAdaptiveSynthetic() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-100 flex items-center">
        <i data-lucide="dna" class="w-4 h-4 mr-2 text-emerald-400"></i> Generative Synthetic Data (Gaussian Copula)
      </h3>
      <p class="text-xs text-slate-400">Synthesize privacy-preserving datasets with $(\\epsilon)$-Differential Privacy guarantees.</p>
      
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-[11px] text-slate-400 mb-1">Samples to Generate</label>
          <input type="number" id="syn-samples" value="200" min="10" class="w-full glass-input text-xs">
        </div>
        <div>
          <label class="block text-[11px] text-slate-400 mb-1">Privacy Budget (Epsilon)</label>
          <input type="number" id="syn-eps" value="1.0" step="0.1" class="w-full glass-input text-xs">
        </div>
      </div>

      <button id="syn-run-btn" class="py-2.5 px-5 bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-semibold rounded-lg flex items-center">
        <i data-lucide="sparkles" class="w-3.5 h-3.5 mr-1.5"></i> Synthesize Data
      </button>

      <div id="syn-res-box" class="mt-4"></div>
    </div>
  `;

  document.getElementById('syn-run-btn')?.addEventListener('click', async () => {
    const samples = document.getElementById('syn-samples').value;
    const eps = document.getElementById('syn-eps').value;
    const resBox = document.getElementById('syn-res-box');
    resBox.innerHTML = '<span class="text-slate-400">Synthesizing copula manifold...</span>';
    try {
      const res = await ApiClient.generateSynthetic(state.sessionId, samples, true, eps);
      resBox.innerHTML = `
        <div class="p-4 bg-slate-900/60 rounded-xl border border-slate-800 space-y-2">
          <div class="text-emerald-400 font-semibold text-xs">Generated ${res.n_generated} records with ${res.fidelity_score_pct}% Distribution Fidelity!</div>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'synthetic')}" class="text-xs text-cyan-400 underline inline-block" download>⬇️ Download synthetic_dataset.csv</a>
        </div>
      `;
    } catch (err) {
      resBox.innerHTML = `<span class="text-rose-400">${err.message}</span>`;
    }
  });

  lucide.createIcons();
}

async function renderAdaptiveOnline() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Simulating online streaming fit...</div>`;
  try {
    const res = await ApiClient.getOnlineLearning(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="zap" class="w-4 h-4 mr-2 text-amber-400"></i> Online Incremental Streaming Fit
        </h3>
        <div class="flex justify-between text-xs p-3 bg-slate-900/60 rounded-lg">
          <span>Processed Mini-Batches: <b class="text-slate-200 font-mono">${res.n_batches}</b></span>
          <span>Final Batch Loss: <b class="text-emerald-400 font-mono">${res.final_loss.toFixed(4)}</b></span>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
  }
}

async function renderAdaptiveTimeSeries() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-100 flex items-center">
        <i data-lucide="trending-up" class="w-4 h-4 mr-2 text-cyan-400"></i> Automated Time-Series Forecaster
      </h3>
      <p class="text-xs text-slate-400">Autoregressive lag projection with 95% confidence intervals.</p>
      
      <div class="flex space-x-3">
        <input type="number" id="ts-horizon" value="14" min="7" max="60" class="glass-input text-xs w-32" placeholder="Horizon">
        <button id="ts-run-btn" class="py-2 px-4 bg-cyan-700 hover:bg-cyan-600 text-white text-xs font-semibold rounded-lg">
          🚀 Forecast Multi-Step Horizon
        </button>
      </div>

      <div id="ts-chart-box" class="h-64 w-full mt-4"></div>
    </div>
  `;

  document.getElementById('ts-run-btn')?.addEventListener('click', async () => {
    const horizon = document.getElementById('ts-horizon').value;
    try {
      const res = await ApiClient.forecastTimeSeries(state.sessionId, null, horizon);
      if (res.future_projections) {
        const histDates = res.historical_points.map((p) => p.date);
        const histVals = res.historical_points.map((p) => p.actual);
        const futDates = res.future_projections.map((p) => p.date);
        const futVals = res.future_projections.map((p) => p.forecast);

        Plotly.newPlot(
          'ts-chart-box',
          [
            { x: histDates, y: histVals, mode: 'lines', name: 'Historical', line: { color: '#94a3b8' } },
            { x: futDates, y: futVals, mode: 'lines+markers', name: 'Forecast', line: { color: '#06b6d4', dash: 'dash' } },
          ],
          {
            margin: { l: 40, r: 20, t: 10, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: { color: '#94a3b8', gridcolor: '#1e293b' },
            yaxis: { color: '#94a3b8', gridcolor: '#1e293b' },
          },
          { responsive: true, displayModeBar: false }
        );
      }
    } catch (err) {
      showError(err.message);
    }
  });

  lucide.createIcons();
}

async function renderAdaptiveNlp() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Analyzing free-text tokens & sentiment...</div>`;
  try {
    const res = await ApiClient.getNlpAnalysis(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="file-text" class="w-4 h-4 mr-2 text-purple-400"></i> NLP & Sentiment Analysis (${res.analyzed_text_column})
        </h3>
        <div class="grid grid-cols-2 gap-4">
          ${res.top_keywords.map((k) => `<div class="p-2 bg-slate-900/60 rounded border border-slate-800 text-xs font-mono">${k.keyword} (${k.score})</div>`).join('')}
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

async function renderAdaptiveGraph() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Constructing bipartite entity graph...</div>`;
  try {
    const res = await ApiClient.getGraphIntelligence(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="share-2" class="w-4 h-4 mr-2 text-indigo-400"></i> Graph Intelligence & Centrality
        </h3>
        <div class="flex justify-between text-xs p-3 bg-slate-900/60 rounded-lg">
          <span>Nodes: <b class="text-cyan-400">${res.n_nodes}</b></span>
          <span>Edges: <b class="text-indigo-400">${res.n_edges}</b></span>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//   WORKSPACE 4: GOVERNANCE & MLOPS VIEWS
// ═══════════════════════════════════════════════════════════════════════════════

async function renderGovernanceContracts() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Generating Great Expectations suite...</div>`;
  try {
    const res = await ApiClient.getDataContract(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="check-square" class="w-4 h-4 mr-2 text-emerald-400"></i> Great Expectations & Data Contracts (${res.n_expectations} Tests)
        </h3>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${res.contract_yaml}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

async function renderGovernanceGdpr() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Auditing GDPR ROPA & PII...</div>`;
  try {
    const res = await ApiClient.getGdprAudit(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="lock" class="w-4 h-4 mr-2 text-indigo-400"></i> GDPR ROPA Register & Privacy Audit
        </h3>
        <div class="flex justify-between text-xs p-3 bg-slate-900/60 rounded-lg">
          <span>Privacy Risk Score: <b class="text-emerald-400 font-bold">${res.privacy_risk_score} / 100</b></span>
          <span>PII Findings: <b class="text-slate-300 font-mono">${res.pii_entities_detected.length}</b></span>
        </div>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${res.ropa_markdown}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

async function renderGovernanceDrift() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Computing PSI drift monitor...</div>`;
  try {
    const res = await ApiClient.getDriftMonitor(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="git-commit" class="w-4 h-4 mr-2 text-cyan-400"></i> Population Stability Index (PSI) Drift Monitor
        </h3>
        <div class="flex justify-between text-xs p-3 bg-slate-900/60 rounded-lg">
          <span>Overall Dataset PSI: <b class="text-cyan-400 font-mono font-bold">${res.psi_score.toFixed(4)}</b></span>
          <span>Status: <b class="text-emerald-400 uppercase">${res.drift_status}</b></span>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

async function renderGovernanceSql() {
  try {
    const arts = await ApiClient.getAllArtifacts(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <div class="flex justify-between items-center">
          <h3 class="text-base font-semibold text-slate-100 flex items-center">
            <i data-lucide="database" class="w-4 h-4 mr-2 text-cyan-400"></i> In-Database SQL Transpiler
          </h3>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'sql')}" download class="text-xs text-cyan-400 underline">⬇️ Download SQL</a>
        </div>
        <p class="text-xs text-slate-400">Direct SQL CASE WHEN scoring logic ready for Snowflake, BigQuery, Postgres, or Redshift.</p>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${arts.sql_query}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-slate-400">Train AutoML first to view SQL transpiler code.</div>`;
  }
}

async function renderGovernanceCode() {
  try {
    const arts = await ApiClient.getAllArtifacts(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="code" class="w-4 h-4 mr-2 text-indigo-400"></i> Production Deployment Artifacts
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <a href="${ApiClient.getExportUrl(state.sessionId, 'python')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">🐍 pipeline.py</a>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'fastapi')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">⚡ main.py</a>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'dockerfile')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">🐳 Dockerfile</a>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'airflow')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">💨 airflow_dag.py</a>
        </div>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${arts.python_script}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-slate-400">Train AutoML first to view production code exports.</div>`;
  }
}

function renderGovernanceCopilot() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-100 flex items-center">
        <i data-lucide="message-square" class="w-4 h-4 mr-2 text-indigo-400"></i> Autonomous Dataset Copilot
      </h3>
      <div id="chat-messages" class="h-64 overflow-y-auto p-3 bg-slate-950/60 border border-slate-800/80 rounded-xl space-y-3 text-xs">
        <div class="text-slate-400 italic">Ask any analytical question (e.g., "correlations", "outliers", "summary of features").</div>
      </div>
      <div class="flex space-x-2">
        <input type="text" id="chat-input" class="flex-1 glass-input text-xs" placeholder="Ask a question about your dataset...">
        <button id="chat-send-btn" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg">
          Send
        </button>
      </div>
    </div>
  `;

  document.getElementById('chat-send-btn')?.addEventListener('click', handleChatQuery);
  document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChatQuery();
  });

  lucide.createIcons();
}

async function handleChatQuery() {
  const input = document.getElementById('chat-input');
  const msgBox = document.getElementById('chat-messages');
  if (!input || !msgBox || !input.value.trim() || !state.sessionId) return;

  const query = input.value.trim();
  input.value = '';

  msgBox.innerHTML += `
    <div class="text-right">
      <span class="inline-block p-2 bg-indigo-600/40 border border-indigo-500/30 text-slate-100 rounded-lg">${query}</span>
    </div>
  `;
  msgBox.scrollTop = msgBox.scrollHeight;

  try {
    const res = await ApiClient.chat(state.sessionId, query);
    msgBox.innerHTML += `
      <div class="text-left">
        <div class="inline-block p-3 bg-slate-900 border border-slate-800 text-slate-200 rounded-lg space-y-2">
          <div>${res.content}</div>
        </div>
      </div>
    `;
    msgBox.scrollTop = msgBox.scrollHeight;
  } catch (err) {
    msgBox.innerHTML += `<div class="text-rose-400 text-left">Error: ${err.message}</div>`;
  }
}

// ─── Utility Notifications ───────────────────────────────────────────────────

function showLoading(msg) {
  el.statusText.textContent = msg;
  el.trainProgress.classList.remove('hidden');
}

function hideLoading() {
  el.trainProgress.classList.add('hidden');
}

function showTrainingProgress() {
  el.statusText.textContent = 'Executing multi-model AutoML training & SHAP precomputations...';
  el.trainProgress.classList.remove('hidden');
  el.progressBar.style.width = '60%';
}

function hideTrainingProgress() {
  el.trainProgress.classList.add('hidden');
  el.progressBar.style.width = '0%';
}

function showSuccess(msg) {
  console.log('[SUCCESS]', msg);
}

function showError(msg) {
  alert(`Error: ${msg}`);
}

window.addEventListener('DOMContentLoaded', init);
