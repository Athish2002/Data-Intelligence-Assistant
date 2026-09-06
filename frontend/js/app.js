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
import { setupWorkbenchFeatures, inspectModel, renderInspectorContent, toggleInspector, closeInspector } from './inspector.js';
import { setupCommandPalette } from './palette.js';

// Import Views
import { renderExecutiveDashboard } from './views/dashboard.js';
import {
  renderCoreOverview,
  renderCoreSchema,
  renderCoreReadiness,
  renderCoreInsights,
  renderCoreExecutive,
} from './views/core.js';
import {
  renderModelRequiredState,
  renderAutoMLLeaderboard,
  renderAutoMLCurves,
  renderAutoMLExplainability,
  renderAutoMLRoiOptimizer,
  renderAutoMLSimulator,
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

window.inspectModel = inspectModel;
window.applyTheme = applyTheme;
window.loadDemoFromCard = loadDemoFromCard;
window.renderCurrentView = renderCurrentView;

// ─── Theme Manager ──────────────────────────────────────────────────────────

function setupThemeAndModeManager() {
  const selector = document.getElementById('theme-selector');
  const modeBtn = document.getElementById('mode-toggle-btn');
  let saved = localStorage.getItem('dia-theme');
  if (saved !== 'beige-black' && saved !== 'tokyo-sand') {
    saved = 'beige-black';
  }

  applyTheme(saved);

  if (selector) {
    selector.value = saved;
    selector.addEventListener('change', (e) => {
      applyTheme(e.target.value);
    });
  }

  if (modeBtn) {
    modeBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'beige-black';
      const next = current === 'tokyo-sand' ? 'beige-black' : 'tokyo-sand';
      applyTheme(next);
    });
  }
}

function applyTheme(themeName) {
  const finalTheme = themeName === 'tokyo-sand' ? 'tokyo-sand' : 'beige-black';
  document.documentElement.setAttribute('data-theme', finalTheme);
  localStorage.setItem('dia-theme', finalTheme);

  const selector = document.getElementById('theme-selector');
  if (selector) selector.value = finalTheme;

  const modeBtn = document.getElementById('mode-toggle-btn');
  if (modeBtn) {
    const isLight = finalTheme === 'tokyo-sand';
    modeBtn.innerHTML = isLight
      ? `<i id="mode-toggle-icon" data-lucide="sun" class="w-3.5 h-3.5 text-amber-500"></i>`
      : `<i id="mode-toggle-icon" data-lucide="moon" class="w-3.5 h-3.5 text-[#00e575]"></i>`;
    modeBtn.title = isLight ? 'Switch to Dark Mode (Primary Black)' : 'Switch to Light Mode (Tokyo Sand)';
    lucide.createIcons();
  }

  window.dispatchEvent(new Event('resize'));
  if (state.datasetMeta) {
    renderCurrentView();
  }
}

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
    el.systemHealthBadge.innerHTML = `
      <span class="live-pulse mr-1.5"></span>
      <span class="font-mono text-[10px] font-semibold text-[#4ade80]">CPU ${health.cpu_cores}C &bull; ${health.gpu_available ? 'GPU LIVE' : 'CPU'} &bull; v${health.version}</span>
    `;
  } catch (err) {
    el.systemHealthBadge.innerHTML = `<span class="text-[10px] text-rose-400">API Offline</span>`;
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

function handleTrainSuccess(res) {
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
    lucide.createIcons();
    return;
  }

  // If no dataset loaded yet and user visits other workspace, show helpful onboarding empty-state
  if (!state.datasetMeta) {
    el.tabContent.innerHTML = `
      <div class="glass-card p-8 text-center max-w-xl mx-auto my-12 space-y-4">
        <div class="w-12 h-12 rounded-full mx-auto flex items-center justify-center border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
          <i data-lucide="database" class="w-6 h-6 text-[#00e575]"></i>
        </div>
        <h3 class="text-base font-bold font-mono tracking-tight" style="color: var(--text-primary);">
          Dataset Connection Required
        </h3>
        <p class="text-xs leading-relaxed" style="color: var(--text-muted);">
          Workspace <strong class="text-[#00e575]">${w.toUpperCase()}</strong> requires an active dataset. Ingest a CSV from the left dock or select a pre-configured benchmark from the Executive Dashboard to proceed.
        </p>
        <div class="flex items-center justify-center space-x-3 pt-2">
          <button onclick="switchWorkspace('dashboard', 'overview')" class="px-4 py-2 rounded-lg font-bold text-xs border hover:bg-white/5 cursor-pointer flex items-center space-x-2" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-primary);">
            <i data-lucide="layout-dashboard" class="w-4 h-4 text-[#00e575]"></i>
            <span>Go to Executive Dashboard</span>
          </button>
          <button onclick="window.quickLoadBenchmark('Telecom Customer Churn')" class="px-4 py-2 rounded-lg font-bold text-xs uppercase tracking-wider cursor-pointer shadow-sm active:scale-95" style="background-color: var(--btn-primary-bg); color: var(--btn-primary-text);">
            Quick-Load Benchmark
          </button>
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
  setupThemeAndModeManager();
  setupCustomCursor();
  setupWorkbenchFeatures();
  setupCommandPalette();
  lucide.createIcons();
  setupEventListeners();
  await loadSystemHealth();
  await loadDemoList();
  renderSubtabBar();
  renderCurrentView();
  startSystemHealthBadgePolling();
}

window.addEventListener('DOMContentLoaded', init);
