/**
 * frontend/js/state.js
 * ────────────────────
 * Central Reactive Application State, Workspace Definitions, and DOM Cache.
 */

export const state = {
  sessionId: null,
  datasetMeta: null,
  pipelineResult: null,
  activeWorkspace: 'dashboard',
  activeSubtab: 'overview',
};

export const WORKSPACE_SUBTABS = {
  dashboard: [
    { id: 'overview', label: 'Executive Mission Control', icon: 'layout-dashboard' },
  ],
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

export const el = {
  get demoSelect() { return document.getElementById('demo-select'); },
  get loadDemoBtn() { return document.getElementById('load-demo-btn'); },
  get csvFileInput() { return document.getElementById('csv-file-input'); },
  get uploadDropzone() { return document.getElementById('upload-dropzone'); },
  get goalInput() { return document.getElementById('goal-input'); },
  get targetColInput() { return document.getElementById('target-col-input'); },
  get autodetectBtn() { return document.getElementById('autodetect-btn'); },
  get trainBtn() { return document.getElementById('train-btn'); },
  get trainProgress() { return document.getElementById('train-progress'); },
  get progressBar() { return document.getElementById('progress-bar'); },
  get statusText() { return document.getElementById('status-text'); },
  get workspaceNav() { return document.getElementById('workspace-nav'); },
  get subtabBar() { return document.getElementById('subtab-bar'); },
  get tabContent() { return document.getElementById('tab-content'); },
  get activeDatasetBadge() { return document.getElementById('active-dataset-badge'); },
  get activeModelBadge() { return document.getElementById('active-model-badge'); },
  get systemHealthBadge() { return document.getElementById('system-health-badge'); },

  // Workbench Style 1 Dock & Inspector Elements
  get leftDock() { return document.getElementById('left-dock'); },
  get toggleDockBtn() { return document.getElementById('toggle-dock-btn'); },
  get inspectorDrawer() { return document.getElementById('inspector-drawer'); },
  get toggleInspectorBtn() { return document.getElementById('toggle-inspector-btn'); },
  get closeInspectorBtn() { return document.getElementById('close-inspector-btn'); },
  get inspectorContent() { return document.getElementById('inspector-content'); },
  get commandPaletteModal() { return document.getElementById('command-palette-modal'); },
  get commandPaletteTrigger() { return document.getElementById('command-palette-trigger'); },
  get commandInput() { return document.getElementById('command-input'); },
  get commandResults() { return document.getElementById('command-results'); },
  get hudDataset() { return document.getElementById('hud-dataset'); },
  get hudModel() { return document.getElementById('hud-model'); },
  get hudCmdShortcut() { return document.getElementById('hud-cmd-shortcut'); },
  get hudDockShortcut() { return document.getElementById('hud-dock-shortcut'); },
  get hudInspectorShortcut() { return document.getElementById('hud-inspector-shortcut'); },
  get modeToggleBtn() { return document.getElementById('mode-toggle-btn'); },
  get cursorDot() { return document.getElementById('cursor-dot'); },
  get cursorRing() { return document.getElementById('cursor-ring'); },
};

export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}
