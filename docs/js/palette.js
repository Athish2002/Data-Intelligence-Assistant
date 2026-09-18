/**
 * frontend/js/palette.js
 * ──────────────────────
 * Quick Command Palette Modal (⌘K) Controller.
 */

import { state, el } from './state.js';
import { toggleDock, toggleInspector } from './inspector.js';

export const COMMAND_LIST = [
  { id: 'ws-automl-board', title: 'Jump to: 2. Autonomous ML &bull; Model Leaderboard', category: 'Workspace', run: () => window.switchWorkspace('automl', 'leaderboard') },
  { id: 'ws-automl-cockpit', title: 'Jump to: 2. Autonomous ML &bull; Champion vs. Challenger Split Cockpit', category: 'Workspace', run: () => window.switchWorkspace('automl', 'cockpit') },
  { id: 'ws-automl-simulator', title: 'Jump to: 2. Autonomous ML &bull; What-If & Algorithmic Recourse Playground', category: 'Workspace', run: () => window.switchWorkspace('automl', 'simulator') },
  { id: 'ws-automl-stress', title: 'Jump to: 2. Autonomous ML &bull; Monte Carlo Macro Stress Testing', category: 'Workspace', run: () => window.switchWorkspace('automl', 'stress') },
  { id: 'ws-automl-conformal', title: 'Jump to: 2. Autonomous ML &bull; Conformal Prediction & Uncertainty Sets', category: 'Workspace', run: () => window.switchWorkspace('automl', 'conformal') },
  { id: 'ws-automl-shap', title: 'Jump to: 2. Autonomous ML &bull; TreeSHAP Attributions', category: 'Workspace', run: () => window.switchWorkspace('automl', 'explainability') },
  { id: 'ws-core-symbolic', title: 'Jump to: 1. Data Intelligence &bull; Symbolic Feature Discovery & SQL', category: 'Workspace', run: () => window.switchWorkspace('core', 'symbolic') },
  { id: 'ws-adaptive-causal', title: 'Jump to: 3. Adaptive AI &bull; Interactive Causal DAG Studio', category: 'Workspace', run: () => window.switchWorkspace('adaptive', 'causal') },
  { id: 'ws-adaptive-bandits', title: 'Jump to: 3. Adaptive AI &bull; LinUCB Contextual Bandits', category: 'Workspace', run: () => window.switchWorkspace('adaptive', 'bandits') },
  { id: 'ws-gov-drift', title: 'Jump to: 4. Governance &bull; Drift Sentinel & Multivariate MMD', category: 'Workspace', run: () => window.switchWorkspace('governance', 'drift') },
  { id: 'ws-gov-dossier', title: 'Jump to: 4. Governance &bull; Executive Regulatory Audit Dossier', category: 'Workspace', run: () => window.switchWorkspace('governance', 'dossier') },
  { id: 'ws-gov-contracts', title: 'Jump to: 4. Governance &bull; Data Quality Contracts', category: 'Workspace', run: () => window.switchWorkspace('governance', 'contracts') },
  { id: 'nav-landing-page', title: 'Navigate to: 🏠 Landing Page (/)', category: 'Navigation', run: () => { window.location.href = '/'; } },
  { id: 'ws-dashboard-telemetry', title: 'Jump to: 📊 Executive Mission Control', category: 'Workspace', run: () => window.switchWorkspace('dashboard', 'mission_control') },
  { id: 'ws-dashboard-arch', title: 'Jump to: 🏛️ Full-Stack System Architecture Blueprint', category: 'Workspace', run: () => window.switchWorkspace('dashboard', 'system_arch') },
  { id: 'act-export-dossier', title: 'Action: Export Standalone Regulatory Audit Dossier (HTML)', category: 'Action', run: () => { if (state.sessionId) window.open(`/api/v1/export/${state.sessionId}/dossier`, '_blank'); } },
  { id: 'act-dock', title: 'Action: Toggle Left Ingestion Dock (⌘B)', category: 'Action', run: () => toggleDock() },
  { id: 'act-inspector', title: 'Action: Toggle Model & Feature Inspector (⌘I)', category: 'Action', run: () => toggleInspector() },
  { id: 'act-train', title: 'Action: Launch Autonomous ML Pipeline', category: 'Action', run: () => el.trainBtn?.click() },
  { id: 'act-system-monitor', title: 'System: View System Health & Resource Monitor', category: 'Action', run: () => window.openSystemMetrics?.() },
  { id: 'act-system-gc', title: 'System: Purge Inactive Sessions & Run GC', category: 'Action', run: () => window.triggerManualGC?.() },
  { id: 'act-mode-toggle', title: 'Action: Toggle Dark / Light Mode', category: 'Action', run: () => document.getElementById('mode-toggle-btn')?.click() },
  { id: 'act-aurora-toggle', title: 'Action: Toggle Cyber Aurora Ambient Background', category: 'Action', run: () => window.toggleAurora?.() },
  { id: 'theme-obsidian', title: 'Theme: 1. Obsidian &bull; Cyber Aurora (Dark Mode)', category: 'Theme', run: () => window.applyTheme?.('obsidian') },
  { id: 'theme-black', title: 'Theme: 2. Beige Flat Black (Classic Dark)', category: 'Theme', run: () => window.applyTheme?.('beige-black') },
  { id: 'theme-sand', title: 'Theme: 3. Tokyo Sand (Light Mode)', category: 'Theme', run: () => window.applyTheme?.('tokyo-sand') },
];

let selectedCommandIndex = 0;

export function setupCommandPalette() {
  // Global keyboard shortcuts (⌘K / Ctrl+K and ESC)
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      toggleCommandPalette();
    } else if (e.key === 'Escape' && el.commandPaletteModal && !el.commandPaletteModal.classList.contains('hidden')) {
      e.preventDefault();
      closeCommandPalette();
    }
  });

  if (el.commandPaletteTrigger) {
    el.commandPaletteTrigger.addEventListener('click', () => openCommandPalette());
  }
  if (el.hudCmdShortcut) {
    el.hudCmdShortcut.addEventListener('click', () => openCommandPalette());
  }
  if (el.commandPaletteModal) {
    el.commandPaletteModal.addEventListener('click', (e) => {
      if (e.target === el.commandPaletteModal) closeCommandPalette();
    });
  }

  if (el.commandInput) {
    el.commandInput.addEventListener('input', (e) => {
      renderCommandResults(e.target.value);
    });

    el.commandInput.addEventListener('keydown', (e) => {
      const items = el.commandResults.querySelectorAll('.command-item');
      if (!items.length) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedCommandIndex = (selectedCommandIndex + 1) % items.length;
        updateCommandSelection(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedCommandIndex = (selectedCommandIndex - 1 + items.length) % items.length;
        updateCommandSelection(items);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const activeItem = items[selectedCommandIndex];
        if (activeItem) activeItem.click();
      }
    });
  }
}

export function openCommandPalette() {
  if (!el.commandPaletteModal) return;
  el.commandPaletteModal.classList.remove('hidden');
  selectedCommandIndex = 0;
  if (el.commandInput) {
    el.commandInput.value = '';
    el.commandInput.focus();
  }
  renderCommandResults('');
}

export function closeCommandPalette() {
  if (!el.commandPaletteModal) return;
  el.commandPaletteModal.classList.add('hidden');
}

export function toggleCommandPalette() {
  if (!el.commandPaletteModal) return;
  if (el.commandPaletteModal.classList.contains('hidden')) {
    openCommandPalette();
  } else {
    closeCommandPalette();
  }
}

export function renderCommandResults(query = '') {
  if (!el.commandResults) return;
  const q = query.toLowerCase().trim();
  const filtered = COMMAND_LIST.filter(
    (c) => !q || c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q)
  );

  if (!filtered.length) {
    el.commandResults.innerHTML = `
      <div class="p-4 text-center text-xs" style="color: var(--text-muted);">
        No matching commands or destinations found.
      </div>
    `;
    return;
  }

  el.commandResults.innerHTML = filtered
    .map(
      (cmd, idx) => `
      <div class="command-item ${idx === selectedCommandIndex ? 'selected' : ''}" data-idx="${idx}">
        <div class="flex items-center space-x-2">
          <span class="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded border" style="background-color: var(--tag-bg); color: var(--tag-text); border-color: var(--card-border);">${cmd.category}</span>
          <span class="text-xs font-mono" style="color: var(--text-primary);">${cmd.title}</span>
        </div>
        <span class="text-[10px] font-mono text-stone-500">&crarr;</span>
      </div>
    `
    )
    .join('');

  el.commandResults.querySelectorAll('.command-item').forEach((item, idx) => {
    item.addEventListener('click', () => {
      closeCommandPalette();
      filtered[idx].run();
    });
    item.addEventListener('mouseenter', () => {
      selectedCommandIndex = idx;
      updateCommandSelection(el.commandResults.querySelectorAll('.command-item'));
    });
  });
}

export function updateCommandSelection(items) {
  items.forEach((it, idx) => {
    if (idx === selectedCommandIndex) {
      it.classList.add('selected');
      it.scrollIntoView({ block: 'nearest' });
    } else {
      it.classList.remove('selected');
    }
  });
}

window.openCommandPalette = openCommandPalette;
window.closeCommandPalette = closeCommandPalette;
window.toggleCommandPalette = toggleCommandPalette;
