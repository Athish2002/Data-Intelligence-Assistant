/**
 * frontend/js/inspector.js
 * ────────────────────────
 * Model & Deep Inspection Slide-out Drawer Controller.
 */

import { state, el } from './state.js';

export function setupWorkbenchFeatures() {
  if (el.toggleDockBtn) {
    el.toggleDockBtn.addEventListener('click', () => toggleDock());
  }
  if (el.hudDockShortcut) {
    el.hudDockShortcut.addEventListener('click', () => toggleDock());
  }

  if (el.toggleInspectorBtn) {
    el.toggleInspectorBtn.addEventListener('click', () => toggleInspector());
  }
  if (el.closeInspectorBtn) {
    el.closeInspectorBtn.addEventListener('click', () => toggleInspector(false));
  }
  if (el.hudInspectorShortcut) {
    el.hudInspectorShortcut.addEventListener('click', () => toggleInspector());
  }

  window.addEventListener('keydown', (e) => {
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    const modifier = isMac ? e.metaKey : e.ctrlKey;

    if (modifier && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      toggleDock();
    } else if (modifier && e.key.toLowerCase() === 'i') {
      e.preventDefault();
      toggleInspector();
    } else if (modifier && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      window.toggleCommandPalette?.();
    } else if (e.key === 'Escape') {
      if (el.commandPaletteModal && !el.commandPaletteModal.classList.contains('hidden')) {
        window.closeCommandPalette?.();
      } else if (el.inspectorDrawer && !el.inspectorDrawer.classList.contains('hidden')) {
        toggleInspector(false);
      }
    }
  });
}

export function toggleDock(forceState = null) {
  if (!el.leftDock) return;
  if (forceState === true) {
    el.leftDock.classList.remove('dock-collapsed');
    el.toggleDockBtn?.classList.remove('rotated');
  } else if (forceState === false) {
    el.leftDock.classList.add('dock-collapsed');
    el.toggleDockBtn?.classList.add('rotated');
  } else {
    el.leftDock.classList.toggle('dock-collapsed');
    el.toggleDockBtn?.classList.toggle('rotated');
  }
  window.dispatchEvent(new Event('resize'));
}

export function toggleInspector(forceState = null) {
  if (!el.inspectorDrawer) return;
  const isCurrentlyOpen = !el.inspectorDrawer.classList.contains('hidden');
  const shouldOpen = forceState !== null ? forceState : !isCurrentlyOpen;

  if (shouldOpen) {
    el.inspectorDrawer.classList.remove('hidden');
    renderInspectorContent();
  } else {
    el.inspectorDrawer.classList.add('hidden');
  }
  window.dispatchEvent(new Event('resize'));
}

export function closeInspector() {
  toggleInspector(false);
}

export function inspectModel(modelName = null) {
  toggleInspector(true);
  renderInspectorContent(modelName);
}

export function renderInspectorContent(modelName = null) {
  if (!el.inspectorContent) return;
  const p = state.pipelineResult;
  const d = state.datasetMeta;

  if (!p) {
    el.inspectorContent.innerHTML = `
      <div class="p-6 text-center text-xs font-mono" style="color: var(--text-muted);">
        <i data-lucide="info" class="w-6 h-6 mx-auto mb-2 opacity-50"></i>
        <span>No pipeline trained yet. Run AutoML to view deep telemetry and feature importances.</span>
      </div>
    `;
    lucide.createIcons();
    return;
  }

  const targetModel = modelName
    ? p.models_evaluated.find((m) => m.label.toLowerCase() === modelName.toLowerCase()) || p.models_evaluated[0]
    : p.models_evaluated.find((m) => m.is_best) || p.models_evaluated[0];

  const metrics = targetModel ? targetModel.metrics : {};
  const shapFeats = p.feature_importance ? Object.entries(p.feature_importance).slice(0, 10) : [];

  el.inspectorContent.innerHTML = `
    <div class="space-y-6 text-xs font-mono">
      <div>
        <div class="flex items-center justify-between mb-1.5">
          <span class="text-[10px] uppercase font-bold tracking-wider" style="color: var(--text-muted);">Inspecting Candidate</span>
          ${targetModel && targetModel.is_best ? '<span class="text-[9px] uppercase font-bold px-1.5 py-0.5 rounded bg-[#00e575] text-black">Champion</span>' : ''}
        </div>
        <h4 class="text-sm font-bold truncate" style="color: var(--text-primary);">${targetModel ? targetModel.label : 'N/A'}</h4>
        <p class="text-[11px] mt-0.5" style="color: var(--text-muted);">Cross-Validation: 5-Fold Stratified &bull; Metric: ${p.metric_used}</p>
      </div>

      <div>
        <h5 class="text-[10px] uppercase font-bold tracking-wider mb-2" style="color: var(--text-muted);">Validation Metrics Matrix</h5>
        <div class="grid grid-cols-2 gap-2">
          ${Object.entries(metrics)
            .map(([k, v]) => `
              <div class="p-2 rounded border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
                <div class="text-[9px] uppercase text-stone-500 truncate" title="${k}">${k}</div>
                <div class="text-xs font-bold text-[#00e575] mt-0.5">${typeof v === 'number' ? v.toFixed(4) : v}</div>
              </div>
            `).join('')}
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between mb-2">
          <h5 class="text-[10px] uppercase font-bold tracking-wider" style="color: var(--text-muted);">Top SHAP Attributions</h5>
          <span class="text-[10px] text-stone-500">|&Phi;| Impact</span>
        </div>
        <div class="space-y-2">
          ${shapFeats.map(([feat, val]) => {
            const maxVal = shapFeats[0] ? shapFeats[0][1] : 1;
            const pct = Math.min(100, Math.max(5, (val / (maxVal || 1)) * 100));
            return `
              <div>
                <div class="flex justify-between text-[11px] mb-1">
                  <span class="truncate max-w-[140px]" title="${feat}" style="color: var(--text-primary);">${feat}</span>
                  <span class="font-bold text-[#00e575]">${val.toFixed(4)}</span>
                </div>
                <div class="w-full h-1.5 rounded-full overflow-hidden" style="background-color: var(--bg-secondary);">
                  <div class="h-full rounded-full bg-[#00e575]" style="width: ${pct}%;"></div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>

      <div>
        <h5 class="text-[10px] uppercase font-bold tracking-wider mb-2" style="color: var(--text-muted);">Ground Truth & Target</h5>
        <div class="p-2.5 rounded border space-y-1.5" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
          <div class="flex justify-between">
            <span class="text-stone-500">Target Feature:</span>
            <strong class="text-[#00e575] truncate max-w-[120px]">${p.target_col}</strong>
          </div>
          <div class="flex justify-between">
            <span class="text-stone-500">Task Type:</span>
            <span class="uppercase font-semibold" style="color: var(--text-primary);">${p.final_task_type}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-stone-500">Training Samples:</span>
            <span style="color: var(--text-primary);">${d ? (d.n_rows || (d.shape ? d.shape[0] : 'N/A')).toLocaleString() : 'N/A'} rows</span>
          </div>
        </div>
      </div>
    </div>
  `;
  lucide.createIcons();
}

window.toggleDock = toggleDock;
window.toggleInspector = toggleInspector;
window.inspectModel = inspectModel;
