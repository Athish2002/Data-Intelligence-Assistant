/**
 * frontend/js/views/automl.js
 * ───────────────────────────
 * Workspace 2: AutoML, Leaderboard, SHAP Explainability, Simulator, and ROI Optimizer.
 */

import { state, el } from '../state.js';
import { ApiClient } from '../api.js';
import { showError } from '../toast.js';
import { inspectModel } from '../inspector.js';

export function renderModelRequiredState(title, description, icon = 'cpu') {
  return `
    <div class="glass-card p-10 text-center max-w-lg mx-auto my-8 space-y-4">
      <div class="w-16 h-16 rounded-2xl mx-auto flex items-center justify-center border shadow-inner" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
        <i data-lucide="${icon}" class="w-8 h-8 text-[#00e575]"></i>
      </div>
      <div>
        <h3 class="text-base font-bold font-mono tracking-tight" style="color: var(--text-primary);">
          ${title}
        </h3>
        <p class="text-xs max-w-sm mx-auto mt-1.5 leading-relaxed" style="color: var(--text-muted);">
          ${description}
        </p>
      </div>
      <div class="pt-2">
        <button onclick="el.trainBtn.click()" class="px-5 py-2.5 rounded-lg font-bold text-xs uppercase tracking-wider cursor-pointer shadow-lg active:scale-95 inline-flex items-center space-x-2" style="background-color: var(--btn-primary-bg); color: var(--btn-primary-text);">
          <i data-lucide="play" class="w-4 h-4 fill-current"></i>
          <span>Launch Autonomous ML Pipeline</span>
        </button>
      </div>
    </div>
  `;
}

export function renderAutoMLLeaderboard() {
  const p = state.pipelineResult;
  if (!p) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Automated ML Leaderboard Awaiting Training',
      'Execute 5-fold cross-validation across candidate models (LightGBM, Random Forest, Ridge, Soft Voting) and compute TreeSHAP explainability.'
    );
    lucide.createIcons();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="space-y-4">
      <div class="glass-card p-5">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <div class="flex items-center space-x-2">
              <span class="text-[10px] font-mono uppercase px-1.5 py-0.2 rounded border" style="background:#092314; color:#4ade80; border-color:#134e2c;">5-Fold Cross-Validation</span>
              <h3 class="text-sm font-bold font-mono" style="color: var(--text-primary);">Automated Model Leaderboard</h3>
            </div>
            <p class="text-xs mt-1" style="color: var(--text-muted);">Target: <code class="font-mono text-[#00e575]">${p.target_col}</code> &bull; Task: <span class="uppercase font-semibold" style="color: var(--text-primary);">${p.final_task_type}</span></p>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-xs font-mono font-semibold px-2.5 py-1 rounded border" style="background-color: var(--tag-bg); color: var(--tag-text); border-color: var(--card-border);">
              Champion: <strong class="text-[#00e575]">${p.best_model_label}</strong>
            </span>
            <button onclick="window.inspectModel('${p.best_model_label}')" class="px-2.5 py-1 rounded text-xs font-mono font-bold border hover:bg-white/5 transition-all cursor-pointer flex items-center space-x-1" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-primary);">
              <i data-lucide="sidebar" class="w-3.5 h-3.5 text-[#00e575]"></i>
              <span>Inspect in Drawer &rarr;</span>
            </button>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          ${p.models_evaluated
            .map(
              (m) => `
            <div onclick="window.inspectModel('${m.label}')" class="p-3.5 rounded-lg border transition-all hover:translate-y-[-1px] cursor-pointer group" style="background-color: ${m.is_best ? 'var(--card-hover)' : 'var(--bg-secondary)'}; border-color: ${m.is_best ? '#00e575' : 'var(--card-border)'};">
              <div class="flex justify-between items-start mb-2">
                <span class="font-bold text-xs font-mono group-hover:text-[#00e575] transition-colors" style="color: var(--text-primary);">${m.label}</span>
                ${m.is_best ? '<span class="text-[9px] uppercase font-mono font-bold bg-[#00e575] text-black px-1.5 py-0.2 rounded">Champion</span>' : '<span class="text-[9px] font-mono opacity-0 group-hover:opacity-100 transition-opacity text-[#00e575]">Inspect &rarr;</span>'}
              </div>
              <div class="space-y-1 text-xs mt-2 border-t pt-2" style="border-color: var(--card-border);">
                ${Object.entries(m.metrics)
                  .map(([k, v]) => `
                    <div class="flex justify-between font-mono text-[11px]">
                      <span style="color: var(--text-muted);">${k}:</span>
                      <span class="font-semibold" style="color: var(--text-primary);">${typeof v === 'number' ? v.toFixed(4) : v}</span>
                    </div>
                  `)
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
  lucide.createIcons();
}

export function renderAutoMLCurves() {
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

export function renderAutoMLExplainability() {
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

export function renderAutoMLRoiOptimizer() {
  if (!state.pipelineResult) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Business ROI Optimizer Awaiting Champion Model',
      'Fine-tuning decision probability thresholds and calculating profit curves requires an active trained ML model.',
      'dollar-sign'
    );
    lucide.createIcons();
    return;
  }

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

export function renderAutoMLSimulator() {
  const d = state.datasetMeta;
  const p = state.pipelineResult;

  if (!p) {
    renderAutoMLLeaderboard();
    return;
  }

  const bestModel = p.best_model_label;
  const targetCol = p.target_col;
  const isClassification = p.final_task_type === 'classification';

  // Feature values state for simulator
  const featureValues = {};
  const featureTypes = {};

  // Extract features from dataset profile
  if (d.profile && d.profile.columns_info) {
    d.profile.columns_info.forEach((c) => {
      if (c.name !== targetCol) {
        featureTypes[c.name] = c.logical_type;
        if (c.logical_type === 'numeric' || c.logical_type === 'float' || c.logical_type === 'integer') {
          featureValues[c.name] = c.stats ? (c.stats.mean !== undefined ? c.stats.mean : (c.stats.min + c.stats.max) / 2) : 0;
        } else {
          featureValues[c.name] = c.stats && c.stats.top_values && c.stats.top_values.length > 0 ? c.stats.top_values[0].value : 'default';
        }
      }
    });
  }

  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <div class="flex justify-between items-center">
        <div>
          <h3 class="text-base font-semibold text-slate-200 flex items-center">
            <i data-lucide="sliders" class="w-4 h-4 mr-2 text-cyan-400"></i> Interactive What-If Simulator
          </h3>
          <p class="text-xs text-slate-400 mt-0.5">Real-time counterfactual simulation using champion model: <strong class="text-emerald-400">${bestModel}</strong></p>
        </div>
        <span class="text-xs font-mono px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">
          Target: ${targetCol}
        </span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto p-1">
        ${Object.entries(featureValues)
          .map(([name, val]) => {
            const isNum = typeof val === 'number';
            return `
            <div class="p-2.5 bg-slate-900/60 border border-slate-800 rounded-lg">
              <label class="block text-[11px] font-mono text-slate-300 mb-1 truncate" title="${name}">${name}</label>
              ${
                isNum
                  ? `<input type="number" step="any" data-feature="${name}" value="${Number(val).toFixed(2)}" class="w-full glass-input text-xs font-mono sim-input">`
                  : `<input type="text" data-feature="${name}" value="${val}" class="w-full glass-input text-xs font-mono sim-input">`
              }
            </div>
          `;
          })
          .join('')}
      </div>

      <div class="flex items-center space-x-3 pt-2">
        <button id="sim-run-btn" class="py-2 px-5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg flex items-center space-x-2 transition-all">
          <i data-lucide="zap" class="w-3.5 h-3.5"></i>
          <span>Run Real-Time Simulation</span>
        </button>
        <button id="sim-reset-btn" class="py-2 px-3 border border-slate-700 hover:bg-white/5 text-slate-300 text-xs rounded-lg">
          Reset to Defaults
        </button>
      </div>

      <div id="sim-output" class="hidden mt-4 p-4 bg-slate-900/80 border border-slate-800 rounded-xl space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-mono uppercase text-slate-400">Model Prediction</span>
          <span id="sim-prediction" class="text-xl font-bold font-mono text-emerald-400"></span>
        </div>
        <div id="sim-confidence-bar" class="w-full bg-slate-800 rounded-full h-2 overflow-hidden hidden">
          <div id="sim-confidence-fill" class="h-full bg-emerald-400 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
        <div id="sim-confidence-text" class="text-right text-[11px] font-mono text-slate-400"></div>
      </div>
    </div>
  `;

  document.getElementById('sim-run-btn')?.addEventListener('click', async () => {
    const overrides = {};
    document.querySelectorAll('.sim-input').forEach((input) => {
      const feat = input.dataset.feature;
      const raw = input.value;
      overrides[feat] = isNaN(Number(raw)) ? raw : Number(raw);
    });

    const outDiv = document.getElementById('sim-output');
    const predSpan = document.getElementById('sim-prediction');
    outDiv.classList.remove('hidden');
    predSpan.textContent = 'Predicting...';

    try {
      const res = await ApiClient.simulate(state.sessionId, overrides);
      predSpan.textContent = res.prediction;

      if (res.confidence !== undefined && res.confidence !== null) {
        document.getElementById('sim-confidence-bar').classList.remove('hidden');
        const pct = (res.confidence * 100).toFixed(1);
        document.getElementById('sim-confidence-fill').style.width = `${pct}%`;
        document.getElementById('sim-confidence-text').textContent = `Confidence: ${pct}%`;
      }
    } catch (err) {
      predSpan.textContent = `Error: ${err.message}`;
    }
  });

  document.getElementById('sim-reset-btn')?.addEventListener('click', () => {
    renderAutoMLSimulator();
  });

  lucide.createIcons();
}

export async function renderAutoMLActiveLearning() {
  if (!state.pipelineResult) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Active Learning Queue Awaiting Model Predictions',
      'Prioritizing high-entropy borderline samples for expert domain labeling requires predicted class probabilities from a trained model.',
      'user-check'
    );
    lucide.createIcons();
    return;
  }

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


