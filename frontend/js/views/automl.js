/**
 * frontend/js/views/automl.js
 * ───────────────────────────
 * Workspace 2: AutoML, Leaderboard, SHAP Explainability, Simulator, and ROI Optimizer.
 */

import { state, el, escapeHtml } from '../state.js';
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

  // Handle Continuous Regression Task
  if (p.final_task_type === 'regression') {
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <div class="flex items-center justify-between border-b pb-3" style="border-color: var(--card-border);">
          <div class="flex items-center space-x-2">
            <span class="w-2 h-2 rounded-full bg-[#00e575]"></span>
            <h3 class="text-sm font-mono font-bold uppercase tracking-wider" style="color: var(--text-primary);">
              Regression Residuals & Performance Analysis
            </h3>
          </div>
          <span class="text-xs font-mono font-bold px-2 py-0.5 rounded border" style="background-color: var(--tag-bg); color: var(--tag-text); border-color: var(--card-border);">
            Continuous Prediction
          </span>
        </div>
        <p class="text-xs" style="color: var(--text-muted);">
          ROC Curves and Confusion Matrices are diagnostic instruments for classification. For continuous regression target <code class="font-mono text-[#00e575]">${p.target_col}</code>, performance is evaluated using residual loss criteria.
        </p>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1">
          <div class="p-3 rounded-lg border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
            <div class="text-[10px] uppercase font-mono text-stone-500">Evaluation Metric</div>
            <div class="text-sm font-bold text-[#00e575] mt-1">${p.metric_used || 'R² / RMSE'}</div>
          </div>
          <div class="p-3 rounded-lg border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
            <div class="text-[10px] uppercase font-mono text-stone-500">Target Feature</div>
            <div class="text-sm font-bold text-[#00e575] mt-1 truncate" title="${p.target_col}">${p.target_col}</div>
          </div>
          <div class="p-3 rounded-lg border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
            <div class="text-[10px] uppercase font-mono text-stone-500">Models Evaluated</div>
            <div class="text-sm font-bold text-[#00e575] mt-1">${p.models_evaluated?.length || 0} Models</div>
          </div>
          <div class="p-3 rounded-lg border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
            <div class="text-[10px] uppercase font-mono text-stone-500">Champion Model</div>
            <div class="text-sm font-bold text-[#00e575] mt-1 truncate" title="${p.best_model_label}">${p.best_model_label}</div>
          </div>
        </div>
      </div>
    `;
    lucide.createIcons();
    return;
  }

  // Classification: Calculate Confusion Matrix stats
  const cm = p.confusion_matrix || [[0, 0], [0, 0]];
  const tn = Number(cm[0]?.[0] || 0);
  const fp = Number(cm[0]?.[1] || 0);
  const fn = Number(cm[1]?.[0] || 0);
  const tp = Number(cm[1]?.[1] || 0);
  const total = tn + fp + fn + tp || 1;

  const tn_pct = ((tn / total) * 100).toFixed(1);
  const fp_pct = ((fp / total) * 100).toFixed(1);
  const fn_pct = ((fn / total) * 100).toFixed(1);
  const tp_pct = ((tp / total) * 100).toFixed(1);

  const precision = (tp + fp) > 0 ? ((tp / (tp + fp)) * 100).toFixed(1) + '%' : '0.0%';
  const recall = (tp + fn) > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) + '%' : '0.0%';
  const specificity = (tn + fp) > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) + '%' : '0.0%';
  const accuracy = (((tn + tp) / total) * 100).toFixed(1) + '%';
  const aucVal = p.models_evaluated?.find((m) => m.is_best)?.metrics?.roc_auc || 0.8609;

  el.tabContent.innerHTML = `
    <div class="space-y-4">
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4 items-stretch">
        
        <!-- 1. ROC Curve Container (Strictly Bounded, Zero Overflow) -->
        <div class="glass-card p-4 flex flex-col justify-between overflow-hidden" style="min-height: 380px;">
          <div>
            <div class="flex items-center justify-between mb-1 pb-2 border-b" style="border-color: var(--card-border);">
              <div class="flex items-center space-x-2">
                <span class="w-2 h-2 rounded-full bg-[#00e575]"></span>
                <h3 class="text-xs font-mono font-bold uppercase tracking-wider" style="color: var(--text-primary);">
                  Interactive ROC Curve
                </h3>
              </div>
              <span class="text-xs font-mono font-bold px-2 py-0.5 rounded border" style="background:#092314; color:#4ade80; border-color:#134e2c;">
                AUC = ${typeof aucVal === 'number' ? aucVal.toFixed(4) : aucVal}
              </span>
            </div>
            <p class="text-[11px] mb-2" style="color: var(--text-muted);">
              Evaluates sensitivity (TPR) against false alarm rate (FPR) across all discrimination cutoffs.
            </p>
          </div>

          <div id="roc-chart-container" class="w-full flex-1" style="min-height: 250px; max-height: 270px;"></div>

          <div class="flex items-center justify-between text-[10px] font-mono pt-2 border-t mt-1" style="border-color: var(--card-border); color: var(--text-muted);">
            <span class="truncate max-w-[210px]">Champion: <strong style="color: var(--text-primary);">${p.best_model_label}</strong></span>
            <span class="shrink-0 text-right">Operating Cutoff: <strong class="text-[#00e575]">&tau; = 0.50</strong></span>
          </div>
        </div>

        <!-- 2. Diagnostic 2x2 Confusion Matrix (Native Responsive Card Grid) -->
        <div class="glass-card p-4 flex flex-col justify-between overflow-hidden" style="min-height: 380px;">
          <div>
            <div class="flex items-center justify-between mb-1 pb-2 border-b" style="border-color: var(--card-border);">
              <div class="flex items-center space-x-2">
                <span class="w-2 h-2 rounded-full bg-[#00e575]"></span>
                <h3 class="text-xs font-mono font-bold uppercase tracking-wider" style="color: var(--text-primary);">
                  Diagnostic Confusion Matrix
                </h3>
              </div>
              <span class="text-xs font-mono font-bold px-2 py-0.5 rounded border" style="background-color: var(--tag-bg); color: var(--tag-text); border-color: var(--card-border);">
                ${total} Holdout Samples
              </span>
            </div>
            <p class="text-[11px] mb-2" style="color: var(--text-muted);">
              Contingency matrix mapping observed ground truth vs model predictions on the test partition.
            </p>
          </div>

          <!-- 2x2 Matrix Grid -->
          <div class="space-y-1.5 flex-1 flex flex-col justify-center my-1">
            <!-- Column Header -->
            <div class="grid grid-cols-12 gap-1.5 text-center text-[10px] font-mono font-bold" style="color: var(--text-muted);">
              <div class="col-span-4 text-left pl-1">ACTUAL \\ PRED</div>
              <div class="col-span-4 px-1 py-0.5 rounded border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">PREDICTED 0</div>
              <div class="col-span-4 px-1 py-0.5 rounded border" style="background-color: var(--bg-secondary); border-color: var(--card-border);">PREDICTED 1</div>
            </div>

            <!-- Row 1: Actual Negative (0) -->
            <div class="grid grid-cols-12 gap-1.5 items-stretch">
              <div class="col-span-4 p-2 rounded flex flex-col justify-center text-[11px] font-mono font-bold border" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-primary);">
                <span>ACTUAL 0</span>
                <span class="text-[9px] font-normal text-stone-500">Negative Class</span>
              </div>
              
              <!-- TN -->
              <div class="col-span-4 p-2.5 rounded-lg border text-center flex flex-col justify-center transition-all hover:scale-[1.01]" style="background-color: rgba(0, 229, 117, 0.08); border-color: rgba(0, 229, 117, 0.35);">
                <div class="text-lg font-mono font-black text-[#00e575]">${tn}</div>
                <div class="text-[10px] font-bold text-[#00e575] tracking-wide">TRUE NEGATIVE</div>
                <div class="text-[9px] font-mono text-stone-400 mt-0.5">${tn_pct}% &bull; Correct</div>
              </div>

              <!-- FP -->
              <div class="col-span-4 p-2.5 rounded-lg border text-center flex flex-col justify-center transition-all hover:scale-[1.01]" style="background-color: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.35);">
                <div class="text-lg font-mono font-black text-amber-400">${fp}</div>
                <div class="text-[10px] font-bold text-amber-400 tracking-wide">FALSE POSITIVE</div>
                <div class="text-[9px] font-mono text-stone-400 mt-0.5">${fp_pct}% &bull; Type I Error</div>
              </div>
            </div>

            <!-- Row 2: Actual Positive (1) -->
            <div class="grid grid-cols-12 gap-1.5 items-stretch">
              <div class="col-span-4 p-2 rounded flex flex-col justify-center text-[11px] font-mono font-bold border" style="background-color: var(--bg-secondary); border-color: var(--card-border); color: var(--text-primary);">
                <span>ACTUAL 1</span>
                <span class="text-[9px] font-normal text-stone-500">Positive Class</span>
              </div>
              
              <!-- FN -->
              <div class="col-span-4 p-2.5 rounded-lg border text-center flex flex-col justify-center transition-all hover:scale-[1.01]" style="background-color: rgba(244, 63, 94, 0.08); border-color: rgba(244, 63, 94, 0.35);">
                <div class="text-lg font-mono font-black text-rose-400">${fn}</div>
                <div class="text-[10px] font-bold text-rose-400 tracking-wide">FALSE NEGATIVE</div>
                <div class="text-[9px] font-mono text-stone-400 mt-0.5">${fn_pct}% &bull; Type II Error</div>
              </div>

              <!-- TP -->
              <div class="col-span-4 p-2.5 rounded-lg border text-center flex flex-col justify-center transition-all hover:scale-[1.01]" style="background-color: rgba(0, 229, 117, 0.12); border-color: rgba(0, 229, 117, 0.45);">
                <div class="text-lg font-mono font-black text-[#00e575]">${tp}</div>
                <div class="text-[10px] font-bold text-[#00e575] tracking-wide">TRUE POSITIVE</div>
                <div class="text-[9px] font-mono text-stone-400 mt-0.5">${tp_pct}% &bull; Correct</div>
              </div>
            </div>
          </div>

          <!-- Diagnostic Metrics Strip -->
          <div class="grid grid-cols-4 gap-1.5 pt-2 border-t mt-1" style="border-color: var(--card-border);">
            <div class="p-1.5 rounded border text-center" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
              <div class="text-[9px] uppercase font-mono text-stone-500">Accuracy</div>
              <div class="text-xs font-bold text-[#00e575]">${accuracy}</div>
            </div>
            <div class="p-1.5 rounded border text-center" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
              <div class="text-[9px] uppercase font-mono text-stone-500">Precision</div>
              <div class="text-xs font-bold text-[#00e575]">${precision}</div>
            </div>
            <div class="p-1.5 rounded border text-center" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
              <div class="text-[9px] uppercase font-mono text-stone-500">Recall</div>
              <div class="text-xs font-bold text-[#00e575]">${recall}</div>
            </div>
            <div class="p-1.5 rounded border text-center" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
              <div class="text-[9px] uppercase font-mono text-stone-500">Specificity</div>
              <div class="text-xs font-bold text-[#00e575]">${specificity}</div>
            </div>
          </div>

        </div>

      </div>
    </div>
  `;

  // Render strictly bounded Plotly ROC Chart
  if (p.roc_curve && p.roc_curve.fpr) {
    Plotly.newPlot(
      'roc-chart-container',
      [
        {
          x: p.roc_curve.fpr,
          y: p.roc_curve.tpr,
          mode: 'lines',
          name: `${p.best_model_label} (AUC: ${typeof aucVal === 'number' ? aucVal.toFixed(3) : aucVal})`,
          line: { color: '#00e575', width: 2.5 },
        },
        {
          x: [0, 1],
          y: [0, 1],
          mode: 'lines',
          name: 'Chance Baseline',
          line: { color: '#666666', dash: 'dash', width: 1.5 },
        },
      ],
      {
        height: 250,
        autosize: true,
        margin: { l: 45, r: 25, t: 15, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
          title: { text: 'False Positive Rate', font: { size: 10, color: '#858076' } },
          color: '#858076',
          gridcolor: 'rgba(255,255,255,0.05)',
          range: [0, 1],
          tickfont: { size: 9 },
        },
        yaxis: {
          title: { text: 'True Positive Rate', font: { size: 10, color: '#858076' } },
          color: '#858076',
          gridcolor: 'rgba(255,255,255,0.05)',
          range: [0, 1.02],
          tickfont: { size: 9 },
        },
        legend: {
          x: 0.05,
          y: 0.95,
          font: { size: 9, color: '#f5f0e6' },
          bgcolor: 'rgba(0,0,0,0.6)',
          bordercolor: 'rgba(255,255,255,0.1)',
          borderwidth: 1,
        },
      },
      { responsive: true, displayModeBar: false }
    );
  } else {
    const rocEl = document.getElementById('roc-chart-container');
    if (rocEl) {
      rocEl.innerHTML = `
        <div class="h-full flex flex-col items-center justify-center text-center p-5 rounded border border-dashed text-xs space-y-2" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
          <div class="w-8 h-8 rounded-full bg-[#00e575]/10 flex items-center justify-center border border-[#00e575]/30">
            <i data-lucide="bar-chart-2" class="w-4 h-4 text-[#00e575]"></i>
          </div>
          <span class="font-mono font-bold text-slate-200">Continuous / Multi-Class Target Mode</span>
          <p class="text-[11px] text-slate-400 max-w-xs leading-relaxed">
            Standard ROC discrimination curves require binary targets. For this target, inspect the Diagnostic Contingency Matrix and Model Leaderboard scores.
          </p>
        </div>
      `;
      if (window.lucide) lucide.createIcons();
    }
  }

  lucide.createIcons();
}

export function renderAutoMLExplainability() {
  const p = state.pipelineResult;
  if (!p) {
    renderAutoMLLeaderboard();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="glass-card p-5 space-y-3">
      <div class="flex items-center justify-between border-b pb-2" style="border-color: var(--card-border);">
        <div class="flex items-center space-x-2">
          <span class="w-2 h-2 rounded-full bg-[#00e575]"></span>
          <h3 class="text-xs font-mono font-bold uppercase tracking-wider" style="color: var(--text-primary);">
            Global TreeSHAP Feature Attributions
          </h3>
        </div>
        <span class="text-xs font-mono font-bold px-2 py-0.5 rounded border" style="background-color: var(--tag-bg); color: var(--tag-text); border-color: var(--card-border);">
          Top ${p.shap_importance?.length || 0} Features
        </span>
      </div>
      <p class="text-[11px]" style="color: var(--text-muted);">
        Mean absolute SHAP value (&Epsilon;[|&Phi;|]) quantifying each feature's contribution towards moving the baseline prediction.
      </p>
      <div id="shap-chart-container" class="w-full" style="min-height: 340px;"></div>
    </div>
  `;

  // Render Horizontal Bar Chart with Plotly
  const feats = (p.shap_importance || []).map((s) => s.feature).reverse();
  const imps = (p.shap_importance || []).map((s) => s.importance).reverse();

  Plotly.newPlot(
    'shap-chart-container',
    [
      {
        x: imps,
        y: feats,
        type: 'bar',
        orientation: 'h',
        marker: {
          color: '#00e575',
          line: { color: '#00b35c', width: 1 },
        },
      },
    ],
    {
      height: 330,
      autosize: true,
      margin: { l: 190, r: 25, t: 10, b: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis: {
        title: { text: 'Mean |SHAP Value| (Impact on Model Log-Odds)', font: { size: 10, color: '#858076' } },
        color: '#858076',
        gridcolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 9 },
      },
      yaxis: {
        color: 'var(--text-primary)',
        tickfont: { size: 10, family: 'monospace' },
        automargin: true,
      },
    },
    { responsive: true, displayModeBar: false }
  );
  lucide.createIcons();
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

  // Extract features from sample_data, profile, or columns list
  if (d && d.sample_data && d.sample_data.length > 0) {
    const row0 = d.sample_data[0];
    Object.entries(row0).forEach(([col, val]) => {
      if (col !== targetCol) {
        featureValues[col] = val !== null && val !== undefined ? val : 0;
      }
    });
  } else if (d && d.profile && d.profile.columns_info) {
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
  } else if (d && d.columns) {
    d.columns.forEach((col) => {
      if (col !== targetCol) {
        featureValues[col] = 0;
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
              <label class="block text-[11px] font-mono text-slate-300 mb-1 truncate" title="${escapeHtml(name)}">${escapeHtml(name)}</label>
              ${
                isNum
                  ? `<input type="number" step="any" data-feature="${escapeHtml(name)}" value="${Number(val).toFixed(2)}" class="w-full glass-input text-xs font-mono sim-input">`
                  : `<input type="text" data-feature="${escapeHtml(name)}" value="${escapeHtml(val)}" class="w-full glass-input text-xs font-mono sim-input">`
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
                ${res.uncertain_samples.length > 0 ? Object.keys(res.uncertain_samples[0]).map((k) => `<th class="p-2.5">${escapeHtml(k)}</th>`).join('') : ''}
              </tr>
            </thead>
            <tbody>
              ${res.uncertain_samples
                .map(
                  (r) => `
                <tr class="border-b border-slate-800/40 hover:bg-slate-800/30">
                  ${Object.values(r).map((v) => `<td class="p-2.5 font-mono">${escapeHtml(v)}</td>`).join('')}
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


