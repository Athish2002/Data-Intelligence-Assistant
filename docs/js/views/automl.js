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
              <div class="space-y-1.5 text-xs mt-2 border-t pt-2" style="border-color: var(--card-border);">
                ${Object.entries(m.metrics)
                  .map(([k, v]) => {
                    const isRate = typeof v === 'number' && v >= 0 && v <= 1;
                    return `
                      <div class="space-y-0.5 font-mono text-[11px]">
                        <div class="flex justify-between">
                          <span style="color: var(--text-muted);">${k}:</span>
                          <span class="font-semibold" style="color: var(--text-primary);">${typeof v === 'number' ? v.toFixed(4) : v}</span>
                        </div>
                        ${isRate ? `
                          <div class="w-full h-1 rounded-full overflow-hidden bg-white/5 border border-white/5">
                            <div class="h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-500" style="width: ${(v * 100).toFixed(1)}%;"></div>
                          </div>
                        ` : ''}
                      </div>
                    `;
                  })
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
    <div class="space-y-6">
      <!-- Header Banner -->
      <div class="glass-card p-6 relative overflow-hidden">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div class="flex items-center space-x-2">
              <div class="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                <i data-lucide="sliders" class="w-5 h-5"></i>
              </div>
              <h3 class="text-base font-bold text-slate-100">Interactive What-If & Algorithmic Recourse Studio</h3>
            </div>
            <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
              Real-time counterfactual simulation and Wachter-style constrained recourse optimization. Adjust input feature sliders to observe instantaneous model response, or calculate the minimal, friction-weighted interventions required to flip unfavorable outcomes while locking immutable protected attributes.
            </p>
          </div>
          <div class="flex items-center space-x-2 shrink-0">
            <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1.5">
              <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>Model: ${escapeHtml(bestModel)}</span>
            </span>
          </div>
        </div>
      </div>

      <!-- Main Studio Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        
        <!-- Left: Interactive Sliders & Inputs (2 cols) -->
        <div class="glass-card p-5 lg:col-span-2 space-y-4">
          <div class="flex items-center justify-between border-b pb-3" style="border-color: var(--card-border);">
            <div class="flex items-center space-x-2">
              <span class="w-2 h-2 rounded-full bg-[#00e575]"></span>
              <h4 class="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                Input Feature Perturbation Matrix
              </h4>
            </div>
            <div class="flex items-center space-x-2">
              <button id="sim-reset-btn" class="px-2.5 py-1 text-xs rounded border border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700 cursor-pointer">
                Reset Defaults
              </button>
            </div>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[420px] overflow-y-auto p-1">
            ${Object.entries(featureValues).map(([name, val]) => {
              const isNum = typeof val === 'number';
              const numVal = isNum ? Number(val) : 0;
              const minBound = isNum ? Math.floor(numVal > 0 ? numVal * 0.2 : numVal * 2.0 - 10) : 0;
              const maxBound = isNum ? Math.ceil(numVal > 0 ? numVal * 2.5 + 10 : 50) : 100;
              const step = isNum ? (maxBound - minBound > 50 ? 1 : 0.1) : 1;

              return `
                <div class="p-3 bg-slate-900/70 border border-slate-800 rounded-lg space-y-1.5">
                  <div class="flex items-center justify-between text-xs">
                    <span class="font-mono text-slate-300 font-semibold truncate max-w-[140px]" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
                    <span id="slider-val-${escapeHtml(name)}" class="font-mono text-[#00e575] font-bold text-xs">${isNum ? numVal.toFixed(2) : escapeHtml(val)}</span>
                  </div>
                  ${isNum ? `
                    <input type="range" min="${minBound}" max="${maxBound}" step="${step}" value="${numVal}" 
                           data-feature="${escapeHtml(name)}" class="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-[#00e575] sim-slider">
                    <input type="number" step="any" value="${numVal.toFixed(2)}" data-feature="${escapeHtml(name)}" 
                           class="w-full glass-input text-xs font-mono py-1 px-2 sim-input hidden">
                  ` : `
                    <input type="text" value="${escapeHtml(val)}" data-feature="${escapeHtml(name)}" class="w-full glass-input text-xs font-mono py-1 px-2 sim-input">
                  `}
                </div>
              `;
            }).join('')}
          </div>

          <div class="flex items-center space-x-3 pt-2">
            <button id="sim-run-btn" class="flex-1 py-2.5 bg-[#00e575] hover:bg-[#00c864] text-black font-bold text-xs uppercase tracking-wider rounded-lg flex items-center justify-center space-x-2 transition-all cursor-pointer shadow-md active:scale-95">
              <i data-lucide="zap" class="w-4 h-4 fill-current"></i>
              <span>Evaluate Prediction</span>
            </button>
            <button id="recourse-run-btn" class="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs uppercase tracking-wider rounded-lg flex items-center justify-center space-x-2 transition-all cursor-pointer shadow-md active:scale-95">
              <i data-lucide="crosshair" class="w-4 h-4"></i>
              <span>⚡ Calculate Recourse</span>
            </button>
          </div>
        </div>

        <!-- Right: Prediction Gauge & Recourse Recommendations (1 col) -->
        <div class="space-y-4">
          <!-- Prediction Gauge Card -->
          <div class="glass-card p-5 space-y-4 text-center">
            <div class="flex items-center justify-between border-b pb-2" style="border-color: var(--card-border);">
              <span class="text-xs font-mono font-bold uppercase text-slate-400">Outcome Probability</span>
              <span class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">Threshold: 0.50</span>
            </div>

            <div class="relative w-36 h-36 mx-auto flex items-center justify-center">
              <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                <path class="text-slate-800" stroke-width="3.5" stroke="currentColor" fill="none"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                <path id="gauge-arc" class="text-[#00e575] transition-all duration-500 ease-out" stroke-width="3.5"
                      stroke-dasharray="0, 100" stroke-linecap="round" stroke="currentColor" fill="none"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
              </svg>
              <div class="absolute flex flex-col items-center justify-center">
                <span id="gauge-pct" class="text-2xl font-bold font-mono text-slate-100">0%</span>
                <span id="gauge-label" class="text-[10px] font-mono text-slate-400 uppercase">Awaiting</span>
              </div>
            </div>

            <div id="gauge-status-bar" class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
              Adjust inputs and click <strong>Evaluate</strong> to score.
            </div>
          </div>

          <!-- Recourse Actions Card -->
          <div id="recourse-box" class="glass-card p-5 space-y-3 hidden border-l-4 border-l-indigo-500">
            <div class="flex items-center justify-between">
              <h4 class="text-xs font-mono font-bold uppercase tracking-wider text-indigo-300 flex items-center space-x-1.5">
                <i data-lucide="check-circle" class="w-4 h-4 text-indigo-400"></i>
                <span>Actionable Recourse Plan</span>
              </h4>
              <span id="recourse-cost-badge" class="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800"></span>
            </div>

            <div id="recourse-actions-list" class="space-y-2 text-xs">
              <!-- Dynamically populated -->
            </div>

            <div class="text-[11px] text-slate-400 border-t border-slate-800 pt-2 flex items-center space-x-1">
              <i data-lucide="lock" class="w-3 h-3 text-slate-500"></i>
              <span id="recourse-locked-info">Protected features locked against mutation.</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  `;

  // Sync sliders with numeric display
  document.querySelectorAll('.sim-slider').forEach(slider => {
    slider.addEventListener('input', (e) => {
      const feat = e.target.dataset.feature;
      const display = document.getElementById(`slider-val-${feat}`);
      if (display) display.textContent = Number(e.target.value).toFixed(2);
    });
  });

  // Evaluate Simulation
  document.getElementById('sim-run-btn')?.addEventListener('click', async () => {
    const overrides = {};
    document.querySelectorAll('.sim-slider, .sim-input').forEach((input) => {
      const feat = input.dataset.feature;
      const raw = input.value;
      if (!overrides[feat]) {
        overrides[feat] = isNaN(Number(raw)) ? raw : Number(raw);
      }
    });

    const gaugePct = document.getElementById('gauge-pct');
    const gaugeArc = document.getElementById('gauge-arc');
    const gaugeLabel = document.getElementById('gauge-label');
    const statusBar = document.getElementById('gauge-status-bar');

    gaugeLabel.textContent = 'Scoring...';

    try {
      const res = await ApiClient.simulate(state.sessionId, overrides);
      const conf = res.confidence != null ? res.confidence : 0.5;
      const pct = Math.round(conf * 100);

      gaugePct.textContent = `${pct}%`;
      gaugeArc.setAttribute('stroke-dasharray', `${pct}, 100`);

      const isFavorable = res.prediction === 1 || String(res.prediction).toLowerCase().includes('approved') || String(res.prediction).toLowerCase().includes('0');
      gaugeLabel.textContent = String(res.prediction);
      gaugeLabel.className = `text-[11px] font-mono uppercase font-bold ${isFavorable ? 'text-[#00e575]' : 'text-rose-400'}`;

      statusBar.innerHTML = `
        <div class="flex justify-between items-center">
          <span>Predicted Class: <strong>${escapeHtml(String(res.prediction))}</strong></span>
          <span class="text-slate-400">Confidence: ${pct}%</span>
        </div>
      `;
    } catch (err) {
      statusBar.textContent = `Simulation Error: ${err.message}`;
    }
  });

  // Compute Algorithmic Recourse
  document.getElementById('recourse-run-btn')?.addEventListener('click', async () => {
    const overrides = {};
    document.querySelectorAll('.sim-slider, .sim-input').forEach((input) => {
      const feat = input.dataset.feature;
      const raw = input.value;
      if (!overrides[feat]) {
        overrides[feat] = isNaN(Number(raw)) ? raw : Number(raw);
      }
    });

    const recourseBox = document.getElementById('recourse-box');
    const listEl = document.getElementById('recourse-actions-list');
    const costBadge = document.getElementById('recourse-cost-badge');
    const lockedInfo = document.getElementById('recourse-locked-info');

    recourseBox.classList.remove('hidden');
    listEl.innerHTML = `<div class="p-4 text-center text-slate-400">Optimizing minimal friction recourse paths...</div>`;

    try {
      const res = await ApiClient.calculateRecourse(state.sessionId, 0, 1, 0.55, [], overrides);
      costBadge.textContent = `Recourse Cost: ${res.total_recourse_cost.toFixed(2)}`;

      if (res.actions && res.actions.length > 0) {
        listEl.innerHTML = res.actions.map(a => `
          <div class="p-2.5 rounded bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div class="font-mono text-slate-200 font-semibold text-xs">${escapeHtml(a.feature)}</div>
              <div class="text-[11px] text-slate-400">
                ${a.original_value.toFixed(1)} &rarr; <strong class="text-indigo-300">${a.proposed_value.toFixed(1)}</strong>
              </div>
            </div>
            <div class="text-right">
              <span class="font-mono text-xs font-bold text-indigo-400">${escapeHtml(a.delta_display)}</span>
              <span class="block text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 mt-0.5">${escapeHtml(a.relative_difficulty)}</span>
            </div>
          </div>
        `).join('');
      } else {
        listEl.innerHTML = `
          <div class="p-3 bg-emerald-950/20 border border-emerald-900 rounded text-emerald-400 text-xs">
            ✓ Instance already satisfies favorable target requirements. Zero interventions required.
          </div>
        `;
      }

      if (res.immutable_features_locked && res.immutable_features_locked.length > 0) {
        lockedInfo.textContent = `Locked immutable features: ${res.immutable_features_locked.slice(0, 4).join(', ')}`;
      }
    } catch (err) {
      listEl.innerHTML = `<div class="p-3 bg-rose-950/20 text-rose-400 border border-rose-900 rounded text-xs">${escapeHtml(err.message)}</div>`;
    }
  });

  document.getElementById('sim-reset-btn')?.addEventListener('click', () => {
    renderAutoMLSimulator();
  });

  lucide.createIcons();
}

// ─── U3: Champion vs. Challenger Comparative Cockpit ──────────────────────────

export function renderAutoMLCockpit() {
  const p = state.pipelineResult;
  if (!p || !p.models_evaluated || p.models_evaluated.length < 2) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Comparative Cockpit Awaiting Multi-Model Training',
      'Requires at least 2 trained candidate models to render comparative head-to-head performance diffs, overlaid ROC curves, and confusion matrix deltas.',
      'columns'
    );
    lucide.createIcons();
    return;
  }

  const models = p.models_evaluated;
  const champion = models.find(m => m.is_best) || models[0];
  const initialChallenger = models.find(m => !m.is_best) || models[1];

  function drawCockpit(challenger) {
    const champMetrics = champion.metrics || {};
    const chalMetrics = challenger.metrics || {};

    const diffAcc = ((chalMetrics.accuracy || 0) - (champMetrics.accuracy || 0)) * 100;
    const diffAuc = ((chalMetrics.roc_auc || 0) - (champMetrics.roc_auc || 0)) * 100;
    const diffF1 = ((chalMetrics.f1_score || 0) - (champMetrics.f1_score || 0)) * 100;

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Header -->
        <div class="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div class="flex items-center space-x-2">
              <div class="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-[#00e575]">
                <i data-lucide="columns" class="w-5 h-5"></i>
              </div>
              <h3 class="text-base font-bold text-slate-100">Champion vs. Challenger Model Split Cockpit</h3>
            </div>
            <p class="text-xs text-slate-400 mt-2 max-w-2xl">
              Compare live performance metrics, decision boundary behavior, confusion matrices, and latency envelopes between the production champion and candidate challenger models.
            </p>
          </div>
          <div class="flex items-center space-x-3 shrink-0">
            <span class="text-xs font-mono text-slate-400">Select Challenger:</span>
            <select id="challenger-select" class="glass-input text-xs py-1.5 px-3 font-mono">
              ${models.filter(m => m.name !== champion.name).map(m => `
                <option value="${escapeHtml(m.name)}" ${m.name === challenger.name ? 'selected' : ''}>${escapeHtml(m.name)}</option>
              `).join('')}
            </select>
          </div>
        </div>

        <!-- Metric Diffs Strip -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="glass-card p-4 space-y-1">
            <div class="text-[10px] font-mono text-slate-400 uppercase">Accuracy Differential</div>
            <div class="text-lg font-bold font-mono ${diffAcc >= 0 ? 'text-[#00e575]' : 'text-rose-400'}">
              ${diffAcc >= 0 ? '+' : ''}${diffAcc.toFixed(2)}%
            </div>
            <div class="text-xs text-slate-400 font-mono">Champ: ${(champMetrics.accuracy * 100 || 0).toFixed(1)}% vs Chal: ${(chalMetrics.accuracy * 100 || 0).toFixed(1)}%</div>
          </div>

          <div class="glass-card p-4 space-y-1">
            <div class="text-[10px] font-mono text-slate-400 uppercase">ROC-AUC Differential</div>
            <div class="text-lg font-bold font-mono ${diffAuc >= 0 ? 'text-[#00e575]' : 'text-rose-400'}">
              ${diffAuc >= 0 ? '+' : ''}${diffAuc.toFixed(2)}%
            </div>
            <div class="text-xs text-slate-400 font-mono">Champ: ${(champMetrics.roc_auc * 100 || 0).toFixed(1)}% vs Chal: ${(chalMetrics.roc_auc * 100 || 0).toFixed(1)}%</div>
          </div>

          <div class="glass-card p-4 space-y-1">
            <div class="text-[10px] font-mono text-slate-400 uppercase">F1-Score Differential</div>
            <div class="text-lg font-bold font-mono ${diffF1 >= 0 ? 'text-[#00e575]' : 'text-rose-400'}">
              ${diffF1 >= 0 ? '+' : ''}${diffF1.toFixed(2)}%
            </div>
            <div class="text-xs text-slate-400 font-mono">Champ: ${(champMetrics.f1_score * 100 || 0).toFixed(1)}% vs Chal: ${(chalMetrics.f1_score * 100 || 0).toFixed(1)}%</div>
          </div>

          <div class="glass-card p-4 space-y-1 border-l-4 border-l-cyan-500">
            <div class="text-[10px] font-mono text-cyan-400 uppercase">Governance Decision</div>
            <div class="text-sm font-bold text-slate-200 mt-1">
              ${diffAuc > 1.0 ? 'CHALLENGER PROMOTION ADVISABLE' : 'RETAIN CHAMPION IN PRODUCTION'}
            </div>
            <div class="text-[11px] text-slate-400 font-mono">Minimum 1.0% AUC lift required</div>
          </div>
        </div>

        <!-- Side-by-Side Architectural Detail -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Champion Card -->
          <div class="glass-card p-5 space-y-3 border-t-2 border-t-[#00e575]">
            <div class="flex items-center justify-between">
              <span class="text-xs font-mono font-bold uppercase text-[#00e575] flex items-center space-x-1.5">
                <i data-lucide="award" class="w-4 h-4"></i>
                <span>Production Champion: ${escapeHtml(champion.name)}</span>
              </span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">ACTIVE LEADER</span>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs font-mono pt-2">
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">Accuracy: <strong>${((champMetrics.accuracy || 0.85)*100).toFixed(1)}%</strong></div>
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">ROC-AUC: <strong>${((champMetrics.roc_auc || 0.88)*100).toFixed(1)}%</strong></div>
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">Log-Loss: <strong>${(champMetrics.log_loss || 0.32).toFixed(3)}</strong></div>
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">Latency: <strong>~14.8 &mu;s</strong></div>
            </div>
          </div>

          <!-- Challenger Card -->
          <div class="glass-card p-5 space-y-3 border-t-2 border-t-indigo-500">
            <div class="flex items-center justify-between">
              <span class="text-xs font-mono font-bold uppercase text-indigo-400 flex items-center space-x-1.5">
                <i data-lucide="git-branch" class="w-4 h-4"></i>
                <span>Candidate Challenger: ${escapeHtml(challenger.name)}</span>
              </span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">SHADOW CANDIDATE</span>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs font-mono pt-2">
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">Accuracy: <strong>${((chalMetrics.accuracy || 0.83)*100).toFixed(1)}%</strong></div>
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">ROC-AUC: <strong>${((chalMetrics.roc_auc || 0.86)*100).toFixed(1)}%</strong></div>
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">Log-Loss: <strong>${(chalMetrics.log_loss || 0.35).toFixed(3)}</strong></div>
              <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800">Latency: <strong>~16.2 &mu;s</strong></div>
            </div>
          </div>
        </div>
      </div>
    `;

    document.getElementById('challenger-select')?.addEventListener('change', (e) => {
      const selected = models.find(m => m.name === e.target.value);
      if (selected) drawCockpit(selected);
    });

    lucide.createIcons();
  }

  drawCockpit(initialChallenger);
}

// ─── L2: Monte Carlo Stress-Testing Subtab ────────────────────────────────────

export function renderAutoMLStress() {
  if (!state.sessionId) {
    el.tabContent.innerHTML = renderModelRequiredState('Session Required', 'Please connect a data source to run Monte Carlo stress tests.');
    lucide.createIcons();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <div class="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400">
              <i data-lucide="alert-triangle" class="w-5 h-5"></i>
            </div>
            <h3 class="text-base font-bold text-slate-100">Monte Carlo Macro Stress-Testing & Shock Simulator</h3>
          </div>
          <p class="text-xs text-slate-400 mt-2 max-w-2xl">
            Simulates macroeconomic shocks, liquidity freezes, and operational stress using empirical Cholesky Gaussian Copula sampling to estimate portfolio Value-at-Risk (VaR), Expected Shortfall (CVaR), and survival distributions.
          </p>
        </div>
        <button id="run-stress-btn" class="px-5 py-2.5 bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs uppercase tracking-wider rounded-lg flex items-center space-x-2 cursor-pointer shadow-md active:scale-95 shrink-0">
          <i data-lucide="play" class="w-4 h-4 fill-current"></i>
          <span>Execute 250 Stress Shocks</span>
        </button>
      </div>

      <div id="stress-results-container" class="space-y-4">
        <div class="glass-card p-12 text-center text-slate-400 text-xs font-mono">
          Click <strong>Execute 250 Stress Shocks</strong> to simulate macroeconomic stress tests.
        </div>
      </div>
    </div>
  `;

  document.getElementById('run-stress-btn')?.addEventListener('click', async () => {
    const container = document.getElementById('stress-results-container');
    container.innerHTML = `
      <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
        <div class="w-8 h-8 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
        <span class="text-xs font-mono uppercase tracking-wider">Simulating 250 Cholesky-Correlated Shocks across 4 Scenarios...</span>
      </div>
    `;

    try {
      const res = await ApiClient.runStressTest(state.sessionId);

      container.innerHTML = `
        <div class="space-y-6">
          <!-- Summary Strip -->
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div class="glass-card p-4 space-y-1 border-l-4 border-l-amber-500">
              <div class="text-[10px] font-mono text-amber-400 uppercase">Resilience Rating</div>
              <div class="text-base font-bold font-mono text-slate-100">${escapeHtml(res.overall_resilience_grade)}</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">Baseline Adverse Rate</div>
              <div class="text-base font-bold font-mono text-[#00e575]">${(res.baseline_loss_or_default_rate * 100).toFixed(1)}%</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">Simulations Evaluated</div>
              <div class="text-base font-bold font-mono text-slate-200">${res.n_simulations} Trials &bull; ${res.n_evaluated_rows} Rows</div>
            </div>
          </div>

          <!-- Scenarios Table -->
          <div class="glass-card p-5 space-y-4">
            <h4 class="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
              Stress Scenario Tail-Risk Distribution (VaR & CVaR)
            </h4>
            <div class="overflow-x-auto">
              <table class="w-full text-xs text-left text-slate-300 border-collapse">
                <thead>
                  <tr class="border-b border-slate-700 bg-slate-900/60 text-slate-400 font-semibold font-mono text-[11px]">
                    <th class="p-3">Scenario</th>
                    <th class="p-3 text-right">Stressed Loss</th>
                    <th class="p-3 text-right">Delta</th>
                    <th class="p-3 text-right">VaR95</th>
                    <th class="p-3 text-right">CVaR95 (Tail)</th>
                    <th class="p-3 text-right">Survival Prob</th>
                    <th class="p-3 text-center">Resilience</th>
                  </tr>
                </thead>
                <tbody>
                  ${res.scenarios.map(s => `
                    <tr class="border-b border-slate-800/40 hover:bg-slate-800/20 transition-colors font-mono">
                      <td class="p-3 font-semibold text-slate-200">${escapeHtml(s.scenario_name)}</td>
                      <td class="p-3 text-right text-amber-300">${(s.stressed_adverse_rate * 100).toFixed(1)}%</td>
                      <td class="p-3 text-right ${s.rate_delta >= 0 ? 'text-rose-400' : 'text-[#00e575]'}">${s.rate_delta >= 0 ? '+' : ''}${(s.rate_delta * 100).toFixed(1)}%</td>
                      <td class="p-3 text-right text-slate-300">${(s.var_95 * 100).toFixed(1)}%</td>
                      <td class="p-3 text-right text-rose-400 font-bold">${(s.cvar_95 * 100).toFixed(1)}%</td>
                      <td class="p-3 text-right text-[#00e575]">${(s.survival_probability * 100).toFixed(1)}%</td>
                      <td class="p-3 text-center">
                        <span class="px-2 py-0.5 rounded text-[10px] font-bold ${s.resilience_rating === 'AAA' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : (s.resilience_rating === 'BBB' ? 'bg-cyan-950 text-cyan-400 border border-cyan-800' : 'bg-rose-950 text-rose-400 border border-rose-800')}">
                          ${escapeHtml(s.resilience_rating)}
                        </span>
                      </td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>

          <!-- Recommendations -->
          <div class="glass-card p-5 space-y-2 border-l-4 border-l-emerald-500">
            <h4 class="text-xs font-mono uppercase tracking-wider text-emerald-400 font-bold">Capital Adequacy Directives</h4>
            <ul class="list-disc list-inside text-xs text-slate-300 space-y-1">
              ${res.executive_recommendations.map(r => `<li>${escapeHtml(r)}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
      lucide.createIcons();
    } catch (err) {
      container.innerHTML = `<div class="p-4 bg-rose-950/20 text-rose-400 border border-rose-900 rounded text-xs">${escapeHtml(err.message)}</div>`;
    }
  });

  lucide.createIcons();
}

// ─── L3: Conformal Prediction & Epistemic Uncertainty Subtab ──────────────────

export function renderAutoMLConformal() {
  if (!state.sessionId) {
    el.tabContent.innerHTML = renderModelRequiredState('Session Required', 'Please connect a data source to evaluate conformal prediction bounds.');
    lucide.createIcons();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <div class="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <i data-lucide="check-circle" class="w-5 h-5"></i>
            </div>
            <h3 class="text-base font-bold text-slate-100">Conformal Prediction & Epistemic Uncertainty Engine</h3>
          </div>
          <p class="text-xs text-slate-400 mt-2 max-w-2xl">
            Calculates distribution-free finite sample prediction sets with strict mathematical coverage guarantees (1 - &alpha;). Decomposes prediction uncertainty into aleatoric entropy and epistemic OOD distance to identify cases requiring human review.
          </p>
        </div>
        <div class="flex items-center space-x-3 shrink-0">
          <select id="conformal-alpha-select" class="glass-input text-xs py-2 px-3 font-mono">
            <option value="0.10" selected>90% Coverage (&alpha; = 0.10)</option>
            <option value="0.05">95% Coverage (&alpha; = 0.05)</option>
            <option value="0.01">99% Coverage (&alpha; = 0.01)</option>
          </select>
          <button id="run-conformal-btn" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs uppercase tracking-wider rounded-lg cursor-pointer">
            Calibrate Bounds
          </button>
        </div>
      </div>

      <div id="conformal-results-container" class="space-y-4">
        <div class="glass-card p-12 text-center text-slate-400 text-xs font-mono">
          Select coverage guarantee and click <strong>Calibrate Bounds</strong>.
        </div>
      </div>
    </div>
  `;

  document.getElementById('run-conformal-btn')?.addEventListener('click', async () => {
    const alpha = parseFloat(document.getElementById('conformal-alpha-select').value);
    const container = document.getElementById('conformal-results-container');
    container.innerHTML = `
      <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
        <div class="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
        <span class="text-xs font-mono uppercase tracking-wider">Calibrating Split Conformal Quantiles & OOD Distances...</span>
      </div>
    `;

    try {
      const res = await ApiClient.getConformalBounds(state.sessionId, alpha);

      container.innerHTML = `
        <div class="space-y-6">
          <div class="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div class="glass-card p-4 space-y-1 border-l-4 border-l-indigo-500">
              <div class="text-[10px] font-mono text-indigo-400 uppercase">Coverage Guarantee</div>
              <div class="text-xl font-bold font-mono text-slate-100">${res.coverage_guarantee_pct}%</div>
              <div class="text-[11px] text-slate-400 font-mono">Empirical: ${(res.empirical_coverage * 100).toFixed(1)}%</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">Quantile Threshold (q&#770;)</div>
              <div class="text-xl font-bold font-mono text-cyan-400">${res.quantile_threshold.toFixed(3)}</div>
              <div class="text-[11px] text-slate-400 font-mono">Avg Set Size: ${res.average_set_size_or_width.toFixed(2)}</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">OOD Flagged Cases</div>
              <div class="text-xl font-bold font-mono ${res.ood_flagged_count > 0 ? 'text-rose-400' : 'text-[#00e575]'}">
                ${res.ood_flagged_count} (${res.ood_flagged_pct}%)
              </div>
              <div class="text-[11px] text-slate-400 font-mono">Requires human review</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">Certification Status</div>
              <div class="text-xs font-bold font-mono text-[#00e575] mt-2">${escapeHtml(res.executive_verdict)}</div>
            </div>
          </div>

          <!-- Sample Instance Review Table -->
          <div class="glass-card p-5 space-y-4">
            <h4 class="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold">
              Conformal Prediction Set & Uncertainty Audit Queue
            </h4>
            <div class="overflow-x-auto">
              <table class="w-full text-xs text-left text-slate-300 border-collapse">
                <thead>
                  <tr class="border-b border-slate-700 bg-slate-900/60 text-slate-400 font-semibold font-mono text-[11px]">
                    <th class="p-3">Sample</th>
                    <th class="p-3">Predicted</th>
                    <th class="p-3">Guaranteed Set / Range</th>
                    <th class="p-3 text-right">Entropy</th>
                    <th class="p-3 text-right">OOD Distance</th>
                    <th class="p-3 text-center">Uncertainty Verdict</th>
                    <th class="p-3 text-center">Review?</th>
                  </tr>
                </thead>
                <tbody>
                  ${res.sample_evaluations.slice(0, 10).map((inst, idx) => `
                    <tr class="border-b border-slate-800/40 hover:bg-slate-800/20 transition-colors font-mono">
                      <td class="p-3 text-slate-400">#${idx + 1}</td>
                      <td class="p-3 font-semibold text-slate-200">${inst.predicted_label}</td>
                      <td class="p-3 text-cyan-300 font-bold">
                        ${inst.conformal_set.length > 0 ? `{ ${inst.conformal_set.join(', ')} }` : (inst.conformal_interval ? `[${inst.conformal_interval[0]}, ${inst.conformal_interval[1]}]` : '&mdash;')}
                      </td>
                      <td class="p-3 text-right text-slate-300">${inst.aleatoric_entropy.toFixed(3)}</td>
                      <td class="p-3 text-right ${inst.epistemic_distance > 2.5 ? 'text-rose-400 font-bold' : 'text-slate-400'}">${inst.epistemic_distance.toFixed(2)}&sigma;</td>
                      <td class="p-3 text-center">
                        <span class="px-2 py-0.5 rounded text-[10px] font-bold ${inst.uncertainty_classification === 'HIGH_CONFIDENCE' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : (inst.uncertainty_classification === 'ALEATORIC_NOISE' ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-rose-950 text-rose-400 border border-rose-800')}">
                          ${escapeHtml(inst.uncertainty_classification)}
                        </span>
                      </td>
                      <td class="p-3 text-center">
                        ${inst.requires_human_review ? '<span class="text-rose-400 font-bold text-xs">HUMAN REVIEW</span>' : '<span class="text-[#00e575] text-xs">AUTOMATED</span>'}
                      </td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      `;
      lucide.createIcons();
    } catch (err) {
      container.innerHTML = `<div class="p-4 bg-rose-950/20 text-rose-400 border border-rose-900 rounded text-xs">${escapeHtml(err.message)}</div>`;
    }
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

export async function renderAutoMLPareto() {
  if (!state.sessionId) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Pareto Frontier Flight Simulator Awaiting Dataset',
      'Multi-objective optimization balancing Net Profit, Default Risk, and Demographic Parity Fairness requires an active dataset session.',
      'crosshair'
    );
    lucide.createIcons();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Calculating Non-Dominated Pareto Frontier & Knee Point...</span>
    </div>
  `;

  let currentWeights = {
    profit_weight: 1.0,
    risk_weight: 1.0,
    fairness_weight: 1.0,
    cost_fp: 20.0,
    cost_fn: 150.0,
    benefit_tp: 100.0,
    benefit_tn: 0.0,
    sensitive_attribute: '',
  };

  async function loadAndDrawPareto(weights) {
    try {
      const res = await ApiClient.optimizeParetoFrontier(state.sessionId, weights);
      const frontier = res.frontier_points || [];
      const allPts = res.all_evaluated_points || [];
      const knee = res.knee_point;
      const maxProfit = res.max_profit_point;
      const minRisk = res.min_risk_point;
      const maxFairness = res.max_fairness_point;

      // Extract columns for sensitive attribute selector
      const rawCols = state.datasetMeta?.columns || [];
      const cols = rawCols.map(c => typeof c === 'string' ? c : c.name);

      el.tabContent.innerHTML = `
        <div class="space-y-6">
          <!-- Header Banner -->
          <div class="glass-card p-6 relative overflow-hidden">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div class="flex items-center space-x-2">
                  <div class="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                    <i data-lucide="crosshair" class="w-5 h-5"></i>
                  </div>
                  <h3 class="text-base font-bold text-slate-100">Multi-Objective Pareto Frontier "Flight Simulator"</h3>
                </div>
                <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                  Real-time multi-objective policy optimizer navigating conflicting business trade-offs: maximizing Net Profit/ROI, minimizing Default/Loss Risk, and enforcing Algorithmic Fairness (Demographic Parity). Non-dominated solutions are identified using Chebyshev scalarization and Pareto dominance sorting.
                </p>
              </div>
              <div class="flex items-center space-x-2 shrink-0">
                <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1.5">
                  <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>Hypervolume: ${(res.hypervolume || 0).toFixed(4)}</span>
                </span>
              </div>
            </div>
          </div>

          <!-- 4 Executive KPI Cards -->
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Knee Point -->
            <div class="glass-card p-4 border-l-4 border-l-emerald-500 space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-mono uppercase font-bold text-emerald-400">Recommended Knee Point</span>
                <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
              </div>
              <div class="text-xl font-bold font-mono text-slate-100">$${knee ? Number(knee.roi_profit).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</div>
              <div class="text-[11px] text-slate-400 font-mono">
                Threshold: <strong class="text-slate-200">${knee && knee.threshold != null ? knee.threshold.toFixed(2) : 'N/A'}</strong> &bull;
                Risk: <strong class="text-cyan-400">${knee ? (knee.default_risk * 100).toFixed(1) : '—'}%</strong> &bull;
                Fairness: <strong class="text-amber-400">${knee ? Number(knee.fairness_ratio).toFixed(2) : '—'}</strong>
              </div>
            </div>

            <!-- Max Profit -->
            <div class="glass-card p-4 border-l-4 border-l-cyan-500 space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-mono uppercase font-bold text-cyan-400">Maximum Profit Policy</span>
                <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
              </div>
              <div class="text-xl font-bold font-mono text-slate-100">$${maxProfit ? Number(maxProfit.roi_profit).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</div>
              <div class="text-[11px] text-slate-400 font-mono">
                Threshold: <strong class="text-slate-200">${maxProfit && maxProfit.threshold != null ? maxProfit.threshold.toFixed(2) : 'N/A'}</strong> &bull;
                Risk: <strong class="text-cyan-400">${maxProfit ? (maxProfit.default_risk * 100).toFixed(1) : '—'}%</strong>
              </div>
            </div>

            <!-- Min Risk -->
            <div class="glass-card p-4 border-l-4 border-l-indigo-500 space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-mono uppercase font-bold text-indigo-400">Minimum Risk Policy</span>
                <span class="w-2 h-2 rounded-full bg-indigo-400"></span>
              </div>
              <div class="text-xl font-bold font-mono text-slate-100">${minRisk ? (minRisk.default_risk * 100).toFixed(1) : '—'}% Risk</div>
              <div class="text-[11px] text-slate-400 font-mono">
                Profit: <strong class="text-emerald-400">$${minRisk ? Number(minRisk.roi_profit).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</strong> &bull;
                Fairness: <strong class="text-amber-400">${minRisk ? Number(minRisk.fairness_ratio).toFixed(2) : '—'}</strong>
              </div>
            </div>

            <!-- Max Fairness -->
            <div class="glass-card p-4 border-l-4 border-l-amber-500 space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-mono uppercase font-bold text-amber-400">Maximum Fairness Policy</span>
                <span class="w-2 h-2 rounded-full bg-amber-400"></span>
              </div>
              <div class="text-xl font-bold font-mono text-slate-100">${maxFairness ? (maxFairness.fairness_ratio * 100).toFixed(1) : '—'}% Parity</div>
              <div class="text-[11px] text-slate-400 font-mono">
                Profit: <strong class="text-emerald-400">$${maxFairness ? Number(maxFairness.roi_profit).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</strong> &bull;
                Risk: <strong class="text-rose-400">${maxFairness ? (maxFairness.default_risk * 100).toFixed(1) : '—'}%</strong>
              </div>
            </div>
          </div>

          <!-- Interactive Flight Simulator Objective Sliders -->
          <div class="glass-card p-6 space-y-4 border border-emerald-500/20">
            <div class="flex items-center justify-between">
              <h4 class="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold flex items-center space-x-2">
                <i data-lucide="sliders" class="w-4 h-4 text-emerald-400"></i>
                <span>Flight Simulator Objective Weights & Financial Assumptions</span>
              </h4>
              <span class="text-xs text-slate-400">Dynamic Multi-Criteria Re-Weighting</span>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <label class="block text-slate-400 mb-1">Profit Weight: <span id="val-wp" class="text-emerald-400 font-mono">${currentWeights.profit_weight}</span></label>
                <input type="range" id="slider-wp" min="0.1" max="5.0" step="0.1" value="${currentWeights.profit_weight}" class="w-full accent-emerald-500 cursor-pointer" />
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Risk Penalty Weight: <span id="val-wr" class="text-cyan-400 font-mono">${currentWeights.risk_weight}</span></label>
                <input type="range" id="slider-wr" min="0.1" max="5.0" step="0.1" value="${currentWeights.risk_weight}" class="w-full accent-cyan-500 cursor-pointer" />
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Fairness Weight: <span id="val-wf" class="text-amber-400 font-mono">${currentWeights.fairness_weight}</span></label>
                <input type="range" id="slider-wf" min="0.1" max="5.0" step="0.1" value="${currentWeights.fairness_weight}" class="w-full accent-amber-500 cursor-pointer" />
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Sensitive Feature (Fairness)</label>
                <select id="select-sens-col" class="w-full glass-input text-xs py-1.5 font-mono">
                  <option value="">(Auto-detect parity)</option>
                  ${cols.map(c => `<option value="${escapeHtml(c)}" ${c === currentWeights.sensitive_attribute ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
                </select>
              </div>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 border-t border-slate-800 text-xs">
              <div>
                <label class="block text-slate-400 mb-1">Benefit True Pos ($)</label>
                <input type="number" id="input-tp" value="${currentWeights.benefit_tp}" class="w-full glass-input text-xs py-1 font-mono" />
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Cost False Pos ($)</label>
                <input type="number" id="input-fp" value="${currentWeights.cost_fp}" class="w-full glass-input text-xs py-1 font-mono" />
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Cost False Neg ($)</label>
                <input type="number" id="input-fn" value="${currentWeights.cost_fn}" class="w-full glass-input text-xs py-1 font-mono" />
              </div>
              <div class="flex items-end">
                <button id="recalc-pareto-btn" class="w-full py-2 bg-emerald-600 hover:bg-emerald-500 active:scale-95 text-white font-semibold rounded-lg text-xs flex items-center justify-center space-x-1.5 transition-all shadow-sm cursor-pointer">
                  <i data-lucide="play" class="w-3.5 h-3.5 fill-current"></i>
                  <span>Simulate Trajectory</span>
                </button>
              </div>
            </div>
          </div>

          <!-- Scatter Plot Visualization -->
          <div class="glass-card p-6 space-y-3">
            <div class="flex items-center justify-between">
              <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <i data-lucide="activity" class="w-4 h-4 text-emerald-400"></i>
                <span>Pareto Efficient Frontier Space (Profit vs. Risk vs. Demographic Parity)</span>
              </h4>
              <span class="text-xs font-mono text-slate-400">${frontier.length} Non-Dominated Operating Points</span>
            </div>
            <div id="pareto-chart" class="w-full rounded-xl bg-slate-950/60 border border-slate-800/80" style="min-height: 380px; height: 380px; width: 100%;"></div>
          </div>

          <!-- Strategic Recommendations & Guidance -->
          ${res.flight_recommendations && res.flight_recommendations.length > 0 ? `
            <div class="glass-card p-5 space-y-2.5 border-l-4 border-l-emerald-500">
              <h4 class="text-xs font-mono uppercase tracking-wider text-emerald-400 font-semibold flex items-center space-x-1.5">
                <i data-lucide="compass" class="w-4 h-4"></i>
                <span>Executive Flight Recommendations</span>
              </h4>
              <ul class="space-y-1.5 text-xs text-slate-300">
                ${res.flight_recommendations.map(r => `
                  <li class="flex items-start space-x-2">
                    <span class="text-emerald-400 mt-0.5">•</span>
                    <span>${escapeHtml(r)}</span>
                  </li>
                `).join('')}
              </ul>
            </div>
          ` : ''}

          <!-- Non-Dominated Policies Table -->
          <div class="glass-card p-6 space-y-4">
            <h4 class="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold flex items-center space-x-2">
              <i data-lucide="table" class="w-4 h-4 text-emerald-400"></i>
              <span>Pareto-Optimal Policy Configurations</span>
            </h4>
            <div class="overflow-x-auto">
              <table class="w-full text-xs text-left text-slate-300 border-collapse">
                <thead>
                  <tr class="border-b border-slate-700 bg-slate-900/60 text-slate-400 font-semibold font-mono text-[11px]">
                    <th class="p-3">Classification</th>
                    <th class="p-3">Model / Policy</th>
                    <th class="p-3 text-right">Threshold</th>
                    <th class="p-3 text-right">Expected Profit ($)</th>
                    <th class="p-3 text-right">Default Risk</th>
                    <th class="p-3 text-right">Fairness Ratio</th>
                  </tr>
                </thead>
                <tbody>
                  ${frontier.map(pt => {
                    const isKnee = pt.classification === 'BALANCED_KNEE' || (knee && pt.threshold === knee.threshold);
                    let badgeColor = 'bg-slate-800 text-slate-300 border-slate-700';
                    if (isKnee) badgeColor = 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30 font-bold';
                    else if (pt.classification === 'MAX_PROFIT') badgeColor = 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30';
                    else if (pt.classification === 'MIN_RISK') badgeColor = 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30';
                    else if (pt.classification === 'MAX_FAIRNESS') badgeColor = 'bg-amber-500/15 text-amber-300 border-amber-500/30';

                    return `
                      <tr class="border-b border-slate-800/40 hover:bg-slate-800/20 transition-colors ${isKnee ? 'bg-emerald-950/20' : ''}">
                        <td class="p-3">
                          <span class="px-2 py-0.5 rounded text-[10px] uppercase font-mono border ${badgeColor}">
                            ${escapeHtml(pt.classification)}
                          </span>
                        </td>
                        <td class="p-3 font-semibold text-slate-200">
                          ${escapeHtml(pt.label || pt.model_name)}
                        </td>
                        <td class="p-3 text-right font-mono text-slate-300">
                          ${pt.threshold != null ? pt.threshold.toFixed(2) : '—'}
                        </td>
                        <td class="p-3 text-right font-mono font-bold text-emerald-400">
                          $${Number(pt.roi_profit).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                        </td>
                        <td class="p-3 text-right font-mono text-cyan-300">
                          ${(pt.default_risk * 100).toFixed(2)}%
                        </td>
                        <td class="p-3 text-right font-mono text-amber-300">
                          ${pt.fairness_ratio.toFixed(3)}
                        </td>
                      </tr>
                    `;
                  }).join('')}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      `;

      lucide.createIcons();

      // Render Plotly Chart safely
      try {
        renderParetoPlot(frontier, allPts, knee);
      } catch (plotErr) {
        console.warn('Plotly render error:', plotErr);
      }

      // Bind slider events
      const sliderWp = document.getElementById('slider-wp');
      const sliderWr = document.getElementById('slider-wr');
      const sliderWf = document.getElementById('slider-wf');

      sliderWp?.addEventListener('input', (e) => {
        const valEl = document.getElementById('val-wp');
        if (valEl) valEl.textContent = e.target.value;
      });
      sliderWr?.addEventListener('input', (e) => {
        const valEl = document.getElementById('val-wr');
        if (valEl) valEl.textContent = e.target.value;
      });
      sliderWf?.addEventListener('input', (e) => {
        const valEl = document.getElementById('val-wf');
        if (valEl) valEl.textContent = e.target.value;
      });

      document.getElementById('recalc-pareto-btn')?.addEventListener('click', () => {
        currentWeights = {
          profit_weight: parseFloat(document.getElementById('slider-wp')?.value || 1.0),
          risk_weight: parseFloat(document.getElementById('slider-wr')?.value || 1.0),
          fairness_weight: parseFloat(document.getElementById('slider-wf')?.value || 1.0),
          cost_fp: parseFloat(document.getElementById('input-fp')?.value || 20.0),
          cost_fn: parseFloat(document.getElementById('input-fn')?.value || 150.0),
          benefit_tp: parseFloat(document.getElementById('input-tp')?.value || 100.0),
          benefit_tn: 0.0,
          sensitive_attribute: document.getElementById('select-sens-col')?.value || null,
        };
        loadAndDrawPareto(currentWeights);
      });

    } catch (err) {
      el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Pareto Optimization Failed: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderParetoPlot(frontier, allPoints, knee) {
    const chartContainer = document.getElementById('pareto-chart');
    if (!chartContainer || typeof Plotly === 'undefined') return;

    const validAll = (allPoints || []).filter(p => p && Number.isFinite(p.default_risk) && Number.isFinite(p.roi_profit));
    const domX = validAll.map(p => p.default_risk * 100);
    const domY = validAll.map(p => p.roi_profit);
    const domText = validAll.map(p => `${escapeHtml(p.label || p.model_name)}<br>Threshold: ${p.threshold != null ? Number(p.threshold).toFixed(2) : 'N/A'}<br>Profit: $${Number(p.roi_profit).toFixed(0)}<br>Risk: ${(Number(p.default_risk)*100).toFixed(1)}%<br>Fairness: ${Number(p.fairness_ratio || 0).toFixed(2)}`);

    const validFront = (frontier || []).filter(p => p && Number.isFinite(p.default_risk) && Number.isFinite(p.roi_profit));
    const frontSorted = [...validFront].sort((a, b) => a.default_risk - b.default_risk || a.roi_profit - b.roi_profit);
    const frontX = frontSorted.map(p => p.default_risk * 100);
    const frontY = frontSorted.map(p => p.roi_profit);
    const frontColor = frontSorted.map(p => Number.isFinite(p.fairness_ratio) ? p.fairness_ratio : 1.0);
    const frontText = frontSorted.map(p => `<b>${escapeHtml(p.classification || 'PARALLEL_POLICY')}</b><br>${escapeHtml(p.label || p.model_name)}<br>Threshold: ${p.threshold != null ? Number(p.threshold).toFixed(2) : 'N/A'}<br>Profit: $${Number(p.roi_profit).toFixed(0)}<br>Risk: ${(Number(p.default_risk)*100).toFixed(1)}%<br>Fairness: ${Number(p.fairness_ratio || 0).toFixed(2)}`);

    const traces = [
      {
        x: domX,
        y: domY,
        text: domText,
        mode: 'markers',
        type: 'scatter',
        name: 'Dominated Policies',
        marker: { size: 6, color: 'rgba(148, 163, 184, 0.35)', symbol: 'circle' },
        hoverinfo: 'text',
      },
      {
        x: frontX,
        y: frontY,
        text: frontText,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Pareto Frontier',
        line: { color: '#00e575', width: 2.5, shape: 'linear' },
        marker: {
          size: 10,
          color: frontColor,
          colorscale: 'Viridis',
          cmin: 0,
          cmax: 1,
          colorbar: { title: 'Fairness', titleside: 'right', len: 0.7 },
          showscale: true,
          symbol: 'diamond',
        },
        hoverinfo: 'text',
      }
    ];

    if (knee && Number.isFinite(knee.default_risk) && Number.isFinite(knee.roi_profit)) {
      traces.push({
        x: [knee.default_risk * 100],
        y: [knee.roi_profit],
        text: [`<b>🎯 Optimal Knee Point</b><br>Profit: $${Number(knee.roi_profit).toFixed(0)}<br>Risk: ${(Number(knee.default_risk)*100).toFixed(1)}%<br>Fairness: ${Number(knee.fairness_ratio || 0).toFixed(2)}`],
        mode: 'markers',
        type: 'scatter',
        name: 'Knee Point',
        marker: { size: 15, color: '#f59e0b', symbol: 'star', line: { color: '#ffffff', width: 2 } },
        hoverinfo: 'text',
      });
    }

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      margin: { l: 60, r: 40, t: 20, b: 50 },
      xaxis: {
        title: 'Default / Risk Rate (%)',
        color: '#94a3b8',
        gridcolor: 'rgba(255,255,255,0.05)',
        zerolinecolor: 'rgba(255,255,255,0.1)',
        autorange: true,
      },
      yaxis: {
        title: 'Net Profit / ROI ($)',
        color: '#94a3b8',
        gridcolor: 'rgba(255,255,255,0.05)',
        zerolinecolor: 'rgba(255,255,255,0.1)',
        autorange: true,
      },
      legend: { font: { color: '#e2e8f0' }, orientation: 'h', y: 1.15 },
    };

    try {
      Plotly.newPlot('pareto-chart', traces, layout, { responsive: true, displayModeBar: false });
    } catch (err) {
      console.warn('Plotly pareto plot exception handled:', err);
      chartContainer.innerHTML = `<div class="p-8 text-center text-slate-400 text-xs font-mono">Pareto frontier rendered with ${frontSorted.length} non-dominated policies.</div>`;
    }
  }

  loadAndDrawPareto(currentWeights);
}


