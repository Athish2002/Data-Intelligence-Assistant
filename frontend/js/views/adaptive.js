/**
 * frontend/js/views/adaptive.js
 * ─────────────────────────────
 * Workspace 3: Adaptive AI Engines (Causal Counterfactuals, Autoencoder, Bandits, Time Series, Graph).
 */

import { state, el } from '../state.js';
import { ApiClient } from '../api.js';
import { showError } from '../toast.js';
import { renderModelRequiredState } from './automl.js';

export function renderAdaptiveCausal() {
  if (!state.pipelineResult) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Causal Counterfactuals Awaiting Champion Model',
      'Prescriptive counterfactual search and T-Learner uplift modeling require an active trained ML model to simulate intervention shifts.',
      'target'
    );
    lucide.createIcons();
    return;
  }

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

export async function renderAdaptiveAutoencoder() {
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

export async function renderAdaptiveBandits() {
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

export function renderAdaptiveSynthetic() {
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

export async function renderAdaptiveOnline() {
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

export async function renderAdaptiveTimeSeries() {
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

export async function renderAdaptiveNlp() {
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

export async function renderAdaptiveGraph() {
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


