/**
 * frontend/js/views/adaptive.js
 * ─────────────────────────────
 * Workspace 3: Adaptive AI Engines (Causal Counterfactuals, Autoencoder, Bandits, Time Series, Graph).
 */

import { state, el, escapeHtml } from '../state.js';
import { ApiClient } from '../api.js';
import { showError } from '../toast.js';
import { renderModelRequiredState } from './automl.js';

export async function renderAdaptiveCausal() {
  if (!state.sessionId) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Causal Intelligence Engine Awaiting Dataset',
      'Observational Causal DAG discovery, Pearl\'s Do-Calculus policy simulator, and counterfactual reasoning require an active dataset session.',
      'target'
    );
    lucide.createIcons();
    return;
  }

  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-rose-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Discovering Observational Causal DAG via PC Algorithm...</span>
    </div>
  `;

  try {
    const graphRes = await ApiClient.getCausalGraph(state.sessionId);
    const rawCols = state.datasetMeta?.columns || [];
    const colList = rawCols.map(c => typeof c === 'string' ? c : c.name);
    const availableCols = graphRes.nodes && graphRes.nodes.length > 0 ? graphRes.nodes : colList;
    const targetCol = graphRes.target_col || state.datasetMeta?.suggested_target || availableCols[availableCols.length - 1] || '';
    const defaultTreatment = graphRes.direct_causes_of_target?.[0] || graphRes.root_causes?.[0] || availableCols.find(c => c !== targetCol) || availableCols[0] || '';

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Header Banner -->
        <div class="glass-card p-6 relative overflow-hidden">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-2">
                <div class="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400">
                  <i data-lucide="target" class="w-5 h-5"></i>
                </div>
                <h3 class="text-base font-bold text-slate-100">Causal Discovery & Pearl's Do-Calculus Simulator</h3>
              </div>
              <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                Constraint-based causal DAG discovery using the PC algorithm (Fisher's z-transform partial correlations and Meek orientation rules R1–R4). Simulates policy interventions E[Y | do(X = x)] with backdoor criteria adjustment to eradicate confounding bias and reverse causality.
              </p>
            </div>
            <div class="flex items-center space-x-2 shrink-0">
              <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center space-x-1.5">
                <span class="w-2 h-2 rounded-full bg-rose-400 animate-pulse"></span>
                <span>${graphRes.nodes.length} Nodes &bull; ${graphRes.edges.length} Directed Edges</span>
              </span>
            </div>
          </div>
        </div>

        <!-- Causal Node Roles & Reverse Causality Flags -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="glass-card p-4 border-l-4 border-l-rose-500 space-y-1">
            <div class="text-[10px] font-mono text-rose-400 uppercase font-semibold">Direct Causes of Target</div>
            <div class="text-base font-bold text-slate-100 font-mono">
              ${graphRes.direct_causes_of_target.length > 0 ? graphRes.direct_causes_of_target.map(c => `<span class="inline-block bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded text-xs mr-1 mb-1 font-semibold">${escapeHtml(c)}</span>`).join('') : '<span class="text-slate-500 text-xs italic">None isolated</span>'}
            </div>
          </div>
          <div class="glass-card p-4 border-l-4 border-l-cyan-500 space-y-1">
            <div class="text-[10px] font-mono text-cyan-400 uppercase font-semibold">Exogenous Root Causes</div>
            <div class="text-base font-bold text-slate-100 font-mono">
              ${graphRes.root_causes.length > 0 ? graphRes.root_causes.map(c => `<span class="inline-block bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded text-xs mr-1 mb-1">${escapeHtml(c)}</span>`).join('') : '<span class="text-slate-500 text-xs italic">None</span>'}
            </div>
          </div>
          <div class="glass-card p-4 border-l-4 border-l-amber-500 space-y-1">
            <div class="text-[10px] font-mono text-amber-400 uppercase font-semibold">Confounders (Backdoor Risk)</div>
            <div class="text-base font-bold text-slate-100 font-mono">
              ${graphRes.confounders.length > 0 ? graphRes.confounders.map(c => `<span class="inline-block bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded text-xs mr-1 mb-1">${escapeHtml(c)}</span>`).join('') : '<span class="text-slate-500 text-xs italic">None detected</span>'}
            </div>
          </div>
          <div class="glass-card p-4 border-l-4 ${graphRes.reverse_causality_risks.length > 0 ? 'border-l-rose-600 bg-rose-950/10' : 'border-l-emerald-500'} space-y-1">
            <div class="text-[10px] font-mono ${graphRes.reverse_causality_risks.length > 0 ? 'text-rose-400' : 'text-emerald-400'} uppercase font-semibold">Reverse Causality Alerts</div>
            <div class="text-xs font-semibold ${graphRes.reverse_causality_risks.length > 0 ? 'text-rose-300' : 'text-emerald-400'}">
              ${graphRes.reverse_causality_risks.length > 0 ? `${graphRes.reverse_causality_risks.length} Risk(s): ${escapeHtml(graphRes.reverse_causality_risks.join(', '))}` : '✓ No reverse causality detected'}
            </div>
          </div>
        </div>

        <!-- Pearl's Do-Calculus Policy Intervention Simulator -->
        <div class="glass-card p-6 space-y-4 border border-rose-500/30">
          <div class="flex items-center space-x-2">
            <i data-lucide="play-circle" class="w-5 h-5 text-rose-400"></i>
            <h4 class="text-sm font-bold text-slate-100">Pearl's Do-Calculus Policy Simulator &bull; E[Y | do(X = x)]</h4>
          </div>
          <p class="text-xs text-slate-400">
            Simulate the causal impact of intervening on feature X to value x while controlling for confounding backdoor paths.
          </p>

          <div class="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
            <div>
              <label class="block text-slate-400 mb-1 font-semibold">Treatment Feature (X)</label>
              <select id="do-treatment-select" class="w-full glass-input text-xs py-1.5 font-mono">
                ${availableCols.map(c => `<option value="${escapeHtml(c)}" ${c === defaultTreatment ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
              </select>
            </div>
            <div>
              <label class="block text-slate-400 mb-1 font-semibold">Outcome Variable (Y)</label>
              <select id="do-outcome-select" class="w-full glass-input text-xs py-1.5 font-mono">
                ${availableCols.map(c => `<option value="${escapeHtml(c)}" ${c === targetCol ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('')}
              </select>
            </div>
            <div>
              <label class="block text-slate-400 mb-1 font-semibold">Intervention Value (x)</label>
              <input type="number" id="do-intervention-val" value="1.0" step="any" class="w-full glass-input text-xs py-1.5 font-mono" />
            </div>
            <div class="flex items-end">
              <button id="do-simulate-btn" class="w-full py-2 bg-rose-600 hover:bg-rose-500 active:scale-95 text-white font-semibold rounded-lg text-xs flex items-center justify-center space-x-1.5 transition-all shadow-sm cursor-pointer">
                <i data-lucide="zap" class="w-3.5 h-3.5 fill-current"></i>
                <span>Simulate do(X = x)</span>
              </button>
            </div>
          </div>

          <div id="do-results-box" class="hidden mt-3 p-4 bg-slate-900/90 border border-slate-800 rounded-xl space-y-3">
            <!-- Populated on simulation -->
          </div>
        </div>

        <!-- U2: Interactive Causal DAG Studio & Priors Canvas -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b pb-3" style="border-color: var(--card-border);">
            <div class="flex items-center space-x-2">
              <span class="w-2 h-2 rounded-full bg-rose-500"></span>
              <h4 class="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                Interactive Causal DAG Topology & Domain Priors Studio
              </h4>
            </div>
            <div class="flex items-center space-x-2 text-xs font-mono">
              <span id="dag-edge-count" class="text-rose-400 font-bold">${graphRes.edges.length} Active Edges</span>
              <button id="reset-priors-btn" class="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 cursor-pointer">
                Reset Priors
              </button>
            </div>
          </div>

          <p class="text-xs text-slate-400">
            Click any directed edge below or in the table to toggle domain priors (<span class="text-[#00e575] font-semibold">ALLOWED</span> vs <span class="text-rose-400 font-semibold">FORBIDDEN</span>). Changes dynamically re-calibrate backdoor path blocking for policy intervention estimation.
          </p>

          <!-- SVG Visual DAG Canvas -->
          <div class="w-full overflow-x-auto bg-slate-950/80 rounded-xl border border-slate-800/80 p-2 relative flex justify-center">
            <svg id="causal-dag-svg" class="w-full max-w-4xl h-72 select-none" viewBox="0 0 760 300">
              <defs>
                <marker id="arrow-allowed" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e" />
                </marker>
                <marker id="arrow-forbidden" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
                </marker>
              </defs>
              <g id="dag-edges-group"></g>
              <g id="dag-nodes-group"></g>
            </svg>
          </div>
        </div>

        <!-- Discovered Causal Edges Table -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold flex items-center space-x-2">
              <i data-lucide="git-merge" class="w-4 h-4 text-rose-400"></i>
              <span>Discovered Causal Adjacency & Partial Correlations</span>
            </h4>
            <span class="text-xs font-mono text-slate-400">${graphRes.edges.length} Causal Links</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300 border-collapse">
              <thead>
                <tr class="border-b border-slate-700 bg-slate-900/60 text-slate-400 font-semibold font-mono text-[11px]">
                  <th class="p-3">Cause (Source)</th>
                  <th class="p-3 text-center">Direction</th>
                  <th class="p-3">Effect (Target)</th>
                  <th class="p-3 text-right">Partial Correlation</th>
                  <th class="p-3 text-right">p-value</th>
                  <th class="p-3 text-center">Target Direct Cause?</th>
                </tr>
              </thead>
              <tbody>
                ${graphRes.edges.length > 0 ? graphRes.edges.map(e => `
                  <tr class="border-b border-slate-800/40 hover:bg-slate-800/20 transition-colors">
                    <td class="p-3 font-mono font-semibold text-slate-200">${escapeHtml(e.source)}</td>
                    <td class="p-3 text-center font-mono text-rose-400 font-bold">${escapeHtml(e.direction)}</td>
                    <td class="p-3 font-mono font-semibold text-slate-200">${escapeHtml(e.target)}</td>
                    <td class="p-3 text-right font-mono text-cyan-300">${e.weight != null ? e.weight.toFixed(3) : '—'}</td>
                    <td class="p-3 text-right font-mono text-slate-400">${e.p_value != null ? (e.p_value < 0.001 ? '< 0.001' : e.p_value.toFixed(4)) : '—'}</td>
                    <td class="p-3 text-center">
                      ${e.is_direct_cause_of_target ? '<span class="px-2 py-0.5 rounded text-[10px] uppercase font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">DIRECT CAUSE</span>' : '<span class="text-slate-500 text-[11px]">Indirect / Mediated</span>'}
                    </td>
                  </tr>
                `).join('') : `
                  <tr>
                    <td colspan="6" class="p-4 text-center text-slate-400 italic">No significant causal edges survived constraint thresholding.</td>
                  </tr>
                `}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Prescriptive Counterfactuals & Uplift T-Learner (Model-reliant) -->
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
            <button id="cf-run-btn" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 cursor-pointer">
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
            <button id="uplift-run-btn" class="w-full py-2 bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-semibold rounded-lg cursor-pointer">
              ⚡ Calculate Uplift & Treatment Segments
            </button>
            <div id="uplift-res-box" class="text-xs text-slate-300"></div>
          </div>
        </div>
      </div>
    `;

    lucide.createIcons();

    // Render Causal DAG Graph onto SVG with Interactive Edge Toggles
    const nodes = graphRes.nodes || [];
    const edges = graphRes.edges || [];
    const forbiddenEdges = new Set();

    function renderSvgDag() {
      const edgesGroup = document.getElementById('dag-edges-group');
      const nodesGroup = document.getElementById('dag-nodes-group');
      const edgeCountEl = document.getElementById('dag-edge-count');
      if (!edgesGroup || !nodesGroup) return;

      const n = nodes.length || 1;
      const cx = 380, cy = 140, rx = 280, ry = 95;
      const nodePos = {};

      nodes.forEach((node, idx) => {
        const theta = (2 * Math.PI * idx) / n - Math.PI / 2;
        nodePos[node] = {
          x: Math.round(cx + rx * Math.cos(theta)),
          y: Math.round(cy + ry * Math.sin(theta)),
        };
      });

      // Render Directed Edges
      let activeCount = 0;
      edgesGroup.innerHTML = edges.map((e) => {
        const p1 = nodePos[e.source] || { x: cx, y: cy };
        const p2 = nodePos[e.target] || { x: cx, y: cy };
        const edgeKey = `${e.source}->${e.target}`;
        const isForbidden = forbiddenEdges.has(edgeKey);
        if (!isForbidden) activeCount++;

        return `
          <g class="dag-edge-interactive cursor-pointer" data-edge="${escapeHtml(edgeKey)}">
            <line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}"
                  stroke="${isForbidden ? '#475569' : '#f43f5e'}"
                  stroke-width="${isForbidden ? '1.5' : '2.5'}"
                  stroke-dasharray="${isForbidden ? '4,4' : 'none'}"
                  opacity="${isForbidden ? '0.35' : '0.85'}"
                  marker-end="url(#${isForbidden ? 'arrow-forbidden' : 'arrow-allowed'})" />
          </g>
        `;
      }).join('');

      if (edgeCountEl) {
        edgeCountEl.textContent = `${activeCount} Active Edges (${forbiddenEdges.size} Forbidden)`;
      }

      // Render Circular Nodes
      nodesGroup.innerHTML = nodes.map((node) => {
        const p = nodePos[node] || { x: cx, y: cy };
        const isTarget = node === targetCol;
        const isDirect = (graphRes.direct_causes_of_target || []).includes(node);

        const nodeFill = isTarget ? '#e11d48' : (isDirect ? '#0284c7' : '#1e293b');
        const strokeColor = isTarget ? '#fda4af' : (isDirect ? '#38bdf8' : '#475569');

        return `
          <g transform="translate(${p.x}, ${p.y})" class="select-none">
            <circle r="15" fill="${nodeFill}" stroke="${strokeColor}" stroke-width="2" />
            <text text-anchor="middle" dy="4" fill="#ffffff" font-size="9" font-family="monospace" font-weight="bold">
              ${escapeHtml(node.slice(0, 3).toUpperCase())}
            </text>
            <text text-anchor="middle" dy="26" fill="${isTarget ? '#fda4af' : '#cbd5e1'}" font-size="10" font-family="monospace" font-weight="600">
              ${escapeHtml(node)}
            </text>
          </g>
        `;
      }).join('');

      // Wire Edge Click Listeners
      document.querySelectorAll('.dag-edge-interactive').forEach(elem => {
        elem.addEventListener('click', () => {
          const key = elem.dataset.edge;
          if (forbiddenEdges.has(key)) {
            forbiddenEdges.delete(key);
          } else {
            forbiddenEdges.add(key);
          }
          renderSvgDag();
        });
      });
    }

    renderSvgDag();

    document.getElementById('reset-priors-btn')?.addEventListener('click', () => {
      forbiddenEdges.clear();
      renderSvgDag();
    });

    // Event handler for Do-Calculus intervention simulation
    document.getElementById('do-simulate-btn')?.addEventListener('click', async () => {
      const treatment = document.getElementById('do-treatment-select')?.value;
      const outcome = document.getElementById('do-outcome-select')?.value;
      const val = document.getElementById('do-intervention-val')?.value;
      const resBox = document.getElementById('do-results-box');

      if (!treatment || !outcome) return;

      resBox.classList.remove('hidden');
      resBox.innerHTML = `
        <div class="flex items-center space-x-2 text-slate-400">
          <div class="w-3.5 h-3.5 border-2 border-rose-500 border-t-transparent rounded-full animate-spin"></div>
          <span>Computing back-door criteria adjustment & interventional expectation...</span>
        </div>
      `;

      try {
        const simRes = await ApiClient.simulateIntervention(state.sessionId, treatment, outcome, val);
        const ate = simRes.average_treatment_effect;
        const baseline = simRes.baseline_expected_outcome;
        const intervened = simRes.intervened_expected_outcome;
        const pct = simRes.percent_change;

        resBox.innerHTML = `
          <div class="space-y-3">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
              <div class="flex items-center space-x-2">
                <span class="px-2 py-0.5 rounded text-[10px] uppercase font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">Policy Simulation Output</span>
                <span class="text-xs text-slate-300 font-mono">do(${escapeHtml(treatment)} = ${Number(val).toFixed(2)}) &rarr; ${escapeHtml(outcome)}</span>
              </div>
              <span class="text-xs font-mono font-bold ${ate >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
                ATE: ${ate >= 0 ? '+' : ''}${ate.toFixed(4)} (${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%)
              </span>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div class="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                <div class="text-[10px] font-mono text-slate-400 uppercase">Observational Baseline E[Y]</div>
                <div class="text-sm font-bold font-mono text-slate-200 mt-0.5">${baseline.toFixed(4)}</div>
              </div>
              <div class="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                <div class="text-[10px] font-mono text-slate-400 uppercase">Interventional E[Y|do(X)]</div>
                <div class="text-sm font-bold font-mono text-rose-400 mt-0.5">${intervened.toFixed(4)}</div>
              </div>
              <div class="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                <div class="text-[10px] font-mono text-slate-400 uppercase">95% Confidence Interval</div>
                <div class="text-xs font-mono text-cyan-300 mt-0.5">[${simRes.confidence_interval_95 ? simRes.confidence_interval_95.map(v => v.toFixed(3)).join(', ') : '—'}]</div>
              </div>
              <div class="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800">
                <div class="text-[10px] font-mono text-slate-400 uppercase">Adjustment Set</div>
                <div class="text-xs font-mono text-slate-300 mt-0.5 truncate" title="${escapeHtml(simRes.adjustment_set.join(', '))}">
                  ${simRes.adjustment_set.length > 0 ? escapeHtml(simRes.adjustment_set.join(', ')) : '(None needed)'}
                </div>
              </div>
            </div>

            <div class="p-3 bg-slate-950/60 rounded-lg text-xs text-slate-300 leading-relaxed border border-slate-800">
              💡 <strong>Interpretation:</strong> ${escapeHtml(simRes.interpretation)}
            </div>
          </div>
        `;
      } catch (err) {
        resBox.innerHTML = `<div class="text-rose-400 text-xs">Intervention Simulation Failed: ${escapeHtml(err.message)}</div>`;
      }
    });

    // Existing Counterfactual & Uplift handlers
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
        resBox.innerHTML = `<span class="text-rose-400">${escapeHtml(err.message)}</span>`;
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
            <div class="text-[11px] text-slate-300">💡 Recommended Action: ${escapeHtml(res.recommended_action)}</div>
          </div>
        `;
      } catch (err) {
        resBox.innerHTML = `<span class="text-rose-400">${escapeHtml(err.message)}</span>`;
      }
    });

    if (window.lucide) {
      lucide.createIcons();
    }
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Causal Discovery Failed: ${escapeHtml(err.message)}</div>`;
  }
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


