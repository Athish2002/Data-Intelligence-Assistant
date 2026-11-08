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
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b pb-3" style="border-color: var(--card-border);">
            <div class="flex items-center space-x-2">
              <span class="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse"></span>
              <h4 class="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                Interactive Causal DAG Topology & Invariant Studio
              </h4>
            </div>
            <div class="flex flex-wrap items-center gap-2 text-xs font-mono">
              <span id="dag-edge-count" class="text-rose-400 font-bold">${graphRes.edges.length} Active Edges</span>
              <button id="clear-conditioning-btn" class="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 active:scale-95 cursor-pointer transition-all">
                Clear Conditioning Set
              </button>
              <button id="reset-priors-btn" class="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 active:scale-95 cursor-pointer transition-all">
                Reset Priors
              </button>
            </div>
          </div>

          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
            <p class="text-slate-400">
              Click any node to toggle into <strong>Conditioning Set Z</strong> (visual d-separation & collider detection). Click directed edges to toggle domain priors (<span class="text-[#00e575] font-semibold">ALLOWED</span> vs <span class="text-slate-400 font-semibold">FORBIDDEN</span>).
            </p>
            <div id="conditioning-status-badge" class="px-3 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] font-mono font-medium flex items-center space-x-1.5 shrink-0">
              <span class="text-slate-400">Conditioning Set Z:</span>
              <span id="conditioning-set-display" class="text-cyan-400 font-bold">&empty; (Unconditioned)</span>
            </div>
          </div>

          <!-- Dynamic Berkson's Bias / Collider Warning Shelf -->
          <div id="berkson-warning-container" class="space-y-2"></div>

          <!-- SVG Visual DAG Canvas with Active/Blocked Path Highlighting -->
          <div class="w-full overflow-x-auto bg-slate-950/80 rounded-xl border border-slate-800/80 p-2 relative flex justify-center">
            <svg id="causal-dag-svg" class="w-full max-w-4xl h-76 select-none" viewBox="0 0 760 300">
              <defs>
                <marker id="arrow-allowed" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
                </marker>
                <marker id="arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
                </marker>
                <marker id="arrow-rose" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e" />
                </marker>
                <marker id="arrow-blocked" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />
                </marker>
                <marker id="arrow-forbidden" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155" />
                </marker>
                <filter id="glow-z" x="-50%" y="-50%" width="200%" height="200%">
                  <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#10b981" flood-opacity="0.8"/>
                </filter>
              </defs>
              <g id="dag-edges-group"></g>
              <g id="dag-nodes-group"></g>
            </svg>
          </div>
        </div>

        <!-- Axiomatic Conservation Laws & Balance Invariants Shelf -->
        <div class="glass-card p-6 space-y-4" id="invariants-shelf-card">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b pb-3" style="border-color: var(--card-border);">
            <div class="flex items-center space-x-2">
              <div class="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                <i data-lucide="shield-check" class="w-5 h-5"></i>
              </div>
              <div>
                <h4 class="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <span>Axiomatic Conservation Laws & Balance Invariants</span>
                  <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">CLOSED MANIFOLD</span>
                </h4>
                <p class="text-xs text-slate-400 mt-0.5">
                  Discover, verify, and lock physical conservation laws (Kirchhoff / balance Ax = b) and projection manifolds to enforce strict invariance in projections.
                </p>
              </div>
            </div>
            <div class="flex items-center space-x-2 shrink-0">
              <button id="scan-invariants-btn" class="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 active:scale-95 text-white font-semibold rounded-lg text-xs flex items-center space-x-1.5 transition-all shadow-sm cursor-pointer font-mono">
                <i data-lucide="search" class="w-3.5 h-3.5"></i>
                <span>Scan Conservation Laws</span>
              </button>
            </div>
          </div>

          <div id="invariants-cards-container" class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="col-span-full p-6 text-center text-slate-500 text-xs font-mono border border-dashed border-slate-800 rounded-xl">
              Click <strong>[Scan Conservation Laws]</strong> to audit dataset features against closed linear balance axioms, probability simplex invariants, and monotonicity constraints.
            </div>
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

    // Render Interactive Causal DAG Graph with Visual d-Separation & Invariant Studio
    const nodes = graphRes.nodes || [];
    const edges = graphRes.edges || [];
    const forbiddenEdges = new Set();
    const conditioningSet = new Set();

    // Helper: Find all simple paths between start and target in DAG
    function findAllSimplePaths(start, target, maxDepth = 6) {
      const activeEdgesList = edges.filter(e => !forbiddenEdges.has(`${e.source}->${e.target}`));
      const adj = {};
      nodes.forEach(n => { adj[n] = []; });
      activeEdgesList.forEach(e => {
        if (!adj[e.source]) adj[e.source] = [];
        if (!adj[e.target]) adj[e.target] = [];
        adj[e.source].push({ neighbor: e.target, direction: 'forward', edge: e });
        adj[e.target].push({ neighbor: e.source, direction: 'backward', edge: e });
      });

      const paths = [];
      function dfs(curr, targetNode, visited, currentPath) {
        if (currentPath.length > maxDepth) return;
        if (curr === targetNode) {
          paths.push([...currentPath]);
          return;
        }
        for (const nextHop of (adj[curr] || [])) {
          if (!visited.has(nextHop.neighbor)) {
            visited.add(nextHop.neighbor);
            currentPath.push(nextHop);
            dfs(nextHop.neighbor, targetNode, visited, currentPath);
            currentPath.pop();
            visited.delete(nextHop.neighbor);
          }
        }
      }

      visitedSet = new Set([start]);
      dfs(start, target, visitedSet, []);
      return paths;
    }

    // Helper: Determine if a trail is active given conditioning set Z
    function isTrailActive(pathNodes, condSet) {
      if (pathNodes.length <= 2) return true;
      for (let i = 1; i < pathNodes.length - 1; i++) {
        const prev = pathNodes[i - 1];
        const curr = pathNodes[i];
        const next = pathNodes[i + 1];

        // Is curr a collider on this path? (prev -> curr and next -> curr)
        const isCollider = edges.some(e => !forbiddenEdges.has(`${e.source}->${e.target}`) && e.source === prev && e.target === curr) &&
                           edges.some(e => !forbiddenEdges.has(`${e.source}->${e.target}`) && e.source === next && e.target === curr);

        if (isCollider) {
          // Collider is active iff curr in Z or descendant of curr in Z
          if (!condSet.has(curr)) {
            return false;
          }
        } else {
          // Non-collider: active iff NOT in Z
          if (condSet.has(curr)) {
            return false;
          }
        }
      }
      return true;
    }

    function renderSvgDag() {
      const edgesGroup = document.getElementById('dag-edges-group');
      const nodesGroup = document.getElementById('dag-nodes-group');
      const edgeCountEl = document.getElementById('dag-edge-count');
      const condDisplayEl = document.getElementById('conditioning-set-display');
      const berksonContainer = document.getElementById('berkson-warning-container');
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

      // 1. Detect colliders in conditioning set & compute Berkson's Bias Warnings
      const activeEdgesList = edges.filter(e => !forbiddenEdges.has(`${e.source}->${e.target}`));
      const parentMap = {};
      nodes.forEach(node => { parentMap[node] = []; });
      activeEdgesList.forEach(e => {
        if (!parentMap[e.target]) parentMap[e.target] = [];
        parentMap[e.target].push(e.source);
      });

      const berksonWarnings = [];
      conditioningSet.forEach(condNode => {
        const parents = parentMap[condNode] || [];
        if (parents.length >= 2) {
          const parentA = parents[0];
          const parentB = parents[1];
          berksonWarnings.push({
            node: condNode,
            parentA: parentA,
            parentB: parentB,
            message: `⚠️ Berkson's Bias Warning: Conditioning on collider '${condNode}' opens a spurious association between '${parentA}' and '${parentB}'!`,
          });
        }
      });

      if (berksonContainer) {
        if (berksonWarnings.length > 0) {
          berksonContainer.innerHTML = berksonWarnings.map(w => `
            <div class="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-300 text-xs font-mono flex items-center space-x-2 animate-fadeIn">
              <span class="text-amber-400 font-bold shrink-0 text-sm">⚠️</span>
              <span><strong>Berkson's Bias Warning:</strong> Conditioning on collider '<strong>${escapeHtml(w.node)}</strong>' opens a spurious association between '<strong>${escapeHtml(w.parentA)}</strong>' and '<strong>${escapeHtml(w.parentB)}</strong>'!</span>
            </div>
          `).join('');
        } else {
          berksonContainer.innerHTML = '';
        }
      }

      // Update conditioning set badge display
      if (condDisplayEl) {
        if (conditioningSet.size === 0) {
          condDisplayEl.innerHTML = `&empty; <span class="font-normal text-slate-500">(Unconditioned)</span>`;
          condDisplayEl.className = 'text-cyan-400 font-bold';
        } else {
          condDisplayEl.innerHTML = `{ ${Array.from(conditioningSet).map(c => `<span class="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">${escapeHtml(c)}</span>`).join(', ')} }`;
          condDisplayEl.className = 'text-emerald-400 font-bold flex items-center gap-1 flex-wrap';
        }
      }

      // 2. Compute Active vs Blocked Paths across all edges
      // An edge is active if it lies on an unblocked active path (e.g. from treatment/causes to target,
      // or opened via collider conditioning). If conditioning blocks all through-paths, edge becomes dimmed slate.
      const activePathEdges = new Set();
      const blockedPathEdges = new Set();

      edges.forEach(e => {
        const edgeKey = `${e.source}->${e.target}`;
        if (forbiddenEdges.has(edgeKey)) {
          return;
        }

        // Check if edge is directly incident to a conditioned non-collider
        const sourceCond = conditioningSet.has(e.source);
        const targetCond = conditioningSet.has(e.target);

        // Target is a collider if it has >= 2 parents
        const isTargetCollider = (parentMap[e.target] || []).length >= 2;

        if (targetCond && isTargetCollider) {
          // Conditioning on collider opens the incoming paths!
          activePathEdges.add(edgeKey);
        } else if (sourceCond || targetCond) {
          // Conditioned non-collider blocks transmission
          blockedPathEdges.add(edgeKey);
        } else {
          // By default, unconditioned edges on causal flow are active
          activePathEdges.add(edgeKey);
        }
      });

      // Render Directed Edges with distinct active vs blocked styling
      let activeCount = 0;
      edgesGroup.innerHTML = edges.map((e) => {
        const p1 = nodePos[e.source] || { x: cx, y: cy };
        const p2 = nodePos[e.target] || { x: cx, y: cy };
        const edgeKey = `${e.source}->${e.target}`;
        const isForbidden = forbiddenEdges.has(edgeKey);
        const isBlocked = !isForbidden && blockedPathEdges.has(edgeKey) && !activePathEdges.has(edgeKey);
        const isActive = !isForbidden && !isBlocked;

        if (isActive) activeCount++;

        let strokeColor = '#10b981'; // Bold emerald for active path
        let strokeWidth = '2.5';
        let strokeDash = 'none';
        let opacity = '0.9';
        let marker = 'url(#arrow-active)';

        if (isForbidden) {
          strokeColor = '#334155';
          strokeWidth = '1.5';
          strokeDash = '4,4';
          opacity = '0.35';
          marker = 'url(#arrow-forbidden)';
        } else if (isBlocked) {
          strokeColor = '#475569';
          strokeWidth = '1.5';
          strokeDash = '4,4';
          opacity = '0.45';
          marker = 'url(#arrow-blocked)';
        }

        return `
          <g class="dag-edge-interactive cursor-pointer" data-edge="${escapeHtml(edgeKey)}" title="${isForbidden ? 'Forbidden by Domain Prior' : (isBlocked ? 'Path Blocked by Conditioning Set Z' : 'Active Causal Path')}">
            <line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}"
                  stroke="${strokeColor}"
                  stroke-width="${strokeWidth}"
                  stroke-dasharray="${strokeDash}"
                  opacity="${opacity}"
                  marker-end="${marker}" />
          </g>
        `;
      }).join('');

      if (edgeCountEl) {
        edgeCountEl.textContent = `${activeCount} Active Paths (${forbiddenEdges.size} Forbidden, ${blockedPathEdges.size} Blocked)`;
      }

      // Render Circular Nodes with Conditioning Badges & Highlights
      nodesGroup.innerHTML = nodes.map((node) => {
        const p = nodePos[node] || { x: cx, y: cy };
        const isConditioned = conditioningSet.has(node);
        const isTarget = node === targetCol;
        const isDirect = (graphRes.direct_causes_of_target || []).includes(node);

        let nodeFill = isTarget ? '#e11d48' : (isDirect ? '#0284c7' : '#1e293b');
        let strokeColor = isTarget ? '#fda4af' : (isDirect ? '#38bdf8' : '#475569');
        let strokeWidth = '2';

        if (isConditioned) {
          nodeFill = '#047857';
          strokeColor = '#10b981';
          strokeWidth = '2.5';
        }

        return `
          <g transform="translate(${p.x}, ${p.y})"
             class="dag-node-interactive cursor-pointer select-none transition-transform"
             data-node="${escapeHtml(node)}"
             data-conditioned="${isConditioned ? 'true' : 'false'}"
             title="Click to ${isConditioned ? 'remove from' : 'add to'} Conditioning Set Z">
            ${isConditioned ? `
              <circle r="22" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="3,3" filter="url(#glow-z)" class="animate-pulse" />
              <rect x="-16" y="-32" width="32" height="13" rx="3" fill="#064e3b" stroke="#10b981" stroke-width="1" />
              <text text-anchor="middle" dy="-23" fill="#a7f3d0" font-size="8" font-family="monospace" font-weight="bold">Z-SET</text>
            ` : ''}
            <circle r="15" fill="${nodeFill}" stroke="${strokeColor}" stroke-width="${strokeWidth}" />
            <text text-anchor="middle" dy="4" fill="#ffffff" font-size="9" font-family="monospace" font-weight="bold">
              ${escapeHtml(node.slice(0, 3).toUpperCase())}
            </text>
            <text text-anchor="middle" dy="26" fill="${isConditioned ? '#a7f3d0' : (isTarget ? '#fda4af' : '#cbd5e1')}" font-size="10" font-family="monospace" font-weight="${isConditioned ? 'bold' : '600'}">
              ${escapeHtml(node)}
            </text>
          </g>
        `;
      }).join('');

      // Wire Node Click Listeners for Conditioning Set Toggling
      document.querySelectorAll('.dag-node-interactive').forEach(elem => {
        elem.addEventListener('click', (ev) => {
          ev.stopPropagation();
          const nodeName = elem.dataset.node;
          if (conditioningSet.has(nodeName)) {
            conditioningSet.delete(nodeName);
          } else {
            conditioningSet.add(nodeName);
          }
          renderSvgDag();
        });
      });

      // Wire Edge Click Listeners for Forbidden Domain Priors
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

    // Wire Clear Conditioning Set button
    document.getElementById('clear-conditioning-btn')?.addEventListener('click', () => {
      conditioningSet.clear();
      renderSvgDag();
    });

    // Wire Reset Priors button
    document.getElementById('reset-priors-btn')?.addEventListener('click', () => {
      forbiddenEdges.clear();
      renderSvgDag();
    });

    // Wire Axiomatic Invariant Scan & Lock Shelf Handlers
    document.getElementById('scan-invariants-btn')?.addEventListener('click', async () => {
      const container = document.getElementById('invariants-cards-container');
      if (!container) return;
      container.innerHTML = `
        <div class="col-span-full p-6 text-center text-slate-400 text-xs font-mono flex items-center justify-center space-x-2">
          <div class="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
          <span>Auditing dataset features against closed conservation laws (Ax = b)...</span>
        </div>
      `;

      try {
        const res = await ApiClient.scanInvariants(state.sessionId);
        if (!res.laws || res.laws.length === 0) {
          container.innerHTML = `
            <div class="col-span-full p-4 text-center text-slate-400 text-xs font-mono">
              No linear conservation laws detected in active feature subspace.
            </div>
          `;
          return;
        }

        container.innerHTML = res.laws.map((law, idx) => `
          <div class="p-4 bg-slate-900/80 border ${law.is_locked ? 'border-emerald-500/50 bg-emerald-950/10' : 'border-slate-800'} rounded-xl space-y-3 relative overflow-hidden transition-all" id="law-card-${idx}">
            <div class="flex items-start justify-between gap-2">
              <div class="space-y-1">
                <div class="flex items-center space-x-2">
                  <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold ${law.is_locked ? 'bg-emerald-500/20 text-emerald-300' : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'}">
                    ${escapeHtml(law.category)}
                  </span>
                  <h5 class="text-xs font-bold text-slate-200 font-mono">${escapeHtml(law.name)}</h5>
                </div>
                <div class="text-xs font-mono text-emerald-400 font-semibold pl-1">
                  <code>${escapeHtml(law.formula)}</code>
                </div>
              </div>
            </div>

            <div class="grid grid-cols-3 gap-2 text-center text-[11px] font-mono py-1 border-y border-slate-800/80">
              <div class="p-1.5 bg-slate-950/60 rounded">
                <div class="text-[9px] text-slate-500 uppercase">Residual MAE</div>
                <div class="text-slate-200 font-bold tabular-nums">${law.residual_mae.toFixed(4)}</div>
              </div>
              <div class="p-1.5 bg-slate-950/60 rounded">
                <div class="text-[9px] text-slate-500 uppercase">Compliance</div>
                <div class="text-emerald-400 font-bold tabular-nums">${law.compliance_pct.toFixed(1)}%</div>
              </div>
              <div class="p-1.5 bg-slate-950/60 rounded">
                <div class="text-[9px] text-slate-500 uppercase">Tolerance</div>
                <div class="text-slate-400 tabular-nums">&plusmn;${law.tolerance.toFixed(3)}</div>
              </div>
            </div>

            <div class="flex items-center justify-between pt-1" id="law-action-box-${idx}">
              ${law.is_locked ? `
                <span class="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[11px] font-mono font-bold flex items-center space-x-1.5">
                  <i data-lucide="check-circle" class="w-3.5 h-3.5"></i>
                  <span>✓ Locked Physical Axiom (Enforced in Projections)</span>
                </span>
              ` : `
                <button class="lock-invariant-btn px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 active:scale-95 text-white text-xs font-semibold rounded-lg flex items-center space-x-1.5 cursor-pointer font-mono transition-all" data-idx="${idx}" data-law-name="${escapeHtml(law.name)}" data-formula="${escapeHtml(law.formula)}">
                  <i data-lucide="lock" class="w-3 h-3"></i>
                  <span>Lock Invariant</span>
                </button>
              `}
            </div>
          </div>
        `).join('');

        if (window.lucide) lucide.createIcons();

        // Wire lock buttons
        document.querySelectorAll('.lock-invariant-btn').forEach(btn => {
          btn.addEventListener('click', async () => {
            const idx = btn.dataset.idx;
            const lawName = btn.dataset.lawName;
            const formula = btn.dataset.formula;
            btn.disabled = true;
            btn.innerHTML = `<span class="animate-spin mr-1">⌛</span> Locking...`;
            try {
              await ApiClient.lockInvariant(state.sessionId, {
                invariant_name: lawName,
                formula: formula,
              });
              const actionBox = document.getElementById(`law-action-box-${idx}`);
              const card = document.getElementById(`law-card-${idx}`);
              if (card) {
                card.classList.add('border-emerald-500/50', 'bg-emerald-950/10');
              }
              if (actionBox) {
                actionBox.innerHTML = `
                  <span class="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[11px] font-mono font-bold flex items-center space-x-1.5">
                    <i data-lucide="check-circle" class="w-3.5 h-3.5"></i>
                    <span>✓ Locked Physical Axiom (Enforced in Projections)</span>
                  </span>
                `;
                if (window.lucide) lucide.createIcons();
              }
            } catch (e) {
              btn.disabled = false;
              btn.innerHTML = `<i data-lucide="lock" class="w-3 h-3 mr-1"></i> Lock Invariant`;
              showError(`Failed to lock invariant: ${e.message}`);
            }
          });
        });
      } catch (err) {
        container.innerHTML = `<div class="col-span-full p-4 text-center text-rose-400 text-xs font-mono">Scan failed: ${escapeHtml(err.message)}</div>`;
      }
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


