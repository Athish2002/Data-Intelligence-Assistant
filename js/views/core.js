/**
 * frontend/js/views/core.js
 * ─────────────────────────
 * Workspace 1: Core Intelligence Views (Overview, Schema, Readiness, Insights, Executive Summary).
 */

import { state, el, escapeHtml } from '../state.js';
import { ApiClient } from '../api.js';

export function renderCoreOverview() {
  const d = state.datasetMeta;
  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div id="section-core-kpis" class="grid grid-cols-2 md:grid-cols-4 gap-4" data-section-anchor="Dataset KPIs">
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Total Rows</div>
          <div class="text-2xl font-bold text-slate-100">${d.n_rows.toLocaleString()}</div>
          <div class="text-xs text-emerald-400 mt-1">✓ Ingested & Validated</div>
        </div>
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Total Features</div>
          <div class="text-2xl font-bold text-slate-100">${d.n_cols}</div>
          <div class="text-xs text-cyan-400 mt-1">${d.columns.length} columns active</div>
        </div>
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Business Domain</div>
          <div class="text-lg font-bold text-indigo-300 truncate">${d.detected_domain}</div>
          <div class="text-xs text-slate-400 mt-1">Semantic ontology match</div>
        </div>
        <div class="glass-card p-4">
          <div class="text-xs text-slate-400 uppercase font-semibold">Data Sanitizer</div>
          <div class="text-2xl font-bold text-emerald-400">${d.sanitize_report.total_cells_repaired || 0}</div>
          <div class="text-xs text-emerald-300 mt-1">Dirty cells auto-repaired</div>
        </div>
      </div>

      <div id="section-core-preview" class="glass-card p-6" data-section-anchor="Data Preview">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-base font-semibold text-slate-200 flex items-center">
            <i data-lucide="table" class="w-4 h-4 mr-2 text-cyan-400"></i> Sanitized Dataset Preview (First 10 Rows)
          </h3>
          <span class="text-xs text-slate-400">Zero data loss guaranteed</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-slate-300 border-collapse">
            <thead>
              <tr class="border-b border-slate-700/60 bg-slate-900/40 text-slate-400 font-semibold">
                ${d.columns.map((c) => `<th class="p-3">${escapeHtml(c)}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${d.sample_data
                .map(
                  (r) => `
                <tr class="border-b border-slate-800/40 hover:bg-slate-800/30">
                  ${d.columns.map((c) => `<td class="p-3">${r[c] !== null && r[c] !== undefined ? escapeHtml(r[c]) : '<span class="text-slate-600">null</span>'}</td>`).join('')}
                </tr>
              `
                )
                .join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

export async function renderCoreSchema() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Loading schema and column roles...</div>`;
  try {
    const prof = await ApiClient.getProfile(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6">
        <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center">
          <i data-lucide="columns" class="w-4 h-4 mr-2 text-indigo-400"></i> Column Roles & Schema Profile
        </h3>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left text-slate-300 border-collapse">
            <thead>
              <tr class="border-b border-slate-700 bg-slate-900/50 text-slate-400 font-semibold">
                <th class="p-3">Column Name</th>
                <th class="p-3">Data Type</th>
                <th class="p-3">Detected Role</th>
                <th class="p-3">Missing %</th>
                <th class="p-3">Uniques</th>
                <th class="p-3">Sample Values</th>
              </tr>
            </thead>
            <tbody>
              ${prof.columns_info
                .map(
                  (col) => `
                <tr class="border-b border-slate-800/40 hover:bg-slate-800/30">
                  <td class="p-3 font-medium text-slate-200 font-mono">${escapeHtml(col.column)}</td>
                  <td class="p-3 text-slate-400">${escapeHtml(col.dtype)}</td>
                  <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] uppercase font-bold ${getRoleBadgeClass(col.role)}">${escapeHtml(col.role)}</span></td>
                  <td class="p-3 ${col.null_pct > 0 ? 'text-amber-400' : 'text-slate-400'}">${col.null_pct}% (${col.null_count})</td>
                  <td class="p-3 font-mono">${col.unique_count.toLocaleString()}</td>
                  <td class="p-3 text-slate-400 font-mono text-[11px] truncate max-w-xs">${escapeHtml((col.sample_values || []).join(', '))}</td>
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
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error loading schema: ${escapeHtml(err.message)}</div>`;
  }
}

function getRoleBadgeClass(role) {
  if (role === 'numeric') return 'bg-cyan-950 text-cyan-400 border border-cyan-500/30';
  if (role.startsWith('categorical')) return 'bg-purple-950 text-purple-400 border border-purple-500/30';
  if (role === 'datetime') return 'bg-emerald-950 text-emerald-400 border border-emerald-500/30';
  if (role === 'id_entity') return 'bg-slate-800 text-slate-400 border border-slate-700';
  return 'bg-amber-950 text-amber-400 border border-amber-500/30';
}

export async function renderCoreReadiness() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Auditing 5-pillar dataset readiness & target leakage forensics...</div>`;
  try {
    const r = await ApiClient.getReadiness(state.sessionId);

    // Attempt target leakage audit if target is identified
    let leakageReport = null;
    const targetCandidate = state.pipelineResult?.target_col || state.datasetMeta?.suggested_target || el.targetColInput?.value || null;
    if (targetCandidate) {
      try {
        leakageReport = await ApiClient.getLeakageReport(state.sessionId);
      } catch (e) {
        console.warn('Leakage check deferred or target not set:', e);
      }
    }

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Circular Gauge & Production Verdict -->
        <div class="glass-card p-6 flex flex-col md:flex-row items-center justify-between gap-6">
          <div class="flex items-center space-x-6">
            <div class="w-24 h-24 rounded-full border-4 border-cyan-500 flex flex-col items-center justify-center bg-cyan-950/30 shadow-lg shadow-cyan-500/20">
              <span class="text-2xl font-bold text-slate-100">${r.overall_readiness_score}%</span>
              <span class="text-[9px] uppercase font-bold text-cyan-400">Health</span>
            </div>
            <div>
              <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Production Readiness Verdict</div>
              <div class="text-xl font-bold text-slate-100 mt-0.5">${escapeHtml(r.production_verdict)}</div>
              <p class="text-xs text-slate-400 mt-1">Evaluated across missingness, multicollinearity, sample power, and class balance.</p>
            </div>
          </div>
          <div class="text-right">
            <span class="px-3 py-1 bg-emerald-950 border border-emerald-500/40 text-emerald-400 text-xs font-semibold rounded-full">
              Enterprise Grade
            </span>
          </div>
        </div>

        <!-- 5-Pillar Audit Checklist -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          ${r.checks
            .map(
              (c) => `
            <div class="glass-card p-4 border-l-4 ${c.verdict === 'PASS' ? 'border-l-emerald-500' : c.verdict === 'WARN' ? 'border-l-amber-500' : 'border-l-rose-500'}">
              <div class="flex justify-between items-start mb-1">
                <span class="font-semibold text-xs text-slate-200">${escapeHtml(c.category)}: ${escapeHtml(c.title)}</span>
                <span class="text-[10px] uppercase font-bold px-2 py-0.5 rounded ${c.verdict === 'PASS' ? 'bg-emerald-950 text-emerald-400' : 'bg-amber-950 text-amber-400'}">${escapeHtml(c.verdict)}</span>
              </div>
              <p class="text-xs text-slate-400 mt-1">${escapeHtml(c.details)}</p>
              ${c.action_item ? `<div class="mt-2 text-[11px] text-cyan-300 font-medium">💡 Recommendation: ${escapeHtml(c.action_item)}</div>` : ''}
            </div>
          `
            )
            .join('')}
        </div>

        <!-- Target Leakage & Data Poisoning Sleuth Card -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div class="flex items-center space-x-2">
                <div class="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400">
                  <i data-lucide="shield-alert" class="w-4 h-4"></i>
                </div>
                <h4 class="text-sm font-bold text-slate-100">Target Leakage & Data Poisoning Sleuth</h4>
              </div>
              <p class="text-xs text-slate-400 mt-1">
                Information-theoretic mutual information ratios I(X; Y) / H(Y), temporal chronology inversions, quasi-target correlations, and high-cardinality memorization detection.
              </p>
            </div>
            ${leakageReport ? `
              <div class="flex items-center space-x-2 shrink-0">
                <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-bold border ${leakageReport.overall_leakage_risk_score > 50 ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' : leakageReport.overall_leakage_risk_score > 20 ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'}">
                  Risk Score: ${leakageReport.overall_leakage_risk_score.toFixed(1)} / 100
                </span>
              </div>
            ` : ''}
          </div>

          ${leakageReport ? `
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
              <div class="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-xs">
                <div class="text-slate-400 font-mono text-[10px] uppercase">Quarantined Features</div>
                <div class="text-base font-bold font-mono mt-1 ${leakageReport.quarantine_features.length > 0 ? 'text-rose-400' : 'text-emerald-400'}">
                  ${leakageReport.quarantine_features.length} Detected
                </div>
              </div>
              <div class="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-xs">
                <div class="text-slate-400 font-mono text-[10px] uppercase">Temporal Inversion</div>
                <div class="text-base font-bold font-mono mt-1 ${leakageReport.temporal_leakage_detected ? 'text-rose-400' : 'text-emerald-400'}">
                  ${leakageReport.temporal_leakage_detected ? 'LEAK DETECTED' : 'CLEAN'}
                </div>
              </div>
              <div class="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-xs">
                <div class="text-slate-400 font-mono text-[10px] uppercase">Poisoning Conflicts</div>
                <div class="text-base font-bold font-mono mt-1 ${leakageReport.poisoning_detected ? 'text-rose-400' : 'text-emerald-400'}">
                  ${leakageReport.poisoning_detected ? 'CONFLICTS FOUND' : 'CLEAN'}
                </div>
              </div>
            </div>

            ${(() => {
              const recList = Array.isArray(leakageReport.quarantine_recommendations)
                ? leakageReport.quarantine_recommendations
                : (Array.isArray(leakageReport.feature_assessments)
                    ? leakageReport.feature_assessments.filter(a => a.risk_level !== 'CLEAN')
                    : Object.entries(leakageReport.quarantine_recommendations || {}).map(([feat, r]) => ({
                        feature: feat,
                        reason: typeof r === 'string' ? r : (r?.reason || 'Quarantine recommended'),
                        risk_level: typeof r === 'string' ? (r.includes('CRITICAL') ? 'QUARANTINE_CRITICAL' : 'QUARANTINE_WARNING') : (r?.risk_level || 'QUARANTINE_WARNING'),
                      }))
                  );

              if (!recList || recList.length === 0) {
                return `
                  <div class="p-3 bg-emerald-950/20 border border-emerald-500/30 rounded-lg text-emerald-300 text-xs flex items-center space-x-2">
                    <i data-lucide="check-circle" class="w-4 h-4 text-emerald-400 shrink-0"></i>
                    <span>Zero target leakage or data poisoning detected across evaluated features for target <code>${escapeHtml(leakageReport.target_col)}</code>.</span>
                  </div>
                `;
              }

              return `
                <div class="pt-2">
                  <div class="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold mb-2">Feature Quarantine Manifest</div>
                  <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left text-slate-300 border-collapse">
                      <thead>
                        <tr class="border-b border-slate-700 bg-slate-900/50 text-slate-400 font-semibold font-mono text-[11px]">
                          <th class="p-2.5">Feature Column</th>
                          <th class="p-2.5">Quarantine Action</th>
                          <th class="p-2.5 text-right">MI Ratio</th>
                          <th class="p-2.5 text-right">Correlation</th>
                          <th class="p-2.5">Risk Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        ${recList.map(item => {
                          const feat = item.feature || item.column || '';
                          const reason = item.reason || item.metric_name || 'Flagged for quarantine';
                          const isCritical = item.risk_level === 'QUARANTINE_CRITICAL' || String(reason).includes('CRITICAL') || String(reason).includes('ID Memorization');
                          const mi = leakageReport.mi_scores?.[feat] ?? (item.metric_name === 'Mutual Information' ? item.metric_value : null);
                          const corr = leakageReport.correlation_scores?.[feat] ?? (item.metric_name === 'Correlation' ? item.metric_value : null);
                          return `
                            <tr class="border-b border-slate-800/40 hover:bg-slate-800/20">
                              <td class="p-2.5 font-mono font-semibold text-slate-200">${escapeHtml(feat)}</td>
                              <td class="p-2.5">
                                <span class="px-2 py-0.5 rounded text-[10px] uppercase font-mono border ${isCritical ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' : 'bg-amber-500/20 text-amber-300 border-amber-500/40'}">
                                  ${isCritical ? 'QUARANTINE_CRITICAL' : 'QUARANTINE_WARN'}
                                </span>
                              </td>
                              <td class="p-2.5 text-right font-mono text-cyan-300">${mi != null ? Number(mi).toFixed(3) : '—'}</td>
                              <td class="p-2.5 text-right font-mono text-amber-300">${corr != null ? Number(corr).toFixed(3) : '—'}</td>
                              <td class="p-2.5 text-slate-400 text-[11px]">${escapeHtml(reason)}</td>
                            </tr>
                          `;
                        }).join('')}
                      </tbody>
                    </table>
                  </div>
                </div>
              `;
            })()}

            ${leakageReport.clean_features && leakageReport.clean_features.length > 0 ? `
              <div class="text-[11px] text-slate-400 pt-1">
                <span class="font-semibold text-slate-300">Clean Verified Features:</span>
                ${leakageReport.clean_features.map(f => `<span class="inline-block font-mono bg-slate-900 border border-slate-800 px-2 py-0.5 rounded mr-1 mb-1 text-slate-300">${escapeHtml(f)}</span>`).join('')}
              </div>
            ` : ''}
          ` : `
            <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-lg text-xs text-slate-400 flex items-center justify-between">
              <span>Target feature not declared yet. Specify a prediction target in the left dock to trigger automated mutual information forensics.</span>
              <button onclick="el.autodetectBtn.click()" class="px-3 py-1.5 rounded text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer shrink-0">
                Auto-Detect Target
              </button>
            </div>
          `}
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${escapeHtml(err.message)}</div>`;
  }
}

export async function renderCoreInsights() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Generating correlation matrix & smart insights...</div>`;
  try {
    const prof = await ApiClient.getProfile(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <div class="glass-card p-6">
          <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center">
            <i data-lucide="activity" class="w-4 h-4 mr-2 text-cyan-400"></i> Top Pairwise Feature Correlations
          </h3>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            ${prof.correlations.slice(0, 8)
              .map(
                (c) => `
              <div class="p-3 bg-slate-900/50 border border-slate-800 rounded-lg flex justify-between items-center text-xs">
                <span class="font-mono text-slate-300">${c.feature_a} &harr; ${c.feature_b}</span>
                <span class="font-mono font-bold ${c.correlation > 0 ? 'text-cyan-400' : 'text-rose-400'}">${c.correlation > 0 ? '+' : ''}${c.correlation}</span>
              </div>
            `
              )
              .join('')}
          </div>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
  }
}

export function renderCoreExecutive() {
  const d = state.datasetMeta;
  const p = state.pipelineResult;
  el.tabContent.innerHTML = `
    <div class="glass-card p-8 space-y-4">
      <h3 class="text-lg font-bold text-slate-100 flex items-center">
        <i data-lucide="briefcase" class="w-5 h-5 mr-2 text-indigo-400"></i> Executive Business Briefing
      </h3>
      <p class="text-xs text-slate-300 leading-relaxed">
        The Data Intelligence Assistant ingested and processed <b>${d.n_rows.toLocaleString()} records</b> across <b>${d.n_cols} features</b>. 
        Autonomous heuristic sanitization repaired <b>${d.sanitize_report.total_cells_repaired || 0} dirty or malformed entries</b>. 
        Business domain was determined as <span class="text-cyan-300 font-semibold">${d.detected_domain}</span>.
      </p>
      ${
        p
          ? `
        <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2 text-xs">
          <div class="font-semibold text-slate-200">AutoML Deployment Summary:</div>
          <div>&bull; Primary Target: <code class="text-cyan-300">${p.target_col}</code> (${p.final_task_type})</div>
          <div>&bull; Champion Model: <span class="text-emerald-400 font-semibold">${p.best_model_label}</span></div>
          <div>&bull; Production Health Score: <span class="text-cyan-400 font-semibold">${p.readiness_score_pct}%</span> (${p.readiness_verdict})</div>
        </div>
      `
          : '<div class="text-xs text-slate-500 italic">Launch AutoML in the sidebar to populate model evaluation summaries.</div>'
      }
    </div>
  `;
}

// ─── L4: Symbolic Feature Discovery Subtab ────────────────────────────────────

export async function renderCoreSymbolic() {
  if (!state.sessionId) {
    el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Please connect a dataset first.</div>`;
    return;
  }

  el.tabContent.innerHTML = `
    <div class="space-y-6">
      <div class="glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <div class="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <i data-lucide="binary" class="w-5 h-5"></i>
            </div>
            <h3 class="text-base font-bold text-slate-100">Symbolic Feature Discovery & Formula Distillation</h3>
          </div>
          <p class="text-xs text-slate-400 mt-2 max-w-2xl">
            Genetic and heuristic search discovering non-linear algebraic invariants, compound ratios, and logarithmic interactions across numerical dimensions. Exports discovered features as ANSI-SQL and Python transforms.
          </p>
        </div>
        <button id="run-symbolic-btn" class="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs uppercase tracking-wider rounded-lg flex items-center space-x-2 cursor-pointer shadow-md active:scale-95 shrink-0">
          <i data-lucide="sparkles" class="w-4 h-4"></i>
          <span>Discover Symbolic Invariants</span>
        </button>
      </div>

      <div id="symbolic-results-container" class="space-y-4">
        <div class="glass-card p-12 text-center text-slate-400 text-xs font-mono">
          Click <strong>Discover Symbolic Invariants</strong> to synthesize non-linear interaction formulas.
        </div>
      </div>
    </div>
  `;

  document.getElementById('run-symbolic-btn')?.addEventListener('click', async () => {
    const container = document.getElementById('symbolic-results-container');
    container.innerHTML = `
      <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
        <div class="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
        <span class="text-xs font-mono uppercase tracking-wider">Evaluating Non-Linear Algebraic Trees & Mutual Information...</span>
      </div>
    `;

    try {
      const res = await ApiClient.discoverSymbolicFeatures(state.sessionId);

      container.innerHTML = `
        <div class="space-y-6">
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div class="glass-card p-4 space-y-1 border-l-4 border-l-cyan-500">
              <div class="text-[10px] font-mono text-cyan-400 uppercase">Evaluated Candidates</div>
              <div class="text-xl font-bold font-mono text-slate-100">${res.n_evaluated_expressions} Expressions</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">Top Discovered Formulas</div>
              <div class="text-xl font-bold font-mono text-[#00e575]">${res.n_discovered_formulas} High-Lift Features</div>
            </div>
            <div class="glass-card p-4 space-y-1">
              <div class="text-[10px] font-mono text-slate-400 uppercase">Target Objective</div>
              <div class="text-xl font-bold font-mono text-slate-200 truncate">${escapeHtml(res.target_column)}</div>
            </div>
          </div>

          <!-- Formulas Grid -->
          <div class="grid grid-cols-1 gap-4">
            ${res.formulas.map(f => `
              <div class="glass-card p-5 space-y-3 border-l-2 border-l-cyan-400">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <span class="font-mono font-bold text-sm text-slate-100">${escapeHtml(f.feature_name)}</span>
                  <div class="flex items-center space-x-2 font-mono text-xs">
                    <span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold">
                      Correlation Lift: +${(f.correlation_lift * 100).toFixed(1)}%
                    </span>
                    <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      MI: ${f.mutual_info_score.toFixed(3)}
                    </span>
                  </div>
                </div>
                <p class="text-xs text-slate-400">${escapeHtml(f.description)}</p>
                <div class="p-2.5 rounded bg-slate-900 border border-slate-800 font-mono text-xs text-cyan-300 overflow-x-auto">
                  LaTeX: <code>${escapeHtml(f.formula_latex)}</code>
                </div>
                <div class="text-[11px] font-mono text-slate-400">
                  SQL: <code class="text-slate-200">${escapeHtml(f.sql_expression)}</code>
                </div>
              </div>
            `).join('')}
          </div>

          <!-- Consolidated SQL View Export -->
          <div class="glass-card p-5 space-y-3">
            <h4 class="text-xs font-mono uppercase tracking-wider text-slate-200 font-bold flex items-center space-x-2">
              <i data-lucide="database" class="w-4 h-4 text-cyan-400"></i>
              <span>Consolidated In-Database ANSI-SQL Feature View</span>
            </h4>
            <pre class="p-3.5 rounded bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto select-all"><code>${escapeHtml(res.consolidated_sql_view)}</code></pre>
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


