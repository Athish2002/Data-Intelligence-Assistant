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
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
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

      <div class="glass-card p-6">
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
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Auditing 5-pillar dataset readiness...</div>`;
  try {
    const r = await ApiClient.getReadiness(state.sessionId);
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
              <div class="text-xl font-bold text-slate-100 mt-0.5">${r.production_verdict}</div>
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
                <span class="font-semibold text-xs text-slate-200">${c.category}: ${c.title}</span>
                <span class="text-[10px] uppercase font-bold px-2 py-0.5 rounded ${c.verdict === 'PASS' ? 'bg-emerald-950 text-emerald-400' : 'bg-amber-950 text-amber-400'}">${c.verdict}</span>
              </div>
              <p class="text-xs text-slate-400 mt-1">${c.details}</p>
              ${c.action_item ? `<div class="mt-2 text-[11px] text-cyan-300 font-medium">💡 Recommendation: ${c.action_item}</div>` : ''}
            </div>
          `
            )
            .join('')}
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">Error: ${err.message}</div>`;
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
        Autonomous zero-loss sanitization repaired <b>${d.sanitize_report.total_cells_repaired || 0} dirty or malformed entries</b>. 
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


