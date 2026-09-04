/**
 * frontend/js/views/governance.js
 * ───────────────────────────────
 * Workspace 4: Governance & MLOps (Data Contracts, GDPR ROPA, Drift, SQL Transpiler, Copilot).
 */

import { state, el, escapeHtml } from '../state.js';
import { ApiClient } from '../api.js';
import { renderModelRequiredState } from './automl.js';

export async function renderGovernanceContracts() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Generating Great Expectations suite...</div>`;
  try {
    const res = await ApiClient.getDataContract(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="check-square" class="w-4 h-4 mr-2 text-emerald-400"></i> Great Expectations & Data Contracts (${res.n_expectations} Tests)
        </h3>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${res.contract_yaml}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

export async function renderGovernanceGdpr() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Auditing GDPR ROPA & PII...</div>`;
  try {
    const res = await ApiClient.getGdprAudit(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="lock" class="w-4 h-4 mr-2 text-indigo-400"></i> GDPR ROPA Register & Privacy Audit
        </h3>
        <div class="flex justify-between text-xs p-3 bg-slate-900/60 rounded-lg">
          <span>Privacy Risk Score: <b class="text-emerald-400 font-bold">${res.privacy_risk_score} / 100</b></span>
          <span>PII Findings: <b class="text-slate-300 font-mono">${res.pii_entities_detected.length}</b></span>
        </div>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${res.ropa_markdown}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

export async function renderGovernanceDrift() {
  el.tabContent.innerHTML = `<div class="glass-card p-8 text-center text-slate-400">Computing PSI drift monitor...</div>`;
  try {
    const res = await ApiClient.getDriftMonitor(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="git-commit" class="w-4 h-4 mr-2 text-cyan-400"></i> Population Stability Index (PSI) Drift Monitor
        </h3>
        <div class="flex justify-between text-xs p-3 bg-slate-900/60 rounded-lg">
          <span>Overall Dataset PSI: <b class="text-cyan-400 font-mono font-bold">${res.psi_score.toFixed(4)}</b></span>
          <span>Status: <b class="text-emerald-400 uppercase">${res.drift_status}</b></span>
        </div>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400">${err.message}</div>`;
  }
}

export async function renderGovernanceSql() {
  if (!state.pipelineResult) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'In-Database SQL Transpiler Awaiting Champion Model',
      'Transpiling decision tree and linear models directly into Snowflake, BigQuery, Postgres, or Redshift SQL queries requires an active champion model.',
      'database'
    );
    lucide.createIcons();
    return;
  }

  try {
    const arts = await ApiClient.getAllArtifacts(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <div class="flex justify-between items-center">
          <h3 class="text-base font-semibold text-slate-100 flex items-center">
            <i data-lucide="database" class="w-4 h-4 mr-2 text-cyan-400"></i> In-Database SQL Transpiler
          </h3>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'sql')}" download class="text-xs text-cyan-400 underline">⬇️ Download SQL</a>
        </div>
        <p class="text-xs text-slate-400">Direct SQL CASE WHEN scoring logic ready for Snowflake, BigQuery, Postgres, or Redshift.</p>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${arts.sql_query}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-slate-400">Train AutoML first to view SQL transpiler code.</div>`;
  }
}

export async function renderGovernanceCode() {
  if (!state.pipelineResult) {
    el.tabContent.innerHTML = renderModelRequiredState(
      'Production Code & Docker Artifacts Awaiting Pipeline',
      'Generating standalone FastAPI main.py, Airflow DAG, Dockerfile, and Kubernetes deployment manifests requires an executed pipeline.',
      'code'
    );
    lucide.createIcons();
    return;
  }

  try {
    const arts = await ApiClient.getAllArtifacts(state.sessionId);
    el.tabContent.innerHTML = `
      <div class="glass-card p-6 space-y-4">
        <h3 class="text-base font-semibold text-slate-100 flex items-center">
          <i data-lucide="code" class="w-4 h-4 mr-2 text-indigo-400"></i> Production Deployment Artifacts
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <a href="${ApiClient.getExportUrl(state.sessionId, 'python')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">🐍 pipeline.py</a>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'fastapi')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">⚡ main.py</a>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'dockerfile')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">🐳 Dockerfile</a>
          <a href="${ApiClient.getExportUrl(state.sessionId, 'airflow')}" download class="p-3 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-center block text-xs text-cyan-300">💨 airflow_dag.py</a>
        </div>
        <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono overflow-x-auto max-h-96"><code>${arts.python_script}</code></pre>
      </div>
    `;
    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-slate-400">Train AutoML first to view production code exports.</div>`;
  }
}

export function renderGovernanceCopilot() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-4">
      <h3 class="text-base font-semibold text-slate-100 flex items-center">
        <i data-lucide="message-square" class="w-4 h-4 mr-2 text-indigo-400"></i> Autonomous Dataset Copilot
      </h3>
      <div id="chat-messages" class="h-64 overflow-y-auto p-3 bg-slate-950/60 border border-slate-800/80 rounded-xl space-y-3 text-xs">
        <div class="text-slate-400 italic">Ask any analytical question (e.g., "correlations", "outliers", "summary of features").</div>
      </div>
      <div class="flex space-x-2">
        <input type="text" id="chat-input" class="flex-1 glass-input text-xs" placeholder="Ask a question about your dataset...">
        <button id="chat-send-btn" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg">
          Send
        </button>
      </div>
    </div>
  `;

  document.getElementById('chat-send-btn')?.addEventListener('click', handleChatQuery);
  document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChatQuery();
  });

  lucide.createIcons();
}

async function handleChatQuery() {
  const input = document.getElementById('chat-input');
  const msgBox = document.getElementById('chat-messages');
  if (!input || !msgBox || !input.value.trim() || !state.sessionId) return;

  const query = input.value.trim();
  input.value = '';

  msgBox.innerHTML += `
    <div class="text-right">
      <span class="inline-block p-2 bg-indigo-600/40 border border-indigo-500/30 text-slate-100 rounded-lg">${query}</span>
    </div>
  `;
  msgBox.scrollTop = msgBox.scrollHeight;

  try {
    const res = await ApiClient.chat(state.sessionId, query);
    const sourcesHtml = (res.sources && res.sources.length)
      ? `
        <details class="text-slate-400">
          <summary class="cursor-pointer text-indigo-400">📎 Sources (${_escapeHtml(res.engine || 'retrieval_only')})</summary>
          <ul class="list-disc list-inside space-y-1 mt-1">
            ${res.sources.map(src => `<li><code class="text-slate-500">${_escapeHtml(src.source_type)}</code> — ${_escapeHtml(src.snippet)}</li>`).join('')}
          </ul>
        </details>
      `
      : '';
    msgBox.innerHTML += `
      <div class="text-left">
        <div class="inline-block p-3 bg-slate-900 border border-slate-800 text-slate-200 rounded-lg space-y-2">
          <div>${res.content}</div>
          ${sourcesHtml}
        </div>
      </div>
    `;
    msgBox.scrollTop = msgBox.scrollHeight;
  } catch (err) {
    msgBox.innerHTML += `<div class="text-rose-400 text-left">Error: ${err.message}</div>`;
  }
}

function _escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

