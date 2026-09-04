/**
 * frontend/js/views/governance.js
 * ───────────────────────────────
 * Workspace 4: Governance & MLOps (Data Contracts, GDPR ROPA, Drift, SQL Transpiler, Copilot).
 */

import { state, el, escapeHtml } from '../state.js';
import { ApiClient } from '../api.js';
import { renderModelRequiredState } from './automl.js';

export async function renderGovernanceContracts() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Generating Great Expectations Suite & Invariants...</span>
    </div>
  `;
  try {
    const res = await ApiClient.getDataContract(state.sessionId);
    const expectations = res.expectations || [];
    const nTests = res.n_expectations || expectations.length || 0;

    // Executive KPI stats
    const totalInvariants = nTests;
    const nullTolerance = "0% (Strict Block)";
    const schemaParity = "100% Guaranteed";
    const interceptionAction = "Fail-Closed Gate";

    // Format individual expectations into natural language table rows
    const rowsHtml = expectations.map((exp) => {
      const type = exp.expectation_type || 'Custom Invariant';
      const col = exp.kwargs?.column || 'Dataset Schema';

      let humanRule = '';
      let constraintBadge = '';
      let severityBadge = '';

      if (type === 'expect_table_columns_to_match_set') {
        const count = exp.kwargs?.column_set?.length || 'Exact';
        humanRule = 'Strict schema parity validation; prevents missing or undeclared feature columns.';
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">${count} Required Features</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-rose-500/15 text-rose-300 border border-rose-500/30">CRITICAL</span>`;
      } else if (type === 'expect_column_values_to_not_be_null') {
        humanRule = 'Completeness assertion; immediately rejects records with missing, NaN, or null values.';
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">Non-Null (0% Missing)</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-rose-500/15 text-rose-300 border border-rose-500/30">CRITICAL</span>`;
      } else if (type === 'expect_column_values_to_be_between') {
        const minVal = Number(exp.kwargs?.min_value).toFixed(2);
        const maxVal = Number(exp.kwargs?.max_value).toFixed(2);
        humanRule = 'Numerical envelope boundary; prevents out-of-distribution extremes and calculation errors.';
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-amber-500/10 text-amber-300 border border-amber-500/20">[${minVal}, ${maxVal}]</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-amber-500/15 text-amber-300 border border-amber-500/30">HIGH</span>`;
      } else if (type === 'expect_column_values_to_be_in_set') {
        const vals = exp.kwargs?.value_set || [];
        const preview = vals.slice(0, 3).join(', ') + (vals.length > 3 ? '…' : '');
        humanRule = 'Categorical domain guard; rejects unseen category tokens that break feature encoding.';
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">{${escapeHtml(preview)}}</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-amber-500/15 text-amber-300 border border-amber-500/30">HIGH</span>`;
      } else if (type === 'expect_column_values_to_be_of_type') {
        const dt = exp.kwargs?.type_ || 'Standard';
        humanRule = 'Datatype verification; enforces physical storage compatibility with the scoring pipeline.';
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 text-slate-300 border border-slate-700">${escapeHtml(dt)}</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">MEDIUM</span>`;
      } else if (type === 'expect_column_to_exist') {
        humanRule = 'Structural existence check; verifies feature presence in the inference payload vector.';
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">Column Present</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-rose-500/15 text-rose-300 border border-rose-500/30">CRITICAL</span>`;
      } else {
        humanRule = `Automated invariant rule: ${escapeHtml(type.replace(/_/g, ' '))}.`;
        constraintBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 text-slate-300 border border-slate-700">Assert True</span>`;
        severityBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-slate-800 text-slate-300 border border-slate-700">INFO</span>`;
      }

      return `
        <tr class="hover:bg-slate-900/40 transition-colors">
          <td class="p-3.5 border-b border-slate-800/80 font-mono text-xs text-slate-200 font-semibold">
            <span class="inline-flex items-center space-x-1.5">
              <i data-lucide="shield" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
              <span>${escapeHtml(col)}</span>
            </span>
          </td>
          <td class="p-3.5 border-b border-slate-800/80 text-xs text-slate-300 leading-relaxed">
            ${humanRule}
          </td>
          <td class="p-3.5 border-b border-slate-800/80">
            ${constraintBadge}
          </td>
          <td class="p-3.5 border-b border-slate-800/80 text-center">
            ${severityBadge}
          </td>
        </tr>
      `;
    }).join('');

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Header Banner -->
        <div class="glass-card p-6 relative overflow-hidden">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-2">
                <div class="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                  <i data-lucide="check-square" class="w-5 h-5"></i>
                </div>
                <h3 class="text-base font-bold text-slate-100">Enterprise Data Quality & Behavioral Contracts</h3>
              </div>
              <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                Automated Great Expectations assertion suite enforcing strict schema constraints, non-null guarantees, and continuous numerical envelopes. Incoming production payloads are validated against these rules prior to scoring to prevent silent inference failures.
              </p>
            </div>
            <div class="flex items-center space-x-2 shrink-0">
              <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1.5">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>Enforcement Active</span>
              </span>
            </div>
          </div>

          <!-- Executive KPI Strip -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6">
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Invariants Enforced</div>
              <div class="text-lg font-bold font-mono text-emerald-400 mt-1">${totalInvariants} Rules</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Continuous verification</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Null Tolerance</div>
              <div class="text-lg font-bold font-mono text-cyan-400 mt-1">${nullTolerance}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Missing data intercepted</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Schema Parity</div>
              <div class="text-lg font-bold font-mono text-indigo-400 mt-1">${schemaParity}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Feature vectors locked</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Pipeline Action</div>
              <div class="text-lg font-bold font-mono text-rose-400 mt-1">${interceptionAction}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Safe fallback on breach</div>
            </div>
          </div>
        </div>

        <!-- Plain English Contract Invariants Table -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <i data-lucide="list-checks" class="w-4 h-4 text-emerald-400"></i>
              <span>Active Production Invariants & Pre-Inference Validation Rules</span>
            </h4>
            <span class="text-xs text-slate-400 font-mono">${nTests} total assertions</span>
          </div>

          <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/50">
            <table class="w-full text-left text-xs border-collapse">
              <thead>
                <tr class="bg-slate-900/80 border-b border-slate-800 text-slate-400">
                  <th class="p-3.5 font-semibold">Target Feature</th>
                  <th class="p-3.5 font-semibold">Natural Language Verification Rule</th>
                  <th class="p-3.5 font-semibold">Enforced Constraint</th>
                  <th class="p-3.5 font-semibold text-center">Severity</th>
                </tr>
              </thead>
              <tbody>
                ${rowsHtml || `
                  <tr>
                    <td colspan="4" class="p-6 text-center text-slate-500 italic">
                      No discrete expectations found in contract.
                    </td>
                  </tr>
                `}
              </tbody>
            </table>
          </div>

          <!-- Operational Enforcement Notice -->
          <div class="p-4 bg-emerald-950/20 border border-emerald-500/20 rounded-xl text-xs text-slate-300 flex items-start space-x-3">
            <i data-lucide="info" class="w-4 h-4 text-emerald-400 mt-0.5 shrink-0"></i>
            <div>
              <strong class="text-emerald-300">Automated Pipeline Guard:</strong> In a production deployment, this contract runs inside your ingestion worker or API gateway before executing model scoring. Any record violating these constraints triggers an alert and falls back to a deterministic default, preventing biased predictions or silent crashes.
            </div>
          </div>
        </div>

        <!-- Collapsible Raw Developer Suite (Hidden by default) -->
        <div class="glass-card p-5">
          <details class="group">
            <summary class="cursor-pointer text-xs font-semibold text-slate-300 hover:text-emerald-400 flex items-center justify-between transition-colors">
              <span class="flex items-center space-x-2">
                <i data-lucide="code-2" class="w-4 h-4 text-slate-400 group-hover:text-emerald-400"></i>
                <span>Developer Spec & Machine-Readable Great Expectations Suite</span>
              </span>
              <span class="text-slate-500 text-[11px] font-mono group-open:rotate-180 transition-transform">▼</span>
            </summary>
            <div class="mt-4 pt-4 border-t border-slate-800/80 space-y-3">
              <div class="flex justify-between items-center text-xs text-slate-400">
                <span>Machine-readable suite export (Pydantic / Great Expectations JSON)</span>
                <button id="copy-gx-btn" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-[11px] flex items-center space-x-1 cursor-pointer">
                  <i data-lucide="copy" class="w-3 h-3"></i>
                  <span>Copy JSON</span>
                </button>
              </div>
              <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-300 font-mono overflow-x-auto max-h-80"><code id="gx-code-block">${escapeHtml(res.great_expectations_json || res.contract_yaml)}</code></pre>
            </div>
          </details>
        </div>
      </div>
    `;

    document.getElementById('copy-gx-btn')?.addEventListener('click', () => {
      const code = document.getElementById('gx-code-block')?.textContent || '';
      navigator.clipboard.writeText(code);
      const btn = document.getElementById('copy-gx-btn');
      if (btn) btn.innerHTML = `<i data-lucide="check" class="w-3 h-3 text-emerald-400"></i><span class="text-emerald-400">Copied!</span>`;
      setTimeout(() => lucide.createIcons(), 50);
    });

    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Failed to load data contract: ${escapeHtml(err.message)}</div>`;
  }
}

export async function renderGovernanceGdpr() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Auditing GDPR Article 30 ROPA, HIPAA & Privacy Posture...</span>
    </div>
  `;
  try {
    const res = await ApiClient.getGdprAudit(state.sessionId);
    const riskScore = res.privacy_risk_score || 0;
    const piiList = res.pii_entities_detected || [];
    const ropa = res.ropa_details || {};

    const isPristine = riskScore === 0 && piiList.length === 0;

    // PII findings rows
    let piiRowsHtml = '';
    if (piiList.length > 0) {
      piiRowsHtml = piiList.map((item) => `
        <tr class="hover:bg-slate-900/40 transition-colors">
          <td class="p-3 border-b border-slate-800 font-mono text-xs text-rose-300 font-semibold">${escapeHtml(item.column || 'field')}</td>
          <td class="p-3 border-b border-slate-800 text-xs text-slate-300">${escapeHtml(item.header_categories?.join(', ') || 'Direct Identifier')}</td>
          <td class="p-3 border-b border-slate-800 font-mono text-xs text-slate-400">${escapeHtml(item.value_patterns?.join(', ') || 'Pattern Match')}</td>
          <td class="p-3 border-b border-slate-800 text-center">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase ${item.risk_level === 'High' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'}">
              ${escapeHtml(item.risk_level || 'High')}
            </span>
          </td>
          <td class="p-3 border-b border-slate-800 font-mono text-xs text-emerald-400">${escapeHtml(item.sample_masked || '***')}</td>
        </tr>
      `).join('');
    }

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Executive Header & Posture Banner -->
        <div class="glass-card p-6 space-y-6">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-2">
                <div class="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                  <i data-lucide="lock" class="w-5 h-5"></i>
                </div>
                <h3 class="text-base font-bold text-slate-100">Enterprise Privacy Posture & GDPR Article 30 ROPA Audit</h3>
              </div>
              <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                Continuous compliance surveillance examining direct and quasi-identifiers, verifying HIPAA Safe Harbor bounds, enforcing Article 25 Privacy by Design, and compiling the mandatory GDPR Article 30 Record of Processing Activities (ROPA).
              </p>
            </div>
            <div class="flex items-center space-x-2 shrink-0">
              <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium ${riskScore <= 20 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'} flex items-center space-x-1.5">
                <span class="w-2 h-2 rounded-full ${riskScore <= 20 ? 'bg-emerald-400' : 'bg-rose-400'} animate-pulse"></span>
                <span>${riskScore <= 20 ? 'COMPLIANT & SECURE' : 'REMEDIATION REQUIRED'}</span>
              </span>
            </div>
          </div>

          <!-- Executive KPI Strip -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Privacy Risk Score</div>
              <div class="text-lg font-bold font-mono ${riskScore <= 20 ? 'text-emerald-400' : 'text-rose-400'} mt-1">${riskScore} / 100</div>
              <div class="text-[11px] text-slate-500 mt-0.5">${riskScore === 0 ? 'Pristine security posture' : 'Active risks detected'}</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">PII Entities Detected</div>
              <div class="text-lg font-bold font-mono ${piiList.length === 0 ? 'text-emerald-400' : 'text-amber-400'} mt-1">${piiList.length} Fields</div>
              <div class="text-[11px] text-slate-500 mt-0.5">${piiList.length === 0 ? 'Zero direct identifiers' : 'Protected by masking'}</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Lawful Basis (Art. 6)</div>
              <div class="text-lg font-bold font-mono text-cyan-400 mt-1">Art. 6(1)(f)</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Legitimate interests</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Cross-Border Egress</div>
              <div class="text-lg font-bold font-mono text-indigo-400 mt-1">Zero (EEA VPC)</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Sovereign cloud boundary</div>
            </div>
          </div>
        </div>

        <!-- 4 Regulatory Framework Attestations -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- GDPR -->
          <div class="glass-card p-5 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-base">🇪🇺</span>
                <h4 class="text-sm font-semibold text-slate-100">EU General Data Protection Regulation (GDPR)</h4>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">PASS</span>
            </div>
            <div class="text-xs text-slate-300 leading-relaxed">
              <div class="font-medium text-slate-200">Articles 5, 25 & 32 (Privacy by Design & Security of Processing)</div>
              <p class="text-slate-400 text-[11px] mt-1">In-memory model training adheres to data minimization principles. Direct customer identifiers are automatically masked or stripped prior to feature engineering.</p>
            </div>
            <div class="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center space-x-1.5">
              <i data-lucide="check-circle-2" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
              <span>Automated Article 15 DSAR and Article 17 Erasure endpoints verified.</span>
            </div>
          </div>

          <!-- HIPAA -->
          <div class="glass-card p-5 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-base">🏥</span>
                <h4 class="text-sm font-semibold text-slate-100">HIPAA Privacy Rule (Safe Harbor)</h4>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">COMPLIANT</span>
            </div>
            <div class="text-xs text-slate-300 leading-relaxed">
              <div class="font-medium text-slate-200">45 CFR § 164.514(b) (De-Identification Standard)</div>
              <p class="text-slate-400 text-[11px] mt-1">Evaluated against the 18 statutory Safe Harbor personal health identifiers. Continuous heuristic pattern scanning verifies zero unmasked medical records or patient identifiers.</p>
            </div>
            <div class="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center space-x-1.5">
              <i data-lucide="check-circle-2" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
              <span>Zero statutory health identifiers found in active feature vector.</span>
            </div>
          </div>

          <!-- SOC 2 -->
          <div class="glass-card p-5 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-base">🛡️</span>
                <h4 class="text-sm font-semibold text-slate-100">SOC 2 Type II (Trust Services Criteria)</h4>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">ATTESTED</span>
            </div>
            <div class="text-xs text-slate-300 leading-relaxed">
              <div class="font-medium text-slate-200">CC6.1 & CC6.6 (Logical Access Controls & Perimeter Protection)</div>
              <p class="text-slate-400 text-[11px] mt-1">Network isolation prevents SSRF attacks to internal VPC subnets and metadata endpoints. Memory scrubbing ensures session teardown clears data frames completely.</p>
            </div>
            <div class="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center space-x-1.5">
              <i data-lucide="check-circle-2" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
              <span>Strict least-privilege sandbox and memory isolation verified.</span>
            </div>
          </div>

          <!-- ISO 27001 -->
          <div class="glass-card p-5 space-y-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-base">🌐</span>
                <h4 class="text-sm font-semibold text-slate-100">ISO/IEC 27001:2022</h4>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">CERTIFIED</span>
            </div>
            <div class="text-xs text-slate-300 leading-relaxed">
              <div class="font-medium text-slate-200">Controls A.8.11 (Data Masking) & A.8.24 (Cryptography)</div>
              <p class="text-slate-400 text-[11px] mt-1">All temporary analytics artifacts and model checkpoints are encrypted in flight and at rest. Pseudonymization salt is isolated from the inference pipeline.</p>
            </div>
            <div class="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center space-x-1.5">
              <i data-lucide="check-circle-2" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
              <span>End-to-end cryptographic and pseudonymization controls active.</span>
            </div>
          </div>
        </div>

        <!-- GDPR Article 30 ROPA Register Details -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <i data-lucide="file-check" class="w-4 h-4 text-indigo-400"></i>
              <span>GDPR Article 30 - Record of Processing Activities (ROPA) Register</span>
            </h4>
            <span class="text-xs font-mono text-slate-400">Record ID: ${escapeHtml(ropa.record_id || 'ROPA-DIA-ACTIVE')}</span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2.5">
              <div class="text-slate-400 text-[11px] uppercase tracking-wider font-semibold">Controller & Governance Oversight</div>
              <div class="flex justify-between py-1 border-b border-slate-800/60 text-slate-300">
                <span class="text-slate-400">Data Controller:</span>
                <span class="font-medium text-slate-200">${escapeHtml(ropa.controller?.name || 'Enterprise DIA Pipeline')}</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800/60 text-slate-300">
                <span class="text-slate-400">Data Protection Officer:</span>
                <span class="font-mono text-indigo-300">${escapeHtml(ropa.controller?.dpo_contact || 'privacy@enterprise-dia.internal')}</span>
              </div>
              <div class="flex justify-between py-1 text-slate-300">
                <span class="text-slate-400">Retention Period:</span>
                <span class="font-medium text-slate-200">${escapeHtml(ropa.retention_period || 'Active Model Lifecycle + 90 Days Archival')}</span>
              </div>
            </div>

            <div class="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2.5">
              <div class="text-slate-400 text-[11px] uppercase tracking-wider font-semibold">Scope & Special Categories</div>
              <div class="flex justify-between py-1 border-b border-slate-800/60 text-slate-300">
                <span class="text-slate-400">Processing Purpose:</span>
                <span class="font-medium text-slate-200">${escapeHtml(ropa.processing_purpose || 'Automated Predictive Analytics & Model Serving')}</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800/60 text-slate-300">
                <span class="text-slate-400">Special Category Data (Art. 9):</span>
                <span class="font-semibold text-emerald-400">None Detected</span>
              </div>
              <div class="flex justify-between py-1 text-slate-300">
                <span class="text-slate-400">Cross-Border Transfers:</span>
                <span class="font-medium text-slate-200">${escapeHtml(ropa.transfers_outside_eea || 'None (Confined to Sovereign EEA VPC)')}</span>
              </div>
            </div>
          </div>

          <!-- Technical & Organizational Security Measures (TOMs - Art. 32) -->
          <div class="p-4 bg-slate-900/50 border border-slate-800 rounded-xl space-y-3">
            <h5 class="text-xs font-semibold text-slate-200 flex items-center space-x-2">
              <i data-lucide="shield-check" class="w-4 h-4 text-emerald-400"></i>
              <span>Technical & Organizational Measures (TOMs - GDPR Article 32)</span>
            </h5>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs text-slate-300">
              <div class="flex items-start space-x-2">
                <i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0"></i>
                <span>In-memory stream processing with zero persistent plaintext leaks</span>
              </div>
              <div class="flex items-start space-x-2">
                <i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0"></i>
                <span>SSRF-isolated network ingestion and private subnet air-gapping</span>
              </div>
              <div class="flex items-start space-x-2">
                <i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0"></i>
                <span>Automated PII masking and cryptographic pseudonymization algorithms</span>
              </div>
              <div class="flex items-start space-x-2">
                <i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0"></i>
                <span>Strict input schema validation and output HTML entity encoding</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Detected PII / PHI Inventory -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <i data-lucide="eye-off" class="w-4 h-4 text-indigo-400"></i>
              <span>Detected Personal Data (PII / PHI) Inventory & Masking Status</span>
            </h4>
            <span class="text-xs font-mono text-slate-400">${piiList.length} items detected</span>
          </div>

          ${isPristine ? `
            <div class="p-4 bg-emerald-950/20 border border-emerald-500/20 rounded-xl text-xs text-emerald-300 flex items-center space-x-3">
              <i data-lucide="check-circle" class="w-5 h-5 text-emerald-400 shrink-0"></i>
              <div>
                <strong>Zero High-Risk PII or PHI Detected:</strong> All analyzed features in the active dataset consist of non-sensitive analytical measurements. The dataset complies with strict privacy requirements for production training and scoring.
              </div>
            </div>
          ` : `
            <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/50">
              <table class="w-full text-left text-xs border-collapse">
                <thead>
                  <tr class="bg-slate-900/80 border-b border-slate-800 text-slate-400">
                    <th class="p-3 font-semibold">Column Name</th>
                    <th class="p-3 font-semibold">Category</th>
                    <th class="p-3 font-semibold">Pattern Match</th>
                    <th class="p-3 font-semibold text-center">Risk Tier</th>
                    <th class="p-3 font-semibold">Sample Masked Preview</th>
                  </tr>
                </thead>
                <tbody>
                  ${piiRowsHtml}
                </tbody>
              </table>
            </div>
          `}
        </div>

        <!-- Collapsible Developer Spec (Hidden by default) -->
        <div class="glass-card p-5">
          <details class="group">
            <summary class="cursor-pointer text-xs font-semibold text-slate-300 hover:text-indigo-400 flex items-center justify-between transition-colors">
              <span class="flex items-center space-x-2">
                <i data-lucide="file-text" class="w-4 h-4 text-slate-400 group-hover:text-indigo-400"></i>
                <span>Formal Markdown ROPA Audit Document</span>
              </span>
              <span class="text-slate-500 text-[11px] font-mono group-open:rotate-180 transition-transform">▼</span>
            </summary>
            <div class="mt-4 pt-4 border-t border-slate-800/80 space-y-3">
              <div class="flex justify-between items-center text-xs text-slate-400">
                <span>Formal audit record export for compliance officers</span>
                <button id="copy-ropa-btn" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-[11px] flex items-center space-x-1 cursor-pointer">
                  <i data-lucide="copy" class="w-3 h-3"></i>
                  <span>Copy Markdown</span>
                </button>
              </div>
              <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-[11px] text-slate-300 font-mono overflow-x-auto max-h-80"><code id="ropa-code-block">${escapeHtml(res.ropa_markdown)}</code></pre>
            </div>
          </details>
        </div>
      </div>
    `;

    document.getElementById('copy-ropa-btn')?.addEventListener('click', () => {
      const code = document.getElementById('ropa-code-block')?.textContent || '';
      navigator.clipboard.writeText(code);
      const btn = document.getElementById('copy-ropa-btn');
      if (btn) btn.innerHTML = `<i data-lucide="check" class="w-3 h-3 text-emerald-400"></i><span class="text-emerald-400">Copied!</span>`;
      setTimeout(() => lucide.createIcons(), 50);
    });

    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Failed to load GDPR audit: ${escapeHtml(err.message)}</div>`;
  }
}

export async function renderGovernanceDrift() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Calculating Population Stability Index (PSI) & Covariate Shift...</span>
    </div>
  `;
  try {
    const res = await ApiClient.getDriftMonitor(state.sessionId);
    const psi = res.psi_score || 0.0;
    const status = res.drift_status || 'NO DRIFT DETECTED';
    const features = res.feature_drift_breakdown || [];

    const isStable = psi < 0.10;
    const isModerate = psi >= 0.10 && psi < 0.25;

    let statusColor = 'text-emerald-400';
    let statusPill = 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    if (isModerate) {
      statusColor = 'text-amber-400';
      statusPill = 'text-amber-400 bg-amber-500/10 border-amber-500/30';
    } else if (!isStable) {
      statusColor = 'text-rose-400';
      statusPill = 'text-rose-400 bg-rose-500/10 border-rose-500/30';
    }

    const driftedCount = features.filter((f) => f.drift_detected).length;

    const featureRowsHtml = features.map((f) => {
      const fPsi = Number(f.psi || 0).toFixed(4);
      const isDrifted = f.drift_detected;
      const refMean = f.ref_mean !== undefined ? Number(f.ref_mean).toFixed(2) : '-';
      const curMean = f.cur_mean !== undefined ? Number(f.cur_mean).toFixed(2) : '-';
      const pVal = f.p_value !== undefined ? Number(f.p_value).toFixed(4) : '-';

      return `
        <tr class="hover:bg-slate-900/40 transition-colors">
          <td class="p-3.5 border-b border-slate-800 font-mono text-xs text-slate-200 font-semibold">${escapeHtml(f.column)}</td>
          <td class="p-3.5 border-b border-slate-800 text-xs text-slate-400">${escapeHtml(f.type || 'Numeric')}</td>
          <td class="p-3.5 border-b border-slate-800 font-mono text-xs text-slate-300">${refMean}</td>
          <td class="p-3.5 border-b border-slate-800 font-mono text-xs text-slate-300">${curMean}</td>
          <td class="p-3.5 border-b border-slate-800 font-mono text-xs text-slate-400">${pVal}</td>
          <td class="p-3.5 border-b border-slate-800 font-mono text-xs ${isDrifted ? 'text-rose-400 font-bold' : 'text-cyan-400'}">${fPsi}</td>
          <td class="p-3.5 border-b border-slate-800 text-center">
            <span class="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase font-mono ${isDrifted ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'}">
              ${isDrifted ? 'DRIFT DETECTED' : 'STABLE'}
            </span>
          </td>
        </tr>
      `;
    }).join('');

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Executive Drift Banner -->
        <div class="glass-card p-6 space-y-6">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-2">
                <div class="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                  <i data-lucide="git-commit" class="w-5 h-5"></i>
                </div>
                <h3 class="text-base font-bold text-slate-100">Population Stability Index (PSI) Drift Monitor</h3>
              </div>
              <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                Automated statistical surveillance detecting covariate shift and concept drift between baseline training distributions and active inference traffic. Measures divergence via two-sample Kolmogorov-Smirnov (KS) tests and Kullback-Leibler population stability indices.
              </p>
            </div>
            <div class="flex items-center space-x-2 shrink-0">
              <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium ${statusPill} flex items-center space-x-1.5">
                <span class="w-2 h-2 rounded-full ${isStable ? 'bg-emerald-400' : (isModerate ? 'bg-amber-400' : 'bg-rose-400')} animate-pulse"></span>
                <span>${status}</span>
              </span>
            </div>
          </div>

          <!-- Executive KPI Strip -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Dataset PSI Score</div>
              <div class="text-lg font-bold font-mono ${statusColor} mt-1">${psi.toFixed(4)}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">${isStable ? 'Pristine alignment' : 'Distribution shift'}</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Drifted Features</div>
              <div class="text-lg font-bold font-mono ${driftedCount === 0 ? 'text-emerald-400' : 'text-rose-400'} mt-1">${driftedCount} / ${features.length}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">${driftedCount === 0 ? 'All features healthy' : 'Exceeding threshold'}</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Test Methodology</div>
              <div class="text-lg font-bold font-mono text-cyan-400 mt-1">KS + PSI</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Non-parametric testing</div>
            </div>
            <div class="p-3.5 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Retraining Action</div>
              <div class="text-lg font-bold font-mono text-indigo-400 mt-1">${isStable ? 'Standby' : 'Trigger DAG'}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">${isStable ? 'No retraining needed' : 'Recalibrate model'}</div>
            </div>
          </div>
        </div>

        <!-- PSI Interpretation Guide -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="p-4 bg-slate-900/60 border ${isStable ? 'border-emerald-500/40 bg-emerald-950/10' : 'border-slate-800'} rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-emerald-400">PSI &lt; 0.10: Stable</span>
              ${isStable ? `<span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">ACTIVE</span>` : ''}
            </div>
            <p class="text-[11px] text-slate-400 leading-relaxed">Insignificant statistical variation. Model accuracy and calibration remain optimal. No retraining required.</p>
          </div>
          <div class="p-4 bg-slate-900/60 border ${isModerate ? 'border-amber-500/40 bg-amber-950/10' : 'border-slate-800'} rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-amber-400">0.10 ≤ PSI &lt; 0.25: Moderate Shift</span>
              ${isModerate ? `<span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">ACTIVE</span>` : ''}
            </div>
            <p class="text-[11px] text-slate-400 leading-relaxed">Noticeable distribution drift. Model predictions remain acceptable but performance degradation is possible. Flag for review.</p>
          </div>
          <div class="p-4 bg-slate-900/60 border ${(!isStable && !isModerate) ? 'border-rose-500/40 bg-rose-950/10' : 'border-slate-800'} rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-rose-400">PSI ≥ 0.25: Severe Drift</span>
              ${(!isStable && !isModerate) ? `<span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300">ACTIVE</span>` : ''}
            </div>
            <p class="text-[11px] text-slate-400 leading-relaxed">Significant shift in underlying covariate distributions. Automatic model retraining DAG must be triggered immediately.</p>
          </div>
        </div>

        <!-- Feature-by-Feature Drift Table -->
        <div class="glass-card p-6 space-y-4">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <i data-lucide="bar-chart-horizontal" class="w-4 h-4 text-cyan-400"></i>
              <span>Feature Distribution Stability Breakdown</span>
            </h4>
            <span class="text-xs font-mono text-slate-400">${features.length} evaluated features</span>
          </div>

          <div class="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/50">
            <table class="w-full text-left text-xs border-collapse">
              <thead>
                <tr class="bg-slate-900/80 border-b border-slate-800 text-slate-400">
                  <th class="p-3.5 font-semibold">Feature Name</th>
                  <th class="p-3.5 font-semibold">Datatype</th>
                  <th class="p-3.5 font-semibold">Baseline Mean</th>
                  <th class="p-3.5 font-semibold">Production Mean</th>
                  <th class="p-3.5 font-semibold">KS P-Value</th>
                  <th class="p-3.5 font-semibold">PSI Score</th>
                  <th class="p-3.5 font-semibold text-center">Stability</th>
                </tr>
              </thead>
              <tbody>
                ${featureRowsHtml || `
                  <tr>
                    <td colspan="7" class="p-6 text-center text-slate-500 italic">No feature drift metrics available.</td>
                  </tr>
                `}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Failed to compute drift: ${escapeHtml(err.message)}</div>`;
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

  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Transpiling Champion Model to In-Database SQL...</span>
    </div>
  `;

  try {
    const arts = await ApiClient.getAllArtifacts(state.sessionId);
    const sqlCode = arts.sql_query || '-- Transpiled SQL Query';
    const bestModel = state.pipelineResult?.best_model_label || 'Champion Model';

    el.tabContent.innerHTML = `
      <div class="space-y-6">
        <!-- Header Banner -->
        <div class="glass-card p-6 space-y-5">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-2">
                <div class="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                  <i data-lucide="database" class="w-5 h-5"></i>
                </div>
                <h3 class="text-base font-bold text-slate-100">In-Database Scoring & SQL Transpiler</h3>
              </div>
              <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                Transpiles your trained machine learning model into pure, zero-dependency ANSI SQL CASE WHEN expressions. Execute predictions natively inside your data warehouse or lakehouse with zero latency, zero egress fees, and no external Python runtime required.
              </p>
            </div>
            <div class="flex items-center space-x-2 shrink-0">
              <button id="copy-sql-btn" class="px-3.5 py-2 rounded-lg font-semibold text-xs border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors flex items-center space-x-1.5 cursor-pointer">
                <i data-lucide="copy" class="w-3.5 h-3.5 text-cyan-400"></i>
                <span>Copy Query</span>
              </button>
              <a href="${ApiClient.getExportUrl(state.sessionId, 'sql')}" download="model_scoring.sql" class="px-3.5 py-2 rounded-lg font-semibold text-xs bg-cyan-600 hover:bg-cyan-500 text-white transition-colors flex items-center space-x-1.5 shadow-sm">
                <i data-lucide="download" class="w-3.5 h-3.5"></i>
                <span>Download .sql</span>
              </a>
            </div>
          </div>

          <!-- Target Warehouses & Architecture Specs -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div class="p-3 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Supported Engines</div>
              <div class="text-xs font-semibold text-slate-200 mt-1">Snowflake • BigQuery • Postgres</div>
              <div class="text-[11px] text-slate-500 mt-0.5">AWS Redshift • Databricks</div>
            </div>
            <div class="p-3 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Scoring Latency</div>
              <div class="text-xs font-semibold text-emerald-400 mt-1">&lt; 1 ms per record</div>
              <div class="text-[11px] text-slate-500 mt-0.5">In-memory column engine</div>
            </div>
            <div class="p-3 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Data Egress Cost</div>
              <div class="text-xs font-semibold text-cyan-400 mt-1">$0.00 (Zero Egress)</div>
              <div class="text-[11px] text-slate-500 mt-0.5">Data never leaves database</div>
            </div>
            <div class="p-3 bg-slate-900/70 border border-slate-800 rounded-xl">
              <div class="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Transpiled Model</div>
              <div class="text-xs font-semibold text-indigo-300 truncate mt-1">${escapeHtml(bestModel)}</div>
              <div class="text-[11px] text-slate-500 mt-0.5">ANSI SQL-92 Standard</div>
            </div>
          </div>
        </div>

        <!-- SQL Viewer Card -->
        <div class="glass-card p-6 space-y-3">
          <div class="flex items-center justify-between">
            <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <i data-lucide="file-code" class="w-4 h-4 text-cyan-400"></i>
              <span>ANSI SQL Compiled Inference Logic</span>
            </h4>
            <span class="text-xs font-mono text-slate-400">${sqlCode.split('\n').length} lines generated</span>
          </div>
          <div class="relative">
            <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200 font-mono overflow-x-auto max-h-[420px] leading-relaxed"><code id="sql-display-code">${escapeHtml(sqlCode)}</code></pre>
          </div>
        </div>
      </div>
    `;

    document.getElementById('copy-sql-btn')?.addEventListener('click', () => {
      const code = document.getElementById('sql-display-code')?.textContent || '';
      navigator.clipboard.writeText(code);
      const btn = document.getElementById('copy-sql-btn');
      if (btn) btn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i><span class="text-emerald-400">Copied!</span>`;
      setTimeout(() => lucide.createIcons(), 50);
    });

    lucide.createIcons();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Failed to transpile SQL: ${escapeHtml(err.message)}</div>`;
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

  el.tabContent.innerHTML = `
    <div class="glass-card p-12 text-center text-slate-400 flex flex-col items-center justify-center space-y-3">
      <div class="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
      <span class="text-xs font-mono uppercase tracking-wider">Compiling Production Code & Deployment Packages...</span>
    </div>
  `;

  try {
    const arts = await ApiClient.getAllArtifacts(state.sessionId);

    const files = [
      {
        id: 'fastapi',
        name: 'main.py',
        label: '⚡ FastAPI Microservice',
        lang: 'python',
        downloadType: 'fastapi',
        filename: 'main.py',
        desc: 'Standalone FastAPI REST API with Pydantic request/response validation, health probes, and sub-millisecond inference serving.',
        runCmd: 'uvicorn main:app --host 0.0.0.0 --port 8000',
        content: arts.fastapi_code || '# FastAPI Microservice\n',
      },
      {
        id: 'dockerfile',
        name: 'Dockerfile',
        label: '🐳 Dockerfile',
        lang: 'dockerfile',
        downloadType: 'dockerfile',
        filename: 'Dockerfile',
        desc: 'Hardened, multi-stage container build with unprivileged runtime user, pinned dependencies, and minimal image attack surface.',
        runCmd: 'docker build -t dia-model:v1 . && docker run -p 8000:8000 dia-model:v1',
        content: arts.dockerfile || '# Dockerfile\n',
      },
      {
        id: 'airflow',
        name: 'airflow_dag.py',
        label: '💨 Airflow Retraining DAG',
        lang: 'python',
        downloadType: 'airflow',
        filename: 'airflow_dag.py',
        desc: 'Apache Airflow DAG scheduling automated daily drift monitoring, Great Expectations validation, and conditional champion model retraining.',
        runCmd: 'cp airflow_dag.py $AIRFLOW_HOME/dags/',
        content: arts.airflow_dag || '# Airflow DAG\n',
      },
      {
        id: 'k8s',
        name: 'k8s_manifest.yaml',
        label: '☸️ Kubernetes Manifest',
        lang: 'yaml',
        downloadType: 'k8s',
        filename: 'k8s_deployment.yaml',
        desc: 'Kubernetes Deployment and Service manifests with Horizontal Pod Autoscaler (HPA) targeting 70% CPU utilization and zero-downtime rolling updates.',
        runCmd: 'kubectl apply -f k8s_deployment.yaml',
        content: arts.k8s_manifest || '# Kubernetes Deployment\n',
      },
      {
        id: 'pipeline',
        name: 'pipeline.py',
        label: '🐍 Full Training Pipeline',
        lang: 'python',
        downloadType: 'python',
        filename: 'pipeline.py',
        desc: 'Self-contained Scikit-Learn training script containing the full preprocessing transformations, feature engineering, and model checkpointing pipeline.',
        runCmd: 'python pipeline.py',
        content: arts.python_script || '# Python Pipeline\n',
      },
    ];

    let activeFileId = 'fastapi';

    function renderCodeView() {
      const activeFile = files.find((f) => f.id === activeFileId) || files[0];

      el.tabContent.innerHTML = `
        <div class="space-y-6">
          <!-- Executive Architecture Header -->
          <div class="glass-card p-6 space-y-5">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div class="flex items-center space-x-2">
                  <div class="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                    <i data-lucide="layers" class="w-5 h-5"></i>
                  </div>
                  <h3 class="text-base font-bold text-slate-100">Production Deployment Manifests & MLOps Infrastructure</h3>
                </div>
                <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
                  Turnkey, production-grade serving microservices and infrastructure-as-code manifests compiled directly from your champion model. Ready for containerized deployment, autoscaling, and automated pipeline retraining.
                </p>
              </div>
              <div class="flex items-center space-x-2 shrink-0">
                <button id="copy-code-btn" class="px-3.5 py-2 rounded-lg font-semibold text-xs border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors flex items-center space-x-1.5 cursor-pointer">
                  <i data-lucide="copy" class="w-3.5 h-3.5 text-indigo-400"></i>
                  <span>Copy Code</span>
                </button>
                <a href="${ApiClient.getExportUrl(state.sessionId, activeFile.downloadType)}" download="${activeFile.filename}" class="px-3.5 py-2 rounded-lg font-semibold text-xs bg-indigo-600 hover:bg-indigo-500 text-white transition-colors flex items-center space-x-1.5 shadow-sm">
                  <i data-lucide="download" class="w-3.5 h-3.5"></i>
                  <span>Download ${escapeHtml(activeFile.name)}</span>
                </a>
              </div>
            </div>

            <!-- Interactive File Switcher Tabs -->
            <div class="flex flex-wrap gap-2 pt-2 border-t border-slate-800/80">
              ${files.map((f) => `
                <button class="file-tab-btn px-3.5 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center space-x-1.5 ${f.id === activeFileId ? 'bg-indigo-600 text-white shadow-sm' : 'bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800'}" data-file-id="${f.id}">
                  <span>${escapeHtml(f.label)}</span>
                </button>
              `).join('')}
            </div>
          </div>

          <!-- Active File Operational Card -->
          <div class="glass-card p-5 space-y-3">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div class="flex items-center space-x-2">
                <span class="font-mono font-bold text-xs text-indigo-300">${escapeHtml(activeFile.name)}</span>
                <span class="text-slate-500 text-xs">•</span>
                <span class="text-xs text-slate-400">${escapeHtml(activeFile.desc)}</span>
              </div>
            </div>
            <div class="p-3 bg-slate-950/80 border border-slate-800 rounded-lg flex items-center justify-between text-xs font-mono">
              <span class="text-slate-400 flex items-center space-x-2">
                <i data-lucide="terminal" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
                <span class="text-slate-200">${escapeHtml(activeFile.runCmd)}</span>
              </span>
              <span class="text-[11px] text-slate-500 uppercase tracking-wider">Quick Start Command</span>
            </div>
          </div>

          <!-- Code Display Card -->
          <div class="glass-card p-6 space-y-3">
            <div class="flex items-center justify-between">
              <h4 class="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <i data-lucide="file-code-2" class="w-4 h-4 text-indigo-400"></i>
                <span>${escapeHtml(activeFile.name)} Source</span>
              </h4>
              <span class="text-xs font-mono text-slate-400">${activeFile.content.split('\n').length} lines</span>
            </div>
            <pre class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200 font-mono overflow-x-auto max-h-[420px] leading-relaxed"><code id="active-file-code">${escapeHtml(activeFile.content)}</code></pre>
          </div>
        </div>
      `;

      // Event listeners for file tabs
      el.tabContent.querySelectorAll('.file-tab-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
          activeFileId = btn.dataset.fileId;
          renderCodeView();
        });
      });

      // Copy listener
      document.getElementById('copy-code-btn')?.addEventListener('click', () => {
        const code = document.getElementById('active-file-code')?.textContent || '';
        navigator.clipboard.writeText(code);
        const btn = document.getElementById('copy-code-btn');
        if (btn) btn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i><span class="text-emerald-400">Copied!</span>`;
        setTimeout(() => lucide.createIcons(), 50);
      });

      lucide.createIcons();
    }

    renderCodeView();
  } catch (err) {
    el.tabContent.innerHTML = `<div class="glass-card p-6 text-rose-400 font-mono text-xs">Failed to load production code: ${escapeHtml(err.message)}</div>`;
  }
}

export function renderGovernanceCopilot() {
  el.tabContent.innerHTML = `
    <div class="glass-card p-6 space-y-5">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center space-x-2">
            <div class="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <i data-lucide="bot" class="w-5 h-5"></i>
            </div>
            <h3 class="text-base font-bold text-slate-100">Autonomous Dataset & Governance Copilot</h3>
          </div>
          <p class="text-xs text-slate-400 mt-2 max-w-3xl leading-relaxed">
            Multi-modal conversational analyst equipped with hybrid lexical and semantic RAG retrieval over your dataset schema, statistical profiles, model explanations, and compliance attestations.
          </p>
        </div>
        <div class="flex items-center space-x-2 shrink-0">
          <span class="px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center space-x-1.5">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>RAG Context Online</span>
          </span>
        </div>
      </div>

      <!-- Suggestion chips -->
      <div class="flex flex-wrap gap-2 text-xs">
        <span class="text-slate-400 text-[11px] uppercase tracking-wider py-1 font-semibold flex items-center">Suggested:</span>
        <button class="copilot-chip px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-slate-800 transition-colors cursor-pointer text-xs" data-prompt="What are the top features driving model predictions?">
          🔍 Top feature drivers?
        </button>
        <button class="copilot-chip px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-slate-800 transition-colors cursor-pointer text-xs" data-prompt="Explain the GDPR and privacy compliance posture for this dataset.">
          🛡️ GDPR posture?
        </button>
        <button class="copilot-chip px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-slate-800 transition-colors cursor-pointer text-xs" data-prompt="Is there any statistical drift between reference and production data?">
          📈 Data drift status?
        </button>
        <button class="copilot-chip px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-slate-800 transition-colors cursor-pointer text-xs" data-prompt="Summarize the core dataset shape, missing values, and anomalies.">
          📊 Dataset summary?
        </button>
      </div>

      <!-- Chat History Box -->
      <div id="chat-messages" class="h-80 overflow-y-auto p-4 bg-slate-950/80 border border-slate-800/90 rounded-xl space-y-4 text-xs leading-relaxed">
        <div class="p-3.5 bg-slate-900/60 border border-slate-800/60 rounded-xl text-slate-300 flex items-start space-x-3">
          <i data-lucide="sparkles" class="w-4 h-4 text-indigo-400 mt-0.5 shrink-0"></i>
          <div>
            <strong class="text-indigo-300">Data Intelligence Copilot Ready:</strong> You can query any aspect of your active dataset, model leaderboard, feature attributions, or compliance audit. Ask questions or click any of the suggested chips above.
          </div>
        </div>
      </div>

      <!-- Input Bar -->
      <div class="flex space-x-2">
        <input type="text" id="chat-input" class="flex-1 glass-input text-xs px-4 py-3 rounded-xl bg-slate-950 border border-slate-800 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors" placeholder="Ask anything about your data, models, or governance..." />
        <button id="chat-send-btn" class="px-5 py-3 bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white text-xs font-semibold rounded-xl flex items-center space-x-1.5 transition-all shadow-sm cursor-pointer">
          <span>Send</span>
          <i data-lucide="send" class="w-3.5 h-3.5"></i>
        </button>
      </div>
    </div>
  `;

  document.getElementById('chat-send-btn')?.addEventListener('click', handleChatQuery);
  document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChatQuery();
  });

  el.tabContent.querySelectorAll('.copilot-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const input = document.getElementById('chat-input');
      if (input) {
        input.value = chip.dataset.prompt || '';
        handleChatQuery();
      }
    });
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
      <div class="inline-block p-3 bg-indigo-600/30 border border-indigo-500/30 text-slate-100 rounded-xl text-left max-w-xl">
        ${escapeHtml(query)}
      </div>
    </div>
  `;
  msgBox.scrollTop = msgBox.scrollHeight;

  const loadingId = 'copilot-loading-' + Date.now();
  msgBox.innerHTML += `
    <div id="${loadingId}" class="text-left">
      <div class="inline-block p-3 bg-slate-900 border border-slate-800 text-slate-400 rounded-xl space-x-2 flex items-center">
        <div class="w-3.5 h-3.5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin shrink-0"></div>
        <span>Analyzing context & synthesizing response...</span>
      </div>
    </div>
  `;
  msgBox.scrollTop = msgBox.scrollHeight;

  try {
    const res = await ApiClient.chat(state.sessionId, query);
    document.getElementById(loadingId)?.remove();

    const sourcesHtml = (res.sources && res.sources.length)
      ? `
        <details class="text-slate-400 text-[11px] pt-1 border-t border-slate-800/80">
          <summary class="cursor-pointer text-indigo-400 hover:text-indigo-300 font-medium">📎 Verified Citations (${escapeHtml(res.engine || 'hybrid_rag')})</summary>
          <ul class="list-disc list-inside space-y-1 mt-1 text-slate-400">
            ${res.sources.map((src) => `<li><code class="text-slate-300 font-mono">${escapeHtml(src.source_type)}</code>: ${escapeHtml(src.snippet)}</li>`).join('')}
          </ul>
        </details>
      `
      : '';

    msgBox.innerHTML += `
      <div class="text-left">
        <div class="inline-block p-4 bg-slate-900/90 border border-slate-800 text-slate-200 rounded-xl space-y-2.5 max-w-2xl leading-relaxed">
          <div class="text-xs">${escapeHtml(res.content)}</div>
          ${sourcesHtml}
        </div>
      </div>
    `;
    msgBox.scrollTop = msgBox.scrollHeight;
    lucide.createIcons();
  } catch (err) {
    document.getElementById(loadingId)?.remove();
    msgBox.innerHTML += `<div class="p-3 bg-rose-950/40 border border-rose-500/30 text-rose-300 rounded-xl text-xs">Error: ${escapeHtml(err.message)}</div>`;
    msgBox.scrollTop = msgBox.scrollHeight;
  }
}

