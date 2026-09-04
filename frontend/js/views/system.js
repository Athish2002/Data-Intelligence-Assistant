/**
 * frontend/js/views/system.js
 * ───────────────────────────
 * Enterprise System Health & Resource Monitor HUD.
 * Provides real-time process RSS memory gauges, host RAM/CPU telemetry,
 * session store cache inspector, and interactive garbage collection controls.
 */

import { ApiClient } from '../api.js';

let systemPollTimer = null;
let isPollingActive = true;

export async function openSystemMetricsModal() {
  let modal = document.getElementById('system-metrics-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'system-metrics-modal';
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm modal-backdrop-anim';
    document.body.appendChild(modal);
  }

  modal.innerHTML = `
    <div class="glass-card w-full max-w-4xl max-h-[90dvh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden modal-card-anim" style="background-color: var(--card-bg); border-color: var(--card-border);">
      <!-- Header -->
      <div class="px-6 py-4 border-b flex items-center justify-between" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <div class="flex items-center space-x-3">
          <div class="w-8 h-8 rounded-lg flex items-center justify-center bg-[#00e575]/10 border border-[#00e575]/30">
            <i data-lucide="activity" class="w-4 h-4 text-[#00e575]"></i>
          </div>
          <div>
            <h3 class="text-sm font-bold font-mono tracking-tight" style="color: var(--text-primary);">System Health & Resource Monitor</h3>
            <p class="text-[11px] font-mono" style="color: var(--text-muted);">Real-time memory profiling, CPU load, and session cache management</p>
          </div>
        </div>
        <div class="flex items-center space-x-3">
          <div class="flex items-center space-x-1.5 px-2.5 py-1 rounded-full border text-[10px] font-mono pulse-pill" style="background-color: var(--badge-live-bg); border-color: var(--badge-live-border); color: var(--badge-live-text);">
            <span class="w-1.5 h-1.5 rounded-full bg-[#00e575]"></span>
            <span>PROTECTED • AUTO-GC ACTIVE</span>
          </div>
          <button onclick="window.closeSystemMetrics()" class="p-1 rounded-lg hover:bg-white/10 transition-colors cursor-pointer" style="color: var(--text-muted);" title="Close (Esc)">
            <i data-lucide="x" class="w-4 h-4"></i>
          </button>
        </div>
      </div>

      <!-- Content Body -->
      <div id="system-metrics-body" class="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
        <div class="text-center py-12 text-sm font-mono" style="color: var(--text-muted);">
          <i data-lucide="loader-2" class="w-6 h-6 animate-spin mx-auto mb-2 text-[#00e575]"></i>
          Querying hardware telemetry...
        </div>
      </div>

      <!-- Footer Toolbar -->
      <div class="px-6 py-3 border-t flex items-center justify-between font-mono text-[11px]" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <div class="flex items-center space-x-3">
          <button id="trigger-gc-btn" onclick="window.triggerManualGC()" class="px-3 py-1.5 rounded-lg border font-bold hover:bg-white/5 transition-all cursor-pointer flex items-center space-x-2 active:scale-95" style="background-color: var(--card-bg); border-color: var(--card-border); color: var(--text-primary);">
            <i data-lucide="trash-2" class="w-3.5 h-3.5 text-[#00e575]"></i>
            <span>Purge Inactive Sessions & Run GC</span>
          </button>
          <span id="gc-feedback-msg" class="text-[11px] text-[#00e575] font-semibold"></span>
        </div>
        <div class="flex items-center space-x-2">
          <span style="color: var(--text-muted);">Auto-Refresh (3s):</span>
          <button id="toggle-poll-btn" onclick="window.toggleSystemPolling()" class="px-2.5 py-0.5 rounded border text-[10px] font-bold cursor-pointer transition-colors" style="background-color: var(--card-bg); border-color: var(--card-border); color: var(--text-primary);">
            ON
          </button>
          <button onclick="window.refreshSystemMetrics()" class="p-1 rounded hover:bg-white/10 transition-colors cursor-pointer" title="Refresh Now" style="color: var(--text-muted);">
            <i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>
    </div>
  `;

  modal.classList.remove('hidden');
  lucide.createIcons();

  // Load immediate data
  await loadAndRenderMetrics();

  // Start polling
  if (systemPollTimer) clearInterval(systemPollTimer);
  systemPollTimer = setInterval(() => {
    if (isPollingActive && !modal.classList.contains('hidden')) {
      loadAndRenderMetrics();
    }
  }, 3000);
}

export function closeSystemMetricsModal() {
  const modal = document.getElementById('system-metrics-modal');
  if (modal) {
    modal.classList.add('hidden');
  }
  if (systemPollTimer) {
    clearInterval(systemPollTimer);
    systemPollTimer = null;
  }
}

async function loadAndRenderMetrics() {
  const container = document.getElementById('system-metrics-body');
  if (!container) return;

  try {
    const res = await fetch('/api/v1/system/metrics');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderMetricsDashboard(container, data);
    lucide.createIcons();
  } catch (err) {
    container.innerHTML = `
      <div class="p-4 rounded-xl border border-red-900/50 bg-red-950/20 text-red-400 font-mono text-xs">
        Failed to fetch system metrics: ${err.message}
      </div>
    `;
  }
}

function renderMetricsDashboard(container, d) {
  const rssGaugePct = Math.min(100, Math.round((d.process_memory_rss_mb / 1024) * 100));
  const rssColor = d.process_memory_rss_mb < 400 ? '#00e575' : (d.process_memory_rss_mb < 900 ? '#f59e0b' : '#f43f5e');
  const ramColor = d.system_memory_percent < 70 ? '#00e575' : (d.system_memory_percent < 85 ? '#f59e0b' : '#f43f5e');
  const cpuColor = d.cpu_percent < 50 ? '#00e575' : (d.cpu_percent < 80 ? '#f59e0b' : '#f43f5e');

  container.innerHTML = `
    <!-- Top KPI Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
      <!-- 1. Process Memory Footprint -->
      <div class="glass-card p-4 rounded-xl space-y-2 border" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <div class="flex items-center justify-between text-[11px] font-mono" style="color: var(--text-muted);">
          <span>PROCESS RSS RAM</span>
          <i data-lucide="cpu" class="w-3.5 h-3.5 text-[#00e575]"></i>
        </div>
        <div class="text-2xl font-bold font-mono tracking-tight" style="color: ${rssColor};">
          ${d.process_memory_rss_mb} <span class="text-xs font-normal" style="color: var(--text-muted);">MB</span>
        </div>
        <div class="metric-gauge-bar">
          <div class="metric-gauge-fill" style="width: ${rssGaugePct}%; background-color: ${rssColor};"></div>
        </div>
        <div class="flex justify-between text-[10px] font-mono" style="color: var(--text-muted);">
          <span>VMS: ${d.process_memory_vms_mb} MB</span>
          <span>Target: &lt; 500 MB</span>
        </div>
      </div>

      <!-- 2. System RAM Utilization -->
      <div class="glass-card p-4 rounded-xl space-y-2 border" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <div class="flex items-center justify-between text-[11px] font-mono" style="color: var(--text-muted);">
          <span>HOST SYSTEM RAM</span>
          <i data-lucide="hard-drive" class="w-3.5 h-3.5 text-[#00e575]"></i>
        </div>
        <div class="text-2xl font-bold font-mono tracking-tight" style="color: var(--text-primary);">
          ${d.system_memory_percent}%
        </div>
        <div class="metric-gauge-bar">
          <div class="metric-gauge-fill" style="width: ${d.system_memory_percent}%; background-color: ${ramColor};"></div>
        </div>
        <div class="flex justify-between text-[10px] font-mono" style="color: var(--text-muted);">
          <span>Used: ${d.system_memory_used_gb} GB</span>
          <span>Total: ${d.system_memory_total_gb} GB</span>
        </div>
      </div>

      <!-- 3. Host CPU Utilization -->
      <div class="glass-card p-4 rounded-xl space-y-2 border" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <div class="flex items-center justify-between text-[11px] font-mono" style="color: var(--text-muted);">
          <span>CPU UTILIZATION</span>
          <i data-lucide="zap" class="w-3.5 h-3.5 text-[#00e575]"></i>
        </div>
        <div class="text-2xl font-bold font-mono tracking-tight" style="color: var(--text-primary);">
          ${d.cpu_percent}%
        </div>
        <div class="metric-gauge-bar">
          <div class="metric-gauge-fill" style="width: ${d.cpu_percent}%; background-color: ${cpuColor};"></div>
        </div>
        <div class="flex justify-between text-[10px] font-mono" style="color: var(--text-muted);">
          <span>${d.cpu_cores_logical} Logical Cores</span>
          <span>${d.thread_count} Threads</span>
        </div>
      </div>

      <!-- 4. Active In-Memory Sessions -->
      <div class="glass-card p-4 rounded-xl space-y-2 border" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <div class="flex items-center justify-between text-[11px] font-mono" style="color: var(--text-muted);">
          <span>ACTIVE SESSIONS</span>
          <i data-lucide="layers" class="w-3.5 h-3.5 text-[#00e575]"></i>
        </div>
        <div class="text-2xl font-bold font-mono tracking-tight" style="color: #00e575;">
          ${d.active_sessions_count} <span class="text-xs font-normal" style="color: var(--text-muted);">/ ${d.max_sessions_capacity} Max</span>
        </div>
        <div class="metric-gauge-bar">
          <div class="metric-gauge-fill" style="width: ${Math.round((d.active_sessions_count / d.max_sessions_capacity) * 100)}%; background-color: #00e575;"></div>
        </div>
        <div class="flex justify-between text-[10px] font-mono" style="color: var(--text-muted);">
          <span>LRU Protected</span>
          <span>TTL: ${d.session_ttl_minutes}m</span>
        </div>
      </div>
    </div>

    <!-- Defensive Memory Protection Architecture Banner -->
    <div class="p-4 rounded-xl border flex items-start space-x-3 text-xs font-mono" style="background-color: var(--bg-secondary); border-color: var(--card-border);">
      <i data-lucide="shield-check" class="w-5 h-5 text-[#00e575] flex-shrink-0 mt-0.5"></i>
      <div class="space-y-1">
        <span class="font-bold text-sm" style="color: var(--text-primary);">Engine Memory Guardrails Active</span>
        <p class="leading-relaxed" style="color: var(--text-muted);">
          Bounded LRU store caps memory to <strong>${d.max_sessions_capacity} active sessions</strong> with an automatic <strong>${d.session_ttl_minutes}-minute idle eviction policy</strong>. Scikit-Learn Loky multiprocessing is capped at a safe maximum of <strong>4 worker cores</strong> on Windows to prevent thread pool cascades and memory thrashing.
        </p>
      </div>
    </div>

    <!-- Session Store Inspector Table -->
    <div class="space-y-2">
      <div class="flex items-center justify-between font-mono">
        <span class="font-bold text-xs" style="color: var(--text-primary);">Active In-Memory Sessions (${d.sessions_detail.length})</span>
        <span class="text-[11px]" style="color: var(--text-muted);">System Uptime: ${Math.round(d.uptime_seconds)}s • Python ${d.python_version}</span>
      </div>

      <div class="border rounded-xl overflow-hidden font-mono text-xs" style="border-color: var(--card-border); background-color: var(--bg-secondary);">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="border-b text-[10px] uppercase tracking-wider" style="border-color: var(--card-border); color: var(--text-muted); background-color: var(--card-bg);">
              <th class="p-2.5">Session ID</th>
              <th class="p-2.5">Dataset Goal / Objective</th>
              <th class="p-2.5">Shape</th>
              <th class="p-2.5">Memory</th>
              <th class="p-2.5">Best Model</th>
              <th class="p-2.5">Idle / Age</th>
              <th class="p-2.5 text-right">Action</th>
            </tr>
          </thead>
          <tbody class="divide-y" style="border-color: var(--card-border);">
            ${
              d.sessions_detail.length === 0
                ? `<tr><td colspan="7" class="p-6 text-center text-xs" style="color: var(--text-muted);">No active sessions in memory. Ingest a dataset to initialize.</td></tr>`
                : d.sessions_detail.map((s) => `
                  <tr class="hover:bg-white/[0.02] transition-colors">
                    <td class="p-2.5 font-bold text-[#00e575] truncate max-w-[120px]" title="${s.session_id}">
                      ${s.session_id.substring(0, 8)}...
                    </td>
                    <td class="p-2.5 max-w-[200px] truncate" style="color: var(--text-primary);" title="${s.goal}">
                      ${s.goal || 'General Analysis'}
                    </td>
                    <td class="p-2.5" style="color: var(--text-secondary);">
                      ${s.n_rows.toLocaleString()} × ${s.n_cols}
                    </td>
                    <td class="p-2.5 font-bold" style="color: var(--text-primary);">
                      ${s.memory_mb} MB
                    </td>
                    <td class="p-2.5" style="color: var(--text-muted);">
                      ${s.best_model || (s.has_pipeline ? 'Trained' : 'Raw Ingestion')}
                    </td>
                    <td class="p-2.5" style="color: var(--text-muted);">
                      ${Math.round(s.idle_seconds)}s idle / ${Math.round(s.age_seconds)}s age
                    </td>
                    <td class="p-2.5 text-right">
                      <button onclick="window.evictSession('${s.session_id}')" class="px-2 py-0.5 rounded border text-[10px] text-red-400 hover:bg-red-950/40 border-red-900/50 transition-colors cursor-pointer" title="Evict from memory">
                        Evict
                      </button>
                    </td>
                  </tr>
                `).join('')
            }
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// Global Handlers
window.openSystemMetrics = openSystemMetricsModal;
window.closeSystemMetrics = closeSystemMetricsModal;
window.refreshSystemMetrics = loadAndRenderMetrics;

window.toggleSystemPolling = function () {
  isPollingActive = !isPollingActive;
  const btn = document.getElementById('toggle-poll-btn');
  if (btn) {
    btn.textContent = isPollingActive ? 'ON' : 'OFF';
    btn.style.color = isPollingActive ? '#00e575' : '#858076';
  }
};

window.triggerManualGC = async function () {
  const btn = document.getElementById('trigger-gc-btn');
  const feedback = document.getElementById('gc-feedback-msg');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin text-[#00e575]"></i> <span>Sweeping Heap...</span>`;
    lucide.createIcons();
  }

  try {
    const res = await fetch('/api/v1/system/gc', { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (feedback) {
      feedback.textContent = `✓ Swept ${data.unreachable_objects_collected} objects (Reclaimed ${data.reclaimed_mb} MB, Current RSS: ${data.current_rss_mb} MB)`;
      setTimeout(() => {
        if (feedback) feedback.textContent = '';
      }, 5000);
    }
    await loadAndRenderMetrics();
  } catch (err) {
    if (feedback) {
      feedback.textContent = `GC Error: ${err.message}`;
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="trash-2" class="w-3.5 h-3.5 text-[#00e575]"></i> <span>Purge Inactive Sessions & Run GC</span>`;
      lucide.createIcons();
    }
  }
};

window.evictSession = async function (sessionId) {
  if (!confirm(`Are you sure you want to evict session ${sessionId.substring(0, 8)}... from RAM?`)) return;
  try {
    const res = await fetch(`/api/v1/system/sessions/${sessionId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadAndRenderMetrics();
  } catch (err) {
    alert(`Failed to evict session: ${err.message}`);
  }
};

// Keyboard listener for Escape key to close modal
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('system-metrics-modal');
    if (modal && !modal.classList.contains('hidden')) {
      closeSystemMetricsModal();
    }
  }
});
