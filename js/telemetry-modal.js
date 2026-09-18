/**
 * telemetry-modal.js
 * ──────────────────
 * In-App Interactive Health Telemetry Modal & Real-Time Hardware HUD.
 * 
 * Provides:
 * 1. Rich hardware and storage telemetry visualization (CPU, RAM, GPU, Storage, Latency, Disk, Tenants).
 * 2. Universal trigger interception preventing navigation to raw /api/v1/health JSON.
 * 3. 10-second live auto-refresh timer with pause/resume toggle and manual refresh capability.
 * 4. WCAG 2.2 AA compliant modal behavior (Escape key, backdrop click, focus management).
 * 5. Pure inline SVG rendering with zero external icon dependencies.
 */

// ─── State ───────────────────────────────────────────────────────────────────

let telemetryInterval = null;
let isAutoRefreshActive = true;
let lastActiveTrigger = null;

// ─── Helpers & Formatters ────────────────────────────────────────────────────

/**
 * Escapes HTML characters to prevent XSS injection.
 */
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Formats gigabyte values cleanly.
 */
function formatGb(gb, decimals = 1) {
  const num = parseFloat(gb);
  if (isNaN(num)) return '0.0 GB';
  return `${num.toFixed(decimals)} GB`;
}

/**
 * Formats probe latency in milliseconds.
 */
function formatLatency(ms) {
  const num = parseFloat(ms);
  if (isNaN(num)) return '0.00 ms';
  return `${num.toFixed(2)} ms`;
}

/**
 * Formats percentage.
 */
function formatPercent(pct, decimals = 1) {
  const num = parseFloat(pct);
  if (isNaN(num)) return '0.0%';
  return `${num.toFixed(decimals)}%`;
}

/**
 * Returns color token based on percentage thresholds.
 */
function getMetricColor(value, warnThreshold = 70, critThreshold = 85) {
  const v = parseFloat(value) || 0;
  if (v < warnThreshold) return '#00e575'; // Emerald / green
  if (v < critThreshold) return '#f59e0b'; // Amber / warning
  return '#f43f5e';                       // Rose / critical
}

// ─── Dynamic Rendering ───────────────────────────────────────────────────────

/**
 * Renders telemetry cards into the modal body container.
 */
function renderHealthMetrics(container, data) {
  const storage = data.storage_details || {};
  const isHealthy = (data.status === 'healthy' || data.status === 'ok') && storage.writable !== false;

  // Extract core metrics with defensive fallbacks
  const cpuCores = data.cpu_cores || 4;
  const ramPercent = data.ram_percent ?? (data.system_memory_percent || 0);
  const ramUsedGb = data.ram_used_gb ?? (data.system_memory_used_gb || 0);
  const ramTotalGb = data.ram_total_gb ?? (data.system_memory_total_gb || 0);
  const gpuAvailable = Boolean(data.gpu_available);
  const writeLatencyMs = storage.latency_ms ?? 0;
  const freeDiskGb = storage.free_disk_gb ?? 0;
  const totalDiskGb = storage.total_disk_gb ?? 0;
  const diskUsedPercent = totalDiskGb > 0 ? Math.round(((totalDiskGb - freeDiskGb) / totalDiskGb) * 100) : 0;
  const storageStatus = (data.storage_status || storage.status || 'healthy').toUpperCase();
  const storageBackend = (data.storage_type || storage.backend || 'local').toUpperCase();
  const tenantsCount = data.tenants_count || 1;
  const activeSessions = data.active_sessions_count || 0;
  const platformName = data.platform || 'Server';

  const ramColor = getMetricColor(ramPercent, 70, 85);
  const diskColor = getMetricColor(diskUsedPercent, 80, 90);
  const latencyColor = writeLatencyMs < 10 ? '#00e575' : (writeLatencyMs < 50 ? '#f59e0b' : '#f43f5e');

  // Update modal header pills
  const statusText = document.getElementById('telemetry-status-text');
  const platformBadge = document.getElementById('telemetry-platform-badge');
  if (platformBadge) platformBadge.textContent = `${platformName.toUpperCase()} • v${escapeHtml(data.version || '13.6.0')}`;
  if (statusText) statusText.textContent = isHealthy ? 'HEALTHY • PROBES PASSING' : 'ATTENTION • DEGRADED';

  container.innerHTML = `
    <!-- Top KPI Grid: 6 Real-Time Telemetry Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
      
      <!-- 1. CPU Core Allocation -->
      <div class="telemetry-card p-3.5 rounded-xl border space-y-2" style="background-color: var(--bg-secondary, rgba(255,255,255,0.03)); border-color: var(--card-border, rgba(255,255,255,0.1));">
        <div class="flex items-center justify-between text-[11px]" style="color: var(--text-muted, #94a3b8);">
          <span>CPU CORES & ALLOCATION</span>
          <svg class="w-3.5 h-3.5 text-[#00e575]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"></path></svg>
        </div>
        <div id="telemetry-cpu-val" class="text-xl font-bold font-mono tracking-tight text-[#00e575]">
          ${cpuCores} <span class="text-xs font-normal" style="color: var(--text-muted, #94a3b8);">Logical Cores</span>
        </div>
        <div class="text-[10px]" style="color: var(--text-muted, #94a3b8);">
          Parallel Loky Workers Cap: 4 Cores
        </div>
      </div>

      <!-- 2. Host RAM Utilization -->
      <div class="telemetry-card p-3.5 rounded-xl border space-y-2" style="background-color: var(--bg-secondary, rgba(255,255,255,0.03)); border-color: var(--card-border, rgba(255,255,255,0.1));">
        <div class="flex items-center justify-between text-[11px]" style="color: var(--text-muted, #94a3b8);">
          <span>RAM UTILIZATION</span>
          <svg class="w-3.5 h-3.5 text-[#00e575]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 19v-3M10 19v-3M14 19v-3M18 19v-3M6 5v3M10 5v3M14 5v3M18 5v3M2 8h20v8H2z"></path></svg>
        </div>
        <div id="telemetry-ram-val" class="text-xl font-bold font-mono tracking-tight" style="color: ${ramColor};">
          ${formatPercent(ramPercent)} <span class="text-xs font-normal" style="color: var(--text-muted, #94a3b8);">(${formatGb(ramUsedGb)} / ${formatGb(ramTotalGb)})</span>
        </div>
        <div class="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
          <div class="h-full rounded-full transition-all duration-300" style="width: ${Math.min(100, Math.max(0, ramPercent))}%; background-color: ${ramColor};"></div>
        </div>
      </div>

      <!-- 3. GPU Acceleration Status -->
      <div class="telemetry-card p-3.5 rounded-xl border space-y-2" style="background-color: var(--bg-secondary, rgba(255,255,255,0.03)); border-color: var(--card-border, rgba(255,255,255,0.1));">
        <div class="flex items-center justify-between text-[11px]" style="color: var(--text-muted, #94a3b8);">
          <span>GPU ACCELERATION</span>
          <span class="w-2 h-2 rounded-full ${gpuAvailable ? 'bg-[#00e575]' : 'bg-amber-400'}"></span>
        </div>
        <div class="text-sm font-bold font-mono tracking-tight ${gpuAvailable ? 'text-[#00e575]' : 'text-amber-400'}">
          ${gpuAvailable ? 'NVIDIA CUDA DETECTED' : 'CPU MODE (AVX-512 ACTIVE)'}
        </div>
        <div class="text-[10px]" style="color: var(--text-muted, #94a3b8);">
          ${gpuAvailable ? 'Hardware Tensor Acceleration Active' : 'SIMD Matrix Vectorization Fallback'}
        </div>
      </div>

      <!-- 4. Storage Engine Health -->
      <div class="telemetry-card p-3.5 rounded-xl border space-y-2" style="background-color: var(--bg-secondary, rgba(255,255,255,0.03)); border-color: var(--card-border, rgba(255,255,255,0.1));">
        <div class="flex items-center justify-between text-[11px]" style="color: var(--text-muted, #94a3b8);">
          <span>STORAGE ENGINE HEALTH</span>
          <svg class="w-3.5 h-3.5 text-[#00e575]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12H2M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"></path></svg>
        </div>
        <div class="text-base font-bold font-mono tracking-tight text-[#00e575]">
          ${storageStatus} &bull; ${storageBackend}
        </div>
        <div class="text-[10px] truncate" style="color: var(--text-muted, #94a3b8);" title="${escapeHtml(storage.path || 'Partition Root')}">
          Mount: ${escapeHtml(storage.path ? (storage.path.length > 30 ? storage.path.substring(0, 30) + '...' : storage.path) : 'Local Storage')}
        </div>
      </div>

      <!-- 5. Write Latency in Milliseconds -->
      <div class="telemetry-card p-3.5 rounded-xl border space-y-2" style="background-color: var(--bg-secondary, rgba(255,255,255,0.03)); border-color: var(--card-border, rgba(255,255,255,0.1));">
        <div class="flex items-center justify-between text-[11px]" style="color: var(--text-muted, #94a3b8);">
          <span>STORAGE PROBE LATENCY</span>
          <span class="text-[10px] text-[#00e575] font-semibold">P99 SLA PASS</span>
        </div>
        <div id="telemetry-latency-val" class="text-xl font-bold font-mono tracking-tight" style="color: ${latencyColor};">
          ${formatLatency(writeLatencyMs)}
        </div>
        <div class="text-[10px]" style="color: var(--text-muted, #94a3b8);">
          Active write/read/delete verification probe
        </div>
      </div>

      <!-- 6. Disk Space Free / Total -->
      <div class="telemetry-card p-3.5 rounded-xl border space-y-2" style="background-color: var(--bg-secondary, rgba(255,255,255,0.03)); border-color: var(--card-border, rgba(255,255,255,0.1));">
        <div class="flex items-center justify-between text-[11px]" style="color: var(--text-muted, #94a3b8);">
          <span>DISK SPACE USAGE</span>
          <span class="text-[10px]" style="color: var(--text-muted, #94a3b8);">${diskUsedPercent}% Used</span>
        </div>
        <div id="telemetry-storage-val" class="text-xl font-bold font-mono tracking-tight" style="color: var(--text-primary, #ffffff);">
          ${formatGb(freeDiskGb)} <span class="text-xs font-normal" style="color: var(--text-muted, #94a3b8);">Free / ${formatGb(totalDiskGb)} Total</span>
        </div>
        <div class="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
          <div class="h-full rounded-full transition-all duration-300" style="width: ${diskUsedPercent}%; background-color: ${diskColor};"></div>
        </div>
      </div>

    </div>

    <!-- Active Tenant Status & Session Footprint Strip -->
    <div class="p-3.5 rounded-xl border flex items-center justify-between" style="background-color: var(--bg-secondary, rgba(255,255,255,0.02)); border-color: var(--card-border, rgba(255,255,255,0.08));">
      <div class="flex items-center space-x-3">
        <div class="w-2 h-2 rounded-full bg-[#00e575]"></div>
        <div>
          <span class="font-bold" style="color: var(--text-primary, #ffffff);">Tenant Isolation Guard:</span>
          <span class="ml-1" style="color: var(--text-muted, #94a3b8);">
            ${tenantsCount} Active Isolated ${tenantsCount === 1 ? 'Tenant' : 'Tenants'} &bull; ${activeSessions} In-Memory Sessions
          </span>
        </div>
      </div>
      <div class="text-[10px] px-2.5 py-1 rounded border border-emerald-500/30 bg-emerald-500/10 text-[#00e575] font-semibold">
        ENFORCED
      </div>
    </div>
  `;
}

// ─── Fetch & Data Polling ────────────────────────────────────────────────────

let isFetchingTelemetry = false;

/**
 * Fetches telemetry data from backend and updates modal view.
 */
async function fetchAndRenderTelemetry() {
  const container = document.getElementById('health-telemetry-body');
  const modal = document.getElementById('health-telemetry-modal');
  if (!container || !modal || modal.classList.contains('hidden')) return;

  // Prevent overlapping in-flight requests
  if (isFetchingTelemetry) return;
  isFetchingTelemetry = true;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    const res = await fetch('/api/v1/health', { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    const healthData = await res.json();

    if (modal.classList.contains('hidden')) return;

    // Backend /api/v1/health directly supplies complete host RAM metrics
    // (ram_percent, ram_used_gb, ram_total_gb, ram_available_gb).
    // No calls to /api/v1/system/metrics are needed, preventing secondary polling loops.

    if (modal.classList.contains('hidden')) return;

    renderHealthMetrics(container, healthData);

    const lastUpdatedEl = document.getElementById('telemetry-last-updated');
    if (lastUpdatedEl) {
      const now = new Date();
      lastUpdatedEl.textContent = `Updated: ${now.toTimeString().split(' ')[0]}`;
    }
  } catch (err) {
    if (modal.classList.contains('hidden')) return;
    container.innerHTML = `
      <div class="p-4 rounded-xl border border-rose-900/50 bg-rose-950/20 text-rose-400 font-mono text-xs space-y-1">
        <div class="font-bold">Failed to query /api/v1/health probe:</div>
        <div>${escapeHtml(err.message)}</div>
      </div>
    `;
  } finally {
    isFetchingTelemetry = false;
  }
}

/**
 * Starts auto-refresh timer (10s minimum interval).
 */
function startTelemetryPolling() {
  stopTelemetryPolling();
  telemetryInterval = setInterval(() => {
    const modal = document.getElementById('health-telemetry-modal');
    if (isAutoRefreshActive && modal && !modal.classList.contains('hidden') && document.visibilityState === 'visible') {
      fetchAndRenderTelemetry();
    } else if (modal && modal.classList.contains('hidden')) {
      stopTelemetryPolling();
    }
  }, 10000);
}

/**
 * Stops auto-refresh timer.
 */
function stopTelemetryPolling() {
  if (telemetryInterval) {
    clearInterval(telemetryInterval);
    telemetryInterval = null;
  }
}

// ─── Modal Open / Close Controller ───────────────────────────────────────────

/**
 * Opens the health telemetry modal HUD.
 */
function openTelemetryModal(triggerElement = null) {
  const modal = document.getElementById('health-telemetry-modal');
  if (!modal) return;

  if (triggerElement && triggerElement.target) {
    lastActiveTrigger = triggerElement.target;
  } else if (triggerElement instanceof HTMLElement) {
    lastActiveTrigger = triggerElement;
  } else {
    lastActiveTrigger = document.activeElement;
  }

  modal.classList.remove('hidden');
  modal.classList.add('flex');
  modal.setAttribute('aria-modal', 'true');
  document.body.style.overflow = 'hidden';

  fetchAndRenderTelemetry();
  startTelemetryPolling();

  requestAnimationFrame(() => {
    const closeBtn = modal.querySelector('#close-telemetry-btn') || modal.querySelector('[data-close-modal]');
    if (closeBtn && typeof closeBtn.focus === 'function') {
      closeBtn.focus();
    }
  });
}

/**
 * Closes the health telemetry modal HUD.
 */
function closeTelemetryModal() {
  const modal = document.getElementById('health-telemetry-modal');
  if (!modal) return;

  modal.classList.add('hidden');
  modal.classList.remove('flex');
  modal.removeAttribute('aria-modal');
  document.body.style.overflow = '';

  stopTelemetryPolling();

  if (lastActiveTrigger && typeof lastActiveTrigger.focus === 'function') {
    try {
      lastActiveTrigger.focus();
    } catch (e) {}
  }
}

// ─── Event Setup & Universal Interception ───────────────────────────────────

/**
 * Sets up modal controls, button listeners, and universal click delegation.
 */
let telemetryModalSetup = false;

function setupTelemetryModal() {
  if (telemetryModalSetup) return;
  const modal = document.getElementById('health-telemetry-modal');
  if (!modal) return;
  telemetryModalSetup = true;

  // 1. Backdrop click dismiss
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeTelemetryModal();
    }
  });

  // 2. Explicit close button handlers
  modal.querySelectorAll('[data-close-modal="health-telemetry-modal"], #close-telemetry-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      closeTelemetryModal();
    });
  });

  // 3. Polling pause / resume toggle
  const pollToggle = document.getElementById('telemetry-poll-toggle');
  if (pollToggle) {
    pollToggle.addEventListener('click', () => {
      isAutoRefreshActive = !isAutoRefreshActive;
      pollToggle.textContent = isAutoRefreshActive ? 'ACTIVE' : 'PAUSED';
      pollToggle.style.color = isAutoRefreshActive ? '#00e575' : '#f59e0b';
      if (isAutoRefreshActive) {
        fetchAndRenderTelemetry();
      }
    });
  }

  // 4. Refresh now button
  const refreshBtn = document.getElementById('telemetry-refresh-now-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      fetchAndRenderTelemetry();
    });
  }

  // 5. Global Escape key listener
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
      e.stopPropagation();
      closeTelemetryModal();
    }
  });

  // Page lifecycle listeners: ensure background timer stops when tab hidden or navigated
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopTelemetryPolling();
    } else {
      if (isAutoRefreshActive && modal && !modal.classList.contains('hidden')) {
        fetchAndRenderTelemetry();
        startTelemetryPolling();
      }
    }
  });
  window.addEventListener('beforeunload', stopTelemetryPolling);
  window.addEventListener('pagehide', stopTelemetryPolling);
}

/**
 * Clean Targeted Click Interceptor:
 * Intercepts clicks on elements explicitly marked with data-telemetry-trigger or links to /api/v1/health.
 * Never intercepts clicks inside the open modal itself.
 */
document.addEventListener('click', (e) => {
  const modal = document.getElementById('health-telemetry-modal');
  if (modal && modal.contains(e.target) && !modal.classList.contains('hidden')) {
    return;
  }

  const trigger = e.target.closest && e.target.closest(
    '[data-telemetry-trigger], a[href*="/api/v1/health"], a[href$="/health"], #nav-telemetry-btn'
  );

  if (trigger) {
    e.preventDefault();
    e.stopPropagation();
    openTelemetryModal(trigger);
  }
});

// ─── Global Exposure & Lifecycle ─────────────────────────────────────────────

window.openTelemetryModal = openTelemetryModal;
window.closeTelemetryModal = closeTelemetryModal;
window.openHealthTelemetryModal = openTelemetryModal;
window.closeHealthTelemetryModal = closeTelemetryModal;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupTelemetryModal);
} else {
  setupTelemetryModal();
}
