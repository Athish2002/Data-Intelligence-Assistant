/**
 * frontend/js/toast.js
 * ────────────────────
 * Enterprise Floating Notifications and Real-Time Progress HUD.
 */

import { el } from './state.js';

let progressTimerInterval = null;
let progressStageTimeout = null;
let progressStartTime = 0;

function updateNodeState(nodeNum, state) {
  const node = document.getElementById(`p-node-${nodeNum}`);
  if (!node) return;
  node.classList.remove('active', 'completed', 'pending');
  node.classList.add(state);
  if (state === 'completed') {
    node.innerHTML = `✓`;
  } else {
    node.textContent = `${nodeNum}`;
  }
}

export function showLoading(msg) {
  const trainProgress = document.getElementById('train-progress');
  const statusText = document.getElementById('status-text');
  const stageTitle = document.getElementById('progress-stage-title');
  const progressBar = document.getElementById('progress-bar');
  const percentText = document.getElementById('progress-percent');

  if (trainProgress) trainProgress.classList.remove('hidden');
  if (statusText) statusText.textContent = msg;
  if (stageTitle) stageTitle.innerHTML = `<span class="w-2 h-2 rounded-full bg-[#00e575] animate-pulse mr-1.5"></span> Ingestion`;
  if (progressBar) progressBar.style.width = '25%';
  if (percentText) percentText.textContent = '25%';
}

export function hideLoading() {
  const trainProgress = document.getElementById('train-progress');
  if (trainProgress) trainProgress.classList.add('hidden');
}

export function showTrainingProgress() {
  const trainProgress = document.getElementById('train-progress');
  const stageTitle = document.getElementById('progress-stage-title');
  const statusText = document.getElementById('status-text');
  const progressBar = document.getElementById('progress-bar');
  const percentText = document.getElementById('progress-percent');
  const timerText = document.getElementById('progress-timer');

  if (trainProgress) trainProgress.classList.remove('hidden');

  // Reset all 5 nodes
  updateNodeState(1, 'active');
  updateNodeState(2, 'pending');
  updateNodeState(3, 'pending');
  updateNodeState(4, 'pending');
  updateNodeState(5, 'pending');

  if (stageTitle) stageTitle.innerHTML = `<span class="w-2 h-2 rounded-full bg-[#00e575] animate-pulse mr-1.5"></span> Stage 1/5: Ingestion`;
  if (statusText) statusText.textContent = 'Sanitizing types and detecting delimiters...';
  if (progressBar) progressBar.style.width = '15%';
  if (percentText) percentText.textContent = '15%';

  // Start live elapsed timer
  progressStartTime = Date.now();
  if (progressTimerInterval) clearInterval(progressTimerInterval);
  progressTimerInterval = setInterval(() => {
    if (timerText) {
      const elapsedMs = Date.now() - progressStartTime;
      const secs = (elapsedMs / 1000).toFixed(1);
      timerText.textContent = `${secs}s`;
    }
  }, 100);

  // Auto-advance simulated micro-stages while backend trains
  const scheduleStage = (delay, stageNum, title, desc, pct) => {
    return setTimeout(() => {
      // Mark previous nodes completed
      for (let i = 1; i < stageNum; i++) updateNodeState(i, 'completed');
      updateNodeState(stageNum, 'active');

      if (stageTitle) stageTitle.innerHTML = `<span class="w-2 h-2 rounded-full bg-[#00e575] animate-pulse mr-1.5"></span> Stage ${stageNum}/5: ${title}`;
      if (statusText) statusText.textContent = desc;
      if (progressBar) progressBar.style.width = `${pct}%`;
      if (percentText) percentText.textContent = `${pct}%`;
    }, delay);
  };

  const t1 = scheduleStage(1200, 2, 'Feature Profiling', 'Extracting cardinality & distributions...', 35);
  const t2 = scheduleStage(2600, 3, 'AutoML Benchmark', 'Fitting Random Forest, LightGBM & Ensembles...', 65);
  const t3 = scheduleStage(5200, 4, 'Explainability', 'Calculating TreeSHAP & ROC curve coordinates...', 85);
  const t4 = scheduleStage(8000, 5, 'Governance', 'Compiling Great Expectations contract suite...', 95);

  progressStageTimeout = [t1, t2, t3, t4];
}

export function hideTrainingProgress(success = true) {
  if (progressTimerInterval) {
    clearInterval(progressTimerInterval);
    progressTimerInterval = null;
  }
  if (progressStageTimeout) {
    progressStageTimeout.forEach((t) => clearTimeout(t));
    progressStageTimeout = null;
  }

  const trainProgress = document.getElementById('train-progress');
  const stageTitle = document.getElementById('progress-stage-title');
  const statusText = document.getElementById('status-text');
  const progressBar = document.getElementById('progress-bar');
  const percentText = document.getElementById('progress-percent');

  if (success) {
    for (let i = 1; i <= 5; i++) updateNodeState(i, 'completed');
    if (stageTitle) stageTitle.innerHTML = `<span class="text-[#00e575]">✓ Pipeline Complete</span>`;
    if (statusText) statusText.textContent = 'All 5 stages finalized successfully.';
    if (progressBar) progressBar.style.width = '100%';
    if (percentText) percentText.textContent = '100%';

    setTimeout(() => {
      if (trainProgress) trainProgress.classList.add('hidden');
    }, 900);
  } else {
    if (trainProgress) trainProgress.classList.add('hidden');
  }
}


export function showNotification(msg, type = 'error') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'fixed top-14 right-4 z-50 flex flex-col space-y-2 max-w-md w-full pointer-events-none';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = 'p-3.5 rounded-xl border shadow-2xl transition-all duration-300 transform translate-y-2 opacity-0 pointer-events-auto flex items-start space-x-3 text-xs font-mono';
  
  if (type === 'error') {
    toast.style.backgroundColor = '#180a0a';
    toast.style.borderColor = '#7f1d1d';
    toast.style.color = '#fca5a5';
    toast.innerHTML = `
      <span class="text-base">⚠️</span>
      <div class="flex-1 leading-relaxed">
        <div class="font-bold uppercase tracking-wider text-red-400 mb-0.5">Pipeline Notice</div>
        <div class="text-xs text-red-200">${msg}</div>
      </div>
      <button class="text-red-400 hover:text-white cursor-pointer text-sm font-bold ml-2" onclick="this.parentElement.remove()">✕</button>
    `;
  } else {
    toast.style.backgroundColor = '#061c0e';
    toast.style.borderColor = '#14532d';
    toast.style.color = '#86efac';
    toast.innerHTML = `
      <span class="text-base">✅</span>
      <div class="flex-1 leading-relaxed">
        <div class="font-bold uppercase tracking-wider text-emerald-400 mb-0.5">Success</div>
        <div class="text-xs text-emerald-200">${msg}</div>
      </div>
      <button class="text-emerald-400 hover:text-white cursor-pointer text-sm font-bold ml-2" onclick="this.parentElement.remove()">✕</button>
    `;
  }

  container.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-x-4');
    setTimeout(() => toast.remove(), 300);
  }, 7000);
}

export function showSuccess(msg) {
  showNotification(msg, 'success');
}

export function showError(msg) {
  showNotification(msg, 'error');
}
