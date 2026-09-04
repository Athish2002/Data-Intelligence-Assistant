/**
 * frontend/js/toast.js
 * ────────────────────
 * Enterprise Floating Notifications and Real-Time Progress HUD.
 */

import { el } from './state.js';

export function showLoading(msg) {
  if (el.statusText) el.statusText.textContent = msg;
  if (el.trainProgress) el.trainProgress.classList.remove('hidden');
}

export function hideLoading() {
  if (el.trainProgress) el.trainProgress.classList.add('hidden');
}

export function showTrainingProgress() {
  if (el.statusText) el.statusText.textContent = 'Executing multi-model AutoML training & SHAP precomputations...';
  if (el.trainProgress) el.trainProgress.classList.remove('hidden');
  if (el.progressBar) el.progressBar.style.width = '60%';
}

export function hideTrainingProgress() {
  if (el.trainProgress) el.trainProgress.classList.add('hidden');
  if (el.progressBar) el.progressBar.style.width = '0%';
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
