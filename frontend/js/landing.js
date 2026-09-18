/**
 * frontend/js/landing.js
 * ──────────────────────
 * Dedicated controller for the standalone DIA Landing Page.
 * Implements interactive Causal DAG node inspection, benchmark demo modal & 1-click loading,
 * drag-and-drop universal file upload modal with progress, and navigation to /app.
 */

import { setupCustomCursor, setupSpotlightCards } from './cursor.js';
import { initTheme, toggleTheme, applyTheme, getCurrentTheme } from './theme.js';

// Causal Node Metadata for Interactive DAG Inspector
const CAUSAL_NODES_DATA = {
  z: {
    title: 'Confounder (Z)',
    label: 'Credit Score & Age',
    type: 'Backdoor Conditioning Set',
    status: 'Conditioning Satisfied',
    desc: 'Common cause influencing treatment and default risk. Conditioned to block spurious non-causal confounding paths.',
    color: '#818cf8',
    stats: 'Backdoor Path: Blocked • Bias Controlled'
  },
  x: {
    title: 'Treatment do(X)',
    label: 'Interest Rate %',
    type: 'Policy Action Variable',
    status: 'Pearl Rule 2 Identified',
    desc: 'Policy intervention variable. Applying the Do-Operator do(X = x) simulates what would happen if the policy maker forcibly sets the rate, severing natural parental dependencies from Z.',
    color: '#00e575',
    stats: 'Simulated ATE: -12.8% • 95% CI: [-15.2%, -10.4%]'
  },
  m: {
    title: 'Mediator (M)',
    label: 'Debt-to-Income',
    type: 'Intermediate Causal Channel',
    status: 'Direct & Indirect Separation',
    desc: 'Intermediate causal channel transmitting policy effects from Interest Rate to Default Risk. Baron-Kenny decomposition isolates the direct policy effect from mediator-transmitted risk amplification.',
    color: '#38bdf8',
    stats: 'Mediation Ratio: 34.2% • Sobel p < 0.001'
  },
  y: {
    title: 'Outcome (Y)',
    label: 'Default Risk',
    type: 'Target Response Variable',
    status: 'Unbiased Invariant Estimator',
    desc: 'Empirical business outcome modeled under counterfactual policy interventions. Evaluated with 95% conformal prediction intervals guaranteeing finite-sample coverage.',
    color: '#f43f5e',
    stats: 'Baseline Default: 25.6% • Intervention: 12.8%'
  }
};

let lastActiveTrigger = null;

export function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;
  lastActiveTrigger = document.activeElement;
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('role', 'dialog');
  document.body.style.overflow = 'hidden';

  // WCAG 2.2 AA: Move focus to first focusable element inside modal
  requestAnimationFrame(() => {
    const focusable = modal.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])');
    if (focusable.length > 0) {
      focusable[0].focus();
    }
  });
}

export function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  modal.removeAttribute('aria-modal');
  document.body.style.overflow = '';

  // WCAG 2.2 AA: Restore focus to trigger element
  if (lastActiveTrigger && typeof lastActiveTrigger.focus === 'function') {
    try {
      lastActiveTrigger.focus();
    } catch (e) {}
  }
}

// ─── Interactive DAG Node Inspector ──────────────────────────────────────────

export function selectDagNode(nodeKey) {
  const nodeInfo = CAUSAL_NODES_DATA[nodeKey];
  if (!nodeInfo) return;

  // Highlight active node in SVG
  document.querySelectorAll('.dag-svg-node').forEach((el) => {
    el.classList.remove('ring-active');
    el.classList.remove('active');
  });
  const activeEl = document.getElementById(`dag-node-${nodeKey}`);
  if (activeEl) {
    activeEl.classList.add('ring-active');
    activeEl.classList.add('active');
  }

  // Update details card / banner
  const titleEl = document.getElementById('dag-inspector-title');
  const typeEl = document.getElementById('dag-inspector-type');
  const descEl = document.getElementById('dag-inspector-desc');
  const statsEl = document.getElementById('dag-inspector-stats');

  if (titleEl) {
    titleEl.textContent = `${nodeInfo.title.replace(')', ` = ${nodeInfo.label})`)}:`;
    titleEl.style.color = nodeInfo.color;
  }
  if (typeEl) {
    typeEl.textContent = nodeInfo.type;
    typeEl.style.color = nodeInfo.color;
  }
  if (descEl) descEl.textContent = nodeInfo.desc;
  if (statsEl) statsEl.textContent = nodeInfo.stats;
}

// ─── Live Policy Intervention Simulator ───────────────────────────────────────

export function updatePolicyIntervention(val) {
  const rate = parseFloat(val);
  if (isNaN(rate)) return;

  const sliderValBadge = document.getElementById('policy-slider-val');
  if (sliderValBadge) {
    sliderValBadge.textContent = `do(Interest Rate = ${rate.toFixed(1)}%)`;
  }

  // Baseline: Rate = 8.5%, Default Risk = 25.6%
  // Sensitivity coefficient = 3.2
  const baselineRate = 8.5;
  const baselineRisk = 25.6;
  const deltaRate = rate - baselineRate;
  const ate = deltaRate * 3.2;
  const simRisk = Math.max(0.1, Math.min(99.9, baselineRisk + ate));
  const ciMargin = 2.4;
  const ciLow = ate - ciMargin;
  const ciHigh = ate + ciMargin;

  const simDefaultEl = document.getElementById('metric-sim-default');
  const simDeltaEl = document.getElementById('metric-sim-delta');
  const simAteEl = document.getElementById('metric-sim-ate');
  const simCiEl = document.getElementById('metric-sim-ci');

  if (simDefaultEl) {
    simDefaultEl.textContent = `${simRisk.toFixed(1)}%`;
  }
  if (simDeltaEl) {
    const sign = ate >= 0 ? '+' : '';
    simDeltaEl.textContent = `Δ ${sign}${ate.toFixed(1)}% vs Baseline`;
    if (ate <= 0) {
      simDeltaEl.className = 'text-[11px] font-mono text-[#00e575]/90';
    } else {
      simDeltaEl.className = 'text-[11px] font-mono text-amber-400';
    }
  }
  if (simAteEl) {
    const sign = ate >= 0 ? '+' : '';
    simAteEl.textContent = `${sign}${ate.toFixed(1)}%`;
  }
  if (simCiEl) {
    const signLow = ciLow >= 0 ? '+' : '';
    const signHigh = ciHigh >= 0 ? '+' : '';
    simCiEl.textContent = `95% CI: [${signLow}${ciLow.toFixed(1)}%, ${signHigh}${ciHigh.toFixed(1)}%]`;
  }
}

// ─── Universal Ingest (Upload) ───────────────────────────────────────────────

let selectedFile = null;
const ALLOWED_EXTENSIONS = ['.csv', '.tsv', '.parquet', '.json', '.xlsx', '.xls'];

async function parseApiError(res, fallbackMessage) {
  try {
    const data = await res.json();
    if (data && data.detail) {
      if (Array.isArray(data.detail)) {
        return data.detail.map((d) => d.msg || (typeof d === 'string' ? d : JSON.stringify(d))).join(', ');
      }
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    }
    return fallbackMessage;
  } catch {
    return `${fallbackMessage} (HTTP ${res.status})`;
  }
}

export function handleFileSelect(file) {
  if (!file) return;

  const uploadErrorEl = document.getElementById('upload-error-msg');
  if (uploadErrorEl) uploadErrorEl.classList.add('hidden');

  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    if (uploadErrorEl) {
      uploadErrorEl.textContent = `Unsupported file format '${ext}'. Please upload CSV, TSV, Parquet, JSON, or Excel.`;
      uploadErrorEl.classList.remove('hidden');
    }
    return;
  }

  selectedFile = file;

  const fileInfoEl = document.getElementById('upload-file-info');
  const fileNameEl = document.getElementById('upload-file-name');
  const fileSizeEl = document.getElementById('upload-file-size');
  const uploadSubmitBtn = document.getElementById('upload-submit-btn');

  if (fileInfoEl && fileNameEl && fileSizeEl) {
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = formatBytes(file.size);
    fileInfoEl.classList.remove('hidden');
  }
  if (uploadSubmitBtn) {
    uploadSubmitBtn.disabled = false;
    uploadSubmitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
  }
}

export function removeSelectedFile() {
  selectedFile = null;
  const fileInput = document.getElementById('landing-file-input');
  if (fileInput) fileInput.value = '';

  const fileInfoEl = document.getElementById('upload-file-info');
  const uploadSubmitBtn = document.getElementById('upload-submit-btn');
  const uploadErrorEl = document.getElementById('upload-error-msg');

  if (fileInfoEl) fileInfoEl.classList.add('hidden');
  if (uploadErrorEl) uploadErrorEl.classList.add('hidden');
  if (uploadSubmitBtn) {
    uploadSubmitBtn.disabled = true;
    uploadSubmitBtn.classList.add('opacity-50', 'cursor-not-allowed');
  }
}

export async function submitDatasetUpload() {
  if (!selectedFile) return;

  const progressSection = document.getElementById('upload-progress-section');
  const progressBar = document.getElementById('upload-progress-bar');
  const progressText = document.getElementById('upload-progress-text');
  const progressPct = document.getElementById('upload-progress-pct');
  const uploadSubmitBtn = document.getElementById('upload-submit-btn');
  const uploadErrorEl = document.getElementById('upload-error-msg');

  if (uploadErrorEl) uploadErrorEl.classList.add('hidden');
  if (progressSection) progressSection.classList.remove('hidden');
  if (uploadSubmitBtn) {
    uploadSubmitBtn.disabled = true;
    uploadSubmitBtn.textContent = 'Uploading & Sanitizing...';
  }

  const setProgress = (pct, text) => {
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressBar && progressBar.parentElement) progressBar.parentElement.setAttribute('aria-valuenow', pct);
    if (progressPct) progressPct.textContent = `${pct}%`;
    if (progressText) progressText.textContent = text;
  };

  // Step 1: Initial parsing
  setProgress(25, 'Reading bytes and verifying schema...');

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Step 2: Sanitization & profiling
    setProgress(60, 'Executing automated sanitization & profiling...');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    const response = await fetch('/api/v1/ingest/upload', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errDetail = await parseApiError(response, 'Dataset upload failed');
      throw new Error(errDetail);
    }

    const data = await response.json();

    // Step 3: Contract synthesis complete
    setProgress(100, 'Data contract synthesized! Launching workbench...');

    // Store in sessionStorage with fallback for strict private browsing
    let storedInSession = false;
    try {
      sessionStorage.setItem('dia_pending_session', JSON.stringify(data));
      sessionStorage.removeItem('dia_pending_autotrain');
      storedInSession = true;
    } catch (storageErr) {
      console.warn('sessionStorage quota exceeded or disabled, using URL parameter fallback:', storageErr);
      try {
        sessionStorage.setItem('dia_pending_session', JSON.stringify({
          session_id: data.session_id,
          detected_domain: data.detected_domain,
          n_rows: data.n_rows,
          n_cols: data.n_cols
        }));
        storedInSession = true;
      } catch (e) {}
    }

    // Redirect to full analytical workbench
    setTimeout(() => {
      if (storedInSession) {
        window.location.href = '/app';
      } else {
        window.location.href = `/app?session_id=${encodeURIComponent(data.session_id)}`;
      }
    }, 500);
  } catch (err) {
    if (progressSection) progressSection.classList.add('hidden');
    if (uploadSubmitBtn) {
      uploadSubmitBtn.disabled = false;
      uploadSubmitBtn.textContent = 'Upload & Launch Workbench';
    }
    if (uploadErrorEl) {
      uploadErrorEl.textContent = `Upload Error: ${err.message}`;
      uploadErrorEl.classList.remove('hidden');
    }
  }
}

// ─── 1-Click Interactive Benchmark Launcher ──────────────────────────────────

export async function launchDemoBenchmark(demoName) {
  const loadingIndicator = document.getElementById(`demo-loading-${encodeURIComponent(demoName)}`);
  const allBtns = document.querySelectorAll('.demo-launch-btn');

  allBtns.forEach((b) => {
    b.disabled = true;
    b.classList.add('opacity-50');
  });

  if (loadingIndicator) {
    loadingIndicator.classList.remove('hidden');
    const label = loadingIndicator.querySelector('.loading-label');
    if (label) {
      label.textContent = 'Initializing benchmark & training baselines...';
    } else {
      loadingIndicator.textContent = 'Initializing benchmark & training baselines...';
    }
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    const res = await fetch('/api/v1/ingest/demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ demo_name: demoName }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const errDetail = await parseApiError(res, 'Failed to initialize benchmark scenario');
      throw new Error(errDetail);
    }

    const data = await res.json();

    // Store in sessionStorage safely
    try {
      sessionStorage.setItem('dia_pending_session', JSON.stringify(data));
      sessionStorage.setItem('dia_pending_autotrain', 'false');
    } catch (e) {
      console.warn('sessionStorage not available, relying on URL parameter:', e);
    }

    // Redirect to full analytical workbench
    window.location.href = '/app';
  } catch (err) {
    allBtns.forEach((b) => {
      b.disabled = false;
      b.classList.remove('opacity-50');
    });
    if (loadingIndicator) {
      const label = loadingIndicator.querySelector('.loading-label');
      if (label) {
        label.textContent = `Error: ${err.message}`;
        label.classList.add('text-rose-400');
      } else {
        loadingIndicator.textContent = `Error: ${err.message}`;
        loadingIndicator.classList.add('text-rose-400');
      }
    }
  }
}

// ─── Utilities ───────────────────────────────────────────────────────────────

function formatBytes(bytes, decimals = 1) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}


// ─── Setup Event Listeners on DOMContentLoaded ───────────────────────────────

function setupLandingEventListeners() {
  // Initialize Two-Theme Engine (Night / Day Mode)
  try {
    initTheme();
  } catch (err) {
    console.warn('Theme initialization notice:', err);
  }

  // Initialize Custom Precision Cursor & Interactive Spotlight luminescence
  try {
    setupCustomCursor();
  } catch (err) {
    console.warn('Custom cursor initialization notice:', err);
  }
  setupSpotlightCards();

  // Modal Triggers
  const btnLaunchDemo = document.getElementById('btn-launch-demo');
  const btnIngestData = document.getElementById('btn-ingest-data');
  const demoModal = document.getElementById('demo-modal');
  const uploadModal = document.getElementById('upload-modal');

  if (btnLaunchDemo) {
    btnLaunchDemo.addEventListener('click', () => openModal('demo-modal'));
  }
  if (btnIngestData) {
    btnIngestData.addEventListener('click', () => openModal('upload-modal'));
  }

  // Close buttons
  document.querySelectorAll('[data-close-modal]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-close-modal');
      closeModal(targetId);
    });
  });

  // Backdrop click
  [demoModal, uploadModal].forEach((m) => {
    if (m) {
      m.addEventListener('click', (e) => {
        if (e.target === m) closeModal(m.id);
      });
    }
  });

  // ESC and WCAG 2.2 AA Modal Focus Trap keydown listener
  window.addEventListener('keydown', (e) => {
    const activeModal = [demoModal, uploadModal].find((m) => m && !m.classList.contains('hidden'));

    if (e.key === 'Escape') {
      if (typeof window.closeTelemetryModal === 'function') {
        window.closeTelemetryModal();
      }
      if (activeModal) {
        closeModal(activeModal.id);
      }
      return;
    }

    // Modal Focus Trapping on Tab / Shift+Tab
    if (e.key === 'Tab' && activeModal) {
      const focusable = Array.from(
        activeModal.querySelectorAll(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
      ).filter((el) => el.offsetParent !== null && !el.classList.contains('hidden'));

      if (focusable.length === 0) return;

      const firstEl = focusable[0];
      const lastEl = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === firstEl || !activeModal.contains(document.activeElement)) {
          e.preventDefault();
          lastEl.focus();
        }
      } else {
        if (document.activeElement === lastEl || !activeModal.contains(document.activeElement)) {
          e.preventDefault();
          firstEl.focus();
        }
      }
    }
  });

  // Prevent accidental file drop outside dropzone from opening in browser
  window.addEventListener('dragover', (e) => e.preventDefault(), false);
  window.addEventListener('drop', (e) => e.preventDefault(), false);

  // Dropzone Setup
  const dropzone = document.getElementById('landing-dropzone');
  const fileInput = document.getElementById('landing-file-input');

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('border-[#00e575]', 'bg-emerald-500/10');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('border-[#00e575]', 'bg-emerald-500/10');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });
  }

  // Upload actions
  const uploadSubmitBtn = document.getElementById('upload-submit-btn');
  const removeFileBtn = document.getElementById('remove-selected-file-btn');

  if (uploadSubmitBtn) {
    uploadSubmitBtn.addEventListener('click', submitDatasetUpload);
  }
  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', removeSelectedFile);
  }

  // Mobile Menu Toggle & Auto-Dismiss on selection
  const mobileMenuBtn = document.getElementById('mobile-menu-btn');
  const mobileMenu = document.getElementById('mobile-menu');
  if (mobileMenuBtn && mobileMenu) {
    mobileMenuBtn.addEventListener('click', () => {
      mobileMenu.classList.toggle('hidden');
    });

    mobileMenu.querySelectorAll('a, button').forEach((item) => {
      item.addEventListener('click', () => {
        mobileMenu.classList.add('hidden');
      });
    });
  }

  // Policy Intervention Range Slider
  const slider = document.getElementById('causal-policy-slider');
  if (slider) {
    slider.addEventListener('input', (e) => {
      updatePolicyIntervention(e.target.value);
    });
  }

  // Default Select Treatment Node X and initialize policy intervention at 4.5% matching mockup
  selectDagNode('x');
  updatePolicyIntervention(4.5);
}

// Expose globals for onclick attributes
window.openModal = openModal;
window.closeModal = closeModal;
window.selectDagNode = selectDagNode;
window.updatePolicyIntervention = updatePolicyIntervention;
window.launchDemoBenchmark = launchDemoBenchmark;
window.handleFileSelect = handleFileSelect;
window.removeSelectedFile = removeSelectedFile;
window.submitDatasetUpload = submitDatasetUpload;
window.setupSpotlightCards = setupSpotlightCards;
window.setupCustomCursor = setupCustomCursor;
window.initTheme = initTheme;
window.toggleTheme = toggleTheme;
window.applyTheme = applyTheme;
window.getCurrentTheme = getCurrentTheme;

if (document.readyState === 'loading') {
  window.addEventListener('DOMContentLoaded', setupLandingEventListeners);
} else {
  setupLandingEventListeners();
}

