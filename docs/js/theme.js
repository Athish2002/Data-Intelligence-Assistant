/**
 * frontend/js/theme.js
 * ─────────────────────
 * Centralized Two-Theme Engine for Data Intelligence Assistant.
 * Strictly supports 2 polished themes:
 * 1. Night Mode ('night'): Deep Obsidian (#040608) canvas with Cyber Aurora mesh glow.
 * 2. Day Mode ('day'): Warm editorial Tokyo Sand (#f5f2eb) canvas with high-contrast ink text (WCAG 2.2 AA >= 4.5:1).
 *
 * Persisted in localStorage under key 'dia_theme_mode'.
 */

export const THEME_KEY = 'dia_theme_mode';
export const LEGACY_THEME_KEY = 'dia-theme';

export const THEMES = {
  NIGHT: 'night',
  DAY: 'day',
};

let currentTheme = THEMES.NIGHT;
let isTransitioning = false;
let transitionTimeout = null;
const listeners = new Set();

/**
 * Retrieves the stored theme mode, handling legacy key migration.
 */
export function getInitialTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === THEMES.DAY || stored === THEMES.NIGHT) return stored;

    // Migrate from legacy keys
    const legacy = localStorage.getItem(LEGACY_THEME_KEY);
    if (legacy === 'tokyo-sand') return THEMES.DAY;
    if (legacy === 'obsidian' || legacy === 'cyber-aurora') return THEMES.NIGHT;
  } catch (e) {
    console.warn('[Theme] localStorage unavailable, defaulting to night mode:', e);
  }
  return THEMES.NIGHT;
}

/**
 * Initializes the theme engine on page load.
 */
export function initTheme() {
  const theme = getInitialTheme();
  applyTheme(theme, false);
  setupThemeToggleButtons();

  // Listen for storage events across tabs
  window.addEventListener('storage', (e) => {
    if (e.key === THEME_KEY && (e.newValue === THEMES.NIGHT || e.newValue === THEMES.DAY)) {
      applyTheme(e.newValue, false);
    }
  });
}

/**
 * Toggles between Night Mode and Day Mode.
 */
export function toggleTheme() {
  const next = currentTheme === THEMES.NIGHT ? THEMES.DAY : THEMES.NIGHT;
  applyTheme(next, true);
  return next;
}

/**
 * Applies the selected theme to the document root and persists it.
 */
export function applyTheme(theme, animate = true) {
  currentTheme = (theme === 'tokyo-sand' || theme === THEMES.DAY) ? THEMES.DAY : THEMES.NIGHT;
  const root = document.documentElement;

  if (animate) {
    if (isTransitioning) clearTimeout(transitionTimeout);
    isTransitioning = true;
    root.classList.add('theme-in-transition');
    transitionTimeout = setTimeout(() => {
      isTransitioning = false;
      root.classList.remove('theme-in-transition');
    }, 280);
  }

  // Set the data-theme attribute used by style.css
  const dataThemeValue = currentTheme === THEMES.DAY ? 'tokyo-sand' : 'cyber-aurora';
  root.setAttribute('data-theme', dataThemeValue);
  root.setAttribute('data-theme-mode', currentTheme);

  if (currentTheme === THEMES.DAY) {
    root.classList.remove('dark');
    root.classList.add('light');
  } else {
    root.classList.remove('light');
    root.classList.add('dark');
  }

  // Persist to localStorage
  try {
    localStorage.setItem(THEME_KEY, currentTheme);
    localStorage.setItem(LEGACY_THEME_KEY, dataThemeValue);
  } catch (e) {
    console.warn('[Theme] Failed to persist theme preference:', e);
  }

  updateToggleButtonsUI();

  // Notify registered callbacks
  listeners.forEach((fn) => {
    try {
      fn(currentTheme);
    } catch (err) {
      console.error('[Theme] Listener error:', err);
    }
  });

  // Dispatch custom event for Chart.js / Plotly reflow
  window.dispatchEvent(new CustomEvent('dia-theme-changed', {
    detail: { theme: currentTheme, dataTheme: dataThemeValue }
  }));
}

/**
 * Register a listener for theme changes.
 */
export function onThemeChange(callback) {
  listeners.add(callback);
  return () => listeners.delete(callback);
}

/**
 * Updates all theme toggle button icons and labels across the DOM.
 */
export function updateToggleButtonsUI() {
  const isNight = currentTheme === THEMES.NIGHT;
  const buttons = document.querySelectorAll(
    '[data-theme-toggle], [data-action="toggle-theme"], #theme-toggle-btn, #theme-toggle-btn-mobile, #landing-theme-toggle-btn, #mobile-theme-toggle-btn, #mode-toggle-btn'
  );

  buttons.forEach((btn) => {
    btn.setAttribute('aria-label', isNight ? 'Switch to Day Mode (Tokyo Sand)' : 'Switch to Night Mode (Cyber Aurora)');
    btn.setAttribute('title', isNight ? 'Switch to Day Mode (Tokyo Sand)' : 'Switch to Night Mode (Cyber Aurora)');
    btn.setAttribute('data-current-theme', currentTheme);

    const moonIcon = btn.querySelector('.theme-icon-moon');
    const sunIcon = btn.querySelector('.theme-icon-sun');
    if (moonIcon && sunIcon) {
      if (isNight) {
        moonIcon.classList.remove('hidden');
        sunIcon.classList.add('hidden');
      } else {
        moonIcon.classList.add('hidden');
        sunIcon.classList.remove('hidden');
      }
    } else {
      btn.innerHTML = isNight
        ? '<span class="relative w-4 h-4 flex items-center justify-center pointer-events-none"><svg class="theme-icon-moon w-4 h-4 text-[#00e575] transition-all duration-300 transform group-hover:scale-110" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg></span>'
        : '<span class="relative w-4 h-4 flex items-center justify-center pointer-events-none"><svg class="theme-icon-sun w-4 h-4 text-amber-500 transition-all duration-300 transform group-hover:scale-110" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg></span>';
    }

    const label = btn.querySelector('.theme-label-text');
    if (label) {
      label.textContent = isNight ? 'Day Mode' : 'Night Mode';
    }
  });
}

/**
 * Attaches click handlers to all theme toggle buttons.
 */
export function setupThemeToggleButtons() {
  const buttons = document.querySelectorAll(
    '[data-theme-toggle], [data-action="toggle-theme"], #theme-toggle-btn, #theme-toggle-btn-mobile, #landing-theme-toggle-btn, #mobile-theme-toggle-btn, #mode-toggle-btn'
  );
  buttons.forEach((btn) => {
    if (btn._themeToggleAttached) return;
    btn._themeToggleAttached = true;
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      toggleTheme();
    });
  });
  updateToggleButtonsUI();
}

export function getCurrentTheme() {
 return currentTheme;
}

// Global browser attachments for cross-context / testing / inline accessibility
if (typeof window !== 'undefined') {
 window.initTheme = initTheme;
 window.toggleTheme = toggleTheme;
 window.applyTheme = applyTheme;
 window.getCurrentTheme = getCurrentTheme;
 window.onThemeChange = onThemeChange;
}
