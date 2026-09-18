/**
 * frontend/js/cursor.js
 * ─────────────────────
 * Precision Interactive Custom Cursor with text caret suppression,
 * spring-physics dual-ring easing, dynamic scale transitions,
 * luminous ambient aura lighting, click wave ripples, and universal spotlight card tracking.
 */

let cursorInitialized = false;
let spotlightInitialized = false;
let activeSpotlightTarget = null;

export function setupCustomCursor() {
  if (cursorInitialized) return;

  const dot = document.getElementById('cursor-dot');
  const ring = document.getElementById('cursor-ring');
  const aura = document.getElementById('cursor-aura');

  if (!dot && !ring) return;
  cursorInitialized = true;

  // Track touch activity so touch-only interactions don't show custom cursor
  window.addEventListener('touchstart', () => {
    document.body.classList.add('touch-device-active');
  }, { passive: true });

  let mouseX = -999;
  let mouseY = -999;
  let ringX = -999;
  let ringY = -999;
  let auraX = -999;
  let auraY = -999;
  let currentScale = 1.0;
  let isHovering = false;
  let isMouseDown = false;
  let hasMoved = false;

  function updatePosition(x, y) {
    if (isNaN(x) || isNaN(y)) return;
    mouseX = x;
    mouseY = y;

    // Update global CSS custom properties for any CSS consumers
    document.documentElement.style.setProperty('--cx', `${mouseX}px`);
    document.documentElement.style.setProperty('--cy', `${mouseY}px`);
    document.documentElement.style.setProperty('--cursor-x', `${mouseX}px`);
    document.documentElement.style.setProperty('--cursor-y', `${mouseY}px`);

    if (!hasMoved) {
      hasMoved = true;
      ringX = mouseX;
      ringY = mouseY;
      auraX = mouseX;
      auraY = mouseY;

      if (dot) {
        dot.style.visibility = 'visible';
        dot.style.opacity = '1';
        dot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0)`;
      }
      if (ring) {
        ring.style.visibility = 'visible';
        ring.style.opacity = '1';
        ring.style.transform = `translate3d(${ringX}px, ${ringY}px, 0) scale(${currentScale.toFixed(4)})`;
      }
      if (aura) {
        aura.style.visibility = 'visible';
        aura.style.opacity = '1';
        aura.style.transform = `translate3d(${auraX}px, ${auraY}px, 0)`;
      }
    } else {
      if (dot) {
        dot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0)`;
      }
    }
  }

  window.addEventListener('mousemove', (e) => {
    document.body.classList.remove('touch-device-active');
    if (!document.body.classList.contains('has-custom-cursor')) {
      document.body.classList.add('has-custom-cursor');
    }

    updatePosition(e.clientX, e.clientY);

    // Detect if hovering over text fields (show native I-beam)
    const isTextField = e.target && e.target.closest && e.target.closest('input, textarea');
    if (isTextField) {
      if (dot) dot.classList.add('cursor-text-mode');
      if (ring) ring.classList.add('cursor-text-mode');
      if (aura) aura.classList.add('cursor-text-mode');
      return;
    } else {
      if (dot) dot.classList.remove('cursor-text-mode');
      if (ring) ring.classList.remove('cursor-text-mode');
      if (aura) aura.classList.remove('cursor-text-mode');
    }

    // Detect if hovering interactive target
    const target = e.target && e.target.closest && e.target.closest(
      'a, button, select, .benchmark-card, .command-item, .workspace-btn, .subtab-btn, .cursor-pointer, [data-cursor], .dag-svg-node, .spotlight-card, .feature-tile, .pathway-card, .hero-glass-card, [role="button"], input[type="file"], label, .step-node'
    );
    if (target && !isHovering) {
      isHovering = true;
      if (ring) ring.classList.add('cursor-hover');
      document.body.classList.add('cursor-hover');
    } else if (!target && isHovering) {
      isHovering = false;
      if (ring) ring.classList.remove('cursor-hover');
      document.body.classList.remove('cursor-hover');
    }
  }, { passive: true });

  window.addEventListener('mousedown', (e) => {
    isMouseDown = true;
    if (e.clientX !== undefined && e.clientY !== undefined && !isNaN(e.clientX) && !isNaN(e.clientY)) {
      updatePosition(e.clientX, e.clientY);
    }
    if (ring) ring.classList.add('cursor-active');

    // Generate expanding ripple wave animation at cursor click coordinates
    try {
      const ripple = document.createElement('div');
      ripple.className = 'cursor-click-ripple';
      ripple.style.left = `${e.clientX}px`;
      ripple.style.top = `${e.clientY}px`;
      document.body.appendChild(ripple);
      setTimeout(() => ripple.remove(), 580);
    } catch {
      // safe fallback if DOM detached
    }
  });

  window.addEventListener('mouseup', () => {
    isMouseDown = false;
    if (ring) ring.classList.remove('cursor-active');
  });

  document.addEventListener('mouseleave', () => {
    if (dot) dot.style.opacity = '0';
    if (ring) ring.style.opacity = '0';
    if (aura) aura.style.opacity = '0';
  });

  document.addEventListener('mouseenter', (e) => {
    if (e.clientX !== undefined && e.clientY !== undefined && !isNaN(e.clientX) && !isNaN(e.clientY)) {
      updatePosition(e.clientX, e.clientY);
      ringX = e.clientX;
      ringY = e.clientY;
      auraX = e.clientX;
      auraY = e.clientY;
    }
    if (hasMoved) {
      if (dot) dot.style.opacity = '1';
      if (ring) ring.style.opacity = '1';
      if (aura) aura.style.opacity = '1';
    }
  });

  window.addEventListener('focus', () => {
    if (hasMoved && mouseX > 0 && mouseY > 0) {
      ringX = mouseX;
      ringY = mouseY;
      auraX = mouseX;
      auraY = mouseY;
    }
  });

  function renderCursor() {
    if (hasMoved && !isNaN(mouseX) && !isNaN(mouseY) && mouseX > -500 && mouseY > -500) {
      if (isNaN(ringX) || isNaN(ringY) || ringX < -500) {
        ringX = mouseX;
        ringY = mouseY;
      }
      if (isNaN(auraX) || isNaN(auraY) || auraX < -500) {
        auraX = mouseX;
        auraY = mouseY;
      }
      // Spring physics interpolation for outer ring
      ringX += (mouseX - ringX) * 0.20;
      ringY += (mouseY - ringY) * 0.20;

      // Smooth dynamic scale transition on click / hover
      const targetScale = isMouseDown ? 0.72 : (isHovering ? 1.38 : 1.0);
      currentScale += (targetScale - currentScale) * 0.22;

      if (ring) {
        ring.style.transform = `translate3d(${ringX.toFixed(2)}px, ${ringY.toFixed(2)}px, 0) scale(${currentScale.toFixed(4)})`;
      }

      // Smooth lag for ambient aura lighting
      if (aura) {
        auraX += (mouseX - auraX) * 0.12;
        auraY += (mouseY - auraY) * 0.12;
        aura.style.transform = `translate3d(${auraX.toFixed(2)}px, ${auraY.toFixed(2)}px, 0)`;
      }
    }
    requestAnimationFrame(renderCursor);
  }

  requestAnimationFrame(renderCursor);

  // Activate universal spotlight tracking across all cards
  setupSpotlightCards();
}

/**
 * Universal Spotlight Card Follow Effect
 * Computes --mouse-x and --mouse-y relative coordinates on any card
 * using delegated mousemove for high performance and dynamic DOM compatibility.
 */
export function setupSpotlightCards() {
  if (spotlightInitialized) return;
  spotlightInitialized = true;

  window.addEventListener('mousemove', (e) => {
    const card = e.target && e.target.closest && e.target.closest(
      '.spotlight-card, .feature-tile, .hero-glass-card, .pathway-card, .benchmark-card, .glass-card, .bento-card, .metric-card'
    );
    if (card) {
      if (activeSpotlightTarget && activeSpotlightTarget !== card) {
        activeSpotlightTarget.style.setProperty('--mouse-x', '-999px');
        activeSpotlightTarget.style.setProperty('--mouse-y', '-999px');
      }
      activeSpotlightTarget = card;
      const rect = card.getBoundingClientRect();
      card.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
      card.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
    } else if (activeSpotlightTarget) {
      activeSpotlightTarget.style.setProperty('--mouse-x', '-999px');
      activeSpotlightTarget.style.setProperty('--mouse-y', '-999px');
      activeSpotlightTarget = null;
    }
  }, { passive: true });
}
