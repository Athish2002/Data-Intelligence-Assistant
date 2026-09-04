/**
 * frontend/js/cursor.js
 * ─────────────────────
 * Precision Interactive Custom Cursor with text caret suppression and hover physics.
 */

export function setupCustomCursor() {
  const dot = document.getElementById('cursor-dot');
  const ring = document.getElementById('cursor-ring');
  if (!dot || !ring) return;

  const isFinePointer = window.matchMedia('(pointer: fine) and (hover: hover)').matches;
  if (!isFinePointer) return;

  document.body.classList.add('has-custom-cursor');

  let mouseX = -100;
  let mouseY = -100;
  let ringX = -100;
  let ringY = -100;
  let isHovering = false;
  let isMouseDown = false;
  let isInitialized = false;

  window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;

    if (!isInitialized) {
      isInitialized = true;
      ringX = mouseX;
      ringY = mouseY;
      dot.style.visibility = 'visible';
      ring.style.visibility = 'visible';
      dot.style.opacity = '1';
      ring.style.opacity = '1';
    }

    dot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0)`;

    // Detect if hovering over text fields (show native I-beam)
    const isTextField = e.target.closest('input, textarea');
    if (isTextField) {
      dot.classList.add('cursor-text-mode');
      ring.classList.add('cursor-text-mode');
      return;
    } else {
      dot.classList.remove('cursor-text-mode');
      ring.classList.remove('cursor-text-mode');
    }

    // Detect if hovering interactive target
    const target = e.target.closest(
      'a, button, select, .benchmark-card, .command-item, .workspace-btn, .subtab-btn, .cursor-pointer, [data-cursor]'
    );
    if (target && !isHovering) {
      isHovering = true;
      ring.classList.add('cursor-hover');
    } else if (!target && isHovering) {
      isHovering = false;
      ring.classList.remove('cursor-hover');
    }
  });

  window.addEventListener('mousedown', () => {
    isMouseDown = true;
    ring.classList.add('cursor-active');
  });

  window.addEventListener('mouseup', () => {
    isMouseDown = false;
    ring.classList.remove('cursor-active');
  });

  document.addEventListener('mouseleave', () => {
    dot.style.opacity = '0';
    ring.style.opacity = '0';
  });

  document.addEventListener('mouseenter', () => {
    if (isInitialized) {
      dot.style.opacity = '1';
      ring.style.opacity = '1';
    }
  });

  window.addEventListener('focus', () => {
    ringX = mouseX;
    ringY = mouseY;
  });

  function renderCursor() {
    if (isInitialized) {
      ringX += (mouseX - ringX) * 0.18;
      ringY += (mouseY - ringY) * 0.18;
      ring.style.transform = `translate3d(${ringX}px, ${ringY}px, 0)`;
    }
    requestAnimationFrame(renderCursor);
  }

  requestAnimationFrame(renderCursor);
}
