/**
 * Pure geometry for the guided overlays (mission card + tour tooltip).
 *
 * The studio canvas is not a clean modal surface: nodes move, wires are dragged, panels scroll, and
 * the palette / inspector stay interactive while a guide is open. Placement therefore has to satisfy
 * two different constraints:
 *
 *   - never cover the element the learner is being asked to touch (a hard constraint), and
 *   - stay out of the fixed app chrome when any alternative exists (a soft constraint).
 *
 * Keeping the math here makes the overlay presenters small and lets the Node suites exercise every
 * edge case without a browser.
 *
 * File: src/tutorial/overlay_layout.js
 */

const DEFAULT_MARGIN = 12;
const DEFAULT_GAP = 16;

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function normalizeRect(rect) {
  if (!rect) return null;
  const left = Number(rect.left) || 0;
  const top = Number(rect.top) || 0;
  const width = Number.isFinite(rect.width)
    ? Math.max(0, rect.width)
    : Math.max(0, (Number(rect.right) || left) - left);
  const height = Number.isFinite(rect.height)
    ? Math.max(0, rect.height)
    : Math.max(0, (Number(rect.bottom) || top) - top);
  return {
    left,
    top,
    width,
    height,
    right: Number.isFinite(rect.right) ? rect.right : left + width,
    bottom: Number.isFinite(rect.bottom) ? rect.bottom : top + height
  };
}

export function expandRect(rect, pad) {
  const r = normalizeRect(rect);
  if (!r) return null;
  return {
    left: r.left - pad,
    top: r.top - pad,
    width: r.width + pad * 2,
    height: r.height + pad * 2,
    right: r.right + pad,
    bottom: r.bottom + pad
  };
}

export function intersectionArea(a, b) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);
  if (!ra || !rb) return 0;
  const width = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
  const height = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
  if (width <= 0 || height <= 0) return 0;
  return width * height;
}

/** Read a viewport-space rect from a real DOM element (or return null). */
export function elementRect(el) {
  if (!el || typeof el.getBoundingClientRect !== 'function') return null;
  return normalizeRect(el.getBoundingClientRect());
}

/** Measure a set of app-chrome selectors; used only as a soft avoidance list. */
export function measureChrome(doc, selectors) {
  const rects = [];
  if (!doc || typeof doc.querySelector !== 'function') return rects;
  for (const selector of selectors) {
    let el = null;
    try {
      el = doc.querySelector(selector);
    } catch {
      el = null;
    }
    const rect = elementRect(el);
    if (rect && rect.width > 0 && rect.height > 0) rects.push(rect);
  }
  return rects;
}

/** A target bigger than this fraction of the viewport is acting as the workspace backdrop. */
export function isBackdropTarget(rect, viewportWidth, viewportHeight) {
  const r = normalizeRect(rect);
  if (!r) return false;
  const viewportArea = Math.max(1, viewportWidth * viewportHeight);
  return (r.width * r.height) / viewportArea >= 0.5;
}

/**
 * Place a fixed-size floating panel next to `target`.
 *
 * Candidates are tried in `prefer` order, clamped into the viewport, then sorted lexicographically:
 * any placement that avoids the target (plus its 10 px breathing room) outranks every placement that
 * clips it; among those, the least overlap with the app chrome wins, then the preferred order. That
 * keeps the panel off the live control while still docking it over the empty canvas instead of the
 * palette / inspector.
 *
 * `hardAvoid: false` is used for a full-canvas target (intro steps): the panel is allowed to sit in
 * the workspace, but is still steered away from the fixed chrome.
 */
export function placeFloatingPanel(options = {}) {
  const target = normalizeRect(options.target);
  const viewportWidth = Math.max(1, Number(options.viewportWidth) || 1);
  const viewportHeight = Math.max(1, Number(options.viewportHeight) || 1);
  const margin = Number.isFinite(options.margin) ? options.margin : DEFAULT_MARGIN;
  const gap = Number.isFinite(options.gap) ? options.gap : DEFAULT_GAP;
  const hardAvoid = options.hardAvoid !== false;
  const softAvoid = Array.isArray(options.softAvoid) ? options.softAvoid : [];
  const prefer = Array.isArray(options.prefer) && options.prefer.length
    ? options.prefer
    : ['right', 'left', 'below', 'above'];

  const maxWidth = Math.max(1, viewportWidth - margin * 2);
  const maxHeight = Math.max(1, viewportHeight - margin * 2);
  const width = Math.min(Math.max(1, Number(options.width) || 1), maxWidth);
  const height = Math.min(Math.max(1, Number(options.height) || 1), maxHeight);
  const maxX = Math.max(margin, viewportWidth - width - margin);
  const maxY = Math.max(margin, viewportHeight - height - margin);

  const targetRect = target || { left: margin, top: margin, width: 0, height: 0, right: margin, bottom: margin };
  const hardRect = hardAvoid ? expandRect(targetRect, 10) : null;

  const candidates = [];
  prefer.forEach((side, index) => {
    let x = margin;
    let y = margin;
    if (side === 'right') {
      x = targetRect.right + gap;
      y = targetRect.top + targetRect.height / 2 - height / 2;
    } else if (side === 'left') {
      x = targetRect.left - width - gap;
      y = targetRect.top + targetRect.height / 2 - height / 2;
    } else if (side === 'below') {
      x = targetRect.left + targetRect.width / 2 - width / 2;
      y = targetRect.bottom + gap;
    } else if (side === 'above') {
      x = targetRect.left + targetRect.width / 2 - width / 2;
      y = targetRect.top - height - gap;
    } else {
      return;
    }
    const left = clamp(x, margin, maxX);
    const top = clamp(y, margin, maxY);
    const panel = { left, top, width, height, right: left + width, bottom: top + height };
    const targetOverlap = hardRect ? intersectionArea(panel, hardRect) : 0;
    let chromeOverlap = 0;
    for (const rect of softAvoid) chromeOverlap += intersectionArea(panel, rect);
    candidates.push({
      side,
      index,
      x: left,
      y: top,
      width,
      height,
      targetOverlap,
      chromeOverlap
    });
  });

  if (candidates.length === 0) {
    return {
      side: 'center',
      x: clamp(viewportWidth / 2 - width / 2, margin, maxX),
      y: clamp(viewportHeight / 2 - height / 2, margin, maxY),
      width,
      height,
      targetOverlap: 0,
      chromeOverlap: 0
    };
  }

  // Lexicographic, so "do not cover the target" is genuinely hard: any placement that avoids the
  // target beats every placement that clips it, regardless of how much chrome it would cover.
  candidates.sort((a, b) => {
    if ((a.targetOverlap > 0) !== (b.targetOverlap > 0)) return a.targetOverlap > 0 ? 1 : -1;
    if (a.targetOverlap !== b.targetOverlap) return a.targetOverlap - b.targetOverlap;
    if (a.chromeOverlap !== b.chromeOverlap) return a.chromeOverlap - b.chromeOverlap;
    return a.index - b.index;
  });
  const best = candidates[0];
  return {
    side: best.side,
    x: best.x,
    y: best.y,
    width: best.width,
    height: best.height,
    targetOverlap: best.targetOverlap,
    chromeOverlap: best.chromeOverlap
  };
}

/** The fixed studio chrome that a guide should avoid when there is room. */
export const STUDIO_CHROME_SELECTORS = [
  '.ns-topbar',
  '.ns-palette',
  '.ns-right',
  '.ns-statusbar'
];
