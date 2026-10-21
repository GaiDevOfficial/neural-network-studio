/**
 * Node Studio — interactive wire mechanics.
 *
 * Wires are cubic Béziers in world coordinates inside a transformed SVG layer with
 * `pointer-events: none`, so hit-testing is done by sampling the curve geometry instead of
 * DOM events. This module provides:
 *
 *   - pure raycasting helpers (`flattenBezier`, `distanceToPolyline`, `hitTestWires`),
 *   - `collectWires(graph)` — the live wire list with real socket positions and SVG paths,
 *   - `wireMenuBaseModel` — the context-menu data model,
 *   - `WireBreakAnimator` — the collapse animation before a real `disconnect`,
 *   - `WireContextMenu` — a MenuManager popover with the “Break connection” action,
 *   - `WireInteractions` — the coordinator the canvas binds to.
 *
 * Breaking a wire always goes through the page's `onBreak(connectionId)` callback, which performs
 * the real `GraphState.disconnect` + re-inference. Nothing here mutates the graph itself.
 *
 * File: src/canvas/wire_interactions.js
 */

import { getNodeType } from '../graph/node_types.js';
import { socketPosition, bezierPath, pointOnBezier } from '../graph/geometry.js';

export const DEFAULT_HIT_RADIUS = 10;
export const MIN_HIT_RADIUS = 5;
export const MAX_HIT_RADIUS = 22;
export const WIRE_BREAK_MS = 180;

/* ------------------------------------------------------------- raycasting */

/**
 * Sample a wire into a polyline. `segments` is the number of straight pieces; the polyline has
 * `segments + 1` points. Works for any from/to pair the SVG renderer draws.
 */
export function flattenBezier(from, to, segments = 28) {
  const count = Math.max(2, Math.floor(segments) + 1);
  const points = [];
  for (let i = 0; i < count; i++) {
    points.push(pointOnBezier(from, to, i / (count - 1)));
  }
  return points;
}

/** Shortest distance from `p` to segment a→b, plus the closest point. */
export function distancePointToSegment(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  let t = 0;
  if (lengthSq > 0) {
    t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));
  }
  const point = { x: a.x + t * dx, y: a.y + t * dy };
  return { distance: Math.hypot(p.x - point.x, p.y - point.y), point, t };
}

/**
 * Distance from `p` to a polyline, plus the normalized position `t ∈ [0,1]` along the polyline
 * (used for placing labels / starting the break animation at the click point).
 */
export function distanceToPolyline(p, polyline) {
  if (!Array.isArray(polyline) || polyline.length === 0) {
    return { distance: Infinity, t: 0, point: { x: p.x, y: p.y } };
  }
  if (polyline.length === 1) {
    const d = Math.hypot(p.x - polyline[0].x, p.y - polyline[0].y);
    return { distance: d, t: 0, point: { x: polyline[0].x, y: polyline[0].y } };
  }

  const lengths = [0];
  for (let i = 1; i < polyline.length; i++) {
    lengths.push(lengths[i - 1] + Math.hypot(polyline[i].x - polyline[i - 1].x, polyline[i].y - polyline[i - 1].y));
  }
  const total = lengths[lengths.length - 1] || 1;

  let best = { distance: Infinity, t: 0, point: { x: polyline[0].x, y: polyline[0].y } };
  for (let i = 1; i < polyline.length; i++) {
    const seg = distancePointToSegment(p, polyline[i - 1], polyline[i]);
    if (seg.distance < best.distance) {
      const t = (lengths[i - 1] + seg.t * (lengths[i] - lengths[i - 1])) / total;
      best = { distance: seg.distance, t, point: seg.point };
    }
  }
  return best;
}

/** Constant feel across zoom levels: radius shrinks/grows inversely with the view scale. */
export function wireHitRadius(scale, base = DEFAULT_HIT_RADIUS) {
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1;
  return Math.min(MAX_HIT_RADIUS, Math.max(MIN_HIT_RADIUS, base / s));
}

/**
 * Nearest wire to `point` within `radius`.
 * @param {Array} wires entries from `collectWires()` (needs id/fromPos/toPos)
 * @returns {{wire: object, distance: number, t: number, point: {x,y}}|null}
 */
export function hitTestWires(wires, point, radius) {
  let best = null;
  for (const wire of wires || []) {
    if (!wire || !wire.fromPos || !wire.toPos) continue;
    const polyline = wire.polyline || flattenBezier(wire.fromPos, wire.toPos, 28);
    const result = distanceToPolyline(point, polyline);
    if (result.distance <= radius && (!best || result.distance < best.distance)) {
      best = { wire, distance: result.distance, t: result.t, point: result.point };
    }
  }
  return best;
}

/* ------------------------------------------------------------ wire list */

/**
 * Build the live wire list from a real GraphState:
 * `[{ id, from, to, kind, path, polyline, fromPos, toPos, fromLabel, toLabel }]`.
 * Nodes without a registered type are skipped (they cannot compile anyway).
 */
export function collectWires(graph) {
  const wires = [];
  if (!graph) return wires;
  for (const connection of graph.connections.values()) {
    const fromNode = graph.getNode(connection.from.nodeId);
    const toNode = graph.getNode(connection.to.nodeId);
    if (!fromNode || !toNode) continue;
    const fromDef = getNodeType(fromNode.type);
    const toDef = getNodeType(toNode.type);
    if (!fromDef || !toDef) continue;
    const outSpec = (fromDef.outputs || []).find((s) => s.name === connection.from.socket);
    const kind = outSpec ? outSpec.kind : 'tensor';
    const fromPos = socketPosition(fromNode, fromDef, connection.from.socket, 'out');
    const toPos = socketPosition(toNode, toDef, connection.to.socket, 'in');
    wires.push({
      id: connection.id,
      from: { ...connection.from },
      to: { ...connection.to },
      fromType: fromNode.type,
      toType: toNode.type,
      kind,
      fromPos,
      toPos,
      path: bezierPath(fromPos, toPos),
      polyline: flattenBezier(fromPos, toPos, 28),
      fromLabel: `${fromDef.label} · ${connection.from.socket} (${kind})`,
      toLabel: `${toDef.label} · ${connection.to.socket} (${kind})`
    });
  }
  return wires;
}

/* ------------------------------------------------------- menu data model */

/**
 * Base context-menu model consumed by the tier policy in `interactive_modes.js`.
 * `hit` is the result of `hitTestWires` (optional; used for the click position label).
 */
export function wireMenuBaseModel(wire, hit = null) {
  if (!wire) return null;
  return {
    connectionId: wire.id,
    kind: wire.kind,
    title: 'Wire',
    fromType: wire.fromType || null,
    toType: wire.toType || null,
    fromLabel: wire.fromLabel || wire.from.nodeId,
    toLabel: wire.toLabel || wire.to.nodeId,
    detail: `${wire.from.nodeId} → ${wire.to.nodeId}`,
    warning: null,
    position: hit ? { t: hit.t, x: hit.point.x, y: hit.point.y } : null,
    items: [{ id: 'break', label: 'Break connection', danger: true, detail: 'Detach the two sockets and re-run shape inference.' }]
  };
}

/* ---------------------------------------------------------- break animator */

/**
 * Collapse animation for a wire path. The path element is marked `.is-breaking` (CSS handles the
 * dash/opacity transition) and removed from the animation state after `duration` ms via the
 * injectable scheduler, so tests can run on a fake clock.
 */
export class WireBreakAnimator {
  constructor(options = {}) {
    this.duration = Number.isFinite(options.duration) ? options.duration : WIRE_BREAK_MS;
    this.schedule = options.schedule || ((fn, ms) => setTimeout(fn, ms));
    this.playing = new Set();
  }

  /** True while the given path element is mid-animation. */
  isPlaying(pathEl) {
    return Boolean(pathEl && this.playing.has(pathEl));
  }

  play(pathEl, onDone) {
    if (!pathEl || !pathEl.classList) {
      if (onDone) onDone();
      return false;
    }
    if (this.playing.has(pathEl)) return false;
    this.playing.add(pathEl);
    pathEl.classList.add('is-breaking');
    this.schedule(() => {
      this.playing.delete(pathEl);
      if (pathEl.classList) pathEl.classList.remove('is-breaking');
      if (onDone) onDone();
    }, this.duration);
    return true;
  }
}

/* ------------------------------------------------------------- context menu */

const MENU_ID = 'wire-menu';

/**
 * Cursor-anchored context menu registered with MenuManager (Escape / outside-click / injected ×).
 * The page owns the model; this class only renders and forwards `onAction(itemId, model)`.
 */
export class WireContextMenu {
  constructor(options = {}) {
    this.doc = options.document || (typeof document !== 'undefined' ? document : null);
    this.win = options.window || (typeof window !== 'undefined' ? window : null);
    this.menuManager = options.menuManager || null;
    this.menuId = options.menuId || MENU_ID;
    this.onAction = options.onAction || null;
    this.model = null;
    this.panel = null;
    this.head = null;
    this.meta = null;
    this.warning = null;
    this.itemHost = null;
    if (this.doc) this._build();
  }

  _build() {
    const panel = this.doc.createElement('div');
    panel.className = 'ns-menu ns-menu-cursor ns-wire-menu';
    panel.hidden = true;

    this.head = this.doc.createElement('div');
    this.head.className = 'ns-menu-cursor-head';
    panel.appendChild(this.head);

    this.meta = this.doc.createElement('div');
    this.meta.className = 'ns-wire-menu-meta';
    panel.appendChild(this.meta);

    this.warning = this.doc.createElement('div');
    this.warning.className = 'ns-wire-menu-warning';
    this.warning.hidden = true;
    panel.appendChild(this.warning);

    this.itemHost = this.doc.createElement('div');
    this.itemHost.className = 'ns-wire-menu-items';
    panel.appendChild(this.itemHost);

    this.panel = panel;
    // MenuManager owns lifecycle (show/hide/Escape), not mounting: attach the panel once so the
    // popover is actually part of the document (JS-created panels are invisible otherwise).
    if (this.doc && this.doc.body && panel.parentNode !== this.doc.body) {
      this.doc.body.appendChild(panel);
    }
    if (this.menuManager) {
      this.menuManager.register({ id: this.menuId, panel, kind: 'popover' });
    }
  }

  render(model) {
    if (!this.doc || !this.panel) return;
    this.model = model;
    this.head.textContent = model.title || 'Wire';
    this.meta.textContent = `${model.fromLabel}  →  ${model.toLabel}`;
    if (model.warning) {
      this.warning.textContent = model.warning;
      this.warning.hidden = false;
    } else {
      this.warning.textContent = '';
      this.warning.hidden = true;
    }

    this.itemHost.innerHTML = '';
    for (const item of model.items || []) {
      const button = this.doc.createElement('button');
      button.type = 'button';
      button.className = `ns-wire-menu-item${item.danger ? ' is-danger' : ''}`;
      button.dataset.wireAction = item.id;
      button.textContent = item.label;
      if (item.detail) button.title = item.detail;
      button.addEventListener('click', () => this._activate(item));
      this.itemHost.appendChild(button);
    }
  }

  _activate(item) {
    const model = this.model;
    this.close();
    if (this.onAction) this.onAction(item.id, model);
  }

  open(model, position = {}) {
    if (!this.doc || !this.panel) return false;
    this.render(model);
    const margin = 8;
    const maxX = this.win && Number.isFinite(this.win.innerWidth) ? this.win.innerWidth : 1200;
    const maxY = this.win && Number.isFinite(this.win.innerHeight) ? this.win.innerHeight : 800;
    const x = Math.max(margin, Math.min(Number(position.clientX) || 0, maxX - 260));
    const y = Math.max(margin, Math.min(Number(position.clientY) || 0, maxY - 160));
    this.panel.style.position = 'fixed';
    this.panel.style.left = `${x}px`;
    this.panel.style.top = `${y}px`;
    this.panel.style.right = 'auto';
    if (this.menuManager) {
      this.menuManager.open(this.menuId);
      return true;
    }
    this.panel.hidden = false;
    return true;
  }

  close() {
    if (this.menuManager) {
      this.menuManager.close(this.menuId, { reason: 'wire-menu' });
    } else if (this.panel) {
      this.panel.hidden = true;
    }
    this.model = null;
  }

  get active() {
    return Boolean(this.panel && !this.panel.hidden);
  }
}

/* --------------------------------------------------------------- coordinator */

/**
 * Binds the canvas `contextmenu` event to the wire raycast and drives the break lifecycle.
 * All page coupling is via injected callbacks:
 *   getWires() → collectWires(graph)      getScale() → state.view.scale
 *   onBreak(connectionId)                  onMiss(worldPoint)
 *   menuModelFor(wire, hit) → tier-decorated model
 */
export class WireInteractions {
  constructor(options = {}) {
    this.wireLayer = options.wireLayer || null;
    this.menuManager = options.menuManager || null;
    this.getWires = options.getWires || (() => []);
    this.getScale = options.getScale || (() => 1);
    this.toWorld = options.toWorld || ((point) => point);
    this.menuModelFor = options.menuModelFor || wireMenuBaseModel;
    this.onBreak = options.onBreak || null;
    this.onMiss = options.onMiss || null;
    this.onOpen = options.onOpen || null;
    this.animator = options.animator || new WireBreakAnimator();
    this.menu = options.menu || new WireContextMenu({
      document: options.document,
      window: options.window,
      menuManager: this.menuManager,
      onAction: (itemId, model) => this.handleAction(itemId, model)
    });
    this.lastHit = null;
    this._boundContext = (event) => this.handleContextMenu(event);
  }

  attach(canvas) {
    if (!canvas || !canvas.addEventListener) return;
    canvas.addEventListener('contextmenu', this._boundContext);
  }

  detach(canvas) {
    if (!canvas || !canvas.removeEventListener) return;
    canvas.removeEventListener('contextmenu', this._boundContext);
  }

  handleContextMenu(event) {
    if (event && event.preventDefault) event.preventDefault();
    if (!event) return null;
    const client = { x: event.clientX || 0, y: event.clientY || 0 };
    const world = this.toWorld(client);
    const scale = this.getScale();
    const hit = hitTestWires(this.getWires(), world, wireHitRadius(scale));
    if (!hit) {
      this.lastHit = null;
      if (this.onMiss) this.onMiss(world);
      return null;
    }
    this.lastHit = hit;
    const model = this.menuModelFor(hit.wire, hit);
    if (this.onOpen) this.onOpen(hit, model);
    this.menu.open(model, { clientX: client.x, clientY: client.y });
    return hit;
  }

  /** Break the wire represented by the open menu, with the collapse animation first. */
  requestBreak(connectionId) {
    const id = connectionId;
    const path = this.wireLayer && this.wireLayer.querySelector ? this.wireLayer.querySelector(`[data-id="${id}"]`) : null;
    this.animator.play(path, () => {
      if (this.onBreak) this.onBreak(id);
    });
    // Even if the path element was missing, the animation callback still fires synchronously there,
    // so the graph always ends up consistent.
    return true;
  }

  handleAction(itemId, model) {
    if (!model) return false;
    if (itemId === 'break') return this.requestBreak(model.connectionId);
    return false;
  }
}
