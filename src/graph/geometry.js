/**
 * Node Studio — pure canvas geometry helpers.
 *
 * Everything the canvas needs to lay blocks out and hit-test interactions, separated
 * from the DOM so it can be unit-tested in Node (tests/test_graph_engine.js):
 * node rectangles, socket ports, bezier wires (path strings + point evaluation),
 * view transforms (pan/zoom) and socket snapping.
 *
 * File: src/graph/geometry.js
 */

export const NODE_WIDTH = 184;
export const HEADER_H = 30;
export const FIRST_ROW_Y = 24;   // first socket row offset below the header
export const ROW_H = 26;         // vertical spacing between socket rows
export const BOTTOM_PAD = 12;
export const PORT_RADIUS = 6;

/** Rect of a node in world coordinates. */
export function nodeSize(def) {
  const rows = Math.max((def && def.inputs ? def.inputs.length : 0), (def && def.outputs ? def.outputs.length : 0), 1);
  return {
    width: NODE_WIDTH,
    height: HEADER_H + FIRST_ROW_Y + rows * ROW_H + BOTTOM_PAD
  };
}

export function rectFromNode(node, def) {
  const size = nodeSize(def);
  return { x: node.x, y: node.y, w: size.width, h: size.height };
}

export function pointInRect(point, rect) {
  return point.x >= rect.x && point.x <= rect.x + rect.w && point.y >= rect.y && point.y <= rect.y + rect.h;
}

/** World position of one named socket port. */
export function socketPosition(node, def, socketName, direction) {
  const list = direction === 'out' ? def.outputs || [] : def.inputs || [];
  const index = Math.max(0, list.findIndex((s) => s.name === socketName));
  return {
    x: node.x + (direction === 'out' ? NODE_WIDTH : 0),
    y: node.y + HEADER_H + FIRST_ROW_Y + index * ROW_H + ROW_H / 2 - 4
  };
}

/** All ports of a node: [{name, kind, direction, x, y}] */
export function socketsOf(node, def) {
  const ports = [];
  for (const spec of def.inputs || []) {
    const pos = socketPosition(node, def, spec.name, 'in');
    ports.push({ name: spec.name, kind: spec.kind, direction: 'in', x: pos.x, y: pos.y });
  }
  for (const spec of def.outputs || []) {
    const pos = socketPosition(node, def, spec.name, 'out');
    ports.push({ name: spec.name, kind: spec.kind, direction: 'out', x: pos.x, y: pos.y });
  }
  return ports;
}

/* --------------------------------------------------------- wires */

export function bezierControlPoints(from, to) {
  const dx = Math.max(48, Math.abs(to.x - from.x) * 0.5);
  return {
    c1: { x: from.x + dx, y: from.y },
    c2: { x: to.x - dx, y: to.y }
  };
}

/** SVG path string for a wire from → to. */
export function bezierPath(from, to) {
  const { c1, c2 } = bezierControlPoints(from, to);
  return `M ${from.x} ${from.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${to.x} ${to.y}`;
}

/** Point at parameter t ∈ [0,1] on the wire (used by the flow particles). */
export function pointOnBezier(from, to, t) {
  const { c1, c2 } = bezierControlPoints(from, to);
  const u = 1 - t;
  const x = u * u * u * from.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * to.x;
  const y = u * u * u * from.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * to.y;
  return { x, y };
}

/* ------------------------------------------------------- transforms */

/** view = {x, y, scale}: world → screen and back. */
export function worldToScreen(point, view) {
  return { x: (point.x + view.x) * view.scale, y: (point.y + view.y) * view.scale };
}

export function screenToWorld(point, view) {
  return { x: point.x / view.scale - view.x, y: point.y / view.scale - view.y };
}

export function zoomViewAt(view, screenPoint, factor, minScale = 0.35, maxScale = 2.5) {
  const nextScale = Math.min(maxScale, Math.max(minScale, view.scale * factor));
  const world = screenToWorld(screenPoint, view);
  return {
    scale: nextScale,
    x: screenPoint.x / nextScale - world.x,
    y: screenPoint.y / nextScale - world.y
  };
}

/* ---------------------------------------------------------- snapping */

export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Nearest socket port to `point` within `radius`.
 * @param {Array} ports from socketsOf()/socketsOf across nodes
 * @param {object} filter { kind, direction } optional constraints
 */
export function hitTestSocket(ports, point, radius, filter = {}) {
  let best = null;
  let bestDistance = Infinity;
  for (const port of ports) {
    if (filter.kind && port.kind !== filter.kind) continue;
    if (filter.direction && port.direction !== filter.direction) continue;
    const d = distance(port, point);
    if (d <= radius && d < bestDistance) {
      best = port;
      bestDistance = d;
    }
  }
  return best;
}
