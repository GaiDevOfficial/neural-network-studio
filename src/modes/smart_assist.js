/**
 * Smart Assist mode module (Tier 2).
 *
 * The repair/auto-wire algorithms live in `src/graph/smart_assist.js` so they stay pure and testable.
 * This module is the mode-level facade the canvas uses:
 *   - re-exports the repair engine,
 *   - `suggestConnection()` — a real ghost suggestion scored by compiling candidate graphs,
 *   - `snapCandidates()` / `snapTarget()` — wider magnetic socket targeting for Beginner mode.
 *
 * Pure ES module (no DOM). File: src/modes/smart_assist.js
 */

import { compile } from '../graph/node_engine.js';
import { getNodeType } from '../graph/node_types.js';
import {
  assistConnect,
  autoWireOnDrop,
  autoFixGraph,
  autoFixGraphToConvergence,
  instantPreset,
  presetList
} from '../graph/smart_assist.js';

export {
  assistConnect,
  autoWireOnDrop,
  autoFixGraph,
  autoFixGraphToConvergence,
  instantPreset,
  presetList
};

/**
 * Suggest the single wire that most reduces the real compile-error count.
 *
 * Candidates are tried on an isolated clone and scored with the same compiler the Run button uses.
 * Returns `{ from, to, errors, errorReduction, message }` or null when nothing improves.
 */
export function suggestConnection(graph, selectedNodeId, context = {}) {
  const node = selectedNodeId ? graph.getNode(selectedNodeId) : null;
  if (!node) return null;
  const def = getNodeType(node.type);
  if (!def || !def.outputs || def.outputs.length === 0) return null;

  const baseline = compile(graph, context);
  const baselineErrors = baseline.errors ? baseline.errors.length : 0;
  if (baselineErrors === 0) return null;

  let best = null;
  for (const target of graph.nodeList()) {
    if (target.id === node.id) continue;
    const targetDef = getNodeType(target.type);
    if (!targetDef || !targetDef.inputs) continue;
    for (const inSocket of targetDef.inputs) {
      if (graph.incoming(target.id, inSocket.name)) continue;
      for (const outSocket of def.outputs) {
        if (outSocket.kind !== inSocket.kind) continue;
        if (graph.canReach(target.id, node.id)) continue;

        const clone = graph.clone();
        const wired = clone.connect(node.id, outSocket.name, target.id, inSocket.name);
        if (!wired.ok) continue;
        const plan = compile(clone, context);
        const errors = plan.ok ? 0 : (plan.errors ? plan.errors.length : baselineErrors);
        if (errors < baselineErrors && (!best || errors < best.errors)) {
          best = {
            from: { nodeId: node.id, socket: outSocket.name },
            to: { nodeId: target.id, socket: inSocket.name },
            errors,
            errorReduction: baselineErrors - errors
          };
        }
      }
    }
  }

  if (!best) return null;
  const targetNode = graph.getNode(best.to.nodeId);
  const targetDef = getNodeType(targetNode.type);
  best.message = `Connect ${def.label}.${best.from.socket} → ${targetDef.label}.${best.to.socket} (removes ${best.errorReduction} validation issue${best.errorReduction === 1 ? '' : 's'}).`;
  return best;
}

/** Every compatible socket inside the magnetic radius, nearest first. */
export function snapCandidates(ports, worldPoint, drag, radius = 34) {
  if (!drag) return [];
  const wantedDirection = drag.direction === 'out' ? 'in' : 'out';
  const result = [];
  for (const port of ports) {
    if (port.direction !== wantedDirection) continue;
    if (port.kind !== drag.kind) continue;
    if (port.nodeId === drag.nodeId) continue;
    const distance = Math.hypot(port.x - worldPoint.x, port.y - worldPoint.y);
    if (distance <= radius) result.push({ ...port, distance });
  }
  result.sort((a, b) => a.distance - b.distance);
  return result;
}

/** Nearest compatible socket for a wire drag (Beginner snaps at 34 px, Advanced at 18 px). */
export function snapTarget(ports, worldPoint, drag, options = {}) {
  const radius = Number.isFinite(options.radius) ? options.radius : (options.mode === 'assist' ? 34 : 18);
  const candidates = snapCandidates(ports, worldPoint, drag, radius);
  return candidates.length > 0 ? candidates[0] : null;
}
