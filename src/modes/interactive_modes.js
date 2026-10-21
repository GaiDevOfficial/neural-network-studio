/**
 * Node Studio — three-tier interaction policy.
 *
 * One pure layer decides how the interactive mechanics behave per tier, so the canvas code and
 * the two popovers never hard-code vocabulary:
 *
 *   Tier 1 — Absolute Beginner (Playground + friendly missions)
 *     wire menu:  “Cut this wire?” — friendly endpoint names, no socket jargon
 *     quick menu: curated ≤ 8 blocks, friendly groups (“Make it think”, “Turn signals on/off”)
 *
 *   Tier 2 — Guided Beginner (Smart Node Assist)
 *     wire menu:  endpoint + socket kinds; reconnect repairs run through assistConnect
 *     quick menu: all compatible blocks by category, clean-compile badge, shape hints
 *
 *   Tier 3 — Advanced Studio (Full Manual)
 *     wire menu:  raw sockets + a real post-break topology warning (clone → disconnect → infer)
 *     quick menu: raw sockets, tensor shapes, param summaries and exact error codes
 *
 * Pure ES module — no DOM. Exercised by tests/test_quick_node_menu.js and the smoke test.
 *
 * File: src/modes/interactive_modes.js
 */

import { CATEGORY_META } from '../graph/node_types.js';
import { compile } from '../graph/node_engine.js';
import { friendlyLabel } from '../tutorial/explanations_catalog.js';
import { scoreCandidates, buildQuickMenuModel } from '../canvas/quick_node_menu.js';

export const TIER_META = {
  1: {
    id: 1,
    mode: 'playground',
    label: 'Absolute Beginner',
    blurb: 'Friendly names, guided drops, every action explained.'
  },
  2: {
    id: 2,
    mode: 'assist',
    label: 'Guided · Smart Assist',
    blurb: 'Real block names, automatic shape repairs, clean-compile answers.'
  },
  3: {
    id: 3,
    mode: 'manual',
    label: 'Advanced Studio',
    blurb: 'Raw sockets, strict wiring, exact error codes and topology warnings.'
  }
};

function clampTier(tier) {
  const value = Number(tier);
  if (value === 1 || value === 2 || value === 3) return value;
  return 2;
}

/** Mode → tier: playground=1, assist=2, manual=3. */
export function modeToTier(mode) {
  if (mode === 'playground') return 1;
  if (mode === 'manual') return 3;
  return 2;
}

/** The tier the interactions should use right now: an active mission may override the mode. */
export function activeTier(ctx = {}) {
  const missionTier = Number(ctx.missionTier);
  if (Number.isFinite(missionTier) && missionTier >= 1 && missionTier <= 3) return clampTier(missionTier);
  return modeToTier(ctx.mode);
}

/* ------------------------------------------------------------- wire menu */

export function wireMenuPolicy(tier) {
  const t = clampTier(tier);
  if (t === 1) {
    return {
      tier: 1,
      title: 'Cut this wire?',
      friendly: true,
      showSockets: false,
      topologyWarning: false,
      breakLabel: 'Cut the wire',
      breakDetail: 'The blocks stay connected to everything else — only this wire goes away. One undo brings it back.'
    };
  }
  if (t === 3) {
    return {
      tier: 3,
      title: 'Wire — raw sockets',
      friendly: false,
      showSockets: true,
      topologyWarning: true,
      breakLabel: 'Break connection (disconnect)',
      breakDetail: 'Runs graph.disconnect and re-runs shape inference. The warning below is computed on a cloned graph, not guessed.'
    };
  }
  return {
    tier: 2,
    title: 'Wire',
    friendly: false,
    showSockets: true,
    topologyWarning: false,
    breakLabel: 'Break connection',
    breakDetail: 'Detach the two sockets and re-run shape inference. Smart Assist repairs the reconnect if a shape needs a Flatten.'
  };
}

function countCodes(errors) {
  const counts = new Map();
  for (const issue of errors || []) counts.set(issue.code, (counts.get(issue.code) || 0) + 1);
  return counts;
}

/**
 * Decorate a `wireMenuBaseModel` with tier copy and (Advanced only) a real post-break topology
 * warning computed by cloning the graph, disconnecting the wire and re-running `inferGraph`.
 */
export function decorateWireMenu(model, options = {}) {
  if (!model) return null;
  const policy = wireMenuPolicy(options.tier);
  const decorated = {
    ...model,
    tier: policy.tier,
    title: policy.title,
    items: (model.items || []).map((item) => (item.id === 'break'
      ? { ...item, label: policy.breakLabel, detail: policy.breakDetail }
      : { ...item }))
  };

  if (policy.friendly) {
    const from = friendlyLabel(model.fromType || '');
    const to = friendlyLabel(model.toType || '');
    decorated.fromLabel = from;
    decorated.toLabel = to;
    decorated.detail = `${from} → ${to}`;
  } else if (policy.showSockets) {
    decorated.detail = model.detail;
  }

  if (policy.topologyWarning && options.graph) {
    try {
      const context = options.context || {};
      const before = compile(options.graph, context);
      const clone = options.graph.clone();
      // GraphState.clone() round-trips through JSON, and fromJSON mints fresh connection ids.
      // Resolve the cloned wire by its endpoints (id first, endpoints as the reliable fallback).
      const original = options.graph.connections.get
        ? options.graph.connections.get(model.connectionId)
        : null;
      let cloned = clone.connections.get(model.connectionId) || null;
      if (!cloned && original) {
        cloned = Array.from(clone.connections.values()).find((conn) =>
          conn.from.nodeId === original.from.nodeId
          && conn.from.socket === original.from.socket
          && conn.to.nodeId === original.to.nodeId
          && conn.to.socket === original.to.socket) || null;
      }
      if (cloned) clone.disconnect(cloned.id);
      const after = compile(clone, context);

      const beforeCounts = countCodes(before.errors);
      const afterCounts = countCodes(after.errors);
      const added = [];
      for (const [code, count] of afterCounts.entries()) {
        if (count > (beforeCounts.get(code) || 0)) added.push(code);
      }
      const addedCodes = Array.from(new Set(added));
      const beforeWarn = countCodes(before.warnings);
      const afterWarn = countCodes(after.warnings);
      const addedWarnings = [];
      for (const [code, count] of afterWarn.entries()) {
        if (count > (beforeWarn.get(code) || 0)) addedWarnings.push(code);
      }

      if (addedCodes.length > 0) {
        decorated.warning = `This cut breaks the build: ${addedCodes.join(', ')}.`;
      } else if (!before.ok && after.ok) {
        decorated.warning = 'The cut actually makes the build valid — the wire was feeding the broken branch.';
      } else if (after.ok && before.ok) {
        const layersBefore = before.network ? before.network.architecture.length : 0;
        const layersAfter = after.network ? after.network.architecture.length : 0;
        if (layersAfter !== layersBefore) {
          decorated.warning = `The build still compiles, but the network shrinks from ${layersBefore} to ${layersAfter} compiled layer(s).`;
        } else if (addedWarnings.length > 0) {
          decorated.warning = `The build still compiles; the cut adds warnings: ${Array.from(new Set(addedWarnings)).join(', ')}.`;
        } else {
          decorated.warning = 'The build still compiles after this cut — only this wire is removed.';
        }
      } else if (after.errors.length > 0) {
        decorated.warning = `This cut leaves ${after.errors.length} build problem(s): ${Array.from(new Set(after.errors.map((issue) => issue.code))).join(', ')}.`;
      } else {
        decorated.warning = 'The build stays valid after this cut.';
      }
    } catch (err) {
      decorated.warning = `Topology preview failed: ${err.message}`;
    }
  }

  return decorated;
}

/* ------------------------------------------------------------ quick menu */

export function quickMenuPolicy(tier) {
  const t = clampTier(tier);
  if (t === 1) {
    return {
      tier: 1,
      title: 'Grow the brain',
      maxItems: 8,
      intentGroups: true,
      friendly: true,
      showDetail: false,
      showReasons: false,
      showBlocked: true,
      groupOrder: [
        'Feed it examples',
        'Make it think',
        'Turn signals on or off',
        'Score and steer learning',
        'Training controls',
        'Watch it learn'
      ]
    };
  }
  if (t === 3) {
    return {
      tier: 3,
      title: 'Place a block — raw sockets',
      maxItems: 0,
      intentGroups: false,
      friendly: false,
      showDetail: true,
      showReasons: true,
      showBlocked: true,
      groupOrder: Object.keys(CATEGORY_META)
    };
  }
  return {
    tier: 2,
    title: 'Place a block',
    maxItems: 14,
    intentGroups: false,
    friendly: false,
    showDetail: false,
    showReasons: false,
    showBlocked: true,
    groupOrder: Object.keys(CATEGORY_META)
  };
}

/**
 * Full quick-menu pipeline for one drop: score the candidates on the real compiler, then shape the
 * view model for the active tier. Returns `{ tier, policy, rows, model }` so callers (and tests)
 * can inspect the measured rows as well as the display model.
 */
export function buildQuickMenu(options = {}) {
  const { graph, drag, position } = options;
  const ctx = options.ctx || {};
  const tier = activeTier(ctx);
  const policy = quickMenuPolicy(tier);
  const rows = scoreCandidates(graph, drag, {
    mode: ctx.mode,
    context: ctx.context,
    position: position || (drag && drag.position) || { x: 0, y: 0 }
  });
  const model = buildQuickMenuModel({ drag, rows, policy, query: options.query || '' });
  return { tier, policy, rows, model };
}

/** Convenience for the page: the tier for a mission overlay or the current mode. */
export function tierContext(state = {}, missions = null) {
  return {
    mode: state.mode,
    missionTier: missions && missions.active ? missions.tier : 0
  };
}
