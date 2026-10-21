/**
 * Node Studio — drag-wire-to-empty-space quick node menu.
 *
 * When a wire drag ends over empty canvas, the studio intercepts the drop and offers only the
 * blocks that can legally accept (or feed) the dragged socket. Candidate ranking is not a
 * heuristic: every candidate is scored by cloning the live graph, adding the block, performing
 * the wire, and re-running the real `inferGraph` compiler — so “compatible” means “the engine
 * accepts it”, and blocked candidates carry the exact error code they would raise.
 *
 * Pure logic (`baseCandidates`, `scoreCandidates`, `applyQuickAdd`, model builders) is
 * DOM-free; `QuickNodeMenu` is the cursor-anchored popover registered with MenuManager.
 *
 * File: src/canvas/quick_node_menu.js
 */

import { CATEGORY_META, NODE_TYPES, defaultParams, getNodeType, SOURCE_NODE_TYPES, LOSS_NODE_TYPES } from '../graph/node_types.js';
import { inferGraph } from '../graph/node_engine.js';
import { assistConnect } from '../graph/smart_assist.js';
import { explain, friendlyLabel, technicalLine, searchTerms } from '../tutorial/explanations_catalog.js';

export const QUICK_MENU_ID = 'quick-node-menu';

/* ------------------------------------------------------------ candidates */

/**
 * The candidate socket for a dragged kind/direction.
 * An `out` drag needs a candidate INPUT; an `in` drag needs a candidate OUTPUT.
 */
export function socketForCandidate(type, kind, dragDirection) {
  const def = NODE_TYPES[type];
  if (!def || def.supported === false) return null;
  const wanted = dragDirection === 'out' ? (def.inputs || []) : (def.outputs || []);
  const socket = wanted.find((spec) => spec.kind === kind);
  return socket ? { name: socket.name, kind: socket.kind, direction: dragDirection === 'out' ? 'in' : 'out' } : null;
}

/** All supported blocks owning a socket compatible with the drag, before scoring. */
export function baseCandidates(drag) {
  if (!drag || !drag.kind) return [];
  const rows = [];
  for (const [type, def] of Object.entries(NODE_TYPES)) {
    if (def.supported === false) continue;
    const socket = socketForCandidate(type, drag.kind, drag.direction);
    if (!socket) continue;
    rows.push({
      type,
      label: def.label,
      friendly: friendlyLabel(type),
      category: def.category,
      categoryLabel: (CATEGORY_META[def.category] || {}).label || def.category,
      summary: def.summary || '',
      socket: socket.name,
      direction: socket.direction,
      kind: socket.kind,
      aliases: searchTerms(type)
    });
  }
  return rows;
}

/**
 * Score candidates against the real compiler.
 * @param {GraphState} graph live graph (never mutated here)
 * @param {object} drag { nodeId, socket, direction, kind }
 * @param {object} options { mode, context, position, baselineInference }
 * @returns {Array} rows: [{ type, label, socket, applies, errorsAfter, warningsAfter, score, reason }]
 */
export function scoreCandidates(graph, drag, options = {}) {
  const mode = options.mode === 'manual' ? 'manual' : 'assist';
  const context = options.context || {};
  const position = options.position || { x: 0, y: 0 };
  const baseline = options.baselineInference || inferGraph(graph, context);
  const baselineErrors = baseline.errors.length;
  const rows = [];

  for (const candidate of baseCandidates(drag)) {
    const def = getNodeType(candidate.type);
    let score = -1000;
    let applies = false;
    let errorsAfter = baselineErrors;
    let warningsAfter = 0;
    let reason = null;

    try {
      const clone = graph.clone();
      const node = clone.addNode(candidate.type, { x: position.x, y: position.y });
      const refs = candidate.direction === 'in'
        ? { from: { nodeId: drag.nodeId, socket: drag.socket }, to: { nodeId: node.id, socket: candidate.socket } }
        : { from: { nodeId: node.id, socket: candidate.socket }, to: { nodeId: drag.nodeId, socket: drag.socket } };

      const result = mode === 'manual'
        ? clone.connect(refs.from.nodeId, refs.from.socket, refs.to.nodeId, refs.to.socket)
        : assistConnect(clone, refs.from, refs.to, context);

      if (!result.ok) {
        reason = result.error ? `${result.error.code}: ${result.error.message}` : 'The engine refused this connection.';
        score = -1000;
        applies = false;
      } else {
        const inference = inferGraph(clone, context);
        errorsAfter = inference.errors.length;
        warningsAfter = inference.warnings.length;
        const delta = baselineErrors - errorsAfter;
        // Real compiler deltas dominate; warnings break ties. `applies` means the graph compiles.
        score = delta * 100 - warningsAfter;
        applies = errorsAfter === 0;
        if (!applies) {
          if (errorsAfter > baselineErrors) {
            reason = inference.errors.slice(0, 2).map((issue) => `${issue.code}: ${issue.message}`).join(' · ');
          } else if (errorsAfter === baselineErrors && baselineErrors > 0) {
            reason = 'Adds no new errors but does not fix existing ones.';
          } else {
            reason = inference.errors.slice(0, 2).map((issue) => `${issue.code}: ${issue.message}`).join(' · ') || 'The graph does not compile with this block.';
          }
        }
      }
    } catch (err) {
      reason = err.message;
    }

    rows.push({
      ...candidate,
      detail: technicalLine(candidate.type),
      applies,
      score,
      errorsAfter,
      warningsAfter,
      reason
    });
  }

  rows.sort((a, b) => {
    if (a.applies !== b.applies) return a.applies ? -1 : 1;
    if (b.score !== a.score) return b.score - a.score;
    return a.label.localeCompare(b.label);
  });
  return rows;
}

/** Case-insensitive search across labels, summaries and aliases. */
export function filterCandidates(rows, query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return rows.slice();
  return rows.filter((row) => {
    const haystack = `${row.label} ${row.friendly} ${row.summary} ${row.aliases}`.toLowerCase();
    return haystack.includes(q);
  });
}

const INTENT_GROUPS = {
  data: 'Feed it examples',
  layer: 'Make it think',
  activation: 'Turn signals on or off',
  core: 'Score and steer learning',
  logic: 'Training controls',
  output: 'Watch it learn'
};

/**
 * Turn scored rows into the view model consumed by the popover.
 * @param {object} input { drag, rows, policy, query }
 * @param {object} policy from interactive_modes.quickMenuPolicy (tier aware)
 */
export function buildQuickMenuModel({ drag = null, rows = [], policy = {}, query = '' } = {}) {
  const maxItems = Number.isFinite(policy.maxItems) && policy.maxItems > 0 ? policy.maxItems : 0;
  const visible = policy.showBlocked === false ? rows.filter((row) => row.applies) : rows.slice();
  const matched = filterCandidates(visible, query);

  const groupOrder = Array.isArray(policy.groupOrder) ? policy.groupOrder : null;
  const groupLabels = policy.groupLabels || {};
  const tiers = policy.intentGroups ? INTENT_GROUPS : null;

  const buckets = new Map();
  for (const row of matched) {
    const key = tiers ? (tiers[row.category] || 'Other') : row.category;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(row);
  }

  let groups = Array.from(buckets.entries()).map(([key, groupRows]) => ({
    key,
    label: groupLabels[key] || (tiers ? key : (CATEGORY_META[key] ? CATEGORY_META[key].label : key)),
    rows: groupRows
  }));

  if (groupOrder) {
    groups.sort((a, b) => {
      const ia = groupOrder.indexOf(a.key);
      const ib = groupOrder.indexOf(b.key);
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    });
  }
  groups = groups.filter((group) => group.rows.length > 0);

  return {
    title: policy.title || 'Place a block',
    subtitle: drag
      ? `Growing from ${drag.nodeLabel || drag.nodeId} · ${drag.socket} (${drag.kind})`
      : 'Pick a block — it lands at the cursor and is wired instantly.',
    emptyText: 'No compatible block matches that search.',
    groups,
    // `limit` is applied only while the search box is empty, so tier-1 stays curated but the
    // search still reaches every compatible block behind the scenes.
    limit: maxItems,
    count: matched.length,
    friendly: Boolean(policy.friendly),
    showDetail: Boolean(policy.showDetail),
    showReasons: Boolean(policy.showReasons)
  };
}

/* --------------------------------------------------------------- mutation */

/**
 * Actually place the chosen block and wire it to the dragged socket.
 * Beginner/assist mode uses `assistConnect` (rank repairs insert Flatten); Advanced uses strict connect.
 * Rolls the node back if the wire is refused, so a failed pick never leaves a stray block.
 */
export function applyQuickAdd(graph, drag, type, position, options = {}) {
  const mode = options.mode === 'manual' ? 'manual' : 'assist';
  const context = options.context || {};
  const socket = socketForCandidate(type, drag.kind, drag.direction);
  if (!socket) {
    return {
      ok: false,
      error: { code: 'E_NO_SOCKET', message: `No ${drag.kind} socket can accept this wire on ${type}.` }
    };
  }

  const node = graph.addNode(type, {
    x: Math.round(position.x),
    y: Math.round(position.y),
    params: options.params || defaultParams(type)
  });

  const refs = socket.direction === 'in'
    ? { from: { nodeId: drag.nodeId, socket: drag.socket }, to: { nodeId: node.id, socket: socket.name } }
    : { from: { nodeId: node.id, socket: socket.name }, to: { nodeId: drag.nodeId, socket: drag.socket } };

  const result = mode === 'manual'
    ? graph.connect(refs.from.nodeId, refs.from.socket, refs.to.nodeId, refs.to.socket)
    : assistConnect(graph, refs.from, refs.to, context);

  if (!result.ok) {
    graph.removeNode(node.id);
    return { ok: false, error: result.error || { code: 'E_REFUSED', message: 'The engine refused the connection.' } };
  }

  return {
    ok: true,
    node: graph.getNode(node.id),
    connection: result.connection || graph.incoming(refs.to.nodeId, refs.to.socket),
    actions: result.actions || []
  };
}

/* ------------------------------------------------------------ DOM popover */

/**
 * Cursor-anchored searchable block picker. Registered with MenuManager as a popover, so Escape,
 * outside-click and the injected × all close it (the page clears the pending wire on close).
 */
export class QuickNodeMenu {
  constructor(options = {}) {
    this.doc = options.document || (typeof document !== 'undefined' ? document : null);
    this.win = options.window || (typeof window !== 'undefined' ? window : null);
    this.menuManager = options.menuManager || null;
    this.menuId = options.menuId || QUICK_MENU_ID;
    this.onPick = options.onPick || null;
    this.onClose = options.onClose || null;
    this.model = null;
    this.query = '';
    this.buttons = [];
    this.panel = null;
    this.groupsHost = null;
    this.searchInput = null;
    if (this.doc) this._build();
  }

  _build() {
    const panel = this.doc.createElement('div');
    panel.className = 'ns-menu ns-menu-cursor ns-quick-menu';
    panel.hidden = true;

    this.titleEl = this.doc.createElement('div');
    this.titleEl.className = 'ns-quick-title';
    panel.appendChild(this.titleEl);

    this.subtitleEl = this.doc.createElement('div');
    this.subtitleEl.className = 'ns-quick-subtitle';
    panel.appendChild(this.subtitleEl);

    this.searchInput = this.doc.createElement('input');
    this.searchInput.className = 'ns-quick-search';
    this.searchInput.type = 'search';
    this.searchInput.placeholder = 'Search blocks…';
    this.searchInput.setAttribute('aria-label', 'Search blocks');
    this.searchInput.addEventListener('input', () => this.setQuery(this.searchInput.value));
    this.searchInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        this.pickFirst();
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        this.moveActive(event.key === 'ArrowDown' ? 1 : -1);
      }
    });
    panel.appendChild(this.searchInput);

    this.groupsHost = this.doc.createElement('div');
    this.groupsHost.className = 'ns-quick-groups';
    panel.appendChild(this.groupsHost);

    this.emptyEl = this.doc.createElement('div');
    this.emptyEl.className = 'ns-quick-empty';
    panel.appendChild(this.emptyEl);

    this.footerEl = this.doc.createElement('div');
    this.footerEl.className = 'ns-quick-footer';
    this.footerEl.textContent = 'Enter picks the first match · Esc closes · the wire stays connected to the cursor until you choose';
    panel.appendChild(this.footerEl);

    this.panel = panel;
    // MenuManager owns lifecycle (show/hide/Escape), not mounting: attach the panel once so the
    // popover is actually part of the document (JS-created panels are invisible otherwise).
    if (this.doc && this.doc.body && panel.parentNode !== this.doc.body) {
      this.doc.body.appendChild(panel);
    }
    if (this.menuManager) {
      this.menuManager.register({
        id: this.menuId,
        panel,
        kind: 'popover',
        onClose: () => {
          this.query = '';
          if (this.onClose) this.onClose();
        }
      });
    }
  }

  /** Visible rows for the current model + query (also used by the keyboard handlers). */
  visibleRows() {
    if (!this.model) return [];
    const rows = [];
    const limited = !this.query && this.model.limit > 0;
    for (const group of this.model.groups) {
      for (const row of group.rows) {
        if (!limited || rows.length < this.model.limit) rows.push(row);
      }
    }
    return filterCandidates(rows, this.query);
  }

  open(model, position = {}) {
    if (!this.doc || !this.panel) return false;
    this.model = model;
    this.query = '';
    if (this.searchInput) this.searchInput.value = '';
    this.render();
    const margin = 8;
    const maxX = this.win && Number.isFinite(this.win.innerWidth) ? this.win.innerWidth : 1280;
    const maxY = this.win && Number.isFinite(this.win.innerHeight) ? this.win.innerHeight : 800;
    const width = 340;
    const height = 420;
    const x = Math.max(margin, Math.min(Number(position.clientX) || 0, Math.max(margin, maxX - width)));
    const y = Math.max(margin, Math.min(Number(position.clientY) || 0, Math.max(margin, maxY - height)));
    this.panel.style.position = 'fixed';
    this.panel.style.left = `${x}px`;
    this.panel.style.top = `${y}px`;
    this.panel.style.right = 'auto';
    if (this.menuManager) this.menuManager.open(this.menuId);
    else this.panel.hidden = false;
    if (this.searchInput && this.searchInput.focus) this.searchInput.focus();
    return true;
  }

  setQuery(query) {
    this.query = String(query || '');
    this.render();
  }

  render() {
    if (!this.doc || !this.groupsHost) return;
    this.titleEl.textContent = this.model ? this.model.title : 'Place a block';
    this.subtitleEl.textContent = this.model ? this.model.subtitle : '';
    this.groupsHost.innerHTML = '';
    this.buttons = [];

    const active = this.model ? this.visibleRows() : [];
    if (active.length === 0) {
      this.emptyEl.textContent = this.model ? this.model.emptyText : '';
      this.emptyEl.hidden = false;
      return;
    }
    this.emptyEl.hidden = true;

    const limit = !this.query && this.model.limit > 0 ? this.model.limit : 0;
    for (const group of this.model.groups) {
      if (limit > 0 && this.buttons.length >= limit) break;
      const rows = filterCandidates(group.rows, this.query);
      if (rows.length === 0) continue;
      const groupEl = this.doc.createElement('div');
      groupEl.className = 'ns-quick-group';
      const head = this.doc.createElement('div');
      head.className = 'ns-quick-group-head';
      head.textContent = group.label;
      groupEl.appendChild(head);

      for (const row of rows) {
        if (limit > 0 && this.buttons.length >= limit) break;
        const blocked = Boolean(this.model.showReasons && !row.applies && row.reason);
        const button = this.doc.createElement('button');
        button.type = 'button';
        button.className = `ns-quick-item${row.applies ? ' is-clean' : ''}${blocked ? ' is-blocked' : ''}`;
        button.dataset.quickType = row.type;

        const name = this.doc.createElement('span');
        name.className = 'ns-quick-name';
        name.textContent = this.model && this.model.friendly ? row.friendly : row.label;
        button.appendChild(name);

        const detail = this.doc.createElement('span');
        detail.className = 'ns-quick-detail';
        const showReason = this.model && this.model.showReasons && !row.applies && row.reason;
        detail.textContent = showReason
          ? row.reason
          : (this.model && this.model.showDetail ? row.detail : row.summary);
        button.appendChild(detail);

        const socketTag = this.doc.createElement('span');
        socketTag.className = 'ns-quick-socket';
        socketTag.textContent = `${row.socket}:${row.kind}`;
        button.appendChild(socketTag);

        button.addEventListener('click', () => this._pick(row));
        groupEl.appendChild(button);
        this.buttons.push({ row, button });
      }
      this.groupsHost.appendChild(groupEl);
    }
    this.setActive(0);
  }

  setActive(index) {
    if (this.buttons.length === 0) return;
    const bounded = Math.max(0, Math.min(index, this.buttons.length - 1));
    this.activeIndex = bounded;
    this.buttons.forEach((entry, i) => entry.button.classList.toggle('is-active', i === bounded));
  }

  moveActive(delta) {
    if (this.buttons.length === 0) return;
    const next = ((this.activeIndex || 0) + delta + this.buttons.length) % this.buttons.length;
    this.setActive(next);
  }

  pickFirst() {
    if (this.buttons.length === 0) return false;
    this._pick(this.buttons[Math.max(0, this.activeIndex || 0)].row);
    return true;
  }

  _pick(row) {
    // Deliver the pick first: the page consumes the pending wire inside onPick, then close()
    // runs the MenuManager close path (whose onClose clears anything that remains).
    try {
      if (this.onPick) this.onPick(row);
    } finally {
      this.close();
    }
  }

  close() {
    if (this.menuManager) this.menuManager.close(this.menuId, { reason: 'quick-menu' });
    else if (this.panel) this.panel.hidden = true;
  }

  get active() {
    return Boolean(this.panel && !this.panel.hidden);
  }

  destroy() {
    if (this.menuManager) this.menuManager.unregister(this.menuId);
    if (this.panel && this.panel.parentNode) this.panel.parentNode.removeChild(this.panel);
    this.panel = null;
    this.buttons = [];
  }
}

/** Explanation snippet for a candidate row (mission cards reuse this). */
export function candidateExplanation(type) {
  const entry = explain(type);
  return { plain: entry.plain, analogy: entry.analogy, technical: entry.technical };
}
