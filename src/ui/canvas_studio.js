/**
 * Node Studio — interactive canvas application.
 *
 * The full modular editor: palette → draggable blocks → wired graph → live validation →
 * compiled plan → real training through the shared engine worker → live metrics, confusion
 * matrix, prediction probe → exportable artifacts.
 *
 * Everything the page shows is produced by the verified modules underneath:
 *   graph/node_types.js (registry) · graph/graph_state.js (wiring) · graph/node_engine.js
 *   (shape inference + compile) · graph/smart_assist.js (Beginner repairs) ·
 *   graph/data_connectors.js (built-in/CSV/SQL/REST sources) · export/builder.js (artifacts).
 *
 * File: src/ui/canvas_studio.js
 */

import { CATEGORY_META, DATA_SOURCES, NODE_TYPES, SOURCE_NODE_TYPES, TAP_NODE_TYPES, OPTIMIZER_NODE_TYPES, defaultParams, getNodeType } from '../graph/node_types.js';
import { GraphState } from '../graph/graph_state.js';
import { inferGraph, compile, applyClassFilter, schedulerValue } from '../graph/node_engine.js';
import { assistConnect, autoWireOnDrop, autoFixGraphToConvergence, instantPreset, suggestConnection, snapTarget } from '../modes/smart_assist.js';
import { nodeSize, socketsOf, socketPosition, bezierPath, pointOnBezier, screenToWorld, zoomViewAt, hitTestSocket } from '../graph/geometry.js';
import { resolveSourceDataset, ensureSplit } from '../graph/data_connectors.js';
import { DatasetManager } from '../engine/dataset_manager.js';
import { buildExportArtifacts, buildCommands, downloadBytes, downloadText } from '../export/builder.js';
import { TutorialOverlay, TOURS } from './tutorial_engine.js';
import { MenuManager } from './menu_manager.js';
import { ActionBar } from './action_bar.js';
import { NodePalette } from '../canvas/node_palette.js';
import { collectWires, WireInteractions, wireMenuBaseModel } from '../canvas/wire_interactions.js';
import { QuickNodeMenu, applyQuickAdd } from '../canvas/quick_node_menu.js';
import { StepByStepEngine } from '../tutorial/step_by_step_engine.js';
import { activeTier, decorateWireMenu, buildQuickMenu, tierContext } from '../modes/interactive_modes.js';
import { AbsoluteBeginnerPlayground } from '../modes/absolute_beginner.js';
import { CurriculumTour } from '../tutorial/interactive_tour.js';

const MNIST_BASE_PATH = 'data/mnist_01';
const CACHEABLE_SOURCE_TYPES = ['source', 'sql_source', 'api_source'];

/* ------------------------------------------------------------------ state */

const state = {
  graph: null,
  mode: 'assist',
  presetId: null,
  selectedId: null,
  suggestion: null,
  view: { x: 60, y: 60, scale: 1 },
  sourceInfo: new Map(),
  sourceData: new Map(),
  featuresMeta: new Map(),
  dataCache: new Map(),
  customDatasets: {},
  inference: null,
  plan: null,
  weights: null,
  running: false,
  builtSignature: null,
  epochsRun: 0,
  lossHistory: [],
  epochDelayMs: 0,
  particlesOn: true,
  lastScheduledLr: null,
  undoStack: [],
  confusion: null,
  /* Structured action journal. Transient UI state (e.g. `running`) can flip back to false the
     moment a fast run completes, so lessons verify *actions that really happened* here. */
  actions: [],
  actionSeq: 0
};

let worker = null;
let dom = {};
let tutorial = null;
let curriculum = null;
let menus = null;
let actionBar = null;
let palette = null;
let playground = null;
let playgroundPending = null;
let wireDrag = null;
let pendingDrop = null;
let nodeDrag = null;
let panDrag = null;
let validateTimer = null;
let particles = [];
let padState = null;
let wireInteractions = null;
let quickMenu = null;
let missions = null;

/* ------------------------------------------------------------- utilities */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(2)}%`;
}

function basename(filePath) {
  return String(filePath || '').split(/[\\/]/).pop();
}

function compileContext() {
  return { datasetInfoFor: (node) => state.sourceInfo.get(node.id) || null };
}

function appendLog(level, message) {
  if (!dom.logList) return;
  const item = el('li', `ns-log-${level}`);
  const time = new Date().toLocaleTimeString();
  item.innerHTML = `<span class="ns-log-time">${time}</span> <span class="ns-log-msg"></span>`;
  item.querySelector('.ns-log-msg').textContent = message;
  dom.logList.prepend(item);
  while (dom.logList.children.length > 120) dom.logList.removeChild(dom.logList.lastChild);
}

/**
 * Record one real user action in two places at once:
 *   1. the structured journal (`state.actions`) that lesson gates can inspect, and
 *   2. the live guide runners (tour + mission), so the current step can unlock immediately.
 *
 * Gates must not depend on a momentary flag such as `state.running`: a small XOR net can finish
 * before the learner clicks Next, and the lesson would otherwise ask them to press Run again.
 */
function recordAction(actionId, detail = null) {
  if (!actionId) return;
  state.actionSeq += 1;
  state.actions.push({
    id: actionId,
    seq: state.actionSeq,
    at: Date.now(),
    detail: detail || null
  });
  if (state.actions.length > 300) state.actions.shift();
  if (tutorial) tutorial.notify(actionId);
  if (missions) missions.notify(actionId, missionContext());
}

function setStatus(text) {
  if (dom.statusText) dom.statusText.textContent = text;
}

function snapshot() {
  return JSON.stringify({ graph: state.graph.toJSON(), selectedId: state.selectedId });
}

function pushUndo(before = null) {
  const entry = before || snapshot();
  state.undoStack.push(entry);
  if (state.undoStack.length > 60) state.undoStack.shift();
}

function undo() {
  const entry = state.undoStack.pop();
  if (!entry) {
    appendLog('info', 'Nothing to undo.');
    return;
  }
  const parsed = JSON.parse(entry);
  state.graph = GraphState.fromJSON(parsed.graph);
  state.selectedId = parsed.selectedId;
  afterGraphMutation({ skipUndo: true, resolve: true });
  appendLog('action', 'Undo applied.');
}

function selectedNode() {
  return state.selectedId ? state.graph.getNode(state.selectedId) : null;
}

/* ------------------------------------------------------------- palette */

function renderPalette() {
  if (palette) palette.render();
}

function paletteAddAtCenter(type) {
  const rect = dom.canvas.getBoundingClientRect();
  const center = screenToWorld({ x: rect.width / 2, y: rect.height / 3 }, state.view);
  addNodeAt(type, { x: center.x - 90, y: center.y - 40 });
}

/* ---------------------------------------------------------- graph edits */

function addNodeAt(type, worldPos) {
  const before = snapshot();
  const node = state.graph.addNode(type, { x: Math.round(worldPos.x), y: Math.round(worldPos.y) });
  pushUndo(before);
  state.selectedId = node.id;
  state.presetId = null;
  if (state.mode === 'assist' && !(missions && missions.suppressAssist)) {
    const result = autoWireOnDrop(state.graph, node.id, compileContext());
    for (const action of result.actions || []) appendLog('action', action.message || action.code);
  }
  afterGraphMutation({ resolve: false });
  appendLog('action', `Added ${getNodeType(type).label}.`);
  recordAction('node_added', { type, label: getNodeType(type).label });
  if (type === 'sql_source') recordAction('source_added_sql', { type });
}

function removeSelected() {
  const node = selectedNode();
  if (!node) return;
  deleteNode(node.id, { viaTrash: false });
}

/**
 * Remove a block and every piece of state that belongs to it. Wires are severed inside
 * GraphState; per-node data caches and the shared data cache are released here so repeated
 * add/delete cycles cannot grow the app's memory.
 */
function deleteNode(nodeId, { viaTrash = false } = {}) {
  const node = state.graph.getNode(nodeId);
  if (!node) return false;
  const before = snapshot();
  const linkCount = state.graph.connections.size;
  const isSource = SOURCE_NODE_TYPES.includes(node.type);
  const key = isSource ? cacheKeyFor(node) : null;

  state.graph.removeNode(nodeId);
  const severed = linkCount - state.graph.connections.size;

  state.sourceInfo.delete(nodeId);
  state.sourceData.delete(nodeId);
  state.featuresMeta.delete(nodeId);
  if (isSource && key) {
    const stillUsed = state.graph.nodeList().some((other) => SOURCE_NODE_TYPES.includes(other.type) && cacheKeyFor(other) === key);
    if (!stillUsed) state.dataCache.delete(key);
  }

  pushUndo(before);
  if (state.selectedId === nodeId) state.selectedId = null;
  state.presetId = null;
  state.suggestion = null;
  afterGraphMutation({ resolve: false });
  appendLog('action', viaTrash
    ? `Dropped ${getNodeType(node.type).label} in the trash — ${severed} wire(s) severed.`
    : `Removed ${getNodeType(node.type).label}.`);
  recordAction('node_removed', { type: node.type, label: getNodeType(node.type).label, viaTrash, severed });
  return true;
}

function attemptConnect(fromRef, toRef) {
  const fromNode = state.graph.getNode(fromRef.nodeId);
  const toNode = state.graph.getNode(toRef.nodeId);
  const fromLabel = `${getNodeType(fromNode.type).label}.${fromRef.socket}`;
  const toLabel = `${getNodeType(toNode.type).label}.${toRef.socket}`;

  if (state.mode === 'assist') {
    const before = snapshot();
    const result = assistConnect(state.graph, fromRef, toRef, compileContext());
    if (result.ok) {
      pushUndo(before);
      state.presetId = null;
      for (const action of result.actions || []) appendLog('action', action.message || action.code);
      afterGraphMutation({ resolve: true });
      setStatus(`Connected ${fromLabel} → ${toLabel}.`);
      recordAction('node_connected', { from: fromLabel, to: toLabel });
    } else {
      appendLog('warn', result.error ? `${result.error.code}: ${result.error.message}` : 'Connection refused.');
    }
    return;
  }

  const before = snapshot();
  const result = state.graph.connect(fromRef.nodeId, fromRef.socket, toRef.nodeId, toRef.socket);
  if (result.ok) {
    pushUndo(before);
    state.presetId = null;
    afterGraphMutation({ resolve: true });
    setStatus(`Connected ${fromLabel} → ${toLabel}.`);
    recordAction('node_connected', { from: fromLabel, to: toLabel });
  } else {
    appendLog('warn', result.error ? `${result.error.code}: ${result.error.message}` : 'Connection refused.');
  }
}

function applySuggestion() {
  if (!state.suggestion) return;
  const suggestion = state.suggestion;
  attemptConnect(suggestion.from, suggestion.to);
}

/* ------------------------------------- interactive wires, quick-add, missions */

function missionContext() {
  return {
    graph: state.graph,
    mode: state.mode,
    running: state.running,
    actions: state.actions,
    plan: state.plan,
    inference: state.inference
  };
}

/**
 * Keep the guide spotlight / ring / card on the live geometry.
 *
 * `renderNodes()` replaces every node element, so a cached target becomes a detached node and its
 * old screen rect goes stale the moment a learner drags a block. Pan/zoom and inspector rebuilds
 * move targets too. Both presenters expose `refreshPosition()` and re-resolve their target there.
 */
function refreshGuideOverlays() {
  if (missions && typeof missions.refreshPosition === 'function') missions.refreshPosition();
  if (tutorial && typeof tutorial.refreshPosition === 'function') tutorial.refreshPosition();
}

/** Real wire break: one undo entry, graph mutation, re-inference, mission notification. */
function breakWire(connectionId) {
  const connection = state.graph.connections.get(connectionId);
  if (!connection) return false;
  const fromNode = state.graph.getNode(connection.from.nodeId);
  const toNode = state.graph.getNode(connection.to.nodeId);
  const fromLabel = fromNode ? `${getNodeType(fromNode.type).label}.${connection.from.socket}` : connection.from.nodeId;
  const toLabel = toNode ? `${getNodeType(toNode.type).label}.${connection.to.socket}` : connection.to.nodeId;
  const before = snapshot();
  state.graph.disconnect(connectionId);
  pushUndo(before);
  state.presetId = null;
  afterGraphMutation({ resolve: false });
  appendLog('action', `Broke ${fromLabel} → ${toLabel}.`);
  recordAction('wire_broken', { from: fromLabel, to: toLabel, connectionId });
  setStatus(`Disconnected ${fromLabel} → ${toLabel}.`);
  return true;
}

/** Open the searchable quick-add menu for a wire released over empty canvas. */
function openQuickMenuAt(drag, client, world) {
  if (!quickMenu) return false;
  const node = state.graph.getNode(drag.nodeId);
  if (!node) return false;
  const def = getNodeType(node.type);
  const built = buildQuickMenu({
    graph: state.graph,
    drag: {
      nodeId: drag.nodeId,
      socket: drag.socket,
      direction: drag.direction,
      kind: drag.kind,
      nodeLabel: def.label
    },
    position: world,
    ctx: {
      mode: state.mode,
      missionTier: missions && missions.active ? missions.tier : 0,
      context: compileContext()
    }
  });
  pendingDrop = {
    nodeId: drag.nodeId,
    socket: drag.socket,
    direction: drag.direction,
    kind: drag.kind,
    start: { ...drag.start },
    cursor: { ...world }
  };
  renderWires();
  quickMenu.open(built.model, { clientX: client.clientX, clientY: client.clientY });
  if (built.rows.length === 0) appendLog('warn', 'No compatible block can accept this wire.');
  setStatus(`Choose a block for the ${drag.kind} wire — it will be placed and wired at the cursor.`);
  recordAction('quick_menu_opened', { node: def.label, socket: drag.socket, kind: drag.kind });
  return true;
}

/** Place the chosen block at the release point and wire it in the same gesture. */
function applyQuickPick(row) {
  const drop = pendingDrop;
  pendingDrop = null;
  if (!drop || !row) {
    renderWires();
    return;
  }
  const before = snapshot();
  const result = applyQuickAdd(state.graph, {
    nodeId: drop.nodeId,
    socket: drop.socket,
    direction: drop.direction,
    kind: drop.kind
  }, row.type, drop.cursor, { mode: state.mode, context: compileContext() });

  if (!result.ok) {
    appendLog('warn', `${result.error.code}: ${result.error.message}`);
    renderWires();
    return;
  }

  pushUndo(before);
  state.presetId = null;
  state.selectedId = result.node.id;
  for (const action of result.actions || []) appendLog('action', action.message || action.code);
  afterGraphMutation({ resolve: true });
  appendLog('action', `Placed ${getNodeType(row.type).label} and wired it in one gesture.`);
  recordAction('quick_node_added', { type: row.type, label: getNodeType(row.type).label });
  recordAction('node_added', { type: row.type, label: getNodeType(row.type).label, via: 'quick-add' });
  setStatus(`Placed ${getNodeType(row.type).label}.`);
}

/** Page command executed on mission step entry. */
function handleMissionSetup(setup, step, engine) {
  if (!setup) return;
  if (setup.id === 'prepare_mission') {
    if (setup.mode && setup.mode !== state.mode) setMode(setup.mode);
    const before = snapshot();
    state.graph = new GraphState(state.mode === 'manual' ? 'manual' : 'assist');
    pushUndo(before);
    state.selectedId = null;
    state.presetId = null;
    state.weights = null;
    state.builtSignature = null;
    state.lossHistory = [];
    state.epochsRun = 0;
    state.confusion = null;
    state.suggestion = null;
    renderSparkline();
    renderConfusion();
    afterGraphMutation({ resolve: false });
    appendLog('info', 'Mission canvas cleared — build the brain block by block.');
  }
}

/** Resolve a mission target: CSS selector, or { nodeType, socket, direction, pick:'newest' }. */
function resolveMissionTarget(target) {
  if (!target) return null;
  if (typeof target === 'string') return document.querySelector(target);
  if (target.nodeType) {
    const candidates = state.graph.nodeList().filter((node) => {
      if (node.type !== target.nodeType) return false;
      if (!target.socket) return true;
      return socketsOf(node, getNodeType(node.type)).some((port) => port.name === target.socket);
    });
    if (candidates.length === 0) return null;
    const node = target.pick === 'newest' ? candidates[candidates.length - 1] : candidates[0];
    const nodeEl = dom.nodeLayer.querySelector(`.ns-node[data-id="${node.id}"]`);
    if (!nodeEl) return null;
    if (!target.socket) return nodeEl;
    return nodeEl.querySelector(`.ns-socket[data-socket="${target.socket}"][data-dir="${target.direction || 'out'}"]`);
  }
  return null;
}

function afterGraphMutation({ skipUndo = false, resolve = false } = {}) {
  renderNodes();
  renderWires();
  renderInspector();
  validateNow();
  updateProbePanel();
  if (resolve) resolveAllSources().catch((err) => appendLog('error', err.message));
}

/* -------------------------------------------------------------- canvas */

function canvasPoint(event) {
  const rect = dom.canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function allPorts() {
  const ports = [];
  for (const node of state.graph.nodeList()) {
    const def = getNodeType(node.type);
    if (!def) continue;
    for (const port of socketsOf(node, def)) {
      ports.push({ ...port, nodeId: node.id });
    }
  }
  return ports;
}

function markSnapSocket(port) {
  for (const socketEl of dom.nodeLayer.querySelectorAll('.ns-socket.is-snap')) {
    socketEl.classList.remove('is-snap');
  }
  if (!port) return;
  const socketEl = dom.nodeLayer.querySelector(`.ns-socket[data-node="${port.nodeId}"][data-socket="${port.name}"][data-dir="${port.direction}"]`);
  if (socketEl) socketEl.classList.add('is-snap');
}

function clearTrashTarget() {
  if (palette) palette.setTrashTarget(false);
  for (const nodeEl of dom.nodeLayer.querySelectorAll('.ns-node.is-trash-armed')) {
    nodeEl.classList.remove('is-trash-armed');
  }
}

function bindCanvasEvents() {
  const capture = (event) => {
    try {
      dom.canvas.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic/test pointers and some embedded webviews cannot be captured; the drag state
      // below still tracks the gesture, so the interaction degrades gracefully.
    }
  };

  dom.canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    const socketEl = event.target.closest ? event.target.closest('.ns-socket') : null;
    const nodeEl = event.target.closest ? event.target.closest('.ns-node') : null;

    if (socketEl) {
      const nodeId = socketEl.dataset.node;
      const socketName = socketEl.dataset.socket;
      const direction = socketEl.dataset.dir;
      const node = state.graph.getNode(nodeId);
      const def = getNodeType(node.type);
      const start = socketPosition(node, def, socketName, direction);
      const kind = socketEl.dataset.kind;
      wireDrag = { nodeId, socket: socketName, direction, kind, start, cursor: { ...start }, pressX: event.clientX, pressY: event.clientY, moved: false };
      renderWires();
      capture(event);
      return;
    }

    if (nodeEl) {
      const nodeId = nodeEl.dataset.id;
      const node = state.graph.getNode(nodeId);
      state.selectedId = nodeId;
      renderInspector();
      renderNodes();
      nodeDrag = { nodeId, startX: event.clientX, startY: event.clientY, nodeX: node.x, nodeY: node.y, moved: false };
      capture(event);
      return;
    }

    // Background: pan the view and clear selection.
    state.selectedId = null;
    renderInspector();
    renderNodes();
    panDrag = { startX: event.clientX, startY: event.clientY, viewX: state.view.x, viewY: state.view.y };
    capture(event);
  });

  dom.canvas.addEventListener('pointermove', (event) => {
    if (wireDrag) {
      if (Math.abs(event.clientX - wireDrag.pressX) > 6 || Math.abs(event.clientY - wireDrag.pressY) > 6) wireDrag.moved = true;
      wireDrag.cursor = screenToWorld(canvasPoint(event), state.view);
      if (state.mode === 'assist') {
        markSnapSocket(snapTarget(allPorts(), wireDrag.cursor, wireDrag, { mode: 'assist' }));
      }
      dom.canvas.classList.add('is-wire-drop');
      renderWires();
      return;
    }
    if (nodeDrag) {
      const node = state.graph.getNode(nodeDrag.nodeId);
      if (!node) return;
      const dx = (event.clientX - nodeDrag.startX) / state.view.scale;
      const dy = (event.clientY - nodeDrag.startY) / state.view.scale;
      node.x = Math.round(nodeDrag.nodeX + dx);
      node.y = Math.round(nodeDrag.nodeY + dy);
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) nodeDrag.moved = true;
      renderNodes();
      renderWires();
      const overTrash = nodeDrag.moved && palette && palette.isOver(event.clientX, event.clientY);
      if (palette) palette.setTrashTarget(Boolean(overTrash));
      const dragged = dom.nodeLayer.querySelector(`.ns-node[data-id="${nodeDrag.nodeId}"]`);
      if (dragged) dragged.classList.toggle('is-trash-armed', Boolean(overTrash));
      return;
    }
    if (panDrag) {
      state.view.x = panDrag.viewX + (event.clientX - panDrag.startX) / state.view.scale;
      state.view.y = panDrag.viewY + (event.clientY - panDrag.startY) / state.view.scale;
      applyView();
    }
  });

  dom.canvas.addEventListener('pointerup', (event) => {
    if (wireDrag) {
      const world = screenToWorld(canvasPoint(event), state.view);
      const wantedDirection = wireDrag.direction === 'out' ? 'in' : 'out';
      const snapped = state.mode === 'assist'
        ? snapTarget(allPorts(), world, wireDrag, { mode: 'assist' })
        : null;
      const target = snapped || hitTestSocket(allPorts(), world, 18, { kind: wireDrag.kind, direction: wantedDirection });
      const drag = wireDrag;
      wireDrag = null;
      markSnapSocket(null);
      dom.canvas.classList.remove('is-wire-drop');
      if (target && target.nodeId !== drag.nodeId) {
        renderWires();
        if (drag.direction === 'out') {
          attemptConnect({ nodeId: drag.nodeId, socket: drag.socket }, { nodeId: target.nodeId, socket: target.name });
        } else {
          attemptConnect({ nodeId: target.nodeId, socket: target.name }, { nodeId: drag.nodeId, socket: drag.socket });
        }
      } else {
        // Released over empty canvas (or back on its own node): grow the graph from the wire.
        openQuickMenuAt(drag, { clientX: event.clientX, clientY: event.clientY }, world);
      }
    }
    if (nodeDrag) {
      const dropToTrash = nodeDrag.moved && palette && palette.isOver(event.clientX, event.clientY);
      const draggedId = nodeDrag.nodeId;
      nodeDrag = null;
      if (dropToTrash) {
        clearTrashTarget();
        deleteNode(draggedId, { viaTrash: true });
      } else {
        clearTrashTarget();
        renderWires();
      }
    }
    panDrag = null;
  });

  dom.canvas.addEventListener('pointercancel', () => {
    wireDrag = null;
    nodeDrag = null;
    panDrag = null;
    clearTrashTarget();
    markSnapSocket(null);
    dom.canvas.classList.remove('is-wire-drop');
    renderWires();
  });

  dom.canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
    state.view = zoomViewAt(state.view, canvasPoint(event), factor);
    applyView();
  }, { passive: false });

  dom.canvas.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });

  dom.canvas.addEventListener('drop', (event) => {
    event.preventDefault();
    const type = event.dataTransfer.getData('application/x-node-type');
    if (!type || !NODE_TYPES[type]) return;
    const world = screenToWorld(canvasPoint(event), state.view);
    addNodeAt(type, { x: world.x - 90, y: world.y - 30 });
  });

  dom.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
}

function applyView() {
  dom.graphLayer.style.transform = `translate(${state.view.x * state.view.scale}px, ${state.view.y * state.view.scale}px) scale(${state.view.scale})`;
  refreshGuideOverlays();
}

function fitGraphView() {
  const nodes = state.graph.nodeList();
  if (nodes.length === 0) return;
  let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
  for (const node of nodes) {
    const size = nodeSize(getNodeType(node.type));
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x + size.width);
    maxY = Math.max(maxY, node.y + size.height);
  }
  const rect = dom.canvas.getBoundingClientRect();
  const scale = Math.min(1.1, Math.max(0.4, Math.min((rect.width - 120) / (maxX - minX), (rect.height - 140) / (maxY - minY))));
  state.view = {
    scale,
    x: (rect.width / scale - (maxX - minX)) / 2 - minX,
    y: (rect.height / scale - (maxY - minY)) / 2 - minY
  };
  applyView();
}

/* ------------------------------------------------------------- rendering */

function errorCodesByNode() {
  const map = new Map();
  const source = state.plan && !state.plan.ok ? state.plan.errors : (state.inference ? state.inference.errors : []);
  for (const issue of source || []) {
    if (!issue.nodeId) continue;
    if (!map.has(issue.nodeId)) map.set(issue.nodeId, []);
    map.get(issue.nodeId).push(issue);
  }
  return map;
}

function summarizeParams(node) {
  const def = getNodeType(node.type);
  const params = node.params || {};
  switch (node.type) {
    case 'source': {
      const entry = DATA_SOURCES.find((d) => d.id === params.datasetId);
      return [entry ? entry.label : params.datasetId];
    }
    case 'sql_source':
      return [params.db ? `${basename(params.db)}` : 'no file yet', params.query ? params.query.slice(0, 40) : 'no query'];
    case 'api_source':
      return [params.url || 'no URL'];
    case 'dense':
      return [`${params.size} neurons`];
    case 'class_filter':
      return [`keep ${params.keep}`];
    case 'lr_schedule':
      return [params.mode === 'step' ? `×${params.factor} / ${params.every} ep` : 'constant'];
    case 'epoch_loop':
      return [params.mode === 'auto_stop' ? `auto-stop · patience ${params.patience}` : `fixed ${params.maxEpochs} ep`, `batch ${params.batchSize}`];
    case 'adam_optimizer':
    case 'sgd_optimizer':
    case 'momentum_optimizer':
    case 'rmsprop_optimizer':
      return [`lr ${params.learningRate}`, `wd ${params.weightDecay}`];
    default:
      return def.summary && def.summary.length < 60 ? [def.summary] : [];
  }
}

function renderNodes() {
  dom.nodeLayer.innerHTML = '';
  if (dom.studioCanvas) {
    dom.studioCanvas.classList.toggle('is-empty', state.graph.nodeList().length === 0);
  }
  const issues = errorCodesByNode();

  for (const node of state.graph.nodeList()) {
    const def = getNodeType(node.type);
    if (!def) continue;
    const size = nodeSize(def);
    const nodeEl = el('div', 'ns-node');
    nodeEl.dataset.id = node.id;
    nodeEl.dataset.type = node.type;
    nodeEl.style.left = `${node.x}px`;
    nodeEl.style.top = `${node.y}px`;
    nodeEl.style.width = `${size.width}px`;
    nodeEl.style.height = `${size.height}px`;
    if (state.selectedId === node.id) nodeEl.classList.add('is-selected');
    if (issues.has(node.id)) nodeEl.classList.add('has-error');

    const header = el('div', 'ns-node-header');
    header.style.background = CATEGORY_META[def.category].color;
    header.appendChild(el('span', 'ns-node-title', def.label));
    if (issues.has(node.id)) {
      const badge = el('span', 'ns-node-error', '⚠');
      badge.title = issues.get(node.id).map((e) => `${e.code}: ${e.message}`).join('\n');
      header.appendChild(badge);
    }
    nodeEl.appendChild(header);

    const body = el('div', 'ns-node-body');
    for (const line of summarizeParams(node)) {
      body.appendChild(el('div', 'ns-node-line', line));
    }
    nodeEl.appendChild(body);

    const ports = socketsOf(node, def);
    for (const port of ports) {
      const socket = el('div', `ns-socket ns-socket-${port.direction} ns-socket-kind-${port.kind}`);
      socket.dataset.node = node.id;
      socket.dataset.socket = port.name;
      socket.dataset.dir = port.direction;
      socket.dataset.kind = port.kind;
      socket.style.left = `${port.x - node.x - 6}px`;
      socket.style.top = `${port.y - node.y - 6}px`;
      socket.title = `${port.name} (${port.kind})`;
      nodeEl.appendChild(socket);
    }

    dom.nodeLayer.appendChild(nodeEl);
  }

  if (state.suggestion) {
    const target = state.suggestion.to;
    const socketEl = dom.nodeLayer.querySelector(`.ns-socket[data-node="${target.nodeId}"][data-socket="${target.socket}"][data-dir="in"]`);
    if (socketEl) socketEl.classList.add('is-suggested');
  }
  refreshGuideOverlays();
  requestAnimationFrame(positionParticles);
}

function renderWires() {
  dom.wireLayer.innerHTML = '';
  particles = [];

  for (const connection of state.graph.connections.values()) {
    const fromNode = state.graph.getNode(connection.from.nodeId);
    const toNode = state.graph.getNode(connection.to.nodeId);
    if (!fromNode || !toNode) continue;
    const fromDef = getNodeType(fromNode.type);
    const toDef = getNodeType(toNode.type);
    const fromPos = socketPosition(fromNode, fromDef, connection.from.socket, 'out');
    const toPos = socketPosition(toNode, toDef, connection.to.socket, 'in');
    const socketSpec = (fromDef.outputs || []).find((s) => s.name === connection.from.socket);
    const kind = socketSpec ? socketSpec.kind : 'tensor';

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezierPath(fromPos, toPos));
    path.setAttribute('class', `ns-wire ns-wire-${kind}`);
    path.dataset.id = connection.id;
    dom.wireLayer.appendChild(path);

    const particle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    particle.setAttribute('r', kind === 'tensor' ? '3' : '2.2');
    particle.setAttribute('class', `ns-particle ns-particle-${kind}`);
    dom.wireLayer.appendChild(particle);
    particles.push({ circle: particle, from: fromPos, to: toPos, offset: particles.length * 0.13 });
  }

  if (wireDrag) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezierPath(wireDrag.start, wireDrag.cursor));
    path.setAttribute('class', 'ns-wire ns-wire-temp');
    dom.wireLayer.appendChild(path);
  }

  if (pendingDrop) {
    const from = pendingDrop.direction === 'out' ? pendingDrop.start : pendingDrop.cursor;
    const to = pendingDrop.direction === 'out' ? pendingDrop.cursor : pendingDrop.start;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezierPath(from, to));
    path.setAttribute('class', 'ns-wire ns-wire-temp ns-wire-pending');
    dom.wireLayer.appendChild(path);
  }

  if (state.suggestion && !wireDrag) {
    const fromNode = state.graph.getNode(state.suggestion.from.nodeId);
    const toNode = state.graph.getNode(state.suggestion.to.nodeId);
    if (fromNode && toNode) {
      const fromDef = getNodeType(fromNode.type);
      const toDef = getNodeType(toNode.type);
      const fromPos = socketPosition(fromNode, fromDef, state.suggestion.from.socket, 'out');
      const toPos = socketPosition(toNode, toDef, state.suggestion.to.socket, 'in');
      const ghost = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      ghost.setAttribute('d', bezierPath(fromPos, toPos));
      ghost.setAttribute('class', 'ns-wire ns-wire-ghost');
      dom.wireLayer.appendChild(ghost);
    }
  }
  requestAnimationFrame(positionParticles);
}

function positionParticles() {
  const t = performance.now() / 2400;
  for (const particle of particles) {
    if (!state.particlesOn) {
      particle.circle.style.opacity = '0';
      continue;
    }
    const progress = (t + particle.offset) % 1;
    const point = pointOnBezier(particle.from, particle.to, progress);
    particle.circle.setAttribute('cx', point.x);
    particle.circle.setAttribute('cy', point.y);
    particle.circle.style.opacity = state.running ? '0.95' : '0.45';
  }
}

/* ------------------------------------------------------------ inspector */

function renderInspector() {
  const node = selectedNode();
  dom.inspectorBody.innerHTML = '';
  if (!node) {
    dom.inspectorBody.appendChild(el('div', 'ns-muted', 'Select a block on the canvas to edit it.'));
    refreshGuideOverlays();
    return;
  }
  const def = getNodeType(node.type);
  const head = el('div', 'ns-inspector-head');
  const dot = el('span', 'ns-palette-dot');
  dot.style.background = CATEGORY_META[def.category].color;
  head.appendChild(dot);
  head.appendChild(el('strong', '', def.label));
  head.appendChild(el('span', 'ns-muted ns-inspector-id', node.id));
  dom.inspectorBody.appendChild(head);
  dom.inspectorBody.appendChild(el('div', 'ns-inspector-summary', def.summary || ''));

  for (const [name, spec] of Object.entries(def.params || {})) {
    dom.inspectorBody.appendChild(inspectorField(node, name, spec));
  }

  if (node.type === 'source' && node.params.datasetId === 'custom_csv') {
    dom.inspectorBody.appendChild(customCsvControls(node));
  }
  if (node.type === 'sql_source' || node.type === 'api_source') {
    const testBtn = el('button', 'ns-btn ns-btn-small', 'Test data fetch');
    testBtn.addEventListener('click', async () => {
      try {
        setStatus('Fetching data…');
        await resolveNodeData(node, { force: true });
        const dataset = state.sourceData.get(node.id);
        appendLog('ok', `Source OK: ${dataset.X.length} rows · ${dataset.featureNames.length} features · ${dataset.targetNames.length} targets.`);
      } catch (err) {
        appendLog('error', err.message);
      }
      validateNow();
    });
    dom.inspectorBody.appendChild(testBtn);
  }

  if (TAP_NODE_TYPES.includes(node.type)) {
    dom.inspectorBody.appendChild(el('div', 'ns-muted', 'Read-only tap — connect a monitor output into this block to display it.'));
  }

  const deleteBtn = el('button', 'ns-btn ns-btn-danger ns-btn-small', 'Delete block');
  deleteBtn.addEventListener('click', () => removeSelected());
  dom.inspectorBody.appendChild(deleteBtn);
  refreshGuideOverlays();
}

function inspectorField(node, name, spec) {
  const wrap = el('label', 'ns-field');
  wrap.appendChild(el('span', 'ns-field-label', spec.label || name));

  const apply = (value) => {
    const before = snapshot();
    const parsed = defaultParams(node.type, { ...node.params, [name]: value });
    node.params = { ...node.params, ...parsed, [name]: parsed[name] };
    pushUndo(before);
    renderNodes();
    validateNow();
    recordAction('param_changed', { nodeId: node.id, param: name, value: node.params[name] });
  };

  if (spec.type === 'int' || spec.type === 'float') {
    const input = el('input', 'ns-input');
    input.type = 'number';
    input.value = node.params[name];
    if (spec.min !== undefined) input.min = spec.min;
    if (spec.max !== undefined) input.max = spec.max;
    input.step = spec.step || (spec.type === 'int' ? 1 : 0.001);
    input.addEventListener('change', () => apply(input.value));
    wrap.appendChild(input);
  } else if (spec.type === 'select') {
    const select = el('select', 'ns-input');
    for (const option of spec.options || []) {
      const optionEl = el('option', '', option);
      optionEl.value = option;
      if (option === node.params[name]) optionEl.selected = true;
      select.appendChild(optionEl);
    }
    select.addEventListener('change', () => apply(select.value));
    wrap.appendChild(select);
  } else if (spec.type === 'dataset') {
    const select = el('select', 'ns-input');
    for (const source of DATA_SOURCES) {
      const optionEl = el('option', '', source.label);
      optionEl.value = source.id;
      if (source.id === node.params[name]) optionEl.selected = true;
      select.appendChild(optionEl);
    }
    select.addEventListener('change', () => {
      apply(select.value);
      resolveAllSources().catch((err) => appendLog('error', err.message));
      renderInspector();
    });
    wrap.appendChild(select);
  } else if (spec.multiline) {
    const textarea = el('textarea', 'ns-input ns-textarea');
    textarea.value = node.params[name];
    textarea.rows = 3;
    textarea.addEventListener('change', () => apply(textarea.value));
    wrap.appendChild(textarea);
  } else {
    const input = el('input', 'ns-input');
    input.type = 'text';
    input.value = node.params[name];
    if (spec.placeholder) input.placeholder = spec.placeholder;
    input.addEventListener('change', () => {
      apply(input.value);
      if (SOURCE_NODE_TYPES.includes(node.type)) resolveAllSources().catch((err) => appendLog('error', err.message));
    });
    wrap.appendChild(input);
  }
  return wrap;
}

function customCsvControls(node) {
  const wrap = el('div', 'ns-field');
  const fileInput = el('input');
  fileInput.type = 'file';
  fileInput.accept = '.csv,text/csv';
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const dataset = DatasetManager.parseCSV(text, { name: file.name });
      state.customDatasets.custom_csv = dataset;
      appendLog('ok', `CSV loaded: ${basename(file.name)} — ${dataset.X.length} rows, ${dataset.featureNames.length} features, ${dataset.numClasses} classes.`);
      await resolveAllSources();
      renderInspector();
      afterGraphMutation({ resolve: false });
    } catch (err) {
      appendLog('error', `CSV rejected: ${err.message}`);
    }
  });
  wrap.appendChild(el('span', 'ns-field-label', 'CSV file'));
  wrap.appendChild(fileInput);
  const loaded = state.customDatasets.custom_csv;
  if (loaded) wrap.appendChild(el('div', 'ns-muted', `Loaded: ${loaded.X.length} rows · ${loaded.featureNames.length} features`));
  return wrap;
}

/* ------------------------------------------------------------ data layer */

function cacheKeyFor(node) {
  if (node.type === 'source') return `builtin:${node.params.datasetId}`;
  if (node.type === 'sql_source') return `sql:${node.params.db}|${node.params.query}`;
  if (node.type === 'api_source') return `api:${node.params.url}`;
  return node.id;
}

async function resolveNodeData(node, { force = false } = {}) {
  if (node.type === 'source' && node.params.datasetId === 'custom_csv') {
    const dataset = state.customDatasets.custom_csv;
    if (!dataset) throw new Error('No CSV uploaded yet — open the Source block and choose a file.');
    state.sourceData.set(node.id, dataset);
    state.sourceInfo.set(node.id, {
      type: dataset.type, name: dataset.name, numFeatures: dataset.numFeatures,
      numClasses: dataset.numClasses || 2, imageDim: dataset.imageDim || null
    });
    recordFeatureMeta(node.id, dataset);
    return;
  }

  const key = cacheKeyFor(node);
  if (!force && state.dataCache.has(key)) {
    const cached = state.dataCache.get(key);
    state.sourceData.set(node.id, cached.dataset);
    state.sourceInfo.set(node.id, cached.info);
    recordFeatureMeta(node.id, cached.dataset);
    return;
  }

  setStatus('Loading data…');
  const resolved = await resolveSourceDataset(node, {
    customDatasets: state.customDatasets,
    datasetOptions: { basePath: MNIST_BASE_PATH }
  });
  state.dataCache.set(key, resolved);
  state.sourceData.set(node.id, resolved.dataset);
  state.sourceInfo.set(node.id, resolved.info);
  recordFeatureMeta(node.id, resolved.dataset);
  appendLog('ok', `Data ready for ${getNodeType(node.type).label}: ${resolved.info.name} — ${resolved.info.numFeatures} features, ${resolved.info.numClasses} classes.`);
}

function recordFeatureMeta(nodeId, dataset) {
  const rows = dataset.trainX || dataset.X || [];
  const sample = rows[0] || [];
  const mins = new Array(sample.length).fill(Infinity);
  const maxs = new Array(sample.length).fill(-Infinity);
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      if (row[i] < mins[i]) mins[i] = row[i];
      if (row[i] > maxs[i]) maxs[i] = row[i];
    }
  }
  state.featuresMeta.set(nodeId, { mins, maxs, names: dataset.featureNames || [] });
}

async function resolveAllSources() {
  const sources = state.graph.nodeList().filter((n) => SOURCE_NODE_TYPES.includes(n.type));
  for (const node of sources) {
    await resolveNodeData(node);
  }
  validateNow();
  updateDatasetStatus();
}

function updateDatasetStatus() {
  const sources = state.graph.nodeList().filter((n) => SOURCE_NODE_TYPES.includes(n.type));
  const parts = sources.map((node) => {
    const info = state.sourceInfo.get(node.id);
    const dataset = state.sourceData.get(node.id);
    if (!info || !dataset) return `${getNodeType(node.type).label}: not loaded`;
    const rows = dataset.trainX ? dataset.trainX.length : (dataset.X ? dataset.X.length : 0);
    return `${info.name} (${rows} rows)`;
  });
  dom.statusDataset.textContent = parts.join(' · ');
  if (actionBar) {
    if (sources.length === 0) {
      actionBar.setDataset('Dataset: —', 'No dataset source on the canvas');
    } else {
      const firstInfo = state.sourceInfo.get(sources[0].id);
      const name = firstInfo ? firstInfo.name : 'loading…';
      const short = name.length > 24 ? `${name.slice(0, 23)}…` : name;
      actionBar.setDataset(`Dataset: ${short}`, parts.join(' · '));
    }
  }
}

/* ---------------------------------------------------------- validation */

function validateNow() {
  state.inference = inferGraph(state.graph, compileContext());
  state.plan = compile(state.graph, compileContext());
  state.suggestion = (state.mode === 'assist' && state.selectedId && state.plan && !state.plan.ok)
    ? suggestConnection(state.graph, state.selectedId, compileContext())
    : null;
  renderValidation();
  renderNodes();
  renderMetricsHeader();
  if (missions) missions.check(missionContext());
}

function renderValidation() {
  dom.validationPanel.innerHTML = '';
  const plan = state.plan;

  // Empty canvas (or a render before the first validation pass): guide, don't scare.
  if (!plan || state.graph.nodeList().length === 0) {
    dom.validationPanel.appendChild(el('div', 'ns-validation-info',
      'Empty canvas — drag blocks from the left, press a preset, or let Smart Assist build a starter graph.'));
    dom.btnFix.hidden = state.mode !== 'assist';
    dom.statusParams.textContent = '';
    return;
  }

  if (plan.ok) {
    const chip = el('div', 'ns-validation-ok');
    chip.textContent = `✓ Graph valid — ${planLabel(plan)}`;
    dom.validationPanel.appendChild(chip);
    dom.btnFix.hidden = true;
    dom.statusParams.textContent = planLabel(plan);
    return;
  }

  const errors = plan.errors || [];
  if (errors.length === 0) {
    const chip = el('div', 'ns-validation-info', 'Graph incomplete — add blocks to build a full pipeline.');
    dom.validationPanel.appendChild(chip);
    dom.btnFix.hidden = state.mode !== 'assist';
    dom.statusParams.textContent = '';
    return;
  }

  for (const issue of errors.slice(0, 6)) {
    const item = el('div', 'ns-validation-error');
    item.textContent = `${issue.code}: ${issue.message}`;
    if (issue.nodeId && state.graph.getNode(issue.nodeId)) {
      item.classList.add('is-clickable');
      item.addEventListener('click', () => {
        state.selectedId = issue.nodeId;
        renderInspector();
        renderNodes();
      });
    }
    dom.validationPanel.appendChild(item);
  }
  if (errors.length > 6) {
    dom.validationPanel.appendChild(el('div', 'ns-muted', `…and ${errors.length - 6} more.`));
  }
  if (state.suggestion) {
    const suggestion = el('div', 'ns-validation-suggestion');
    suggestion.appendChild(el('div', 'ns-validation-suggestion-text', `Suggestion: ${state.suggestion.message}`));
    const apply = el('button', 'ns-btn ns-btn-accent', 'Apply suggestion');
    apply.type = 'button';
    apply.title = 'Apply this wire (Enter)';
    apply.addEventListener('click', applySuggestion);
    suggestion.appendChild(apply);
    dom.validationPanel.appendChild(suggestion);
  }
  dom.btnFix.hidden = state.mode !== 'assist';
  dom.statusParams.textContent = `${errors.length} issue(s)`;
}

function planLabel(plan) {
  const arch = plan.network.architecture
    .map((layer) => `${layer.size}${layer.activation && layer.activation !== 'linear' ? `·${layer.activation}` : ''}`)
    .join(' → ');
  return `${arch} · ${plan.network.loss} · ${plan.network.optimizer}`;
}

/* ----------------------------------------------------------- run control */

function datasetPayloadFor(dataset, plan) {
  const filtered = plan.data && plan.data.filter ? applyClassFilter(dataset, plan.data.filter.keep) : dataset;
  const split = ensureSplit(filtered, 0.8);
  const classification = (filtered.type || 'classification') !== 'regression';
  const numClasses = classification
    ? (filtered.numClasses || (filtered.targetNames ? filtered.targetNames.length : split.trainY[0].length))
    : 1;
  return {
    payload: {
      trainX: split.trainX,
      trainY: split.trainY,
      testX: split.testX,
      testY: split.testY,
      numFeatures: filtered.numFeatures || split.trainX[0].length,
      numClasses,
      featureNames: filtered.featureNames || [],
      targetNames: filtered.targetNames || []
    },
    dataset: filtered,
    classification
  };
}

function signatureFor(plan, datasetKey) {
  return JSON.stringify({ network: plan.network, datasetKey, input: plan.meta.flatInputSize });
}

async function startRun() {
  if (state.running) return true;

  await resolveAllSources().catch((err) => {
    appendLog('error', err.message);
    return null;
  });

  const plan = compile(state.graph, compileContext());
  state.plan = plan;
  renderValidation();
  if (!plan.ok) {
    appendLog('warn', `Cannot run — ${plan.errors.length} validation issue(s). Press "Fix automatically" or resolve them on the canvas.`);
    setStatus('Validation failed — see the error panel.');
    return false;
  }

  const sourceNode = state.graph.nodeList().find((n) => n.id === state.inference.chain[0]);
  const dataset = state.sourceData.get(sourceNode.id);
  if (!dataset) {
    appendLog('error', 'The dataset for the source block is not loaded yet.');
    return false;
  }

  const { payload, dataset: effective, classification } = datasetPayloadFor(dataset, plan);
  const datasetKey = plan.data.datasetId || plan.data.db || plan.data.url || sourceNode.id;
  const signature = signatureFor(plan, datasetKey);
  const continuing = state.builtSignature === signature && state.weights !== null;

  if (!continuing) {
    if (classification) {
      const expectedClasses = plan.meta.effectiveClasses || payload.numClasses;
      if (plan.network.loss !== 'mse' && payload.trainY[0] && payload.trainY[0].length !== plan.network.architecture[plan.network.architecture.length - 1].size) {
        appendLog('error', `Target shape (${payload.trainY[0].length}) does not match the output size (${plan.network.architecture[plan.network.architecture.length - 1].size}) — check the Class Filter.`);
        return false;
      }
      if (expectedClasses && payload.numClasses !== expectedClasses) {
        appendLog('warn', `Compiler expected ${expectedClasses} classes but the data has ${payload.numClasses}.`);
      }
    }
    worker.postMessage({
      type: 'BUILD_NETWORK',
      payload: {
        architecture: plan.network.architecture,
        hyperparameters: {
          loss: plan.network.loss,
          optimizer: plan.network.optimizer,
          learningRate: plan.network.learningRate,
          weightDecay: plan.network.weightDecay
        }
      }
    });
    worker.postMessage({ type: 'SET_DATASET', payload: payload });
    state.builtSignature = signature;
    state.epochsRun = 0;
    state.lossHistory = [];
    state.confusion = null;
    renderConfusion();
    appendLog('action', `Built network: ${planLabel(plan)} on ${payload.trainX.length} training rows.`);
  } else {
    appendLog('action', `Continuing the current network (epoch ${state.epochsRun}).`);
  }

  let maxEpochs = plan.training.mode === 'fixed' ? plan.training.maxEpochs : null;
  if (plan.training.mode === 'fixed' && continuing) {
    maxEpochs = state.epochsRun + plan.training.maxEpochs;
  }

  worker.postMessage({
    type: 'START_TRAINING',
    payload: {
      maxEpochs,
      earlyStopPatience: plan.training.earlyStopPatience,
      batchSize: plan.training.batchSize,
      epochDelayMs: state.epochDelayMs
    }
  });
  state.running = true;
  state.lastScheduledLr = plan.network.learningRate;
  updateRunButtons();
  setStatus(`Training… (${plan.training.mode === 'fixed' ? `budget ${plan.training.maxEpochs} epochs` : `auto-stop patience ${plan.training.earlyStopPatience}`}, batch ${plan.training.batchSize})`);
  recordAction('training_started', { epochs: maxEpochs, mode: plan.training.mode });
  return true;
}

function updateRunButtons() {
  if (actionBar) {
    actionBar.setTraining(state.running);
    return;
  }
  dom.btnRun.disabled = state.running;
  dom.btnRun.textContent = state.running ? 'Running…' : '▶ Run';
  dom.btnPause.hidden = !state.running;
}

/* ------------------------------------------------------- worker messages */

function onWorkerMessage(event) {
  const { type, payload } = event.data;

  switch (type) {
    case 'NETWORK_BUILT':
      state.weights = payload.weights;
      appendLog('ok', `Network ready (${payload.architecture.length} layers).`);
      updateProbePanel();
      break;

    case 'NETWORK_RESET':
      state.weights = payload.weights;
      break;

    case 'DATASET_READY':
      appendLog('ok', `Worker dataset loaded: ${payload.trainCount} train / ${payload.testCount} test rows.`);
      updateProbePanel();
      break;

    case 'TRAINING_STEP':
      state.epochsRun = payload.epoch;
      state.weights = payload.weights;
      state.lossHistory.push({ epoch: payload.epoch, loss: payload.loss, valLoss: payload.valLoss });
      if (state.lossHistory.length > 2000) state.lossHistory.shift();
      renderMetrics(payload);
      renderSparkline();
      applyScheduledLr(payload.epoch);
      if (playground) playground.handleTrainingStep(payload);
      break;

    case 'TRAINING_COMPLETE':
      state.running = false;
      updateRunButtons();
      appendLog('ok', `Run complete at epoch ${payload.epoch} (${payload.reason}${payload.bestValLoss !== null ? `, best held-out loss ${fmt(payload.bestValLoss)}` : ''}).`);
      setStatus(`Run complete — epoch ${payload.epoch} (${payload.reason}).`);
      requestEvaluation();
      if (playground) playground.handleTrainingComplete(payload);
      recordAction('training_complete', { epoch: payload.epoch, reason: payload.reason });
      break;

    case 'TRAINING_PAUSED':
      state.running = false;
      updateRunButtons();
      appendLog('info', `Paused at epoch ${payload.epoch}.`);
      if (playground) playground.handleTrainingPaused(payload);
      break;

    case 'TEST_SET_EVALUATED':
      if (payload.ok) {
        const isRegression = state.plan && state.plan.network && state.plan.network.loss === 'mse';
        if (isRegression) {
          appendLog('ok', `Held-out MSE: ${fmt(payload.loss, 5)} over ${payload.count} samples.`);
          setStatus(`Held-out MSE ${fmt(payload.loss, 4)} over ${payload.count} samples.`);
        } else {
          appendLog('ok', `Held-out accuracy: ${formatPercent(payload.accuracy)} over ${payload.count} samples.`);
          setStatus(`Held-out accuracy ${formatPercent(payload.accuracy)} over ${payload.count} samples.`);
        }
        requestConfusion();
      }
      if (playground) playground.handleEvaluation(payload);
      break;

    case 'TEST_SET_PREDICTIONS':
      if (payload.ok) {
        state.confusion = payload;
        renderConfusion();
      }
      break;

    case 'PREDICTION_RESULT':
      if (payload.source === 'playground' && playgroundPending && playgroundPending.kind === 'predict') {
        const pending = playgroundPending;
        playgroundPending = null;
        pending.resolve(payload);
      } else {
        renderProbeResult(payload);
        recordAction('predicted', { source: payload.source || 'probe' });
      }
      break;

    case 'ONLINE_STEP_COMPLETE':
      if (payload.weights) state.weights = payload.weights;
      if (payload.source === 'playground' && playgroundPending && playgroundPending.kind === 'correct') {
        const pending = playgroundPending;
        playgroundPending = null;
        pending.resolve(payload);
      } else if (!payload.ok) {
        appendLog('error', `Online correction failed: ${payload.error || 'unknown error'}`);
      } else {
        appendLog('action', `Online correction: loss ${fmt(payload.initialLoss, 5)} → ${fmt(payload.postLoss, 5)} (${payload.lossReduced ? 'reduced' : 'not reduced'}) on the corrected sample.`);
      }
      break;

    case 'ERROR':
      if (playgroundPending) {
        const pending = playgroundPending;
        playgroundPending = null;
        pending.reject(new Error(payload.message));
      }
      appendLog('error', payload.message);
      setStatus(`Error: ${payload.message}`);
      break;

    default:
      break;
  }
}

function requestEvaluation() {
  if (!state.plan || !state.plan.ok || !state.weights) return;
  worker.postMessage({ type: 'EVALUATE_TEST_SET', payload: {} });
}

function requestConfusion() {
  const isRegression = state.plan && state.plan.network && state.plan.network.loss === 'mse';
  if (isRegression) return;
  if (!state.plan.taps || !state.plan.taps.confusion) {
    renderConfusion();
    return;
  }
  worker.postMessage({ type: 'PREDICT_TEST_SET', payload: {} });
}

/* ------------------------------------------------------ playground bridge */

function playgroundClassNames() {
  const source = state.graph.nodeList().find((node) => SOURCE_NODE_TYPES.includes(node.type));
  const dataset = source ? state.sourceData.get(source.id) : null;
  if (dataset && Array.isArray(dataset.targetNames) && dataset.targetNames.length > 0) {
    return dataset.targetNames.map((name) => String(name).replace(/^Digit\s+/i, ''));
  }
  const size = state.plan && state.plan.ok
    ? state.plan.network.architecture[state.plan.network.architecture.length - 1].size
    : 2;
  return Array.from({ length: size }, (_, index) => String(index));
}

async function preparePlayground() {
  if (state.presetId !== 'mnist_classifier') {
    if (state.running) {
      worker.postMessage({ type: 'PAUSE_TRAINING' });
      state.running = false;
      updateRunButtons();
    }
    await applyPreset('mnist_classifier');
  } else {
    await resolveAllSources().catch((err) => appendLog('error', err.message));
  }
  validateNow();
  if (!state.plan || !state.plan.ok) {
    throw new Error('the pre-assembled graph did not compile');
  }
  return { plan: state.plan, weights: state.weights, running: state.running };
}

function playgroundPredict(input) {
  if (!state.weights) return Promise.reject(new Error('the network has not been trained yet'));
  if (playgroundPending) return Promise.reject(new Error('another playground request is still running'));
  return new Promise((resolve, reject) => {
    playgroundPending = { kind: 'predict', resolve, reject };
    worker.postMessage({ type: 'RUN_PREDICTION', payload: { input, source: 'playground' } });
  });
}

function playgroundCorrect(input, trueLabel) {
  if (!state.plan || !state.plan.ok) return Promise.reject(new Error('the graph is not compiled'));
  if (playgroundPending) return Promise.reject(new Error('another playground request is still running'));
  const size = state.plan.network.architecture[state.plan.network.architecture.length - 1].size;
  const target = new Array(size).fill(0);
  if (trueLabel < 0 || trueLabel >= size) return Promise.reject(new Error(`true label ${trueLabel} is outside the output size ${size}`));
  target[trueLabel] = 1;
  return new Promise((resolve, reject) => {
    playgroundPending = { kind: 'correct', resolve, reject };
    worker.postMessage({
      type: 'ONLINE_STEP',
      payload: {
        sample: input,
        target,
        trueLabel,
        steps: 1,
        learningRate: 0.05,
        source: 'playground'
      }
    });
  });
}

function applyScheduledLr(epoch) {
  const schedule = state.plan && state.plan.ok ? state.plan.training.lrSchedule : null;
  if (!schedule || schedule.mode !== 'step') return;
  const multiplier = schedulerValue(schedule, epoch);
  const lr = Number((state.plan.network.learningRate * multiplier).toFixed(6));
  if (lr !== state.lastScheduledLr) {
    state.lastScheduledLr = lr;
    worker.postMessage({ type: 'UPDATE_HYPERPARAMS', payload: { learningRate: lr } });
    appendLog('action', `LR schedule: learning rate → ${lr} at epoch ${epoch}.`);
  }
}

/* --------------------------------------------------------------- panels */

function renderMetricsHeader() {
  if (!dom.metricsMode) return;
  const plan = state.plan;
  if (!plan || !plan.ok) {
    dom.metricsMode.textContent = 'no valid graph';
    return;
  }
  dom.metricsMode.textContent = plan.training.mode === 'fixed'
    ? `fixed ${plan.training.maxEpochs} epochs`
    : `auto-stop · patience ${plan.training.earlyStopPatience}`;
}

function renderMetrics(payload) {
  if (dom.metricEpoch) dom.metricEpoch.textContent = String(payload.epoch);
  if (dom.metricLoss) dom.metricLoss.textContent = fmt(payload.loss, 5);
  if (dom.metricValLoss) dom.metricValLoss.textContent = fmt(payload.valLoss, 5);
  if (dom.metricValAcc) dom.metricValAcc.textContent = payload.valAccuracy === null || payload.valAccuracy === undefined ? '—' : formatPercent(payload.valAccuracy);
  if (dom.metricSpeed) dom.metricSpeed.textContent = `${payload.speed} ep/s`;
  if (dom.metricBest) dom.metricBest.textContent = payload.bestValLoss === null ? '—' : fmt(payload.bestValLoss, 5);
  if (dom.metricStall) dom.metricStall.textContent = String(payload.stallEpochs);
}

function renderSparkline() {
  const canvas = dom.lossSpark;
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const history = state.lossHistory;
  if (history.length < 2) return;
  const windowSize = Math.min(history.length, 300);
  const slice = history.slice(-windowSize);
  const maxVal = Math.max(...slice.map((h) => Math.max(h.loss || 0, h.valLoss || 0)), 1e-6);

  const plot = (key, color) => {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    slice.forEach((point, index) => {
      const x = (index / (slice.length - 1)) * (width - 8) + 4;
      const y = height - 6 - ((point[key] || 0) / maxVal) * (height - 14);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot('loss', '#8fb7e8');
  plot('valLoss', '#d9a441');
}

function renderConfusion() {
  dom.confBody.innerHTML = '';
  const isRegression = state.plan && state.plan.ok && state.plan.network.loss === 'mse';
  if (isRegression) {
    dom.confBody.appendChild(el('div', 'ns-muted', 'Regression model — the held-out MSE readout in the log replaces the matrix.'));
    return;
  }
  const data = state.confusion;
  if (!data || !data.ok) {
    const tapped = state.plan && state.plan.ok && state.plan.taps && state.plan.taps.confusion;
    dom.confBody.appendChild(el('div', 'ns-muted', tapped
      ? 'Run a training session to completion — predictions over the held-out set will fill this matrix.'
      : 'Add a Confusion Matrix block and wire it to the loss monitor to enable this tap.'));
    return;
  }

  const classes = state.plan.meta.effectiveClasses
    || (state.plan.network.architecture[state.plan.network.architecture.length - 1].size);
  const counts = Array.from({ length: classes }, () => new Array(classes).fill(0));
  for (const row of data.predictions) {
    if (counts[row.trueClass] && counts[row.trueClass][row.predictedClass] !== undefined) {
      counts[row.trueClass][row.predictedClass]++;
    }
  }
  const total = data.predictions.length;
  const correct = counts.reduce((sum, row, i) => sum + row[i], 0);

  const table = el('table', 'ns-confusion');
  const headerRow = el('tr');
  headerRow.appendChild(el('th', '', 'true ↓ / pred →'));
  for (let c = 0; c < classes; c++) headerRow.appendChild(el('th', '', `c${c}`));
  table.appendChild(headerRow);
  for (let r = 0; r < classes; r++) {
    const rowEl = el('tr');
    rowEl.appendChild(el('th', '', `c${r}`));
    for (let c = 0; c < classes; c++) {
      const cell = el('td', r === c ? 'is-correct' : (counts[r][c] > 0 ? 'is-wrong' : ''), String(counts[r][c]));
      rowEl.appendChild(cell);
    }
    table.appendChild(rowEl);
  }
  dom.confBody.appendChild(table);
  dom.confChip.textContent = `${correct}/${total} correct · ${formatPercent(correct / total)}`;
}

/* ---------------------------------------------------------- probe panel */

function updateProbePanel() {
  dom.predictBody.innerHTML = '';
  const plan = state.plan;
  if (!plan || !plan.ok) {
    dom.predictBody.appendChild(el('div', 'ns-muted', 'Compile a valid graph to probe the model.'));
    return;
  }
  if (!state.weights) {
    dom.predictBody.appendChild(el('div', 'ns-muted', 'Press Run once — after the network is built you can probe it live.'));
    return;
  }
  const sourceNode = state.graph.nodeList().find((n) => n.id === state.inference.chain[0]);
  const info = state.sourceInfo.get(sourceNode.id);
  const isImage = info && info.imageDim;

  if (isImage) {
    dom.predictBody.appendChild(buildDigitPad(info.imageDim));
  } else {
    dom.predictBody.appendChild(buildFeatureSliders(sourceNode.id));
  }

  const predictBtn = el('button', 'ns-btn ns-btn-small', 'Predict sample');
  predictBtn.addEventListener('click', () => requestProbe());
  dom.predictBody.appendChild(predictBtn);

  const resultWrap = el('div', 'ns-probe-result');
  resultWrap.id = 'probe-result';
  dom.predictBody.appendChild(resultWrap);
}

function buildDigitPad(imageDim) {
  const wrap = el('div', 'ns-pad-wrap');
  const width = imageDim.width || 28;
  const height = imageDim.height || 28;
  const scale = Math.max(3, Math.floor(168 / Math.max(width, height)));
  const canvas = el('canvas', 'ns-pad');
  canvas.width = width * scale;
  canvas.height = height * scale;
  wrap.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  padState = { pixels: new Array(width * height).fill(0), width, height, scale, drawing: false };

  const drawAt = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * height);
    const radius = 2;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = x + dx;
        const py = y + dy;
        if (px < 0 || py < 0 || px >= width || py >= height) continue;
        const strength = Math.max(0, 1 - Math.hypot(dx, dy) / (radius + 0.5));
        const index = py * width + px;
        padState.pixels[index] = Math.min(1, padState.pixels[index] + strength);
      }
    }
    renderPad(ctx, padState);
  };

  canvas.addEventListener('pointerdown', (event) => {
    padState.drawing = true;
    canvas.setPointerCapture(event.pointerId);
    drawAt(event);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (padState.drawing) drawAt(event);
  });
  canvas.addEventListener('pointerup', () => { padState.drawing = false; });
  canvas.addEventListener('pointerleave', () => { padState.drawing = false; });

  const clearBtn = el('button', 'ns-btn ns-btn-small', 'Clear pad');
  clearBtn.addEventListener('click', () => {
    padState.pixels = new Array(width * height).fill(0);
    renderPad(ctx, padState);
  });
  wrap.appendChild(clearBtn);
  return wrap;
}

function renderPad(ctx, pad) {
  const cell = pad.scale;
  for (let y = 0; y < pad.height; y++) {
    for (let x = 0; x < pad.width; x++) {
      const value = Math.round(255 * (1 - pad.pixels[y * pad.width + x]));
      ctx.fillStyle = `rgb(${value},${value},${value})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
}

function buildFeatureSliders(sourceNodeId) {
  const wrap = el('div', 'ns-sliders');
  const meta = state.featuresMeta.get(sourceNodeId);
  if (!meta) {
    wrap.appendChild(el('div', 'ns-muted', 'Feature ranges not measured yet — reload the dataset.'));
    return wrap;
  }
  const controls = [];
  meta.names.forEach((name, index) => {
    const min = meta.mins[index];
    const max = meta.maxs[index];
    const field = el('label', 'ns-field ns-field-inline');
    field.appendChild(el('span', 'ns-field-label', name));
    const range = el('input', 'ns-range');
    range.type = 'range';
    range.min = min;
    range.max = max;
    range.step = (max - min) / 100 || 0.01;
    range.value = (min + max) / 2;
    const readout = el('span', 'ns-range-value', Number(range.value).toFixed(2));
    range.addEventListener('input', () => { readout.textContent = Number(range.value).toFixed(2); });
    field.appendChild(range);
    field.appendChild(readout);
    wrap.appendChild(field);
    controls.push({ index, range });
  });
  wrap._controls = controls;
  return wrap;
}

function requestProbe() {
  const plan = state.plan;
  if (!plan || !plan.ok) return;
  const sourceNode = state.graph.nodeList().find((n) => n.id === state.inference.chain[0]);
  const info = state.sourceInfo.get(sourceNode.id);
  let input = null;

  if (info && info.imageDim && padState) {
    input = padState.pixels.map((value) => Number(value.toFixed(4)));
  } else {
    const sliders = dom.predictBody.querySelector('.ns-sliders');
    if (sliders && sliders._controls) {
      const values = new Array(sliders._controls.length).fill(0);
      for (const control of sliders._controls) values[control.index] = Number(control.range.value);
      input = values;
    }
  }

  if (!input || input.length !== plan.meta.flatInputSize) {
    appendLog('warn', input
      ? `Probe vector has ${input.length} values but the model expects ${plan.meta.flatInputSize}.`
      : 'No probe input yet — draw on the pad or set the sliders.');
    return;
  }

  worker.postMessage({ type: 'RUN_PREDICTION', payload: { input, source: 'node_probe' } });
}

function renderProbeResult(payload) {
  const wrap = dom.predictBody.querySelector('.ns-probe-result');
  if (!wrap) return;
  wrap.innerHTML = '';
  const plan = state.plan;
  const isRegression = plan.network.loss === 'mse';

  if (isRegression) {
    wrap.appendChild(el('div', 'ns-probe-regression', `Prediction: ${fmt(payload.output[0], 5)}`));
    return;
  }

  const targetNames = (state.sourceData.get(state.graph.nodeList().find((n) => n.id === state.inference.chain[0]).id) || {}).targetNames || [];
  payload.probabilities.forEach((probability, index) => {
    const row = el('div', 'ns-probe-row');
    row.appendChild(el('span', 'ns-probe-label', targetNames[index] || `class ${index}`));
    const bar = el('div', 'ns-probe-bar');
    const fill = el('div', 'ns-probe-fill');
    fill.style.width = `${Math.round(probability * 100)}%`;
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(el('span', 'ns-probe-value', formatPercent(probability)));
    if (index === payload.predictedClass) row.classList.add('is-predicted');
    wrap.appendChild(row);
  });
  wrap.appendChild(el('div', 'ns-muted', `Predicted: ${targetNames[payload.predictedClass] || `class ${payload.predictedClass}`} · confidence ${formatPercent(payload.confidence)}`));
}

/* --------------------------------------------------------------- exports */

async function handleExport(kind) {
  if (!state.plan || !state.plan.ok) {
    appendLog('warn', 'Compile a valid graph before exporting.');
    return;
  }

  if (kind === 'commands') {
    appendLog('info', 'Packaging commands:');
    for (const entry of buildCommands()) {
      appendLog('action', `${entry.command}  —  ${entry.detail}`);
    }
    recordAction('export_commands');
    return;
  }

  if (!state.weights) {
    appendLog('warn', 'Run training once before exporting — the exporter ships trained weights.');
    return;
  }

  const artifacts = buildExportArtifacts({
    plan: state.plan,
    weights: state.weights,
    datasetName: state.plan.meta.datasetName,
    epochs: state.epochsRun
  });

  if (kind === 'json') {
    downloadText(JSON.stringify(artifacts.jsonPayload, null, 2), 'node-studio-model.json', 'application/json');
    appendLog('ok', 'Exported model JSON (re-loadable by both studios).');
  } else if (kind === 'python') {
    downloadText(artifacts.pythonSource, 'node_studio_model.py', 'text/x-python');
    appendLog('ok', 'Exported standalone Python model (NumPy).');
  } else if (kind === 'js') {
    downloadText(artifacts.jsSource, 'node_studio_model.mjs', 'text/javascript');
    appendLog('ok', 'Exported standalone JavaScript module.');
  } else if (kind === 'npz') {
    downloadBytes(artifacts.npzBytes, 'node_studio_weights.npz');
    appendLog('ok', 'Exported .npz weight archive (NumPy-compatible).');
  } else if (kind === 'onnx') {
    downloadBytes(artifacts.onnxBytes, 'node_studio_model.onnx', 'application/onnx');
    appendLog('ok', 'Exported ONNX inference graph (opset 13) — load it with onnxruntime.');
  }
  recordAction(`export_${kind}`, { kind });
}

/* ------------------------------------------------------------------ init */

export function initNodeStudio() {
  dom = {
    palette: document.getElementById('palette'),
    studioMain: document.getElementById('studio-main'),
    playgroundSection: document.getElementById('playground'),
    canvas: document.getElementById('studio-canvas'),
    graphLayer: document.getElementById('graph-layer'),
    wireLayer: document.getElementById('wire-layer'),
    studioCanvas: document.getElementById('studio-canvas'),
    nodeLayer: document.getElementById('node-layer'),
    validationPanel: document.getElementById('validation-panel'),
    statusText: document.getElementById('status-text'),
    statusDataset: document.getElementById('status-dataset'),
    statusParams: document.getElementById('status-params'),
    inspectorPanel: document.getElementById('inspector-panel'),
    inspectorBody: document.getElementById('inspector-body'),
    metricsPanel: document.getElementById('panel-metrics'),
    metricsMode: document.getElementById('metrics-mode'),
    metricEpoch: document.getElementById('metric-epoch'),
    metricLoss: document.getElementById('metric-loss'),
    metricValLoss: document.getElementById('metric-val-loss'),
    metricValAcc: document.getElementById('metric-val-acc'),
    metricSpeed: document.getElementById('metric-speed'),
    metricBest: document.getElementById('metric-best'),
    metricStall: document.getElementById('metric-stall'),
    lossSpark: document.getElementById('loss-spark'),
    epochDelay: document.getElementById('epoch-delay'),
    toggleParticles: document.getElementById('toggle-particles'),
    confBody: document.getElementById('conf-body'),
    confChip: document.getElementById('conf-chip'),
    predictBody: document.getElementById('predict-body'),
    logList: document.getElementById('log-list'),
    btnModePlayground: document.getElementById('btn-mode-playground'),
    btnModeAssist: document.getElementById('btn-mode-assist'),
    btnModeManual: document.getElementById('btn-mode-manual'),
    actionDataset: document.getElementById('action-dataset'),
    actionMetrics: document.getElementById('action-metrics'),
    btnPresetMnist: document.getElementById('btn-preset-mnist'),
    btnPresetXor: document.getElementById('btn-preset-xor'),
    btnPresetIris: document.getElementById('btn-preset-iris'),
    btnTour: document.getElementById('btn-tour'),
    tourMenu: document.getElementById('tour-menu'),
    btnGraphMenu: document.getElementById('btn-graph-menu'),
    graphMenu: document.getElementById('graph-menu'),
    btnFix: document.getElementById('btn-fix'),
    btnRun: document.getElementById('btn-run'),
    btnPause: document.getElementById('btn-pause'),
    btnReset: document.getElementById('btn-reset'),
    btnUndo: document.getElementById('btn-undo'),
    btnSaveGraph: document.getElementById('btn-save-graph'),
    btnLoadGraph: document.getElementById('btn-load-graph'),
    fileLoadGraph: document.getElementById('file-load-graph'),
    btnExport: document.getElementById('btn-export'),
    exportMenu: document.getElementById('export-menu')
  };

  state.graph = new GraphState('assist');
  worker = new Worker(new URL('../engine/nn_worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (event) => appendLog('error', `Worker error: ${event.message || 'unknown'}`);

  menus = new MenuManager();
  tutorial = new TutorialOverlay({
    manageEscape: false, // MenuManager owns the Escape stack; this overlay registers with it
    onComplete: () => appendLog('info', 'Tutorial finished.')
  });
  curriculum = new CurriculumTour({ overlay: tutorial, actions: state.actions });
  menus.registerOverlay({
    id: 'tour-overlay',
    isOpen: () => curriculum.active,
    requestClose: () => curriculum.finish()
  });
  menus.registerOverlay({
    id: 'deep-dive-tour',
    isOpen: () => tutorial.active,
    requestClose: () => tutorial.finish()
  });

  missions = new StepByStepEngine({
    menuManager: menus,
    container: document.body,
    resolveTarget: (target) => resolveMissionTarget(target),
    onSetup: (setup, step, engine) => handleMissionSetup(setup, step, engine),
    onStepEnter: (step) => appendLog('info', `Mission step: ${step.title}`),
    onComplete: (mission) => appendLog('ok', `Mission complete: ${mission.title} — every gate was verified against the live graph.`),
    onExit: (mission) => appendLog('info', `Mission exited: ${mission.title}.`),
    onLog: (level, message) => appendLog(level, message)
  });
  menus.registerOverlay({
    id: 'mission-overlay',
    isOpen: () => missions.active,
    requestClose: () => missions.finish('escape')
  });

  quickMenu = new QuickNodeMenu({
    menuManager: menus,
    onPick: (row) => applyQuickPick(row),
    onClose: () => {
      if (pendingDrop) {
        pendingDrop = null;
        renderWires();
      }
    }
  });

  wireInteractions = new WireInteractions({
    wireLayer: dom.wireLayer,
    menuManager: menus,
    getWires: () => collectWires(state.graph),
    getScale: () => state.view.scale,
    toWorld: (client) => {
      const rect = dom.canvas.getBoundingClientRect();
      return screenToWorld({ x: client.x - rect.left, y: client.y - rect.top }, state.view);
    },
    menuModelFor: (wire, hit) => decorateWireMenu(wireMenuBaseModel(wire, hit), {
      tier: activeTier(tierContext(state, missions)),
      graph: state.graph,
      context: compileContext()
    }),
    onBreak: (connectionId) => breakWire(connectionId)
  });
  wireInteractions.attach(dom.canvas);

  actionBar = new ActionBar({
    root: document.querySelector('.ns-topbar'),
    modeButtons: {
      playground: dom.btnModePlayground,
      assist: dom.btnModeAssist,
      manual: dom.btnModeManual
    },
    datasetButton: dom.actionDataset,
    metricsButton: dom.actionMetrics,
    runButton: dom.btnRun,
    pauseButton: dom.btnPause,
    onMode: (mode) => setMode(mode),
    onDataset: () => {
      const source = state.graph.nodeList().find((node) => SOURCE_NODE_TYPES.includes(node.type));
      if (!source) {
        appendLog('warn', 'No dataset source on the canvas — add a Source or SQL block first.');
        return;
      }
      state.selectedId = source.id;
      renderInspector();
      renderNodes();
      if (dom.inspectorPanel.scrollIntoView) dom.inspectorPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },
    onMetrics: () => {
      if (dom.metricsPanel.scrollIntoView) dom.metricsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      dom.metricsPanel.classList.add('is-flash');
      setTimeout(() => dom.metricsPanel.classList.remove('is-flash'), 900);
    }
  });
  actionBar.setMode('assist');
  document.body.dataset.tier = 'assist';

  palette = new NodePalette(dom.palette, {
    onAdd: (type) => paletteAddAtCenter(type)
  });

  playground = new AbsoluteBeginnerPlayground({
    controller: {
      prepare: preparePlayground,
      startTraining: () => startRun(),
      stopTraining: () => worker.postMessage({ type: 'PAUSE_TRAINING' }),
      predict: playgroundPredict,
      correct: playgroundCorrect,
      getClassNames: playgroundClassNames
    },
    onAction: (actionId) => recordAction(actionId, { source: 'playground' }),
    onExit: () => setMode('assist')
  });

  renderPalette();
  bindEvents();
  applyView();
  renderNodes();
  renderWires();
  renderInspector();
  validateNow();
  renderConfusion();
  updateProbePanel();
  updateRunButtons();
  updateDatasetStatus();

  appendLog('info', 'Node Studio ready. Drop blocks from the left, or press a preset to start.');
  setStatus('Ready — build a graph or load a preset.');
  requestAnimationFrame(function tick() {
    positionParticles();
    requestAnimationFrame(tick);
  });
}

function bindEvents() {
  bindCanvasEvents();

  // Mode buttons are bound by ActionBar.
  dom.btnPresetMnist.addEventListener('click', () => applyPreset('mnist_classifier'));
  dom.btnPresetXor.addEventListener('click', () => applyPreset('xor_classifier'));
  dom.btnPresetIris.addEventListener('click', () => applyPreset('iris_multi_class'));

  menus.register({
    id: 'tour-menu',
    trigger: dom.btnTour,
    panel: dom.tourMenu
  });
  menus.register({
    id: 'graph-menu',
    trigger: dom.btnGraphMenu,
    panel: dom.graphMenu
  });
  if (dom.graphMenu) {
    dom.graphMenu.addEventListener('click', (event) => {
      if (event.target.closest('button')) menus.close('graph-menu');
    });
  }
  menus.register({
    id: 'export-menu',
    trigger: dom.btnExport,
    panel: dom.exportMenu,
    onOpen: () => recordAction('export_opened')
  });

  const closeActiveMissionForGuide = () => {
    // A spotlight tour and a build mission both own a full-screen dim + spotlight. Leaving both
    // mounted is exactly what makes the cards fight over the target; the new guide wins.
    if (missions && missions.active) missions.finish('guide');
  };
  for (const button of dom.tourMenu.querySelectorAll('[data-tour-track]')) {
    button.addEventListener('click', () => {
      menus.close('tour-menu');
      closeActiveMissionForGuide();
      const trackId = button.dataset.tourTrack;
      if (curriculum.start(trackId)) appendLog('info', `Started curriculum track "${button.textContent.trim()}".`);
    });
  }
  for (const tourId of Object.keys(TOURS)) {
    const button = dom.tourMenu.querySelector(`[data-tour-id="${tourId}"]`);
    if (button) button.addEventListener('click', () => {
      menus.close('tour-menu');
      closeActiveMissionForGuide();
      tutorial.start(tourId, { actions: state.actions });
    });
  }
  for (const button of dom.tourMenu.querySelectorAll('[data-mission]')) {
    button.addEventListener('click', () => {
      menus.close('tour-menu');
      // Close a spotlit tour silently before the mission takes over the screen.
      if (tutorial && tutorial.active) tutorial.finish(true);
      const missionId = button.dataset.mission;
      if (missions && missions.start(missionId, missionContext())) {
        appendLog('info', `Started mission "${button.textContent.trim()}".`);
      } else {
        appendLog('warn', `Unknown mission "${missionId}".`);
      }
    });
  }

  dom.btnFix.addEventListener('click', () => {
    const before = snapshot();
    const result = autoFixGraphToConvergence(state.graph, compileContext());
    if (result.actions.length > 0) {
      pushUndo(before);
      for (const action of result.actions) appendLog('action', action.message || action.code);
      afterGraphMutation({ resolve: true });
      setStatus(`Smart Assist applied ${result.actions.length} fix(es).`);
    } else {
      appendLog('info', 'Smart Assist found nothing safe to fix automatically.');
    }
  });

  dom.btnRun.addEventListener('click', () => startRun());
  dom.btnPause.addEventListener('click', () => worker.postMessage({ type: 'PAUSE_TRAINING' }));
  dom.btnReset.addEventListener('click', () => {
    worker.postMessage({ type: 'PAUSE_TRAINING' });
    worker.postMessage({ type: 'RESET_NETWORK' });
    state.lossHistory = [];
    state.epochsRun = 0;
    state.builtSignature = null;
    state.confusion = null;
    renderSparkline();
    renderConfusion();
    appendLog('action', 'Network reset to fresh weights (dataset kept).');
  });

  dom.btnUndo.addEventListener('click', () => undo());
  dom.btnSaveGraph.addEventListener('click', () => {
    downloadText(JSON.stringify(state.graph.toJSON(), null, 2), 'node-studio-graph.json', 'application/json');
    appendLog('ok', 'Graph saved.');
  });
  dom.btnLoadGraph.addEventListener('click', () => dom.fileLoadGraph.click());
  dom.fileLoadGraph.addEventListener('change', async () => {
    const file = dom.fileLoadGraph.files[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const loaded = GraphState.fromJSON(parsed);
      pushUndo();
      state.graph = loaded;
      state.selectedId = null;
      state.presetId = null;
      if (loaded.loadReport && (loaded.loadReport.droppedNodes.length || loaded.loadReport.droppedConnections.length)) {
        appendLog('warn', `Graph loaded with drops: ${loaded.loadReport.droppedNodes.length} unknown block(s), ${loaded.loadReport.droppedConnections.length} dangling wire(s).`);
      } else {
        appendLog('ok', 'Graph loaded.');
      }
      state.weights = null;
      state.builtSignature = null;
      afterGraphMutation();
      await resolveAllSources();
      fitGraphView();
    } catch (err) {
      appendLog('error', `Graph load failed: ${err.message}`);
    }
  });

  for (const button of dom.exportMenu.querySelectorAll('[data-export]')) {
    button.addEventListener('click', () => {
      menus.close('export-menu');
      handleExport(button.dataset.export);
    });
  }

  dom.epochDelay.addEventListener('input', () => {
    const value = parseInt(dom.epochDelay.value, 10);
    state.epochDelayMs = value >= 100 ? 0 : Math.max(0, Math.floor((100 - value) * 2.5));
    if (state.weights) worker.postMessage({ type: 'UPDATE_HYPERPARAMS', payload: { epochDelayMs: state.epochDelayMs } });
    recordAction('epoch_delay_changed', { value, delayMs: state.epochDelayMs });
  });

  dom.toggleParticles.addEventListener('change', () => {
    state.particlesOn = dom.toggleParticles.checked;
    recordAction('particles_toggled', { enabled: state.particlesOn });
  });

  window.addEventListener('blur', clearTrashTarget);

  document.addEventListener('keydown', (event) => {
    const inField = event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && !inField) {
      event.preventDefault();
      undo();
    } else if (event.key === 'Enter' && !inField && state.suggestion) {
      event.preventDefault();
      applySuggestion();
    } else if ((event.key === 'Delete' || event.key === 'Backspace') && !inField) {
      if (selectedNode()) {
        event.preventDefault();
        removeSelected();
      }
    }
  });
}

async function setMode(mode) {
  if (!['playground', 'assist', 'manual'].includes(mode)) return;
  if (mode === state.mode) {
    recordAction(`mode_${mode}`, { alreadyActive: true });
    return;
  }

  const leavingPlayground = state.mode === 'playground';
  state.mode = mode;
  state.graph.mode = mode === 'manual' ? 'manual' : 'assist';
  state.suggestion = null;
  document.body.dataset.tier = mode;
  if (actionBar) actionBar.setMode(mode);

  const play = mode === 'playground';
  if (dom.studioMain) dom.studioMain.hidden = play;
  if (dom.playgroundSection) dom.playgroundSection.hidden = !play;

  if (play) {
    appendLog('info', 'Playground mode — a locked, pre-assembled brain with a drawing pad.');
    await playground.open();
    recordAction('mode_playground');
    refreshGuideOverlays();
    return;
  }

  if (leavingPlayground) {
    if (playground) playground.close();
    fitGraphView();
    validateNow();
    updateProbePanel();
  }
  renderValidation();
  appendLog('info', mode === 'assist'
    ? 'Beginner mode — Smart Assist repairs wiring and suggests the next wire.'
    : 'Advanced mode — full manual wiring, strict validation.');
  recordAction(`mode_${mode}`);
  refreshGuideOverlays();
}

async function applyPreset(id) {
  const preset = instantPreset(id);
  if (!preset) {
    appendLog('warn', `Unknown preset "${id}".`);
    return;
  }
  pushUndo();
  state.graph = preset;
  state.graph.mode = state.mode === 'manual' ? 'manual' : 'assist';
  state.selectedId = null;
  state.presetId = id;
  state.weights = null;
  state.builtSignature = null;
  state.confusion = null;
  state.lossHistory = [];
  state.epochsRun = 0;
  afterGraphMutation();
  renderConfusion();
  renderSparkline();
  await resolveAllSources().catch((err) => appendLog('error', err.message));
  fitGraphView();
  renderValidation();
  appendLog('ok', `Preset loaded: ${id === 'mnist_classifier' ? 'MNIST 0-vs-1 classifier' : id === 'xor_classifier' ? 'XOR classifier' : 'Iris multi-class'}. Press Run to train.`);
  recordAction(`preset_${id === 'mnist_classifier' ? 'mnist' : id === 'xor_classifier' ? 'xor' : 'iris'}`, { preset: id });
}
