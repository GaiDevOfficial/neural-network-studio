/**
 * Interactive wire mechanics — right-click raycasting, context menu, animated break.
 *
 * Verifies the exact contract the canvas depends on:
 *   - wires are hit-tested by sampling the real cubic Bézier (not DOM events),
 *   - the hit radius scales with zoom but clamps,
 *   - the nearest wire wins when two wires are close,
 *   - the context menu is a MenuManager popover (Escape / outside-click / injected ×),
 *   - "Break connection" runs the collapse animation FIRST, then performs exactly one real
 *     `GraphState.disconnect`, and the live wire list reflects the deletion,
 *   - a missing path element still results in a consistent graph (graceful degradation).
 *
 * Runs against the real GraphState + geometry with the DOM doubles in tests/fake_dom.js.
 *
 * File: tests/test_wire_interactions.js
 */

import { GraphState } from '../src/graph/graph_state.js';
import { MenuManager } from '../src/ui/menu_manager.js';
import { FakeDocument, FakeElement, FakeEvent } from './fake_dom.js';
import {
  DEFAULT_HIT_RADIUS,
  MIN_HIT_RADIUS,
  MAX_HIT_RADIUS,
  flattenBezier,
  distancePointToSegment,
  distanceToPolyline,
  wireHitRadius,
  hitTestWires,
  collectWires,
  wireMenuBaseModel,
  WireBreakAnimator,
  WireContextMenu,
  WireInteractions
} from '../src/canvas/wire_interactions.js';
import { decorateWireMenu } from '../src/modes/interactive_modes.js';
import { inferGraph } from '../src/graph/node_engine.js';

let passed = 0;
let failed = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  [PASS] ${name}`);
  } else {
    failed++;
    console.error(`  [FAIL] ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function fakeWindow() {
  return {
    innerWidth: 1280,
    innerHeight: 800,
    addEventListener() {},
    removeEventListener() {}
  };
}

const XOR_INFO = { datasetId: 'xor', type: 'classification', name: 'XOR', numFeatures: 2, numClasses: 2, imageDim: null };
const CONTEXT = { datasetInfoFor: (node) => (node.type === 'source' ? XOR_INFO : null) };

function buildChain() {
  const graph = new GraphState('assist');
  const source = graph.addNode('source', { x: 40, y: 140, params: { datasetId: 'xor' } });
  const dense = graph.addNode('dense', { x: 420, y: 140, params: { size: 8 } });
  const relu = graph.addNode('relu', { x: 800, y: 140 });
  const first = graph.connect(source.id, 'out', dense.id, 'in');
  const second = graph.connect(dense.id, 'out', relu.id, 'in');
  if (!first.ok || !second.ok) throw new Error('Fixture wiring failed.');
  return { graph, source, dense, relu, first: first.connection, second: second.connection };
}

function fakeScheduler() {
  const queue = [];
  return {
    queue,
    schedule(fn, ms) {
      queue.push({ fn, ms });
      return queue.length;
    },
    runAll() {
      while (queue.length > 0) queue.shift().fn();
    }
  };
}

const MNIST_INFO = { datasetId: 'mnist_01', type: 'classification', name: 'MNIST 0 vs 1', numFeatures: 784, numClasses: 2, imageDim: { width: 28, height: 28 } };
const MNIST_CONTEXT = { datasetInfoFor: (node) => (node.type === 'source' ? MNIST_INFO : null) };

/** The compile-valid 0-vs-1 classifier used to measure real topology warnings. */
function buildFullClassifier() {
  const graph = new GraphState('assist');
  const source = graph.addNode('source', { x: 0, y: 0, params: { datasetId: 'mnist_01' } });
  const flatten = graph.addNode('flatten', { x: 200, y: 0 });
  const d1 = graph.addNode('dense', { x: 400, y: 0, params: { size: 16 } });
  const relu = graph.addNode('relu', { x: 600, y: 0 });
  const d2 = graph.addNode('dense', { x: 800, y: 0, params: { size: 2 } });
  const sigmoid = graph.addNode('sigmoid', { x: 1000, y: 0 });
  const loss = graph.addNode('bce_loss', { x: 1200, y: 0 });
  const opt = graph.addNode('adam_optimizer', { x: 1200, y: 260 });
  const loop = graph.addNode('epoch_loop', { x: 1200, y: 420 });
  const wire = (a, as, b, bs) => {
    const result = graph.connect(a.id, as, b.id, bs);
    if (!result.ok) throw new Error(`Fixture wiring failed: ${result.error.code}`);
  };
  wire(source, 'out', flatten, 'in');
  wire(flatten, 'out', d1, 'in');
  wire(d1, 'out', relu, 'in');
  wire(relu, 'out', d2, 'in');
  wire(d2, 'out', sigmoid, 'in');
  wire(sigmoid, 'out', loss, 'in');
  wire(opt, 'cfg', loss, 'opt');
  wire(loop, 'cfg', loss, 'loop');
  return { graph, opt, loop };
}

function main() {
  console.log('=====================================================');
  console.log('  WIRE INTERACTIONS — raycast, context menu, break');
  console.log('=====================================================');

  /* ---------------------------------------------------------- polyline math */
  console.log('--- bézier sampling + distances ---');
  const straightFrom = { x: 0, y: 0 };
  const straightTo = { x: 100, y: 0 };
  const polyline = flattenBezier(straightFrom, straightTo, 28);
  check('flattenBezier returns segments + 1 points', polyline.length === 29, String(polyline.length));
  check('polyline starts exactly at the source socket', polyline[0].x === 0 && polyline[0].y === 0);
  check('polyline ends exactly at the target socket', Math.abs(polyline[28].x - 100) < 1e-9 && Math.abs(polyline[28].y) < 1e-9);

  const curved = flattenBezier({ x: 0, y: 0 }, { x: 200, y: 120 }, 28);
  const chordDistance = distancePointToSegment(curved[7], { x: 0, y: 0 }, { x: 200, y: 120 }).distance;
  check('the samples follow the real bézier, not the straight chord', chordDistance > 4, `quarter-point is ${chordDistance.toFixed(2)} off the chord`);

  const seg = distancePointToSegment({ x: 50, y: 10 }, { x: 0, y: 0 }, { x: 100, y: 0 });
  check('point-to-segment distance is exact', Math.abs(seg.distance - 10) < 1e-9, String(seg.distance));
  check('point-to-segment reports the clamped t', Math.abs(seg.t - 0.5) < 1e-9, String(seg.t));
  const endSeg = distancePointToSegment({ x: -20, y: 0 }, { x: 0, y: 0 }, { x: 100, y: 0 });
  check('past the end clamps to the endpoint', endSeg.t === 0 && Math.abs(endSeg.distance - 20) < 1e-9);

  const line = [{ x: 0, y: 0 }, { x: 100, y: 0 }];
  const mid = distanceToPolyline({ x: 50, y: 4 }, line);
  check('polyline distance uses the nearest segment', Math.abs(mid.distance - 4) < 1e-9);
  check('polyline t is normalized along the arc', Math.abs(mid.t - 0.5) < 1e-9, String(mid.t));

  check('hit radius default is 10 world units', wireHitRadius(1) === DEFAULT_HIT_RADIUS);
  check('hit radius shrinks when zoomed in and clamps at 5', wireHitRadius(4) === MIN_HIT_RADIUS, String(wireHitRadius(4)));
  check('hit radius grows when zoomed out and clamps at 22', wireHitRadius(0.4) === MAX_HIT_RADIUS, String(wireHitRadius(0.4)));
  check('hit radius survives invalid scales', wireHitRadius(0) === DEFAULT_HIT_RADIUS && wireHitRadius(NaN) === DEFAULT_HIT_RADIUS);

  /* ------------------------------------------------------------- live wires */
  console.log('--- collectWires on the real graph ---');
  const chain = buildChain();
  const wires = collectWires(chain.graph);
  check('both connections become live wires', wires.length === 2, String(wires.length));
  const firstWire = wires.find((wire) => wire.id === chain.first.id);
  check('wire carries its connection id', Boolean(firstWire && firstWire.id === chain.first.id));
  check('wire exposes real socket endpoints', firstWire.fromPos.x === 40 + 184 && firstWire.toPos.x === 420, `${firstWire.fromPos.x}→${firstWire.toPos.x}`);
  check('wire exposes node types for tier copy', firstWire.fromType === 'source' && firstWire.toType === 'dense');
  check('wire carries an SVG path and a sampled polyline', firstWire.path.startsWith('M ') && firstWire.polyline.length === 29);
  check('wire labels use the registry names and socket kinds', firstWire.fromLabel.includes('Dataset Source') && firstWire.fromLabel.includes('tensor'));

  console.log('--- hit-testing ---');
  const midPoint = firstWire.polyline[14];
  const hit = hitTestWires(wires, midPoint, 10);
  check('a click exactly on the curve hits that wire', Boolean(hit && hit.wire.id === chain.first.id));
  check('hit distance is ~0 on the curve', hit && hit.distance < 1e-6, String(hit && hit.distance));
  const offCurve = { x: midPoint.x, y: midPoint.y + 40 };
  check('a click well off the curve misses', hitTestWires(wires, offCurve, 10) === null);
  const nearMiss = { x: midPoint.x, y: midPoint.y + 14 };
  check('outside the radius misses', hitTestWires(wires, nearMiss, 10) === null);
  check('inside a larger radius hits', Boolean(hitTestWires(wires, nearMiss, 20)));

  const secondWire = wires.find((wire) => wire.id === chain.second.id);
  const secondMid = secondWire.polyline[14];
  const secondHit = hitTestWires(wires, secondMid, 10);
  check('when two wires exist the nearest one wins', Boolean(secondHit && secondHit.wire.id === chain.second.id));

  /* ----------------------------------------------------------- menu model */
  console.log('--- menu model + tier decoration ---');
  const base = wireMenuBaseModel(firstWire, hit);
  check('base model names the connection', base.connectionId === chain.first.id);
  check('base model carries raw endpoint labels', base.fromLabel.includes('out') && base.toLabel.includes('in'));
  check('base model offers exactly the break action', base.items.length === 1 && base.items[0].id === 'break' && base.items[0].danger === true);

  const advancedModel = decorateWireMenu(base, { tier: 3, graph: chain.graph, context: CONTEXT });
  check('advanced tier renames the action to a raw disconnect', advancedModel.items[0].label.includes('disconnect'));
  check('advanced tier computes a real topology warning', typeof advancedModel.warning === 'string' && advancedModel.warning.length > 0, advancedModel.warning);
  check('the warning reflects the real compiler delta', /E_[A-Z_]/.test(advancedModel.warning), advancedModel.warning);

  const friendlyModel = decorateWireMenu(base, { tier: 1, graph: chain.graph, context: CONTEXT });
  check('tier 1 uses the same block names as the canvas and hides sockets', friendlyModel.fromLabel === 'Dataset Source' && friendlyModel.toLabel === 'Dense Layer', `${friendlyModel.fromLabel} → ${friendlyModel.toLabel}`);
  check('tier 1 keeps the friendly break label', friendlyModel.items[0].label === 'Cut the wire');

  const validBase = wireMenuBaseModel(secondWire, secondHit);
  const keepGraph = decorateWireMenu(validBase, { tier: 3, graph: chain.graph, context: CONTEXT });
  check('cutting a non-terminal wire reports a real consequence', /E_[A-Z_]|valid/.test(keepGraph.warning || ''), keepGraph.warning);

  const full = buildFullClassifier();
  const optWire = collectWires(full.graph).find((wire) => wire.fromType === 'adam_optimizer' && wire.toType === 'bce_loss');
  const optModel = decorateWireMenu(wireMenuBaseModel(optWire, null), { tier: 3, graph: full.graph, context: MNIST_CONTEXT });
  check('cutting the optimizer predicts the real E_MISSING_CONFIG build break', /E_MISSING_CONFIG/.test(optModel.warning || ''), optModel.warning);

  const tapGraph = buildFullClassifier().graph;
  const tapLoss = tapGraph.nodeList().find((node) => node.type === 'bce_loss');
  const tap = tapGraph.addNode('metrics_gauge', { x: 1500, y: 300 });
  tapGraph.connect(tapLoss.id, 'monitor', tap.id, 'in');
  const tapWire = collectWires(tapGraph).find((wire) => wire.toType === 'metrics_gauge');
  const tapModel = decorateWireMenu(wireMenuBaseModel(tapWire, null), { tier: 3, graph: tapGraph, context: MNIST_CONTEXT });
  check('cutting a non-essential monitor tap reports that the build still compiles', /still compiles/.test(tapModel.warning || ''), tapModel.warning);

  /* -------------------------------------------------------------- animator */
  console.log('--- break animator ---');
  const doc = new FakeDocument();
  const scheduler = fakeScheduler();
  const animator = new WireBreakAnimator({ schedule: scheduler.schedule, duration: 180 });
  const pathEl = new FakeElement('path', doc);
  let animatedGraphBroken = false;
  const started = animator.play(pathEl, () => { animatedGraphBroken = true; });
  check('first play starts the animation', started === true);
  check('the path gets the is-breaking class', pathEl.classList.contains('is-breaking'));
  check('the graph is NOT touched while the collapse animates', animatedGraphBroken === false);
  check('a second play of the same path is refused', animator.play(pathEl, () => {}) === false);
  check('the scheduler was asked for the real duration', scheduler.queue.length === 1 && scheduler.queue[0].ms === 180);
  scheduler.runAll();
  check('the completion callback runs after the animation', animatedGraphBroken === true);
  check('the class is cleared when the animation finishes', !pathEl.classList.contains('is-breaking'));
  let immediate = false;
  animator.play(null, () => { immediate = true; });
  check('a missing path element degrades to an immediate callback', immediate === true);

  /* ----------------------------------------------------------- context menu */
  console.log('--- WireContextMenu + MenuManager ---');
  const win = fakeWindow();
  const manager = new MenuManager({ document: doc, window: win });
  let action = null;
  const menu = new WireContextMenu({ document: doc, window: win, menuManager: manager, onAction: (id, model) => { action = { id, model }; } });
  check('the menu panel mounts onto the document', doc.body.contains(menu.panel) && doc.querySelector('.ns-wire-menu') === menu.panel);
  check('the menu panel starts hidden', menu.panel.hidden === true);
  menu.open(base, { clientX: 400, clientY: 300 });
  check('opening goes through MenuManager', manager.isOpen('wire-menu') === true);
  check('the panel is visible after open', menu.panel.hidden === false);
  check('the header shows the decorated title', menu.head.textContent === 'Wire');
  check('the meta line shows both endpoints', menu.meta.textContent.includes('Dataset Source') && menu.meta.textContent.includes('Dense Layer'));
  check('exactly one break button is rendered', menu.itemHost.children.length === 1);
  check('the break button carries the danger class', menu.itemHost.children[0].classList.contains('is-danger'));
  menu.itemHost.children[0].click();
  check('clicking break closes the menu', manager.isOpen('wire-menu') === false);
  check('clicking break reports the action + model', action && action.id === 'break' && action.model.connectionId === chain.first.id);

  menu.open(base, { clientX: 400, clientY: 300 });
  doc.dispatchEvent(new FakeEvent('keydown', { key: 'Escape' }));
  check('Escape closes the wire menu (universal close contract)', manager.isOpen('wire-menu') === false);
  menu.open(base, { clientX: 400, clientY: 300 });
  check('the injected × exists for the wire menu', Boolean(menu.panel.querySelector('.ns-menu-close')));
  menu.panel.querySelector('.ns-menu-close').click();
  check('the explicit × closes the wire menu', manager.isOpen('wire-menu') === false);

  /* ------------------------------------------- end-to-end animated break */
  console.log('--- end-to-end right-click break (animated, real disconnect) ---');
  const breakChain = buildChain();
  const liveBefore = collectWires(breakChain.graph).length;
  const breakScheduler = fakeScheduler();
  const breakPath = new FakeElement('path', doc);
  const wireLayer = {
    querySelector(selector) {
      return selector.includes(breakChain.first.id) ? breakPath : null;
    }
  };
  const brokenIds = [];
  const interactions = new WireInteractions({
    document: doc,
    window: win,
    menuManager: manager,
    wireLayer,
    getWires: () => collectWires(breakChain.graph),
    getScale: () => 1,
    toWorld: (point) => point,
    menuModelFor: (wire, clickHit) => decorateWireMenu(wireMenuBaseModel(wire, clickHit), {
      tier: 2,
      graph: breakChain.graph,
      context: CONTEXT
    }),
    animator: new WireBreakAnimator({ schedule: breakScheduler.schedule, duration: 180 }),
    onBreak: (connectionId) => {
      brokenIds.push(connectionId);
      breakChain.graph.disconnect(connectionId);
    }
  });

  const target = collectWires(breakChain.graph).find((wire) => wire.id === breakChain.first.id).polyline[14];
  const contextEvent = { clientX: target.x, clientY: target.y, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
  const raycastHit = interactions.handleContextMenu(contextEvent);
  check('contextmenu is prevented (no browser menu)', contextEvent.defaultPrevented === true);
  check('the right-click raycast hits the real wire', Boolean(raycastHit && raycastHit.wire.id === breakChain.first.id));
  check('the context menu is open after the right-click', manager.isOpen('wire-menu') === true);
  check('the interactions menu owns the open panel', interactions.menu.active === true);
  check('the wire path is marked before the detach', (() => {
    interactions.menu.itemHost.children[0].click();
    return breakPath.classList.contains('is-breaking');
  })());
  check('the graph still has its wire during the animation', breakChain.graph.connections.size === liveBefore);
  breakScheduler.runAll();
  check('exactly one disconnect was performed', brokenIds.length === 1 && brokenIds[0] === breakChain.first.id, JSON.stringify(brokenIds));
  check('the connection is gone from the model graph', breakChain.graph.connections.has(breakChain.first.id) === false);
  check('the live wire list reflects the deletion', collectWires(breakChain.graph).length === liveBefore - 1);
  check('the animation class is cleaned up after the break', !breakPath.classList.contains('is-breaking'));
  check('re-inference after the cut produces real errors', inferGraph(breakChain.graph, CONTEXT).errors.length > 0);

  console.log('--- miss + graceful path lookup ---');
  const missEvent = { clientX: 5, clientY: 5, preventDefault() {} };
  check('a right-click on empty canvas does not open the menu', interactions.handleContextMenu(missEvent) === null && manager.isOpen('wire-menu') === false);

  const graceful = buildChain();
  const gracefulScheduler = fakeScheduler();
  let gracefulBreak = null;
  const gracefulInteractions = new WireInteractions({
    document: doc,
    window: win,
    menuManager: manager,
    wireLayer: { querySelector: () => null },
    getWires: () => collectWires(graceful.graph),
    getScale: () => 1,
    toWorld: (point) => point,
    animator: new WireBreakAnimator({ schedule: gracefulScheduler.schedule }),
    onBreak: (id) => { gracefulBreak = id; graceful.graph.disconnect(id); }
  });
  gracefulInteractions.requestBreak(graceful.second.id);
  check('a missing SVG path still breaks the graph immediately', gracefulBreak === graceful.second.id && !graceful.graph.connections.has(graceful.second.id));

  manager.destroy();
  doc.dispatchEvent(new FakeEvent('keydown', { key: 'Escape' }));
  check('destroyed manager no longer consumes Escape', manager.isOpen('wire-menu') === false);

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main();
