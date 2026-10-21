/**
 * Step-by-step build missions — real gates, real solvability, overlay contract.
 *
 * Three layers:
 *   1. Curriculum integrity: every mission step has a valid concept from the explanation
 *      catalogue, a real target, and an action gate expressed as a compiler/graph predicate.
 *   2. `MissionRunner` gating: Next refuses until the live predicate passes; `notify` only
 *      satisfies the matching step; prev/skip/progress behave.
 *   3. Solvability simulation: each mission is built end-to-end with the REAL GraphState,
 *      assistConnect and `applyQuickAdd` mutations — every gate must flip false→true from the
 *      action, and the final graphs must compile (Mission 3 is a full regression pipeline).
 *
 * Plus the presenter: `StepByStepEngine` against the fake DOM (spotlight target, explanation
 * modes, gated Next, its own ×, MenuManager Escape, natural completion).
 *
 * File: tests/test_step_by_step.js
 */

import { GraphState } from '../src/graph/graph_state.js';
import { compile, inferGraph } from '../src/graph/node_engine.js';
import { getNodeType } from '../src/graph/node_types.js';
import { MenuManager } from '../src/ui/menu_manager.js';
import { FakeDocument, FakeElement, FakeEvent } from './fake_dom.js';
import {
  MISSIONS,
  MISSION_IDS,
  VALIDATORS,
  evaluateValidation,
  MissionRunner,
  StepByStepEngine
} from '../src/tutorial/step_by_step_engine.js';
import { explain, catalogCoverage } from '../src/tutorial/explanations_catalog.js';
import { applyQuickAdd } from '../src/canvas/quick_node_menu.js';
import {
  intersectionArea,
  isBackdropTarget,
  normalizeRect,
  placeFloatingPanel
} from '../src/tutorial/overlay_layout.js';

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
  const listeners = new Map();
  return {
    innerWidth: 1280,
    innerHeight: 800,
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(handler);
    },
    removeEventListener(type, handler) {
      if (listeners.has(type)) listeners.get(type).delete(handler);
    },
    dispatch(type, event = {}) {
      const handlers = listeners.get(type);
      if (handlers) for (const handler of Array.from(handlers)) handler(event);
    },
    listenerCount(type) {
      return listeners.has(type) ? listeners.get(type).size : 0;
    }
  };
}

const XOR_INFO = { datasetId: 'xor', type: 'classification', name: 'XOR', numFeatures: 2, numClasses: 2, imageDim: null };
const MNIST_INFO = { datasetId: 'mnist_01', type: 'classification', name: 'MNIST 0 vs 1', numFeatures: 784, numClasses: 2, imageDim: { width: 28, height: 28 } };
const REGRESSION_INFO = { datasetId: 'linear_regression', type: 'regression', name: 'Non-Linear Regression', numFeatures: 1, numClasses: 1, imageDim: null };

function infoFor(datasetId) {
  if (datasetId === 'xor') return XOR_INFO;
  if (datasetId === 'linear_regression') return REGRESSION_INFO;
  return MNIST_INFO;
}

function contextFor(graph) {
  return {
    datasetInfoFor: (node) => {
      if (node.type !== 'source') return null;
      return infoFor(node.params.datasetId);
    }
  };
}

function ctxOf(graph, mode, running = false) {
  const context = contextFor(graph);
  return {
    graph,
    mode,
    running,
    plan: compile(graph, context),
    inference: inferGraph(graph, context)
  };
}

function byType(graph, type) {
  return graph.nodeList().filter((node) => node.type === type);
}

function newest(graph, type) {
  const list = byType(graph, type);
  return list.length ? list[list.length - 1] : null;
}

function outDrag(node) {
  return { nodeId: node.id, socket: 'out', direction: 'out', kind: 'tensor' };
}

function inDrag(node, socket, kind) {
  return { nodeId: node.id, socket, direction: 'in', kind };
}

function quick(graph, drag, type, mode, position) {
  const result = applyQuickAdd(graph, drag, type, position, { mode, context: contextFor(graph) });
  if (!result.ok) throw new Error(`quick-add ${type} failed: ${result.error.code}: ${result.error.message}`);
  return result.node;
}

function buildValidXor() {
  const graph = new GraphState('assist');
  const source = graph.addNode('source', { x: 0, y: 0, params: { datasetId: 'xor' } });
  const dense = graph.addNode('dense', { x: 250, y: 0, params: { size: 8 } });
  const relu = graph.addNode('relu', { x: 500, y: 0 });
  const out = graph.addNode('dense', { x: 750, y: 0, params: { size: 2 } });
  const sigmoid = graph.addNode('sigmoid', { x: 1000, y: 0 });
  const loss = graph.addNode('bce_loss', { x: 1250, y: 0 });
  const opt = graph.addNode('adam_optimizer', { x: 1250, y: 260 });
  const loop = graph.addNode('epoch_loop', { x: 1250, y: 420 });
  graph.connect(source.id, 'out', dense.id, 'in');
  graph.connect(dense.id, 'out', relu.id, 'in');
  graph.connect(relu.id, 'out', out.id, 'in');
  graph.connect(out.id, 'out', sigmoid.id, 'in');
  graph.connect(sigmoid.id, 'out', loss.id, 'in');
  graph.connect(opt.id, 'cfg', loss.id, 'opt');
  graph.connect(loop.id, 'cfg', loss.id, 'loop');
  return { graph, source, dense, relu, out, sigmoid, loss, opt, loop };
}

/* ------------------------------------------------------------ simulation */

/**
 * Perform the exact user action each mission step asks for, using the real primitives the page
 * uses. Returns the finished graph (and asserts each gate flips correctly).
 */
function simulateMission(mission) {
  const context = { datasetInfoFor: (node) => (node.type === 'source' ? infoFor(node.params.datasetId) : null) };
  const graph = new GraphState(mission.tier === 3 ? 'manual' : 'assist');
  let mode = 'assist';
  let running = false;
  const journal = [];
  const runner = new MissionRunner(mission);
  const failures = [];
  const pending = [];
  let guard = 0;

  const report = (id, detail = null) => journal.push({ id, seq: journal.length + 1, at: Date.now(), detail });

  const stateCtx = () => ({
    graph,
    mode,
    running,
    actions: journal,
    plan: compile(graph, context),
    inference: inferGraph(graph, context)
  });

  const addSource = (datasetId = 'mnist_01') => graph.addNode('source', { x: 0, y: 0, params: { datasetId } });

  const actions = {
    /* Mission 1 */
    'fb-source': () => addSource(),
    'fb-xor': () => { byType(graph, 'source')[0].params.datasetId = 'xor'; },
    'fb-dense': () => quick(graph, outDrag(byType(graph, 'source')[0]), 'dense', mode, { x: 300, y: 0 }),
    'fb-relu': () => quick(graph, outDrag(newest(graph, 'dense')), 'relu', mode, { x: 600, y: 0 }),
    'fb-dense2': () => quick(graph, outDrag(byType(graph, 'relu')[0]), 'dense', mode, { x: 900, y: 0 }),
    'fb-sigmoid': () => quick(graph, outDrag(newest(graph, 'dense')), 'sigmoid', mode, { x: 1200, y: 0 }),
    'fb-size': () => { newest(graph, 'dense').params.size = 2; },
    'fb-loss': () => quick(graph, outDrag(byType(graph, 'sigmoid')[0]), 'bce_loss', mode, { x: 1500, y: 0 }),
    'fb-opt': () => quick(graph, inDrag(byType(graph, 'bce_loss')[0], 'opt', 'config'), 'adam_optimizer', mode, { x: 1400, y: 320 }),
    'fb-loop': () => quick(graph, inDrag(byType(graph, 'bce_loss')[0], 'loop', 'config'), 'epoch_loop', mode, { x: 1400, y: 480 }),
    'fb-run': () => { running = true; report('training_started'); },

    /* Mission 2 */
    'db-start': () => { mode = 'assist'; },
    'db-source': () => addSource('mnist_01'),
    'db-dense': () => quick(graph, outDrag(byType(graph, 'source')[0]), 'dense', mode, { x: 300, y: 0 }),
    'db-relu': () => quick(graph, outDrag(newest(graph, 'dense')), 'relu', mode, { x: 600, y: 0 }),
    'db-dense2': () => quick(graph, outDrag(byType(graph, 'relu')[0]), 'dense', mode, { x: 900, y: 0 }),
    'db-sigmoid': () => quick(graph, outDrag(newest(graph, 'dense')), 'sigmoid', mode, { x: 1200, y: 0 }),
    'db-size': () => { newest(graph, 'dense').params.size = 2; },
    'db-loss': () => quick(graph, outDrag(byType(graph, 'sigmoid')[0]), 'bce_loss', mode, { x: 1500, y: 0 }),
    'db-snip': () => {
      const relu = byType(graph, 'relu')[0];
      const outgoing = Array.from(graph.connections.values()).find((conn) => conn.from.nodeId === relu.id && conn.to.socket === 'in');
      if (!outgoing) throw new Error('no ReLU → Dense wire to snip');
      graph.disconnect(outgoing.id);
    },
    'db-rewire': () => {
      const sigmoid = byType(graph, 'sigmoid')[0];
      const feeder = graph.incoming(sigmoid.id, 'in');
      const relu = byType(graph, 'relu')[0];
      const result = graph.connect(relu.id, 'out', feeder.from.nodeId, 'in');
      if (!result.ok) throw new Error(`rewire failed: ${result.error.code}`);
    },
    'db-opt': () => quick(graph, inDrag(byType(graph, 'bce_loss')[0], 'opt', 'config'), 'adam_optimizer', mode, { x: 1400, y: 320 }),
    'db-loop': () => quick(graph, inDrag(byType(graph, 'bce_loss')[0], 'loop', 'config'), 'epoch_loop', mode, { x: 1400, y: 480 }),
    'db-run': () => { running = true; report('training_started'); },

    /* Mission 3 */
    'cp-mode': () => { mode = 'manual'; },
    'cp-source': () => addSource('mnist_01'),
    'cp-dataset': () => { byType(graph, 'source')[0].params.datasetId = 'linear_regression'; },
    'cp-flatten': () => quick(graph, outDrag(byType(graph, 'source')[0]), 'flatten', mode, { x: 300, y: 0 }),
    'cp-dense': () => quick(graph, outDrag(byType(graph, 'flatten')[0]), 'dense', mode, { x: 600, y: 0 }),
    'cp-relu': () => quick(graph, outDrag(newest(graph, 'dense')), 'relu', mode, { x: 900, y: 0 }),
    'cp-dense2': () => quick(graph, outDrag(byType(graph, 'relu')[0]), 'dense', mode, { x: 1200, y: 0 }),
    'cp-linear': () => quick(graph, outDrag(newest(graph, 'dense')), 'linear', mode, { x: 1500, y: 0 }),
    'cp-size': () => { newest(graph, 'dense').params.size = 1; },
    'cp-mse': () => quick(graph, outDrag(byType(graph, 'linear')[0]), 'mse_loss', mode, { x: 1800, y: 0 }),
    'cp-snip': () => {
      const linear = byType(graph, 'linear')[0];
      const outgoing = Array.from(graph.connections.values()).find((conn) => conn.from.nodeId === linear.id && conn.to.socket === 'in');
      if (!outgoing) throw new Error('no Linear → MSE wire to snip');
      graph.disconnect(outgoing.id);
    },
    'cp-rewire': () => {
      const result = graph.connect(byType(graph, 'linear')[0].id, 'out', byType(graph, 'mse_loss')[0].id, 'in');
      if (!result.ok) throw new Error(`rewire failed: ${result.error.code}`);
    },
    'cp-opt': () => {
      const loss = byType(graph, 'mse_loss')[0];
      quick(graph, inDrag(loss, 'opt', 'config'), 'rmsprop_optimizer', mode, { x: 1700, y: 320 });
      quick(graph, inDrag(loss, 'loop', 'config'), 'epoch_loop', mode, { x: 1700, y: 480 });
    },
    'cp-schedule': () => quick(graph, inDrag(byType(graph, 'mse_loss')[0], 'schedule', 'config'), 'lr_schedule', mode, { x: 1700, y: 640 }),
    'cp-gauge': () => quick(graph, { nodeId: byType(graph, 'mse_loss')[0].id, socket: 'monitor', direction: 'out', kind: 'monitor' }, 'metrics_gauge', mode, { x: 2100, y: 320 }),
    'cp-run': () => { running = true; report('training_started'); }
  };

  while (!runner.finished && guard < 100) {
    guard += 1;
    const step = runner.current;
    if (!step) break;

    if (step.action) {
      const before = runner.check(stateCtx());
      if (!before.ok) pending.push(step.id);
      try {
        if (actions[step.id]) actions[step.id]();
      } catch (err) {
        failures.push(`${step.id}: mutation threw ${err.message}`);
        break;
      }
      const after = runner.check(stateCtx());
      if (!after.ok) {
        failures.push(`${step.id}: gate still unmet — ${after.reason}`);
        break;
      }
    } else {
      const info = runner.check(stateCtx());
      if (!info.ok) {
        failures.push(`${step.id}: info step reported unmet`);
        break;
      }
    }

    const advance = runner.next();
    if (!advance.advanced) {
      failures.push(`${step.id}: Next refused — ${advance.reason}`);
      break;
    }
  }

  if (!runner.finished) failures.push(`mission did not finish (index ${runner.index}/${runner.total})`);

  return { graph, context, failures, pending, finished: runner.finished, mode, running };
}

function main() {
  console.log('=====================================================');
  console.log('  STEP-BY-STEP MISSIONS — gates, solvability, overlay');
  console.log('=====================================================');

  /* -------------------------------------------------------- curriculum */
  console.log('--- curriculum integrity ---');
  check('exactly three build missions ship', MISSION_IDS.length === 3 && Object.keys(MISSIONS).every((id) => MISSIONS[id].id === id));
  check('missions cover all three tiers', MISSION_IDS.map((id) => MISSIONS[id].tier).sort().join(',') === '1,2,3');
  check('every mission starts from an empty canvas setup', MISSIONS.build_first_brain.steps[0].setup.id === 'prepare_mission'
    && MISSIONS.build_digit_classifier.steps[0].setup.id === 'prepare_mission'
    && MISSIONS.build_custom_pipeline.steps.some((step) => step.setup && step.setup.id === 'prepare_mission'));
  check('mission 1 builds XOR in assist mode', MISSIONS.build_first_brain.steps[0].setup.mode === 'assist');
  check('mission 3 starts in strict manual mode', MISSIONS.build_custom_pipeline.steps.find((step) => step.setup).setup.mode === 'manual');

  const stepIds = new Set();
  let duplicateStep = null;
  let missingConcept = null;
  let missingTarget = null;
  let badRule = null;
  for (const mission of Object.values(MISSIONS)) {
    for (const step of mission.steps) {
      const key = `${mission.id}/${step.id}`;
      if (stepIds.has(key)) duplicateStep = key;
      stepIds.add(key);
      if (!step.concept || !explain(step.concept) || !explain(step.concept).plain) missingConcept = key;
      if (!step.target) missingTarget = key;
      const rules = !step.action || !step.action.validate
        ? []
        : (Array.isArray(step.action.validate) ? step.action.validate : [step.action.validate]);
      for (const rule of rules) {
        if (!VALIDATORS[rule.rule]) badRule = `${key}:${rule.rule}`;
      }
    }
  }
  check('every step id is unique', duplicateStep === null, duplicateStep || '');
  check('every step has a real explanation concept', missingConcept === null, missingConcept || '');
  check('every step highlights a target', missingTarget === null, missingTarget || '');
  check('every gate uses a registered validator rule', badRule === null, badRule || '');
  check('the explanation catalogue covers every concept', catalogCoverage().missing.length === 0, catalogCoverage().missing.join(','));

  const gatedSteps = Object.values(MISSIONS).flatMap((mission) => mission.steps).filter((step) => step.action);
  check('the missions are genuinely action-gated (not click-through)', gatedSteps.length >= 30, String(gatedSteps.length));
  check('mission 2 includes a real snip step', MISSIONS.build_digit_classifier.steps.some((step) => step.id === 'db-snip' && step.action.validate.some((rule) => rule.rule === 'wireBroken')));
  check('mission 3 includes strict regression wiring', MISSIONS.build_custom_pipeline.steps.some((step) => step.id === 'cp-mse'));

  /* --------------------------------------------------------- validators */
  console.log('--- live validators ---');
  const fixture = buildValidXor();
  const ctx = ctxOf(fixture.graph, 'assist');
  check('modeIs reads the live mode', evaluateValidation({ rule: 'modeIs', params: { mode: 'assist' } }, ctx).ok === true);
  check('modeIs rejects the wrong mode', evaluateValidation({ rule: 'modeIs', params: { mode: 'manual' } }, ctx).ok === false);
  check('hasNode finds the real source block', evaluateValidation({ rule: 'hasNode', params: { type: 'source' } }, ctx).ok === true);
  check('hasNode fails for a missing block', evaluateValidation({ rule: 'hasNode', params: { type: 'flatten' } }, ctx).ok === false);
  check('nodeCount counts the two dense layers', evaluateValidation({ rule: 'nodeCount', params: { type: 'dense', min: 2 } }, ctx).ok === true);
  check('nodeCount rejects an unmet minimum', evaluateValidation({ rule: 'nodeCount', params: { type: 'dense', min: 3 } }, ctx).ok === false);
  check('wireExists sees a real connection', evaluateValidation({ rule: 'wireExists', params: { fromType: 'source', toType: 'dense' } }, ctx).ok === true);
  check('wireExists respects socket names', evaluateValidation({ rule: 'wireExists', params: { fromType: 'adam_optimizer', fromSocket: 'cfg', toType: 'bce_loss', toSocket: 'opt' } }, ctx).ok === true);
  check('wireExists fails on a missing wire', evaluateValidation({ rule: 'wireExists', params: { fromType: 'sigmoid', toType: 'relu' } }, ctx).ok === false);
  check('pathExists walks the real chain', evaluateValidation({ rule: 'pathExists', params: { fromType: 'source', toType: 'bce_loss' } }, ctx).ok === true);
  check('pathExists is directional', evaluateValidation({ rule: 'pathExists', params: { fromType: 'bce_loss', toType: 'source' } }, ctx).ok === false);
  check('socketWired sees the optimizer', evaluateValidation({ rule: 'socketWired', params: { nodeType: 'bce_loss', socket: 'opt' } }, ctx).ok === true);
  check('socketWired checks the feeding type', evaluateValidation({ rule: 'socketWired', params: { nodeType: 'bce_loss', socket: 'opt', fromType: 'epoch_loop' } }, ctx).ok === false);
  check('feedingParam sees the output size', evaluateValidation({ rule: 'feedingParam', params: { sourceType: 'dense', targetType: 'sigmoid', param: 'size', value: 2 } }, ctx).ok === true);
  check('feedingParam rejects the wrong value', evaluateValidation({ rule: 'feedingParam', params: { sourceType: 'dense', targetType: 'sigmoid', param: 'size', value: 5 } }, ctx).ok === false);
  check('paramEquals compares coerced numbers', evaluateValidation({ rule: 'paramEquals', params: { type: 'source', param: 'datasetId', value: 'xor' } }, ctx).ok === true);
  check('compiles accepts the buildable fixture', evaluateValidation({ rule: 'compiles' }, ctx).ok === true);
  check('running reads the real run flag', evaluateValidation({ rule: 'running' }, { ...ctx, running: false }).ok === false);
  check('actionSeen reads the persistent action journal',
    evaluateValidation({ rule: 'actionSeen', params: { id: 'training_started' } },
      { ...ctx, actions: [{ id: 'node_added' }, { id: 'training_started' }] }).ok === true);
  check('actionSeen rejects an action that was never logged',
    evaluateValidation({ rule: 'actionSeen', params: { id: 'training_started' } },
      { ...ctx, actions: [{ id: 'node_added' }] }).ok === false);

  const latchMission = MISSIONS.build_first_brain;
  const latchRunner = new MissionRunner(latchMission);
  latchRunner.index = latchMission.steps.findIndex((step) => step.id === 'fb-run');
  const runJournal = [{ id: 'training_started', seq: 1, at: Date.now(), detail: null }];
  const startedCtx = { ...ctxOf(fixture.graph, 'assist', true), actions: runJournal };
  check('the Run gate passes from the action journal', latchRunner.check(startedCtx).ok === true);
  const finishedCtx = { ...startedCtx, running: false };
  check('the Run gate stays satisfied after a fast run finishes', latchRunner.check(finishedCtx).ok === true);
  const advanceAfterRun = latchRunner.next();
  check('the learner can advance to the results step after the run finished',
    advanceAfterRun.advanced === true && latchRunner.current.id === 'fb-done');

  const broken = fixture.graph.clone();
  const sigmoidConn = Array.from(broken.connections.values()).find((conn) => broken.getNode(conn.from.nodeId).type === 'sigmoid');
  broken.disconnect(sigmoidConn.id);
  const brokenCtx = ctxOf(broken, 'assist');
  check('wireBroken sees the severed wire', evaluateValidation({ rule: 'wireBroken', params: { fromType: 'sigmoid', toType: 'bce_loss' } }, brokenCtx).ok === true);
  check('wireBroken rejects an intact wire', evaluateValidation({ rule: 'wireBroken', params: { fromType: 'source', toType: 'dense' } }, ctx).ok === false);
  check('arrays require every rule', evaluateValidation([{ rule: 'hasNode', params: { type: 'dense' } }, { rule: 'hasNode', params: { type: 'flatten' } }], ctx).ok === false);
  check('arrays pass when every rule passes', evaluateValidation([{ rule: 'hasNode', params: { type: 'dense' } }, { rule: 'wireExists', params: { fromType: 'source', toType: 'dense' } }], ctx).ok === true);
  check('an unknown rule fails loudly', evaluateValidation({ rule: 'does_not_exist' }, ctx).ok === false);
  check('compiles fails on a cut graph', evaluateValidation({ rule: 'compiles' }, brokenCtx).ok === false);

  /* ------------------------------------------------------- runner gating */
  console.log('--- MissionRunner gating ---');
  const runner = new MissionRunner(MISSIONS.build_first_brain);
  const emptyGraph = new GraphState('assist');
  const emptyCtx = ctxOf(emptyGraph, 'assist');
  check('the mission opens on the intro step', runner.index === 0 && runner.current.id === 'fb-start');
  check('the intro step can be advanced freely', runner.check(emptyCtx).ok === true && runner.next().advanced === true);
  check('the next step is the source placement', runner.current.id === 'fb-source');
  check('Next is refused while the graph has no source', (() => {
    const result = runner.next();
    return result.advanced === false && runner.index === 1;
  })());
  check('an unrelated event does not satisfy a predicate step', runner.notify('preset_loaded', emptyCtx).ok === false);
  check('a matching event with an unmet predicate still refuses', runner.notify('node_added', emptyCtx).ok === false);
  emptyGraph.addNode('source', { x: 0, y: 0 });
  check('the gate flips once the real block exists', runner.notify('node_added', ctxOf(emptyGraph, 'assist')).ok === true);
  check('Next advances after the verified action', runner.next().advanced === true);
  check('the dataset step is now active', runner.current.id === 'fb-xor');
  check('progress reports the truthful position', runner.progress().index === 2 && runner.progress().total === 13);
  check('prev moves back without losing the satisfied gate', runner.prev().moved === true && runner.index === 1);
  check('the previously completed step still allows Next', runner.next().advanced === true && runner.index === 2);
  check('skip refuses a required step', runner.skip().advanced === false);
  const optionalRunner = new MissionRunner({ id: 'synthetic', steps: [{ id: 'one', title: 'optional', text: '', target: '#x', concept: 'wiring', optional: true }, { id: 'two', title: 'two', text: '', target: '#x', concept: 'wiring' }] });
  check('skip advances an optional step', optionalRunner.skip().advanced === true && optionalRunner.index === 1);

  /* ----------------------------------------------------- solvability */
  console.log('--- full solvability (real graph builds) ---');
  const first = simulateMission(MISSIONS.build_first_brain);
  check('Mission 1 completes every gate with real actions', first.finished && first.failures.length === 0, first.failures.join(' | '));
  check('Mission 1 refused the gates before the actions', first.pending.length >= 10, String(first.pending.length));
  check('Mission 1 produced a running XOR graph', first.running === true && compile(first.graph, first.context).ok === true);

  const second = simulateMission(MISSIONS.build_digit_classifier);
  check('Mission 2 completes every gate with real actions', second.finished && second.failures.length === 0, second.failures.join(' | '));
  check('Mission 2 really cut and re-wired a wire', second.failures.length === 0
    && second.graph.incoming(byType(second.graph, 'sigmoid')[0].id, 'in') !== null
    && byType(second.graph, 'flatten').length === 1);
  check('Mission 2 produced a compile-valid MNIST classifier', compile(second.graph, second.context).ok === true);
  const secondPlan = compile(second.graph, second.context);
  check('the MNIST classifier has the trained shape 784→…→2', secondPlan.meta.inputShape.join('×') === '28×28' && secondPlan.meta.outputSize === 2, JSON.stringify(secondPlan.meta));

  const third = simulateMission(MISSIONS.build_custom_pipeline);
  check('Mission 3 completes every gate with real actions', third.finished && third.failures.length === 0, third.failures.join(' | '));
  check('Mission 3 ran in strict manual mode', third.mode === 'manual');
  const thirdPlan = compile(third.graph, third.context);
  check('Mission 3 produced a compile-valid regression pipeline', thirdPlan.ok === true, thirdPlan.errors.map((issue) => issue.code).join(','));
  check('the regression pipeline ends in one real output', thirdPlan.ok && thirdPlan.meta.outputSize === 1, thirdPlan.ok ? JSON.stringify(thirdPlan.meta) : '');
  check('the regression pipeline is configured with MSE + RMSprop', thirdPlan.ok && thirdPlan.network.loss === 'mse' && thirdPlan.network.optimizer === 'rmsprop');
  check('the pipeline carries a real LR schedule', thirdPlan.ok && thirdPlan.training.lrSchedule !== null);
  check('the monitor tap is part of the compiled plan', thirdPlan.ok && thirdPlan.taps.gauges.length === 1);

  /* ------------------------------------------------------ overlay geometry */
  console.log('--- overlay geometry ---');
  const paletteRect = { left: 0, top: 80, width: 236, height: 600, right: 236, bottom: 680 };
  const rightOfPalette = placeFloatingPanel({
    target: paletteRect,
    width: 384,
    height: 460,
    viewportWidth: 1280,
    viewportHeight: 800,
    softAvoid: [paletteRect, { left: 0, top: 0, width: 1280, height: 56, right: 1280, bottom: 56 }]
  });
  check('a palette anchor pushes the card onto the canvas, not over the shelf',
    rightOfPalette.side === 'right' && rightOfPalette.targetOverlap === 0 && rightOfPalette.chromeOverlap === 0,
    JSON.stringify(rightOfPalette));
  check('the palette-adjacent card is inside the viewport',
    rightOfPalette.x >= 12 && rightOfPalette.y >= 12
    && rightOfPalette.x + rightOfPalette.width <= 1268
    && rightOfPalette.y + rightOfPalette.height <= 788,
    JSON.stringify(rightOfPalette));

  const inspectorRect = { left: 950, top: 60, width: 320, height: 700, right: 1270, bottom: 760 };
  const leftOfInspector = placeFloatingPanel({
    target: inspectorRect,
    width: 384,
    height: 460,
    viewportWidth: 1280,
    viewportHeight: 800,
    softAvoid: [inspectorRect]
  });
  check('an inspector anchor pushes the card onto the canvas, not over the inspector',
    leftOfInspector.side === 'left' && leftOfInspector.targetOverlap === 0,
    JSON.stringify(leftOfInspector));

  const backdrop = { left: 236, top: 56, width: 720, height: 690, right: 956, bottom: 746 };
  const backdropPlacement = placeFloatingPanel({
    target: backdrop,
    width: 384,
    height: 460,
    viewportWidth: 1280,
    viewportHeight: 800,
    hardAvoid: false,
    softAvoid: [paletteRect, inspectorRect]
  });
  check('a full-canvas intro step docks the card over the workspace, off the fixed panels',
    backdropPlacement.chromeOverlap === 0
    && backdropPlacement.x >= 12 && backdropPlacement.y >= 12
    && backdropPlacement.x + backdropPlacement.width <= 1268
    && backdropPlacement.y + backdropPlacement.height <= 788,
    JSON.stringify(backdropPlacement));
  check('backdrop detection uses the viewport area',
    isBackdropTarget({ left: 0, top: 0, width: 1000, height: 700 }, 1280, 800) === true
    && isBackdropTarget({ left: 0, top: 0, width: 200, height: 200 }, 1280, 800) === false);
  check('viewport rect normalisation accepts right/bottom-only rects',
    normalizeRect({ left: 10, top: 20, right: 110, bottom: 120 }).width === 100
    && normalizeRect({ left: 10, top: 20, right: 110, bottom: 120 }).height === 100);
  check('intersection maths is exact',
    intersectionArea({ left: 0, top: 0, width: 50, height: 50 }, { left: 25, top: 25, width: 50, height: 50 }) === 625
    && intersectionArea({ left: 0, top: 0, width: 50, height: 50 }, { left: 60, top: 60, width: 20, height: 20 }) === 0);

  /* --------------------------------------------------------- presenter */
  console.log('--- StepByStepEngine overlay ---');
  const doc = new FakeDocument();
  const win = fakeWindow();
  const container = new FakeElement('div', doc);
  const runButton = new FakeElement('button', doc);
  runButton.id = 'btn-run';
  const canvasTarget = new FakeElement('div', doc);
  const targets = { '#btn-run': runButton, '#studio-canvas': canvasTarget };
  let completed = 0;
  let setups = 0;
  const missions = new StepByStepEngine({
    document: doc,
    window: win,
    container,
    resolveTarget: (target) => (typeof target === 'string' ? targets[target] || null : null),
    onSetup: () => { setups++; },
    onComplete: () => { completed++; }
  });
  const manager = new MenuManager({ document: doc, window: win });
  manager.registerOverlay({
    id: 'mission-overlay',
    isOpen: () => missions.active,
    requestClose: () => missions.finish('escape')
  });

  // The overlay gets its own empty graph: the runner-gating fixture above already gained a source.
  const overlayGraph = new GraphState('assist');
  const overlayCtx = () => ctxOf(overlayGraph, 'assist');

  check('the overlay starts inactive', missions.active === false);
  missions.start('build_first_brain', overlayCtx());
  check('starting a mission activates the overlay', missions.active === true && missions.root.hidden === false);
  check('the run setup command fired once', setups === 1);
  check('the overlay exposes the mission tier', missions.tier === 1 && missions.suppressAssist === true);
  check('the progress starts at step 1', missions.progressText.textContent === 'Step 1 of 13', missions.progressText.textContent);
  check('the explanation card is populated', missions.explainPlain.textContent.length > 20 && missions.explainChip.textContent.length > 0);
  check('the current target is highlighted for the learner', canvasTarget.classList.contains('is-mission-target'));

  const cardRect = () => ({
    left: parseFloat(missions.card.style.left) || 0,
    top: parseFloat(missions.card.style.top) || 0,
    width: parseFloat(missions.card.style.width) || 384,
    height: missions.card.offsetHeight || 460
  });
  canvasTarget.setBoundingRect({ left: 40, top: 280, right: 240, bottom: 400 });
  missions.refreshPosition();
  check('moving a target re-anchors the spotlight and pulse ring',
    parseFloat(missions.spotlight.style.left) === 34
    && parseFloat(missions.spotlight.style.top) === 274
    && parseFloat(missions.ring.style.left) === 34
    && parseFloat(missions.ring.style.top) === 274,
    `${missions.spotlight.style.left},${missions.spotlight.style.top}`);
  check('the card avoids the moved target',
    intersectionArea(cardRect(), { left: 40, top: 280, width: 200, height: 120 }) === 0,
    JSON.stringify(cardRect()));
  const firstCardLeft = cardRect().left;
  canvasTarget.setBoundingRect({ left: 300, top: 100, right: 500, bottom: 300 });
  missions.refreshPosition();
  check('the card dodges a target dragged underneath it',
    cardRect().left !== firstCardLeft
    && intersectionArea(cardRect(), { left: 300, top: 100, width: 200, height: 200 }) === 0,
    `${firstCardLeft} -> ${cardRect().left}`);
  const stableCardLeft = cardRect().left;
  canvasTarget.setBoundingRect({ left: 60, top: 600, right: 200, bottom: 700 });
  win.dispatch('scroll');
  check('captured scroll re-anchors the spotlight', parseFloat(missions.spotlight.style.left) === 54
    && parseFloat(missions.spotlight.style.top) === 594,
    `${missions.spotlight.style.left},${missions.spotlight.style.top}`);
  check('scroll leaves a still-valid card stable', cardRect().left === stableCardLeft,
    `${stableCardLeft} -> ${cardRect().left}`);
  missions.card.offsetHeight = 700;
  missions.refreshPosition();
  check('a taller card stays fully inside the viewport',
    cardRect().top >= 12 && cardRect().top + cardRect().height <= 788,
    JSON.stringify(cardRect()));
  missions.card.offsetHeight = 460;
  win.innerWidth = 700;
  win.dispatch('resize');
  check('resize re-places the card inside the smaller viewport',
    cardRect().left >= 12 && cardRect().left + cardRect().width <= 688
    && cardRect().top >= 12 && cardRect().top + cardRect().height <= 788,
    JSON.stringify(cardRect()));
  win.innerWidth = 1280;
  win.dispatch('resize');

  check('the technical detail starts collapsed', !missions.explainBox.classList.contains('is-tech-open'));
  missions.techToggle.click();
  check('the technical toggle expands the detail', missions.explainBox.classList.contains('is-tech-open'));
  missions.techToggle.click();
  check('the technical toggle collapses again', !missions.explainBox.classList.contains('is-tech-open'));

  const advance = missions.next(overlayCtx());
  check('the intro step advances the overlay', advance.advanced === true && missions.runner.index === 1);
  check('the previous target highlight is cleared', !canvasTarget.classList.contains('is-mission-target'));
  const blocked = missions.next(overlayCtx());
  check('the overlay refuses Next before the action', blocked.advanced === false && missions.runner.index === 1);
  check('the Next button is disabled while pending', missions.nextBtn.disabled === true);
  check('the hint explains the pending action', !missions.hint.classList.contains('is-met') && missions.hint.textContent.length > 0);

  overlayGraph.addNode('source', { x: 0, y: 0 });
  const notified = missions.notify('node_added', overlayCtx());
  check('notify satisfies the real predicate', notified.known === true && notified.ok === true);
  check('the Next button enables after verification', missions.nextBtn.disabled === false);
  check('the hint flips to the verified state', missions.hint.classList.contains('is-met'));
  missions.nextBtn.click();
  check('clicking Next moves to the dataset step', missions.runner.current.id === 'fb-xor');
  check('Back returns to the previous step', missions.backBtn.disabled === false && (missions.prev(overlayCtx()).moved === true) && missions.runner.current.id === 'fb-source');

  const overlayX = missions.card.children.find((child) => child.className === 'mission-close');
  check('the mission card has its own explicit ×', Boolean(overlayX) && overlayX.textContent === '×');
  overlayX.click();
  check('× exits the mission and hides the overlay', missions.active === false && missions.root.hidden === true);
  check('exiting unregisters the scroll tracking listener', win.listenerCount('scroll') === 0,
    String(win.listenerCount('scroll')));

  missions.start('build_first_brain', overlayCtx());
  doc.dispatchEvent(new FakeEvent('keydown', { key: 'Escape' }));
  check('Escape exits the registered mission overlay', missions.active === false);
  check('Escape also unregisters the scroll tracking listener', win.listenerCount('scroll') === 0,
    String(win.listenerCount('scroll')));

  missions.start('build_first_brain', overlayCtx());
  missions.runner.index = missions.runner.total - 1;
  missions.check(overlayCtx());
  const finished = missions.next(overlayCtx());
  check('finishing the last step completes the mission', finished.finished === true && completed === 1);
  check('completion hides the overlay and marks it inactive', missions.active === false && missions.root.hidden === true);

  manager.destroy();

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main();
