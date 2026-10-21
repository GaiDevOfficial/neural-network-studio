/**
 * Node Studio — graph engine verification suite (pure logic, no DOM).
 *
 * Covers: block registry integrity vs the real engine, GraphState wiring invariants,
 * tensor shape inference, the loss-pairing matrix, compile plans, Smart Assist repairs
 * (including a malformed-graph fuzz pass that must never throw), presets, geometry,
 * the scheduler/class-filter math and the tutorial state machine.
 *
 * File: tests/test_graph_engine.js
 */

import { NODE_TYPES, CATEGORY_META, paletteSections, defaultParams, ACTIVATION_NODE_TYPES, LOSS_NODE_TYPES, OPTIMIZER_NODE_TYPES } from '../src/graph/node_types.js';
import { GraphState } from '../src/graph/graph_state.js';
import { inferGraph, compile, applyClassFilter, schedulerValue, parseKeepList } from '../src/graph/node_engine.js';
import { assistConnect, autoFixGraph, autoFixGraphToConvergence, autoWireOnDrop, instantPreset } from '../src/graph/smart_assist.js';
import { socketsOf, socketPosition, bezierPath, pointOnBezier, screenToWorld, worldToScreen, zoomViewAt, hitTestSocket, NODE_WIDTH } from '../src/graph/geometry.js';
import { TutorialRunner, TOURS } from '../src/ui/tutorial_engine.js';
import { inlineMarkdown } from '../src/tutorial/inline_text.js';
import { Activations } from '../src/engine/activations.js';
import { LossFunctions } from '../src/engine/loss_functions.js';

let passed = 0;
let failed = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  [PASS] ${name}`);
  } else {
    failed++;
    console.log(`  [FAIL] ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function section(title) {
  console.log(`\n-----------------------------------------------------`);
  console.log(`  ${title}`);
  console.log(`-----------------------------------------------------`);
}

/* ------------------------------------------------------------- fixtures */

const MNIST_INFO = { type: 'classification', numFeatures: 784, numClasses: 2, imageDim: { width: 28, height: 28 }, name: 'MNIST 0 vs 1' };
const XOR_INFO = { type: 'classification', numFeatures: 2, numClasses: 2, imageDim: null, name: 'XOR' };
const IRIS_INFO = { type: 'classification', numFeatures: 4, numClasses: 3, imageDim: null, name: 'Iris' };
const REG_INFO = { type: 'regression', numFeatures: 1, numClasses: 1, imageDim: null, name: 'Regression' };

const INFO = { mnist_01: MNIST_INFO, xor: XOR_INFO, iris: IRIS_INFO, linear_regression: REG_INFO };
const ctx = { datasetInfoFor: (node) => INFO[node.params.datasetId] || null };

/* ================================================================= */
console.log('=====================================================');
console.log('   NODE STUDIO — GRAPH ENGINE VERIFICATION SUITE     ');
console.log('=====================================================');

section('PART 1 — Block registry integrity');

{
  const missing = Object.entries(NODE_TYPES).filter(([, def]) => !CATEGORY_META[def.category]);
  check('every block has a declared palette category', missing.length === 0, missing.map(([t]) => t).join(', '));

  const badSockets = [];
  for (const [type, def] of Object.entries(NODE_TYPES)) {
    const sockets = [...(def.inputs || []), ...(def.outputs || [])];
    for (const socket of sockets) {
      if (!socket.name || !['tensor', 'config', 'monitor'].includes(socket.kind)) badSockets.push(`${type}.${socket.name}`);
    }
  }
  check('every socket is well-formed (name + kind)', badSockets.length === 0, badSockets.join(', '));

  const badParams = [];
  for (const [type, def] of Object.entries(NODE_TYPES)) {
    for (const [name, spec] of Object.entries(def.params || {})) {
      if (spec.default === undefined) badParams.push(`${type}.${name}`);
      if ((spec.type === 'int' || spec.type === 'float') && (!Number.isFinite(spec.default))) badParams.push(`${type}.${name}:bad default`);
      if (spec.type === 'int' && !Number.isInteger(spec.default)) badParams.push(`${type}.${name}:not int`);
    }
  }
  check('every parameter has a valid default', badParams.length === 0, badParams.join(', '));

  const badActivations = ACTIVATION_NODE_TYPES.filter((t) => !Activations[NODE_TYPES[t].activation]);
  check('activation blocks map to real engine activations', badActivations.length === 0, badActivations.join(', '));

  const badLosses = LOSS_NODE_TYPES.filter((t) => !LossFunctions[NODE_TYPES[t].loss]);
  check('loss blocks map to real engine losses', badLosses.length === 0, badLosses.join(', '));

  const badOptimizers = OPTIMIZER_NODE_TYPES.filter((t) => !['adam', 'sgd', 'momentum', 'rmsprop'].includes(NODE_TYPES[t].optimizer));
  check('optimizer blocks map to engine optimizers', badOptimizers.length === 0, badOptimizers.join(', '));

  const { sections, planned } = paletteSections();
  check('palette exposes all categories', sections.length === Object.keys(CATEGORY_META).length, `${sections.length} sections`);
  check('planned blocks are conv2d/maxpool/dropout with reasons', planned.length === 3 && planned.every((p) => p.reason && p.reason.length > 20), planned.map((p) => p.type).join(','));

  const clamped = defaultParams('dense', { size: 99999 });
  check('parameter clamping enforces declared ranges', clamped.size === 512, `got ${clamped.size}`);
}

section('PART 2 — GraphState wiring invariants');

{
  const graph = new GraphState('assist');
  const source = graph.addNode('source');
  const dense = graph.addNode('dense');

  const ok = graph.connect(source.id, 'out', dense.id, 'in');
  check('a valid tensor wire connects', ok.ok === true);

  const dup = graph.connect(source.id, 'out', dense.id, 'in');
  check('duplicate wire into an input is refused (E_DUP_INPUT)', !dup.ok && dup.error.code === 'E_DUP_INPUT');

  const loss = graph.addNode('bce_loss');
  const wrongKind = graph.connect(source.id, 'out', loss.id, 'opt');
  check('kind mismatch is refused (E_KIND_MISMATCH)', !wrongKind.ok && wrongKind.error.code === 'E_KIND_MISMATCH');

  const unknownRefused = (() => { try { graph.addNode('does_not_exist'); return false; } catch (e) { return true; } })();
  check('unknown block types are refused loudly', unknownRefused === true);

  const activation = graph.addNode('relu');
  graph.connect(dense.id, 'out', activation.id, 'in');

  // Cycle probe: Q → P is legal (P's input is free); closing P → Q would create a loop.
  const p = graph.addNode('dense', { params: { size: 4 } });
  const q = graph.addNode('dense', { params: { size: 4 } });
  graph.connect(q.id, 'out', p.id, 'in');
  const cycle = graph.connect(p.id, 'out', q.id, 'in');
  check('cycle-creating wires are refused (E_CYCLE)', !cycle.ok && cycle.error.code === 'E_CYCLE', cycle.error ? cycle.error.code : 'no error');

  const selfLoop = graph.connect(source.id, 'out', source.id, 'out');
  check('self-loop is refused', !selfLoop.ok, JSON.stringify(selfLoop));

  const before = graph.nodes.size;
  graph.removeNode(activation.id);
  check('removing a node removes its wires', graph.nodes.size === before - 1 && graph.outgoing(dense.id).every((c) => c.to.nodeId !== activation.id));

  const order = graph.topologicalOrder();
  check('topological order puts the source before its consumer', order.indexOf(source.id) < order.indexOf(dense.id));

  const clone = GraphState.fromJSON(graph.toJSON());
  check('JSON round-trip preserves nodes and wires',
    clone.nodes.size === graph.nodes.size && clone.connections.size === graph.connections.size);

  const dropped = GraphState.fromJSON({
    mode: 'assist',
    nodes: [{ id: 'a', type: 'source', x: 0, y: 0, params: {} }, { id: 'b', type: 'warp_drive', x: 0, y: 0, params: {} }],
    connections: [{ id: 'c1', from: { nodeId: 'a', socket: 'out' }, to: { nodeId: 'b', socket: 'in' } }]
  });
  check('fromJSON drops unknown blocks and reports it', dropped.nodes.size === 1 && dropped.loadReport.droppedNodes.length === 1);
}

section('PART 3 — Tensor shape inference & loss pairing');

{
  // Source shape: image grid vs feature vector.
  const g1 = new GraphState('manual');
  const s1 = g1.addNode('source', { params: { datasetId: 'mnist_01' } });
  const r1 = inferGraph(g1, ctx);
  check('image source emits a [28,28] grid', JSON.stringify(r1.shapes.get(s1.id)) === '[28,28]', JSON.stringify(r1.shapes.get(s1.id)));

  const g2 = new GraphState('manual');
  const s2 = g2.addNode('source', { params: { datasetId: 'xor' } });
  const r2 = inferGraph(g2, ctx);
  check('tabular source emits a [2] vector', JSON.stringify(r2.shapes.get(s2.id)) === '[2]');

  // Dense directly on an image → rank mismatch quoting the fix.
  const g3 = new GraphState('manual');
  const s3 = g3.addNode('source', { params: { datasetId: 'mnist_01' } });
  const d3 = g3.addNode('dense', { params: { size: 16 } });
  g3.connect(s3.id, 'out', d3.id, 'in');
  const r3 = inferGraph(g3, ctx);
  const rankErr = r3.errors.find((e) => e.code === 'E_RANK_MISMATCH');
  check('Dense on a rank-2 tensor is an E_RANK_MISMATCH quoting the fix', Boolean(rankErr && rankErr.message.includes('Flatten')), rankErr ? rankErr.message : 'no error');

  // Flatten resolves it: [28,28] → [784].
  const f3 = g3.addNode('flatten');
  g3.disconnect(g3.incoming(d3.id, 'in').id);
  g3.connect(s3.id, 'out', f3.id, 'in');
  g3.connect(f3.id, 'out', d3.id, 'in');
  const r3b = inferGraph(g3, ctx);
  check('Flatten resolves [28,28] → [784]', JSON.stringify(r3b.shapes.get(f3.id)) === '[784]');
  check('Dense then emits [16]', JSON.stringify(r3b.shapes.get(d3.id)) === '[16]');

  // Softmax requires rank 1.
  const g4 = new GraphState('manual');
  const s4 = g4.addNode('source', { params: { datasetId: 'mnist_01' } });
  const sm4 = g4.addNode('softmax');
  g4.connect(s4.id, 'out', sm4.id, 'in');
  const r4 = inferGraph(g4, ctx);
  check('Softmax on a grid is refused (rank)', r4.errors.some((e) => e.code === 'E_RANK_MISMATCH' && e.nodeId === sm4.id));

  // Loss pairing matrix.
  const makeChain = (datasetId, sizes, acts, lossType, { connectOpt = true } = {}) => {
    const g = new GraphState('manual');
    const source = g.addNode('source', { params: { datasetId } });
    let prev = { id: source.id, socket: 'out' };
    sizes.forEach((size, i) => {
      const dense = g.addNode('dense', { params: { size } });
      g.connect(prev.id, prev.socket, dense.id, 'in');
      const act = g.addNode(acts[i]);
      g.connect(dense.id, 'out', act.id, 'in');
      prev = { id: act.id, socket: 'out' };
    });
    const loss = g.addNode(lossType);
    g.connect(prev.id, prev.socket, loss.id, 'in');
    const opt = g.addNode('adam_optimizer');
    if (connectOpt) g.connect(opt.id, 'cfg', loss.id, 'opt');
    const loop = g.addNode('epoch_loop');
    g.connect(loop.id, 'cfg', loss.id, 'loop');
    return { g, source, loss, opt, loop };
  };

  const bceNoSigmoid = makeChain('xor', [8, 2], ['relu', 'relu'], 'bce_loss');
  const rBce = inferGraph(bceNoSigmoid.g, ctx);
  check('BCE without Sigmoid is an E_PAIRING', rBce.errors.some((e) => e.code === 'E_PAIRING' && e.message.includes('Sigmoid')));

  const bceBad = makeChain('xor', [8, 3], ['relu', 'sigmoid'], 'bce_loss');
  const rBadSize = inferGraph(bceBad.g, ctx);
  check('BCE with a 3-value output is an E_SIZE_MISMATCH', rBadSize.errors.some((e) => e.code === 'E_SIZE_MISMATCH'));

  const cceNoSoftmax = makeChain('iris', [12, 3], ['relu', 'relu'], 'cce_loss');
  const rCce = inferGraph(cceNoSoftmax.g, ctx);
  check('CCE without Softmax is an E_PAIRING', rCce.errors.some((e) => e.code === 'E_PAIRING' && e.message.includes('Softmax')));

  const good = makeChain('iris', [12, 3], ['relu', 'softmax'], 'cce_loss');
  const rGood = inferGraph(good.g, ctx);
  check('Iris 12→3 Softmax + CCE validates clean', rGood.errors.length === 0, rGood.errors.map((e) => e.code).join(','));

  const plan = compile(good.g, ctx);
  check('compile extracts the fused architecture', plan.ok && JSON.stringify(plan.network.architecture) === JSON.stringify([{ size: 4 }, { size: 12, activation: 'relu' }, { size: 3, activation: 'softmax' }]), JSON.stringify(plan.errors));
  check('compile maps optimizer + hyperparameters', plan.ok && plan.network.optimizer === 'adam' && plan.network.learningRate === 0.03);
  check('compile reads the epoch-loop config', plan.ok && plan.training.mode === 'auto_stop' && plan.training.maxEpochs === null && plan.training.earlyStopPatience === 150 && plan.training.batchSize === 16);

  const noOpt = makeChain('xor', [8, 2], ['relu', 'sigmoid'], 'bce_loss', { connectOpt: false });
  const planNoOpt = compile(noOpt.g, ctx);
  check('missing optimizer blocks compile (E_MISSING_CONFIG)', !planNoOpt.ok && planNoOpt.errors.some((e) => e.code === 'E_MISSING_CONFIG'), JSON.stringify(planNoOpt.errors.map((e) => e.code)));

  // Class filter semantics.
  const filtered = makeChain('iris', [12, 2], ['relu', 'sigmoid'], 'bce_loss');
  const filter = filtered.g.addNode('class_filter', { params: { keep: '0,1' } });
  const denseFirst = filtered.g.nodeList().find((n) => n.type === 'dense');
  const firstConn = filtered.g.incoming(denseFirst.id, 'in');
  filtered.g.disconnect(firstConn.id);
  filtered.g.connect(filter.id, 'out', denseFirst.id, 'in');
  filtered.g.connect(firstConn.from.nodeId, firstConn.from.socket, filter.id, 'in');
  const rFilter = inferGraph(filtered.g, ctx);
  check('Class Filter reduces classes for loss checks (3 → 2)', rFilter.errors.length === 0, rFilter.errors.map((e) => e.message).join(' | '));
  const planFiltered = compile(filtered.g, ctx);
  check('compiled plan carries the class filter', planFiltered.ok && planFiltered.data.filter && JSON.stringify(planFiltered.data.filter.keep) === '[0,1]');

  const badFilter = makeChain('iris', [12, 2], ['relu', 'sigmoid'], 'bce_loss');
  const bf = badFilter.g.addNode('class_filter', { params: { keep: '0,9' } });
  const bfDense = badFilter.g.nodeList().find((n) => n.type === 'dense');
  const bfConn = badFilter.g.incoming(bfDense.id, 'in');
  badFilter.g.disconnect(bfConn.id);
  badFilter.g.connect(bf.id, 'out', bfDense.id, 'in');
  badFilter.g.connect(bfConn.from.nodeId, bfConn.from.socket, bf.id, 'in');
  const rBadFilter = inferGraph(badFilter.g, ctx);
  check('Class Filter out-of-range index is E_PARAM_RANGE', rBadFilter.errors.some((e) => e.code === 'E_PARAM_RANGE' && e.nodeId === bf.id));

  // Unresolved data.
  const gUnresolved = new GraphState('manual');
  const sU = gUnresolved.addNode('source', { params: { datasetId: 'custom_csv' } });
  const rU = inferGraph(gUnresolved, ctx);
  check('unresolved source info is E_DATA_UNRESOLVED', rU.errors.some((e) => e.code === 'E_DATA_UNRESOLVED' && e.nodeId === sU.id));
}

section('PART 4 — Smart Assist (Beginner Mode)');

{
  // assistConnect inserts a Flatten between an image source and a Dense.
  const g = new GraphState('assist');
  const source = g.addNode('source', { params: { datasetId: 'mnist_01' } });
  const dense = g.addNode('dense', { params: { size: 16 } });
  const res = assistConnect(g, { nodeId: source.id, socket: 'out' }, { nodeId: dense.id, socket: 'in' }, ctx);
  const flattens = g.nodeList().filter((n) => n.type === 'flatten');
  check('assistConnect auto-inserts Flatten for rank-2 → Dense', res.ok && flattens.length === 1, JSON.stringify(res.actions));
  check('the repair is logged for the user', res.actions.some((a) => a.code === 'A_INSERT_FLATTEN'));

  // A wrong-kind assist connect is refused, not silently fixed.
  const gKind = new GraphState('assist');
  const opt = gKind.addNode('adam_optimizer');
  const lossK = gKind.addNode('bce_loss');
  const kindRes = assistConnect(gKind, { nodeId: opt.id, socket: 'cfg' }, { nodeId: lossK.id, socket: 'in' }, ctx);
  check('assistConnect refuses kind mismatches', !kindRes.ok && kindRes.error.code === 'E_KIND_MISMATCH');

  // autoFixGraph on an empty graph: fabricates nothing data-wise, reports the rest.
  const empty = new GraphState('assist');
  const fixedEmpty = autoFixGraphToConvergence(empty, ctx);
  check('autoFixGraph on an empty canvas adds a default source', empty.nodeList().some((n) => n.type === 'source'));
  check('autoFixGraph adds loss + optimizer + loop defaults', ['bce_loss', 'mse_loss', 'cce_loss'].some((t) => empty.nodeList().some((n) => n.type === t)) && empty.nodeList().some((n) => OPTIMIZER_NODE_TYPES.includes(n.type)) && empty.nodeList().some((n) => n.type === 'epoch_loop'));
  check('autoFixGraph never throws and returns an action log', Array.isArray(fixedEmpty.actions) && fixedEmpty.remainingErrors.every((e) => e.code !== 'E_ASSIST_INTERNAL'));

  // A graph missing only the optimizer gets it attached, then compiles.
  const gMissOpt = new GraphState('assist');
  const sM = gMissOpt.addNode('source', { params: { datasetId: 'xor' } });
  const dM1 = gMissOpt.addNode('dense', { params: { size: 8 } });
  const aM1 = gMissOpt.addNode('relu');
  const dM2 = gMissOpt.addNode('dense', { params: { size: 2 } });
  const aM2 = gMissOpt.addNode('sigmoid');
  const lM = gMissOpt.addNode('bce_loss');
  const loopM = gMissOpt.addNode('epoch_loop');
  gMissOpt.connect(sM.id, 'out', dM1.id, 'in');
  gMissOpt.connect(dM1.id, 'out', aM1.id, 'in');
  gMissOpt.connect(aM1.id, 'out', dM2.id, 'in');
  gMissOpt.connect(dM2.id, 'out', aM2.id, 'in');
  gMissOpt.connect(aM2.id, 'out', lM.id, 'in');
  gMissOpt.connect(loopM.id, 'cfg', lM.id, 'loop');
  const fix1 = autoFixGraphToConvergence(gMissOpt, ctx);
  const plan1 = compile(gMissOpt, ctx);
  check('missing optimizer is auto-attached and compiles clean', plan1.ok && fix1.actions.some((a) => a.code === 'A_ADD_OPTIMIZER'), JSON.stringify(plan1.errors.map((e) => e.message)));

  // Wrong activation before BCE → replaced by Sigmoid; wrong output size → resized to 2.
  const gWrong = new GraphState('assist');
  const sW = gWrong.addNode('source', { params: { datasetId: 'xor' } });
  const dW = gWrong.addNode('dense', { params: { size: 5 } });
  const aW = gWrong.addNode('relu');
  const lW = gWrong.addNode('bce_loss');
  gWrong.connect(sW.id, 'out', dW.id, 'in');
  gWrong.connect(dW.id, 'out', aW.id, 'in');
  gWrong.connect(aW.id, 'out', lW.id, 'in');
  autoFixGraphToConvergence(gWrong, ctx);
  const planW = compile(gWrong, ctx);
  check('assist replaces a wrong activation with Sigmoid', planW.ok && planW.network.architecture[1].activation === 'sigmoid', JSON.stringify(planW.errors.map((e) => e.message)));
  check('assist resizes the output Dense to the loss requirement (2)', planW.ok && planW.network.architecture[1].size === 2);

  // Two activations in a row: the earlier one is removed (last wins, no silent override).
  const gTwo = new GraphState('assist');
  const s2 = gTwo.addNode('source', { params: { datasetId: 'xor' } });
  const d2 = gTwo.addNode('dense', { params: { size: 2 } });
  const a2a = gTwo.addNode('tanh');
  const a2b = gTwo.addNode('sigmoid');
  const l2 = gTwo.addNode('bce_loss');
  gTwo.connect(s2.id, 'out', d2.id, 'in');
  gTwo.connect(d2.id, 'out', a2a.id, 'in');
  gTwo.connect(a2a.id, 'out', a2b.id, 'in');
  gTwo.connect(a2b.id, 'out', l2.id, 'in');
  autoFixGraphToConvergence(gTwo, ctx);
  const planTwo = compile(gTwo, ctx);
  check('consecutive activations are repaired (no silent override)', planTwo.ok && planTwo.network.architecture[1].activation === 'sigmoid', JSON.stringify(planTwo.errors.map((e) => e.message)));
  check('the removed activation is gone from the canvas', !gTwo.getNode(a2a.id));

  // Fuzz: malformed random graphs must never throw the assist machine.
  let fuzzThrew = 0;
  for (let seed = 1; seed <= 30; seed++) {
    const fuzz = new GraphState('assist');
    const types = ['source', 'dense', 'relu', 'sigmoid', 'softmax', 'bce_loss', 'cce_loss', 'mse_loss', 'adam_optimizer', 'epoch_loop', 'flatten', 'class_filter', 'metrics_gauge', 'confusion_matrix'];
    let rng = seed * 2654435761 % 4294967296;
    const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 4294967296; };
    const nodes = [];
    const count = 2 + Math.floor(rand() * 6);
    for (let i = 0; i < count; i++) {
      nodes.push(fuzz.addNode(types[Math.floor(rand() * types.length)], {
        x: rand() * 500, params: { datasetId: 'xor', size: 1 + Math.floor(rand() * 10) }
      }));
    }
    for (let i = 0; i < nodes.length * 2; i++) {
      const from = nodes[Math.floor(rand() * nodes.length)];
      const to = nodes[Math.floor(rand() * nodes.length)];
      const defFrom = NODE_TYPES[from.type];
      const defTo = NODE_TYPES[to.type];
      if (!defFrom.outputs || !defTo.inputs || defFrom.outputs.length === 0 || defTo.inputs.length === 0) continue;
      fuzz.connect(from.id, defFrom.outputs[0].name, to.id, defTo.inputs[0].name);
    }
    try {
      autoFixGraphToConvergence(fuzz, ctx);
      compile(fuzz, ctx);
    } catch (err) {
      fuzzThrew++;
      console.log(`    fuzz seed ${seed} threw: ${err.message}`);
    }
  }
  check('30 malformed random graphs auto-fix + compile without exceptions', fuzzThrew === 0, `${fuzzThrew} threw`);

  // autoWireOnDrop: dropping a Dense onto a canvas with a source wires it.
  const gDrop = new GraphState('assist');
  const sDrop = gDrop.addNode('source', { params: { datasetId: 'xor' } });
  const dDrop = gDrop.addNode('dense', { params: { size: 8 } });
  autoWireOnDrop(gDrop, dDrop.id, ctx);
  check('autoWireOnDrop attaches a dropped Dense to the chain end', gDrop.incoming(dDrop.id, 'in') !== null);
}

section('PART 5 — One-click presets compile clean');

{
  const mnist = instantPreset('mnist_classifier');
  const planM = compile(mnist, ctx);
  check('MNIST preset compiles with zero errors', planM.ok, JSON.stringify((planM.errors || []).map((e) => e.message)));
  check('MNIST preset architecture: 784 → 16/8/2 relu…sigmoid', planM.ok && JSON.stringify(planM.network.architecture) === JSON.stringify([
    { size: 784 }, { size: 16, activation: 'relu' }, { size: 8, activation: 'relu' }, { size: 2, activation: 'sigmoid' }
  ]));
  check('MNIST preset includes gauges + confusion + probe taps', planM.ok && planM.taps.gauges.length >= 1 && planM.taps.confusion === true && planM.taps.predict === true);

  const xor = instantPreset('xor_classifier');
  const planX = compile(xor, ctx);
  check('XOR preset compiles (fixed 40-epoch budget)', planX.ok && planX.training.maxEpochs === 40 && planX.training.mode === 'fixed');

  const iris = instantPreset('iris_multi_class');
  const planI = compile(iris, ctx);
  check('Iris preset compiles (softmax + CCE, 3 outputs)', planI.ok && planI.network.loss === 'categorical_crossentropy' && planI.meta.outputSize === 3);

  check('unknown preset returns null', instantPreset('nope') === null);
}

section('PART 6 — Geometry');

{
  const node = { x: 100, y: 50 };
  const def = NODE_TYPES.dense;
  const inPos = socketPosition(node, def, 'in', 'in');
  const outPos = socketPosition(node, def, 'out', 'out');
  check('input socket sits on the left edge', inPos.x === 100);
  check('output socket sits on the right edge', outPos.x === 100 + NODE_WIDTH);

  const ports = socketsOf({ x: 0, y: 0 }, NODE_TYPES.bce_loss);
  check('socket map covers every declared port', ports.length === (NODE_TYPES.bce_loss.inputs.length + NODE_TYPES.bce_loss.outputs.length));

  const from = { x: 0, y: 0 };
  const to = { x: 200, y: 100 };
  const path = bezierPath(from, to);
  check('bezier path string is a cubic from→to', path.startsWith('M 0 0 C ') && path.endsWith('200 100'));
  const p0 = pointOnBezier(from, to, 0);
  const p1 = pointOnBezier(from, to, 1);
  check('bezier evaluation hits both endpoints', p0.x === 0 && p0.y === 0 && Math.abs(p1.x - 200) < 1e-9 && Math.abs(p1.y - 100) < 1e-9);

  const view = { x: 30, y: -20, scale: 1.5 };
  const world = { x: 12, y: 7 };
  const screen = worldToScreen(world, view);
  const back = screenToWorld(screen, view);
  check('screen↔world transforms are exact inverses', Math.abs(back.x - world.x) < 1e-9 && Math.abs(back.y - world.y) < 1e-9);

  const anchor = { x: 400, y: 300 };
  const beforeWorld = screenToWorld(anchor, view);
  const zoomed = zoomViewAt(view, anchor, 1.2);
  const afterWorld = screenToWorld(anchor, zoomed);
  check('zoom keeps the anchor point fixed in world space', Math.abs(afterWorld.x - beforeWorld.x) < 1e-9 && Math.abs(afterWorld.y - beforeWorld.y) < 1e-9);

  const monitorPort = ports.find((port) => port.kind === 'monitor');
  const hit = hitTestSocket(ports, { x: monitorPort.x + 4, y: monitorPort.y + 3 }, 12, { direction: 'out', kind: 'monitor' });
  check('socket snapping finds the nearest matching port', hit !== null && hit.kind === 'monitor');
  const miss = hitTestSocket(ports, { x: 500, y: 500 }, 12);
  check('socket snapping misses when nothing is near', miss === null);
}

section('PART 7 — Scheduler, class filter, tutorial machine');

{
  check('constant schedule keeps multiplier 1', schedulerValue({ mode: 'constant' }, 57) === 1);
  check('step schedule holds before the first boundary', schedulerValue({ mode: 'step', factor: 0.5, every: 100 }, 100) === 1);
  check('step schedule halves after the boundary', schedulerValue({ mode: 'step', factor: 0.5, every: 100 }, 101) === 0.5);
  check('step schedule applies repeatedly', schedulerValue({ mode: 'step', factor: 0.5, every: 100 }, 201) === 0.25);

  check('parseKeepList accepts "0,1"', JSON.stringify(parseKeepList('0,1')) === '[0,1]');
  check('parseKeepList rejects junk', parseKeepList('a,b') === null);

  const dataset = {
    name: 'Toy 3-class', type: 'classification',
    featureNames: ['a', 'b'], targetNames: ['c0', 'c1', 'c2'],
    numFeatures: 2, numClasses: 3,
    X: [[0, 0], [1, 0], [0, 1], [1, 1]],
    y: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
  };
  const filtered = applyClassFilter(dataset, [0, 2]);
  check('applyClassFilter keeps the selected rows only', filtered.X.length === 3 && filtered.y.length === 3);
  check('applyClassFilter remaps one-hot targets to the new class count',
    filtered.numClasses === 2 && JSON.stringify(filtered.y[1]) === '[0,1]');
  let threw = false;
  try { applyClassFilter({ type: 'regression', X: [], y: [] }, [0]); } catch (e) { threw = true; }
  check('applyClassFilter refuses regression datasets', threw === true);

  const tour = TOURS.first_ai;
  const runner = new TutorialRunner(tour);
  check('runner starts at step 1', runner.index === 0 && runner.step.id === 't1-welcome');
  runner.notify('wrong_action');
  check('notify ignores unrelated actions', runner.index === 0);
  const journalRunner = new TutorialRunner(tour, { actions: [{ id: 'preset_mnist', seq: 1 }] });
  check('a gesture already in the page journal satisfies the tour gate',
    journalRunner.isSatisfied() === true);
  const unrelatedJournal = new TutorialRunner(tour, { actions: [{ id: 'node_added', seq: 1 }] });
  check('an unrelated journal entry does not satisfy the tour gate',
    unrelatedJournal.isSatisfied() === false);
  check('lesson copy renders bold emphasis without exposing asterisks',
    inlineMarkdown('Press **Run** now.') === 'Press <strong>Run</strong> now.');
  check('lesson copy escapes HTML before adding its own markup',
    inlineMarkdown('<b>x</b> **ok**') === '&lt;b&gt;x&lt;/b&gt; <strong>ok</strong>');
  runner.notify('preset_mnist');
  check('notify advances on the gated action', runner.index === 1);
  runner.prev();
  check('prev steps back', runner.index === 0);
  while (!runner.done) runner.next();
  check('runner completes and reports done', runner.done === true && runner.step === null);

  const tourIssues = [];
  for (const [tourId, tourDef] of Object.entries(TOURS)) {
    if (!tourDef.steps || tourDef.steps.length < 3) tourIssues.push(`${tourId}:too short`);
    const ids = new Set();
    for (const step of tourDef.steps) {
      if (!step.target || typeof step.target !== 'string' || !step.target.startsWith('#')) tourIssues.push(`${tourId}.${step.id}:bad target`);
      if (!step.title || !step.text) tourIssues.push(`${tourId}.${step.id}:missing text`);
      if (ids.has(step.id)) tourIssues.push(`${tourId}.${step.id}:duplicate`);
      ids.add(step.id);
    }
  }
  check('all three tours are well-formed', tourIssues.length === 0, tourIssues.join(', '));
}

/* ------------------------------------------------------------------ */
console.log('\n=====================================================');
console.log(`   RESULT: ${passed} passed, ${failed} failed`);
console.log('=====================================================');
process.exit(failed === 0 ? 0 : 1);
