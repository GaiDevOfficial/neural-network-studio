/**
 * Drag-wire-to-empty quick-add menu — socket filtering, real compiler scoring, one-gesture add.
 *
 * Verifies:
 *   - candidates are limited to blocks whose real socket kind/direction matches the drag,
 *   - unsupported registry blocks (Conv2D/MaxPool/Dropout) never appear,
 *   - every candidate is scored by cloning the live graph and running the real compiler,
 *   - assist mode lets Smart Assist insert Flatten; manual mode does not,
 *   - `applyQuickAdd` rolls back completely when the engine refuses a connection,
 *   - the tier policies shape the view model (tier 1 curated friendly groups, tier 3 raw error
 *     codes) and search reaches blocks beyond the curated first screen,
 *   - the popover honours the universal close contract (Escape, outside click, injected ×) and
 *     delivers its pick before closing.
 *
 * File: tests/test_quick_node_menu.js
 */

import { GraphState } from '../src/graph/graph_state.js';
import { compile, inferGraph } from '../src/graph/node_engine.js';
import { MenuManager } from '../src/ui/menu_manager.js';
import { FakeDocument, FakeElement, FakeEvent } from './fake_dom.js';
import {
  baseCandidates,
  socketForCandidate,
  scoreCandidates,
  filterCandidates,
  buildQuickMenuModel,
  applyQuickAdd,
  QuickNodeMenu,
  candidateExplanation
} from '../src/canvas/quick_node_menu.js';
import {
  modeToTier,
  activeTier,
  quickMenuPolicy,
  wireMenuPolicy,
  buildQuickMenu
} from '../src/modes/interactive_modes.js';

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
  return { innerWidth: 1280, innerHeight: 800, addEventListener() {}, removeEventListener() {} };
}

const XOR_INFO = { datasetId: 'xor', type: 'classification', name: 'XOR', numFeatures: 2, numClasses: 2, imageDim: null };
const XOR_CONTEXT = { datasetInfoFor: (node) => (node.type === 'source' ? XOR_INFO : null) };
const MNIST_INFO = { datasetId: 'mnist_01', type: 'classification', name: 'MNIST 0 vs 1', numFeatures: 784, numClasses: 2, imageDim: { width: 28, height: 28 } };
const MNIST_CONTEXT = { datasetInfoFor: (node) => (node.type === 'source' ? MNIST_INFO : null) };

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
  const wire = (a, as, b, bs) => {
    const result = graph.connect(a.id, as, b.id, bs);
    if (!result.ok) throw new Error(`Fixture wiring failed: ${result.error.code}`);
    return result.connection;
  };
  wire(source, 'out', dense, 'in');
  wire(dense, 'out', relu, 'in');
  wire(relu, 'out', out, 'in');
  wire(out, 'out', sigmoid, 'in');
  wire(sigmoid, 'out', loss, 'in');
  wire(opt, 'cfg', loss, 'opt');
  wire(loop, 'cfg', loss, 'loop');
  return { graph, source, dense, relu, out, sigmoid, loss, opt, loop };
}

function main() {
  console.log('=====================================================');
  console.log('  QUICK NODE MENU — real candidates, real scoring');
  console.log('=====================================================');

  /* --------------------------------------------------------------- tiers */
  console.log('--- three-tier policies ---');
  check('playground maps to tier 1', modeToTier('playground') === 1);
  check('assist maps to tier 2', modeToTier('assist') === 2);
  check('manual maps to tier 3', modeToTier('manual') === 3);
  check('a mission tier overrides the mode tier', activeTier({ mode: 'assist', missionTier: 3 }) === 3);
  check('no mission keeps the mode tier', activeTier({ mode: 'manual', missionTier: 0 }) === 3);
  check('an invalid mission tier falls back to the mode', activeTier({ mode: 'assist', missionTier: 99 }) === 2);

  const t1 = quickMenuPolicy(1);
  const t2 = quickMenuPolicy(2);
  const t3 = quickMenuPolicy(3);
  check('tier 1 curates to 8 friendly intent groups', t1.maxItems === 8 && t1.intentGroups === true && t1.friendly === true);
  check('tier 1 hides raw error reasons', t1.showReasons === false && t1.showDetail === false);
  check('tier 2 shows all compatible blocks by category', t2.maxItems === 14 && t2.intentGroups === false);
  check('tier 3 shows raw sockets, details and reasons', t3.maxItems === 0 && t3.showDetail === true && t3.showReasons === true);
  check('tier 1 wire menu is friendly', wireMenuPolicy(1).friendly === true && wireMenuPolicy(1).showSockets === false);
  check('tier 3 wire menu computes topology warnings', wireMenuPolicy(3).topologyWarning === true);

  /* ------------------------------------------------------------ candidates */
  console.log('--- socket filtering ---');
  const denseIn = socketForCandidate('dense', 'tensor', 'out');
  check('an out-drag targets a candidate input', Boolean(denseIn && denseIn.name === 'in' && denseIn.direction === 'in'));
  const adamOut = socketForCandidate('adam_optimizer', 'config', 'in');
  check('an in-drag targets a candidate output', Boolean(adamOut && adamOut.name === 'cfg' && adamOut.direction === 'out'));
  check('a config block is not offered for tensor wires', socketForCandidate('adam_optimizer', 'tensor', 'out') === null);
  check('unsupported blocks are never offered', socketForCandidate('conv2d', 'tensor', 'out') === null && socketForCandidate('dropout', 'tensor', 'out') === null && socketForCandidate('maxpool2d', 'tensor', 'out') === null);

  const tensorDrag = { nodeId: 'n1', socket: 'out', direction: 'out', kind: 'tensor' };
  const tensorCandidates = baseCandidates(tensorDrag);
  const tensorTypes = tensorCandidates.map((row) => row.type);
  check('tensor candidates include layers and losses', tensorTypes.includes('dense') && tensorTypes.includes('relu') && tensorTypes.includes('bce_loss'));
  check('tensor candidates exclude sources', !tensorTypes.includes('source') && !tensorTypes.includes('sql_source'));
  check('tensor candidates exclude config-only blocks', !tensorTypes.includes('adam_optimizer') && !tensorTypes.includes('epoch_loop'));
  check('tensor candidates exclude monitor taps', !tensorTypes.includes('metrics_gauge') && !tensorTypes.includes('prediction_view'));
  check('every candidate carries a compatible socket', tensorCandidates.every((row) => row.kind === 'tensor' && row.direction === 'in' && row.socket === 'in'));

  const configDrag = { nodeId: 'n2', socket: 'opt', direction: 'in', kind: 'config' };
  const configTypes = baseCandidates(configDrag).map((row) => row.type);
  check('config candidates include all four optimizers', ['adam_optimizer', 'sgd_optimizer', 'momentum_optimizer', 'rmsprop_optimizer'].every((type) => configTypes.includes(type)));
  check('config candidates include schedule and loop', configTypes.includes('lr_schedule') && configTypes.includes('epoch_loop'));
  check('config candidates exclude tensor layers', !configTypes.includes('dense') && !configTypes.includes('relu'));

  const monitorDrag = { nodeId: 'n3', socket: 'monitor', direction: 'out', kind: 'monitor' };
  const monitorTypes = baseCandidates(monitorDrag).map((row) => row.type);
  check('monitor candidates are exactly the taps', monitorTypes.includes('metrics_gauge') && monitorTypes.includes('confusion_matrix') && monitorTypes.includes('prediction_view'));
  check('monitor candidates exclude tensors', !monitorTypes.includes('dense') && !monitorTypes.includes('sigmoid'));

  /* --------------------------------------------------------------- scoring */
  console.log('--- compiler-scored candidates ---');
  const chain = buildValidXor();
  const baseline = inferGraph(chain.graph, XOR_CONTEXT);
  check('the fixture graph is compile-valid', compile(chain.graph, XOR_CONTEXT).ok === true);

  const fromSigmoid = { nodeId: chain.sigmoid.id, socket: 'out', direction: 'out', kind: 'tensor' };
  const rows = scoreCandidates(chain.graph, fromSigmoid, { mode: 'assist', context: XOR_CONTEXT, position: { x: 1400, y: 0 } });
  check('scoring returns every compatible block', rows.length === baseCandidates(fromSigmoid).length, `${rows.length}`);
  check('a clean extra Dense keeps the graph compiling', (() => {
    const dense = rows.find((row) => row.type === 'dense');
    return dense && dense.applies === true && dense.errorsAfter === 0;
  })());
  check('candidate rows carry the real technical line', rows.every((row) => typeof row.detail === 'string' && row.detail.length > 0));
  check('candidate rows carry the real friendly alias', rows.find((row) => row.type === 'relu').friendly.length > 0);
  check('unsupported blocks never appear in scored rows', !rows.some((row) => ['conv2d', 'maxpool2d', 'dropout'].includes(row.type)));

  const sourceOnly = new GraphState('manual');
  const src = sourceOnly.addNode('source', { x: 0, y: 0, params: { datasetId: 'mnist_01' } });
  const sourceDrag = { nodeId: src.id, socket: 'out', direction: 'out', kind: 'tensor' };
  const manualBaseline = inferGraph(sourceOnly, MNIST_CONTEXT);
  const manualRows = scoreCandidates(sourceOnly, sourceDrag, { mode: 'manual', context: MNIST_CONTEXT, position: { x: 300, y: 0 } });
  const manualDense = manualRows.find((row) => row.type === 'dense');
  check('manual mode measures the real rank error instead of hiding it', manualDense.applies === false && manualDense.errorsAfter > manualBaseline.errors.length, `${manualDense.errorsAfter} vs ${manualBaseline.errors.length}`);
  check('the blocked row names the real compiler error', /E_RANK_MISMATCH/.test(manualDense.reason || ''), manualDense.reason);

  const assistRows = scoreCandidates(sourceOnly, sourceDrag, { mode: 'assist', context: MNIST_CONTEXT, position: { x: 300, y: 0 } });
  const assistDense = assistRows.find((row) => row.type === 'dense');
  check('assist mode adds no new errors (Smart Assist will insert Flatten)', assistDense.errorsAfter === manualBaseline.errors.length, `${assistDense.errorsAfter}`);
  check('assist rows still report honestly that the graph is not complete yet', assistDense.applies === false && assistDense.errorsAfter > 0);

  const found = filterCandidates(rows, 'switch');
  check('search reaches a friendly alias ("switch" matches ReLU)', found.some((row) => row.type === 'relu'), found.map((row) => row.type).join(','));
  const exact = filterCandidates(rows, 'rectifier');
  check('a unique alias returns exactly that block', exact.length === 1 && exact[0].type === 'relu', exact.map((row) => row.type).join(','));
  check('search matches the registry label', filterCandidates(rows, 'sigmoid').some((row) => row.type === 'sigmoid'));
  check('an unmatched search returns nothing', filterCandidates(rows, 'zzzz').length === 0);

  /* ---------------------------------------------------------------- model */
  console.log('--- tier-shaped view model ---');
  const tier3Model = buildQuickMenuModel({ drag: { ...fromSigmoid, nodeLabel: 'Sigmoid' }, rows, policy: t3 });
  check('tier 3 model exposes raw details + reasons', tier3Model.showDetail === true && tier3Model.showReasons === true);
  check('tier 3 groups every compatible block', tier3Model.groups.reduce((total, group) => total + group.rows.length, 0) === rows.length);
  check('the subtitle names the dragged socket', tier3Model.subtitle.includes('Sigmoid') && tier3Model.subtitle.includes('out (tensor)'));

  const tier1Model = buildQuickMenuModel({ drag: { ...fromSigmoid, nodeLabel: 'Sigmoid' }, rows, policy: t1 });
  check('tier 1 model uses friendly labels', tier1Model.friendly === true && tier1Model.showDetail === false);
  check('tier 1 model keeps a curated display cap', tier1Model.limit === 8);
  check('tier 1 groups are intent names', tier1Model.groups.every((group) => t1.groupOrder.includes(group.label)), tier1Model.groups.map((group) => group.label).join(','));
  check('tier 1 still keeps all rows available for search', tier1Model.groups.reduce((total, group) => total + group.rows.length, 0) === rows.length);

  const built = buildQuickMenu({
    graph: chain.graph,
    drag: { ...fromSigmoid, nodeLabel: 'Sigmoid' },
    position: { x: 1400, y: 0 },
    ctx: { mode: 'assist', missionTier: 3, context: XOR_CONTEXT }
  });
  check('buildQuickMenu applies the mission tier override', built.tier === 3 && built.policy.showReasons === true);
  check('buildQuickMenu returns the measured rows and view model', Array.isArray(built.rows) && built.model.groups.length > 0);

  /* -------------------------------------------------------------- mutation */
  console.log('--- applyQuickAdd (real graph mutation) ---');
  const assistGraph = new GraphState('assist');
  const assistSource = assistGraph.addNode('source', { x: 0, y: 0, params: { datasetId: 'mnist_01' } });
  const added = applyQuickAdd(assistGraph, { nodeId: assistSource.id, socket: 'out', direction: 'out', kind: 'tensor' }, 'dense', { x: 420, y: 180 }, { mode: 'assist', context: MNIST_CONTEXT });
  check('assist quick-add succeeds', added.ok === true);
  const addedDense = assistGraph.nodeList().find((node) => node.type === 'dense');
  const flatten = assistGraph.nodeList().find((node) => node.type === 'flatten');
  check('Smart Assist actually inserted the Flatten block', Boolean(flatten));
  check('the inserted Flatten sits between source and Dense', Boolean(assistGraph.incoming(flatten.id, 'in')) && assistGraph.incoming(addedDense.id, 'in').from.nodeId === flatten.id);
  check('the repair is reported as a real action', (added.actions || []).some((action) => action.code === 'A_INSERT_FLATTEN'));
  check('the new block landed at the release point', addedDense.x === 420 && addedDense.y === 180);

  const manualGraph = new GraphState('manual');
  const manualSource = manualGraph.addNode('source', { x: 0, y: 0, params: { datasetId: 'mnist_01' } });
  const manualAdded = applyQuickAdd(manualGraph, { nodeId: manualSource.id, socket: 'out', direction: 'out', kind: 'tensor' }, 'dense', { x: 420, y: 180 }, { mode: 'manual', context: MNIST_CONTEXT });
  check('manual quick-add still places the requested block', manualAdded.ok === true);
  check('manual mode inserts no Flatten (strict wiring)', !manualGraph.nodeList().some((node) => node.type === 'flatten'));
  const manualPlan = compile(manualGraph, MNIST_CONTEXT);
  check('manual rank mismatch is a real compile error', manualPlan.ok === false && manualPlan.errors.some((issue) => issue.code === 'E_RANK_MISMATCH'));

  const optAdded = applyQuickAdd(chain.graph, { nodeId: chain.loss.id, socket: 'opt', direction: 'in', kind: 'config' }, 'rmsprop_optimizer', { x: 1100, y: 380 }, { mode: 'assist', context: XOR_CONTEXT });
  check('an in-drag wires the candidate output into the dragged input', optAdded.ok === true && optAdded.connection.from.nodeId === optAdded.node.id && optAdded.connection.to.socket === 'opt');
  check('replaceable sockets are re-wired, not duplicated', chain.graph.incoming(chain.loss.id, 'opt').from.nodeId === optAdded.node.id);
  check('the graph stays compile-valid after the replacement', compile(chain.graph, XOR_CONTEXT).ok === true);

  const beforeBadPick = chain.graph.nodes.size;
  const badPick = applyQuickAdd(chain.graph, { nodeId: chain.loss.id, socket: 'opt', direction: 'in', kind: 'config' }, 'dense', { x: 0, y: 0 }, { mode: 'assist', context: XOR_CONTEXT });
  check('a candidate without the right socket kind is refused', badPick.ok === false && badPick.error.code === 'E_NO_SOCKET');
  check('a refused pick leaves no stray node', chain.graph.nodes.size === beforeBadPick);

  const occupiedGraph = new GraphState('manual');
  const oSource = occupiedGraph.addNode('source', { x: 0, y: 0, params: { datasetId: 'xor' } });
  const oDense = occupiedGraph.addNode('dense', { x: 300, y: 0 });
  const oLoss = occupiedGraph.addNode('bce_loss', { x: 600, y: 0 });
  occupiedGraph.connect(oSource.id, 'out', oDense.id, 'in');
  occupiedGraph.connect(oDense.id, 'out', oLoss.id, 'in');
  const occupiedBefore = occupiedGraph.nodes.size;
  const occupiedPick = applyQuickAdd(occupiedGraph, { nodeId: oLoss.id, socket: 'in', direction: 'in', kind: 'tensor' }, 'relu', { x: 450, y: 200 }, { mode: 'manual', context: XOR_CONTEXT });
  check('manual mode refuses a wire into an occupied input', occupiedPick.ok === false && occupiedPick.error.code === 'E_DUP_INPUT');
  check('the refused node is rolled back from the graph', occupiedGraph.nodes.size === occupiedBefore && occupiedGraph.getNode('node_4') === null);
  check('the existing wire is untouched after the refusal', occupiedGraph.incoming(oLoss.id, 'in').from.nodeId === oDense.id);

  const explanation = candidateExplanation('sigmoid');
  check('candidate explanations cover plain, analogy and technical', explanation.plain.length > 0 && explanation.analogy.length > 0 && explanation.technical.length > 0);

  /* ------------------------------------------------------------- popover */
  console.log('--- QuickNodeMenu popover ---');
  const doc = new FakeDocument();
  const win = fakeWindow();
  const manager = new MenuManager({ document: doc, window: win });
  let picked = null;
  let closes = 0;
  const menu = new QuickNodeMenu({
    document: doc,
    window: win,
    menuManager: manager,
    onPick: (row) => { picked = row; },
    onClose: () => { closes++; }
  });
  check('the popover panel mounts onto the document', doc.body.contains(menu.panel) && doc.querySelector('.ns-quick-menu') === menu.panel);

  const syntheticRows = [
    { type: 'dense', label: 'Dense Layer', friendly: 'Brain cell', category: 'layer', categoryLabel: 'Layers', summary: 'Neurons that learn weights.', aliases: 'dense layer neurons fc', socket: 'in', direction: 'in', kind: 'tensor', detail: 'y = xW + b', applies: true, score: 0, errorsAfter: 0, warningsAfter: 0, reason: null },
    { type: 'relu', label: 'ReLU', friendly: 'Light switch', category: 'activation', categoryLabel: 'Activations', summary: 'Off for negatives, on for positives.', aliases: 'relu switch nonlinear', socket: 'in', direction: 'in', kind: 'tensor', detail: 'max(0, x)', applies: true, score: 0, errorsAfter: 0, warningsAfter: 0, reason: null },
    { type: 'linear', label: 'Linear', friendly: 'No switch', category: 'activation', categoryLabel: 'Activations', summary: 'Pass the signal through unchanged.', aliases: 'linear identity regression', socket: 'in', direction: 'in', kind: 'tensor', detail: 'y = x', applies: true, score: 0, errorsAfter: 0, warningsAfter: 0, reason: null }
  ];
  const cappedPolicy = { ...quickMenuPolicy(1), maxItems: 2 };
  const model = buildQuickMenuModel({ drag: { nodeId: 'node_1', socket: 'out', direction: 'out', kind: 'tensor', nodeLabel: 'Dataset Source' }, rows: syntheticRows, policy: cappedPolicy });
  check('the popover model carries the drag in its subtitle', model.subtitle.includes('Dataset Source'));

  menu.open(model, { clientX: 500, clientY: 200 });
  check('open goes through MenuManager', manager.isOpen('quick-node-menu') === true);
  check('the curated cap limits the first screen', menu.buttons.length === 2, `${menu.buttons.length}`);
  check('friendly tier names are rendered', menu.buttons[0].button.children.some((child) => child.className === 'ns-quick-name' && child.textContent === 'Brain cell'));
  check('each row shows its socket tag', menu.buttons[0].button.children.some((child) => child.className === 'ns-quick-socket' && child.textContent === 'in:tensor'));
  check('search reaches a block beyond the curated cap', (() => {
    menu.setQuery('identity');
    return menu.buttons.length === 1 && menu.buttons[0].row.type === 'linear';
  })());
  check('clearing the search restores the curated cap', (() => {
    menu.setQuery('');
    return menu.buttons.length === 2;
  })());
  check('the menu panel is positioned at the cursor', menu.panel.style.left === '500px' && menu.panel.style.top === '200px', `${menu.panel.style.left},${menu.panel.style.top}`);

  menu.pickFirst();
  check('Enter/pick delivers the active row', picked && picked.type === 'dense');
  check('picking closes the menu', manager.isOpen('quick-node-menu') === false);
  check('onClose fires exactly once after the pick', closes === 1 && menu.query === '');

  menu.open(model, { clientX: 10, clientY: 10 });
  doc.dispatchEvent(new FakeEvent('keydown', { key: 'Escape' }));
  check('Escape closes the quick menu', manager.isOpen('quick-node-menu') === false && closes === 2);
  menu.open(model, { clientX: 10, clientY: 10 });
  check('the injected × exists', Boolean(menu.panel.querySelector('.ns-menu-close')));
  menu.panel.querySelector('.ns-menu-close').click();
  check('the explicit × closes the quick menu', manager.isOpen('quick-node-menu') === false && closes === 3);

  menu.open(model, { clientX: 10, clientY: 10 });
  menu.setActive(1);
  menu.moveActive(1);
  check('arrow navigation wraps within the visible rows', menu.activeIndex === 0);
  menu.moveActive(-1);
  check('arrow navigation wraps backwards', menu.activeIndex === 1);
  menu.close();

  menu.destroy();
  check('destroy unregisters the menu', manager.get('quick-node-menu') === null);
  manager.destroy();
  check('destroying the manager frees the document listeners', doc.listenerCount('keydown') === 0 && doc.listenerCount('pointerdown') === 0);

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main();
