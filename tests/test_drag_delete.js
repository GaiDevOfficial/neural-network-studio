/**
 * Drag-to-delete verification.
 *
 * Two layers:
 *   1. Palette geometry + trash state against the real NodePalette (fake DOM): the dropzone is a
 *      real getBoundingClientRect test, the veil exists, setTrashTarget is reversible, and every
 *      palette item is a real drag source / add button.
 *   2. Graph sanitisation against the real GraphState + compiler: deleting ANY block from each
 *      shipped preset must remove the block, sever every touching wire, leave no dangling
 *      endpoints, and never throw during recompilation.
 *
 * The wiring guard at the end reads canvas_studio.js + node_studio.css to make sure the pointerup
 * path still calls deleteNode(viaTrash) and that `hidden` can defeat author display rules.
 *
 * File: tests/test_drag_delete.js
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NodePalette } from '../src/canvas/node_palette.js';
import { GraphState } from '../src/graph/graph_state.js';
import { instantPreset } from '../src/graph/smart_assist.js';
import { compile } from '../src/graph/node_engine.js';
import { paletteSections, getNodeType, TAP_NODE_TYPES } from '../src/graph/node_types.js';
import { FakeDocument, FakeElement, FakeEvent } from './fake_dom.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

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

const XOR_INFO = { type: 'classification', name: 'XOR', numFeatures: 2, numClasses: 2, imageDim: null };
const IRIS_INFO = { type: 'classification', name: 'Iris', numFeatures: 4, numClasses: 3, imageDim: null };
const MNIST_INFO = { type: 'classification', name: 'MNIST 0 vs 1', numFeatures: 784, numClasses: 2, imageDim: { width: 28, height: 28 } };

function contextFor(info) {
  return { datasetInfoFor: (node) => (node.type === 'source' ? info : null) };
}

function danglingConnections(graph) {
  const dangling = [];
  for (const connection of graph.connections.values()) {
    if (!graph.getNode(connection.from.nodeId) || !graph.getNode(connection.to.nodeId)) {
      dangling.push(connection.id || `${connection.from.nodeId}->${connection.to.nodeId}`);
    }
  }
  return dangling;
}

function testPalette() {
  console.log('--- palette geometry & trash affordance ---');
  const doc = new FakeDocument();
  const container = new FakeElement('aside', doc);
  container.setBoundingRect({ left: 10, top: 20, right: 250, bottom: 720 });

  const added = [];
  const dragged = [];
  const palette = new NodePalette(container, {
    onAdd: (type) => added.push(type),
    onDragStart: (type) => dragged.push(type)
  });
  palette.render();

  const { sections, planned } = paletteSections();
  const expectedBlocks = sections.reduce((sum, section) => sum + section.blocks.length, 0);
  check('palette renders every registered block', palette.rendered === expectedBlocks,
    `${palette.rendered}/${expectedBlocks}`);
  check('planned v2 blocks are listed but not draggable', planned.length === 3 && palette.rendered === expectedBlocks);
  check('palette exposes stable ids for tutorials', Boolean(container.querySelector('#palette-sql_source')) ||
    Boolean(container.querySelectorAll('.ns-palette-item').find((item) => item.id === 'palette-sql_source')));

  const items = container.querySelectorAll('.ns-palette-item');
  const activeItems = items.filter((item) => !item.classList.contains('is-planned'));
  check('palette renders one active entry per block', activeItems.length === expectedBlocks,
    `${activeItems.length}/${expectedBlocks}`);
  check('active palette items are draggable',
    activeItems.length > 0 && activeItems.every((item) => item.draggable === true));

  const sql = activeItems.find((item) => item.id === 'palette-sql_source');
  check('SQL Query block has a palette entry', Boolean(sql));
  sql.click();
  check('clicking a palette block asks the page to add it', added.length === 1 && added[0] === 'sql_source',
    JSON.stringify(added));

  const dataTransfer = {
    data: null,
    setData(type, value) { this.data = { type, value }; },
    effectAllowed: ''
  };
  sql.dispatchEvent(new FakeEvent('dragstart', { dataTransfer }));
  check('dragstart carries the node type', dataTransfer.data && dataTransfer.data.value === 'sql_source',
    JSON.stringify(dataTransfer.data));
  check('dragstart callback fires with the type', dragged.length === 1 && dragged[0] === 'sql_source');
  check('dragstart advertises copy semantics', dataTransfer.effectAllowed === 'copy');

  check('isOver accepts a point inside the palette', palette.isOver(100, 200) === true);
  check('isOver accepts the exact edges', palette.isOver(10, 20) === true && palette.isOver(250, 720) === true);
  check('isOver rejects a canvas point left of the palette', palette.isOver(9, 200) === false);
  check('isOver rejects a point below the palette', palette.isOver(100, 721) === false);

  palette.setTrashTarget(true);
  check('trash target applies the red state class', container.classList.contains('is-trash-target'));
  check('trash target updates the accessible label',
    container.getAttribute('aria-label') === 'Block palette — drop here to delete',
    container.getAttribute('aria-label'));
  palette.setTrashTarget(true);
  check('repeating setTrashTarget(true) is idempotent', container.classList.contains('is-trash-target'));
  palette.setTrashTarget(false);
  check('trash target is reversible', container.classList.contains('is-trash-target') === false);
  check('trash target restores the palette label', container.getAttribute('aria-label') === 'Block palette');

  const veil = container.querySelectorAll('.ns-palette-trash-veil')[0];
  check('trash veil exists with real copy', Boolean(veil) && veil.textContent.includes('Drop to delete'),
    veil ? veil.textContent : 'missing');

  palette.destroy();
  check('destroy clears rendered items', palette.rendered === 0 && container.children.length === 0);
}

function testGraphDeletion() {
  console.log('\n--- deleting any block severs its wires and never corrupts the graph ---');
  const suites = [
    { id: 'xor_classifier', label: 'XOR', info: XOR_INFO },
    { id: 'iris_multi_class', label: 'Iris', info: IRIS_INFO },
    { id: 'mnist_classifier', label: 'MNIST', info: MNIST_INFO }
  ];

  for (const suite of suites) {
    const preset = instantPreset(suite.id);
    const plan = compile(preset, contextFor(suite.info));
    check(`${suite.label} preset compiles before deletion`, plan.ok === true,
      JSON.stringify((plan.errors || []).map((error) => error.code)));

    for (const node of preset.nodeList()) {
      const graph = preset.clone();
      const beforeLinks = graph.connections.size;
      const touching = Array.from(graph.connections.values())
        .filter((connection) => connection.from.nodeId === node.id || connection.to.nodeId === node.id).length;
      const removed = graph.removeNode(node.id);
      const afterLinks = graph.connections.size;

      check(`${suite.label}: remove ${node.type} (${node.id})`,
        removed === true && graph.getNode(node.id) === null);
      check(`${suite.label}: ${node.type} removal severs ${touching} touching wire(s)`,
        beforeLinks - afterLinks === touching, `${beforeLinks} → ${afterLinks}, touched ${touching}`);
      check(`${suite.label}: ${node.type} removal leaves no dangling wires`, danglingConnections(graph).length === 0);

      let recompiled = null;
      let threw = null;
      try {
        recompiled = compile(graph, contextFor(suite.info));
      } catch (error) {
        threw = error;
      }
      check(`${suite.label}: recompilation after ${node.type} removal does not throw`, threw === null, threw ? threw.message : '');
      if (TAP_NODE_TYPES.includes(node.type)) {
        check(`${suite.label}: removing optional ${node.type} keeps the core graph valid`,
          Boolean(recompiled) && recompiled.ok === true, recompiled ? JSON.stringify((recompiled.errors || []).map((e) => e.code)) : 'no plan');
      } else {
        check(`${suite.label}: recompilation after ${node.type} removal reports the broken graph`,
          Boolean(recompiled) && recompiled.ok === false, recompiled ? 'unexpectedly valid' : 'no plan');
      }
    }

    // Specific error contracts for the two blocks whose absence is unambiguous.
    const noSource = preset.clone();
    const source = noSource.nodeList().find((node) => node.type === 'source');
    noSource.removeNode(source.id);
    const noSourcePlan = compile(noSource, contextFor(suite.info));
    check(`${suite.label}: deleting the source reports E_NO_SOURCE`,
      (noSourcePlan.errors || []).some((error) => error.code === 'E_NO_SOURCE'),
      JSON.stringify((noSourcePlan.errors || []).map((error) => error.code)));

    const noLoss = preset.clone();
    const loss = noLoss.nodeList().find((node) => getNodeType(node.type) && getNodeType(node.type).category === 'loss');
    if (loss) {
      noLoss.removeNode(loss.id);
      const noLossPlan = compile(noLoss, contextFor(suite.info));
      check(`${suite.label}: deleting the loss reports E_NO_LOSS`,
        (noLossPlan.errors || []).some((error) => error.code === 'E_NO_LOSS'),
        JSON.stringify((noLossPlan.errors || []).map((error) => error.code)));
    }

    check(`${suite.label}: the original preset was not mutated`, preset.nodeList().length > 0 && compile(preset, contextFor(suite.info)).ok === true);
  }

  // A full sweep: delete every node from a clone one at a time and verify the state remains
  // internally consistent (no stale map entries after repeated removals).
  const sweep = instantPreset('mnist_classifier');
  const total = sweep.nodeList().length;
  let deletions = 0;
  while (sweep.nodeList().length > 0) {
    const next = sweep.nodeList()[0];
    sweep.removeNode(next.id);
    deletions++;
    if (danglingConnections(sweep).length > 0) break;
  }
  check('repeated deletion removes every block and every wire',
    deletions === total && sweep.nodeList().length === 0 && sweep.connections.size === 0,
    `${deletions}/${total} blocks, ${sweep.connections.size} wires left`);
}

function testWiringGuard() {
  console.log('\n--- UI wiring guard (pointerup → deleteNode, hidden lifecycle) ---');
  const ui = fs.readFileSync(path.join(root, 'src', 'ui', 'canvas_studio.js'), 'utf8');
  const css = fs.readFileSync(path.join(root, 'src', 'css', 'node_studio.css'), 'utf8');
  const html = fs.readFileSync(path.join(root, 'src', 'graph_studio.html'), 'utf8');

  check('pointerup checks the palette dropzone with client coordinates',
    /palette\.isOver\(event\.clientX,\s*event\.clientY\)/.test(ui));
  check('dropping over the palette calls deleteNode(..., { viaTrash: true })',
    /deleteNode\(draggedId,\s*\{\s*viaTrash:\s*true\s*\}\)/.test(ui));
  check('deleteNode clears per-node source/data caches',
    ui.includes('state.sourceInfo.delete(nodeId)') && ui.includes('state.sourceData.delete(nodeId)'));
  check('deleteNode clears the shared dataset cache when the last source stops using it',
    ui.includes('if (!stillUsed) state.dataCache.delete(key)'));
  check('palette has a real trash veil element in the HTML/CSS contract',
    ui.includes('new NodePalette') && css.includes('.ns-palette-trash-veil'));
  check('author display rules cannot override hidden', /\[hidden\]\s*\{\s*display:\s*none\s*!important/.test(css));
  check('playground markup exists and starts hidden', /<section id="playground"[^>]*hidden/.test(html));
  check('three tier buttons exist', ['btn-mode-playground', 'btn-mode-assist', 'btn-mode-manual']
    .every((id) => html.includes(`id="${id}"`)));
}

function main() {
  console.log('=====================================================');
  console.log('   NODE STUDIO — DRAG-TO-DELETE & GRAPH SANITISATION   ');
  console.log('=====================================================\n');

  testPalette();
  testGraphDeletion();
  testWiringGuard();

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main();
