/**
 * PHASE 3 — Node Graph Verification.
 *
 * Proves that models BUILT FROM BLOCKS (the visual graph) converge on real tasks:
 *
 *  Part 1: the MNIST preset is compiled and driven through the REAL nn_worker.js RPC path
 *          (SET_DATASET → BUILD_NETWORK → START_TRAINING → EVALUATE_TEST_SET →
 *          PREDICT_TEST_SET) on the shipped real MNIST 0-vs-1 corpus. The Confusion Matrix
 *          tap's per-sample predictions are cross-checked against the evaluator: the two
 *          accuracies must agree exactly.
 *  Part 2: the XOR preset trains to convergence in the engine.
 *  Part 3: a custom CSV dataset is resolved through the real connector stack
 *          (DatasetManager.parseCSV + streamBatches) and trained via its compiled graph.
 *  Part 4: a Class Filter block remaps a 3-class dataset (Iris) to a 2-class problem end
 *          to end — one-hot targets, filter, chain, train.
 *
 * The worker is loaded into Node with shimmed worker globals, exactly as in
 * tests/test_training_continuity.js.
 *
 * File: tests/test_node_convergence.js
 */

import { loadMnistForTests, trainEpochs, formatPercent, banner } from './mnist_test_utils.js';
import { NeuralNetwork } from '../src/engine/neural_network.js';
import { DatasetManager } from '../src/engine/dataset_manager.js';
import { GraphState } from '../src/graph/graph_state.js';
import { instantPreset, autoFixGraphToConvergence } from '../src/graph/smart_assist.js';
import { compile, applyClassFilter } from '../src/graph/node_engine.js';
import { resolveSourceDataset, datasetInfo, ensureSplit, streamBatches } from '../src/graph/data_connectors.js';

const failures = [];
const checks = { passed: 0, total: 0 };

function check(condition, label) {
  checks.total++;
  if (condition) {
    checks.passed++;
    console.log(`  [PASS] ${label}`);
  } else {
    failures.push(label);
    console.error(`  [FAIL] ${label}`);
  }
}

function installWorkerGlobals() {
  const inbox = new Map();

  globalThis.self = { onmessage: null };
  globalThis.postMessage = (message) => {
    const waiter = inbox.get(message.type);
    if (waiter) {
      inbox.delete(message.type);
      waiter.resolve(message.payload);
    }
  };

  function waitFor(type, timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        inbox.delete(type);
        reject(new Error(`Timed out after ${timeoutMs} ms waiting for worker message '${type}'`));
      }, timeoutMs);
      inbox.set(type, {
        resolve: (payload) => { clearTimeout(timer); resolve(payload); }
      });
    });
  }

  function send(type, payload, expectType) {
    const waiter = expectType ? waitFor(expectType) : null;
    globalThis.self.onmessage({ data: { type, payload } });
    return waiter;
  }

  return { send };
}

function accuracyOn(network, X, y) {
  const predictions = network.predict(X);
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const predicted = predictions[i].indexOf(Math.max(...predictions[i]));
    const truth = y[i].indexOf(Math.max(...y[i]));
    if (predicted === truth) correct++;
  }
  return correct / X.length;
}

async function main() {
  banner('NODE STUDIO — VISUAL GRAPH CONVERGENCE (Phase 3 verification)');

  /* ============================ PART 1 ============================ */
  banner('PART 1 — MNIST preset through the real worker RPC path');

  const mnist = await loadMnistForTests({ maxTrain: 2500, maxTest: 800 });
  console.log(`  Corpus: ${mnist.name}`);
  console.log(`  Train ${mnist.trainX.length} / Test ${mnist.testX.length} samples, input grid ${mnist.imageDim.width}×${mnist.imageDim.height}`);

  const bridge = installWorkerGlobals();
  await import('../src/engine/nn_worker.js');
  const { send } = bridge;

  const mnistCtx = { datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(mnist) : null) };
  const mnistGraph = instantPreset('mnist_classifier');
  const plan = compile(mnistGraph, mnistCtx);
  check(plan.ok === true, `preset graph compiles clean (${(plan.errors || []).map((e) => e.code).join(',') || 'no errors'})`);
  check(plan.meta && JSON.stringify(plan.meta.inputShape) === '[28,28]', `compiler sees the real input grid ${JSON.stringify(plan.meta && plan.meta.inputShape)}`);

  await send('BUILD_NETWORK', {
    architecture: plan.network.architecture,
    hyperparameters: {
      loss: plan.network.loss,
      optimizer: plan.network.optimizer,
      learningRate: plan.network.learningRate,
      weightDecay: plan.network.weightDecay
    }
  }, 'NETWORK_BUILT');

  const datasetReady = await send('SET_DATASET', {
    trainX: mnist.trainX,
    trainY: mnist.trainY,
    testX: mnist.testX,
    testY: mnist.testY,
    numFeatures: mnist.numFeatures,
    numClasses: mnist.numClasses,
    featureNames: mnist.featureNames,
    targetNames: mnist.targetNames
  }, 'DATASET_READY');
  check(datasetReady.trainCount === mnist.trainX.length, `worker holds the real corpus (${datasetReady.trainCount} train rows)`);

  const EPOCH_BUDGET = 10;
  const complete = await send('START_TRAINING', {
    maxEpochs: EPOCH_BUDGET,
    batchSize: plan.training.batchSize,
    epochDelayMs: 0,
    earlyStopPatience: null
  }, 'TRAINING_COMPLETE');
  check(complete.epoch === EPOCH_BUDGET, `run completed at the exact budget (epoch ${complete.epoch})`);

  const report = await send('EVALUATE_TEST_SET', {}, 'TEST_SET_EVALUATED');
  check(report.ok === true && report.accuracy >= 0.95,
    `node-built model reaches ${formatPercent(report.accuracy)} held-out accuracy in ${EPOCH_BUDGET} epochs on ${report.count} real digits (target ≥ 95%)`);

  const perSample = await send('PREDICT_TEST_SET', {}, 'TEST_SET_PREDICTIONS');
  check(perSample.ok === true && perSample.count === report.count,
    `PREDICT_TEST_SET returns one prediction per held-out sample (${perSample.count})`);

  const confusion = [[0, 0], [0, 0]];
  for (const row of perSample.predictions) confusion[row.trueClass][row.predictedClass]++;
  const confusionAccuracy = (confusion[0][0] + confusion[1][1]) / report.count;
  check(Math.abs(confusionAccuracy - report.accuracy) < 1e-12,
    `confusion-matrix accuracy equals the evaluator exactly (${formatPercent(confusionAccuracy)} = ${formatPercent(report.accuracy)})`);
  console.log(`  Confusion matrix: [[${confusion[0].join(', ')}], [${confusion[1].join(', ')}]]`);

  /* ============================ PART 2 ============================ */
  banner('PART 2 — XOR preset converges in the engine');

  const xor = DatasetManager.getBuiltinDataset('xor');
  const xorSplit = DatasetManager.splitTrainTest(xor.X, xor.y, 0.8);
  const xorCtx = { datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(xor) : null) };
  const xorPlan = compile(instantPreset('xor_classifier'), xorCtx);
  check(xorPlan.ok === true, 'XOR graph compiles clean');

  const xorNet = new NeuralNetwork({
    architecture: xorPlan.network.architecture,
    loss: xorPlan.network.loss,
    optimizer: xorPlan.network.optimizer,
    learningRate: xorPlan.network.learningRate,
    weightDecay: xorPlan.network.weightDecay
  });
  const xorHistory = trainEpochs(xorNet, xorSplit.trainX, xorSplit.trainY, { epochs: xorPlan.training.maxEpochs, batchSize: 16 });
  const xorAcc = accuracyOn(xorNet, xorSplit.testX, xorSplit.testY);
  check(xorAcc >= 0.9, `XOR converges to ${formatPercent(xorAcc)} held-out within the preset's fixed 40-epoch budget (final train loss ${xorHistory[xorHistory.length - 1].trainLoss.toFixed(5)})`);

  /* ============================ PART 3 ============================ */
  banner('PART 3 — Custom CSV dataset through the connector stack');

  // A deliberately learnable two-feature CSV with string targets ("yes"/"no").
  let seedState = 7;
  const rand = () => { seedState = (seedState * 1664525 + 1013904223) >>> 0; return seedState / 4294967296; };
  const csvLines = ['feature_a,feature_b,verdict'];
  for (let i = 0; i < 220; i++) {
    const yes = i % 2 === 0;
    const base = yes ? 6 : 1;
    csvLines.push(`${(base + rand() * 2).toFixed(3)},${(base + rand() * 2).toFixed(3)},${yes ? 'yes' : 'no'}`);
  }
  const csvDataset = DatasetManager.parseCSV(csvLines.join('\n'), { name: 'Convergence CSV' });
  check(csvDataset.type === 'classification' && csvDataset.numClasses === 2,
    `CSV parsed as 2-class classification (targets: ${csvDataset.targetNames.join('/')})`);

  const csvGraph = new GraphState('assist');
  const csvSource = csvGraph.addNode('source', { params: { datasetId: 'custom_csv' } });
  const cD1 = csvGraph.addNode('dense', { params: { size: 8 } });
  const cA1 = csvGraph.addNode('relu');
  const cD2 = csvGraph.addNode('dense', { params: { size: 6 } });
  const cA2 = csvGraph.addNode('relu');
  const cD3 = csvGraph.addNode('dense', { params: { size: 2 } });
  const cA3 = csvGraph.addNode('sigmoid');
  const cLoss = csvGraph.addNode('bce_loss');
  csvGraph.connect(csvSource.id, 'out', cD1.id, 'in');
  csvGraph.connect(cD1.id, 'out', cA1.id, 'in');
  csvGraph.connect(cA1.id, 'out', cD2.id, 'in');
  csvGraph.connect(cD2.id, 'out', cA2.id, 'in');
  csvGraph.connect(cA2.id, 'out', cD3.id, 'in');
  csvGraph.connect(cD3.id, 'out', cA3.id, 'in');
  csvGraph.connect(cA3.id, 'out', cLoss.id, 'in');

  const assist = autoFixGraphToConvergence(csvGraph, {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(csvDataset) : null)
  });
  check(assist.remainingErrors.length === 0, `Smart Assist completed the wiring (${assist.actions.map((a) => a.code).join(', ') || 'nothing to fix'})`);

  const resolved = await resolveSourceDataset(csvSource, { customDatasets: { custom_csv: csvDataset } });
  check(resolved.dataset === csvDataset, 'source resolution returns the uploaded dataset object');

  const csvCtx = { datasetInfoFor: (node) => (node.type === 'source' ? resolved.info : null) };
  const csvPlan = compile(csvGraph, csvCtx);
  check(csvPlan.ok === true, `CSV graph compiles clean (${(csvPlan.errors || []).map((e) => e.code).join(',') || 'no errors'})`);

  const csvSplit = ensureSplit(csvDataset, 0.8);
  const csvNet = new NeuralNetwork({
    architecture: csvPlan.network.architecture,
    loss: csvPlan.network.loss,
    optimizer: csvPlan.network.optimizer,
    learningRate: csvPlan.network.learningRate,
    weightDecay: csvPlan.network.weightDecay
  });
  trainEpochs(csvNet, csvSplit.trainX, csvSplit.trainY, { epochs: 30, batchSize: 16 });
  const csvAcc = accuracyOn(csvNet, csvSplit.testX, csvSplit.testY);
  check(csvAcc >= 0.95, `CSV-sourced model reaches ${formatPercent(csvAcc)} held-out accuracy (target ≥ 95%)`);

  const batches = [...streamBatches(csvSplit, { batchSize: 16, seed: 42 })];
  const batchSampleTotal = batches.reduce((sum, batch) => sum + batch.size, 0);
  check(batches.length === Math.ceil(csvSplit.trainX.length / 16), `streamBatches yields ${batches.length} real mini-batches`);
  check(batchSampleTotal === csvSplit.trainX.length, `streamed batches cover every training row exactly once (${batchSampleTotal})`);
  check(batches.every((batch) => batch.batchX.every((row) => row.length === 2) && batch.batchY.every((row) => row.length === 2)),
    'every streamed batch carries 2-feature tensors and 2-value one-hot targets');

  /* ============================ PART 4 ============================ */
  banner('PART 4 — Class Filter remaps 3-class Iris to a 2-class problem');

  const iris = DatasetManager.getBuiltinDataset('iris');
  const irisSubset = applyClassFilter(iris, [0, 1]);
  const irisSplit = DatasetManager.splitTrainTest(irisSubset.X, irisSubset.y, 0.8);
  check(irisSubset.numClasses === 2 && irisSubset.targetNames.length === 2,
    `filter keeps 2 of 3 classes (${irisSubset.targetNames.join('/')}, ${irisSubset.X.length} rows)`);

  const irisGraph = new GraphState('assist');
  const iSource = irisGraph.addNode('source', { params: { datasetId: 'iris' } });
  const iFilter = irisGraph.addNode('class_filter', { params: { keep: '0,1' } });
  const iD1 = irisGraph.addNode('dense', { params: { size: 12 } });
  const iA1 = irisGraph.addNode('relu');
  const iD2 = irisGraph.addNode('dense', { params: { size: 2 } });
  const iA2 = irisGraph.addNode('sigmoid');
  const iLoss = irisGraph.addNode('bce_loss');
  irisGraph.connect(iSource.id, 'out', iFilter.id, 'in');
  irisGraph.connect(iFilter.id, 'out', iD1.id, 'in');
  irisGraph.connect(iD1.id, 'out', iA1.id, 'in');
  irisGraph.connect(iA1.id, 'out', iD2.id, 'in');
  irisGraph.connect(iD2.id, 'out', iA2.id, 'in');
  irisGraph.connect(iA2.id, 'out', iLoss.id, 'in');
  autoFixGraphToConvergence(irisGraph, {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(iris) : null)
  });

  const irisPlan = compile(irisGraph, {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(iris) : null)
  });
  check(irisPlan.ok === true && irisPlan.data.filter && JSON.stringify(irisPlan.data.filter.keep) === '[0,1]',
    `filtered graph compiles clean and carries the filter (${(irisPlan.errors || []).map((e) => e.code).join(',') || 'no errors'})`);

  const irisNet = new NeuralNetwork({
    architecture: irisPlan.network.architecture,
    loss: irisPlan.network.loss,
    optimizer: irisPlan.network.optimizer,
    learningRate: irisPlan.network.learningRate,
    weightDecay: irisPlan.network.weightDecay
  });
  trainEpochs(irisNet, irisSplit.trainX, irisSplit.trainY, { epochs: 40, batchSize: 16 });
  const irisAcc = accuracyOn(irisNet, irisSplit.testX, irisSplit.testY);
  check(irisAcc >= 0.9, `filtered Iris model reaches ${formatPercent(irisAcc)} held-out accuracy (target ≥ 90%)`);

  /* ------------------------------------------------------------------ */
  console.log('\n=====================================================');
  console.log(`   RESULT: ${checks.passed}/${checks.total} passed`);
  if (failures.length > 0) {
    console.log('   Failed checks:');
    for (const failure of failures) console.log(`     - ${failure}`);
    process.exit(1);
  }
  process.exit(0);
}

main().catch((err) => {
  console.error('\n[FATAL]', err);
  process.exit(1);
});
