/**
 * TRAINING LIFECYCLE + INPUT-SHAPE GUARD TESTS
 *
 * Regression fences for two field-reported defects:
 *
 *  1. "The AI trains for a minute and then stops."
 *     A Start run used to be hard-capped at 500 epochs, and clicking Start again did nothing
 *     (currentEpoch >= maxEpochs completed instantly). Runs are now unbounded and auto-stop
 *     on convergence: the worker keeps training until the held-out loss has not improved for
 *     `earlyStopPatience` epochs (or the user pauses). This suite drives the real nn_worker.js
 *     past epoch 500 with maxEpochs:null to prove there is no hidden cap, then verifies the
 *     convergence auto-stop, and that finite maxEpochs still caps exactly (harness contract).
 *
 *  2. "Could not add sample: Expected a 2-value input tensor."
 *     The Digit Studio pad emits a 784-value tensor; with a 2-input dataset loaded, the engine
 *     silently computed predictions from the first 2 values of that tensor (forwardSample
 *     truncates) and every correction/add was rejected with a cryptic error. The worker now
 *     refuses every mismatched tensor with an explicit message on all three RPC paths
 *     (prediction / online step / add sample), and the UI switches the studio off for
 *     non-28x28 datasets (see canvas_feedback.setTaskCompatibility - DOM-side, not tested here).
 *
 *  3. "How can I save my training progress?" - the studio could export a model but never load
 *     one back. Part 4 drives the exact export payload (ModelExporter.buildExportPayload)
 *     through the worker's IMPORT_MODEL RPC and verifies the restored network is
 *     weight-identical, keeps its saved loss / learning rate / epoch counter, rejects
 *     malformed files without touching the live network, and continues training from the
 *     restored epoch.
 *
 * The worker is loaded into Node with shimmed worker globals, exactly as in
 * tests/test_online_feedback.js, so the real RPC flow is exercised end to end.
 *
 * File: tests/test_training_continuity.js
 */

import { DatasetManager } from '../src/engine/dataset_manager.js';
import { NeuralNetwork } from '../src/engine/neural_network.js';
import { ModelExporter } from '../src/ui/export_import.js';
import { banner } from './mnist_test_utils.js';

const ARCHITECTURE = [
  { size: 2, activation: 'linear' },
  { size: 8, activation: 'relu' },
  { size: 6, activation: 'relu' },
  { size: 2, activation: 'sigmoid' }
];
const HYPERPARAMS = { loss: 'binary_crossentropy', optimizer: 'adam', learningRate: 0.03, weightDecay: 0.0001 };

const CEILING_PROBE_EPOCH = 500; // the old hard cap this suite must sail past

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function installWorkerGlobals() {
  const inbox = new Map();
  const log = [];

  globalThis.self = { onmessage: null };
  globalThis.postMessage = (message) => {
    log.push(message);
    const waiter = inbox.get(message.type);
    if (waiter) {
      inbox.delete(message.type);
      waiter.resolve(message.payload);
    }
  };

  function waitFor(type, timeoutMs = 60000) {
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

  async function send(type, payload, expectType) {
    const waiter = expectType ? waitFor(expectType) : null;
    globalThis.self.onmessage({ data: { type, payload } });
    return waiter ? waiter : null;
  }

  return { send, log };
}

async function main() {
  banner('TRAINING CONTINUITY + INPUT-SHAPE GUARDS (real worker module)');

  const workerBridge = installWorkerGlobals();
  await import('../src/engine/nn_worker.js');
  const { send, log } = workerBridge;

  // ---- Build a small 2-input network + dataset (same shapes as the Screenshot bug) --------
  const dataset = DatasetManager.getBuiltinDataset('xor');
  const split = DatasetManager.splitTrainTest(dataset.X, dataset.y, 0.8);

  const built = await send('BUILD_NETWORK', { architecture: ARCHITECTURE, hyperparameters: HYPERPARAMS }, 'NETWORK_BUILT');
  check(built.weights.length === ARCHITECTURE.length - 1 && built.weights[0].inputSize === 2,
    'network built with a 2-value input layer (the shape the studio used to violate)');

  const datasetReady = await send('SET_DATASET', {
    trainX: split.trainX,
    trainY: split.trainY,
    testX: split.testX,
    testY: split.testY,
    numFeatures: 2,
    numClasses: 2,
    featureNames: ['x', 'y'],
    targetNames: ['Class A', 'Class B']
  }, 'DATASET_READY');
  const initialTrainCount = datasetReady.trainCount;
  check(initialTrainCount === split.trainX.length, `worker received ${initialTrainCount} training samples`);

  // ---- Part 1: input-shape guards ----------------------------------------------------------
  banner('PART 1 - input-shape guards (the screenshot bug)');

  const wrongPrediction = await send('RUN_PREDICTION', { input: new Array(784).fill(0.25), source: 'studio' }, 'ERROR');
  check(/784/.test(wrongPrediction.message) && /expects 2/.test(wrongPrediction.message),
    `prediction with a 784-value tensor is refused with the real sizes ("${wrongPrediction.message.slice(0, 60)}...")`);

  const goodPrediction = await send('RUN_PREDICTION', { input: [0.5, -0.5], source: 'studio' }, 'PREDICTION_RESULT');
  check(goodPrediction.output.length === 2 && Math.abs(goodPrediction.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-9,
    'a correctly-shaped tensor still gets a real forward pass (probabilities sum to 1)');

  const badStep = await send('ONLINE_STEP', { sample: new Array(784).fill(0.25), target: [1, 0], steps: 1 }, 'ONLINE_STEP_COMPLETE');
  check(badStep.ok === false && /784/.test(badStep.error) && /expects 2/.test(badStep.error),
    'online correction with a 784-value tensor is rejected before touching any weight');

  const badAdd = await send('ADD_TRAINING_SAMPLE', { sample: new Array(784).fill(0.25), target: [1, 0], label: 0 }, 'SAMPLE_ADDED');
  check(badAdd.ok === false && /784/.test(badAdd.error) && /expects 2/.test(badAdd.error),
    'add-sample with a 784-value tensor reports both sizes instead of "Expected a 2-value input tensor."');

  const goodStep = await send('ONLINE_STEP', { sample: [1, 0], target: [1, 0], steps: 1 }, 'ONLINE_STEP_COMPLETE');
  check(goodStep.ok === true && Number.isFinite(goodStep.initialLoss) && Number.isFinite(goodStep.postLoss),
    'a correctly-shaped correction still runs a real bounded gradient step');

  const goodAdd = await send('ADD_TRAINING_SAMPLE', { sample: [1, 0], target: [1, 0], label: 0 }, 'SAMPLE_ADDED');
  check(goodAdd.ok === true && goodAdd.totalSamples === initialTrainCount + 1,
    'a correctly-shaped sample is still appended to the live training set');

  // ---- Part 2: no hidden epoch cap ---------------------------------------------------------
  banner('PART 2 - unbounded runs (no more 500-epoch wall)');

  const firstCap = await send('START_TRAINING', { maxEpochs: 4, batchSize: 16, epochDelayMs: 0, earlyStopPatience: null }, 'TRAINING_COMPLETE');
  check(firstCap.epoch === 4 && firstCap.reason === 'epoch-cap',
    `finite maxEpochs still caps exactly (epoch ${firstCap.epoch}, reason ${firstCap.reason})`);

  const secondCap = await send('START_TRAINING', { maxEpochs: 9, batchSize: 16, epochDelayMs: 0, earlyStopPatience: null }, 'TRAINING_COMPLETE');
  check(secondCap.epoch === 9 && secondCap.reason === 'epoch-cap',
    `a second Start continues the same run past its previous cap (epoch 4 -> ${secondCap.epoch})`);

  // Unbounded run: must sail past the old 500-epoch wall without completing.
  const probeStartIndex = log.length;
  await send('START_TRAINING', { maxEpochs: null, earlyStopPatience: null, batchSize: 16, epochDelayMs: 0 });

  let latestStep = null;
  let completedEarly = false;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const recent = log.slice(probeStartIndex);
    completedEarly = recent.some((m) => m.type === 'TRAINING_COMPLETE');
    if (completedEarly) break;
    const steps = recent.filter((m) => m.type === 'TRAINING_STEP');
    latestStep = steps.length > 0 ? steps[steps.length - 1].payload : null;
    if (latestStep && latestStep.epoch > CEILING_PROBE_EPOCH) break;
    await sleep(25);
  }

  check(!completedEarly, 'an unbounded run does not complete on its own (no hidden epoch cap)');
  check(latestStep !== null && latestStep.epoch > CEILING_PROBE_EPOCH,
    `training sailed past the old 500-epoch cap (latest reported epoch: ${latestStep ? latestStep.epoch : 'none'})`);
  check(latestStep !== null && typeof latestStep.stallEpochs === 'number' && latestStep.bestValLoss !== null,
    'live TRAINING_STEP payloads carry the real convergence readout (stallEpochs / bestValLoss)');

  const paused = await send('PAUSE_TRAINING', {}, 'TRAINING_PAUSED');
  check(paused.epoch > CEILING_PROBE_EPOCH, `pause reports the true epoch beyond 500 (${paused.epoch})`);

  // ---- Part 3: convergence auto-stop -------------------------------------------------------
  banner('PART 3 - convergence auto-stop (train while it still helps)');

  const converged = await send('START_TRAINING', { maxEpochs: null, earlyStopPatience: 30, batchSize: 16, epochDelayMs: 0 }, 'TRAINING_COMPLETE');
  check(converged.reason === 'converged' && converged.stallEpochs === 30,
    `run auto-stopped on convergence after ${converged.stallEpochs} flat epochs (reason: ${converged.reason})`);
  check(Number.isFinite(converged.bestValLoss) && converged.bestValLoss >= 0,
    `best held-out loss is a real measured number (${converged.bestValLoss})`);
  check(converged.epoch >= paused.epoch + 30,
    `convergence check cannot stop before patience elapses (epoch ${converged.epoch} >= ${paused.epoch} + 30)`);

  const report = await send('EVALUATE_TEST_SET', {}, 'TEST_SET_EVALUATED');
  check(report.ok === true && report.accuracy >= 0.9,
    `the long-run model is genuinely trained: held-out accuracy ${(report.accuracy * 100).toFixed(1)}% on ${report.count} samples`);

  // ---- Part 4: model save/load round-trip (the "Load Model" contract) ----------------------
  banner('PART 4 - model save/load round-trip (Load Model contract)');

  // A reference network plus the exact payload the studio's Export JSON button writes.
  const refNet = new NeuralNetwork({
    architecture: ARCHITECTURE,
    loss: 'binary_crossentropy',
    optimizer: 'adam',
    learningRate: 0.03,
    weightDecay: 0.0001
  });
  const exported = ModelExporter.buildExportPayload(
    refNet.getWeightsSnapshot(),
    refNet.architecture,
    { dataset: 'XOR Problem', datasetId: 'xor', epochs: 42 },
    { loss: 'binary_crossentropy', optimizer: 'adam', learningRate: 0.03, weightDecay: 0.0001 }
  );
  const reloaded = JSON.parse(JSON.stringify(exported)); // exactly what a saved file goes through
  check(reloaded.format === 'NeuralNetworkStudio-v1' && reloaded.loss === 'binary_crossentropy' && reloaded.metadata.epochs === 42,
    'export payload round-trips through JSON with format tag, training config and epoch metadata');

  const imported = await send('IMPORT_MODEL', reloaded, 'MODEL_IMPORTED');
  check(imported.epoch === 42 && imported.loss === 'binary_crossentropy' && imported.learningRate === 0.03,
    `imported model resumes the saved configuration (epoch ${imported.epoch}, loss ${imported.loss}, lr ${imported.learningRate})`);

  const reference = refNet.getWeightsSnapshot();
  const weightsMatch = imported.weights.length === reference.length &&
    imported.weights.every((layer, l) =>
      layer.weights.every((row, i) => row.every((value, j) => value === reference[l].weights[i][j])) &&
      layer.biases.every((value, i) => value === reference[l].biases[i]));
  check(weightsMatch, 'every imported weight and bias is bit-identical to the saved model');

  const probe = [0.35, -0.6];
  const expectedOutput = refNet.forwardSample(probe).output;
  const probePrediction = await send('RUN_PREDICTION', { input: probe, source: 'test' }, 'PREDICTION_RESULT');
  check(probePrediction.output.every((value, i) => Math.abs(value - expectedOutput[i]) < 1e-12),
    'the imported network predicts identically to the saved reference model');

  const badLayers = await send('IMPORT_MODEL', Object.assign({}, reloaded, { layers: reloaded.layers.slice(0, 1) }), 'ERROR');
  check(/weight layers/.test(badLayers.message), 'a truncated weight file is rejected with the real layer count');

  const badShapeLayers = reloaded.layers.map((layer, index) => index === 0
    ? Object.assign({}, layer, { weights: layer.weights.slice(0, 1) })
    : layer);
  const badShape = await send('IMPORT_MODEL', Object.assign({}, reloaded, { layers: badShapeLayers }), 'ERROR');
  check(/does not match the architecture/.test(badShape.message), 'a shape-mismatched layer is rejected before touching any weight');

  const survived = await send('RUN_PREDICTION', { input: probe, source: 'test' }, 'PREDICTION_RESULT');
  check(survived.output.every((value, i) => Math.abs(value - expectedOutput[i]) < 1e-12),
    'a failed import leaves the current network untouched');

  const continued = await send('START_TRAINING', { maxEpochs: 45, batchSize: 16, epochDelayMs: 0, earlyStopPatience: null }, 'TRAINING_COMPLETE');
  check(continued.epoch === 45 && continued.reason === 'epoch-cap',
    `training continues from the restored epoch (42 -> ${continued.epoch})`);

  const afterTraining = await send('RUN_PREDICTION', { input: probe, source: 'test' }, 'PREDICTION_RESULT');
  check(afterTraining.output.some((value, i) => Math.abs(value - expectedOutput[i]) > 1e-9),
    'continued training measurably moved the imported weights');

  banner(failures.length === 0 ? 'TRAINING CONTINUITY TEST PASSED' : 'TRAINING CONTINUITY TEST FAILED');
  console.log(`  checks passed: ${checks.passed}/${checks.total}`);
  if (failures.length === 0) {
    console.log('  [PASS] Unbounded training + convergence auto-stop + input-shape guards + model save/load round-trip verified end-to-end\n');
    return 0;
  }
  for (const failure of failures) console.error(`  [FAIL] ${failure}`);
  console.error('');
  return 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error('\n[FAIL] Training continuity test crashed:', err);
    process.exit(1);
  });
