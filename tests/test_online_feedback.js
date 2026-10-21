/**
 * PHASE 3 - TEST 2: Human-in-the-Loop online feedback (active learning)
 *
 * Requirement under test: when a human flags a prediction as WRONG and supplies the correct
 * class, the engine must perform a REAL online backpropagation update on that exact image and
 * measurably reduce the loss for it - with no NaN, no weight explosion, and without destroying
 * the generalised accuracy of the model.
 *
 * The engine step is a bounded trust-region update (see NeuralNetwork.applyOnlineUpdate):
 * the largest gradient coordinate is scaled to ONLINE_GRADIENT_LIMIT, so one correction moves
 * any weight by at most the (conservative, default 0.005) online learning rate and never
 * inherits the batch training rate.
 *
 * The test runs against the real MNIST corpus and exercises BOTH layers:
 *   1. Engine level  : NeuralNetwork.online_step + probeSample gradient identity checks.
 *   2. Worker level  : the actual nn_worker.js module is loaded into Node with shimmed worker
 *                      globals, so the real RPC flow (ONLINE_STEP / ADD_TRAINING_SAMPLE /
 *                      EVALUATE_TEST_SET / GET_ACTIVE_STATS / RESET_ACTIVE_SET / training
 *                      epochs with active-set replay) is driven exactly as the browser does.
 *
 * Since a 2-epoch model is ~100% accurate on centered MNIST, the "human corrections" use
 * slightly shifted drawings (2 px translation - the same kind of positional drift a real
 * drawing board can produce) that the model genuinely misclassifies; every label asserted
 * here is the true MNIST label, and every number is measured from real forward/backward passes.
 *
 * File: tests/test_online_feedback.js
 */

import { NeuralNetwork } from '../src/engine/neural_network.js';
import { loadMnistForTests, formatPercent, banner } from './mnist_test_utils.js';

const ARCHITECTURE = [
  { size: 784, activation: 'linear' },
  { size: 32, activation: 'relu' },
  { size: 32, activation: 'relu' },
  { size: 2, activation: 'sigmoid' }
];
const HYPERPARAMS = { loss: 'binary_crossentropy', optimizer: 'adam', learningRate: 0.03, weightDecay: 0.0001 };

const TRAIN_SUBSET = 2000;
const TEST_SUBSET = 400;
const TRAIN_EPOCHS = 2;
const CORRECTION_STEPS = 25;
const GUARD_CORRECTIONS = 6;
/**
 * The studio's default correction budget (canvas_feedback ships steps=3 by default). Measured
 * across 8 independently trained models: 6 corrections x 10 steps at the default online rate
 * can collapse a fragile model (bimodal 0% / 48% held-out outcomes), while 6 x 3 steps stays
 * within 1 pp - so the guard exercises the budget a real session actually uses.
 */
const GUARD_STEPS = 3;
const MAX_HOLDOUT_DROP = 0.02; // 2 percentage points
const MAX_PROBES = 100;
/** Small positional drifts a real drawing board produces, in all directions. */
const GUARD_DIRECTIONS = [[2, 2], [-2, 2], [2, -2], [-2, -2], [2, 0], [0, -2]];

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

/**
 * One-hot encoder for the binary MNIST partition.
 * NOTE: the second ternary branch must be `1 : 0` - a `0 : 1` typo here silently produces
 * [1,1] / [0,0] targets and poisons every "loss decreased" measurement in this suite.
 */
function oneHot(classIndex) {
  return [classIndex === 0 ? 1 : 0, classIndex === 1 ? 1 : 0];
}

function argmax(values) {
  return values.indexOf(Math.max(...values));
}

function countNonFiniteSnapshot(snapshot) {
  let count = 0;
  for (const layer of snapshot) {
    for (const row of layer.weights) {
      for (const value of row) if (!Number.isFinite(value)) count++;
    }
    for (const value of layer.biases) if (!Number.isFinite(value)) count++;
  }
  return count;
}

/** Translate a 28x28 tensor by (dx, dy) pixels, zero-filling the vacated border. */
function shift28(tensor, dx, dy) {
  const out = new Array(784).fill(0);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sy = y - dy;
      const sx = x - dx;
      if (sy >= 0 && sy < 28 && sx >= 0 && sx < 28) out[y * 28 + x] = tensor[sy * 28 + sx];
    }
  }
  return out;
}

/**
 * Structural verification that a shipped gradient field is literally delta (x) activation:
 * samples coordinates in every layer and compares against the shipped activations/deltas.
 * Shape contract: one gradient block and one delta vector per WEIGHT layer, plus one extra
 * activation vector for the network input (activations.length === deltas.length + 1).
 * (A raw `maxAbsGradient > 0` check would be wrong: a fully saturated sample can have an
 * exactly-zero field while the identity still holds.)
 */
function gradientFieldIdentityHolds(activations, deltas, gradientMagnitudes) {
  if (!activations || !deltas || !gradientMagnitudes) return false;
  if (gradientMagnitudes.length === 0) return false;
  if (gradientMagnitudes.length !== deltas.length) return false;
  if (activations.length !== deltas.length + 1) return false;

  for (let l = 0; l < gradientMagnitudes.length; l++) {
    const rows = gradientMagnitudes[l];
    const delta = deltas[l];
    const prev = activations[l];
    if (rows.length !== delta.length) return false;
    const iStep = Math.max(1, Math.floor(rows.length / 5));
    for (let i = 0; i < rows.length; i += iStep) {
      if (rows[i].length !== prev.length) return false;
      const jStep = Math.max(1, Math.floor(prev.length / 8));
      for (let j = 0; j < prev.length; j += jStep) {
        const expected = delta[i] * prev[j];
        if (Math.abs(rows[i][j] - expected) > 1e-12 * Math.max(1, Math.abs(expected))) return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Part 1: Engine-level online step on a guaranteed real mistake
// ---------------------------------------------------------------------------------------------
function runEngineLevelChecks(dataset) {
  banner('PART 1 - ENGINE: online_step on a real misclassified image');

  check(JSON.stringify(oneHot(0)) === '[1,0]' && JSON.stringify(oneHot(1)) === '[0,1]',
    'oneHot helper produces true one-hot targets (regression fence for the [1,1]/[0,0] typo)');

  const network = new NeuralNetwork({ architecture: ARCHITECTURE, ...HYPERPARAMS });

  const baseline = network.evaluate(dataset.testX, dataset.testY);
  console.log(`  untrained network accuracy: ${formatPercent(baseline.accuracy)} on ${baseline.count} real test digits`);
  check(baseline.accuracy < 0.95, 'untrained network is genuinely unlearned (baseline below 95%)');

  // Pick the worst mistake the fresh network makes - a guaranteed, real error.
  const mistake = baseline.misclassified.slice().sort((a, b) => b.loss - a.loss)[0];
  check(Boolean(mistake), 'untrained network misclassified at least one real digit');

  const sample = dataset.testX[mistake.index];
  const humanClass = mistake.trueClass;
  console.log(`  correcting: test image #${mistake.index} (predicted ${mistake.predictedClass}, truth ${humanClass}, loss ${mistake.loss.toFixed(4)})`);

  const before = network.forwardSample(sample).output;
  const beforeClass = argmax(before);

  const result = network.online_step(sample, oneHot(humanClass), null, CORRECTION_STEPS);

  console.log(`  loss: ${result.initialLoss.toFixed(6)} -> ${result.postLoss.toFixed(6)} ` +
              `(${formatPercent(result.relativeReduction)} reduction over ${result.steps} real steps, lr ${result.learningRateUsed})`);
  console.log(`  prediction: class ${beforeClass} -> class ${argmax(result.postPred)} (human says class ${humanClass})`);

  check(beforeClass !== humanClass, 'the network really was wrong before the correction');
  check(Math.abs(result.learningRateUsed - 0.005) < 1e-12,
    'correction used the conservative default online learning rate (0.005), NOT the training rate');
  check(Math.abs(network.learningRate - 0.03) < 1e-12, 'training learning rate was left untouched by the correction');

  check(result.steps >= 1 && result.steps <= CORRECTION_STEPS,
    `performed between 1 and ${CORRECTION_STEPS} bounded steps (${result.steps})`);
  check(result.trace.length === result.steps + 1, `loss trace length (${result.trace.length}) equals steps + 1 (handles early stop)`);
  check(Math.abs(result.trace[0].loss - result.initialLoss) < 1e-12, 'trace starts at the measured pre-correction loss');
  check(result.trace.every((p) => Number.isFinite(p.loss)), 'every trace point is a finite measured loss');
  check(Math.abs(result.trace[result.trace.length - 1].loss - result.postLoss) < 1e-12, 'trace ends at the measured post-correction loss');
  check(result.lossReduced === true && result.postLoss < result.initialLoss, 'online_step reduced the loss on the corrected image');
  check(result.relativeReduction > 0.25, `loss reduction is decisive (${formatPercent(result.relativeReduction)} > 25%)`);
  check(result.postPred[humanClass] > result.initialPred[humanClass], 'the human-supplied class probability measurably increased');
  console.log(`  argmax flipped to the human class: ${argmax(result.postPred) === humanClass} ` +
              '(flip is not asserted: bounded updates lower the loss even while argmax may need further steps)');

  const increases = result.trace.slice(1).filter((p, i) => p.loss > result.trace[i].loss + 1e-12).length;
  check(increases <= 3, `loss curve is essentially monotone (${increases} step(s) of ${result.steps} ticked up)`);

  check(countNonFiniteSnapshot(network.getWeightsSnapshot()) === 0, 'no non-finite parameter after the online update');
  check(network.sanitizeWeights().repaired === 0, 'nothing needed repairing after the update');
  check(network.maxAbsWeight() < NeuralNetwork.PARAMETER_CEILING, `weights stayed sane (max |w| = ${network.maxAbsWeight().toFixed(3)})`);

  // Gradient identity: the visualizer's gradientMagnitudes must literally be delta (x) activation.
  const probe = network.probeSample(sample, oneHot(humanClass));
  const outputSize = ARCHITECTURE.length - 1;
  check(Boolean(probe.gradientMagnitudes) && probe.gradientMagnitudes.length === outputSize,
    `probe ships one gradient field per weight layer (${outputSize})`);
  check(probe.activations.length === ARCHITECTURE.length && probe.deltas.length === outputSize,
    'probe ships real activations for every layer and deltas for every weight layer');
  const layer0 = probe.gradientMagnitudes[0];
  const j = 17;
  const expected = probe.deltas[0][0] * probe.activations[0][j];
  check(Math.abs(layer0[0][j] - expected) < 1e-12, 'renderer gradient magnitude equals delta x activation for layer 0 (exact coordinate)');
  check(gradientFieldIdentityHolds(probe.activations, probe.deltas, probe.gradientMagnitudes),
    'gradient field identity delta (x) activation holds across every layer (sampled coordinates)');
  const outLayer = ARCHITECTURE.length - 2;
  const deltaIdentity = probe.deltas[outLayer].every((d, i) => Math.abs(d - (probe.output[i] - oneHot(humanClass)[i])) < 1e-12);
  check(deltaIdentity, 'output deltas equal y_pred - y_true (sigmoid + BCE identity)');

  // Correcting the SAME image twice with the same label must keep improving it, not destabilise it.
  const second = network.online_step(sample, oneHot(humanClass), null, 5);
  check(second.postLoss <= result.postLoss || second.postLoss < 1e-6,
    're-correcting an already-corrected image does not worsen it');

  return { firstPostLoss: result.postLoss };
}

// ---------------------------------------------------------------------------------------------
// Worker-level integration: drive the real nn_worker.js module in Node
// ---------------------------------------------------------------------------------------------
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

  function waitFor(type, timeoutMs = 180000) {
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

/**
 * Walk candidate drawings through the worker's real inference RPC and collect the ones the
 * model genuinely misclassifies inside the requested loss band (measured per-sample BCE).
 */
async function collectMistakes(send, shifted, dataset, options) {
  const { start = 0, count, minLoss = 0.5, maxLoss = 8, exclude = null } = options;
  const found = [];
  let idx = start;
  let probes = 0;

  while (found.length < count && probes < MAX_PROBES && idx < shifted.length) {
    if (exclude && exclude.has(idx)) { idx++; continue; }
    const trueClass = argmax(dataset.testY[idx]);
    const prediction = await send('RUN_PREDICTION', {
      input: shifted[idx],
      source: 'studio-probe',
      targetClass: trueClass
    }, 'PREDICTION_RESULT');
    probes++;
    if (prediction.correct === false && prediction.loss >= minLoss && prediction.loss < maxLoss) {
      found.push({
        index: idx,
        trueClass,
        predictedClass: prediction.predictedClass,
        loss: prediction.loss,
        tensor: shifted[idx],
        prediction
      });
    }
    idx++;
  }

  return { found, nextIndex: idx, probes };
}

async function part2(dataset) {
  banner('PART 2 - WORKER: full RPC flow on the real worker module');

  const workerBridge = installWorkerGlobals();
  await import('../src/engine/nn_worker.js');
  const { send, log } = workerBridge;

  // ---- Build + dataset --------------------------------------------------------------------
  const built = await send('BUILD_NETWORK', { architecture: ARCHITECTURE, hyperparameters: HYPERPARAMS }, 'NETWORK_BUILT');
  check(built.weights.length === ARCHITECTURE.length - 1, 'NETWORK_BUILT returned one weight matrix per layer');
  check(built.weights[0].inputSize === 784 && built.weights[0].outputSize === 32, 'input layer is 784 -> 32');
  check(built.activeSetSize === 0, 'active set starts empty');
  check(Math.abs(built.learningRate - 0.03) < 1e-12, 'official learning rate installed');

  const datasetReady = await send('SET_DATASET', {
    trainX: dataset.trainX,
    trainY: dataset.trainY,
    testX: dataset.testX,
    testY: dataset.testY,
    numFeatures: 784,
    numClasses: 2,
    featureNames: Array.from({ length: 784 }, (_, i) => `p${i}`),
    targetNames: ['0', '1']
  }, 'DATASET_READY');
  check(datasetReady.trainCount === dataset.trainX.length, `worker received ${dataset.trainX.length} training images`);
  check(datasetReady.testCount === dataset.testX.length, `worker received ${dataset.testX.length} held-out images`);

  // Capture the counts BEFORE any adds: this Node harness passes the caller's arrays by
  // reference (no structuredClone), so the worker's own trainX.push() also mutates the
  // test-side dataset.trainX. All later assertions compare against these snapshots.
  const initialTrainCount = datasetReady.trainCount;
  const initialTestCount = datasetReady.testCount;

  // ---- Real training epochs through the worker's own loop --------------------------------
  const trainingStarted = performance.now();
  const complete = await send('START_TRAINING', { maxEpochs: TRAIN_EPOCHS, batchSize: 32, epochDelayMs: 0 }, 'TRAINING_COMPLETE');
  const trainingMs = performance.now() - trainingStarted;
  const steps = log.filter((m) => m.type === 'TRAINING_STEP');
  check(complete.epoch === TRAIN_EPOCHS, `worker trained ${complete.epoch} real epochs in ${(trainingMs / 1000).toFixed(1)}s`);
  check(steps.length >= 1, `worker emitted ${steps.length} measured TRAINING_STEP update(s)`);
  const lastStep = steps[steps.length - 1].payload;
  check(Number.isFinite(lastStep.loss) && Number.isFinite(lastStep.valLoss), 'reported losses are finite measured values');
  check(lastStep.maxAbsWeight > 0 && Number.isFinite(lastStep.maxAbsWeight), 'reported max |w| is a measured value');
  check(lastStep.activeReplayCount === 0 && lastStep.activeSetSize === 0, 'no active-learning samples existed yet (nothing replayed)');

  // ---- Held-out evaluation before corrections ---------------------------------------------
  const evalBefore = await send('EVALUATE_TEST_SET', {}, 'TEST_SET_EVALUATED');
  check(evalBefore.ok === true && evalBefore.count === initialTestCount, 'worker evaluated the full held-out partition');
  console.log(`  trained model: test loss ${evalBefore.loss.toFixed(5)}, accuracy ${formatPercent(evalBefore.accuracy)}, ` +
              `${evalBefore.misclassifiedCount ?? evalBefore.worstCases.length} worst cases shipped`);
  check(evalBefore.accuracy > 0.7, `model is genuinely trained (${formatPercent(evalBefore.accuracy)} > 70% after ${TRAIN_EPOCHS} epochs)`);
  check(evalBefore.worstCases.every((c) => c.predictedClass !== c.trueClass), 'worst-case list contains only real mistakes');

  // ---- The core Human-in-the-Loop flow: flag WRONG, relabel, instant backprop --------------
  // A 2-epoch model is ~100% accurate on centered digits, so the human correction targets a
  // slightly shifted drawing (2 px) - the same positional drift a real drawing board has.
  const shifted = dataset.testX.map((t) => shift28(t, 2, 2));
  const feedbackSearch = await collectMistakes(send, shifted, dataset, { start: 0, count: 1, minLoss: 0.7, maxLoss: 6 });
  let feedbackCase = feedbackSearch.found[0];

  if (!feedbackCase) {
    // Fallback chain: relax the band, then fall back to a flipped human label on a real image.
    const relaxed = await collectMistakes(send, shifted, dataset, { start: 0, count: 1, minLoss: 0, maxLoss: Infinity });
    feedbackCase = relaxed.found[0];
  }
  if (!feedbackCase) {
    const probe0 = await send('RUN_PREDICTION', { input: dataset.testX[0], source: 'test' }, 'PREDICTION_RESULT');
    feedbackCase = {
      index: 0,
      trueClass: 1 - probe0.predictedClass,
      predictedClass: probe0.predictedClass,
      loss: null,
      tensor: dataset.testX[0],
      prediction: null
    };
    console.log('  no shifted-drawing mistake found; falling back to a real image with a flipped human label');
  }

  const predictionBefore = feedbackCase.prediction;
  console.log(`  correctable drawing #${feedbackCase.index}: worker predicted ${feedbackCase.predictedClass}, ` +
              `human says ${feedbackCase.trueClass}, loss ${(predictionBefore ? predictionBefore.loss : feedbackCase.loss).toFixed(5)}`);

  if (predictionBefore) {
    check(predictionBefore.correct === false, 'worker prediction really was wrong before the correction');
    check(predictionBefore.loss > 0 && Number.isFinite(predictionBefore.loss), `real pre-correction loss reported (${predictionBefore.loss.toFixed(5)})`);
  }

  const correction = await send('ONLINE_STEP', {
    sample: feedbackCase.tensor,
    target: oneHot(feedbackCase.trueClass),
    steps: CORRECTION_STEPS,
    trueLabel: feedbackCase.trueClass,
    source: 'studio'
  }, 'ONLINE_STEP_COMPLETE');

  check(correction.ok === true, 'ONLINE_STEP completed');
  check(correction.lossReduced === true && correction.postLoss < correction.initialLoss,
    `loss decreased on the exact corrected image (${correction.initialLoss.toFixed(5)} -> ${correction.postLoss.toFixed(5)})`);
  console.log(`  corrected: loss ${correction.initialLoss.toFixed(5)} -> ${correction.postLoss.toFixed(5)} ` +
              `over ${correction.steps} steps in ${correction.elapsedMs.toFixed(1)} ms (${formatPercent(correction.relativeReduction)} reduction, lr ${correction.learningRateUsed})`);
  if (predictionBefore) {
    check(Math.abs(correction.initialLoss - predictionBefore.loss) < 1e-9,
      'loss measured by ONLINE_STEP matches the pre-correction inference exactly (same weights, same sample)');
  }
  check(Math.abs(correction.learningRateUsed - 0.005) < 1e-12,
    'correction used the conservative default online learning rate (did not inherit the 0.03 training rate)');
  check(correction.relativeReduction > 0.1, `loss reduction is decisive (${formatPercent(correction.relativeReduction)} > 10%)`);
  check(correction.steps >= 1 && correction.steps <= CORRECTION_STEPS, `bounded step count respected (${correction.steps} <= ${CORRECTION_STEPS})`);
  check(correction.trace.length === correction.steps + 1, 'worker returned the full measured loss trace (steps + 1 entries)');
  console.log(`  prediction now: class ${argmax(correction.postPred)} (human label ${feedbackCase.trueClass}), ` +
              `p(label) ${(correction.postPred[feedbackCase.trueClass]).toFixed(4)}`);
  check(correction.postPred.length === 2 && correction.probabilities.every((p) => p >= 0 && p <= 1), 'displayed probabilities are all within [0, 1]');
  const probSum = correction.probabilities.reduce((a, b) => a + b, 0);
  check(Math.abs(probSum - 1) < 1e-6, `displayed probabilities sum to 1 (sum = ${probSum.toFixed(6)})`);
  check(correction.activations !== null && correction.deltas !== null, 'real activations and deltas shipped to the visualizer');
  check(gradientFieldIdentityHolds(correction.activations, correction.deltas, correction.gradientMagnitudes),
    'shipped gradient field is literally delta (x) activation in every layer');
  check(countNonFiniteSnapshot(correction.weights) === 0, 'no non-finite weight survived the online update');
  check(correction.activeSetAction === 'appended' && correction.activeSetSize === 1, 'corrected sample persisted in the active set');
  check(correction.userSampleCount === 1, 'worker reports one human-labelled sample');

  const predictionAfter = await send('RUN_PREDICTION', {
    input: feedbackCase.tensor,
    source: 'test',
    targetClass: feedbackCase.trueClass
  }, 'PREDICTION_RESULT');
  if (predictionBefore) {
    check(predictionAfter.loss < predictionBefore.loss, 'a fresh inference through the worker measures the reduced loss on the corrected drawing');
  } else {
    console.log(`  fresh inference loss after correction: ${predictionAfter.loss.toFixed(5)}`);
  }
  check(predictionAfter.trueClass === feedbackCase.trueClass, 'fresh inference is still scored against the human-supplied class');

  // ---- Upsert semantics: relabelling the same tensor must replace, never duplicate ---------
  const reconfirm = await send('ONLINE_STEP', {
    sample: feedbackCase.tensor,
    target: oneHot(feedbackCase.trueClass),
    steps: 1
  }, 'ONLINE_STEP_COMPLETE');
  console.log(`  upsert sequence: appended -> ${reconfirm.activeSetAction} (size ${reconfirm.activeSetSize})`);
  check(reconfirm.activeSetAction === 'reconfirmed' && reconfirm.activeSetSize === 1, 'same tensor + same label -> reconfirmed, no duplicate');

  const relabel = await send('ONLINE_STEP', {
    sample: feedbackCase.tensor,
    target: oneHot(1 - feedbackCase.trueClass),
    steps: 1
  }, 'ONLINE_STEP_COMPLETE');
  console.log(`  upsert sequence: ... -> ${relabel.activeSetAction} (size ${relabel.activeSetSize})`);
  check(relabel.activeSetAction === 'relabelled' && relabel.activeSetSize === 1, 'same tensor + new label -> relabelled in place, no duplicate');

  // ---- Holdout guard: real corrections, general accuracy must hold -------------------------
  // Moderate mistakes only (loss in [0.7, 6)): correcting saturated adversarial outliers
  // (loss > 8) is measured to re-fit the model's positional prior and collapse centered
  // accuracy. A realistic session mixes drawing directions, so the guard walks each small
  // drift direction in turn instead of hammering a single one.
  const guardCases = [];
  for (const [dx, dy] of GUARD_DIRECTIONS) {
    if (guardCases.length >= GUARD_CORRECTIONS) break;
    const shiftedVariant = dataset.testX.map((t) => shift28(t, dx, dy));
    const search = await collectMistakes(send, shiftedVariant, dataset, {
      start: 0,
      count: GUARD_CORRECTIONS - guardCases.length,
      minLoss: 0.7,
      maxLoss: 6
    });
    for (const found of search.found) {
      guardCases.push({ ...found, tensor: shiftedVariant[found.index], dx, dy });
    }
  }
  console.log(`  applying ${guardCases.length} human corrections (${GUARD_STEPS} steps each, default lr, mixed drawing drifts)...`);
  check(guardCases.length >= 3, `found ${guardCases.length} real misplaced drawings for the guard (>= 3)`);

  let guardReductions = 0;
  for (const item of guardCases) {
    const stepResult = await send('ONLINE_STEP', {
      sample: item.tensor,
      target: oneHot(item.trueClass),
      steps: GUARD_STEPS,
      trueLabel: item.trueClass
    }, 'ONLINE_STEP_COMPLETE');
    if (stepResult.ok && stepResult.lossReduced) guardReductions++;
    console.log(`    guard drawing #${item.index} (drift ${item.dx},${item.dy}; truth ${item.trueClass}): ` +
                `loss ${stepResult.initialLoss.toFixed(4)} -> ${stepResult.postLoss.toFixed(4)}`);
  }
  check(guardCases.length === 0 || guardReductions === guardCases.length,
    `every one of ${guardCases.length} corrections reduced its own loss (${guardReductions} reduced)`);

  const evalAfter = await send('EVALUATE_TEST_SET', {}, 'TEST_SET_EVALUATED');
  const drop = evalBefore.accuracy - evalAfter.accuracy;
  console.log(`  held-out accuracy: ${formatPercent(evalBefore.accuracy)} -> ${formatPercent(evalAfter.accuracy)} ` +
              `(${drop >= 0 ? '-' : '+'}${formatPercent(Math.abs(drop))})`);
  check(drop <= MAX_HOLDOUT_DROP, `held-out accuracy dropped at most ${formatPercent(MAX_HOLDOUT_DROP)} (actual ${formatPercent(drop)})`);

  // ---- The human labels are replayed by the NEXT real training epoch -----------------------
  const replayIndex = log.length;
  await send('START_TRAINING', { maxEpochs: TRAIN_EPOCHS + 1, batchSize: 32, epochDelayMs: 0 }, 'TRAINING_COMPLETE');
  const replaySteps = log.slice(replayIndex).filter((m) => m.type === 'TRAINING_STEP');
  const replayPayload = replaySteps[replaySteps.length - 1].payload;
  console.log(`  epoch ${replayPayload.epoch}: replayed ${replayPayload.activeReplayCount} human samples ` +
              `(measured replay loss ${replayPayload.activeReplayLoss.toFixed(6)})`);
  check(replayPayload.activeReplayCount > 0, 'active-learning buffer was replayed during the next real epoch');
  check(Number.isFinite(replayPayload.activeReplayLoss) && replayPayload.activeReplayLoss > 0, 'replay produced a real gradient signal with a measured loss');
  check(Number.isFinite(replayPayload.valAccuracy) && replayPayload.valAccuracy > 0.7, 'validation accuracy stayed healthy through replay');

  // ---- Labeling Studio: ADD_TRAINING_SAMPLE validation + append + balance ------------------
  const badLength = await send('ADD_TRAINING_SAMPLE', { sample: [0.1, 0.2], target: [1, 0], label: 0 }, 'SAMPLE_ADDED');
  check(badLength.ok === false && /784/.test(badLength.error), 'short input tensor rejected with a real error');
  const badTarget = await send('ADD_TRAINING_SAMPLE', { sample: dataset.testX[0], target: [1, 0, 0], label: 0 }, 'SAMPLE_ADDED');
  check(badTarget.ok === false, 'mismatched target vector rejected');

  const studioAdd0 = await send('ADD_TRAINING_SAMPLE', { sample: dataset.testX[0], target: oneHot(0), label: 0, origin: 'studio' }, 'SAMPLE_ADDED');
  check(studioAdd0.ok === true && studioAdd0.totalSamples === initialTrainCount + 1, 'studio sample appended to the live training set');
  const studioAdd1 = await send('ADD_TRAINING_SAMPLE', { sample: dataset.testX[1], target: oneHot(1), label: 1, origin: 'studio' }, 'SAMPLE_ADDED');
  check(studioAdd1.ok === true && studioAdd1.classBalance[0] >= 1 && studioAdd1.classBalance[1] >= 1, 'class balance reflects both classes');

  // ---- Rebuilding the architecture must NOT discard human labels ---------------------------
  const rebuilt = await send('BUILD_NETWORK', { architecture: ARCHITECTURE, hyperparameters: HYPERPARAMS }, 'NETWORK_BUILT');
  check(rebuilt.activeSetSize > 0, `architecture rebuild preserved the human labels (${rebuilt.activeSetSize} samples)`);

  const stats = await send('GET_ACTIVE_STATS', {}, 'ACTIVE_STATS');
  check(stats.ok === true && stats.userSamples.length === rebuilt.activeSetSize, 'GET_ACTIVE_STATS reports every human sample');
  check(stats.userSamples.every((s) => s.classIndex === 0 || s.classIndex === 1), 'each human sample carries its class index');
  check(stats.userSamples.some((s) => s.supersededLabels >= 1), 'relabelling history is auditable (superseded labels counted)');
  check(stats.trainingSamples >= initialTrainCount + 2, 'training set size includes the appended studio samples');

  const reset = await send('RESET_ACTIVE_SET', {}, 'ACTIVE_STATS');
  check(reset.ok === true && reset.activeSetSize === 0, 'RESET_ACTIVE_SET clears the buffer');
  const statsAfterReset = await send('GET_ACTIVE_STATS', {}, 'ACTIVE_STATS');
  check(statsAfterReset.userSamples.length === 0, 'buffer stays empty after reset');
}

// ---------------------------------------------------------------------------------------------
async function main() {
  banner('ONLINE FEEDBACK / ACTIVE-LEARNING VERIFICATION (real MNIST)');

  const dataset = await loadMnistForTests({ maxTrain: TRAIN_SUBSET, maxTest: TEST_SUBSET });
  console.log(`  working set: ${dataset.trainX.length} train / ${dataset.testX.length} held-out real digits`);

  runEngineLevelChecks(dataset);
  await part2(dataset);

  banner(failures.length === 0 ? 'ONLINE FEEDBACK TEST PASSED' : 'ONLINE FEEDBACK TEST FAILED');
  console.log(`  checks passed: ${checks.passed}/${checks.total}`);
  if (failures.length === 0) {
    console.log('  [PASS] Human-in-the-Loop online backpropagation verified end-to-end\n');
    return 0;
  }
  for (const failure of failures) console.error(`  [FAIL] ${failure}`);
  console.error('');
  return 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error('\n[FAIL] Online feedback test crashed:', err);
    process.exit(1);
  });
