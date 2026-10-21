/**
 * Background Dedicated Web Worker for Neural Network Training
 * Runs matrix math and gradient descent asynchronously to guarantee 60 FPS UI rendering
 * Neural Network Studio Platform
 */

import { NeuralNetwork } from './neural_network.js';

let network = null;
let dataset = null;
let trainX = [];
let trainY = [];
let testX = [];
let testY = [];

/**
 * Persistent Human-in-the-Loop buffer.
 *
 * Owned by the worker (not by a network instance) so that rebuilding the architecture or
 * resetting the weights never silently discards the labels a human just supplied.
 * Duplicate tensors are replaced rather than appended: correcting the same drawing twice
 * must not train the model on two contradictory labels.
 */
const activeSamples = [];

/** Upsert a labelled tensor, returning what happened. */
function upsertActiveSample(tensor, target, origin = 'user') {
  const vector = tensor instanceof Float64Array ? tensor : Float64Array.from(tensor);

  for (let i = 0; i < activeSamples.length; i++) {
    const existing = activeSamples[i].x;
    if (existing.length !== vector.length) continue;

    let identical = true;
    for (let j = 0; j < vector.length; j++) {
      if (Math.abs(existing[j] - vector[j]) > 1e-9) {
        identical = false;
        break;
      }
    }
    if (identical) {
      const previous = activeSamples[i].y;
      activeSamples[i].y = Array.from(target);
      activeSamples[i].origin = origin;
      activeSamples[i].supersededLabels = (activeSamples[i].supersededLabels || 0) + 1;
      const previousClass = previous.indexOf(Math.max(...previous));
      const newClass = Array.from(target).indexOf(Math.max(...target));
      return {
        action: previousClass === newClass ? 'reconfirmed' : 'relabelled',
        previousClass,
        newClass,
        activeSetSize: activeSamples.length
      };
    }
  }

  activeSamples.push({ x: vector, y: Array.from(target), origin, addedAt: Date.now() });
  return { action: 'appended', activeSetSize: activeSamples.length };
}


let isTraining = false;
let currentEpoch = 0;
// null = unbounded run: train until the held-out loss stops improving (auto-stop) or the user
// pauses. A finite number is the exact old "N epochs" budget (used by the automated harness).
let maxEpochs = null;
let batchSize = 16;
let learningRate = 0.03;
let epochDelayMs = 0; // Delay between epochs for slow-motion pedagogical view
let trainingLoopTimer = null;
let lastEpochTime = performance.now();
// Lazily-built payload for the last completed epoch - lets the chunk scheduler emit a real
// TRAINING_STEP for chunks whose final epoch never qualified as the chunk's "last" epoch.
let pendingSnapshotBuilder = null;

// Auto-stop bookkeeping, fed exclusively by the real held-out loss measured after each epoch.
// The window is scoped to the current run: every Start opens a fresh convergence window.
let earlyStopPatience = 150; // epochs without held-out improvement before a run completes
let bestValLoss = Infinity;
let stallEpochs = 0;
// Safety ceiling for unbounded runs: a run that keeps improving by a hair every 149 epochs
// must still terminate before it occupies the worker forever.
const HARD_RUN_CEILING = 100000;

// Event Listener for UI Commands
self.onmessage = function (e) {
  const { type, payload } = e.data;

  switch (type) {
    case 'BUILD_NETWORK':
      buildNetwork(payload);
      break;

    case 'SET_DATASET':
      setDataset(payload);
      break;

    case 'START_TRAINING':
      startTraining(payload);
      break;

    case 'PAUSE_TRAINING':
      pauseTraining();
      break;

    case 'STEP_TRAINING':
      stepTraining();
      break;

    case 'RESET_NETWORK':
      resetNetwork();
      break;

    case 'UPDATE_HYPERPARAMS':
      updateHyperparameters(payload);
      break;

    case 'RUN_PREDICTION':
      runPrediction(payload);
      break;

    case 'ONLINE_STEP':
      executeOnlineStep(payload);
      break;

    case 'ADD_TRAINING_SAMPLE':
      addTrainingSample(payload);
      break;

    case 'EVALUATE_TEST_SET':
      evaluateTestSet(payload);
      break;

    case 'PREDICT_TEST_SET':
      predictTestSet(payload);
      break;

    case 'GET_ACTIVE_STATS':
      reportActiveStats();
      break;

    case 'RESET_ACTIVE_SET':
      resetActiveSet();
      break;

    case 'EVALUATE_GRID':
      evaluateGrid(payload);
      break;

    case 'IMPORT_MODEL':
      importModel(payload);
      break;

    default:
      console.warn('Unknown worker command:', type);
  }
};

function buildNetwork(payload) {
  const { architecture, hyperparameters } = payload;
  network = new NeuralNetwork({
    architecture,
    loss: hyperparameters.loss || 'mse',
    optimizer: hyperparameters.optimizer || 'adam',
    learningRate: hyperparameters.learningRate || 0.03,
    weightDecay: hyperparameters.weightDecay || 0.0001
  });

  // Re-attach the persisted Human-in-the-Loop buffer to the new parameter set.
  network.activeSamples = activeSamples;

  currentEpoch = 0;
  bestValLoss = Infinity;
  stallEpochs = 0;
  postMessage({
    type: 'NETWORK_BUILT',
    payload: {
      weights: network.getWeightsSnapshot(),
      architecture: network.architecture,
      learningRate: network.learningRate,
      rerandomized: true,
      activeSetSize: activeSamples.length
    }
  });
}

function setDataset(payload) {
  dataset = payload;
  trainX = payload.trainX || [];
  trainY = payload.trainY || [];
  testX = payload.testX || [];
  testY = payload.testY || [];

  currentEpoch = 0;
  bestValLoss = Infinity;
  stallEpochs = 0;
  postMessage({
    type: 'DATASET_READY',
    payload: {
      trainCount: trainX.length,
      testCount: testX.length,
      numFeatures: payload.numFeatures,
      numClasses: payload.numClasses,
      featureNames: payload.featureNames,
      targetNames: payload.targetNames
    }
  });
}

function updateHyperparameters(params) {
  if (!network) return;
  if (params.learningRate !== undefined) {
    network.learningRate = params.learningRate;
    network.optimizer.lr = params.learningRate;
  }
  if (params.batchSize !== undefined) {
    batchSize = Math.max(1, params.batchSize);
  }
  if (params.epochDelayMs !== undefined) {
    epochDelayMs = params.epochDelayMs;
  }
}

function startTraining(payload) {
  if (!network || trainX.length === 0) {
    // Without both a network and a dataset an epoch loop would spin forever without advancing.
    postMessage({
      type: 'ERROR',
      payload: { message: 'Cannot start training before a network and a dataset are loaded.' }
    });
    return;
  }
  if (payload) {
    if (payload.maxEpochs !== undefined) {
      maxEpochs = payload.maxEpochs === null ? null : Math.max(1, Math.floor(payload.maxEpochs));
    }
    if (payload.batchSize !== undefined) batchSize = payload.batchSize;
    if (payload.epochDelayMs !== undefined) epochDelayMs = payload.epochDelayMs;
    if (payload.earlyStopPatience !== undefined) {
      earlyStopPatience = payload.earlyStopPatience === null
        ? null
        : Math.max(1, Math.floor(payload.earlyStopPatience));
    }
  }
  // Every Start opens a fresh convergence window (best held-out loss measured within this run).
  bestValLoss = Infinity;
  stallEpochs = 0;
  isTraining = true;
  lastEpochTime = performance.now();
  scheduleNextEpoch();
}

function pauseTraining() {
  isTraining = false;
  if (trainingLoopTimer) {
    clearTimeout(trainingLoopTimer);
    trainingLoopTimer = null;
  }
  postMessage({ type: 'TRAINING_PAUSED', payload: { epoch: currentEpoch } });
}

function resetNetwork() {
  pauseTraining();
  currentEpoch = 0;
  bestValLoss = Infinity;
  stallEpochs = 0;
  if (network && network.architecture) {
    network.buildFromArchitecture(network.architecture);
    postMessage({
      type: 'NETWORK_RESET',
      payload: {
        weights: network.getWeightsSnapshot()
      }
    });
  }
}

function stepTraining() {
  if (!network || trainX.length === 0) return;
  executeEpoch(true);
}

function completeTraining(reason) {
  isTraining = false;
  postMessage({
    type: 'TRAINING_COMPLETE',
    payload: {
      epoch: currentEpoch,
      reason,
      stallEpochs,
      bestValLoss: Number.isFinite(bestValLoss) ? bestValLoss : null
    }
  });
}

/**
 * Auto-stop: end the run once the held-out loss has not improved for `earlyStopPatience`
 * epochs. This replaces the old fixed 500-epoch cap - a run now continues exactly as long as
 * it is still measurably learning, and never stops mid-improvement.
 */
function maybeAutoStop() {
  if (!isTraining) return false;
  if (earlyStopPatience !== null && stallEpochs >= earlyStopPatience) {
    completeTraining('converged');
    return true;
  }
  return false;
}

function scheduleNextEpoch() {
  if (!isTraining) return;

  const effectiveCap = maxEpochs !== null ? maxEpochs : HARD_RUN_CEILING;
  if (currentEpoch >= effectiveCap) {
    completeTraining('epoch-cap');
    return;
  }

  if (epochDelayMs > 0) {
    trainingLoopTimer = setTimeout(() => {
      executeEpoch();
      if (maybeAutoStop()) return;
      scheduleNextEpoch();
    }, epochDelayMs);
  } else {
    // Run chunk of epochs for high throughput, posting updates every ~30ms
    const startTime = performance.now();
    let emittedInChunk = false;

    while (isTraining && currentEpoch < effectiveCap && performance.now() - startTime < 30) {
      const isLastInChunk = (performance.now() - startTime >= 25) || (currentEpoch + 1 >= effectiveCap);
      executeEpoch(isLastInChunk);
      if (isLastInChunk) emittedInChunk = true;
      // Check per epoch (not per chunk) so the run stops exactly when patience elapses.
      if (maybeAutoStop()) return;
    }

    if (maybeAutoStop()) return;
    // A chunk can end on an epoch that was never flagged as the chunk's "last" one (the flag
    // is judged before the epoch runs, and a single slow epoch already exceeds the 30ms
    // window). Post that completed epoch's snapshot so the UI receives a real live update
    // at any epoch speed - previously a run of slow epochs emitted nothing until the end.
    if (!emittedInChunk && pendingSnapshotBuilder) {
      postMessage({ type: 'TRAINING_STEP', payload: pendingSnapshotBuilder() });
    }
    trainingLoopTimer = setTimeout(scheduleNextEpoch, 0);
  }
}

function executeEpoch(shouldEmitSnapshot = true) {
  if (!network || trainX.length === 0) return;

  currentEpoch++;
  const numSamples = trainX.length;
  const actualBatchSize = Math.min(batchSize, numSamples);

  // Mini-batch stochastic shuffle
  const indices = Array.from({ length: numSamples }, (_, i) => i);
  for (let i = numSamples - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  let totalLoss = 0;
  let totalAcc = 0;
  let batchCount = 0;
  let lastBatchActivations = null;
  let lastBatchDeltas = null;

  for (let start = 0; start < numSamples; start += actualBatchSize) {
    const end = Math.min(start + actualBatchSize, numSamples);
    const bIndices = indices.slice(start, end);
    const bX = bIndices.map(idx => trainX[idx]);
    const bY = bIndices.map(idx => trainY[idx]);

    // Attach step hooks for step capture
    network.hooks.onBackwardStep = (stepData) => {
      lastBatchActivations = stepData.activations;
      lastBatchDeltas = stepData.deltas;
    };

    const res = network.trainBatch(bX, bY);
    totalLoss += res.loss;
    totalAcc += res.accuracy;
    batchCount++;
  }

  const trainLoss = totalLoss / batchCount;
  const trainAcc = totalAcc / batchCount;

  // Active-learning replay: the corrected/drawn samples persist in the training set and
  // are replayed every epoch (a real, measurable gradient signal for the user's labels).
  const replay = replayActiveSet();

  // Parameter health sweep (catches an overflow on the epoch that produced it).
  const sanitizeReport = network.sanitizeWeights();

  // Validation metrics
  let valLoss = trainLoss;
  let valAcc = trainAcc;
  if (testX.length > 0) {
    const valPreds = network.predict(testX);
    let vLossSum = 0;
    let vCorrect = 0;
    for (let i = 0; i < testX.length; i++) {
      vLossSum += network.lossFunction.calculate(valPreds[i], testY[i]);
      if (testY[i].length > 1) {
        const pClass = valPreds[i].indexOf(Math.max(...valPreds[i]));
        const tClass = testY[i].indexOf(Math.max(...testY[i]));
        if (pClass === tClass) vCorrect++;
      } else {
        if (Math.abs(valPreds[i][0] - testY[i][0]) < 0.25) vCorrect++;
      }
    }
    valLoss = vLossSum / testX.length;
    valAcc = vCorrect / testX.length;
  }

  // Convergence bookkeeping on the real held-out loss. "Improvement" must beat the best loss
  // seen so far in this run by at least 0.1% (plus an absolute floor to ignore float noise).
  if (Number.isFinite(valLoss)) {
    if (!Number.isFinite(bestValLoss)) {
      bestValLoss = valLoss;
      stallEpochs = 0;
    } else {
      const improvementThreshold = Math.max(1e-6, bestValLoss * 0.001);
      if (valLoss < bestValLoss - improvementThreshold) {
        bestValLoss = valLoss;
        stallEpochs = 0;
      } else {
        stallEpochs++;
      }
    }
  }

  // Calculate speed
  const now = performance.now();
  const speed = 1000 / Math.max(1, now - lastEpochTime);
  lastEpochTime = now;

  // Build the snapshot lazily: epochs inside a fast chunk pay the snapshot cost only when
  // the payload is actually posted (see the chunk scheduler below).
  pendingSnapshotBuilder = () => ({
    epoch: currentEpoch,
    loss: trainLoss,
    accuracy: trainAcc,
    valLoss,
    valAccuracy: valAcc,
    speed: speed.toFixed(1),
    activations: lastBatchActivations,
    deltas: lastBatchDeltas,
    weights: network.getWeightsSnapshot(),
    activeReplayCount: replay.count,
    activeReplayLoss: replay.loss,
    activeSetSize: countUserSamples(),
    repairedParameters: sanitizeReport.repaired,
    totalRepairedParameters: sanitizeReport.total,
    // Live convergence readout: how long the held-out loss has been flat within this run.
    stallEpochs,
    bestValLoss: Number.isFinite(bestValLoss) ? bestValLoss : null,
    maxAbsWeight: network.maxAbsWeight()
  });

  if (shouldEmitSnapshot) {
    postMessage({ type: 'TRAINING_STEP', payload: pendingSnapshotBuilder() });
  }
}

/**
 * One real gradient pass over the human-labelled / corrected samples.
 * Returns the measured mean loss over that buffer (0 when the buffer is empty).
 */
function replayActiveSet() {
  const buffer = activeSamples;
  if (buffer.length === 0) return { count: 0, loss: 0 };

  const limit = Math.min(buffer.length, 32);
  const batchX = [];
  const batchY = [];
  for (let i = 0; i < limit; i++) {
    batchX.push(buffer[i].x);
    batchY.push(buffer[i].y);
  }

  const result = network.trainBatch(batchX, batchY);
  return { count: limit, loss: result.loss };
}

/**
 * Inference on a single sample.
 *
 * When the caller supplies the true class (`targetClass`) the reply also carries the real
 * loss for that sample, which is what the Human-in-the-Loop panel compares against after
 * an online correction step.
 */
function runPrediction(payload) {
  if (!network) {
    postMessage({ type: 'ERROR', payload: { message: 'No network built yet.', source: payload && payload.source } });
    return;
  }
  const { input, source = 'generic', targetClass = null, sampleId = null } = payload || {};

  if (!input || input.length === 0) {
    postMessage({ type: 'ERROR', payload: { message: 'Empty input tensor rejected.', source } });
    return;
  }

  // Silently truncating a longer tensor (what forwardSample would otherwise do) produces a
  // real-looking but meaningless prediction; refuse the mismatch and say exactly what it is.
  const expectedInputs = network.architecture[0].size;
  if (input.length !== expectedInputs) {
    postMessage({
      type: 'ERROR',
      payload: {
        message: `Input tensor has ${input.length} values but the active network expects ${expectedInputs}. ` +
          'Load a dataset that matches this input (the 28x28 MNIST 0-vs-1 corpus for drawings).',
        source
      }
    });
    return;
  }

  let target = null;
  if (targetClass !== null && targetClass !== undefined) {
    const numClasses = network.architecture[network.architecture.length - 1].size;
    target = new Array(numClasses).fill(0);
    if (targetClass >= 0 && targetClass < numClasses) target[targetClass] = 1;
  }

  const probe = network.probeSample(input, target);
  const probabilities = normalizeProbabilities(probe.output);

  postMessage({
    type: 'PREDICTION_RESULT',
    payload: {
      source,
      sampleId,
      input: Array.from(input),
      output: probe.output,
      probabilities,
      activations: probe.activations,
      predictedClass: probe.predictedClass,
      confidence: probabilities[probe.predictedClass],
      targetClass,
      trueClass: probe.targetClass,
      loss: probe.loss,
      correct: target ? probe.predictedClass === probe.targetClass : null
    }
  });
}

/** Convert raw output activations into a probability distribution for display. */
function normalizeProbabilities(output) {
  const sum = output.reduce((acc, value) => acc + (value > 0 ? value : 0), 0);
  if (!(sum > 0)) {
    return output.map(() => 1 / Math.max(1, output.length));
  }
  return output.map((value) => (value > 0 ? value / sum : 0));
}

function evaluateGrid(payload) {
  if (!network) return;
  const { gridPoints } = payload; // [[x1, x2], ...]
  const predictions = network.predict(gridPoints);
  postMessage({
    type: 'GRID_EVALUATED',
    payload: {
      predictions
    }
  });
}

/**
 * Human-in-the-Loop core: run a REAL single-sample backpropagation update for the
 * corrected label and report the measured before/after loss together with the backward
 * gradient magnitudes that drive the visualizer.
 */
function executeOnlineStep(payload) {
  if (!network) {
    postMessage({ type: 'ONLINE_STEP_COMPLETE', payload: { ok: false, error: 'No network built yet.' } });
    return;
  }

  const { sample, target, learningRate = null, steps = 1, trueLabel = null, source = 'studio' } = payload || {};

  if (!sample || sample.length === 0) {
    postMessage({ type: 'ONLINE_STEP_COMPLETE', payload: { ok: false, error: 'Empty sample rejected.' } });
    return;
  }
  if (sample.length !== network.architecture[0].size) {
    postMessage({
      type: 'ONLINE_STEP_COMPLETE',
      payload: {
        ok: false,
        error: `Input tensor has ${sample.length} values but the active network expects ${network.architecture[0].size}.`
      }
    });
    return;
  }
  if (!target || target.length !== network.architecture[network.architecture.length - 1].size) {
    postMessage({
      type: 'ONLINE_STEP_COMPLETE',
      payload: { ok: false, error: `Target vector must have ${network.architecture[network.architecture.length - 1].size} entries.` }
    });
    return;
  }

  // Capture real backward signals for the visualizer (last training sample of the step).
  let capturedActivations = null;
  let capturedDeltas = null;
  network.hooks.onBackwardStep = (stepData) => {
    capturedActivations = stepData.activations;
    capturedDeltas = stepData.deltas;
  };

  const started = performance.now();
  const result = network.online_step(sample, target, learningRate, steps);
  const elapsedMs = performance.now() - started;

  // Real gradient field for the backward frame captured during the correction:
  // gradientMagnitudes[l][i][j] = capturedDeltas[l][i] * capturedActivations[l][j] - the same
  // delta (x) activation math the update itself used. Deriving it from a separate post-update
  // probe (as before) shipped deltas belonging to a different frame than the shipped activations.
  const sanitizeReport = network.sanitizeWeights();
  let gradientMagnitudes = null;
  let maxAbsGradient = 0;
  if (capturedDeltas && capturedActivations) {
    gradientMagnitudes = capturedDeltas.map((delta, l) => {
      const prev = capturedActivations[l];
      return Array.from(delta, (d) => {
        const row = new Float64Array(prev.length);
        for (let j = 0; j < prev.length; j++) {
          const g = d * prev[j];
          row[j] = g;
          const abs = Math.abs(g);
          if (abs > maxAbsGradient) maxAbsGradient = abs;
        }
        return row;
      });
    });
  }

  // The corrected sample is by definition active-learning material: remember it so the
  // next training epochs replay it instead of applying a one-shot update.
  const enqueue = upsertActiveSample(sample, target, 'correction');

  network.hooks.onBackwardStep = null;

  postMessage({
    type: 'ONLINE_STEP_COMPLETE',
    payload: {
      ok: true,
      source,
      trueLabel,
      steps: result.steps,
      initialLoss: result.initialLoss,
      postLoss: result.postLoss,
      lossDelta: result.lossDelta,
      lossReduced: result.lossReduced,
      relativeReduction: result.relativeReduction,
      initialPred: result.initialPred,
      postPred: result.postPred,
      probabilities: normalizeProbabilities(result.postPred),
      trace: result.trace,
      learningRateUsed: result.learningRateUsed,
      elapsedMs,
      sanitizedParameters: sanitizeReport.repaired,
      activations: capturedActivations,
      deltas: capturedDeltas,
      gradientMagnitudes,
      maxAbsGradient,
      activeSetAction: enqueue.action,
      activeSetSize: enqueue.activeSetSize || 0,
      userSampleCount: activeSamples.length,
      weights: network.getWeightsSnapshot()
    }
  });
}

/** Append a labelled sample to the live training set (Labeling Studio). */
function addTrainingSample(payload) {
  if (!network) {
    postMessage({ type: 'SAMPLE_ADDED', payload: { ok: false, error: 'No network built yet.' } });
    return;
  }

  const { sample, target, label = null, origin = 'user' } = payload || {};

  if (!sample) {
    postMessage({
      type: 'SAMPLE_ADDED',
      payload: { ok: false, error: 'No sample tensor received.' }
    });
    return;
  }
  if (sample.length !== network.architecture[0].size) {
    postMessage({
      type: 'SAMPLE_ADDED',
      payload: {
        ok: false,
        error: `Input tensor has ${sample.length} values but the active network expects ${network.architecture[0].size}.`
      }
    });
    return;
  }
  if (!target || target.length !== network.architecture[network.architecture.length - 1].size) {
    postMessage({
      type: 'SAMPLE_ADDED',
      payload: { ok: false, error: 'Target vector length does not match the output layer.' }
    });
    return;
  }

  if (!trainX) trainX = [];
  if (!trainY) trainY = [];
  trainX.push(Array.from(sample));
  trainY.push(Array.from(target));

  const enqueue = upsertActiveSample(sample, target, origin);

  postMessage({
    type: 'SAMPLE_ADDED',
    payload: {
      ok: true,
      label,
      origin,
      activeSetAction: enqueue.action,
      totalSamples: trainX.length,
      activeSetSize: enqueue.activeSetSize || 0,
      userSampleCount: activeSamples.length,
      classBalance: classBalance(trainY)
    }
  });
}

function countUserSamples() {
  return activeSamples.length;
}

function classBalance(labels) {
  const counts = {};
  for (const label of labels) {
    const cls = label.indexOf(Math.max(...label));
    counts[cls] = (counts[cls] || 0) + 1;
  }
  return counts;
}

/** Run a real evaluation over the held-out test partition. */
function evaluateTestSet(payload) {
  if (!network || testX.length === 0) {
    postMessage({ type: 'TEST_SET_EVALUATED', payload: { ok: false, error: 'No test partition loaded.' } });
    return;
  }
  const limit = payload && payload.limit ? Math.min(payload.limit, testX.length) : testX.length;
  const started = performance.now();
  const report = network.evaluate(testX.slice(0, limit), testY.slice(0, limit));

  postMessage({
    type: 'TEST_SET_EVALUATED',
    payload: {
      ok: true,
      count: report.count,
      loss: report.loss,
      accuracy: report.accuracy,
      elapsedMs: performance.now() - started,
      worstCases: report.misclassified
        .slice()
        .sort((a, b) => b.loss - a.loss)
        .slice(0, 8)
        .map((m) => ({
          index: m.index,
          predictedClass: m.predictedClass,
          trueClass: m.trueClass,
          loss: m.loss,
          // Ship the pixels of the worst mistakes so the studio can show real failures.
          tensor: Array.from(testX[m.index])
        }))
    }
  });
}

/**
 * Per-sample predictions over (a slice of) the held-out test partition.
 * The Node Studio's Confusion Matrix tap consumes these — real forward passes, argmaxed
 * per sample, never a re-derived statistic.
 */
function predictTestSet(payload) {
  if (!network || testX.length === 0) {
    postMessage({ type: 'TEST_SET_PREDICTIONS', payload: { ok: false, error: 'No test partition loaded.' } });
    return;
  }
  const limit = payload && payload.limit ? Math.min(payload.limit, testX.length) : testX.length;
  const predictions = network.predict(testX.slice(0, limit));
  const rows = [];
  for (let i = 0; i < limit; i++) {
    const vector = predictions[i];
    let predictedClass = 0;
    for (let c = 1; c < vector.length; c++) if (vector[c] > vector[predictedClass]) predictedClass = c;
    const truth = testY[i];
    let trueClass = 0;
    for (let c = 1; c < truth.length; c++) if (truth[c] > truth[trueClass]) trueClass = c;
    rows.push({ index: i, predictedClass, trueClass, confidence: vector[predictedClass] });
  }
  postMessage({ type: 'TEST_SET_PREDICTIONS', payload: { ok: true, count: limit, predictions: rows } });
}

function reportActiveStats() {
  postMessage({
    type: 'ACTIVE_STATS',
    payload: {
      ok: true,
      trainingSamples: trainX ? trainX.length : 0,
      activeSetSize: activeSamples.length,
      userSamples: activeSamples.map((entry) => ({
        origin: entry.origin,
        addedAt: entry.addedAt,
        supersededLabels: entry.supersededLabels || 0,
        classIndex: entry.y.indexOf(Math.max(...entry.y)),
        sample: Array.from(entry.x)
      })),
      classBalance: trainY ? classBalance(trainY) : {}
    }
  });
}

function resetActiveSet() {
  activeSamples.length = 0;
  postMessage({ type: 'ACTIVE_STATS', payload: { ok: true, activeSetSize: 0, reset: true } });
}

function importModel(payload) {
  if (isTraining) pauseTraining(); // never swap the network mid-run; the data stays, the run stops

  let imported;
  try {
    imported = NeuralNetwork.fromJSON(payload);
    imported.activeSamples = activeSamples;
  } catch (err) {
    postMessage({
      type: 'ERROR',
      payload: { message: 'Failed to import model: ' + err.message + '.' }
    });
    return;
  }

  network = imported;
  // Resume the saved run's continuity: the epoch counter continues from the exported session,
  // and the convergence window restarts (the next run measures fresh held-out improvement).
  const savedEpochs = payload && payload.metadata && payload.metadata.epochs;
  currentEpoch = Number.isFinite(savedEpochs) ? Math.max(0, Math.floor(savedEpochs)) : 0;
  bestValLoss = Infinity;
  stallEpochs = 0;

  postMessage({
    type: 'MODEL_IMPORTED',
    payload: {
      weights: network.getWeightsSnapshot(),
      architecture: network.architecture,
      activeSetSize: activeSamples.length,
      epoch: currentEpoch,
      loss: network.lossType,
      learningRate: network.learningRate,
      metadata: (payload && payload.metadata) || null
    }
  });
}

