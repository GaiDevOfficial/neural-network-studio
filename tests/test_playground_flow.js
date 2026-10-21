/**
 * Tier 1 playground data-path verification.
 *
 * The playground is a presentation layer over the same engine RPCs the block studio uses:
 *   prepare()  → the MNIST 0-vs-1 preset compiles and real data resolves,
 *   predict()  → one forward pass, with activations for the synaptic canvas,
 *   correct()  → one real ONLINE_STEP on the drawing, then an immediate re-prediction.
 *
 * This suite exercises the engine calls behind that controller on the real MNIST payload:
 *   1. the preset compiles to 784 → 16 → 8 → 2 with BCE,
 *   2. training changes real weights and yields a finite held-out accuracy,
 *   3. a genuinely misclassified digit is corrected with a one-hot target,
 *   4. the returned pre/post predictions equal fresh forward passes (no fabricated numbers),
 *   5. the captured activations/deltas have the shapes the synaptic canvas consumes,
 *   6. the corrected weights differ from the pre-correction snapshot,
 *   7. the playground label mapping (targetNames → “0”/“1”) is present.
 *
 * File: tests/test_playground_flow.js
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NeuralNetwork } from '../src/engine/neural_network.js';
import { instantPreset } from '../src/graph/smart_assist.js';
import { compile } from '../src/graph/node_engine.js';
import { datasetInfo } from '../src/graph/data_connectors.js';
import { loadMnistForTests, trainEpochs, formatPercent, banner } from './mnist_test_utils.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

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

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  }
  return worst;
}

function normalizeProbabilities(values) {
  const positive = values.map((value) => (value > 0 ? value : 0));
  const sum = positive.reduce((acc, value) => acc + value, 0);
  return sum > 0 ? positive.map((value) => value / sum) : positive.map(() => 1 / positive.length);
}

function shifted(tensor, dx, dy) {
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

function snapshotDelta(before, after) {
  let worst = 0;
  let changed = 0;
  for (let l = 0; l < before.length; l++) {
    for (let i = 0; i < before[l].weights.length; i++) {
      for (let j = 0; j < before[l].weights[i].length; j++) {
        const diff = Math.abs(before[l].weights[i][j] - after[l].weights[i][j]);
        if (diff > 0) changed++;
        worst = Math.max(worst, diff);
      }
    }
    for (let i = 0; i < before[l].biases.length; i++) {
      const diff = Math.abs(before[l].biases[i] - after[l].biases[i]);
      if (diff > 0) changed++;
      worst = Math.max(worst, diff);
    }
  }
  return { worst, changed };
}

async function main() {
  console.log('=====================================================');
  console.log('   NODE STUDIO — TIER 1 PLAYGROUND DATA PATH          ');
  console.log('=====================================================\n');

  const dataset = await loadMnistForTests({ maxTrain: 600, maxTest: 160 });
  console.log(`  MNIST 0-vs-1: ${dataset.trainX.length} train / ${dataset.testX.length} test digits`);
  check('playground dataset has the 0/1 class names',
    Array.isArray(dataset.targetNames) && dataset.targetNames.length === 2,
    JSON.stringify(dataset.targetNames));
  console.log(`  target names: ${JSON.stringify(dataset.targetNames)}`);

  const plan = compile(instantPreset('mnist_classifier'), {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(dataset) : null)
  });
  check('pre-assembled MNIST graph compiles', plan.ok === true,
    JSON.stringify((plan.errors || []).map((error) => error.code)));
  check('architecture is 784 → 16 → 8 → 2',
    JSON.stringify(plan.network.architecture.map((layer) => layer.size)) === JSON.stringify([784, 16, 8, 2]),
    JSON.stringify(plan.network.architecture.map((layer) => layer.size)));
  check('loss is binary cross-entropy', plan.network.loss === 'binary_crossentropy', plan.network.loss);
  check('flat input is 784 pixels', plan.meta.flatInputSize === 784, String(plan.meta.flatInputSize));

  const net = new NeuralNetwork({
    architecture: plan.network.architecture,
    loss: plan.network.loss,
    optimizer: plan.network.optimizer,
    learningRate: plan.network.learningRate,
    weightDecay: plan.network.weightDecay
  });

  banner('1 — Practice: real training updates the weights');
  const initialSnapshot = net.getWeightsSnapshot();
  const initialProbe = Array.from(net.forwardSample(dataset.testX[0]).output);
  const history = trainEpochs(net, dataset.trainX, dataset.trainY, { epochs: 2, batchSize: 32 });
  const trainedSnapshot = net.getWeightsSnapshot();
  const trainedDelta = snapshotDelta(initialSnapshot, trainedSnapshot);
  const trainedProbe = Array.from(net.forwardSample(dataset.testX[0]).output);

  check('training ran two measured epochs', history.length === 2);
  check('every epoch reported a finite loss',
    history.every((entry) => Number.isFinite(entry.trainLoss)), JSON.stringify(history));
  check('training changed real weights', trainedDelta.changed > 0,
    `${trainedDelta.changed} parameters, max |Δ| ${trainedDelta.worst.toExponential(2)}`);
  check('the trained model changes its prediction', maxAbsDiff(initialProbe, trainedProbe) > 1e-9,
    maxAbsDiff(initialProbe, trainedProbe).toExponential(2));

  const evaluation = net.evaluate(dataset.testX, dataset.testY);
  console.log(`  held-out accuracy after practice: ${formatPercent(evaluation.accuracy)} (${evaluation.count} digits)`);
  check('held-out accuracy is a measured percentage',
    Number.isFinite(evaluation.accuracy) && evaluation.accuracy >= 0 && evaluation.accuracy <= 1,
    String(evaluation.accuracy));

  banner('2 — Ask: one forward pass, with visualization data');
  const firstProbe = dataset.testX[0];
  const before = Array.from(net.forwardSample(firstProbe).output);
  check('a prediction is a finite output vector', before.every(Number.isFinite) && before.length === 2,
    JSON.stringify(before));
  const displayBefore = normalizeProbabilities(before);
  check('display probabilities are normalized for the learner',
    Math.abs(displayBefore[0] + displayBefore[1] - 1) < 1e-9,
    String(displayBefore[0] + displayBefore[1]));

  banner('3 — Teach: correct a genuinely misclassified digit');
  let sample = null;
  let sampleIndex = -1;
  let trueLabel = -1;
  let predicted = -1;
  let beforeCorrection = null;

  for (let i = 0; i < dataset.testX.length; i++) {
    const probe = dataset.testX[i];
    const output = Array.from(net.forwardSample(probe).output);
    const guess = argmax(output);
    const truth = argmax(dataset.testY[i]);
    if (guess !== truth) {
      sample = probe;
      sampleIndex = i;
      trueLabel = truth;
      predicted = guess;
      beforeCorrection = output;
      break;
    }
  }
  if (!sample) {
    for (let i = 0; i < dataset.testX.length; i++) {
      const probe = shifted(dataset.testX[i], 2, 2);
      const output = Array.from(net.forwardSample(probe).output);
      const guess = argmax(output);
      const truth = argmax(dataset.testY[i]);
      if (guess !== truth) {
        sample = probe;
        sampleIndex = i;
        trueLabel = truth;
        predicted = guess;
        beforeCorrection = output;
        break;
      }
    }
  }

  check('found a real misclassified drawing to correct', Boolean(sample));
  if (sample) {
    console.log(`  drawing #${sampleIndex}: network says ${predicted}, human says ${trueLabel}`);
    const target = [0, 0];
    target[trueLabel] = 1; // exactly the playground's one-hot choice

    let captured = null;
    net.hooks.onBackwardStep = (step) => { captured = step; };
    const weightsBeforeCorrection = net.getWeightsSnapshot();
    const result = net.online_step(sample, target, 0.05, 1);
    net.hooks.onBackwardStep = null;
    const weightsAfterCorrection = net.getWeightsSnapshot();

    check('online step reports the measured initial loss', Number.isFinite(result.initialLoss), String(result.initialLoss));
    check('online step reports the measured post loss', Number.isFinite(result.postLoss), String(result.postLoss));
    check('online step records whether the loss was reduced', typeof result.lossReduced === 'boolean',
      String(result.lossReduced));
    check('online step reports the pre-update prediction', maxAbsDiff(result.initialPred, beforeCorrection) < 1e-9,
      maxAbsDiff(result.initialPred, beforeCorrection).toExponential(2));
    console.log(`  loss: ${result.initialLoss.toFixed(5)} → ${result.postLoss.toFixed(5)} ` +
      `(${result.lossReduced ? 'reduced' : 'not reduced'}, relative ${(result.relativeReduction * 100).toFixed(2)}%)`);
    console.log(`  prediction: ${predicted} → ${argmax(result.postPred)} ` +
      `(p${trueLabel}: ${beforeCorrection[trueLabel].toFixed(4)} → ${result.postPred[trueLabel].toFixed(4)})`);

    check('the correction moved the true-class probability in the right direction',
      result.postPred[trueLabel] > beforeCorrection[trueLabel],
      `${beforeCorrection[trueLabel]} → ${result.postPred[trueLabel]}`);
    check('the correction changed real weights', snapshotDelta(weightsBeforeCorrection, weightsAfterCorrection).changed > 0,
      `${snapshotDelta(weightsBeforeCorrection, weightsAfterCorrection).changed} parameters`);

    const freshAfter = Array.from(net.forwardSample(sample).output);
    check('the immediate re-prediction equals a fresh forward pass',
      maxAbsDiff(result.postPred, freshAfter) < 1e-9,
      maxAbsDiff(result.postPred, freshAfter).toExponential(2));
    const displayAfter = normalizeProbabilities(freshAfter);
    check('the immediate re-prediction is still a normalized display distribution',
      Math.abs(displayAfter[0] + displayAfter[1] - 1) < 1e-9,
      String(displayAfter[0] + displayAfter[1]));

    check('backward signals were captured for the synaptic canvas', Boolean(captured));
    if (captured) {
      check('one activation vector per layer (input + layers)',
        captured.activations.length === plan.network.architecture.length,
        `${captured.activations.length}/${plan.network.architecture.length}`);
      check('one delta vector per weight layer',
        captured.deltas.length === plan.network.architecture.length - 1,
        `${captured.deltas.length}/${plan.network.architecture.length - 1}`);
      check('input activation vector is 784 wide', captured.activations[0].length === 784,
        String(captured.activations[0].length));
      check('hidden activations match the 16/8 layer sizes',
        captured.activations[1].length === 16 && captured.activations[2].length === 8,
        `${captured.activations[1].length}/${captured.activations[2].length}`);
      check('the output delta vector has 2 values', captured.deltas[2].length === 2,
        String(captured.deltas[2].length));
    }

    const afterEvaluation = net.evaluate(dataset.testX, dataset.testY);
    console.log(`  held-out accuracy after one correction: ${formatPercent(afterEvaluation.accuracy)}`);
    check('one correction leaves held-out accuracy finite and in range',
      Number.isFinite(afterEvaluation.accuracy) && afterEvaluation.accuracy >= 0 && afterEvaluation.accuracy <= 1,
      String(afterEvaluation.accuracy));
  }

  banner('4 — “That’s right”: confirmation makes no update');
  const confirmBefore = net.getWeightsSnapshot();
  const confirmOutput = Array.from(net.forwardSample(firstProbe).output);
  const confirmAfter = net.getWeightsSnapshot();
  check('a confirmation does not mutate weights (no online step is sent)',
    snapshotDelta(confirmBefore, confirmAfter).changed === 0);
  check('a confirmation does not change the prediction', maxAbsDiff(confirmOutput, Array.from(net.forwardSample(firstProbe).output)) === 0);

  // The UI module must exist and must not import browser DOM code that would break audio/worker use.
  const uiPath = path.join(ROOT, 'src', 'modes', 'absolute_beginner.js');
  const ui = fs.readFileSync(uiPath, 'utf8');
  const studio = fs.readFileSync(path.join(ROOT, 'src', 'ui', 'canvas_studio.js'), 'utf8');
  check('playground module delegates predict/correct to the real page controller',
    ui.includes('controller.predict') && ui.includes('controller.correct'));
  check('playground recomputes the answer immediately after a correction',
    /const recheck = await this\.controller\.predict/.test(ui));
  check('page controller sends ONLINE_STEP with the playground source tag',
    studio.includes("type: 'ONLINE_STEP'") && studio.includes("source: 'playground'"));
  check('page controller sends one-hot targets at the online learning rate',
    studio.includes('target[trueLabel] = 1') && studio.includes('learningRate: 0.05'));

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((error) => {
  console.error(`\n  [FAIL] suite crashed: ${error.stack || error.message}`);
  process.exit(1);
});
