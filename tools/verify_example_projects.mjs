/**
 * Example-projects verification sweep (shipped with the examples pack).
 *
 * Trains every project from the real engine — `DatasetManager` for parsing/splits,
 * `NeuralNetwork` + `trainEpochs` for real forward/backward passes — and prints the
 * measured results that `example_projects.md` quotes. Nothing here is simulated:
 * every accuracy / MAE below comes from `network.evaluate()` (or `predict()`) on
 * held-out rows that were never trained on.
 *
 * Gate: every classification project must reach a MEDIAN held-out accuracy >= 95%
 * across the 10 seeded 80/20 splits (tiny test sets cannot promise a worst-case
 * number; the full distribution is printed and quoted), every regression project a
 * median MAE <= 10% of the target range, and the real MNIST project >= 95%.
 *
 * Usage:  node tools/verify_example_projects.mjs
 *
 * File: tools/verify_example_projects.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { DatasetManager } from '../src/engine/dataset_manager.js';
import { NeuralNetwork } from '../src/engine/neural_network.js';
import { trainEpochs, loadMnistForTests } from '../tests/mnist_test_utils.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/* ------------------------------------------------------------------ helpers */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function splitSeeded(X, y, ratio, seed) {
  const rng = mulberry32(seed);
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const tc = Math.floor(n * ratio);
  const out = { trainX: [], trainY: [], testX: [], testY: [] };
  for (let i = 0; i < n; i++) {
    const row = X[idx[i]], lab = y[idx[i]];
    if (i < tc) { out.trainX.push(row); out.trainY.push(lab); }
    else { out.testX.push(row); out.testY.push(lab); }
  }
  return out;
}

const median = (values) => {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * The engine initializes weights with Math.random() (Glorot/He limits, real draws).
 * That is right for the product but makes a verification sweep non-reproducible, so
 * this tool temporarily substitutes a deterministic PRNG around each run. The training
 * math itself is untouched — same forward/backward code, just a seeded starting point.
 */
function withSeededRandom(seed, fn) {
  const rng = mulberry32(seed);
  const original = Math.random;
  Math.random = rng;
  try {
    return fn();
  } finally {
    Math.random = original;
  }
}

function buildNetwork(inFeatures, numClasses, job) {
  const arch = [{ size: inFeatures }];
  for (const h of job.hidden) arch.push({ size: h, activation: job.hiddenAct || 'relu' });
  arch.push({
    size: job.regression ? 1 : numClasses,
    activation: job.regression ? 'linear' : job.outputAct
  });
  return new NeuralNetwork({
    architecture: arch,
    loss: job.regression ? 'mse' : job.loss,
    optimizer: 'adam',
    learningRate: job.lr,
    weightDecay: 0.0001
  });
}

function sweep(name, ds, job) {
  const values = [];
  const firstBetter = [];

  for (const seed of SEEDS) {
    const value = withSeededRandom(1013 + seed * 7919, () => {
      const split = splitSeeded(ds.X, ds.y, 0.8, seed);
      const net = buildNetwork(ds.numFeatures, ds.numClasses, job);
      let first = null;

      trainEpochs(net, split.trainX, split.trainY, {
        epochs: job.epochs,
        batchSize: job.batch,
        seed: 777,
        onEpoch: (m) => {
          if (first === null && !job.regression) {
            const ev = net.evaluate(split.testX, split.testY);
            if (ev.accuracy >= 0.95) first = m.epoch;
          }
        }
      });

      if (job.regression) {
        const preds = net.predict(split.testX);
        let mae = 0;
        for (let i = 0; i < split.testX.length; i++) mae += Math.abs(preds[i][0] - split.testY[i][0]);
        return mae / split.testX.length;
      }
      firstBetter.push(first);
      return net.evaluate(split.testX, split.testY).accuracy * 100;
    });
    values.push(value);
  }

  const summary = job.regression
    ? {
      kind: 'regression', name,
      medianMAE: +median(values).toFixed(3),
      worstMAE: +Math.max(...values).toFixed(3),
      testCount: 'multiple'
    }
    : {
      kind: 'classification', name,
      perfect: values.filter((a) => a >= 99.99).length,
      atLeast95: values.filter((a) => a >= 95).length,
      medianAcc: +median(values).toFixed(1),
      worstAcc: +Math.min(...values).toFixed(1),
      medianEpochsTo95: (() => {
        const reached = firstBetter.filter((v) => typeof v === 'number');
        return reached.length ? median(reached) : null;
      })()
    };
  summary.distribution = values.map((v) => +v.toFixed(1));

  console.log(
    job.regression
      ? `  ${name.padEnd(30)} MAE median ${summary.medianMAE} worst ${summary.worstMAE}`
      : `  ${name.padEnd(30)} median ${summary.medianAcc}% worst ${summary.worstAcc}% | ` +
        `>=95% in ${summary.atLeast95}/10, perfect ${summary.perfect}/10 | median epoch to 95%: ${summary.medianEpochsTo95}`
  );
  return summary;
}

/* --------------------------------------------------------------------- main */

async function main() {
  const results = { csv: [], builtins: [], mnist: null };

  console.log('=== Classification projects (CSV, 10 seeded 80/20 splits) ===');
  const csvClass = [
    { file: 'spam_screener.csv', name: 'Spam Screener', hidden: [8, 6], outputAct: 'sigmoid', loss: 'binary_crossentropy', lr: 0.03, batch: 8, epochs: 200 },
    { file: 'student_pass_fail.csv', name: 'Student Early-Warning', hidden: [32, 16], outputAct: 'sigmoid', loss: 'binary_crossentropy', lr: 0.03, batch: 8, epochs: 1000 },
    { file: 'bike_decision.csv', name: 'Should I Bother?', hidden: [32, 16], outputAct: 'sigmoid', loss: 'binary_crossentropy', lr: 0.03, batch: 8, epochs: 1000 },
    { file: 'workout_zones.csv', name: 'Workout Zones', hidden: [10, 6], outputAct: 'softmax', loss: 'categorical_crossentropy', lr: 0.03, batch: 8, epochs: 200 }
  ];
  for (const job of csvClass) {
    const text = fs.readFileSync(path.join(ROOT, 'projects', job.file), 'utf8');
    const ds = DatasetManager.parseCSV(text, { name: job.name });
    const summary = sweep(`${job.name} (${job.file})`, ds, job);
    summary.file = job.file;
    summary.rows = ds.X.length;
    summary.features = ds.numFeatures;
    summary.classes = ds.numClasses;
    results.csv.push(summary);
  }

  console.log('\n=== Regression projects (CSV, 10 seeded 80/20 splits) ===');
  const csvReg = [
    { file: 'energy_forecast.csv', name: 'Energy Forecaster', hidden: [16, 16], regression: true },
    { file: 'crop_yield.csv', name: 'Crop Yield Responder', hidden: [16, 16], regression: true }
  ];
  for (const job of csvReg) {
    const text = fs.readFileSync(path.join(ROOT, 'projects', job.file), 'utf8');
    const ds = DatasetManager.parseCSV(text, { name: job.name });
    const summary = sweep(`${job.name} (${job.file})`, ds, { ...job, lr: 0.05, batch: 8, epochs: 800 });
    summary.file = job.file;
    summary.rows = ds.X.length;
    summary.features = ds.numFeatures;
    results.csv.push(summary);
  }

  console.log('\n=== Built-in projects (real generators, 10 seeds) ===');
  const builtins = [
    { name: 'Flower ID (Iris)', builtin: 'iris', dsSeed: 4001, hidden: [16, 8], outputAct: 'softmax', loss: 'categorical_crossentropy', lr: 0.03, batch: 8, epochs: 600, required: true },
    { name: 'Shape lab: single-neuron', builtin: 'moons', dsSeed: 4002, hidden: [2], outputAct: 'sigmoid', loss: 'binary_crossentropy', lr: 0.05, batch: 16, epochs: 300, required: false },
    { name: 'Shape lab: deep [16,16]', builtin: 'moons', dsSeed: 4002, hidden: [16, 16], outputAct: 'sigmoid', loss: 'binary_crossentropy', lr: 0.05, batch: 16, epochs: 300, required: true },
    { name: 'Mini OCR (8x8 digits)', builtin: 'digits', dsSeed: 4004, hidden: [16], outputAct: 'softmax', loss: 'categorical_crossentropy', lr: 0.03, batch: 16, epochs: 200, required: true }
  ];
  for (const job of builtins) {
    // The built-in generators draw their points with Math.random(); seed that draw so the
    // certificate covers one exact, reproducible dataset per project.
    const ds = withSeededRandom(job.dsSeed, () => DatasetManager.getBuiltinDataset(job.builtin));
    const summary = sweep(job.name, ds, job);
    summary.required = job.required;
    summary.dsSeed = job.dsSeed;
    results.builtins.push(summary);
  }

  console.log('\n=== Real MNIST project (playground preset 784-16-8-2, payload 2500/800) ===');
  const mnist = await loadMnistForTests({ maxTrain: 2500, maxTest: 800 });
  const net = withSeededRandom(31337, () => new NeuralNetwork({
    architecture: [{ size: 784 }, { size: 16, activation: 'relu' }, { size: 8, activation: 'relu' }, { size: 2, activation: 'sigmoid' }],
    loss: 'binary_crossentropy', optimizer: 'adam', learningRate: 0.05, weightDecay: 0.0001
  }));
  const history = trainEpochs(net, mnist.trainX, mnist.trainY, { epochs: 10, batchSize: 32, seed: 20261018 });
  const evalMnist = net.evaluate(mnist.testX, mnist.testY);
  results.mnist = {
    epochs: history.length,
    finalTrainLoss: +history[history.length - 1].trainLoss.toFixed(4),
    accuracy: +(evalMnist.accuracy * 100).toFixed(2),
    testCount: evalMnist.count
  };
  console.log(`  MNIST held-out: ${results.mnist.accuracy}% on ${results.mnist.testCount} images ` +
    `(10 epochs, final train loss ${results.mnist.finalTrainLoss})`);

  /* ------------------------------------------------------------ pass gate */
  const failures = [];
  for (const p of results.csv) {
    if (p.kind === 'classification' && p.medianAcc < 95) failures.push(`${p.name}: median ${p.medianAcc}% < 95%`);
    if (p.kind === 'regression') {
      const range = p.file === 'crop_yield.csv' ? 2.51 : 3.89; // measured target range of the shipped CSVs
      if (p.medianMAE > range * 0.1) failures.push(`${p.name}: median MAE ${p.medianMAE} > 10% of range`);
    }
  }
  for (const b of results.builtins) {
    if (b.required && b.kind === 'classification' && b.medianAcc < 95) failures.push(`${b.name}: median ${b.medianAcc}% < 95%`);
  }
  if (results.mnist.accuracy < 95) failures.push(`MNIST: ${results.mnist.accuracy}% < 95%`);

  console.log('\n=== VERDICT ===');
  if (failures.length === 0) {
    console.log(
      `ALL PROJECTS LEARNABLE — ${results.csv.length} CSV projects + ` +
      `${results.builtins.filter((b) => b.required).length} gated built-in configs + MNIST verified ` +
      `(plus the intentionally-struggling single-neuron lab config measured above)`
    );
    return 0;
  }
  console.log('FAILED:');
  for (const f of failures) console.log('  - ' + f);
  return 1;
}

main().then((code) => process.exit(code)).catch((err) => {
  console.error('verification crashed:', err);
  process.exit(1);
});
