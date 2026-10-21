/**
 * Shared MNIST test utilities.
 *
 * Loads the shipped uint8 payload straight off disk and decodes it through the SAME
 * MNISTLoader code path the browser uses, so the verification suite measures the
 * production pipeline rather than a test-only reimplementation.
 *
 * File: tests/mnist_test_utils.js
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MNISTLoader } from '../src/engine/mnist_loader.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = path.resolve(HERE, '..');
export const MNIST_DIR = path.join(PROJECT_ROOT, 'src', 'data', 'mnist_01');

/**
 * @param {object} options { maxTrain, maxTest, verify true }
 * @returns {Promise<object>} dataset descriptor with trainX/trainY/testX/testY
 */
export async function loadMnistForTests(options = {}) {
  const metaPath = path.join(MNIST_DIR, 'meta.json');
  if (!fs.existsSync(metaPath)) {
    throw new Error(
      `Missing ${metaPath}. Run "python tools/fetch_mnist.py" to download and export the real MNIST corpus.`
    );
  }

  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  const buffers = {
    trainX: new Uint8Array(fs.readFileSync(path.join(MNIST_DIR, 'trainX.bin'))),
    trainY: new Uint8Array(fs.readFileSync(path.join(MNIST_DIR, 'trainY.bin'))),
    testX: new Uint8Array(fs.readFileSync(path.join(MNIST_DIR, 'testX.bin'))),
    testY: new Uint8Array(fs.readFileSync(path.join(MNIST_DIR, 'testY.bin')))
  };

  if (options.verify !== false) {
    const integrity = await MNISTLoader.verifyIntegrity(meta, buffers);
    if (!integrity.ok) {
      throw new Error(`MNIST payload failed integrity verification: ${integrity.mismatches.join('; ')}`);
    }
  }

  const dataset = MNISTLoader.fromBuffers(meta, buffers, { sourcePath: MNIST_DIR, ...options });
  dataset.meta = meta;
  return dataset;
}

/** Deterministic Fisher-Yates shuffle (seeded) so test conditions are reproducible. */
export function seededShuffle(array, seed = 1337) {
  let state = seed >>> 0;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
  return array;
}

/** Mini-batch trainer with real shuffling, returning measured epoch metrics. */
export function trainEpochs(network, trainX, trainY, options = {}) {
  const epochs = options.epochs || 10;
  const batchSize = options.batchSize || 32;
  const seed = options.seed || 20261018;
  const onEpoch = options.onEpoch || (() => {});
  const history = [];

  const indices = trainX.map((_, i) => i);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    seededShuffle(indices, seed + epoch);

    let lossSum = 0;
    let accSum = 0;
    let batches = 0;

    for (let start = 0; start < indices.length; start += batchSize) {
      const end = Math.min(start + batchSize, indices.length);
      const batchX = [];
      const batchY = [];
      for (let i = start; i < end; i++) {
        batchX.push(trainX[indices[i]]);
        batchY.push(trainY[indices[i]]);
      }
      const result = network.trainBatch(batchX, batchY);
      lossSum += result.loss;
      accSum += result.accuracy;
      batches++;
    }

    const metrics = {
      epoch,
      trainLoss: lossSum / Math.max(1, batches),
      trainAccuracy: accSum / Math.max(1, batches)
    };
    history.push(metrics);
    onEpoch(metrics);
  }

  return history;
}

export function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

export function banner(title) {
  const line = '='.repeat(Math.max(60, title.length + 8));
  console.log(`\n${line}\n  ${title}\n${line}`);
}
