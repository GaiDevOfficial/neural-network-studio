/**
 * PHASE 3 — ONNX export verification.
 *
 * The exporter writes the ONNX protobuf by hand (src/export/onnx_writer.js). This suite proves the
 * bytes are a real ONNX model and that its numbers match the JS engine:
 *
 *   1. XOR (Relu → Sigmoid) trained in the real engine,
 *   2. Iris (Relu → Softmax) trained in the real engine,
 *   3. the MNIST 0-vs-1 preset architecture (Relu → Sigmoid, 784 inputs),
 *   4. a mixed-activation network (LeakyRelu → Tanh → Sigmoid),
 *   5. a linear regression network (linear output, MSE).
 *
 * Each model is written to disk, validated with the official `onnx.checker`, executed with
 * onnxruntime, and compared against the same inputs run through `NeuralNetwork.forwardSample`.
 * No expected outputs are hard-coded: both sides are executed.
 *
 * Requirements: `pip install onnx onnxruntime`.
 *
 * File: tests/test_onnx_export.js
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { NeuralNetwork } from '../src/engine/neural_network.js';
import { DatasetManager } from '../src/engine/dataset_manager.js';
import { instantPreset } from '../src/graph/smart_assist.js';
import { compile } from '../src/graph/node_engine.js';
import { datasetInfo } from '../src/graph/data_connectors.js';
import { buildOnnxBytes, ONNX_INFO } from '../src/export/onnx_writer.js';
import { buildExportArtifacts } from '../src/export/builder.js';
import { trainEpochs } from './mnist_test_utils.js';

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

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  }
  return worst;
}

function maxDiffNested(actual, expected) {
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, maxAbsDiff(actual[i], expected[i]));
  }
  return worst;
}

function trainFromPlan(plan, dataset, epochs) {
  const split = DatasetManager.splitTrainTest(dataset.X, dataset.y, 0.8);
  const net = new NeuralNetwork({
    architecture: plan.network.architecture,
    loss: plan.network.loss,
    optimizer: plan.network.optimizer,
    learningRate: plan.network.learningRate,
    weightDecay: plan.network.weightDecay
  });
  trainEpochs(net, split.trainX, split.trainY, { epochs, batchSize: 16 });
  return net;
}

/** Deterministic pseudo-random values in [-1, 1] so failures are reproducible. */
function seededVector(seed, length) {
  const values = [];
  for (let i = 0; i < length; i++) {
    const t = Math.sin((i + 1) * 12.9898 + seed * 78.233) * 43758.5453;
    values.push(Number(((t - Math.floor(t)) * 2 - 1).toFixed(6)));
  }
  return values;
}

function runChecker(payloadPath) {
  const script = path.join(process.cwd(), 'tools', 'onnx_parity_check.py');
  const result = spawnSync('python', [script, payloadPath], {
    encoding: 'utf8',
    timeout: 120000
  });
  if (result.status !== 0) {
    throw new Error(
      `onnx parity checker failed (${result.status}):\n${result.stderr || result.stdout}\n` +
      'Install the checker with: pip install onnx onnxruntime'
    );
  }
  return JSON.parse(result.stdout);
}

function verifyModel({ label, weights, probes, plan = null, workDir, tolerance = 1e-5 }) {
  console.log(`--- ${label} ---`);
  const net = new NeuralNetwork({
    architecture: [
      { size: weights[0].inputSize },
      ...weights.map((layer) => ({ size: layer.outputSize, activation: layer.activation }))
    ],
    loss: plan && plan.network ? plan.network.loss : 'mse',
    optimizer: 'adam',
    learningRate: 0.01
  });
  net.layers.forEach((layer, index) => {
    layer.weights = weights[index].weights.map((row) => Float64Array.from(row));
    layer.biases = Float64Array.from(weights[index].biases);
  });

  const engineOutputs = probes.map((probe) => Array.from(net.forwardSample(probe).output));
  const bytes = buildOnnxBytes({ plan, weights, datasetName: label, epochs: 7 });
  const modelPath = path.join(workDir, `${label.replace(/[^a-z0-9]+/gi, '_')}.onnx`);
  fs.writeFileSync(modelPath, Buffer.from(bytes));

  check(`${label}: exporter returns non-trivial protobuf bytes`, bytes.length > 100, `${bytes.length} bytes`);

  const payloadPath = path.join(workDir, `${path.basename(modelPath, '.onnx')}_probes.json`);
  fs.writeFileSync(payloadPath, JSON.stringify({ model: modelPath, inputs: probes }), 'utf8');
  const result = runChecker(payloadPath);

  check(`${label}: official ONNX checker accepts the model`, result.ok === true);
  check(`${label}: IR version is ${ONNX_INFO.irVersion}`, result.ir_version === ONNX_INFO.irVersion, String(result.ir_version));
  check(`${label}: opset is ${ONNX_INFO.opset}`, result.opset === ONNX_INFO.opset, String(result.opset));
  check(`${label}: initializer count is 2 × layers`, result.initializers === weights.length * 2, String(result.initializers));
  check(`${label}: input shape is [1, ${weights[0].inputSize}]`,
    JSON.stringify(result.input_dims) === JSON.stringify([1, weights[0].inputSize]), JSON.stringify(result.input_dims));
  check(`${label}: output shape is [1, ${weights[weights.length - 1].outputSize}]`,
    JSON.stringify(result.output_dims) === JSON.stringify([1, weights[weights.length - 1].outputSize]), JSON.stringify(result.output_dims));

  const worst = maxDiffNested(result.outputs, engineOutputs);
  check(`${label}: onnxruntime matches the JS engine`, worst <= tolerance,
    `max |Δ| = ${worst.toExponential(3)} (tolerance ${tolerance.toExponential(1)})`);
  console.log(`         max |Δ| = ${worst.toExponential(3)} · ${bytes.length} bytes · ${result.graph_nodes} nodes`);

  const artifacts = plan
    ? buildExportArtifacts({ plan, weights, datasetName: label, epochs: 7 })
    : null;
  if (artifacts) {
    check(`${label}: builder exposes the ONNX artifact`,
      artifacts.onnxBytes instanceof Uint8Array && artifacts.onnxBytes.length === bytes.length);
  }
}

function main() {
  console.log('=====================================================');
  console.log('   NODE STUDIO — ONNX EXPORT VERIFICATION             ');
  console.log('=====================================================\n');

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nnstudio_onnx_'));

  const probe = spawnSync('python', ['-c', 'import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)'], {
    encoding: 'utf8'
  });
  if (probe.status !== 0) {
    console.error('  [FAIL] onnx + onnxruntime are required for this suite.');
    console.error('         python -m pip install onnx onnxruntime');
    process.exit(1);
  }
  console.log(`  checker: onnx ${probe.stdout.trim().split(/\s+/)[0]}, onnxruntime ${probe.stdout.trim().split(/\s+/)[1]}\n`);

  /* 1 — XOR, trained */
  const xor = DatasetManager.getBuiltinDataset('xor');
  const xorPlan = compile(instantPreset('xor_classifier'), {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(xor) : null)
  });
  check('XOR graph compiles', xorPlan.ok === true, JSON.stringify((xorPlan.errors || []).map((error) => error.code)));
  const xorNet = trainFromPlan(xorPlan, xor, 40);
  verifyModel({
    label: 'xor',
    weights: xorNet.getWeightsSnapshot(),
    probes: [[0, 0], [0, 1], [1, 0], [1, 1], [0.25, 0.75]],
    plan: xorPlan,
    workDir,
    tolerance: 1e-5
  });

  /* 2 — Iris, trained Softmax */
  const iris = DatasetManager.getBuiltinDataset('iris');
  const irisPlan = compile(instantPreset('iris_multi_class'), {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(iris) : null)
  });
  check('Iris graph compiles', irisPlan.ok === true, JSON.stringify((irisPlan.errors || []).map((error) => error.code)));
  const irisSplit = DatasetManager.splitTrainTest(iris.X, iris.y, 0.8);
  verifyModel({
    label: 'iris',
    weights: trainFromPlan(irisPlan, iris, 10).getWeightsSnapshot(),
    probes: irisSplit.testX.slice(0, 3),
    plan: irisPlan,
    workDir,
    tolerance: 1e-5
  });

  /* 3 — MNIST 0-vs-1 preset architecture, random initialization (784-input parity) */
  const mnistInfo = {
    type: 'classification',
    name: 'MNIST 0 vs 1',
    numFeatures: 784,
    numClasses: 2,
    imageDim: { width: 28, height: 28 }
  };
  const mnistPlan = compile(instantPreset('mnist_classifier'), {
    datasetInfoFor: (node) => (node.type === 'source' ? mnistInfo : null)
  });
  check('MNIST preset compiles (784 → 16 → 8 → 2)',
    mnistPlan.ok === true && mnistPlan.meta.flatInputSize === 784,
    JSON.stringify((mnistPlan.errors || []).map((error) => error.code)));
  const mnistNet = new NeuralNetwork({
    architecture: mnistPlan.network.architecture,
    loss: mnistPlan.network.loss,
    optimizer: 'adam',
    learningRate: 0.001
  });
  verifyModel({
    label: 'mnist_0_vs_1',
    weights: mnistNet.getWeightsSnapshot(),
    probes: [seededVector(1, 784), seededVector(2, 784)],
    plan: mnistPlan,
    workDir,
    tolerance: 2e-5
  });

  /* 4 — LeakyRelu + Tanh + Sigmoid */
  const mixedNet = new NeuralNetwork({
    architecture: [
      { size: 3 },
      { size: 6, activation: 'leaky_relu' },
      { size: 4, activation: 'tanh' },
      { size: 2, activation: 'sigmoid' }
    ],
    loss: 'bce',
    optimizer: 'adam',
    learningRate: 0.01
  });
  verifyModel({
    label: 'mixed_activations',
    weights: mixedNet.getWeightsSnapshot(),
    probes: [seededVector(3, 3), seededVector(4, 3)],
    workDir,
    tolerance: 1e-5
  });

  /* 5 — Linear regression output */
  const linearNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 5, activation: 'relu' },
      { size: 1, activation: 'linear' }
    ],
    loss: 'mse',
    optimizer: 'sgd',
    learningRate: 0.01
  });
  verifyModel({
    label: 'linear_output',
    weights: linearNet.getWeightsSnapshot(),
    probes: [[0.5, -0.25], [-1, 1]],
    workDir,
    tolerance: 1e-5
  });

  /* 6 — Refusals */
  console.log('--- exporter refusals ---');
  let refusedUntrained = false;
  try {
    buildOnnxBytes({ weights: [] });
  } catch {
    refusedUntrained = true;
  }
  check('empty/untrained weight snapshots are refused', refusedUntrained);

  let refusedShape = false;
  try {
    const bad = linearNet.getWeightsSnapshot();
    bad[0].weights.pop();
    buildOnnxBytes({ weights: bad });
  } catch {
    refusedShape = true;
  }
  check('malformed weight matrices are refused', refusedShape);

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main();
