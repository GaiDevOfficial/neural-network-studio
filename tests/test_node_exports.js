/**
 * PHASE 3 — Export artifact verification.
 *
 * Trains two compiled graphs (XOR → Sigmoid/BCE, Iris → Softmax/CCE) in the real engine,
 * exports them through src/export/builder.js, and then measures each artifact for real:
 *
 *   - the generated Python file is executed with the system Python + NumPy and its demo
 *     output must match the engine's forward pass within 1e-9,
 *   - the generated JavaScript module is imported in-process and must match within 1e-12,
 *   - the generated .npz is opened with Python's zipfile + numpy: array shapes, weight
 *     sums and the embedded meta.json must match the snapshot,
 *   - the model JSON payload must round-trip through NeuralNetwork.fromJSON with identical
 *     predictions,
 *   - the packaging commands must be the real repo scripts.
 *
 * Nothing here re-implements the generated code — it is executed.
 *
 * File: tests/test_node_exports.js
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { NeuralNetwork } from '../src/engine/neural_network.js';
import { DatasetManager } from '../src/engine/dataset_manager.js';
import { instantPreset } from '../src/graph/smart_assist.js';
import { compile } from '../src/graph/node_engine.js';
import { datasetInfo } from '../src/graph/data_connectors.js';
import { buildExportArtifacts, buildCommands } from '../src/export/builder.js';
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

function runPython(scriptPath, extraArgs = []) {
  const args = extraArgs.map((a) => `"${a}"`).join(' ');
  const result = spawnSync(`python "${scriptPath}"${args ? ` ${args}` : ''}`, {
    shell: true,
    encoding: 'utf8',
    timeout: 120000
  });
  if (result.status !== 0) {
    throw new Error(`python ${path.basename(scriptPath)} failed (${result.status}):\n${result.stderr || result.stdout}`);
  }
  return result.stdout;
}

function parsePythonDemoOutput(stdout) {
  const line = stdout.split(/\r?\n/).find((l) => l.trim().startsWith('output:'));
  if (!line) throw new Error(`generated script printed no output line:\n${stdout}`);
  return JSON.parse(line.slice(line.indexOf('output:') + 'output:'.length).trim());
}

/** Import a generated Python model as a module and predict arbitrary inputs through it. */
function pythonPredictions(modelPath, inputs) {
  const probePath = path.join(path.dirname(modelPath), 'probe_predict.py');
  fs.writeFileSync(probePath, [
    'import importlib.util, json, sys',
    'spec = importlib.util.spec_from_file_location("generated_model", sys.argv[1])',
    'module = importlib.util.module_from_spec(spec)',
    'spec.loader.exec_module(module)',
    'net = module.ExportedNeuralNetwork()',
    'inputs = json.loads(sys.argv[2])',
    'print("predictions:", json.dumps([net.predict(x).tolist() for x in inputs]))'
  ].join('\n'), 'utf8');
  const stdout = runPython(probePath, [modelPath, JSON.stringify(inputs)]);
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith('predictions:'));
  if (!line) throw new Error(`probe printed no predictions line:\n${stdout}`);
  return JSON.parse(line.slice('predictions:'.length).trim());
}

function maxDiffNested(actual, expected) {
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, maxAbsDiff(actual[i], expected[i]));
  }
  return worst;
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    worst = Math.max(worst, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
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

async function main() {
  console.log('=====================================================');
  console.log('   NODE STUDIO — EXPORT ARTIFACT VERIFICATION         ');
  console.log('=====================================================\n');

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nnstudio_exports_'));

  /* ---------------------------------------------------------------- XOR (Sigmoid) */

  console.log('--- XOR model (Sigmoid output, standalone Python + JS parity) ---');
  const xor = DatasetManager.getBuiltinDataset('xor');
  const xorPlan = compile(instantPreset('xor_classifier'), {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(xor) : null)
  });
  check('XOR graph compiles', xorPlan.ok === true, JSON.stringify((xorPlan.errors || []).map((e) => e.code)));

  const xorNet = trainFromPlan(xorPlan, xor, 40);
  const xorWeights = xorNet.getWeightsSnapshot();
  const xorArtifacts = buildExportArtifacts({ plan: xorPlan, weights: xorWeights, datasetName: xor.name, epochs: 40 });

  const demoInput = new Array(xorPlan.network.architecture[0].size).fill(0.5);
  const engineOut = Array.from(xorNet.forwardSample(demoInput).output);

  // Standalone Python
  const pyPath = path.join(workDir, 'xor_model.py');
  fs.writeFileSync(pyPath, xorArtifacts.pythonSource, 'utf8');
  const pyOut = parsePythonDemoOutput(runPython(pyPath));
  check(`generated Python runs under real Python/NumPy and predicts [${pyOut.map((v) => v.toFixed(6)).join(', ')}]`,
    pyOut.length === engineOut.length);
  check(`generated Python matches the engine (max |Δ| = ${maxAbsDiff(pyOut, engineOut).toExponential(2)})`,
    maxAbsDiff(pyOut, engineOut) < 1e-9);

  const xorProbeInputs = [[0.1, 0.9], [0.75, 0.25], [0.02, 0.98]];
  const xorProbeEngine = xorProbeInputs.map((inp) => Array.from(xorNet.forwardSample(inp).output));
  const xorProbePy = pythonPredictions(pyPath, xorProbeInputs);
  check(`generated Python matches the engine on arbitrary inputs (max |Δ| = ${maxDiffNested(xorProbePy, xorProbeEngine).toExponential(2)})`,
    maxDiffNested(xorProbePy, xorProbeEngine) < 1e-9);

  // Standalone JavaScript
  const jsPath = path.join(workDir, 'xor_model.mjs');
  fs.writeFileSync(jsPath, xorArtifacts.jsSource, 'utf8');
  const jsModule = await import(pathToFileURL(jsPath).href);
  const jsNet = new jsModule.ExportedNeuralNetwork();
  const jsOut = jsNet.predict(demoInput);
  check(`generated JS module matches the engine (max |Δ| = ${maxAbsDiff(jsOut, engineOut).toExponential(2)})`,
    maxAbsDiff(jsOut, engineOut) < 1e-12);
  check('generated JS embeds the model metadata', jsModule.MODEL_META.origin === 'node_studio' && jsModule.MODEL_META.loss === xorPlan.network.loss);

  // Model JSON round-trip
  const loaded = NeuralNetwork.fromJSON(xorArtifacts.jsonPayload);
  const loadedOut = Array.from(loaded.forwardSample(demoInput).output);
  check(`model JSON reloads through NeuralNetwork.fromJSON with identical output (max |Δ| = ${maxAbsDiff(loadedOut, engineOut).toExponential(2)})`,
    maxAbsDiff(loadedOut, engineOut) < 1e-12);
  check('model JSON carries loss + optimizer + training config',
    xorArtifacts.jsonPayload.loss === xorPlan.network.loss &&
    xorArtifacts.jsonPayload.optimizer === xorPlan.network.optimizer &&
    xorArtifacts.jsonPayload.metadata.epochs === 40);

  /* ------------------------------------------------------------------------- NPZ */

  const npzPath = path.join(workDir, 'xor_model.npz');
  fs.writeFileSync(npzPath, Buffer.from(xorArtifacts.npzBytes));
  const npzCheckPath = path.join(workDir, 'check_npz.py');
  fs.writeFileSync(npzCheckPath, [
    'import json, sys, zipfile',
    'import numpy as np',
    'archive = zipfile.ZipFile(sys.argv[1])',
    'names = sorted(archive.namelist())',
    'w = np.load(archive.open("layer0_weights.npy"))',
    'b = np.load(archive.open("layer0_biases.npy"))',
    'meta = json.loads(archive.read("meta.json").decode("utf-8"))',
    'print(json.dumps({',
    '  "names": names,',
    '  "shape_w": list(w.shape), "shape_b": list(b.shape),',
    '  "sum_w": float(w.sum()), "sum_b": float(b.sum()),',
    '  "origin": meta.get("origin"), "loss": meta.get("loss"),',
    '}))'
  ].join('\n'), 'utf8');
  const npzReport = JSON.parse(runPython(npzCheckPath, [npzPath]).trim().split(/\r?\n/).pop());

  const expectedNames = ['meta.json'];
  xorWeights.forEach((_, idx) => expectedNames.push(`layer${idx}_biases.npy`, `layer${idx}_weights.npy`));
  expectedNames.sort();
  check('NPZ opens with Python zipfile and contains one .npy pair per layer + meta.json',
    JSON.stringify(npzReport.names) === JSON.stringify(expectedNames), JSON.stringify(npzReport.names));

  const layer0 = xorWeights[0];
  check(`NPZ layer0 arrays load via numpy with the engine's shape (${npzReport.shape_w.join('×')} weights / ${npzReport.shape_b.join('×')} biases)`,
    JSON.stringify(npzReport.shape_w) === JSON.stringify([layer0.outputSize, layer0.inputSize]) &&
    JSON.stringify(npzReport.shape_b) === JSON.stringify([layer0.outputSize]));

  const jsSumW = layer0.weights.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0);
  const jsSumB = layer0.biases.reduce((s, v) => s + v, 0);
  check(`NPZ weight values are the trained weights (numpy ΣW matches JS ΣW within 1e-9)`,
    Math.abs(npzReport.sum_w - jsSumW) < 1e-9 && Math.abs(npzReport.sum_b - jsSumB) < 1e-9,
    `npz ΣW=${npzReport.sum_w.toFixed(9)} js ΣW=${jsSumW.toFixed(9)}`);
  check('NPZ meta.json carries the node-studio origin and loss', npzReport.origin === 'node_studio' && npzReport.loss === xorPlan.network.loss);

  /* ---------------------------------------------------------------- Iris (Softmax) */

  console.log('\n--- Iris model (Softmax output, parity for the softmax path) ---');
  const iris = DatasetManager.getBuiltinDataset('iris');
  const irisPlan = compile(instantPreset('iris_multi_class'), {
    datasetInfoFor: (node) => (node.type === 'source' ? datasetInfo(iris) : null)
  });
  check('Iris graph compiles (softmax + CCE)', irisPlan.ok === true && irisPlan.network.loss === 'categorical_crossentropy',
    JSON.stringify((irisPlan.errors || []).map((e) => e.code)));

  const irisNet = trainFromPlan(irisPlan, iris, 60);
  const irisWeights = irisNet.getWeightsSnapshot();
  const irisArtifacts = buildExportArtifacts({ plan: irisPlan, weights: irisWeights, datasetName: iris.name, epochs: 60 });

  const irisInput = new Array(irisPlan.network.architecture[0].size).fill(0.25);
  const irisDemoInput = new Array(irisPlan.network.architecture[0].size).fill(0.5);
  const irisEngineOut = Array.from(irisNet.forwardSample(irisInput).output);
  const irisEngineDemoOut = Array.from(irisNet.forwardSample(irisDemoInput).output);
  check('engine softmax output is a real probability vector', Math.abs(irisEngineOut.reduce((s, v) => s + v, 0) - 1) < 1e-9);

  const irisPyPath = path.join(workDir, 'iris_model.py');
  fs.writeFileSync(irisPyPath, irisArtifacts.pythonSource, 'utf8');
  const irisPyOut = parsePythonDemoOutput(runPython(irisPyPath));
  check(`generated Python softmax matches the engine on the demo input (max |Δ| = ${maxAbsDiff(irisPyOut, irisEngineDemoOut).toExponential(2)})`,
    maxAbsDiff(irisPyOut, irisEngineDemoOut) < 1e-9);

  const irisProbeInputs = [irisInput, [0.9, 0.1, 0.5, 0.3], [1.0, 0.0, 0.0, 0.0]];
  const irisProbeEngine = irisProbeInputs.map((inp) => Array.from(irisNet.forwardSample(inp).output));
  const irisProbePy = pythonPredictions(irisPyPath, irisProbeInputs);
  check(`generated Python softmax matches the engine on arbitrary inputs (max |Δ| = ${maxDiffNested(irisProbePy, irisProbeEngine).toExponential(2)})`,
    maxDiffNested(irisProbePy, irisProbeEngine) < 1e-9);
  check('generated Python softmax vectors sum to 1 for every probed input',
    irisProbePy.every((vector) => Math.abs(vector.reduce((s, v) => s + v, 0) - 1) < 1e-9));

  const irisJsPath = path.join(workDir, 'iris_model.mjs');
  fs.writeFileSync(irisJsPath, irisArtifacts.jsSource, 'utf8');
  const irisJsModule = await import(pathToFileURL(irisJsPath).href);
  const irisJsOut = new irisJsModule.ExportedNeuralNetwork().predict(irisInput);
  check(`generated JS softmax matches the engine (max |Δ| = ${maxAbsDiff(irisJsOut, irisEngineOut).toExponential(2)})`,
    maxAbsDiff(irisJsOut, irisEngineOut) < 1e-12);

  /* ------------------------------------------------------------------ Commands */

  const commands = buildCommands();
  check('packaging commands are the real repo scripts',
    commands.some((c) => c.command === 'npm run build:exe') &&
    commands.some((c) => c.command.includes('--web-only')) &&
    commands.some((c) => c.command === 'npm start'));

  /* ------------------------------------------------------------------ Teardown */

  fs.rmSync(workDir, { recursive: true, force: true });

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('\n[FATAL]', err);
  process.exit(1);
});
