/**
 * Automated Edge-Case & Numerical Guardrails Test Suite
 * Tests Malformed CSVs, Zero-Variance Inputs, NaN Protection, Rapid Resets,
 * Learning-Rate Explosions, Empty Batches and Degenerate Single-Class Batches.
 * Neural Network Studio Platform
 */

import { NeuralNetwork } from '../src/engine/neural_network.js';
import { DatasetManager } from '../src/engine/dataset_manager.js';

function runEdgeCaseTests() {
  console.log('====================================================');
  console.log('     EDGE CASE & GUARDRAIL INTEGRITY TEST SUITE     ');
  console.log('====================================================\n');

  let passed = 0;
  let total = 7;

  // Test 1: Malformed CSV with missing values, string headers, trailing lines
  console.log('[Test 1/7] Malformed CSV Parsing & Imputation...');
  const dirtyCsv = `
    "Feature A", "Feature B", "Target Label"
    1.2, 3.4, "Cat"
    5.6, NaN, "Dog"
    , 7.8, "Cat"
    9.0, 1.2, "Dog"
    
    11.1, 13.5, "Cat"
  `;

  try {
    const parsed = DatasetManager.parseCSV(dirtyCsv);
    if (parsed.X.length === 5 && parsed.numFeatures === 2 && parsed.numClasses === 2) {
      // Check that no NaN exists in parsed X
      const hasNaN = parsed.X.some(row => row.some(v => isNaN(v)));
      if (!hasNaN) {
        console.log('  [PASS] Malformed CSV parsed & NaN imputed cleanly.\n');
        passed++;
      } else {
        console.error('  [FAIL] Parsed dataset contained NaN values!\n');
      }
    } else {
      console.error('  [FAIL] Parsed row count mismatch:', parsed.X.length);
    }
  } catch (err) {
    console.error('  [FAIL] CSV parser threw unexpected error:', err);
  }

  // Test 2: Zero-Variance Feature Protection (Constant Columns)
  console.log('[Test 2/7] Zero-Variance Feature Guardrail...');
  const constantFeaturesMatrix = [
    [5.0, 1.0],
    [5.0, 2.0],
    [5.0, 3.0],
    [5.0, 4.0]
  ];
  const normResult = DatasetManager.normalize(constantFeaturesMatrix);
  const col0IsZero = normResult.data.every(row => row[0] === 0.0);
  const col1HasNoNaN = normResult.data.every(row => !isNaN(row[1]));

  if (col0IsZero && col1HasNoNaN && normResult.params[0].isConstant) {
    console.log('  [PASS] Zero-variance constant feature correctly clamped to 0.0 with no division by zero.\n');
    passed++;
  } else {
    console.error('  [FAIL] Zero-variance column caused NaN or improper scaling!\n');
  }

  // Test 3: Numerical Stability under Extreme Inputs
  console.log('[Test 3/7] Extreme Floating-Point Input Guardrail...');
  const extremeNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 4, activation: 'sigmoid' },
      { size: 2, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam'
  });

  const extremeInputs = [
    [1e8, -1e8],
    [-1e9, 1e9],
    [0.0, 0.0]
  ];
  const extremeTargets = [
    [1, 0],
    [0, 1],
    [1, 0]
  ];

  let nanDetected = false;
  for (let step = 0; step < 10; step++) {
    const stepRes = extremeNet.trainBatch(extremeInputs, extremeTargets);
    if (isNaN(stepRes.loss)) nanDetected = true;
  }
  const extremePreds = extremeNet.predict(extremeInputs);
  const hasNanPred = extremePreds.some(row => row.some(v => isNaN(v)));

  if (!nanDetected && !hasNanPred) {
    console.log('  [PASS] Numerical stability verified: Extreme inputs produced finite weights and probabilities.\n');
    passed++;
  } else {
    console.error('  [FAIL] Extreme inputs resulted in NaN values!\n');
  }

  // Test 4: Model Serialization & Deserialization Parity
  console.log('[Test 4/7] Model JSON Export / Import Numerical Parity...');
  const originalNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 6, activation: 'relu' },
      { size: 2, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam'
  });
  originalNet.trainBatch([[0.2, -0.4], [0.8, 0.1]], [[1, 0], [0, 1]]);

  const modelJson = originalNet.toJSON();
  const restoredNet = NeuralNetwork.fromJSON(modelJson);

  const testSample = [0.5, -0.3];
  const predOrig = originalNet.forwardSample(testSample).output;
  const predRestored = restoredNet.forwardSample(testSample).output;

  let maxDiff = 0;
  for (let i = 0; i < predOrig.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(predOrig[i] - predRestored[i]));
  }

  if (maxDiff < 1e-12) {
    console.log(`  [PASS] JSON Export / Import verified with exact floating point parity (delta < 1e-12).\n`);
    passed++;
  } else {
    console.error(`  [FAIL] Parity mismatch between original and restored model: ${maxDiff}\n`);
  }

  // Test 5: Learning-Rate Explosion Guardrails
  console.log('[Test 5/7] Learning-Rate Explosion & Optimizer Clamp...');
  const lrNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 6, activation: 'sigmoid' },
      { size: 2, activation: 'sigmoid' }
    ],
    loss: 'binary_crossentropy',
    optimizer: 'adam',
    learningRate: 0.03
  });

  const clampedHigh = lrNet.setLearningRate(1e9);
  const clampedQuantum = lrNet.setLearningRate(Number.MAX_VALUE);
  const clampedNegative = lrNet.setLearningRate(-5);
  const clampedZero = lrNet.setLearningRate(0);
  const clampedNaN = lrNet.setLearningRate(NaN);
  lrNet.setLearningRate(0.02);

  const clampWorks =
    clampedHigh === NeuralNetwork.MAX_LEARNING_RATE &&
    clampedQuantum === NeuralNetwork.MAX_LEARNING_RATE &&
    clampedNegative === NeuralNetwork.MIN_LEARNING_RATE &&
    clampedZero === NeuralNetwork.MIN_LEARNING_RATE &&
    clampedNaN === NeuralNetwork.MIN_LEARNING_RATE &&
    Math.abs(lrNet.learningRate - 0.02) < 1e-15 &&
    Math.abs(lrNet.optimizer.lr - 0.02) < 1e-15;

  // Even with the clamp DISABLED on purpose (adversarial internal bypass), a 1e9 learning
  // rate must not corrupt the model into NaN/Inf, and probabilities must stay valid.
  lrNet.learningRate = 1e9;
  const bombX = [[1.0, 0.5], [0.2, 0.9], [0.7, 0.1], [0.4, 0.4]];
  const bombY = [[1, 0], [0, 1], [1, 0], [0, 1]];
  let bombFinite = true;
  let bombLoss = 0;
  for (let step = 0; step < 10; step++) {
    const res = lrNet.trainBatch(bombX, bombY);
    bombLoss = res.loss;
    if (!Number.isFinite(res.loss)) bombFinite = false;
  }
  const bombPreds = lrNet.predict(bombX);
  const bombProbsValid = bombPreds.every(row => row.every(p => Number.isFinite(p) && p >= 0 && p <= 1));
  const bombRepair = lrNet.sanitizeWeights();

  const bombSafe =
    clampWorks &&
    bombFinite &&
    Number.isFinite(bombLoss) &&
    bombProbsValid &&
    Number.isFinite(lrNet.maxAbsWeight()) &&
    typeof bombRepair.repaired === 'number';

  if (bombSafe) {
    console.log(`  [PASS] 1e9 learning rate clamped to ${NeuralNetwork.MAX_LEARNING_RATE}; adversarial bypass kept loss ` +
                `finite (${bombLoss.toFixed(4)}), probabilities in [0,1], repairs ${bombRepair.repaired}.\n`);
    passed++;
  } else {
    console.error(`  [FAIL] Learning-rate explosion produced invalid state ` +
                  `(clamps ok=${clampWorks}, loss finite=${bombFinite}, probs valid=${bombProbsValid}).\n`);
  }

  // Test 6: Empty / Null Batch Robustness
  console.log('[Test 6/7] Empty Batch & Missing Dataset Guardrails...');
  const emptyNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 4, activation: 'relu' },
      { size: 2, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam'
  });

  const emptyTrain = emptyNet.trainBatch([], []);
  const nullTrain = emptyNet.trainBatch(null, null);
  const emptyEval = emptyNet.evaluate([], []);
  const emptyPredict = emptyNet.predict([]);
  const bareNet = new NeuralNetwork({ loss: 'mse' });
  const bareTrain = bareNet.trainBatch([[0.5, 0.5]], [[1]]);

  const emptySafe =
    emptyTrain.batchSize === 0 && emptyTrain.loss === 0 && emptyTrain.accuracy === 0 &&
    nullTrain.batchSize === 0 &&
    emptyEval.count === 0 && emptyEval.loss === 0 && emptyEval.accuracy === 0 && emptyEval.misclassified.length === 0 &&
    emptyPredict.length === 0 &&
    bareTrain.batchSize === 0 && Number.isFinite(bareTrain.loss);

  if (emptySafe) {
    console.log('  [PASS] Empty/null batches and architecture-less networks return honest zero metrics without NaN.\n');
    passed++;
  } else {
    console.error('  [FAIL] Empty batch handling produced unexpected metrics!\n');
  }

  // Test 7: Degenerate Single-Class Batch (constant target and constant input)
  console.log('[Test 7/7] Degenerate Single-Class Batch Stability...');
  const singleClassNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 6, activation: 'relu' },
      { size: 2, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam',
    learningRate: 0.05
  });

  const constantX = Array.from({ length: 8 }, () => [0.7, 0.2]);
  const constantY = Array.from({ length: 8 }, () => [1, 0]);
  let firstLoss = null;
  let lastLoss = 0;
  let singleFinite = true;
  for (let step = 0; step < 30; step++) {
    const res = singleClassNet.trainBatch(constantX, constantY);
    if (firstLoss === null) firstLoss = res.loss;
    lastLoss = res.loss;
    if (!Number.isFinite(res.loss)) singleFinite = false;
  }
  const singlePreds = singleClassNet.predict(constantX);
  const singleValid = singlePreds.every(row =>
    row.length === 2 &&
    row.every(p => Number.isFinite(p) && p >= 0 && p <= 1) &&
    Math.abs(row[0] + row[1] - 1) < 1e-9);
  const singleConverged = singlePreds.every(row => row[0] >= row[1]);

  if (singleFinite && singleValid && singleConverged && lastLoss < firstLoss) {
    console.log(`  [PASS] Constant single-class batch stays stable: loss ${firstLoss.toFixed(4)} -> ${lastLoss.toFixed(4)}, ` +
                `softmax outputs remain a valid distribution.\n`);
    passed++;
  } else {
    console.error(`  [FAIL] Degenerate single-class batch destabilised (finite=${singleFinite}, ` +
                  `valid=${singleValid}, converged=${singleConverged}, loss ${firstLoss} -> ${lastLoss}).\n`);
  }

  console.log(`====================================================`);
  console.log(`   GUARDRAILS SUMMARY: ${passed}/${total} PASSED`);
  console.log(`====================================================\n`);

  if (passed !== total) {
    process.exit(1);
  }
}

runEdgeCaseTests();
