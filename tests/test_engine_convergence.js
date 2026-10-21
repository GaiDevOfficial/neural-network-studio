/**
 * Automated Node.js Convergence Test Suite
 * Tests JavaScript Neural Engine on XOR, Concentric Circles, and Iris Flower Datasets
 * Neural Network Studio Platform
 */

import { NeuralNetwork } from '../src/engine/neural_network.js';
import { DatasetManager } from '../src/engine/dataset_manager.js';

function runConvergenceTests() {
  console.log('====================================================');
  console.log('   NEURAL NETWORK ENGINE CONVERGENCE TEST SUITE     ');
  console.log('====================================================\n');

  let testsPassed = 0;
  let totalTests = 4;

  // 1. XOR Test
  console.log('[Test 1/4] Testing XOR Non-linear Convergence...');
  const xorData = DatasetManager.generateXOR(120, 0.04);
  const xorNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 8, activation: 'relu' },
      { size: 6, activation: 'relu' },
      { size: 2, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam',
    learningRate: 0.08
  });

  let initialXorLoss = 0;
  for (let epoch = 0; epoch < 180; epoch++) {
    const res = xorNet.trainBatch(xorData.X, xorData.y);
    if (epoch === 0) initialXorLoss = res.loss;
  }

  const xorPreds = xorNet.predict(xorData.X);
  let xorCorrect = 0;
  for (let i = 0; i < xorData.X.length; i++) {
    const predClass = xorPreds[i].indexOf(Math.max(...xorPreds[i]));
    const trueClass = xorData.y[i].indexOf(Math.max(...xorData.y[i]));
    if (predClass === trueClass) xorCorrect++;
  }
  const xorAcc = xorCorrect / xorData.X.length;
  console.log(`  -> XOR Accuracy: ${(xorAcc * 100).toFixed(1)}% (Target: >= 90%)`);

  if (xorAcc >= 0.90) {
    console.log('  [PASS] XOR Convergence Verified.\n');
    testsPassed++;
  } else {
    console.error('  [FAIL] XOR Convergence below threshold!\n');
  }

  // 2. Concentric Circles Test
  console.log('[Test 2/3] Testing Concentric Circles Radial Convergence...');
  const circleData = DatasetManager.generateCircles(160, 0.04);
  const circleNet = new NeuralNetwork({
    architecture: [
      { size: 2 },
      { size: 8, activation: 'tanh' },
      { size: 8, activation: 'tanh' },
      { size: 2, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam',
    learningRate: 0.05
  });

  for (let epoch = 0; epoch < 160; epoch++) {
    circleNet.trainBatch(circleData.X, circleData.y);
  }

  const circlePreds = circleNet.predict(circleData.X);
  let circleCorrect = 0;
  for (let i = 0; i < circleData.X.length; i++) {
    const predClass = circlePreds[i].indexOf(Math.max(...circlePreds[i]));
    const trueClass = circleData.y[i].indexOf(Math.max(...circleData.y[i]));
    if (predClass === trueClass) circleCorrect++;
  }
  const circleAcc = circleCorrect / circleData.X.length;
  console.log(`  -> Circles Accuracy: ${(circleAcc * 100).toFixed(1)}% (Target: >= 85%)`);

  if (circleAcc >= 0.85) {
    console.log('  [PASS] Concentric Circles Convergence Verified.\n');
    testsPassed++;
  } else {
    console.error('  [FAIL] Concentric Circles Convergence below threshold!\n');
  }

  // 3. Iris Multiclass Test
  console.log('[Test 3/3] Testing Fisher\'s Iris 3-Class Multiclass Convergence...');
  const irisData = DatasetManager.getIrisDataset();
  const irisNet = new NeuralNetwork({
    architecture: [
      { size: 4 },
      { size: 8, activation: 'relu' },
      { size: 3, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam',
    learningRate: 0.05
  });

  for (let epoch = 0; epoch < 150; epoch++) {
    irisNet.trainBatch(irisData.X, irisData.y);
  }

  const irisPreds = irisNet.predict(irisData.X);
  let irisCorrect = 0;
  for (let i = 0; i < irisData.X.length; i++) {
    const predClass = irisPreds[i].indexOf(Math.max(...irisPreds[i]));
    const trueClass = irisData.y[i].indexOf(Math.max(...irisData.y[i]));
    if (predClass === trueClass) irisCorrect++;
  }
  const irisAcc = irisCorrect / irisData.X.length;
  console.log(`  -> Iris Accuracy: ${(irisAcc * 100).toFixed(1)}% (Target: >= 85%)`);

  if (irisAcc >= 0.85) {
    console.log('  [PASS] Iris Multiclass Convergence Verified.\n');
    testsPassed++;
  } else {
    console.error('  [FAIL] Iris Multiclass Convergence below threshold!\n');
  }

  // 4. 8x8 Optical Digits Image Classification Test
  console.log('[Test 4/4] Testing 8x8 Optical Digits Image Classification (64 inputs -> 4 classes)...');
  const digitsData = DatasetManager.generateDigits(100, 0.05);
  const digitsNet = new NeuralNetwork({
    architecture: [
      { size: 64 },
      { size: 16, activation: 'relu' },
      { size: 4, activation: 'softmax' }
    ],
    loss: 'categorical_crossentropy',
    optimizer: 'adam',
    learningRate: 0.05
  });

  for (let epoch = 0; epoch < 100; epoch++) {
    digitsNet.trainBatch(digitsData.X, digitsData.y);
  }

  const digitsPreds = digitsNet.predict(digitsData.X);
  let digitsCorrect = 0;
  for (let i = 0; i < digitsData.X.length; i++) {
    const predClass = digitsPreds[i].indexOf(Math.max(...digitsPreds[i]));
    const trueClass = digitsData.y[i].indexOf(Math.max(...digitsData.y[i]));
    if (predClass === trueClass) digitsCorrect++;
  }
  const digitsAcc = digitsCorrect / digitsData.X.length;
  console.log(`  -> Digits Accuracy: ${(digitsAcc * 100).toFixed(1)}% (Target: >= 90%)`);

  if (digitsAcc >= 0.90) {
    console.log('  [PASS] 8x8 Optical Digits Image Convergence Verified.\n');
    testsPassed++;
  } else {
    console.error('  [FAIL] 8x8 Optical Digits Convergence below threshold!\n');
  }

  console.log(`====================================================`);
  console.log(`   TEST SUMMARY: ${testsPassed}/${totalTests} PASSED`);
  console.log(`====================================================\n`);

  if (testsPassed !== totalTests) {
    process.exit(1);
  }
}

runConvergenceTests();
