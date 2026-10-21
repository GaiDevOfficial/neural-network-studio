/**
 * PHASE 3 - TEST 1: Convergence on the REAL binary MNIST corpus (0 vs 1)
 *
 * Requirement under test: the engine must reach > 95% classification accuracy on a
 * binary 0-vs-1 handwritten digit dataset within 10 epochs.
 *
 * The data is the published MNIST corpus (verified against its MD5 digests, then against
 * the SHA-256 digests recorded in meta.json), decoded through the production loader and
 * evaluated on the ORIGINAL held-out test partition - digits the model never sees while
 * training. Nothing in this file is simulated: every number printed is a measured value.
 *
 * File: tests/test_mnist_convergence.js
 */

import { NeuralNetwork } from '../src/engine/neural_network.js';
import { loadMnistForTests, trainEpochs, formatPercent, banner } from './mnist_test_utils.js';
import { MNISTLoader } from '../src/engine/mnist_loader.js';

const EPOCH_BUDGET = 10;
const ACCURACY_TARGET = 0.95;

// The exact configuration the studio applies when the MNIST dataset is selected:
// ArchitectureBuilder defaults for a 784-input / 2-class problem plus the app's default
// optimizer (Adam), learning rate and batch size.
const ARCHITECTURE = [
  { size: 784, activation: 'linear' },
  { size: 32, activation: 'relu' },
  { size: 32, activation: 'relu' },
  { size: 2, activation: 'sigmoid' }
];
const OPTIMIZER = 'adam';
const LEARNING_RATE = 0.03;
const BATCH_SIZE = 32;

async function main() {
  banner('MNIST CONVERGENCE VERIFICATION (real corpus, 0 vs 1)');

  const dataset = await loadMnistForTests();
  const { meta } = dataset;

  console.log('Dataset provenance');
  console.log(`  corpus ............ ${meta.provenance}`);
  console.log(`  mirror ............ ${meta.sourceMirror}`);
  console.log(`  selection ......... ${meta.subsetPolicy}`);
  console.log(`  train images ...... ${dataset.trainX.length} (per class ${JSON.stringify(meta.train.perClass)})`);
  console.log(`  held-out images ... ${dataset.testX.length} (per class ${JSON.stringify(meta.test.perClass)})`);
  console.log(`  tensor size ....... ${dataset.numFeatures} floats per image, values in [0, 1]`);
  console.log(`  sha256 (trainX) ... ${meta.files['trainX.bin'].sha256.slice(0, 32)}...  (verified)`);

  console.log('\nFirst training image, decoded from the shipped bytes:');
  for (const line of MNISTLoader.toAscii(dataset.trainX[0]).split('\n')) {
    console.log(`  |${line}|`);
  }
  console.log(`  label: class ${dataset.trainY[0].indexOf(1)}`);

  console.log('\nBuilding the network');
  const network = new NeuralNetwork({
    architecture: ARCHITECTURE,
    loss: 'binary_crossentropy',
    optimizer: OPTIMIZER,
    learningRate: LEARNING_RATE,
    weightDecay: 0.0001
  });
  console.log(`  architecture: ${ARCHITECTURE.map((l) => l.size).join(' -> ')}`);
  console.log(`  parameters: ${countParameters(network)}`);
  console.log(`  optimizer: ${OPTIMIZER}, lr=${LEARNING_RATE}, batch=${BATCH_SIZE}`);

  console.log(`\nTraining (max ${EPOCH_BUDGET} epochs)`);
  console.log('  epoch | train loss | train acc | test loss | test acc');
  console.log('  ------+------------+-----------+-----------+----------');

  let achievedEpoch = null;
  let finalTestAccuracy = 0;
  let finalTestLoss = 0;

  trainEpochs(network, dataset.trainX, dataset.trainY, {
    epochs: EPOCH_BUDGET,
    batchSize: BATCH_SIZE,
    onEpoch: (metrics) => {
      const evaluation = network.evaluate(dataset.testX, dataset.testY);
      finalTestAccuracy = evaluation.accuracy;
      finalTestLoss = evaluation.loss;

      console.log(
        `  ${String(metrics.epoch).padStart(5)} | ` +
        `${metrics.trainLoss.toFixed(6).padStart(10)} | ` +
        `${formatPercent(metrics.trainAccuracy).padStart(9)} | ` +
        `${evaluation.loss.toFixed(6).padStart(9)} | ` +
        `${formatPercent(evaluation.accuracy).padStart(8)}`
      );

      if (achievedEpoch === null && evaluation.accuracy > ACCURACY_TARGET) {
        achievedEpoch = metrics.epoch;
      }
    }
  });

  // Final honest report
  const finalEvaluation = network.evaluate(dataset.testX, dataset.testY);
  const confusion = buildConfusion(finalEvaluation, dataset.testY);

  console.log('\nFinal evaluation on the held-out real test partition');
  console.log(`  images evaluated ... ${finalEvaluation.count}`);
  console.log(`  mean loss .......... ${finalEvaluation.loss.toFixed(6)}`);
  console.log(`  accuracy ........... ${formatPercent(finalEvaluation.accuracy)}`);
  console.log(`  misclassified ...... ${finalEvaluation.misclassified.length}`);
  console.log('  confusion matrix   [predicted 0, predicted 1]');
  console.log(`    true 0 ........... [${confusion.true0.predicted0}, ${confusion.true0.predicted1}]`);
  console.log(`    true 1 ........... [${confusion.true1.predicted0}, ${confusion.true1.predicted1}]`);

  const worst = finalEvaluation.misclassified.slice().sort((a, b) => b.loss - a.loss)[0];
  if (worst) {
    console.log(`\n  Worst mistake: test image #${worst.index}, predicted ${worst.predictedClass}, ` +
                `truth ${worst.trueClass}, loss ${worst.loss.toFixed(5)}`);
    for (const line of MNISTLoader.toAscii(dataset.testX[worst.index]).split('\n')) {
      console.log(`    |${line}|`);
    }
  }

  // ---------------- Assertions ----------------
  const failures = [];

  if (achievedEpoch === null) {
    failures.push(
      `Accuracy never exceeded ${formatPercent(ACCURACY_TARGET)} within ${EPOCH_BUDGET} epochs ` +
      `(best ${formatPercent(finalEvaluation.accuracy)})`
    );
  } else if (achievedEpoch > EPOCH_BUDGET) {
    failures.push(`Accuracy target reached only at epoch ${achievedEpoch}`);
  }

  if (!(finalEvaluation.accuracy > ACCURACY_TARGET)) {
    failures.push(`Final accuracy ${formatPercent(finalEvaluation.accuracy)} <= ${formatPercent(ACCURACY_TARGET)}`);
  }

  if (!(finalEvaluation.loss < 0.5) || !Number.isFinite(finalEvaluation.loss)) {
    failures.push(`Final loss ${finalEvaluation.loss} is not a healthy finite value`);
  }

  const nonFinite = countNonFinite(network);
  if (nonFinite > 0) {
    failures.push(`${nonFinite} non-finite parameters found after training`);
  }

  banner(failures.length === 0 ? 'CONVERGENCE TEST PASSED' : 'CONVERGENCE TEST FAILED');
  if (failures.length === 0) {
    console.log(`  target .......... > ${formatPercent(ACCURACY_TARGET)} within ${EPOCH_BUDGET} epochs`);
    console.log(`  achieved ........ ${formatPercent(finalTestAccuracy)} at epoch ${achievedEpoch}`);
    console.log('  [PASS] Real MNIST 0-vs-1 convergence verified\n');
    return 0;
  }

  for (const failure of failures) console.error(`  [FAIL] ${failure}`);
  console.error('');
  return 1;
}

function countParameters(network) {
  return network.layers.reduce((acc, layer) => acc + layer.inputSize * layer.outputSize + layer.outputSize, 0);
}

function countNonFinite(network) {
  let count = 0;
  for (const layer of network.layers) {
    for (let i = 0; i < layer.outputSize; i++) {
      if (!Number.isFinite(layer.biases[i])) count++;
      const row = layer.weights[i];
      for (let j = 0; j < layer.inputSize; j++) {
        if (!Number.isFinite(row[j])) count++;
      }
    }
  }
  return count;
}

function buildConfusion(evaluation, testY) {
  const confusion = {
    true0: { predicted0: 0, predicted1: 0 },
    true1: { predicted0: 0, predicted1: 0 }
  };
  // Reconstruct counts from aggregate accuracy + misclassification list.
  const total = testY.length;
  let true0 = 0;
  let true1 = 0;
  for (const label of testY) {
    if (label.indexOf(1) === 0) true0++;
    else true1++;
  }
  let true0Wrong = 0;
  let true1Wrong = 0;
  for (const miss of evaluation.misclassified) {
    if (miss.trueClass === 0) true0Wrong++;
    else true1Wrong++;
  }
  confusion.true0.predicted0 = true0 - true0Wrong;
  confusion.true0.predicted1 = true0Wrong;
  confusion.true1.predicted0 = true1Wrong;
  confusion.true1.predicted1 = true1 - true1Wrong;
  confusion.total = total;
  return confusion;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error('\n[FAIL] Convergence test crashed:', err);
    process.exit(1);
  });
