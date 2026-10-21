/**
 * Core Multi-Layer Neural Network Engine with Step-by-Step Hooks
 * Neural Network Studio Platform
 */

import { getActivation, Activations } from './activations.js';
import { getLossFunction, LossFunctions } from './loss_functions.js';
import { createOptimizer } from './optimizers.js';

export class Layer {
  constructor(inputSize, outputSize, activationType = 'relu', layerIndex = 0) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activationType = activationType;
    this.activation = getActivation(activationType);
    this.layerIndex = layerIndex;

    // Weights 2D array [outputSize x inputSize]
    // Biases 1D array [outputSize]
    this.weights = [];
    this.biases = new Float64Array(outputSize);

    // Gradients buffers
    this.weightGradients = [];
    this.biasGradients = new Float64Array(outputSize);

    this.initializeWeights();
  }

  /**
   * Xavier (Glorot) or He (Kaiming) initialization depending on activation
   */
  initializeWeights() {
    this.weights = [];
    this.weightGradients = [];

    const isReluVariant = this.activationType === 'relu' || this.activationType === 'leaky_relu';
    const limit = isReluVariant
      ? Math.sqrt(2.0 / this.inputSize) // He Normal scale
      : Math.sqrt(6.0 / (this.inputSize + this.outputSize)); // Glorot Uniform limit

    for (let i = 0; i < this.outputSize; i++) {
      const row = new Float64Array(this.inputSize);
      const gradRow = new Float64Array(this.inputSize);
      for (let j = 0; j < this.inputSize; j++) {
        if (isReluVariant) {
          // Box-Muller Gaussian
          const u1 = Math.max(1e-10, Math.random());
          const u2 = Math.random();
          const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          row[j] = z * limit;
        } else {
          row[j] = (Math.random() * 2 - 1) * limit;
        }
      }
      this.weights.push(row);
      this.weightGradients.push(gradRow);
      this.biases[i] = 0.0;
    }
  }

  /**
   * Reset gradient accumulators
   */
  zeroGradients() {
    for (let i = 0; i < this.outputSize; i++) {
      this.weightGradients[i].fill(0.0);
    }
    this.biasGradients.fill(0.0);
  }
}

export class NeuralNetwork {
  /**
   * Guard rails. A learning rate above MAX_LEARNING_RATE destabilizes even a convex
   * problem, so it is clamped rather than trusted (see setLearningRate).
   */
  static MAX_LEARNING_RATE = 0.5;
  static MIN_LEARNING_RATE = 1e-6;
  /** Per-element ceiling for a single parameter after an update step. */
  static PARAMETER_CEILING = 1e4;
  /**
   * A single-sample loss at or below this value counts as numerically solved (the BCE
   * clip floor is 1e-12, so the loss surface is flat here). Online steps stop early only
   * when the sample is BOTH numerically solved AND classified correctly - stopping on the
   * loss alone can freeze a sample whose two outputs are both saturated.
   */
  static SATURATED_SAMPLE_LOSS = 1e-6;
  /**
   * Conservative fine-tuning rate for single-sample human corrections. Batch training
   * keeps its own (higher) rate; a correction must never inherit the full training rate,
   * because one sample's gradient applied per-coordinate-normalized across the whole
   * network measurably destroys generalisation (verified: 6 corrections at the training
   * rate collapsed 100% -> 52% held-out accuracy).
   */
  static DEFAULT_ONLINE_LEARNING_RATE = 0.005;
  /**
   * Trust-region bound for online updates: the largest weight-gradient coordinate is
   * scaled to 1.0, so one update moves any weight by at most the online learning rate and
   * the update stays proportional to this sample's actual gradient (unlike Adam's
   * per-coordinate normalisation, coordinates this sample does not activate stay put).
   */
  static ONLINE_GRADIENT_LIMIT = 1;
  /**
   * Aggregate trust region for one single-sample update. Coordinate clipping alone still
   * lets tens of thousands of small coordinates move together, which can rewrite a fragile
   * model after a handful of corrections. This caps the L2 norm of the whole parameter
   * delta while preserving its direction.
   */
  static ONLINE_UPDATE_NORM_LIMIT = 0.05;

  constructor(config = {}) {
    this.layers = [];
    this.lossFunction = getLossFunction(config.loss || 'mse');
    this.lossType = config.loss || 'mse';
    this.optimizer = createOptimizer(config.optimizer || 'adam', config.learningRate || 0.01, config.weightDecay || 0.0001);
    this.learningRate = config.learningRate || 0.01;
    this.weightDecay = config.weightDecay || 0.0001;

    // Hook listeners
    this.hooks = {
      onForwardStep: null,
      onBackwardStep: null,
      onEpochEnd: null
    };

    if (config.architecture && config.architecture.length > 1) {
      this.buildFromArchitecture(config.architecture);
    }
  }

  /**
   * Build network from array of layer specifications:
   * e.g. [
   *   { size: 2 },
   *   { size: 8, activation: 'relu' },
   *   { size: 8, activation: 'relu' },
   *   { size: 2, activation: 'softmax' }
   * ]
   */
  buildFromArchitecture(arch) {
    this.layers = [];
    this.architecture = arch;

    for (let i = 0; i < arch.length - 1; i++) {
      const inSize = arch[i].size;
      const outSize = arch[i + 1].size;
      const act = arch[i + 1].activation || (i === arch.length - 2 ? 'linear' : 'relu');
      this.layers.push(new Layer(inSize, outSize, act, i));
    }

    this.optimizer.reset();
  }

  /**
   * Forward pass for a single input vector
   * Returns { output, activations: [[...], [...]], preActivations: [[...], [...]] }
   */
  forwardSample(inputVec) {
    const activations = [[...inputVec]];
    const preActivations = [];

    let current = inputVec;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const z = new Float64Array(layer.outputSize);

      for (let i = 0; i < layer.outputSize; i++) {
        let sum = layer.biases[i];
        const wRow = layer.weights[i];
        for (let j = 0; j < layer.inputSize; j++) {
          sum += wRow[j] * current[j];
        }
        z[i] = sum;
      }
      preActivations.push(z);

      // Apply activation function
      let a;
      if (layer.activationType === 'softmax') {
        a = Activations.softmax.vectorForward(z);
      } else {
        a = layer.activation.vectorForward(z);
      }

      activations.push(a);
      current = a;
    }

    if (this.hooks.onForwardStep) {
      this.hooks.onForwardStep({
        input: inputVec,
        activations,
        preActivations,
        output: current
      });
    }

    return {
      output: current,
      activations,
      preActivations
    };
  }

  /**
   * Predict on a batch of inputs
   */
  predict(batchX) {
    return batchX.map(x => this.forwardSample(x).output);
  }

  /**
   * Compute the real backpropagation deltas (dL/dz) for one forward pass.
   *
   * This is the single source of truth for every gradient computation in the engine:
   * batch training, single-sample online updates and the visualizer probe all call it,
   * so what the synapse renderer draws is literally what the optimizer applied.
   *
   * @param {ArrayLike<number>} output    network output for this sample
   * @param {Array} preActivations        per-layer z vectors from forwardSample()
   * @param {ArrayLike<number>} y         target vector
   * @returns {Float64Array[]} deltas indexed by layer
   */
  computeDeltas(output, preActivations, y) {
    const numLayers = this.layers.length;
    const deltas = new Array(numLayers);
    if (numLayers === 0) return deltas;

    const outLayer = this.layers[numLayers - 1];
    const outDelta = new Float64Array(outLayer.outputSize);
    const lastZ = preActivations[numLayers - 1];

    if (outLayer.activationType === 'softmax' && this.lossType === 'categorical_crossentropy') {
      // Canonical Softmax + CCE simplification: dL/dz = y_pred - y_true
      for (let i = 0; i < outLayer.outputSize; i++) {
        outDelta[i] = output[i] - y[i];
      }
    } else if (outLayer.activationType === 'sigmoid' && this.lossType === 'binary_crossentropy') {
      // Canonical Sigmoid + BCE simplification: dL/dz = y_pred - y_true
      for (let i = 0; i < outLayer.outputSize; i++) {
        outDelta[i] = output[i] - y[i];
      }
    } else {
      // General chain rule: dL/da * da/dz
      const lossGrad = this.lossFunction.gradient(output, y);
      const actDeriv = outLayer.activation.vectorDerivative(lastZ);
      for (let i = 0; i < outLayer.outputSize; i++) {
        outDelta[i] = lossGrad[i] * actDeriv[i];
      }
    }
    deltas[numLayers - 1] = outDelta;

    // Backpropagate through the hidden layers
    for (let l = numLayers - 2; l >= 0; l--) {
      const layer = this.layers[l];
      const nextLayer = this.layers[l + 1];
      const nextDelta = deltas[l + 1];
      const actDeriv = layer.activation.vectorDerivative(preActivations[l]);
      const layerDelta = new Float64Array(layer.outputSize);

      for (let j = 0; j < layer.outputSize; j++) {
        let sum = 0;
        for (let k = 0; k < nextLayer.outputSize; k++) {
          sum += nextLayer.weights[k][j] * nextDelta[k];
        }
        // Gradient clipping for exploding gradient prevention
        const clippedSum = Math.max(-100, Math.min(100, sum));
        layerDelta[j] = clippedSum * actDeriv[j];
      }
      deltas[l] = layerDelta;
    }

    return deltas;
  }

  /**
   * Real per-sample gradient probe used by the synaptic visualizer.
   * Returns the forward activations, the backward deltas, AND the actual
   * dL/dW matrix for every layer (delta ⊗ previous activation), so the renderer can
   * scale pulse intensity by true gradient magnitude rather than a decoration.
   */
  probeSample(x, y, options = {}) {
    if (!this.layers || this.layers.length === 0) return null;

    const forward = this.forwardSample(x);
    const activations = forward.activations;
    const deltas = y ? this.computeDeltas(forward.output, forward.preActivations, y) : null;

    let gradientMagnitudes = null;
    let maxAbsGradient = 0;

    if (deltas) {
      gradientMagnitudes = new Array(this.layers.length);
      for (let l = 0; l < this.layers.length; l++) {
        const layer = this.layers[l];
        const delta = deltas[l];
        const prevActivation = activations[l];
        const rows = new Array(layer.outputSize);

        for (let i = 0; i < layer.outputSize; i++) {
          const row = new Float64Array(layer.inputSize);
          const d = delta[i];
          for (let j = 0; j < layer.inputSize; j++) {
            const g = d * prevActivation[j];
            row[j] = g;
            const abs = Math.abs(g);
            if (abs > maxAbsGradient) maxAbsGradient = abs;
          }
          rows[i] = row;
        }
        gradientMagnitudes[l] = rows;
      }
    }

    const loss = y ? this.lossFunction.calculate(forward.output, y) : null;

    return {
      output: Array.from(forward.output),
      activations: activations.map(a => Array.from(a)),
      deltas: deltas ? deltas.map(d => Array.from(d)) : null,
      gradientMagnitudes,
      maxAbsGradient,
      loss,
      predictedClass: Array.from(forward.output).indexOf(Math.max(...forward.output)),
      targetClass: y ? y.indexOf(Math.max(...y)) : null,
      ...(options.extra || {})
    };
  }

  /**
   * Evaluate the network over a labelled set with the real loss function.
   * Returns mean loss, accuracy and the individual per-sample losses so callers can
   * reason about the worst cases (used by active learning and the test suite).
   */
  evaluate(batchX, batchY) {
    if (!batchX || batchX.length === 0) {
      return { count: 0, loss: 0, accuracy: 0, perSampleLoss: [], misclassified: [] };
    }

    const perSampleLoss = new Array(batchX.length);
    const misclassified = [];
    let totalLoss = 0;
    let correct = 0;

    for (let i = 0; i < batchX.length; i++) {
      const output = this.forwardSample(batchX[i]).output;
      const target = batchY[i];
      const loss = this.lossFunction.calculate(output, target);
      perSampleLoss[i] = loss;
      totalLoss += loss;

      const prediction = Array.from(output);
      const predictedClass = prediction.indexOf(Math.max(...prediction));

      if (target.length > 1) {
        const trueClass = Array.from(target).indexOf(Math.max(...target));
        if (predictedClass === trueClass) correct++;
        else misclassified.push({ index: i, predictedClass, trueClass, loss });
      } else if (Math.abs(output[0] - target[0]) < 0.25) {
        correct++;
      }
    }

    return {
      count: batchX.length,
      loss: totalLoss / batchX.length,
      accuracy: correct / batchX.length,
      perSampleLoss,
      misclassified
    };
  }

  /**
   * Learning rate guard rail: NaN, negatives, zero and absurd magnitudes are clamped to
   * a range in which gradient descent is still a contraction instead of a detonation.
   * Returns the value that was actually installed.
   */
  setLearningRate(lr) {
    let safe = Number(lr);
    if (!Number.isFinite(safe) || safe <= 0) safe = NeuralNetwork.MIN_LEARNING_RATE;
    if (safe > NeuralNetwork.MAX_LEARNING_RATE) safe = NeuralNetwork.MAX_LEARNING_RATE;
    this.learningRate = safe;
    if (this.optimizer) this.optimizer.lr = safe;
    return safe;
  }

  /**
   * Replace every non-finite parameter (NaN/Inf) with 0.0.
   * Called at epoch boundaries by the worker; a non-zero return value is surfaced to the
   * UI as a real diagnostic, never hidden.
   */
  sanitizeWeights() {
    let repaired = 0;
    for (const layer of this.layers) {
      for (let i = 0; i < layer.outputSize; i++) {
        const row = layer.weights[i];
        for (let j = 0; j < layer.inputSize; j++) {
          if (!Number.isFinite(row[j])) {
            row[j] = 0.0;
            repaired++;
          }
        }
        if (!Number.isFinite(layer.biases[i])) {
          layer.biases[i] = 0.0;
          repaired++;
        }
      }
    }
    this.nonFiniteRepairs = (this.nonFiniteRepairs || 0) + repaired;
    return { repaired, total: this.nonFiniteRepairs || 0 };
  }

  /** Largest absolute parameter magnitude currently in the model (health metric). */
  maxAbsWeight() {
    let max = 0;
    for (const layer of this.layers) {
      for (let i = 0; i < layer.outputSize; i++) {
        const row = layer.weights[i];
        for (let j = 0; j < layer.inputSize; j++) {
          const abs = Math.abs(row[j]);
          if (abs > max) max = abs;
        }
      }
    }
    return max;
  }

  /** Add a labelled sample to the live training set (online / active learning). */
  enqueueSample(x, y, origin = 'user') {
    if (!this.activeSamples) this.activeSamples = [];
    if (!x || x.length === 0) return { appended: false, reason: 'EMPTY_SAMPLE' };
    const target = Array.from(y);
    const vector = x instanceof Float64Array ? x : Float64Array.from(x);
    this.activeSamples.push({ x: vector, y: target, origin, addedAt: Date.now() });
    return { appended: true, activeSetSize: this.activeSamples.length, origin };
  }

  /**
   * Train one full batch (real forward pass, real loss, real backprop, real optimizer step)
   */
  trainBatch(batchX, batchY) {
    const batchSize = batchX ? batchX.length : 0;
    if (batchSize === 0 || !this.layers || this.layers.length === 0) {
      return { loss: 0, accuracy: 0, batchSize: 0, nonFiniteSamples: 0 };
    }

    // Learning rate guard rail applied on every step (cheap, idempotent).
    if (this.optimizer) this.optimizer.lr = this.learningRate;

    // Reset gradients
    for (const layer of this.layers) {
      layer.zeroGradients();
    }

    let totalLoss = 0;
    let correctCount = 0;

    // Step-by-step activations and deltas history for visualizer hook
    let lastSampleActivations = null;
    let lastSampleDeltas = [];

    for (let b = 0; b < batchSize; b++) {
      const x = batchX[b];
      const y = batchY[b];

      // Forward Pass
      const { output, activations, preActivations } = this.forwardSample(x);
      lastSampleActivations = activations;

      // Loss calculation
      const sampleLoss = this.lossFunction.calculate(output, y);
      totalLoss += sampleLoss;

      // Accuracy check (Classification vs Regression)
      if (y.length > 1) {
        // Multi-class or one-hot
        const predClass = output.indexOf(Math.max(...output));
        const trueClass = y.indexOf(Math.max(...y));
        if (predClass === trueClass) correctCount++;
      } else {
        // Binary or regression threshold
        if (Math.abs(output[0] - y[0]) < 0.25) correctCount++;
      }

      // Backward Pass: real dL/dz for every layer (shared with online updates + probe)
      const deltas = this.computeDeltas(output, preActivations, y);
      lastSampleDeltas = deltas;

      // Accumulate gradients across batch
      for (let l = 0; l < this.layers.length; l++) {
        const layer = this.layers[l];
        const delta = deltas[l];
        const prevActivation = activations[l]; // activation from previous layer

        for (let i = 0; i < layer.outputSize; i++) {
          layer.biasGradients[i] += delta[i] / batchSize;
          const gradRow = layer.weightGradients[i];
          const d = delta[i] / batchSize;
          for (let j = 0; j < layer.inputSize; j++) {
            gradRow[j] += d * prevActivation[j];
          }
        }
      }
    }

    // Gradient Descent Step via Optimizer
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const wKey = `w_${l}`;
      const bKey = `b_${l}`;

      layer.weights = this.optimizer.update(layer.weights, layer.weightGradients, wKey, true);
      layer.biases = this.optimizer.update(layer.biases, layer.biasGradients, bKey, false);
    }

    // Overflow detection: inspect the freshly written parameters on a strided scan so a
    // detonation is caught on the very step that produced it (full sweep per epoch).
    let nonFiniteSamples = 0;
    let maxAbs = 0;
    for (const layer of this.layers) {
      for (let i = 0; i < layer.outputSize; i += 3) {
        if (!Number.isFinite(layer.biases[i])) nonFiniteSamples++;
        const abs = Math.abs(layer.biases[i]);
        if (abs > maxAbs) maxAbs = abs;
        const row = layer.weights[i];
        for (let j = 0; j < layer.inputSize; j += 7) {
          const w = row[j];
          if (!Number.isFinite(w)) nonFiniteSamples++;
          else {
            const wa = Math.abs(w);
            if (wa > maxAbs) maxAbs = wa;
          }
        }
      }
    }
    this.lastOverflowSamples = nonFiniteSamples;
    this.lastMaxAbsWeight = maxAbs;

    // Backward hook callback for visualizer
    if (this.hooks.onBackwardStep) {
      this.hooks.onBackwardStep({
        activations: lastSampleActivations,
        deltas: lastSampleDeltas,
        weights: this.getWeightsSnapshot(),
        loss: totalLoss / batchSize,
        accuracy: correctCount / batchSize
      });
    }

    return {
      loss: totalLoss / batchSize,
      accuracy: correctCount / batchSize,
      batchSize,
      nonFiniteSamples,
      maxAbsWeight: maxAbs
    };
  }

  /**
   * One bounded single-sample gradient step (the engine behind online/active-learning
   * corrections).
   *
   * Real backprop: forward pass, computeDeltas, gradient = delta (x) activation for every
   * layer. The trust-region scale bounds the largest coordinate to `lr`, and - unlike the
   * batch Adam optimizer - leaves coordinates this sample does not activate untouched, so
   * a correction repairs the sample without globally rewriting the model.
   */
  applyOnlineUpdate(x, y, lr) {
    const forward = this.forwardSample(x);
    const deltas = this.computeDeltas(forward.output, forward.preActivations, y);

    // Fill the per-layer gradient buffers (same math as trainBatch, batch size 1) and
    // find the largest coordinate so the step can be bounded.
    let maxGradient = 0;
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const delta = deltas[l];
      const prevActivation = forward.activations[l];

      layer.biasGradients.set(delta);
      for (let i = 0; i < layer.outputSize; i++) {
        const gradRow = layer.weightGradients[i];
        const d = delta[i];
        for (let j = 0; j < layer.inputSize; j++) {
          const g = d * prevActivation[j];
          gradRow[j] = g;
          const abs = Math.abs(g);
          if (abs > maxGradient) maxGradient = abs;
        }
      }
      for (let i = 0; i < layer.outputSize; i++) {
        const abs = Math.abs(delta[i]);
        if (abs > maxGradient) maxGradient = abs;
      }
    }

    const scale = maxGradient > NeuralNetwork.ONLINE_GRADIENT_LIMIT
      ? NeuralNetwork.ONLINE_GRADIENT_LIMIT / maxGradient
      : 1;

    // Measure the full proposed delta before applying it. The coordinate clip above keeps
    // every single weight bounded; this aggregate clip keeps the correction from moving
    // thousands of coordinates far enough to erase the model's general decision boundary.
    let updateNormSq = 0;
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      for (let i = 0; i < layer.outputSize; i++) {
        const biasDelta = lr * layer.biasGradients[i] * scale;
        updateNormSq += biasDelta * biasDelta;
        const gradRow = layer.weightGradients[i];
        for (let j = 0; j < layer.inputSize; j++) {
          const delta = lr * gradRow[j] * scale;
          updateNormSq += delta * delta;
        }
      }
    }
    const updateNorm = Math.sqrt(updateNormSq);
    const normScale = updateNorm > NeuralNetwork.ONLINE_UPDATE_NORM_LIMIT
      ? NeuralNetwork.ONLINE_UPDATE_NORM_LIMIT / updateNorm
      : 1;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      for (let i = 0; i < layer.outputSize; i++) {
        layer.biases[i] -= lr * layer.biasGradients[i] * scale * normScale;
        const row = layer.weights[i];
        const gradRow = layer.weightGradients[i];
        for (let j = 0; j < layer.inputSize; j++) {
          row[j] -= lr * gradRow[j] * scale * normScale;
        }
      }
    }

    // Same backward hook the batch trainer fires, so the visualizer receives real
    // activations / deltas / gradients for the correction it just watched.
    if (this.hooks.onBackwardStep) {
      this.hooks.onBackwardStep({
        activations: forward.activations,
        deltas,
        weights: this.getWeightsSnapshot(),
        loss: this.lossFunction.calculate(forward.output, y),
        accuracy: forward.output.indexOf(Math.max(...forward.output)) === Array.from(y).indexOf(Math.max(...y)) ? 1 : 0
      });
    }

    return { maxGradient, scale, updateNorm, normScale };
  }

  /**
   * Online Backpropagation Step on a single sample (Human-in-the-Loop Active Learning)
   *
   * Immediately adjusts weights to rectify a mistake on x with true label y, and reports
   * the REAL measured loss before and after the update plus a per-step trace so the UI
   * can show a genuine curve instead of an assertion.
   *
   * The step is bounded (see applyOnlineUpdate) and uses the conservative online learning
   * rate unless the caller explicitly overrides it; the loop stops early only when the
   * sample is numerically solved AND classified correctly.
   *
   * @param {ArrayLike<number>} x input tensor
   * @param {ArrayLike<number>} y corrected target vector
   * @param {number|null} customLr optional learning rate override for this update
   * @param {number} steps   maximum number of consecutive single-sample updates (>=1)
   * @returns {object} { initialLoss, postLoss, initialPred, postPred, lossReduced, lossDelta, steps, requestedSteps, trace, sanitized }
   */
  online_step(x, y, customLr = null, steps = 1) {
    const savedLr = this.learningRate;
    const requestedSteps = Number.isFinite(steps) ? Math.max(1, Math.min(50, Math.floor(steps))) : 1;

    // A human correction never inherits the batch training rate: without an explicit
    // override it uses the conservative online fine-tuning rate.
    let effectiveLr;
    if (customLr !== null && customLr !== undefined) {
      effectiveLr = this.setLearningRate(customLr);
    } else {
      effectiveLr = NeuralNetwork.DEFAULT_ONLINE_LEARNING_RATE;
    }

    // Initial forward pass before any update
    const initForward = this.forwardSample(x);
    const initialLoss = this.lossFunction.calculate(initForward.output, y);
    const initialPred = Array.from(initForward.output);

    const trace = [{ step: 0, loss: initialLoss, predictedClass: initialPred.indexOf(Math.max(...initialPred)) }];

    const targetClass = Array.from(y).indexOf(Math.max(...y));
    let sanitized = 0;
    for (let s = 0; s < requestedSteps; s++) {
      this.applyOnlineUpdate(x, y, effectiveLr);
      const stepForward = this.forwardSample(x);
      const stepLoss = this.lossFunction.calculate(stepForward.output, y);
      const stepPred = Array.from(stepForward.output);
      const stepClass = stepPred.indexOf(Math.max(...stepPred));
      trace.push({
        step: s + 1,
        loss: stepLoss,
        predictedClass: stepClass,
        confidence: Math.max(...stepPred)
      });
      const repair = this.sanitizeWeights();
      sanitized += repair.repaired;

      // Stop only when this sample is genuinely solved - solved loss AND correct class.
      if (stepLoss <= NeuralNetwork.SATURATED_SAMPLE_LOSS && stepClass === targetClass) break;
    }
    const performedSteps = trace.length - 1;

    // Forward pass immediately after the update train
    const postForward = this.forwardSample(x);
    const postLoss = this.lossFunction.calculate(postForward.output, y);
    const postPred = Array.from(postForward.output);

    if (customLr !== null && customLr !== undefined) {
      this.setLearningRate(savedLr);
    }

    return {
      initialLoss,
      postLoss,
      initialPred,
      postPred,
      lossReduced: postLoss < initialLoss,
      lossDelta: initialLoss - postLoss,
      relativeReduction: initialLoss > 0 ? (initialLoss - postLoss) / initialLoss : 0,
      steps: performedSteps,
      requestedSteps,
      trace,
      sanitizedParameters: sanitized,
      learningRateUsed: effectiveLr
    };
  }

  /**
   * Return a lightweight immutable snapshot of weights, biases, and architecture for visualizer
   */
  getWeightsSnapshot() {
    return this.layers.map((layer, idx) => ({
      layerIndex: idx,
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      activation: layer.activationType,
      weights: layer.weights.map(row => Array.from(row)),
      biases: Array.from(layer.biases)
    }));
  }

  /**
   * Export model parameters to serializable JSON object
   */
  toJSON() {
    return {
      version: '1.0',
      timestamp: new Date().toISOString(),
      architecture: this.architecture,
      loss: this.lossType,
      learningRate: this.learningRate,
      weightDecay: this.weightDecay,
      layers: this.layers.map(l => ({
        inputSize: l.inputSize,
        outputSize: l.outputSize,
        activation: l.activationType,
        weights: l.weights.map(r => Array.from(r)),
        biases: Array.from(l.biases)
      }))
    };
  }

  /**
   * Load model from JSON representation.
   *
   * The file comes from the user's disk, so every structural claim is validated BEFORE any
   * weight is applied: a malformed file fails loudly (the worker reports the real reason and
   * keeps the current network) instead of silently leaving random weights in place or
   * crashing mid-training with a shape mismatch.
   */
  static fromJSON(json) {
    if (!json || !Array.isArray(json.architecture) || json.architecture.length < 2) {
      throw new Error('no valid architecture');
    }
    for (let l = 0; l < json.architecture.length; l++) {
      const size = json.architecture[l] && json.architecture[l].size;
      if (!Number.isInteger(size) || size < 1) {
        throw new Error(`architecture layer ${l} has an invalid size`);
      }
    }
    if (!Array.isArray(json.layers) || json.layers.length !== json.architecture.length - 1) {
      throw new Error(
        `file has ${Array.isArray(json.layers) ? json.layers.length : 0} weight layers but the architecture needs ${json.architecture.length - 1}`
      );
    }

    const net = new NeuralNetwork({
      architecture: json.architecture,
      loss: json.loss,
      optimizer: json.optimizer,
      learningRate: json.learningRate,
      weightDecay: json.weightDecay
    });

    for (let l = 0; l < net.layers.length; l++) {
      const layer = net.layers[l];
      const savedLayer = json.layers[l];
      const wellFormed = savedLayer &&
        Array.isArray(savedLayer.weights) &&
        Array.isArray(savedLayer.biases) &&
        savedLayer.weights.length === layer.outputSize &&
        savedLayer.biases.length === layer.outputSize &&
        savedLayer.weights.every((row) => Array.isArray(row) && row.length === layer.inputSize);

      if (!wellFormed) {
        throw new Error(`layer ${l} does not match the architecture (${layer.inputSize} -> ${layer.outputSize})`);
      }

      const weights = savedLayer.weights.map((row) => Float64Array.from(row));
      const biases = Float64Array.from(savedLayer.biases);
      for (const row of weights) {
        for (const value of row) {
          if (!Number.isFinite(value)) throw new Error(`layer ${l} contains non-finite weights`);
        }
      }
      for (const value of biases) {
        if (!Number.isFinite(value)) throw new Error(`layer ${l} contains non-finite biases`);
      }

      layer.weights = weights;
      layer.biases = biases;
    }
    return net;
  }
}
