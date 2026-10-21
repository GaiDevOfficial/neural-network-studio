/**
 * Matrix Neural Network Engine (Pure JavaScript Vectorized Core)
 * Supports Multi-Layer Perceptrons, Online Backpropagation, Adam/SGD Optimizers,
 * Loss Functions, and Real-Time Step Hooks for Visualizers.
 *
 * File: src/engine/matrix_nn.js
 */

export class Activations {
  static relu(x) {
    return Math.max(0, x);
  }

  static reluDeriv(x) {
    return x > 0 ? 1.0 : 0.0;
  }

  static leakyRelu(x, alpha = 0.01) {
    return x > 0 ? x : alpha * x;
  }

  static leakyReluDeriv(x, alpha = 0.01) {
    return x > 0 ? 1.0 : alpha;
  }

  static sigmoid(x) {
    const clamped = Math.max(-50, Math.min(50, x));
    return 1.0 / (1.0 + Math.exp(-clamped));
  }

  static sigmoidDeriv(x) {
    const s = Activations.sigmoid(x);
    return s * (1.0 - s);
  }

  static tanh(x) {
    const clamped = Math.max(-50, Math.min(50, x));
    return Math.tanh(clamped);
  }

  static tanhDeriv(x) {
    const t = Math.tanh(Math.max(-50, Math.min(50, x)));
    return 1.0 - t * t;
  }

  static linear(x) {
    return x;
  }

  static linearDeriv() {
    return 1.0;
  }

  static softmax(vector) {
    let max = -Infinity;
    for (let i = 0; i < vector.length; i++) {
      if (vector[i] > max) max = vector[i];
    }
    const exps = new Float64Array(vector.length);
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
      const e = Math.exp(Math.max(-50, Math.min(50, vector[i] - max)));
      exps[i] = e;
      sum += e;
    }
    const safeSum = sum < 1e-12 ? 1e-12 : sum;
    for (let i = 0; i < vector.length; i++) {
      exps[i] /= safeSum;
    }
    return exps;
  }
}

export function getActivation(name = 'relu') {
  const n = String(name).toLowerCase();
  switch (n) {
    case 'relu':
      return { fn: Activations.relu, deriv: Activations.reluDeriv, name: 'relu' };
    case 'leaky_relu':
      return { fn: Activations.leakyRelu, deriv: Activations.leakyReluDeriv, name: 'leaky_relu' };
    case 'sigmoid':
      return { fn: Activations.sigmoid, deriv: Activations.sigmoidDeriv, name: 'sigmoid' };
    case 'tanh':
      return { fn: Activations.tanh, deriv: Activations.tanhDeriv, name: 'tanh' };
    case 'softmax':
      return { fn: Activations.linear, deriv: Activations.linearDeriv, name: 'softmax' };
    default:
      return { fn: Activations.linear, deriv: Activations.linearDeriv, name: 'linear' };
  }
}

export class MatrixLayer {
  constructor(inputSize, outputSize, activationName = 'relu', layerIndex = 0) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activationName = activationName.toLowerCase();
    this.activation = getActivation(activationName);
    this.layerIndex = layerIndex;

    // Weights: [outputSize][inputSize]
    this.weights = [];
    this.weightGradients = [];
    this.biases = new Float64Array(outputSize);
    this.biasGradients = new Float64Array(outputSize);

    // Adam optimizer moments
    this.m_w = [];
    this.v_w = [];
    this.m_b = new Float64Array(outputSize);
    this.v_b = new Float64Array(outputSize);

    this.initWeights();
  }

  initWeights() {
    this.weights = [];
    this.weightGradients = [];
    this.m_w = [];
    this.v_w = [];

    const isRelu = this.activationName === 'relu' || this.activationName === 'leaky_relu';
    // He or Glorot standard initialization
    const limit = isRelu
      ? Math.sqrt(2.0 / this.inputSize)
      : Math.sqrt(6.0 / (this.inputSize + this.outputSize));

    for (let i = 0; i < this.outputSize; i++) {
      const row = new Float64Array(this.inputSize);
      const gradRow = new Float64Array(this.inputSize);
      const mwRow = new Float64Array(this.inputSize);
      const vwRow = new Float64Array(this.inputSize);

      for (let j = 0; j < this.inputSize; j++) {
        if (isRelu) {
          // Box-Muller Gaussian normal distribution
          const u1 = Math.max(1e-10, Math.random());
          const u2 = Math.random();
          const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          row[j] = z * limit;
        } else {
          row[j] = (Math.random() * 2.0 - 1.0) * limit;
        }
      }
      this.weights.push(row);
      this.weightGradients.push(gradRow);
      this.m_w.push(mwRow);
      this.v_w.push(vwRow);
      this.biases[i] = 0.0;
    }
  }

  zeroGradients() {
    for (let i = 0; i < this.outputSize; i++) {
      this.weightGradients[i].fill(0.0);
    }
    this.biasGradients.fill(0.0);
  }
}

export class MatrixNeuralNetwork {
  constructor(config = {}) {
    this.layers = [];
    this.lossType = (config.loss || 'binary_crossentropy').toLowerCase();
    this.optimizerType = (config.optimizer || 'adam').toLowerCase();
    this.learningRate = config.learningRate || 0.03;
    this.weightDecay = config.weightDecay || 0.0001;

    // Adam hyperparams
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;
    this.timeStep = 0;

    // Hooks
    this.hooks = {
      on_iteration: null, // hook: on_iteration(weights, activations, gradients)
      on_forward: null,
      on_backward: null
    };

    if (config.architecture && config.architecture.length > 1) {
      this.build(config.architecture);
    }
  }

  build(architecture) {
    this.layers = [];
    this.architecture = architecture;
    this.timeStep = 0;

    for (let i = 0; i < architecture.length - 1; i++) {
      const inSize = architecture[i].size;
      const outSize = architecture[i + 1].size;
      const act = architecture[i + 1].activation || (i === architecture.length - 2 ? 'sigmoid' : 'relu');
      this.layers.push(new MatrixLayer(inSize, outSize, act, i));
    }
  }

  /**
   * Forward pass for a single input vector
   * Returns { output, activations: [...], preActivations: [...] }
   */
  forwardSample(inputVec) {
    const activations = [new Float64Array(inputVec)];
    const preActivations = [];

    let current = activations[0];

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

      let a;
      if (layer.activationName === 'softmax') {
        a = Activations.softmax(z);
      } else {
        a = new Float64Array(layer.outputSize);
        for (let i = 0; i < layer.outputSize; i++) {
          a[i] = layer.activation.fn(z[i]);
        }
      }

      activations.push(a);
      current = a;
    }

    if (this.hooks.on_forward) {
      this.hooks.on_forward({ input: inputVec, output: current, activations, preActivations });
    }

    return { output: current, activations, preActivations };
  }

  /**
   * Predict on an array of input samples
   */
  predict(samples) {
    return samples.map(sample => Array.from(this.forwardSample(sample).output));
  }

  /**
   * Compute scalar loss given predicted vector and target vector
   */
  computeLoss(predicted, target) {
    if (this.lossType === 'mse') {
      let sum = 0;
      for (let i = 0; i < target.length; i++) {
        const diff = predicted[i] - target[i];
        sum += diff * diff;
      }
      return (0.5 * sum) / target.length;
    }

    if (this.lossType === 'binary_crossentropy' || this.lossType === 'bce') {
      let sum = 0;
      for (let i = 0; i < target.length; i++) {
        const y = target[i];
        const p = Math.max(1e-12, Math.min(1.0 - 1e-12, predicted[i]));
        sum += -(y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p));
      }
      return sum / target.length;
    }

    // Categorical Cross-Entropy
    let sum = 0;
    for (let i = 0; i < target.length; i++) {
      if (target[i] > 0) {
        const p = Math.max(1e-12, Math.min(1.0, predicted[i]));
        sum += -target[i] * Math.log(p);
      }
    }
    return sum;
  }

  /**
   * Train one mini-batch using Backpropagation and Adam/SGD
   */
  trainBatch(batchX, batchY) {
    const batchSize = batchX.length;
    if (batchSize === 0) return { loss: 0, accuracy: 0 };

    for (const layer of this.layers) {
      layer.zeroGradients();
    }

    let totalLoss = 0;
    let correctCount = 0;

    let lastActivations = null;
    let lastDeltas = null;

    for (let b = 0; b < batchSize; b++) {
      const x = batchX[b];
      const y = batchY[b];

      // 1. Forward Pass
      const { output, activations, preActivations } = this.forwardSample(x);
      lastActivations = activations;

      // 2. Loss & Accuracy
      const sampleLoss = this.computeLoss(output, y);
      totalLoss += sampleLoss;

      if (y.length > 1) {
        let maxPred = -Infinity, maxPredIdx = 0;
        let maxTrue = -Infinity, maxTrueIdx = 0;
        for (let k = 0; k < y.length; k++) {
          if (output[k] > maxPred) { maxPred = output[k]; maxPredIdx = k; }
          if (y[k] > maxTrue) { maxTrue = y[k]; maxTrueIdx = k; }
        }
        if (maxPredIdx === maxTrueIdx) correctCount++;
      } else {
        if (Math.abs(output[0] - y[0]) < 0.25) correctCount++;
      }

      // 3. Backward Pass (Deltas calculation)
      const numLayers = this.layers.length;
      const deltas = new Array(numLayers);

      // Output layer delta
      const outLayer = this.layers[numLayers - 1];
      const outDelta = new Float64Array(outLayer.outputSize);
      const lastZ = preActivations[numLayers - 1];

      if ((outLayer.activationName === 'softmax' && this.lossType === 'categorical_crossentropy') ||
          (outLayer.activationName === 'sigmoid' && (this.lossType === 'binary_crossentropy' || this.lossType === 'bce'))) {
        // Analytical simplification: dL/dz = y_pred - y_true
        for (let i = 0; i < outLayer.outputSize; i++) {
          outDelta[i] = output[i] - y[i];
        }
      } else {
        // General chain rule: dL/da * da/dz
        for (let i = 0; i < outLayer.outputSize; i++) {
          const dL_da = output[i] - y[i];
          const da_dz = outLayer.activation.deriv(lastZ[i]);
          outDelta[i] = dL_da * da_dz;
        }
      }
      deltas[numLayers - 1] = outDelta;

      // Propagate backward through hidden layers
      for (let l = numLayers - 2; l >= 0; l--) {
        const layer = this.layers[l];
        const nextLayer = this.layers[l + 1];
        const nextDelta = deltas[l + 1];
        const z = preActivations[l];
        const layerDelta = new Float64Array(layer.outputSize);

        for (let j = 0; j < layer.outputSize; j++) {
          let sum = 0;
          for (let k = 0; k < nextLayer.outputSize; k++) {
            sum += nextLayer.weights[k][j] * nextDelta[k];
          }
          // Clamping prevents numerical explosion
          const clipped = Math.max(-100, Math.min(100, sum));
          layerDelta[j] = clipped * layer.activation.deriv(z[j]);
        }
        deltas[l] = layerDelta;
      }
      lastDeltas = deltas;

      // Accumulate gradients across batch
      for (let l = 0; l < numLayers; l++) {
        const layer = this.layers[l];
        const delta = deltas[l];
        const prevA = activations[l];

        for (let i = 0; i < layer.outputSize; i++) {
          layer.biasGradients[i] += delta[i] / batchSize;
          const gradRow = layer.weightGradients[i];
          for (let j = 0; j < layer.inputSize; j++) {
            gradRow[j] += (delta[i] * prevA[j]) / batchSize;
          }
        }
      }
    }

    // 4. Update Weights via Optimizer
    this.stepOptimizer();

    // Hook notification: on_iteration(weights, activations, gradients)
    if (this.hooks.on_iteration) {
      this.hooks.on_iteration(this.getWeightsSnapshot(), lastActivations, lastDeltas);
    }

    return {
      loss: totalLoss / batchSize,
      accuracy: correctCount / batchSize
    };
  }

  /**
   * Apply optimizer update (Adam or SGD) with weight decay
   */
  stepOptimizer() {
    this.timeStep++;
    const t = this.timeStep;
    const lr = this.learningRate;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];

      if (this.optimizerType === 'adam') {
        const b1 = this.beta1;
        const b2 = this.beta2;
        const eps = this.epsilon;
        const biasCorrection1 = 1.0 - Math.pow(b1, t);
        const biasCorrection2 = 1.0 - Math.pow(b2, t);

        for (let i = 0; i < layer.outputSize; i++) {
          // Biases
          const gb = layer.biasGradients[i];
          layer.m_b[i] = b1 * layer.m_b[i] + (1.0 - b1) * gb;
          layer.v_b[i] = b2 * layer.v_b[i] + (1.0 - b2) * gb * gb;
          const mHatB = layer.m_b[i] / biasCorrection1;
          const vHatB = layer.v_b[i] / biasCorrection2;
          layer.biases[i] -= (lr * mHatB) / (Math.sqrt(vHatB) + eps);

          // Weights
          const wRow = layer.weights[i];
          const gradRow = layer.weightGradients[i];
          const mwRow = layer.m_w[i];
          const vwRow = layer.v_w[i];

          for (let j = 0; j < layer.inputSize; j++) {
            const gw = gradRow[j] + this.weightDecay * wRow[j];
            mwRow[j] = b1 * mwRow[j] + (1.0 - b1) * gw;
            vwRow[j] = b2 * vwRow[j] + (1.0 - b2) * gw * gw;
            const mHatW = mwRow[j] / biasCorrection1;
            const vHatW = vwRow[j] / biasCorrection2;
            wRow[j] -= (lr * mHatW) / (Math.sqrt(vHatW) + eps);
          }
        }
      } else {
        // Vanilla SGD
        for (let i = 0; i < layer.outputSize; i++) {
          layer.biases[i] -= lr * layer.biasGradients[i];
          const wRow = layer.weights[i];
          const gradRow = layer.weightGradients[i];
          for (let j = 0; j < layer.inputSize; j++) {
            wRow[j] -= lr * (gradRow[j] + this.weightDecay * wRow[j]);
          }
        }
      }
    }
  }

  /**
   * Online Backpropagation Step on a single sample (Human-in-the-Loop Active Learning)
   * Immediately adjusts weights to rectify mistake on x with true label y.
   * Returns { initialLoss, postLoss, initialPred, postPred, lossDelta }
   */
  online_step(x, y, customLr = null) {
    const savedLr = this.learningRate;
    if (customLr !== null) this.learningRate = customLr;

    // Initial forward pass before update
    const initForward = this.forwardSample(x);
    const initialLoss = this.computeLoss(initForward.output, y);
    const initialPred = Array.from(initForward.output);

    // Run single-sample gradient update
    this.trainBatch([x], [y]);

    // Forward pass immediately after update
    const postForward = this.forwardSample(x);
    const postLoss = this.computeLoss(postForward.output, y);
    const postPred = Array.from(postForward.output);

    if (customLr !== null) this.learningRate = savedLr;

    return {
      initialLoss,
      postLoss,
      initialPred,
      postPred,
      lossReduced: postLoss < initialLoss,
      lossDelta: initialLoss - postLoss
    };
  }

  getWeightsSnapshot() {
    return this.layers.map(l => ({
      layerIndex: l.layerIndex,
      inputSize: l.inputSize,
      outputSize: l.outputSize,
      activation: l.activationName,
      weights: l.weights.map(row => Array.from(row)),
      biases: Array.from(l.biases)
    }));
  }

  loadWeightsSnapshot(snapshot) {
    if (!snapshot || snapshot.length !== this.layers.length) return false;
    for (let l = 0; l < this.layers.length; l++) {
      const snap = snapshot[l];
      const layer = this.layers[l];
      for (let i = 0; i < layer.outputSize; i++) {
        layer.biases[i] = snap.biases[i];
        for (let j = 0; j < layer.inputSize; j++) {
          layer.weights[i][j] = snap.weights[i][j];
        }
      }
    }
    return true;
  }
}
