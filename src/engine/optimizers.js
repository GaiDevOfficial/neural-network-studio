/**
 * Optimizers with State Buffers and L2 Regularization
 * Neural Network Studio Platform
 */

export class Optimizer {
  constructor(learningRate = 0.03, weightDecay = 0.0001) {
    this.lr = learningRate;
    this.weightDecay = weightDecay;
    this.iterations = 0;
  }

  // Update a 2D weight matrix or 1D bias vector
  update(param, grad, stateKey, isWeight = true) {
    throw new Error('update() must be implemented by subclass');
  }

  reset() {
    this.iterations = 0;
  }
}

export class SGDOptimizer extends Optimizer {
  constructor(learningRate = 0.05, weightDecay = 0.0001) {
    super(learningRate, weightDecay);
  }

  update(param, grad, stateKey, isWeight = true) {
    if (isWeight) {
      // 2D Matrix (Weights)
      const rows = param.length;
      const cols = param[0].length;
      const newParam = new Array(rows);
      for (let i = 0; i < rows; i++) {
        newParam[i] = new Float64Array(cols);
        for (let j = 0; j < cols; j++) {
          let g = grad[i][j];
          if (isWeight && this.weightDecay > 0) {
            g += this.weightDecay * param[i][j];
          }
          newParam[i][j] = param[i][j] - this.lr * g;
        }
      }
      return newParam;
    } else {
      // 1D Vector (Biases)
      const len = param.length;
      const newParam = new Float64Array(len);
      for (let i = 0; i < len; i++) {
        newParam[i] = param[i] - this.lr * grad[i];
      }
      return newParam;
    }
  }
}

export class MomentumOptimizer extends Optimizer {
  constructor(learningRate = 0.03, momentum = 0.9, weightDecay = 0.0001) {
    super(learningRate, weightDecay);
    this.momentum = momentum;
    this.velocities = new Map();
  }

  update(param, grad, stateKey, isWeight = true) {
    if (isWeight) {
      const rows = param.length;
      const cols = param[0].length;
      if (!this.velocities.has(stateKey)) {
        const vInit = [];
        for (let i = 0; i < rows; i++) vInit.push(new Float64Array(cols));
        this.velocities.set(stateKey, vInit);
      }
      const v = this.velocities.get(stateKey);
      const newParam = new Array(rows);
      for (let i = 0; i < rows; i++) {
        newParam[i] = new Float64Array(cols);
        for (let j = 0; j < cols; j++) {
          let g = grad[i][j];
          if (isWeight && this.weightDecay > 0) {
            g += this.weightDecay * param[i][j];
          }
          v[i][j] = this.momentum * v[i][j] + this.lr * g;
          newParam[i][j] = param[i][j] - v[i][j];
        }
      }
      return newParam;
    } else {
      const len = param.length;
      if (!this.velocities.has(stateKey)) {
        this.velocities.set(stateKey, new Float64Array(len));
      }
      const v = this.velocities.get(stateKey);
      const newParam = new Float64Array(len);
      for (let i = 0; i < len; i++) {
        v[i] = this.momentum * v[i] + this.lr * grad[i];
        newParam[i] = param[i] - v[i];
      }
      return newParam;
    }
  }

  reset() {
    super.reset();
    this.velocities.clear();
  }
}

export class AdamOptimizer extends Optimizer {
  constructor(learningRate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0.0001) {
    super(learningRate, weightDecay);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.m = new Map();
    this.v = new Map();
  }

  update(param, grad, stateKey, isWeight = true) {
    this.iterations++;
    const t = Math.max(1, Math.floor(this.iterations / 2) + 1);
    const beta1Correction = 1 - Math.pow(this.beta1, t);
    const beta2Correction = 1 - Math.pow(this.beta2, t);

    if (isWeight) {
      const rows = param.length;
      const cols = param[0].length;
      if (!this.m.has(stateKey)) {
        const mInit = [];
        const vInit = [];
        for (let i = 0; i < rows; i++) {
          mInit.push(new Float64Array(cols));
          vInit.push(new Float64Array(cols));
        }
        this.m.set(stateKey, mInit);
        this.v.set(stateKey, vInit);
      }
      const mArr = this.m.get(stateKey);
      const vArr = this.v.get(stateKey);
      const newParam = new Array(rows);

      for (let i = 0; i < rows; i++) {
        newParam[i] = new Float64Array(cols);
        for (let j = 0; j < cols; j++) {
          let g = grad[i][j];
          if (isWeight && this.weightDecay > 0) {
            g += this.weightDecay * param[i][j];
          }
          // Biased first and second moment estimate
          mArr[i][j] = this.beta1 * mArr[i][j] + (1 - this.beta1) * g;
          vArr[i][j] = this.beta2 * vArr[i][j] + (1 - this.beta2) * g * g;

          // Bias-corrected moment estimates
          const mHat = mArr[i][j] / beta1Correction;
          const vHat = vArr[i][j] / beta2Correction;

          newParam[i][j] = param[i][j] - (this.lr * mHat) / (Math.sqrt(Math.max(0, vHat)) + this.epsilon);
        }
      }
      return newParam;
    } else {
      const len = param.length;
      if (!this.m.has(stateKey)) {
        this.m.set(stateKey, new Float64Array(len));
        this.v.set(stateKey, new Float64Array(len));
      }
      const mArr = this.m.get(stateKey);
      const vArr = this.v.get(stateKey);
      const newParam = new Float64Array(len);

      for (let i = 0; i < len; i++) {
        const g = grad[i];
        mArr[i] = this.beta1 * mArr[i] + (1 - this.beta1) * g;
        vArr[i] = this.beta2 * vArr[i] + (1 - this.beta2) * g * g;

        const mHat = mArr[i] / beta1Correction;
        const vHat = vArr[i] / beta2Correction;

        newParam[i] = param[i] - (this.lr * mHat) / (Math.sqrt(Math.max(0, vHat)) + this.epsilon);
      }
      return newParam;
    }
  }

  reset() {
    super.reset();
    this.m.clear();
    this.v.clear();
  }
}

export class RMSpropOptimizer extends Optimizer {
  constructor(learningRate = 0.01, decayRate = 0.9, epsilon = 1e-8, weightDecay = 0.0001) {
    super(learningRate, weightDecay);
    this.decayRate = decayRate;
    this.epsilon = epsilon;
    this.cache = new Map();
  }

  update(param, grad, stateKey, isWeight = true) {
    if (isWeight) {
      const rows = param.length;
      const cols = param[0].length;
      if (!this.cache.has(stateKey)) {
        const cInit = [];
        for (let i = 0; i < rows; i++) cInit.push(new Float64Array(cols));
        this.cache.set(stateKey, cInit);
      }
      const cache = this.cache.get(stateKey);
      const newParam = new Array(rows);
      for (let i = 0; i < rows; i++) {
        newParam[i] = new Float64Array(cols);
        for (let j = 0; j < cols; j++) {
          let g = grad[i][j];
          if (isWeight && this.weightDecay > 0) {
            g += this.weightDecay * param[i][j];
          }
          cache[i][j] = this.decayRate * cache[i][j] + (1 - this.decayRate) * g * g;
          newParam[i][j] = param[i][j] - (this.lr * g) / (Math.sqrt(cache[i][j]) + this.epsilon);
        }
      }
      return newParam;
    } else {
      const len = param.length;
      if (!this.cache.has(stateKey)) {
        this.cache.set(stateKey, new Float64Array(len));
      }
      const cache = this.cache.get(stateKey);
      const newParam = new Float64Array(len);
      for (let i = 0; i < len; i++) {
        const g = grad[i];
        cache[i] = this.decayRate * cache[i] + (1 - this.decayRate) * g * g;
        newParam[i] = param[i] - (this.lr * g) / (Math.sqrt(cache[i]) + this.epsilon);
      }
      return newParam;
    }
  }

  reset() {
    super.reset();
    this.cache.clear();
  }
}

export function createOptimizer(type, lr = 0.02, weightDecay = 0.0001) {
  const key = (type || 'adam').toLowerCase();
  switch (key) {
    case 'sgd':
      return new SGDOptimizer(lr, weightDecay);
    case 'momentum':
      return new MomentumOptimizer(lr, 0.9, weightDecay);
    case 'rmsprop':
      return new RMSpropOptimizer(lr, 0.9, 1e-8, weightDecay);
    case 'adam':
    default:
      return new AdamOptimizer(lr, 0.9, 0.999, 1e-8, weightDecay);
  }
}
