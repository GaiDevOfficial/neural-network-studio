/**
 * Activation Functions and Analytical Derivatives
 * Neural Network Studio Platform
 */

export const Activations = {
  relu: {
    name: 'ReLU',
    forward: (x) => Math.max(0, x),
    derivative: (x) => (x > 0 ? 1 : 0),
    vectorForward: (vec) => vec.map((x) => Math.max(0, x)),
    vectorDerivative: (vec) => vec.map((x) => (x > 0 ? 1 : 0))
  },

  leaky_relu: {
    name: 'Leaky ReLU',
    alpha: 0.01,
    forward: (x, alpha = 0.01) => (x > 0 ? x : alpha * x),
    derivative: (x, alpha = 0.01) => (x > 0 ? 1 : alpha),
    vectorForward: (vec, alpha = 0.01) => vec.map((x) => (x > 0 ? x : alpha * x)),
    vectorDerivative: (vec, alpha = 0.01) => vec.map((x) => (x > 0 ? 1 : alpha))
  },

  sigmoid: {
    name: 'Sigmoid',
    forward: (x) => {
      // Numerical stability clamp to prevent overflow/underflow
      const clamped = Math.max(-50, Math.min(50, x));
      return 1 / (1 + Math.exp(-clamped));
    },
    derivative: (x) => {
      const s = Activations.sigmoid.forward(x);
      return s * (1 - s);
    },
    vectorForward: (vec) => vec.map((x) => Activations.sigmoid.forward(x)),
    vectorDerivative: (vec) => {
      return vec.map((x) => {
        const s = Activations.sigmoid.forward(x);
        return s * (1 - s);
      });
    }
  },

  tanh: {
    name: 'Tanh',
    forward: (x) => {
      const clamped = Math.max(-50, Math.min(50, x));
      return Math.tanh(clamped);
    },
    derivative: (x) => {
      const t = Math.tanh(Math.max(-50, Math.min(50, x)));
      return 1 - t * t;
    },
    vectorForward: (vec) => vec.map((x) => Math.tanh(Math.max(-50, Math.min(50, x)))),
    vectorDerivative: (vec) => {
      return vec.map((x) => {
        const t = Math.tanh(Math.max(-50, Math.min(50, x)));
        return 1 - t * t;
      });
    }
  },

  linear: {
    name: 'Linear',
    forward: (x) => x,
    derivative: () => 1,
    vectorForward: (vec) => [...vec],
    vectorDerivative: (vec) => vec.map(() => 1)
  },

  softmax: {
    name: 'Softmax',
    // Vector-level activation
    vectorForward: (vec) => {
      const maxVal = Math.max(...vec);
      const exps = vec.map((x) => Math.exp(Math.max(-50, Math.min(50, x - maxVal))));
      const sum = exps.reduce((acc, val) => acc + val, 0);
      return exps.map((val) => (sum === 0 ? 1 / vec.length : val / sum));
    },
    // Softmax derivative is combined directly with Categorical Cross Entropy for numerical stability
    vectorDerivative: (vec) => {
      const probs = Activations.softmax.vectorForward(vec);
      return probs.map((p) => p * (1 - p));
    }
  }
};

export function getActivation(type) {
  const key = (type || 'relu').toLowerCase().replace(/[\s-]/g, '_');
  return Activations[key] || Activations.relu;
}
