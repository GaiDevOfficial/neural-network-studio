/**
 * Loss Functions and Gradients
 * Neural Network Studio Platform
 */

const EPSILON = 1e-12;

export const LossFunctions = {
  mse: {
    name: 'Mean Squared Error',
    calculate: (yPred, yTrue) => {
      let sum = 0;
      for (let i = 0; i < yPred.length; i++) {
        const diff = yPred[i] - yTrue[i];
        sum += diff * diff;
      }
      return sum / yPred.length;
    },
    gradient: (yPred, yTrue) => {
      const n = yPred.length;
      return yPred.map((yp, i) => (2 * (yp - yTrue[i])) / n);
    }
  },

  binary_crossentropy: {
    name: 'Binary Cross-Entropy',
    calculate: (yPred, yTrue) => {
      let sum = 0;
      for (let i = 0; i < yPred.length; i++) {
        const yp = Math.max(EPSILON, Math.min(1 - EPSILON, yPred[i]));
        const yt = yTrue[i];
        sum += - (yt * Math.log(yp) + (1 - yt) * Math.log(1 - yp));
      }
      return sum / yPred.length;
    },
    gradient: (yPred, yTrue) => {
      const n = yPred.length;
      return yPred.map((yp, i) => {
        const clampedYp = Math.max(EPSILON, Math.min(1 - EPSILON, yp));
        const yt = yTrue[i];
        return ((clampedYp - yt) / (clampedYp * (1 - clampedYp))) / n;
      });
    }
  },

  categorical_crossentropy: {
    name: 'Categorical Cross-Entropy',
    calculate: (yPred, yTrue) => {
      let sum = 0;
      for (let i = 0; i < yPred.length; i++) {
        const yp = Math.max(EPSILON, Math.min(1.0, yPred[i]));
        const yt = yTrue[i];
        if (yt > 0) {
          sum -= yt * Math.log(yp);
        }
      }
      return sum;
    },
    // When combined with Softmax in the output layer: dL/dz = yPred - yTrue
    gradient: (yPred, yTrue) => {
      return yPred.map((yp, i) => {
        const clampedYp = Math.max(EPSILON, yp);
        return - (yTrue[i] / clampedYp);
      });
    }
  }
};

export function getLossFunction(type) {
  const key = (type || 'mse').toLowerCase().replace(/[\s-]/g, '_');
  return LossFunctions[key] || LossFunctions.mse;
}
