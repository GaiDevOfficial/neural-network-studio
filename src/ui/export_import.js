/**
 * Model Import/Export & Python / PyTorch Code Generator
 * Neural Network Studio Platform
 */

export class ModelExporter {
  /**
   * The exact payload the "Export JSON" button writes (and "Load Model" reads back).
   * A pure function so the Node verification harness can build/inspect the same file the
   * browser produces. Training-relevant configuration rides along at the top level
   * (`loss` / `optimizer` / `learningRate` / `weightDecay`) so a re-imported model keeps
   * training with the same rules instead of silently falling back to engine defaults.
   */
  static buildExportPayload(weightsSnapshot, architecture, metadata = {}, hyperparameters = {}) {
    return {
      format: 'NeuralNetworkStudio-v1',
      exportedAt: new Date().toISOString(),
      architecture,
      loss: hyperparameters.loss,
      optimizer: hyperparameters.optimizer,
      learningRate: hyperparameters.learningRate,
      weightDecay: hyperparameters.weightDecay,
      metadata,
      layers: weightsSnapshot
    };
  }

  static exportJSON(weightsSnapshot, architecture, metadata = {}, hyperparameters = {}) {
    const payload = ModelExporter.buildExportPayload(weightsSnapshot, architecture, metadata, hyperparameters);
    const jsonStr = JSON.stringify(payload, null, 2);
    ModelExporter.downloadFile(jsonStr, 'neural_model.json', 'application/json');
  }

  static generatePythonCode(weightsSnapshot, architecture) {
    let py = `# Neural Network Studio - Exported Python/NumPy Standalone Model\n`;
    py += `import numpy as np\n\n`;

    py += `class ExportedNeuralNetwork:\n`;
    py += `    def __init__(self):\n`;
    py += `        self.weights = []\n`;
    py += `        self.biases = []\n\n`;

    weightsSnapshot.forEach((l, idx) => {
      py += `        # Layer ${idx}: ${l.inputSize} -> ${l.outputSize} (${l.activation})\n`;
      py += `        self.weights.append(np.array(${JSON.stringify(l.weights)}, dtype=np.float64))\n`;
      py += `        self.biases.append(np.array(${JSON.stringify(l.biases)}, dtype=np.float64))\n\n`;
    });

    py += `    def relu(self, x):\n        return np.maximum(0, x)\n\n`;
    py += `    def sigmoid(self, x):\n        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))\n\n`;
    py += `    def tanh(self, x):\n        return np.tanh(x)\n\n`;
    py += `    def softmax(self, x):\n        exp_x = np.exp(x - np.max(x))\n        return exp_x / np.sum(exp_x)\n\n`;

    py += `    def predict(self, x):\n`;
    py += `        a = np.array(x, dtype=np.float64)\n`;

    weightsSnapshot.forEach((l, idx) => {
      py += `        # Forward Layer ${idx}\n`;
      py += `        z = np.dot(self.weights[${idx}], a) + self.biases[${idx}]\n`;
      if (l.activation === 'relu') {
        py += `        a = self.relu(z)\n`;
      } else if (l.activation === 'sigmoid') {
        py += `        a = self.sigmoid(z)\n`;
      } else if (l.activation === 'tanh') {
        py += `        a = self.tanh(z)\n`;
      } else if (l.activation === 'softmax') {
        py += `        a = self.softmax(z)\n`;
      } else {
        py += `        a = z\n`;
      }
    });

    py += `        return a\n\n`;
    py += `if __name__ == '__main__':\n`;
    py += `    net = ExportedNeuralNetwork()\n`;
    const dummyInput = new Array(architecture[0].size).fill(0.5);
    py += `    test_input = ${JSON.stringify(dummyInput)}\n`;
    py += `    output = net.predict(test_input)\n`;
    py += `    print("Network output:", output)\n`;

    return py;
  }

  static downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  }
}
