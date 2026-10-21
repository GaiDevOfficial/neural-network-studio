/**
 * Node Studio — code generators for exported graphs.
 *
 * Emits standalone, dependency-light source code that performs the SAME forward pass the
 * studio's engine performs (same weight layout, same activation math, same numeric clamps),
 * so a generated file is a faithful artifact of the wired model — verified against the
 * engine numerically by tests/test_node_exports.js, not just "generated without error".
 *
 * File: src/export/codegen.js
 */

const ACTIVATION_LIST = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'linear', 'softmax'];

export const LEAKY_ALPHA = 0.01;

/** Human/JSON-safe metadata for an exported graph. */
export function buildModelMeta({ plan, datasetName = null, epochs = null, origin = 'node_studio' }) {
  return {
    origin,
    platform: 'Neural Network Studio — Node Studio',
    exportedAt: new Date().toISOString(),
    dataset: datasetName || (plan && plan.meta ? plan.meta.datasetName : null),
    inputShape: plan && plan.meta ? plan.meta.inputShape : null,
    inputKind: plan && plan.meta ? plan.meta.inputKind : null,
    outputSize: plan && plan.meta ? plan.meta.outputSize : null,
    architecture: plan && plan.network ? plan.network.architecture : null,
    loss: plan && plan.network ? plan.network.loss : null,
    optimizer: plan && plan.network ? plan.network.optimizer : null,
    learningRate: plan && plan.network ? plan.network.learningRate : null,
    weightDecay: plan && plan.network ? plan.network.weightDecay : null,
    training: plan ? plan.training : null,
    epochs,
    chain: plan ? plan.chain : null
  };
}

/** One-line description used in file headers. */
export function planLabel(plan) {
  if (!plan || !plan.network) return 'compiled network';
  const arch = plan.network.architecture
    .map((l) => `${l.size}${l.activation && l.activation !== 'linear' ? `(${l.activation})` : ''}`)
    .join(' → ');
  return `${arch} · ${plan.network.loss} · ${plan.network.optimizer}`;
}

/* ------------------------------------------------------------------ Python */

export function generatePythonModel({ plan, weights, metadata = {} }) {
  const lines = [];
  lines.push('# Neural Network Studio — Node Studio export');
  lines.push(`# graph: ${planLabel(plan)}`);
  if (metadata.dataset) lines.push(`# dataset: ${metadata.dataset}`);
  lines.push(`# generated: ${new Date().toISOString()}`);
  lines.push('# Forward pass mirrors the studio engine exactly (same weight layout + clamps).');
  lines.push('');
  lines.push('import numpy as np');
  lines.push('');
  lines.push(`MODEL_META = ${jsonToPython(metadata, 1)}`);
  lines.push('');
  lines.push('class ExportedNeuralNetwork:');
  lines.push('    def __init__(self):');
  lines.push('        self.weights = []');
  lines.push('        self.biases = []');
  lines.push('        self.activations = []');

  weights.forEach((layer, idx) => {
    lines.push(`        # Layer ${idx}: ${layer.inputSize} -> ${layer.outputSize} (${layer.activation})`);
    lines.push(`        self.weights.append(np.array(${JSON.stringify(layer.weights)}, dtype=np.float64))`);
    lines.push(`        self.biases.append(np.array(${JSON.stringify(layer.biases)}, dtype=np.float64))`);
    lines.push(`        self.activations.append(${JSON.stringify(layer.activation || 'linear')})`);
  });

  lines.push('');
  lines.push('    def _relu(self, x): return np.maximum(0.0, x)');
  lines.push('    def _leaky_relu(self, x): return np.where(x > 0, x, 0.01 * x)');
  lines.push('    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))');
  lines.push('    def _tanh(self, x): return np.tanh(np.clip(x, -50.0, 50.0))');
  lines.push('    def _linear(self, x): return x');
  lines.push('    def _softmax(self, x):');
  lines.push('        e = np.exp(np.clip(x - np.max(x), -50.0, 50.0))');
  lines.push('        s = float(np.sum(e))');
  lines.push('        return e / s if s != 0.0 else np.full_like(x, 1.0 / len(x))');
  lines.push('');
  lines.push('    def predict(self, x):');
  lines.push('        a = np.asarray(x, dtype=np.float64)');
  lines.push('        for w, b, act in zip(self.weights, self.biases, self.activations):');
  lines.push('            a = getattr(self, "_" + act)(w @ a + b)');
  lines.push('        return a');
  lines.push('');
  lines.push('if __name__ == "__main__":');
  lines.push('    net = ExportedNeuralNetwork()');
  const demoInput = new Array((plan && plan.network && plan.network.architecture.length > 0)
    ? plan.network.architecture[0].size
    : (weights[0] ? weights[0].inputSize : 1)).fill(0.5);
  lines.push(`    demo_input = ${JSON.stringify(demoInput)}`);
  lines.push('    out = net.predict(demo_input)');
  lines.push('    print("output:", out.tolist())');
  return lines.join('\n') + '\n';
}

/* ---------------------------------------------------------------- JavaScript */

export function generateJavaScriptModel({ plan, weights, metadata = {} }) {
  const lines = [];
  lines.push('// Neural Network Studio — Node Studio export');
  lines.push(`// graph: ${planLabel(plan)}`);
  if (metadata.dataset) lines.push(`// dataset: ${metadata.dataset}`);
  lines.push(`// generated: ${new Date().toISOString()}`);
  lines.push('');
  lines.push(`export const MODEL_META = ${JSON.stringify(metadata, null, 2)};`);
  lines.push('');
  lines.push('export class ExportedNeuralNetwork {');
  lines.push('  constructor() {');
  lines.push('    this.layers = [');
  weights.forEach((layer) => {
    lines.push(`      // ${layer.inputSize} -> ${layer.outputSize} (${layer.activation})`);
    lines.push(`      { weights: ${JSON.stringify(layer.weights)}, biases: ${JSON.stringify(layer.biases)}, activation: ${JSON.stringify(layer.activation || 'linear')} },`);
  });
  lines.push('    ];');
  lines.push('  }');
  lines.push('');
  lines.push('  static relu(x) { return x > 0 ? x : 0; }');
  lines.push(`  static leakyRelu(x) { return x > 0 ? x : ${LEAKY_ALPHA} * x; }`);
  lines.push('  static sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))); }');
  lines.push('  static tanh(x) { return Math.tanh(Math.max(-50, Math.min(50, x))); }');
  lines.push('');
  lines.push('  predict(input) {');
  lines.push('    let a = input.slice();');
  lines.push('    for (const layer of this.layers) {');
  lines.push('      const z = layer.weights.map((row) => row.reduce((sum, w, i) => sum + w * a[i], 0) + 0);');
  lines.push('      const out = z.map((value, i) => value + layer.biases[i]);');
  lines.push('      if (layer.activation === "relu") a = out.map((v) => ExportedNeuralNetwork.relu(v));');
  lines.push('      else if (layer.activation === "leaky_relu") a = out.map((v) => ExportedNeuralNetwork.leakyRelu(v));');
  lines.push('      else if (layer.activation === "sigmoid") a = out.map((v) => ExportedNeuralNetwork.sigmoid(v));');
  lines.push('      else if (layer.activation === "tanh") a = out.map((v) => ExportedNeuralNetwork.tanh(v));');
  lines.push('      else if (layer.activation === "softmax") {');
  lines.push('        const maxVal = Math.max(...out);');
  lines.push('        const exps = out.map((v) => Math.exp(Math.max(-50, Math.min(50, v - maxVal))));');
  lines.push('        const sum = exps.reduce((s, v) => s + v, 0);');
  lines.push('        a = exps.map((v) => (sum === 0 ? 1 / out.length : v / sum));');
  lines.push('      } else a = out;');
  lines.push('    }');
  lines.push('    return a;');
  lines.push('  }');
  lines.push('}');
  lines.push('');
  lines.push('export default ExportedNeuralNetwork;');
  return lines.join('\n') + '\n';
}

/* ---------------------------------------------------------------- helpers */

function jsonToPython(value, indent) {
  const pad = ' '.repeat(indent);
  const inner = ' '.repeat(indent + 4);
  if (value === null || value === undefined) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'None';
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    return `[\n${value.map((v) => inner + jsonToPython(v, indent + 4)).join(',\n')}\n${pad}]`;
  }
  const keys = Object.keys(value);
  if (keys.length === 0) return '{}';
  return `{\n${keys.map((k) => `${inner}${JSON.stringify(k)}: ${jsonToPython(value[k], indent + 4)}`).join(',\n')}\n${pad}}`;
}
