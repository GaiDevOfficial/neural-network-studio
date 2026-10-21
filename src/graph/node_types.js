/**
 * Node Studio — central block (node type) registry.
 *
 * Every draggable block on the visual canvas is declared here exactly once: its sockets,
 * tunable parameters, palette category and engine mapping. The canvas UI, the shape-inference
 * engine and the compiler all read from this single source of truth, so a block can never
 * render one thing and compile into another.
 *
 * Categories use the studio's tactile developer palette (see node_studio.css):
 *   data = amber, layers = blue, activations = green, loss/optimizers = coral,
 *   logic = warm slate, output = muted teal.
 *
 * File: src/graph/node_types.js
 */

export const CATEGORY_META = {
  data: { label: 'Data', color: '#d9a441' },
  layer: { label: 'Layers', color: '#4a8fd4' },
  activation: { label: 'Activations', color: '#4fae6a' },
  core: { label: 'Loss & Optimizers', color: '#e0664d' },
  logic: { label: 'Logic & Training', color: '#8f8878' },
  output: { label: 'Output & Gauges', color: '#5aa9a2' }
};

/** Built-in dataset ids offered by the Dataset Source block (kept in sync with DatasetManager). */
export const DATA_SOURCES = [
  { id: 'mnist_01', label: 'MNIST 0 vs 1 (real 28×28 corpus)', image: true },
  { id: 'synthetic_01', label: 'Stroke Fixture 28×28 (0 vs 1)', image: true },
  { id: 'digits', label: 'Template Digits 8×8 (4 classes)', image: true },
  { id: 'xor', label: 'XOR (2 features)' },
  { id: 'circles', label: 'Concentric Circles (2 features)' },
  { id: 'moons', label: 'Two Moons (2 features)' },
  { id: 'spiral', label: 'Two Spirals (2 features)' },
  { id: 'iris', label: 'Iris (4 features, 3 classes)' },
  { id: 'linear_regression', label: 'Non-linear Regression (1 feature)', regression: true },
  { id: 'custom_csv', label: 'Uploaded CSV (choose file on the page)', custom: true }
];

/**
 * Node type definitions.
 * `supported: false` blocks are shown in the palette under "Planned" with their reason —
 * they are never wired into a compile path that would pretend to execute them.
 */
export const NODE_TYPES = {
  // ---------------------------------------------------------------- Data
  source: {
    category: 'data', label: 'Dataset Source', order: 10, singleton: 'source',
    summary: 'Selects the dataset this graph trains on. Emits an image grid ([h,w]) for image corpora or a flat feature vector ([n]) for tables.',
    inputs: [],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {
      datasetId: { type: 'dataset', default: 'mnist_01', label: 'Dataset' }
    }
  },
  sql_source: {
    category: 'data', label: 'SQL Query', order: 20, singleton: 'source',
    summary: 'Streams rows from a SQLite database through the local Data API (desktop app / dev server). The last column becomes the target.',
    inputs: [],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {
      db: { type: 'string', default: '', label: 'SQLite file', placeholder: 'C:\\data\\records.sqlite' },
      query: { type: 'string', default: 'SELECT * FROM data', label: 'SELECT query', multiline: true }
    }
  },
  api_source: {
    category: 'data', label: 'REST API', order: 30, singleton: 'source',
    summary: 'Fetches JSON rows over HTTP: an array of objects or {columns, rows}. The last column becomes the target.',
    inputs: [],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {
      url: { type: 'string', default: '', label: 'URL', placeholder: 'https://example.com/data.json' }
    }
  },

  // ---------------------------------------------------------------- Layers
  flatten: {
    category: 'layer', label: 'Flatten', order: 10,
    summary: 'Reshapes an image grid into a flat feature vector (28×28 → 784). No-op on vectors.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },
  dense: {
    category: 'layer', label: 'Dense Layer', order: 20,
    summary: 'Fully-connected layer of N neurons. Without a following Activation block it is linear.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {
      size: { type: 'int', min: 1, max: 512, default: 16, label: 'Neurons' }
    }
  },
  conv2d: {
    category: 'layer', label: 'Conv 2D', order: 30, supported: false,
    reason: 'Engine v1 is a Dense-stack engine (verified MLP). Conv2D needs 3D tensor kernels + their backward pass — planned for engine v2, never simulated.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: { filters: { type: 'int', min: 1, max: 64, default: 8, label: 'Filters' } }
  },
  maxpool: {
    category: 'layer', label: 'Max Pool', order: 40, supported: false,
    reason: 'Planned with engine v2 (convolutional stacks). Not available in engine v1.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: { size: { type: 'int', min: 2, max: 4, default: 2, label: 'Pool size' } }
  },
  dropout: {
    category: 'layer', label: 'Dropout', order: 50, supported: false,
    reason: 'Dropout needs train/eval-mode masking inside the verified engine — planned, not yet implemented. Use weight decay (optimizer block) meanwhile.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: { rate: { type: 'float', min: 0.05, max: 0.9, step: 0.05, default: 0.2, label: 'Drop rate' } }
  },

  // ---------------------------------------------------------------- Activations
  relu: {
    category: 'activation', label: 'ReLU', order: 10, activation: 'relu',
    summary: 'max(0, x) — the default hidden activation.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },
  leaky_relu: {
    category: 'activation', label: 'Leaky ReLU', order: 20, activation: 'leaky_relu',
    summary: 'x if x > 0 else 0.01·x — keeps a small gradient for dead units.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },
  tanh: {
    category: 'activation', label: 'Tanh', order: 30, activation: 'tanh',
    summary: 'Squashes to (−1, 1). Useful for centered hidden representations.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },
  sigmoid: {
    category: 'activation', label: 'Sigmoid', order: 40, activation: 'sigmoid',
    summary: 'Squashes to (0, 1). Required pair for Binary Cross-Entropy.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },
  softmax: {
    category: 'activation', label: 'Softmax', order: 50, activation: 'softmax',
    summary: 'Turns a class vector into probabilities. Required pair for Categorical Cross-Entropy (rank-1 input).',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },
  linear: {
    category: 'activation', label: 'Linear', order: 60, activation: 'linear',
    summary: 'Identity passthrough — the natural pair for MSE regression outputs.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {}
  },

  // ---------------------------------------------------------------- Loss
  bce_loss: {
    category: 'core', label: 'Binary Cross-Entropy', order: 10, loss: 'binary_crossentropy',
    summary: 'Two-class loss. Must be fed by a Sigmoid output of size 2.',
    inputs: [
      { name: 'in', kind: 'tensor' },
      { name: 'opt', kind: 'config' },
      { name: 'loop', kind: 'config' },
      { name: 'schedule', kind: 'config' }
    ],
    outputs: [{ name: 'monitor', kind: 'monitor' }],
    params: {}
  },
  cce_loss: {
    category: 'core', label: 'Categorical Cross-Entropy', order: 20, loss: 'categorical_crossentropy',
    summary: 'Multi-class loss. Must be fed by a Softmax output whose size equals the number of classes.',
    inputs: [
      { name: 'in', kind: 'tensor' },
      { name: 'opt', kind: 'config' },
      { name: 'loop', kind: 'config' },
      { name: 'schedule', kind: 'config' }
    ],
    outputs: [{ name: 'monitor', kind: 'monitor' }],
    params: {}
  },
  mse_loss: {
    category: 'core', label: 'MSE Loss', order: 30, loss: 'mse',
    summary: 'Mean-squared error — regression loss; any output activation, usual size 1.',
    inputs: [
      { name: 'in', kind: 'tensor' },
      { name: 'opt', kind: 'config' },
      { name: 'loop', kind: 'config' },
      { name: 'schedule', kind: 'config' }
    ],
    outputs: [{ name: 'monitor', kind: 'monitor' }],
    params: {}
  },

  // ---------------------------------------------------------------- Optimizers
  adam_optimizer: {
    category: 'core', label: 'Adam', order: 40, optimizer: 'adam',
    summary: 'Adaptive moments — the studio default.',
    inputs: [],
    outputs: [{ name: 'cfg', kind: 'config' }],
    params: {
      learningRate: { type: 'float', min: 0.001, max: 0.2, step: 0.001, default: 0.03, label: 'Learning rate' },
      weightDecay: { type: 'float', min: 0, max: 0.01, step: 0.0001, default: 0.0001, label: 'Weight decay' }
    }
  },
  sgd_optimizer: {
    category: 'core', label: 'SGD', order: 50, optimizer: 'sgd',
    summary: 'Plain stochastic gradient descent.',
    inputs: [],
    outputs: [{ name: 'cfg', kind: 'config' }],
    params: {
      learningRate: { type: 'float', min: 0.001, max: 0.2, step: 0.001, default: 0.05, label: 'Learning rate' },
      weightDecay: { type: 'float', min: 0, max: 0.01, step: 0.0001, default: 0.0001, label: 'Weight decay' }
    }
  },
  momentum_optimizer: {
    category: 'core', label: 'Momentum', order: 60, optimizer: 'momentum',
    summary: 'SGD + velocity smoothing.',
    inputs: [],
    outputs: [{ name: 'cfg', kind: 'config' }],
    params: {
      learningRate: { type: 'float', min: 0.001, max: 0.2, step: 0.001, default: 0.03, label: 'Learning rate' },
      weightDecay: { type: 'float', min: 0, max: 0.01, step: 0.0001, default: 0.0001, label: 'Weight decay' }
    }
  },
  rmsprop_optimizer: {
    category: 'core', label: 'RMSprop', order: 70, optimizer: 'rmsprop',
    summary: 'Running average of squared gradients.',
    inputs: [],
    outputs: [{ name: 'cfg', kind: 'config' }],
    params: {
      learningRate: { type: 'float', min: 0.001, max: 0.2, step: 0.001, default: 0.02, label: 'Learning rate' },
      weightDecay: { type: 'float', min: 0, max: 0.01, step: 0.0001, default: 0.0001, label: 'Weight decay' }
    }
  },

  // ---------------------------------------------------------------- Logic
  lr_schedule: {
    category: 'logic', label: 'LR Scheduler', order: 10,
    summary: 'Step decay: multiply the optimizer learning rate by a factor every N epochs (applied live during a run).',
    inputs: [],
    outputs: [{ name: 'cfg', kind: 'config' }],
    params: {
      mode: { type: 'select', options: ['constant', 'step'], default: 'step', label: 'Mode' },
      factor: { type: 'float', min: 0.1, max: 0.95, step: 0.05, default: 0.5, label: 'Decay factor' },
      every: { type: 'int', min: 5, max: 500, default: 100, label: 'Every N epochs' }
    }
  },
  epoch_loop: {
    category: 'logic', label: 'Epoch Loop', order: 20,
    summary: 'Run controller config: auto-stop on convergence (held-out loss flat for N epochs) or a fixed epoch budget.',
    inputs: [],
    outputs: [{ name: 'cfg', kind: 'config' }],
    params: {
      mode: { type: 'select', options: ['auto_stop', 'fixed'], default: 'auto_stop', label: 'Stop rule' },
      maxEpochs: { type: 'int', min: 1, max: 100000, default: 50, label: 'Fixed epochs' },
      patience: { type: 'int', min: 10, max: 1000, default: 150, label: 'Auto-stop patience' },
      batchSize: { type: 'int', min: 1, max: 64, default: 16, label: 'Batch size' }
    }
  },
  class_filter: {
    category: 'logic', label: 'Class Filter', order: 30,
    summary: 'Keeps a subset of classes and remaps the targets (e.g. “0,1” filters a multi-class dataset to two classes). Classification datasets only.',
    inputs: [{ name: 'in', kind: 'tensor' }],
    outputs: [{ name: 'out', kind: 'tensor' }],
    params: {
      keep: { type: 'string', default: '0,1', label: 'Keep classes (comma list)' }
    }
  },

  // ---------------------------------------------------------------- Output taps
  metrics_gauge: {
    category: 'output', label: 'Loss / Accuracy Gauge', order: 10,
    summary: 'Live training readout: epoch, training loss, held-out loss and accuracy.',
    inputs: [{ name: 'in', kind: 'monitor' }],
    outputs: [],
    params: {}
  },
  confusion_matrix: {
    category: 'output', label: 'Confusion Matrix', order: 20,
    summary: 'Real per-class hit counts on the held-out set, computed from live predictions when a run completes.',
    inputs: [{ name: 'in', kind: 'monitor' }],
    outputs: [],
    params: {}
  },
  prediction_view: {
    category: 'output', label: 'Prediction View', order: 30,
    summary: 'Probe the trained network: a drawing pad for image models, feature sliders for tables. Runs a real forward pass.',
    inputs: [{ name: 'in', kind: 'monitor' }],
    outputs: [],
    params: {}
  }
};

export const TENSOR_NODE_TYPES = ['source', 'sql_source', 'api_source', 'flatten', 'dense', 'class_filter'];
export const ACTIVATION_NODE_TYPES = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax', 'linear'];
export const LOSS_NODE_TYPES = ['bce_loss', 'cce_loss', 'mse_loss'];
export const OPTIMIZER_NODE_TYPES = ['adam_optimizer', 'sgd_optimizer', 'momentum_optimizer', 'rmsprop_optimizer'];
export const SOURCE_NODE_TYPES = ['source', 'sql_source', 'api_source'];
export const TAP_NODE_TYPES = ['metrics_gauge', 'confusion_matrix', 'prediction_view'];

export function getNodeType(type) {
  return NODE_TYPES[type] || null;
}

export function isSupported(type) {
  const def = NODE_TYPES[type];
  return Boolean(def && def.supported !== false);
}

/** Palette layout: supported blocks grouped by category, planned blocks collected separately. */
export function paletteSections() {
  const sections = [];
  const planned = [];
  for (const [key, meta] of Object.entries(CATEGORY_META)) {
    const types = Object.entries(NODE_TYPES)
      .filter(([, def]) => def.category === key)
      .sort((a, b) => (a[1].order || 0) - (b[1].order || 0));
    const supported = types.filter(([, def]) => def.supported !== false).map(([type, def]) => ({ type, ...def }));
    const unsupported = types.filter(([, def]) => def.supported === false).map(([type, def]) => ({ type, ...def }));
    if (supported.length > 0) sections.push({ category: key, meta, blocks: supported });
    planned.push(...unsupported);
  }
  return { sections, planned };
}

/** Merge declared defaults with overrides (unknown keys dropped, ranges clamped). */
export function defaultParams(type, overrides = {}) {
  const def = NODE_TYPES[type];
  if (!def) return {};
  const params = {};
  for (const [name, spec] of Object.entries(def.params || {})) {
    params[name] = spec.default;
  }
  for (const [name, value] of Object.entries(overrides || {})) {
    const spec = (def.params || {})[name];
    if (!spec) continue;
    if (spec.type === 'int' || spec.type === 'float') {
      let num = Number(value);
      if (!Number.isFinite(num)) continue;
      if (spec.min !== undefined) num = Math.max(spec.min, num);
      if (spec.max !== undefined) num = Math.min(spec.max, num);
      params[name] = spec.type === 'int' ? Math.round(num) : num;
    } else {
      params[name] = String(value);
    }
  }
  return params;
}
