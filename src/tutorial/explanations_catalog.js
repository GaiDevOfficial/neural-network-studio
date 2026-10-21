/**
 * Node Studio — explanations catalog.
 *
 * One repository of educational content for every supported block and for the core concepts the
 * tutorials teach (wiring, forward passes, backpropagation, loss pairing, quick-add, sniping).
 * Every entry carries three layers so the same key can drive the Absolute-Beginner vocabulary,
 * the Smart-Assist vocabulary and the Advanced Studio vocabulary:
 *
 *   plain      — jargon-free, one breath
 *   analogy    — an everyday metaphor (ReLU = a light switch)
 *   technical  — the exact engine behavior (max(0,x), pairing matrix, tensor shapes)
 *   signals    — what the moving particles / neuron glow actually represent for this block
 *   tips       — practical wiring advice shown in the quick-add menu and mission cards
 *
 * Pure data + pure helpers; no DOM at import time. Fixture: tests/test_step_by_step.js proves
 * every supported palette block has an entry and that no help string is empty or a placeholder.
 *
 * File: src/tutorial/explanations_catalog.js
 */

import { NODE_TYPES, ACTIVATION_NODE_TYPES, LOSS_NODE_TYPES, OPTIMIZER_NODE_TYPES } from '../graph/node_types.js';

/* ------------------------------------------------------------------ entries */

export const EXPLANATIONS = {
  /* ------------------------------------------------------------- data */
  source: {
    title: 'Dataset',
    category: 'data',
    friendlyLabel: 'Example pictures',
    aliases: ['data', 'dataset', 'mnist', 'pictures', 'examples', 'input data'],
    plain: 'Dataset Source chooses the examples the network learns from. Every training round shows it one batch of those examples.',
    analogy: 'A stack of flash cards. The network studies the cards, and the held-out cards are the surprise quiz.',
    technical: 'Emits the dataset tensor: an image grid [h, w] for image corpora or a flat feature vector [n] for tables. Exactly one source block is allowed per graph; the compiler splits train/held-out with the dataset manifest.',
    signals: 'Amber particles leaving this block carry the raw example (pixels or features) into the first layer.',
    tips: ['Drop it first — every working graph starts here.', 'SQL and REST blocks are sources too; only one source per graph.']
  },
  sql_source: {
    title: 'SQL Query',
    category: 'data',
    friendlyLabel: 'Rows from a database',
    aliases: ['sql', 'sqlite', 'database', 'query', 'db'],
    plain: 'Reads rows from a SQLite file and turns them into training examples.',
    analogy: 'A librarian who reads exactly one shelf (your SELECT) and hands you the rows in small batches.',
    technical: 'The bundled read-only Data API executes one SELECT (SELECT-only, mode=ro enforced server-side) and streams windows as JSON; the last column becomes the target, then the same CSV normalization and split run.',
    signals: 'Particles here carry streamed rows rather than pixels; the compiler shapes them exactly like a CSV upload.',
    tips: ['Open the Inspector to set the .sqlite path and query.', 'The last selected column is always the target.']
  },
  api_source: {
    title: 'REST API',
    category: 'data',
    friendlyLabel: 'Rows from the web',
    aliases: ['rest', 'json', 'http', 'api', 'endpoint'],
    plain: 'Fetches JSON rows from a URL and turns them into training examples.',
    analogy: 'Ordering a dataset online: you give the address, the data arrives ready to sort.',
    technical: 'Accepts an array of objects or { columns, rows }; the last column becomes the target. Normalization matches the CSV pipeline.',
    signals: 'Particles carry HTTP-fetched rows; the worker reports fetch errors as real error codes, never fake placeholders.',
    tips: ['CORS must allow the page origin when running from the browser.', 'Set the URL in the Inspector.']
  },

  /* ----------------------------------------------------------- layers */
  flatten: {
    title: 'Flatten',
    category: 'layer',
    friendlyLabel: 'Untangle the picture',
    aliases: ['flatten', 'reshape', 'vector', 'grid to list'],
    plain: 'Flatten turns a picture grid into one long list of numbers so the Dense Layers can read it.',
    analogy: 'Taking a chessboard and reading every square out loud one row at a time.',
    technical: 'Reshapes [h, w] → [h·w]; a no-op on vectors. Smart Assist inserts one automatically when a Dense or Loss block receives a rank-2 tensor.',
    signals: 'Particles enter as a 2-D grid and leave as a single stream — the pixel count never changes, only the shape.',
    tips: ['28×28 becomes 784 values.', 'You rarely need to add it manually in Beginner mode.']
  },
  dense: {
    title: 'Dense Layer',
    category: 'layer',
    friendlyLabel: 'Brain cell',
    aliases: ['dense', 'layer', 'neurons', 'fully connected', 'fc', 'weights'],
    plain: 'A Dense Layer is a set of neurons that mixes the incoming numbers with learned weights and passes the result on.',
    analogy: 'A committee where every member listens to every incoming signal, weighs what matters, and votes.',
    technical: 'Fully connected layer: y = W·x + b. W is [size, inFeatures], b is [size]. Without a following Activation block the layer is linear.',
    signals: 'Line thickness between the layers is the real |weight|; cyan is positive, pink is negative. Weight lines change after every backprop step.',
    tips: ['Set the neuron count in the Inspector.', 'The last Dense must match the loss: 2 for BCE, class count for CCE, 1 for regression.']
  },
  conv2d: {
    title: 'Conv 2D (planned)',
    category: 'layer',
    friendlyLabel: 'Pattern scanner (later)',
    aliases: ['conv', 'convolution', 'kernel'],
    plain: 'A future block that scans small patches of a picture for patterns.',
    analogy: 'A magnifying glass that slides across a photo looking for edges and curves.',
    technical: 'Not available: engine v1 is a verified Dense-stack engine. Conv2D needs 3-D tensor kernels and their backward pass (engine v2). The block is rendered disabled with this reason — it is never simulated.',
    signals: 'Planned — no wire may claim to run it until the engine implements it.',
    tips: ['Use Flatten + Dense for image models today.']
  },
  maxpool: {
    title: 'Max Pool (planned)',
    category: 'layer',
    friendlyLabel: 'Shrinker (later)',
    aliases: ['pool', 'maxpool', 'downsample'],
    plain: 'A future block that shrinks a picture while keeping the strongest signals.',
    analogy: 'Summarizing a paragraph by keeping only the boldest word of each sentence.',
    technical: 'Planned with the convolutional stack (engine v2); not implemented in engine v1. Shown disabled with the reason.',
    signals: 'Planned — no fake downsampling is performed.',
    tips: []
  },
  dropout: {
    title: 'Dropout (planned)',
    category: 'layer',
    friendlyLabel: 'Practice handicap (later)',
    aliases: ['dropout', 'regularization', 'mask'],
    plain: 'A future block that randomly hides neurons during training so the network does not memorize.',
    analogy: 'Training a sports team with random players benched, so everyone learns every position.',
    technical: 'Requires train/eval-mode masking inside the verified engine; planned, not implemented. Weight decay on the optimizer is the supported regularizer today.',
    signals: 'Planned — no random masking is simulated.',
    tips: ['Use weight decay in the optimizer block meanwhile.']
  },

  /* ------------------------------------------------------ activations */
  relu: {
    title: 'ReLU',
    category: 'activation',
    friendlyLabel: 'Light switch',
    aliases: ['relu', 'activation', 'switch', 'nonlinear', 'rectifier'],
    plain: 'ReLU passes positive signals through and turns negative signals into zero. This non-linearity lets the network learn curves instead of straight lines.',
    analogy: 'A light switch: below zero, the light is off; at or above zero, the signal flows.',
    technical: 'f(x) = max(0, x); derivative 1 for x > 0 else 0. The default hidden activation.',
    signals: 'Watch the particles: negative activations are cut, positive ones continue at full strength. Dead units stay dark across epochs.',
    tips: ['Use it after every hidden Dense layer.', 'If many neurons stay dark, try Leaky ReLU.']
  },
  leaky_relu: {
    title: 'Leaky ReLU',
    category: 'activation',
    friendlyLabel: 'Dimmer switch',
    aliases: ['leaky', 'leakyrelu', 'negative slope', 'dimmer'],
    plain: 'Like the light switch, but negatives are not fully off — a tiny trickle still gets through so no neuron ever completely gives up.',
    analogy: 'A dimmer switch that never goes fully black.',
    technical: 'f(x) = x if x > 0 else 0.01·x. Keeps a small gradient for negative units.',
    signals: 'Negative signals leave as a faint trickle instead of vanishing.',
    tips: ['Good when you see many dead ReLU neurons.']
  },
  tanh: {
    title: 'Tanh',
    category: 'activation',
    friendlyLabel: 'Balance scale',
    aliases: ['tanh', 'hyperbolic', 'centered'],
    plain: 'Squashes any signal into the range −1…1, centered on zero.',
    analogy: 'A balance scale that always settles between fully left and fully right.',
    technical: 'f(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ), output (−1, 1). Useful for centered hidden representations.',
    signals: 'Particles arrive with mixed signs and leave bounded; the color mix shows the sign balance.',
    tips: ['Works well in small hidden layers; Sigmoid/Softmax belong at the output.']
  },
  sigmoid: {
    title: 'Sigmoid',
    category: 'activation',
    friendlyLabel: 'Answer squasher',
    aliases: ['sigmoid', 'probability', 'binary', 'squash'],
    plain: 'Sigmoid turns any number into a confidence between 0 and 1 — the output shape Binary Cross-Entropy expects for a yes/no decision.',
    analogy: 'A judge who always gives a verdict between “definitely no” and “definitely yes”.',
    technical: 'f(x) = 1/(1+e⁻ˣ), output (0, 1). Required output activation for Binary Cross-Entropy (size 2).',
    signals: 'The final particles split into two lanes whose brightness is the real class probability.',
    tips: ['Pair it with Binary Cross-Entropy.', 'Never stack two activations in a row — the second would override the first.']
  },
  softmax: {
    title: 'Softmax',
    category: 'activation',
    friendlyLabel: 'Vote splitter',
    aliases: ['softmax', 'multiclass', 'probabilities', 'vote'],
    plain: 'Turns a list of scores into percentages that add up to 100 %, so you can pick the winner.',
    analogy: 'Splitting a pie among candidates — the strongest score gets the biggest slice.',
    technical: 'Softmax over a rank-1 vector; outputs sum to 1. Required output activation for Categorical Cross-Entropy whose size equals the number of classes.',
    signals: 'Each output lane glows with its real probability; the brightest lane is the prediction.',
    tips: ['Pair it with Categorical Cross-Entropy.', 'The preceding Dense size must equal the class count.']
  },
  linear: {
    title: 'Linear',
    category: 'activation',
    friendlyLabel: 'Straight pass',
    aliases: ['linear', 'identity', 'regression', 'passthrough'],
    plain: 'Lets the number through unchanged — used when the answer is a quantity, not a class.',
    analogy: 'An open window: nothing is reshaped on the way through.',
    technical: 'Identity: f(x) = x, derivative 1. The natural output pair for MSE regression (usual size 1).',
    signals: 'Particles pass unchanged; the output gauge shows the raw predicted value.',
    tips: ['Use it for regression with MSE Loss.']
  },

  /* ------------------------------------------------------------ loss */
  bce_loss: {
    title: 'Binary Cross-Entropy',
    category: 'core',
    friendlyLabel: 'Report card (two choices)',
    aliases: ['bce', 'binary', 'crossentropy', 'loss', 'error'],
    plain: 'Binary Cross-Entropy scores how wrong the yes/no answer was. Training means making this number smaller. It needs a Sigmoid output with exactly two values.',
    analogy: 'A teacher grading a two-choice quiz: confident and wrong is punished hardest.',
    technical: 'BCE = −[y·log(p) + (1−y)·log(1−p)]. Requires a Sigmoid output activation of size 2. Attach an optimizer to "opt" and an Epoch Loop to "loop".',
    signals: 'Coral particles carry the error signal; the loss readout drops as backpropagation repairs the weights.',
    tips: ['Sigmoid + size 2 + BCE is the 0-vs-1 digit setup.', 'The two output probabilities are independent and need not sum to 1.']
  },
  cce_loss: {
    title: 'Categorical Cross-Entropy',
    category: 'core',
    friendlyLabel: 'Report card (many choices)',
    aliases: ['cce', 'categorical', 'multiclass', 'crossentropy', 'loss'],
    plain: 'Categorical Cross-Entropy scores a many-choice answer. It needs a Softmax output with one value per class.',
    analogy: 'A teacher grading a multiple-choice test where exactly one answer is right.',
    technical: 'CCE = −Σ y·log(p). Requires a Softmax output whose size equals the number of classes.',
    signals: 'The error particle travels back from the chosen class lane, pushing the wrong probabilities down.',
    tips: ['Softmax + class-sized Dense + CCE is the Iris setup.']
  },
  mse_loss: {
    title: 'MSE Loss',
    category: 'core',
    friendlyLabel: 'Distance score',
    aliases: ['mse', 'mean squared', 'regression', 'loss', 'distance'],
    plain: 'MSE Loss measures how far a predicted quantity is from the true quantity, and squares it so big misses count more.',
    analogy: 'Grading by how many meters a dart missed the bullseye — doubling the miss quadruples the penalty.',
    technical: 'MSE = mean((y − ŷ)²). Any flat output size; the natural pair is Linear with size 1. Regression datasets only.',
    signals: 'The loss readout is the distance; backpropagation reduces it every epoch.',
    tips: ['Pair with Linear output for regression.']
  },

  /* ------------------------------------------------------- optimizers */
  adam_optimizer: {
    title: 'Adam Optimizer',
    category: 'core',
    friendlyLabel: 'Smart coach',
    aliases: ['adam', 'optimizer', 'learning rate', 'training'],
    plain: 'Adam decides how big each learning step should be. It adapts its per-weight step sizes as training goes on, so it usually just works.',
    analogy: 'A coach who shortens the stride on tricky ground and lengthens it on smooth ground.',
    technical: 'Adaptive moments (β₁=0.9, β₂=0.999, ε=1e-8) over the gradient; the learning rate is tunable in the Inspector. Attach its "cfg" output to the loss "opt" socket.',
    signals: 'Particles pulse on every backward pass; the pulse cadence and the loss sparkline are the same real epoch stream.',
    tips: ['Default for almost every graph.', 'Too-large learning rate makes the loss jump around.']
  },
  sgd_optimizer: {
    title: 'SGD',
    category: 'core',
    friendlyLabel: 'Steady walker',
    aliases: ['sgd', 'stochastic', 'gradient descent', 'optimizer'],
    plain: 'SGD takes simple, steady steps in the downhill direction.',
    analogy: 'Walking downhill by always placing your foot in the steepest direction.',
    technical: 'W ← W − lr·∇W. Add momentum for smoother steps.',
    signals: 'The weight lines move a little on every epoch; the loss curve steps down steadily.',
    tips: ['Try momentum or Adam for faster convergence.']
  },
  momentum_optimizer: {
    title: 'Momentum',
    category: 'core',
    friendlyLabel: 'Rolling ball',
    aliases: ['momentum', 'velocity', 'optimizer', 'smooth'],
    plain: 'Momentum is SGD with inertia: it keeps rolling through small bumps.',
    analogy: 'A ball rolling downhill that does not stop at every pebble.',
    technical: 'SGD with velocity smoothing v = μv − lr·∇W; W ← W + v.',
    signals: 'Weight lines keep their direction across epochs instead of dithering.',
    tips: ['Good when the loss curve is jittery with plain SGD.']
  },
  rmsprop_optimizer: {
    title: 'RMSprop',
    category: 'core',
    friendlyLabel: 'Shock absorber',
    aliases: ['rmsprop', 'adaptive', 'optimizer'],
    plain: 'RMSprop calms each weight by how bumpy its own gradient history has been.',
    analogy: 'A shock absorber that softens only the roads that are actually bumpy.',
    technical: 'Scales learning by a running average of squared gradients; good for non-stationary objectives.',
    signals: 'The loss curve smooths out; individual weight lines stop thrashing.',
    tips: ['A solid choice for regression pipelines.']
  },

  /* ------------------------------------------------------------ logic */
  lr_schedule: {
    title: 'LR Scheduler',
    category: 'logic',
    friendlyLabel: 'Step ladder',
    aliases: ['schedule', 'learning rate', 'decay', 'step'],
    plain: 'LR Scheduler shrinks the learning rate over time so the network can settle instead of bouncing around the answer.',
    analogy: 'A runner who takes big strides early and short, careful steps near the finish line.',
    technical: 'Step decay: multiply the optimizer learning rate by `factor` every `every` epochs, applied live during a run. Attach "cfg" to the loss "schedule" socket.',
    signals: 'The loss curve flattens as the steps become smaller; no artificial smoothing is applied.',
    tips: ['Optional, but useful for long runs.']
  },
  epoch_loop: {
    title: 'Epoch Loop',
    category: 'logic',
    friendlyLabel: 'Practice timer',
    aliases: ['loop', 'epoch', 'batch', 'training', 'timer'],
    plain: 'Epoch Loop sets how long training lasts — either a fixed number of epochs or auto-stop when the held-out score stops improving.',
    analogy: 'A practice timer that rings when the player stops getting better.',
    technical: 'Run controller: auto-stop on a flat held-out loss for `patience` epochs, or a fixed `maxEpochs` budget; `batchSize` controls samples per update. Attach to the loss "loop" socket.',
    signals: 'Each epoch pulse is one real forward/backward pass over a batch.',
    tips: ['Auto-stop keeps the best held-out checkpoint.']
  },
  class_filter: {
    title: 'Class Filter',
    category: 'logic',
    friendlyLabel: 'Pick your classes',
    aliases: ['filter', 'classes', 'subset', 'keep', 'binary'],
    plain: 'Keeps only some classes and relabels them, so a many-class dataset can become a yes/no problem.',
    analogy: 'Sorting a mixed box of fruit and keeping only apples and oranges.',
    technical: 'Keeps the listed class indices, remaps the one-hot targets, and preserves the input shape. Classification datasets only.',
    signals: 'Class lanes outside the keep list disappear; the remaining lanes renumber from zero.',
    tips: ['“0,1” turns a 10-class corpus into the 0-vs-1 task.']
  },

  /* ----------------------------------------------------------- taps */
  metrics_gauge: {
    title: 'Loss / Accuracy Gauge',
    category: 'output',
    friendlyLabel: 'Scoreboard',
    aliases: ['gauge', 'metrics', 'loss', 'accuracy', 'scoreboard'],
    plain: 'Loss / Accuracy Gauge shows how well the network is doing: epoch, training loss, held-out loss and accuracy.',
    analogy: 'The scoreboard at a practice session.',
    technical: 'Taps the loss "monitor" output; numbers arrive from the worker per epoch and are never simulated.',
    signals: 'Reads the same epoch stream that drives the particles.',
    tips: ['Held-out loss is the honest one — training loss can be memorized.']
  },
  confusion_matrix: {
    title: 'Confusion Matrix',
    category: 'output',
    friendlyLabel: 'Mistake map',
    aliases: ['confusion', 'matrix', 'mistakes', 'per-class'],
    plain: 'Shows exactly which answers get confused with which, counted on the held-out set.',
    analogy: 'A seating chart of every wrong guess.',
    technical: 'Per-class hit counts computed from live per-sample predictions when a run completes; tied to the loss monitor socket.',
    signals: 'Counts are measured predictions, not estimates.',
    tips: ['A diagonal-heavy matrix means clean classes.']
  },
  prediction_view: {
    title: 'Prediction View',
    category: 'output',
    friendlyLabel: 'Try-it pad',
    aliases: ['prediction', 'probe', 'try', 'inference', 'sliders'],
    plain: 'Lets you give the trained network a new example and see its real answer.',
    analogy: 'A practice test with a fresh question after studying.',
    technical: 'Draws a 28×28 pad for image models or feature sliders for tables; each probe is a real forward pass through the worker.',
    signals: 'The output bars are the real activations returned by the engine.',
    tips: ['Unseen examples are the honest test.']
  },

  /* -------------------------------------------------------- concepts */
  wiring: {
    title: 'Wires',
    category: 'concept',
    friendlyLabel: 'Wires',
    aliases: ['wire', 'connect', 'link', 'edge'],
    plain: 'Wires carry numbers from one block to the next. The network only learns from blocks that are wired into the chain.',
    analogy: 'Jump-lead cables: no cable, no current.',
    technical: 'Typed sockets: tensor (data), config (optimizer/loop/schedule) and monitor (loss taps). Every input accepts one wire; the graph must stay acyclic. Right-click a wire to snip it; drag from a socket to empty space to grow the graph.',
    signals: 'Particles travel only along live wires; thickness is the real |weight| where weights exist.',
    tips: ['A wire with no particles is unconnected or its source failed to resolve.']
  },
  forward_pass: {
    title: 'Forward Pass',
    category: 'concept',
    friendlyLabel: 'Signal forward',
    aliases: ['forward', 'inference', 'signal', 'prediction'],
    plain: 'The network reads an example and produces an answer: numbers flow left to right through every wire.',
    analogy: 'A relay race where each runner passes the baton to the next.',
    technical: 'For each layer: z = W·x + b, a = f(z). Matrices multiply exactly as verified in the engine tests.',
    signals: 'The moving particles are the forward signal; their speed is one real epoch cadence.'
  },
  backprop: {
    title: 'Backpropagation',
    category: 'concept',
    friendlyLabel: 'Learning from mistakes',
    aliases: ['backprop', 'gradient', 'backward', 'learning', 'mistake'],
    plain: 'After seeing how wrong the answer was, the network walks backwards and nudges every weight in the direction that reduces the mistake.',
    analogy: 'A coach reviewing a lost match, player by player, telling each one what to change.',
    technical: 'The error signal δ is propagated from the loss through each layer using the chain rule; the optimizer applies W ← W − lr·∂L/∂W.',
    signals: 'On the backprop pass the particles run right-to-left; the weight line colors shift as weights change sign.'
  },
  loss: {
    title: 'Loss',
    category: 'concept',
    friendlyLabel: 'Score of wrongness',
    aliases: ['loss', 'error', 'objective', 'cost'],
    plain: 'One number that says how wrong the network is right now. Learning means making it smaller.',
    analogy: 'A golf score: lower is better.',
    technical: 'BCE, CCE or MSE, matched to the output activation and dataset type. The compiler refuses mismatched pairs with a real error code.',
    signals: 'The loss sparkline is the raw epoch stream from the worker.'
  },
  training: {
    title: 'Training',
    category: 'concept',
    friendlyLabel: 'Practice',
    aliases: ['train', 'practice', 'fit', 'run'],
    plain: 'Showing the network many examples, one batch at a time, until the score stops improving.',
    analogy: 'Practice drills: repetition plus feedback.',
    technical: 'The worker runs forward → loss → backward → optimizer per batch; auto-stop watches the held-out loss and keeps the best checkpoint.',
    signals: 'Epoch particles burst on every batch; metrics update from real measurements.'
  },
  inference: {
    title: 'Inference',
    category: 'concept',
    friendlyLabel: 'Asking the answer',
    aliases: ['inference', 'predict', 'probe', 'ask'],
    plain: 'Using the trained network on a brand-new example.',
    analogy: 'The real exam after the practice drills.',
    technical: 'A single forward pass with no weight updates; used by the prediction probe, the playground pad and ONLINE_STEP re-predictions.',
    signals: 'Only forward particles run; the weights do not change.'
  },
  quick_add: {
    title: 'Quick-add',
    category: 'concept',
    friendlyLabel: 'Grow the graph',
    aliases: ['quick', 'add', 'spawn', 'menu', 'drop'],
    plain: 'Drag a wire into empty space, and the studio offers only the blocks that fit — pick one and it is born already wired.',
    analogy: 'Pulling a cable into open air and having the right gadget appear on the end.',
    technical: 'Candidates are filtered by socket kind/direction and ranked by clone-compiling the real graph; selecting a block performs addNode + connect (with Smart-Assist rank repair) in one gesture.',
    signals: 'A dashed pending wire follows the gesture until the block lands.'
  },
  snip: {
    title: 'Snip a wire',
    category: 'concept',
    friendlyLabel: 'Cut a wire',
    aliases: ['snip', 'cut', 'break', 'right-click', 'disconnect'],
    plain: 'Right-click a wire to cut it. Cutting is safe: one undo brings it back.',
    analogy: 'Scissors on a jumper cable — the current stops, the ends stay put.',
    technical: 'Break runs a real `graph.disconnect` after a 180 ms collapse animation, clears the compiled plan, and re-runs shape inference. The Advanced menu previews the exact post-break error codes.',
    signals: 'The wire fades and collapses; downstream blocks immediately show their real validation state.'
  }
};

/* ------------------------------------------------------------------ helpers */

const FALLBACK = {
  title: 'Block',
  category: 'logic',
  friendlyLabel: 'Block',
  aliases: [],
  plain: 'A building block of the network.',
  analogy: null,
  technical: 'See the block summary in the palette.',
  signals: 'Real engine data only.',
  tips: []
};

/** Look an explanation up by node type or concept key; never throws, never returns a fake. */
export function explain(key) {
  const entry = EXPLANATIONS[key];
  if (entry) {
    const def = NODE_TYPES[key];
    // Curriculum copy may add a metaphor, but a block is always called by its registry label.
    return def ? { ...entry, title: def.label, friendlyLabel: def.label } : entry;
  }
  const def = NODE_TYPES[key];
  if (def) {
    return {
      ...FALLBACK,
      title: def.label,
      category: def.category,
      friendlyLabel: def.label,
      aliases: [key, def.label.toLowerCase()],
      plain: def.summary || FALLBACK.plain,
      technical: def.reason || def.summary || FALLBACK.technical
    };
  }
  return { ...FALLBACK, title: String(key || 'Block'), aliases: [] };
}

/**
 * Display label used by the Tier-1 quick menu and mission cards.
 *
 * Node blocks always use the exact registry label (`Dataset Source`, `Dense Layer`, `ReLU`, …)
 * so the tutorial and the canvas name the same thing. The beginner metaphor belongs in the
 * `analogy` line, not in the block title.
 */
export function friendlyLabel(type) {
  if (NODE_TYPES[type]) return NODE_TYPES[type].label;
  const entry = EXPLANATIONS[type];
  return entry && entry.friendlyLabel ? entry.friendlyLabel : String(type || 'Block');
}

/** Category label for grouping. */
export function categoryLabel(type) {
  const entry = EXPLANATIONS[type];
  return entry && entry.category ? entry.category : 'logic';
}

/** One-line technical summary for the Advanced quick menu (sockets + params). */
export function technicalLine(type) {
  const def = NODE_TYPES[type];
  if (!def) return explain(type).technical;
  const ins = (def.inputs || []).map((s) => `${s.name}:${s.kind}`).join(', ') || '—';
  const outs = (def.outputs || []).map((s) => `${s.name}:${s.kind}`).join(', ') || '—';
  const params = Object.entries(def.params || {}).map(([name, spec]) => `${name}=${spec.default}`).join(' ');
  return `${ins} → ${outs}${params ? ` · ${params}` : ''}`;
}

/** Search terms for the quick-add filter. */
export function searchTerms(entryOrType) {
  const entry = typeof entryOrType === 'string' ? explain(entryOrType) : (entryOrType || FALLBACK);
  return [entry.title, entry.friendlyLabel, ...(entry.aliases || [])]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

/** Keys that must exist for the tutorial content to be considered complete. */
export function catalogCoverage() {
  const required = [
    ...Object.keys(NODE_TYPES).filter((type) => NODE_TYPES[type].supported !== false),
    'wiring', 'forward_pass', 'backprop', 'loss', 'training', 'inference', 'quick_add', 'snip'
  ];
  const missing = required.filter((key) => !EXPLANATIONS[key]);
  return { required, missing };
}

export { ACTIVATION_NODE_TYPES, LOSS_NODE_TYPES, OPTIMIZER_NODE_TYPES };
