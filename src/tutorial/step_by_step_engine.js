/**
 * Node Studio — step-by-step build mission engine.
 *
 * The curriculum in `interactive_tour.js` spotlights existing UI; this module makes the learner
 * *construct* the network block by block. Every gate is a real predicate over the live
 * `GraphState` (or a real event the page reports), so a mission cannot advance on a pretending
 * click: `wireExists('source','dense')` is only true when that connection is actually in the
 * graph the compiler runs on.
 *
 * Two layers, deliberately separated (mirroring `tutorial_engine.js`):
 *   - `MissionRunner`: the pure step machine — gate evaluation, notify/check, next/prev/skip,
 *     progress. No DOM; unit-tested in `tests/test_step_by_step.js`.
 *   - `StepByStepEngine`: the presenter — spotlight cutout, pulsing dropzone ring over the real
 *     target element, explanation card (plain / analogy / technical from `explanations_catalog`),
 *     gated Next button, its own `×`, and a MenuManager overlay registration handled by the page.
 *
 * File: src/tutorial/step_by_step_engine.js
 */

import { getNodeType } from '../graph/node_types.js';
import { explain } from './explanations_catalog.js';
import { inlineMarkdown } from './inline_text.js';
import {
  expandRect,
  intersectionArea,
  isBackdropTarget,
  measureChrome,
  normalizeRect,
  placeFloatingPanel,
  STUDIO_CHROME_SELECTORS
} from './overlay_layout.js';

/* ------------------------------------------------------------ validators */

function graphNodes(ctx) {
  return ctx && ctx.graph && typeof ctx.graph.nodeList === 'function' ? ctx.graph.nodeList() : [];
}

function looseEquals(a, b) {
  if (a === b) return true;
  const na = Number(a);
  const nb = Number(b);
  if (a !== '' && b !== '' && Number.isFinite(na) && Number.isFinite(nb)) return na === nb;
  return String(a) === String(b);
}

function typeList(params) {
  if (Array.isArray(params.types)) return params.types;
  if (params.type) return [params.type];
  return [];
}

/**
 * Read the page's real action journal. `ctx.actions` is an array of
 * `{ id, seq, at, detail }` records produced by `recordAction()`; `ctx.missionEvents`
 * remains supported for embedders/tests that only want the runner's own event counts.
 */
function actionEntries(ctx) {
  if (!ctx) return [];
  const entries = [];
  if (Array.isArray(ctx.actions)) entries.push(...ctx.actions);
  if (Array.isArray(ctx.actionLog)) entries.push(...ctx.actionLog);
  for (const key of ['actions', 'actionLog', 'events', 'missionEvents']) {
    const value = ctx[key];
    if (value instanceof Map) {
      for (const [id, count] of value.entries()) {
        for (let i = 0; i < count; i += 1) entries.push({ id });
      }
    }
  }
  return entries;
}

function actionWasSeen(ctx, params) {
  const wanted = [].concat(params.id || params.ids || []).filter(Boolean);
  if (wanted.length === 0) return null;
  const entries = actionEntries(ctx);
  const found = wanted.find((id) => entries.some((entry) => entry && entry.id === id));
  return found || null;
}

function pairsOf(graph) {
  const pairs = [];
  for (const conn of graph.connections.values()) {
    const from = graph.getNode(conn.from.nodeId);
    const to = graph.getNode(conn.to.nodeId);
    if (from && to) pairs.push({ conn, from, to });
  }
  return pairs;
}

function labelOf(type) {
  const def = getNodeType(type);
  return def ? def.label : type;
}

/**
 * `ValidationRule`: `{ evaluate(params, ctx) -> { ok, reason }, describe(params) -> string }`.
 * Every rule reads real graph/state facts; none of them consults a scripted flag.
 */
export const VALIDATORS = {
  modeIs: {
    evaluate(params, ctx) {
      const ok = ctx.mode === params.mode;
      return { ok, reason: ok ? null : `Switch to ${params.label || params.mode}.` };
    },
    describe(params) {
      return `mode is “${params.mode}”`;
    }
  },
  hasNode: {
    evaluate(params, ctx) {
      const wanted = typeList(params);
      const min = Number.isFinite(params.min) ? params.min : 1;
      const count = graphNodes(ctx).filter((node) => wanted.includes(node.type)).length;
      return {
        ok: count >= min,
        reason: count >= min ? null : `Place the ${wanted.map(labelOf).join(' or ')} block (${count}/${min}).`
      };
    },
    describe(params) {
      return `graph contains ${typeList(params).map(labelOf).join(' or ')}`;
    }
  },
  nodeCount: {
    evaluate(params, ctx) {
      const wanted = typeList(params);
      const min = Number.isFinite(params.min) ? params.min : 1;
      const nodes = graphNodes(ctx);
      const count = wanted.length === 0 ? nodes.length : nodes.filter((node) => wanted.includes(node.type)).length;
      return {
        ok: count >= min,
        reason: count >= min ? null : `The mission needs at least ${min} ${wanted.length ? wanted.map(labelOf).join('/') : 'block'}(s) — currently ${count}.`
      };
    },
    describe(params) {
      return `at least ${params.min} ${typeList(params).map(labelOf).join('/') || 'block'}(s) on the canvas`;
    }
  },
  wireExists: {
    evaluate(params, ctx) {
      if (!ctx.graph) return { ok: false, reason: 'No graph available.' };
      const found = pairsOf(ctx.graph).some(({ conn, from, to }) => {
        if (params.fromType && from.type !== params.fromType) return false;
        if (params.toType && to.type !== params.toType) return false;
        if (params.fromSocket && conn.from.socket !== params.fromSocket) return false;
        if (params.toSocket && conn.to.socket !== params.toSocket) return false;
        return true;
      });
      return {
        ok: found,
        reason: found ? null : `Wire ${labelOf(params.fromType)} → ${labelOf(params.toType)} is missing.`
      };
    },
    describe(params) {
      return `a wire from ${labelOf(params.fromType)} to ${labelOf(params.toType)}`;
    }
  },
  wireBroken: {
    evaluate(params, ctx) {
      if (!ctx.graph) return { ok: false, reason: 'No graph available.' };
      const stillThere = pairsOf(ctx.graph).some(({ conn, from, to }) => {
        if (params.fromType && from.type !== params.fromType) return false;
        if (params.toType && to.type !== params.toType) return false;
        if (params.fromSocket && conn.from.socket !== params.fromSocket) return false;
        if (params.toSocket && conn.to.socket !== params.toSocket) return false;
        return true;
      });
      return {
        ok: !stillThere,
        reason: stillThere ? `Cut the wire ${labelOf(params.fromType)} → ${labelOf(params.toType)} first.` : null
      };
    },
    describe(params) {
      return `no wire between ${labelOf(params.fromType)} and ${labelOf(params.toType)}`;
    }
  },
  pathExists: {
    evaluate(params, ctx) {
      if (!ctx.graph) return { ok: false, reason: 'No graph available.' };
      const sources = graphNodes(ctx).filter((node) => node.type === params.fromType);
      const targets = graphNodes(ctx).filter((node) => node.type === params.toType);
      const found = sources.some((source) => targets.some((target) => ctx.graph.canReach(source.id, target.id)));
      return {
        ok: found,
        reason: found ? null : `The signal must flow from ${labelOf(params.fromType)} to ${labelOf(params.toType)} again.`
      };
    },
    describe(params) {
      return `a connected path ${labelOf(params.fromType)} → ${labelOf(params.toType)}`;
    }
  },
  socketWired: {
    evaluate(params, ctx) {
      if (!ctx.graph) return { ok: false, reason: 'No graph available.' };
      const nodes = graphNodes(ctx).filter((node) => node.type === params.nodeType);
      const found = nodes.some((node) => {
        const conn = ctx.graph.incoming(node.id, params.socket);
        if (!conn) return false;
        if (!params.fromType) return true;
        const source = ctx.graph.getNode(conn.from.nodeId);
        return source && source.type === params.fromType;
      });
      return {
        ok: found,
        reason: found ? null : `Attach a block to the “${params.socket}” socket of ${labelOf(params.nodeType)}.`
      };
    },
    describe(params) {
      return `the ${labelOf(params.nodeType)} “${params.socket}” socket is wired`;
    }
  },
  feedingParam: {
    evaluate(params, ctx) {
      if (!ctx.graph) return { ok: false, reason: 'No graph available.' };
      const targets = graphNodes(ctx).filter((node) => node.type === params.targetType);
      const found = targets.some((target) => {
        const conn = ctx.graph.incoming(target.id, params.toSocket || 'in');
        if (!conn) return false;
        const source = ctx.graph.getNode(conn.from.nodeId);
        if (!source || source.type !== params.sourceType) return false;
        return looseEquals(source.params[params.param], params.value);
      });
      return {
        ok: found,
        reason: found ? null : `Set ${labelOf(params.sourceType)}.${params.param} = ${params.value} on the block that feeds ${labelOf(params.targetType)}.`
      };
    },
    describe(params) {
      return `${labelOf(params.sourceType)} feeding ${labelOf(params.targetType)} has ${params.param} = ${params.value}`;
    }
  },
  paramEquals: {
    evaluate(params, ctx) {
      const wanted = typeList(params);
      const nodes = graphNodes(ctx).filter((node) => wanted.includes(node.type));
      const found = nodes.some((node) => looseEquals(node.params[params.param], params.value));
      const detail = params.valueLabel || params.value;
      return {
        ok: found,
        reason: found ? null : `Set ${wanted.map(labelOf).join('/')} “${params.param}” to ${detail}.`
      };
    },
    describe(params) {
      return `${typeList(params).map(labelOf).join('/')}.${params.param} = ${params.valueLabel || params.value}`;
    }
  },
  compiles: {
    evaluate(params, ctx) {
      const ok = Boolean(ctx.plan && ctx.plan.ok);
      return { ok, reason: ok ? null : 'The graph must compile without errors.' };
    },
    describe() {
      return 'the graph compiles';
    }
  },
  /**
   * Persistent user-action gate. Use this for one-shot/transient events such as pressing Run:
   * the event stays true after training completes, unlike `ctx.running`. The journal is written
   * by the page on every real gesture, and the runner also seeds it from `missionEvents`.
   */
  actionSeen: {
    evaluate(params, ctx) {
      const found = actionWasSeen(ctx, params);
      const ok = found !== null;
      const label = params.label || [].concat(params.id || params.ids || []).join(' or ');
      return {
        ok,
        reason: ok ? null : (params.reason || `Perform the real action first: ${label}.`)
      };
    },
    describe(params) {
      return `user action “${[].concat(params.id || params.ids || []).join(' or ')}” was logged`;
    }
  },
  /** Kept for backwards compatibility; new lessons should prefer `actionSeen` for one-shot events. */
  running: {
    evaluate(params, ctx) {
      const ok = ctx.running === true;
      return { ok, reason: ok ? null : 'Start a real training run.' };
    },
    describe() {
      return 'training is running';
    }
  }
};

/** Evaluate a validation spec: object, rule string, or an array of both (all must pass). */
export function evaluateValidation(validate, ctx) {
  if (!validate) return { ok: true, reason: null };
  if (Array.isArray(validate)) {
    for (const entry of validate) {
      const result = evaluateValidation(entry, ctx);
      if (!result.ok) return result;
    }
    return { ok: true, reason: null };
  }
  const spec = typeof validate === 'string' ? { rule: validate } : validate;
  const validator = VALIDATORS[spec.rule];
  if (!validator) return { ok: false, reason: `Unknown validation rule "${spec.rule}".` };
  try {
    const result = validator.evaluate(spec.params || {}, ctx || {});
    return {
      ok: Boolean(result.ok),
      reason: result.ok ? null : (result.reason || validator.describe(spec.params || {}))
    };
  } catch (err) {
    return { ok: false, reason: `Validation failed: ${err.message}` };
  }
}

/* -------------------------------------------------------------- missions */

const W = {
  source: { type: 'source' },
  flatten: { type: 'flatten' }
};

export const MISSIONS = {
  /* ------------------------------------------------------- Mission 1 (tier 1) */
  build_first_brain: {
    id: 'build_first_brain',
    tier: 1,
    title: 'Lesson 7 — Build an XOR Network from Scratch',
    goal: 'Place every block and wire yourself, then train a real network that solves XOR.',
    blurb: 'The canvas starts empty. Each step unlocks only after the real graph matches the instruction.',
    suppressAssist: true,
    steps: [
      {
        id: 'fb-start',
        title: 'Build XOR from a blank canvas',
        concept: 'wiring',
        text: 'XOR is a small logic problem: 00 → 0, 01 → 1, 10 → 1, 11 → 0. A network needs a hidden layer to solve it. You will add each block in order. The green **Next** button unlocks only when the real graph matches the instruction.',
        target: '#studio-canvas',
        setup: { id: 'prepare_mission', mode: 'assist' }
      },
      {
        id: 'fb-source',
        title: '1. Add a Dataset Source',
        concept: 'source',
        text: 'In the left **Data** palette, click or drag **Dataset Source** onto the canvas. This block supplies the training examples. There is no data inside the block yet; the next step chooses the XOR corpus.',
        target: '#palette-source',
        action: {
          id: 'node_added',
          hint: 'Click or drag Dataset Source.',
          validate: { rule: 'hasNode', params: W.source }
        }
      },
      {
        id: 'fb-xor',
        title: '2. Select the XOR dataset',
        concept: 'training',
        text: 'The new **Dataset Source** is selected. In the **Inspector** on the right, set **Dataset** to **XOR (2 features)**. The block will emit two input numbers and one binary target.',
        target: '#inspector-body',
        action: {
          id: 'param_changed',
          hint: 'Select Dataset Source, open the Inspector, and choose XOR (2 features).',
          validate: { rule: 'paramEquals', params: { type: 'source', param: 'datasetId', value: 'xor', valueLabel: '“XOR (2 features)”' } }
        }
      },
      {
        id: 'fb-dense',
        title: '3. Add the first Dense Layer',
        concept: 'quick_add',
        text: 'Drag from the **out** socket on the right side of **Dataset Source** into empty canvas. A **Place Node** menu appears. Choose **Dense Layer** — it is placed and wired automatically. A Dense Layer is a set of neurons with learnable weights.',
        target: { nodeType: 'source', socket: 'out', direction: 'out' },
        action: {
          id: 'quick_node_added',
          hint: 'Drag Dataset Source output into empty space, then choose Dense Layer.',
          validate: { rule: 'wireExists', params: { fromType: 'source', toType: 'dense' } }
        }
      },
      {
        id: 'fb-relu',
        title: '4. Add the ReLU activation',
        concept: 'relu',
        text: 'Drag from the **out** socket of the first **Dense Layer** into empty space and choose **ReLU**. ReLU passes positive values through and turns negative values into zero. Without a nonlinear activation, stacking layers would still behave like one linear layer.',
        target: { nodeType: 'dense', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from the Dense Layer output, then choose ReLU.',
          validate: { rule: 'wireExists', params: { fromType: 'dense', toType: 'relu' } }
        }
      },
      {
        id: 'fb-dense2',
        title: '5. Add the second Dense Layer',
        concept: 'dense',
        text: 'Drag from the **out** socket of **ReLU** into empty space and choose **Dense Layer**. This layer combines the features from the first Dense Layer into the final decision.',
        target: { nodeType: 'relu', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from ReLU, then choose another Dense Layer.',
          validate: [
            { rule: 'wireExists', params: { fromType: 'relu', toType: 'dense' } },
            { rule: 'nodeCount', params: { type: 'dense', min: 2 } }
          ]
        }
      },
      {
        id: 'fb-sigmoid',
        title: '6. Add the Sigmoid output',
        concept: 'sigmoid',
        text: 'Drag from the **out** socket of the second **Dense Layer** into empty space and choose **Sigmoid**. Sigmoid maps each output score to a probability between 0 and 1, which is what Binary Cross-Entropy expects.',
        target: { nodeType: 'dense', socket: 'out', direction: 'out', pick: 'newest' },
        action: {
          hint: 'Drag from the last Dense Layer output, then choose Sigmoid.',
          validate: { rule: 'wireExists', params: { fromType: 'dense', toType: 'sigmoid' } }
        }
      },
      {
        id: 'fb-size',
        title: '7. Set the output Dense Layer to 2 neurons',
        concept: 'dense',
        text: 'The Dense Layer immediately before **Sigmoid** must emit one score per class. Select that **Dense Layer**, then in the Inspector set **Neurons** to **2**. The first score means class 0; the second means class 1.',
        target: '#inspector-body',
        action: {
          id: 'param_changed',
          hint: 'Select the Dense Layer feeding Sigmoid and set Neurons = 2.',
          validate: { rule: 'feedingParam', params: { sourceType: 'dense', targetType: 'sigmoid', param: 'size', value: 2 } }
        }
      },
      {
        id: 'fb-loss',
        title: '8. Add Binary Cross-Entropy',
        concept: 'bce_loss',
        text: 'Drag from the **out** socket of **Sigmoid** into empty space and choose **Binary Cross-Entropy**. This loss compares the predicted probabilities with the correct 0/1 label. Training means minimizing this number.',
        target: { nodeType: 'sigmoid', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from Sigmoid, then choose Binary Cross-Entropy.',
          validate: { rule: 'wireExists', params: { fromType: 'sigmoid', toType: 'bce_loss' } }
        }
      },
      {
        id: 'fb-opt',
        title: '9. Attach the Adam optimizer',
        concept: 'adam_optimizer',
        text: 'The **Binary Cross-Entropy** block has an **opt** input on its left for the optimizer. Drag from **opt** into empty space and choose **Adam**. Adam computes the weight update from the real gradients and keeps a per-weight adaptive step size.',
        target: { nodeType: 'bce_loss', socket: 'opt', direction: 'in' },
        action: {
          hint: 'Drag from the loss opt input, then choose Adam.',
          validate: { rule: 'socketWired', params: { nodeType: 'bce_loss', socket: 'opt' } }
        }
      },
      {
        id: 'fb-loop',
        title: '10. Attach the Epoch Loop',
        concept: 'epoch_loop',
        text: 'Drag from the **loop** input on the **Binary Cross-Entropy** block into empty space and choose **Epoch Loop**. It sets the epoch budget, batch size and early-stopping rule that the training worker follows.',
        target: { nodeType: 'bce_loss', socket: 'loop', direction: 'in' },
        action: {
          hint: 'Drag from the loss loop input, then choose Epoch Loop.',
          validate: { rule: 'socketWired', params: { nodeType: 'bce_loss', socket: 'loop' } }
        }
      },
      {
        id: 'fb-run',
        title: '11. Press Run',
        concept: 'training',
        text: 'Press **Run**. The worker performs real forward passes, backpropagation and Adam updates on the XOR data. This step is recorded from your Run action, so it stays complete even when a short run finishes before you press Next.',
        target: '#btn-run',
        action: {
          id: 'training_started',
          hint: 'Press Run. The step is remembered even if a short run finishes first.',
          validate: { rule: 'actionSeen', params: { id: 'training_started', label: 'pressing Run' } }
        }
      },
      {
        id: 'fb-done',
        title: '12. Read the results',
        concept: 'inference',
        text: 'In the **Training** panel, check the training loss, held-out loss and accuracy. Then open **Prediction View**, enter the four XOR input pairs and compare the predicted class with the expected answer. Low loss and high accuracy mean the graph really learned XOR.',
        target: '#panel-metrics'
      }
    ]
  },

  /* ------------------------------------------------------- Mission 2 (tier 2) */
  build_digit_classifier: {
    id: 'build_digit_classifier',
    tier: 2,
    title: 'Lesson 8 — Build the MNIST Digit Classifier',
    goal: 'Build the real MNIST 0-vs-1 pipeline block by block, then break and repair a wire.',
    blurb: 'Smart Assist handles shape repairs, but every block and every wire is placed by you.',
    suppressAssist: true,
    steps: [
      {
        id: 'db-start',
        title: '1. Switch to Beginner · Assist',
        concept: 'wiring',
        text: 'Press **Beginner · Assist**. In this tier, dropping a block can auto-wire it, and **Fix automatically** can insert shape repairs. The preview still runs the real compiler, so it cannot hide an invalid graph.',
        target: '#btn-mode-assist',
        setup: { id: 'prepare_mission', mode: 'assist' },
        action: {
          id: 'mode_assist',
          hint: 'Press Beginner · Assist.',
          validate: { rule: 'modeIs', params: { mode: 'assist', label: 'Beginner · Assist' } }
        }
      },
      {
        id: 'db-source',
        title: '2. Add a Dataset Source',
        concept: 'source',
        text: 'Click or drag **Dataset Source** from the Data palette. It starts on the real MNIST 0-vs-1 corpus: 28×28 handwritten digit images, each labelled 0 or 1.',
        target: '#palette-source',
        action: {
          id: 'node_added',
          hint: 'Click or drag Dataset Source.',
          validate: { rule: 'hasNode', params: W.source }
        }
      },
      {
        id: 'db-dense',
        title: '3. Add a Dense Layer and let Assist insert Flatten',
        concept: 'flatten',
        text: 'Drag from the **out** socket of **Dataset Source** into empty space and choose **Dense Layer**. Because the source emits a 28×28 image grid, Smart Assist inserts **Flatten** (28×28 → 784) between them. Check that both wires exist: Dataset Source → Flatten → Dense Layer.',
        target: { nodeType: 'source', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from Dataset Source, choose Dense Layer, and let Assist insert Flatten.',
          validate: [
            { rule: 'hasNode', params: W.flatten },
            { rule: 'wireExists', params: { fromType: 'source', toType: 'flatten' } },
            { rule: 'wireExists', params: { fromType: 'flatten', toType: 'dense' } }
          ]
        }
      },
      {
        id: 'db-relu',
        title: '4. Add ReLU',
        concept: 'relu',
        text: 'Drag from the **out** socket of the **Dense Layer** into empty space and choose **ReLU**. ReLU keeps positive activations and sets negative ones to zero, which lets the network learn non-linear decision boundaries.',
        target: { nodeType: 'dense', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from the Dense Layer output, then choose ReLU.',
          validate: { rule: 'wireExists', params: { fromType: 'dense', toType: 'relu' } }
        }
      },
      {
        id: 'db-dense2',
        title: '5. Add the second Dense Layer',
        concept: 'dense',
        text: 'Drag from the **out** socket of **ReLU** into empty space and choose **Dense Layer**. The network now has two learned layers. The first creates features; the second combines them into class scores.',
        target: { nodeType: 'relu', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from ReLU, then choose another Dense Layer.',
          validate: [
            { rule: 'wireExists', params: { fromType: 'relu', toType: 'dense' } },
            { rule: 'nodeCount', params: { type: 'dense', min: 2 } }
          ]
        }
      },
      {
        id: 'db-sigmoid',
        title: '6. Add Sigmoid',
        concept: 'sigmoid',
        text: 'Drag from the **out** socket of the last **Dense Layer** into empty space and choose **Sigmoid**. It turns each class score into a probability between 0 and 1.',
        target: { nodeType: 'dense', socket: 'out', direction: 'out', pick: 'newest' },
        action: {
          hint: 'Drag from the last Dense Layer output, then choose Sigmoid.',
          validate: { rule: 'wireExists', params: { fromType: 'dense', toType: 'sigmoid' } }
        }
      },
      {
        id: 'db-size',
        title: '7. Set the output Dense Layer to 2 neurons',
        concept: 'dense',
        text: 'The **Dense Layer** that feeds **Sigmoid** needs one output score per class. Select it and set **Neurons** to **2** in the Inspector. The labels are 0 and 1, so two outputs are required.',
        target: '#inspector-body',
        action: {
          id: 'param_changed',
          hint: 'Select the Dense Layer feeding Sigmoid and set Neurons = 2.',
          validate: { rule: 'feedingParam', params: { sourceType: 'dense', targetType: 'sigmoid', param: 'size', value: 2 } }
        }
      },
      {
        id: 'db-loss',
        title: '8. Add Binary Cross-Entropy',
        concept: 'bce_loss',
        text: 'Drag from the **out** socket of **Sigmoid** into empty space and choose **Binary Cross-Entropy**. It compares the predicted probability for each class with the correct 0/1 target.',
        target: { nodeType: 'sigmoid', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from Sigmoid, then choose Binary Cross-Entropy.',
          validate: { rule: 'wireExists', params: { fromType: 'sigmoid', toType: 'bce_loss' } }
        }
      },
      {
        id: 'db-snip',
        title: '9. Break the ReLU → Dense Layer wire',
        concept: 'snip',
        text: 'Right-click the wire from **ReLU** to the second **Dense Layer** and choose **Break connection**. The wire disappears and shape inference runs again. This is safe: **Ctrl+Z** restores the graph.',
        target: { nodeType: 'relu', socket: 'out', direction: 'out' },
        action: {
          hint: 'Right-click the ReLU → Dense Layer wire and choose Break connection.',
          validate: [
            { rule: 'wireBroken', params: { fromType: 'relu', toType: 'dense' } },
            { rule: 'pathExists', params: { fromType: 'dense', toType: 'sigmoid' } }
          ]
        }
      },
      {
        id: 'db-rewire',
        title: '10. Reconnect ReLU → Dense Layer by hand',
        concept: 'wiring',
        text: 'Drag from the **out** socket of **ReLU** and release exactly on the **in** socket of the second **Dense Layer**. This is a direct socket connection, not the quick-add menu.',
        target: { nodeType: 'relu', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag ReLU output onto the second Dense Layer input.',
          validate: { rule: 'pathExists', params: { fromType: 'relu', toType: 'sigmoid' } }
        }
      },
      {
        id: 'db-opt',
        title: '11. Attach the Adam optimizer',
        concept: 'adam_optimizer',
        text: 'Drag from the **opt** input on **Binary Cross-Entropy** into empty space and choose **Adam**. Adam uses the real gradients and adaptive per-weight step sizes to update the network.',
        target: { nodeType: 'bce_loss', socket: 'opt', direction: 'in' },
        action: {
          hint: 'Drag from the loss opt input, then choose Adam.',
          validate: { rule: 'socketWired', params: { nodeType: 'bce_loss', socket: 'opt' } }
        }
      },
      {
        id: 'db-loop',
        title: '12. Attach the Epoch Loop',
        concept: 'epoch_loop',
        text: 'Drag from the **loop** input on **Binary Cross-Entropy** into empty space and choose **Epoch Loop**. Auto-stop watches held-out loss and keeps the best checkpoint; a fixed budget runs a set number of epochs.',
        target: { nodeType: 'bce_loss', socket: 'loop', direction: 'in' },
        action: {
          hint: 'Drag from the loss loop input, then choose Epoch Loop.',
          validate: { rule: 'socketWired', params: { nodeType: 'bce_loss', socket: 'loop' } }
        }
      },
      {
        id: 'db-run',
        title: '13. Press Run',
        concept: 'training',
        text: 'Press **Run**. The worker performs real forward passes, backpropagation and Adam updates on MNIST. The recorded Run action keeps this step complete even if training stops before you click Next.',
        target: '#btn-run',
        action: {
          id: 'training_started',
          hint: 'Press Run. The step is remembered even if a short run finishes first.',
          validate: { rule: 'actionSeen', params: { id: 'training_started', label: 'pressing Run' } }
        }
      },
      {
        id: 'db-done',
        title: '14. Check the classifier',
        concept: 'inference',
        text: 'Read the held-out accuracy in the **Training** panel and inspect the **Confusion Matrix** for false positives and false negatives. Then open **Prediction View** and test a drawing. You built the graph and repaired a real wire.',
        target: '#panel-metrics'
      }
    ]
  },

  /* ------------------------------------------------------- Mission 3 (tier 3) */
  build_custom_pipeline: {
    id: 'build_custom_pipeline',
    tier: 3,
    title: 'Lesson 9 — Advanced Regression Pipeline',
    goal: 'Build a manual regression graph, break and repair a loss wire, and pair loss, optimizer, schedule and loop.',
    blurb: 'No automatic repairs. Every socket is your decision; the compiler reports real error codes when a pairing is wrong.',
    suppressAssist: true,
    steps: [
      {
        id: 'cp-mode',
        title: '1. Switch to Advanced · Manual',
        concept: 'wiring',
        text: 'Press **Advanced · Manual**. Automatic repairs are disabled: sockets connect exactly where you drop them, and every invalid connection produces a real validation code.',
        target: '#btn-mode-manual',
        action: {
          id: 'mode_manual',
          hint: 'Press Advanced · Manual.',
          validate: { rule: 'modeIs', params: { mode: 'manual', label: 'Advanced · Manual' } }
        }
      },
      {
        id: 'cp-start',
        title: '2. Start with a blank canvas',
        concept: 'wiring',
        text: 'The canvas is cleared for this lesson. In Manual mode nothing is inserted for you. Build the graph from the dataset to the loss, then attach the optimizer, schedule, loop and monitor.',
        target: '#studio-canvas',
        setup: { id: 'prepare_mission', mode: 'manual' }
      },
      {
        id: 'cp-source',
        title: '3. Add a Dataset Source',
        concept: 'source',
        text: 'Click or drag **Dataset Source** from the Data palette. The next step changes it from classification to regression data.',
        target: '#palette-source',
        action: {
          id: 'node_added',
          hint: 'Click or drag Dataset Source.',
          validate: { rule: 'hasNode', params: W.source }
        }
      },
      {
        id: 'cp-dataset',
        title: '4. Select the Non-linear Regression dataset',
        concept: 'training',
        text: 'With **Dataset Source** selected, open the Inspector and set **Dataset** to **Non-linear Regression (1 feature)**. The target is a continuous number, not a class label.',
        target: '#inspector-body',
        action: {
          id: 'param_changed',
          hint: 'Select Dataset Source and choose Non-linear Regression (1 feature).',
          validate: { rule: 'paramEquals', params: { type: 'source', param: 'datasetId', value: 'linear_regression', valueLabel: '“Non-linear Regression”' } }
        }
      },
      {
        id: 'cp-flatten',
        title: '5. Add Flatten',
        concept: 'flatten',
        text: 'Drag from the **out** socket of **Dataset Source** into empty space and choose **Flatten**. Manual mode will not insert it for you. Flatten turns the source tensor into a feature vector the Dense Layers can consume.',
        target: { nodeType: 'source', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from Dataset Source, then choose Flatten.',
          validate: { rule: 'wireExists', params: { fromType: 'source', toType: 'flatten' } }
        }
      },
      {
        id: 'cp-dense',
        title: '6. Add the first Dense Layer',
        concept: 'dense',
        text: 'Drag from the **out** socket of **Flatten** and choose **Dense Layer**. The default 16 neurons mix the input feature into a learned representation.',
        target: { nodeType: 'flatten', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from Flatten, then choose Dense Layer.',
          validate: { rule: 'wireExists', params: { fromType: 'flatten', toType: 'dense' } }
        }
      },
      {
        id: 'cp-relu',
        title: '7. Add ReLU',
        concept: 'relu',
        text: 'Drag from the **out** socket of **Dense Layer** and choose **ReLU**. Regression still needs a non-linear activation; otherwise the stacked layers collapse into one linear function.',
        target: { nodeType: 'dense', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from the Dense Layer output, then choose ReLU.',
          validate: { rule: 'wireExists', params: { fromType: 'dense', toType: 'relu' } }
        }
      },
      {
        id: 'cp-dense2',
        title: '8. Add the second Dense Layer',
        concept: 'dense',
        text: 'Drag from the **out** socket of **ReLU** and choose **Dense Layer**. This layer will produce the final hidden representation before the regression output.',
        target: { nodeType: 'relu', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from ReLU, then choose Dense Layer.',
          validate: [
            { rule: 'wireExists', params: { fromType: 'relu', toType: 'dense' } },
            { rule: 'nodeCount', params: { type: 'dense', min: 2 } }
          ]
        }
      },
      {
        id: 'cp-linear',
        title: '9. Add the Linear output',
        concept: 'linear',
        text: 'Drag from the **out** socket of the second **Dense Layer** and choose **Linear**. Linear passes the raw score through unchanged, which is what a regression target needs.',
        target: { nodeType: 'dense', socket: 'out', direction: 'out', pick: 'newest' },
        action: {
          hint: 'Drag from the last Dense Layer output, then choose Linear.',
          validate: { rule: 'wireExists', params: { fromType: 'dense', toType: 'linear' } }
        }
      },
      {
        id: 'cp-size',
        title: '10. Set the output Dense Layer to 1 neuron',
        concept: 'dense',
        text: 'Select the **Dense Layer** that feeds **Linear** and set **Neurons** to **1**. The model predicts one continuous quantity.',
        target: '#inspector-body',
        action: {
          id: 'param_changed',
          hint: 'Select the Dense Layer feeding Linear and set Neurons = 1.',
          validate: { rule: 'feedingParam', params: { sourceType: 'dense', targetType: 'linear', param: 'size', value: 1 } }
        }
      },
      {
        id: 'cp-mse',
        title: '11. Add MSE Loss',
        concept: 'mse_loss',
        text: 'Drag from the **out** socket of **Linear** and choose **MSE Loss**. Mean Squared Error is the matching loss for a continuous target: it squares the distance between prediction and label.',
        target: { nodeType: 'linear', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag from Linear, then choose MSE Loss.',
          validate: { rule: 'wireExists', params: { fromType: 'linear', toType: 'mse_loss' } }
        }
      },
      {
        id: 'cp-snip',
        title: '12. Break the Linear → MSE Loss wire',
        concept: 'snip',
        text: 'Right-click the wire from **Linear** to **MSE Loss** and choose **Break connection**. In Advanced mode the menu first predicts the validation damage, then removes the wire.',
        target: { nodeType: 'linear', socket: 'out', direction: 'out' },
        action: {
          hint: 'Right-click the Linear → MSE Loss wire and choose Break connection.',
          validate: [
            { rule: 'wireBroken', params: { fromType: 'linear', toType: 'mse_loss' } },
            { rule: 'pathExists', params: { fromType: 'dense', toType: 'linear' } }
          ]
        }
      },
      {
        id: 'cp-rewire',
        title: '13. Reconnect Linear → MSE Loss',
        concept: 'wiring',
        text: 'Drag from the **out** socket of **Linear** and release exactly on the **in** socket of **MSE Loss**. The graph compiles again only when the connection is real.',
        target: { nodeType: 'linear', socket: 'out', direction: 'out' },
        action: {
          hint: 'Drag Linear output onto MSE Loss input.',
          validate: { rule: 'wireExists', params: { fromType: 'linear', toType: 'mse_loss' } }
        }
      },
      {
        id: 'cp-opt',
        title: '14. Attach RMSprop and Epoch Loop',
        concept: 'rmsprop_optimizer',
        text: 'Two connections on **MSE Loss**: drag from **opt** and choose **RMSprop**; drag from **loop** and choose **Epoch Loop**. RMSprop adapts each weight’s step size, and the loop controls epochs and early stopping.',
        target: { nodeType: 'mse_loss', socket: 'opt', direction: 'in' },
        action: {
          hint: 'Wire both the opt and loop sockets on MSE Loss.',
          validate: [
            { rule: 'socketWired', params: { nodeType: 'mse_loss', socket: 'opt' } },
            { rule: 'socketWired', params: { nodeType: 'mse_loss', socket: 'loop' } }
          ]
        }
      },
      {
        id: 'cp-schedule',
        title: '15. Attach the LR Scheduler',
        concept: 'lr_schedule',
        text: 'Drag from the **schedule** input on **MSE Loss** and choose **LR Scheduler**. The optimizer’s learning rate then follows a real step-decay schedule during training.',
        target: { nodeType: 'mse_loss', socket: 'schedule', direction: 'in' },
        action: {
          hint: 'Drag from the loss schedule input, then choose LR Scheduler.',
          validate: { rule: 'socketWired', params: { nodeType: 'mse_loss', socket: 'schedule' } }
        }
      },
      {
        id: 'cp-gauge',
        title: '16. Attach the Loss / Accuracy Gauge',
        concept: 'metrics_gauge',
        text: 'Drag from the **monitor** output on **MSE Loss** into empty space and choose **Loss / Accuracy Gauge**. It is a read-only monitor fed by the real epoch stream.',
        target: { nodeType: 'mse_loss', socket: 'monitor', direction: 'out' },
        action: {
          hint: 'Drag from the loss monitor output, then choose the gauge.',
          validate: { rule: 'wireExists', params: { fromType: 'mse_loss', toType: 'metrics_gauge' } }
        }
      },
      {
        id: 'cp-run',
        title: '17. Press Run',
        concept: 'backprop',
        text: 'Press **Run**. Manual wiring means any missing or incompatible connection appears as a real compiler error before training starts. A valid graph trains with real backpropagation and RMSprop updates.',
        target: '#btn-run',
        action: {
          id: 'training_started',
          hint: 'Press Run. The step is remembered even if a short run finishes first.',
          validate: { rule: 'actionSeen', params: { id: 'training_started', label: 'pressing Run' } }
        }
      },
      {
        id: 'cp-done',
        title: '18. Read the held-out MSE',
        concept: 'inference',
        text: 'The **Training** panel reports held-out MSE: lower is better, and it is measured on examples the optimizer did not train on. Export the graph as JSON, Python, JavaScript or ONNX, or undo the last cut to repeat the wire repair.',
        target: '#panel-metrics'
      }
    ]
  }
};

export const MISSION_IDS = Object.keys(MISSIONS);

/* ---------------------------------------------------------- step machine */

/**
 * Pure mission step machine. Gates come from the page via `check(ctx)` (graph predicates) and
 * `notify(actionId, ctx)` (real UI events). `next()` refuses while the current step's gate is
 * unmet, which is exactly the "action-gated progress" contract.
 */
export class MissionRunner {
  constructor(mission) {
    if (!mission || !Array.isArray(mission.steps) || mission.steps.length === 0) {
      throw new Error('MissionRunner needs a mission with at least one step.');
    }
    this.mission = mission;
    this.index = 0;
    this.satisfied = new Set();
    this.events = new Map();
    this.gateStatus = null;
    this._ctx = null;
    this.finished = false;
  }

  get total() {
    return this.mission.steps.length;
  }

  get current() {
    return this.finished ? null : this.mission.steps[this.index];
  }

  eventCount(actionId) {
    return this.events.get(actionId) || 0;
  }

  /** Evaluate the current step's gate against real state. Marks the gate satisfied when it passes. */
  check(ctx) {
    const live = { ...(ctx || this._ctx || {}), missionEvents: this.events };
    this._ctx = live;
    const step = this.current;
    if (!step) return { ok: true, reason: null };
    if (!step.action) {
      this.gateStatus = { ok: true, reason: null };
      return this.gateStatus;
    }
    const validate = step.action.validate;
    if (validate) {
      const result = evaluateValidation(validate, live);
      if (result.ok) this.satisfied.add(step.id);
      this.gateStatus = {
        ok: Boolean(result.ok),
        reason: result.ok ? null : (result.reason || step.action.hint || null)
      };
      return this.gateStatus;
    }
    const ok = this.satisfied.has(step.id);
    this.gateStatus = { ok, reason: ok ? null : (step.action.hint || null) };
    return this.gateStatus;
  }

  /** Report a real UI action. Predicate steps stay predicate-driven; event-only steps latch. */
  notify(actionId, ctx) {
    this.events.set(actionId, this.eventCount(actionId) + 1);
    const step = this.current;
    if (!step || !step.action) return { known: false, ok: false, reason: null };
    const matches = step.action.id === actionId;
    if (matches && !step.action.validate) this.satisfied.add(step.id);
    const result = this.check(ctx);
    return { known: matches, ok: result.ok, reason: result.reason };
  }

  /** True when the current step may be completed. */
  gateMet(ctx) {
    return this.check(ctx).ok;
  }

  next() {
    if (this.finished) return { advanced: false, finished: true };
    const step = this.current;
    const gate = this.gateStatus || this.check(this._ctx);
    if (step.action && !gate.ok) {
      return { advanced: false, finished: false, reason: gate.reason || 'Complete the step first.' };
    }
    if (this.index >= this.total - 1) {
      this.finished = true;
      return { advanced: true, finished: true };
    }
    this.index += 1;
    this.gateStatus = null;
    return { advanced: true, finished: false };
  }

  prev() {
    if (this.index === 0 && !this.finished) return { moved: false };
    if (this.finished) this.finished = false;
    else this.index -= 1;
    this.gateStatus = null;
    return { moved: true };
  }

  skip() {
    const step = this.current;
    if (!step) return { advanced: false, finished: true };
    if (!step.optional) return { advanced: false, finished: false, reason: 'This step is required — complete it to continue.' };
    return this.next();
  }

  progress() {
    return { index: this.index, total: this.total, done: this.finished, satisfied: this.satisfied.size };
  }
}

/* -------------------------------------------------------------- presenter */

/**
 * The mission overlay. `StepByStepEngine` drives `MissionRunner` and presents the spotlight,
 * the pulsing target ring and the explanation card. The page injects:
 *   resolveTarget(target)  -> element|null   (handles CSS selectors and {nodeType, socket} targets)
 *   onSetup(setup, step)                     (page command on step entry, e.g. clear_graph)
 *   onStepEnter/onStepLeave/onComplete/onLog
 */
export class StepByStepEngine {
  constructor(options = {}) {
    this.doc = options.document || (typeof document !== 'undefined' ? document : null);
    this.win = options.window || (typeof window !== 'undefined' ? window : null);
    this.container = options.container || (this.doc && this.doc.body ? this.doc.body : null);
    this.menuManager = options.menuManager || null;
    this.menuId = options.menuId || 'mission-overlay';
    this.resolveTarget = options.resolveTarget || ((target) => this._defaultResolve(target));
    this.onSetup = options.onSetup || null;
    this.onStepEnter = options.onStepEnter || null;
    this.onStepLeave = options.onStepLeave || null;
    this.onComplete = options.onComplete || null;
    this.onExit = options.onExit || null;
    this.onLog = options.onLog || null;
    this.runner = null;
    this.visible = false;
    this.lastStep = null;
    this.targetEl = null;
    this._ctx = null;
    this._techOpen = false;
    this._placement = null;
    this._forcePlace = true;
    this._onResize = () => {
      this._forcePlace = true;
      this._position();
    };
    this._onScroll = () => this._position();
    this.root = null;
    if (this.doc) this._build();
  }

  /* ------------------------------------------------------------ public API */

  get active() {
    return Boolean(this.runner && this.visible && !this.runner.finished);
  }

  get mission() {
    return this.runner ? this.runner.mission : null;
  }

  get tier() {
    return this.mission ? this.mission.tier : 0;
  }

  /** While a mission runs, drop-time auto-wiring is suppressed so construction stays in the user's hands. */
  get suppressAssist() {
    return Boolean(this.active && this.mission && this.mission.suppressAssist !== false);
  }

  start(missionId, ctx) {
    const mission = MISSIONS[missionId];
    if (!mission || !this.doc) return false;
    if (this.runner) this.finish('restart');
    this.runner = new MissionRunner(mission);
    this.visible = true;
    this._ctx = ctx || {};
    this._techOpen = false;
    if (this.root) this.root.hidden = false;
    this._listen();
    this._enterStep(0, this._ctx);
    if (this.onLog) this.onLog('info', `Mission started: ${mission.title}. ${mission.goal}`);
    this.check(this._ctx);
    return true;
  }

  /** Button-driven navigation uses the most recent live page context. */
  _context(ctx) {
    if (ctx) this._ctx = ctx;
    return ctx || this._ctx || {};
  }

  /** Refresh the gate + presentation against live page state. Idempotent. */
  check(ctx) {
    if (!this.runner || !this.visible) return { ok: false, reason: null };
    const live = this._context(ctx);
    const gate = this.runner.check(live);
    if (this.runner.finished) {
      this._complete();
      return gate;
    }
    this._render(live, gate);
    return gate;
  }

  /** Report a real UI action; may satisfy the gate and re-render. */
  notify(actionId, ctx) {
    if (!this.runner || !this.visible) return { known: false, ok: false, reason: null };
    const live = this._context(ctx);
    const result = this.runner.notify(actionId, live);
    this._render(live, this.runner.gateStatus);
    return result;
  }

  /**
   * Re-measure the live target and keep the spotlight / ring / card glued to it.
   *
   * Canvas node drags replace the node DOM on every pointer move, so the presenter cannot cache the
   * target element: it has to resolve it again and reposition. The page calls this from
   * `renderNodes()`, `applyView()` and `renderInspector()`; the presenter also listens for viewport
   * resize and captured scroll events. Safe to call when the overlay is inactive.
   */
  refreshPosition() {
    if (!this.active || !this.visible) return;
    this._position();
  }

  next(ctx) {
    if (!this.runner) return { advanced: false, finished: true };
    const live = this._context(ctx);
    this.runner.check(live);
    const result = this.runner.next();
    if (result.advanced && !result.finished) this._enterStep(this.runner.index, live);
    if (result.finished) this._complete();
    return result;
  }

  prev(ctx) {
    if (!this.runner) return { moved: false };
    const live = this._context(ctx);
    const result = this.runner.prev();
    if (result.moved) this._enterStep(this.runner.index, live);
    return result;
  }

  skip(ctx) {
    if (!this.runner) return { advanced: false, finished: true };
    const live = this._context(ctx);
    this.runner.check(live);
    const result = this.runner.skip();
    if (result.advanced && !result.finished) this._enterStep(this.runner.index, live);
    if (result.finished) this._complete();
    return result;
  }

  finish(reason = 'exit') {
    if (!this.runner) return false;
    const mission = this.runner.mission;
    this._clearTargetClass();
    this.runner.finished = true;
    this.runner = null;
    this.visible = false;
    this.lastStep = null;
    if (this.root) this.root.hidden = true;
    this._unlisten();
    if (this.onLog && reason !== 'complete' && reason !== 'restart') this.onLog('info', `Mission exited: ${mission.title}.`);
    if (reason === 'exit' && this.onExit) this.onExit(mission);
    return true;
  }

  /* ---------------------------------------------------------------- build */

  _build() {
    const doc = this.doc;
    const root = doc.createElement('div');
    root.className = 'mission-root';
    root.hidden = true;

    this.spotlight = doc.createElement('div');
    this.spotlight.className = 'mission-spotlight';
    this.spotlight.hidden = true;
    root.appendChild(this.spotlight);

    this.ring = doc.createElement('div');
    this.ring.className = 'mission-ring';
    this.ring.hidden = true;
    root.appendChild(this.ring);

    const card = doc.createElement('div');
    card.className = 'mission-card';

    const close = doc.createElement('button');
    close.type = 'button';
    close.className = 'mission-close';
    close.textContent = '×';
    close.setAttribute('aria-label', 'Exit the mission');
    close.addEventListener('click', () => this.finish('exit'));
    card.appendChild(close);

    this.kicker = doc.createElement('div');
    this.kicker.className = 'mission-kicker';
    card.appendChild(this.kicker);

    this.goal = doc.createElement('div');
    this.goal.className = 'mission-goal';
    card.appendChild(this.goal);

    const progressRow = doc.createElement('div');
    progressRow.className = 'mission-progress-row';
    this.progressText = doc.createElement('span');
    this.progressText.className = 'mission-progress-text';
    this.progressPct = doc.createElement('span');
    this.progressPct.className = 'mission-progress-pct';
    progressRow.appendChild(this.progressText);
    progressRow.appendChild(this.progressPct);
    card.appendChild(progressRow);

    const progress = doc.createElement('div');
    progress.className = 'mission-progress';
    this.progressFill = doc.createElement('div');
    this.progressFill.className = 'mission-progress-fill';
    progress.appendChild(this.progressFill);
    card.appendChild(progress);

    this.stepTitle = doc.createElement('div');
    this.stepTitle.className = 'mission-step-title';
    card.appendChild(this.stepTitle);

    this.stepText = doc.createElement('div');
    this.stepText.className = 'mission-step-text';
    card.appendChild(this.stepText);

    this.explainBox = doc.createElement('div');
    this.explainBox.className = 'mission-explain';
    this.explainChip = doc.createElement('div');
    this.explainChip.className = 'mission-explain-chip';
    this.explainPlain = doc.createElement('div');
    this.explainPlain.className = 'mission-explain-plain';
    this.explainAnalogy = doc.createElement('div');
    this.explainAnalogy.className = 'mission-explain-analogy';
    this.techToggle = doc.createElement('button');
    this.techToggle.type = 'button';
    this.techToggle.className = 'mission-tech-toggle';
    this.techToggle.textContent = 'Show the technical detail';
    this.techToggle.addEventListener('click', () => {
      this._techOpen = !this._techOpen;
      this.explainBox.classList.toggle('is-tech-open', this._techOpen);
      this.techToggle.textContent = this._techOpen ? 'Hide the technical detail' : 'Show the technical detail';
      // The card grows when the detail opens; re-measure instead of letting it run off-screen.
      this._forcePlace = true;
      this._position();
    });
    this.explainTech = doc.createElement('div');
    this.explainTech.className = 'mission-explain-tech';
    this.explainBox.appendChild(this.explainChip);
    this.explainBox.appendChild(this.explainPlain);
    this.explainBox.appendChild(this.explainAnalogy);
    this.explainBox.appendChild(this.techToggle);
    this.explainBox.appendChild(this.explainTech);
    card.appendChild(this.explainBox);

    this.hint = doc.createElement('div');
    this.hint.className = 'mission-hint';
    card.appendChild(this.hint);

    const actions = doc.createElement('div');
    actions.className = 'mission-actions';
    this.backBtn = doc.createElement('button');
    this.backBtn.type = 'button';
    this.backBtn.className = 'ns-btn mission-back';
    this.backBtn.textContent = '← Back';
    this.backBtn.addEventListener('click', () => this.prev());
    this.skipBtn = doc.createElement('button');
    this.skipBtn.type = 'button';
    this.skipBtn.className = 'ns-btn mission-skip';
    this.skipBtn.textContent = 'Skip step';
    this.skipBtn.addEventListener('click', () => this.skip());
    this.nextBtn = doc.createElement('button');
    this.nextBtn.type = 'button';
    this.nextBtn.className = 'ns-btn ns-btn-primary mission-next';
    this.nextBtn.textContent = 'Next →';
    this.nextBtn.addEventListener('click', () => this.next());
    actions.appendChild(this.backBtn);
    actions.appendChild(this.skipBtn);
    actions.appendChild(this.nextBtn);
    card.appendChild(actions);

    root.appendChild(card);
    this.card = card;
    this.root = root;
    if (this.container && this.container.appendChild) this.container.appendChild(root);
  }

  /* ------------------------------------------------------------- rendering */

  _enterStep(index, ctx) {
    const step = this.runner ? this.runner.mission.steps[index] : null;
    if (!step) return;
    if (this.onStepLeave && this.lastStep) this.onStepLeave(this.lastStep, this);
    this.lastStep = step;
    if (this.onSetup && step.setup) this.onSetup(step.setup, step, this);
    if (this.onStepEnter) this.onStepEnter(step, this);
    this._techOpen = false;
    this._placement = null;
    this._forcePlace = true;
    if (this.explainBox) this.explainBox.classList.remove('is-tech-open');
    if (this.techToggle) this.techToggle.textContent = 'Show the technical detail';
    this._render(ctx || {}, this.runner.gateStatus);
  }

  _render(ctx, gate) {
    if (!this.runner || !this.stepTitle) return;
    const mission = this.runner.mission;
    const step = this.runner.current;
    if (!step) return;
    const total = this.runner.total;
    const index = this.runner.index;
    const status = gate || this.runner.check(ctx || {});

    this.kicker.textContent = `Step-by-step build · ${mission.title.replace(/^Lesson\s+\d+\s+[—–-]\s*/u, '')}`;
    this.goal.textContent = mission.goal;
    this.progressText.textContent = `Step ${index + 1} of ${total}`;
    const pct = Math.round(((index + (status.ok ? 1 : 0.35)) / total) * 100);
    this.progressPct.textContent = `${Math.min(100, pct)}%`;
    if (this.progressFill) this.progressFill.style.width = `${Math.min(100, pct)}%`;

    this.stepTitle.textContent = step.title;
    this.stepText.innerHTML = inlineMarkdown(step.text);

    const entry = explain(step.concept);
    this.explainChip.textContent = entry.title;
    this.explainPlain.textContent = entry.plain;
    if (entry.analogy) {
      this.explainAnalogy.textContent = `Think of it like this: ${entry.analogy}`;
      this.explainAnalogy.hidden = false;
    } else {
      this.explainAnalogy.textContent = '';
      this.explainAnalogy.hidden = true;
    }
    this.explainTech.textContent = entry.signals ? `${entry.technical} ${entry.signals}` : entry.technical;

    const gateMet = status.ok;
    this.hint.textContent = gateMet
      ? '✓ Step verified against the live graph — press Next.'
      : (step.action
        ? (step.action.hint || status.reason || 'Complete the action to continue.')
        : 'Read the explanation, then press Next.');
    this.hint.classList.toggle('is-met', gateMet);

    if (this.nextBtn) {
      this.nextBtn.disabled = !gateMet;
      this.nextBtn.textContent = index >= total - 1 ? 'Finish mission' : 'Next →';
    }
    if (this.backBtn) this.backBtn.disabled = index === 0;
    if (this.skipBtn) this.skipBtn.hidden = !step.optional;

    this._position();
  }

  _defaultResolve(target) {
    if (typeof target === 'string' && this.doc && this.doc.querySelector) return this.doc.querySelector(target);
    return null;
  }

  _currentTargetEl() {
    const step = this.runner ? this.runner.current : null;
    if (!step || !step.target) return null;
    try {
      return this.resolveTarget(step.target) || null;
    } catch {
      return null;
    }
  }

  _clearTargetClass() {
    if (this.targetEl && this.targetEl.classList) this.targetEl.classList.remove('is-mission-target');
    this.targetEl = null;
  }

  _position() {
    if (!this.visible || !this.spotlight || !this.card) return;
    const margin = 12;
    const winW = this.win && Number.isFinite(this.win.innerWidth) ? this.win.innerWidth : 1280;
    const winH = this.win && Number.isFinite(this.win.innerHeight) ? this.win.innerHeight : 800;

    this._clearTargetClass();
    const targetEl = this._currentTargetEl();
    this.targetEl = targetEl;

    const rawRect = targetEl && targetEl.getBoundingClientRect ? targetEl.getBoundingClientRect() : null;
    const rect = rawRect ? normalizeRect(rawRect) : null;
    const hasTarget = Boolean(rect && rect.width >= 1 && rect.height >= 1);

    // Width has to be settled before measuring: the card's height depends on wrapping, and the
    // placement maths needs the *rendered* height (max-height makes the card scroll internally).
    const cardW = Math.max(1, Math.min(384, winW - margin * 2));
    this.card.style.width = `${cardW}px`;
    const measuredH = this.card.offsetHeight;
    const cardH = Math.min(measuredH > 0 ? measuredH : 460, Math.max(1, winH - margin * 2));

    if (!hasTarget) {
      this.spotlight.hidden = true;
      this.ring.hidden = true;
      this._placement = null;
      this._forcePlace = false;
      this.card.style.left = `${Math.max(margin, (winW - cardW) / 2)}px`;
      this.card.style.top = `${Math.max(margin, (winH - cardH) / 2)}px`;
      return;
    }

    if (targetEl.classList) targetEl.classList.add('is-mission-target');

    // A socket is a 12 px dot; placing the card next to the dot would still hide the node body and
    // the wire's natural drag path. Likewise, a palette button or inspector field lives inside a
    // full-height panel. Anchor the card to the whole block/panel so it docks over the canvas
    // instead of slicing through the fixed chrome.
    let avoidRect = rect;
    if (targetEl.closest) {
      const anchorEl = targetEl.closest('.ns-node')
        || targetEl.closest('.ns-palette, .ns-right, .ns-topbar, .ns-statusbar');
      if (anchorEl && anchorEl !== targetEl && anchorEl.getBoundingClientRect) {
        const anchorRect = normalizeRect(anchorEl.getBoundingClientRect());
        if (anchorRect && anchorRect.width > 0 && anchorRect.height > 0) avoidRect = anchorRect;
      }
    }

    const backdrop = targetEl.id === 'studio-canvas'
      || isBackdropTarget(avoidRect, winW, winH);
    const isTiny = rect.width < 44 || rect.height < 44;
    const pad = isTiny ? 10 : 6;
    const left = Math.max(4, rect.left - pad);
    const top = Math.max(4, rect.top - pad);
    const width = Math.max(2, Math.min(winW - left - 4, rect.width + pad * 2));
    const height = Math.max(2, Math.min(winH - top - 4, rect.height + pad * 2));

    for (const el of [this.spotlight, this.ring]) {
      el.hidden = false;
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
      el.style.width = `${width}px`;
      el.style.height = `${height}px`;
    }

    const stepId = this.runner && this.runner.current ? this.runner.current.id : null;
    const sizeChanged = !this._placement
      || Math.abs(this._placement.width - cardW) > 1
      || Math.abs(this._placement.height - cardH) > 1;
    const canKeep = !this._forcePlace
      && !sizeChanged
      && this._placement
      && this._placement.stepId === stepId
      && this._placementStillValid(this._placement, avoidRect, !backdrop, winW, winH, margin);

    if (!canKeep) {
      const chrome = measureChrome(this.doc, STUDIO_CHROME_SELECTORS);
      const placed = placeFloatingPanel({
        target: avoidRect,
        width: cardW,
        height: cardH,
        viewportWidth: winW,
        viewportHeight: winH,
        margin,
        gap: 16,
        prefer: ['right', 'left', 'below', 'above'],
        hardAvoid: !backdrop,
        softAvoid: chrome
      });
      this._placement = {
        stepId,
        x: placed.x,
        y: placed.y,
        width: cardW,
        height: cardH
      };
    }

    this._forcePlace = false;
    this.card.style.left = `${this._placement.x}px`;
    this.card.style.top = `${this._placement.y}px`;
  }

  /** A stable card is only allowed to stay put if it is still on-screen and off the live target. */
  _placementStillValid(placement, avoidRect, hardAvoid, winW, winH, margin) {
    if (!placement) return false;
    const panel = {
      left: placement.x,
      top: placement.y,
      width: placement.width,
      height: placement.height,
      right: placement.x + placement.width,
      bottom: placement.y + placement.height
    };
    if (panel.left < margin - 1 || panel.top < margin - 1
      || panel.right > winW - margin + 1 || panel.bottom > winH - margin + 1) {
      return false;
    }
    if (hardAvoid && avoidRect && intersectionArea(panel, expandRect(avoidRect, 10)) > 0) return false;
    return true;
  }

  _complete() {
    const mission = this.runner ? this.runner.mission : null;
    this._clearTargetClass();
    this.runner = null;
    this.visible = false;
    this.lastStep = null;
    if (this.root) this.root.hidden = true;
    this._unlisten();
    if (mission && this.onComplete) this.onComplete(mission);
  }

  _listen() {
    if (!this.win || !this.win.addEventListener) return;
    this.win.addEventListener('resize', this._onResize);
    // Capture-phase scroll also catches the palette / inspector scroll containers, which do not
    // bubble. Without this the spotlight stays pinned while the target scrolls out from under it.
    this.win.addEventListener('scroll', this._onScroll, true);
  }

  _unlisten() {
    if (!this.win || !this.win.removeEventListener) return;
    this.win.removeEventListener('resize', this._onResize);
    this.win.removeEventListener('scroll', this._onScroll, true);
  }
}
