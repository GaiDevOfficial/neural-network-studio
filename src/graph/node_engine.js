/**
 * Node Studio — node graph evaluation engine.
 *
 * Converts a visually wired graph into an executable plan for the platform's verified
 * engine: real tensor-shape inference, structured validation errors, and a compiled
 * {network, training, data, taps} plan that the run controller posts to the existing
 * worker RPC protocol. Pure and synchronous — data resolution (CSV/SQL/REST fetches)
 * happens before compile(), and the dataset *shape info* arrives through `context`.
 *
 * File: src/graph/node_engine.js
 */

import { getNodeType, SOURCE_NODE_TYPES, LOSS_NODE_TYPES, OPTIMIZER_NODE_TYPES, ACTIVATION_NODE_TYPES } from './node_types.js';

/* ------------------------------------------------------------------ errors */

export const ERROR_HELP = {
  E_NO_SOURCE: 'Add a Dataset Source / SQL Query / REST API block and wire it into the chain.',
  E_MULTI_SOURCE: 'A graph trains on exactly one dataset — remove the extra source blocks.',
  E_DATA_UNRESOLVED: 'The source has no dataset attached yet (upload the CSV / fetch the query first).',
  E_NO_LOSS: 'Every graph needs one Loss block as its terminal.',
  E_MULTI_LOSS: 'Wire a single Loss block; remove the extras.',
  E_MISSING_INPUT: 'A block in the chain has no incoming wire.',
  E_DUP_INPUT: 'Input sockets accept exactly one wire.',
  E_KIND_MISMATCH: 'Only sockets of the same kind (tensor/config/monitor) can be wired together.',
  E_CYCLE: 'The graph must stay a directed acyclic graph.',
  E_SELF_LOOP: 'A block cannot feed itself.',
  E_BAD_SOCKET: 'Unknown socket.',
  E_BAD_NODE: 'Missing block.',
  E_RANK_MISMATCH: 'Dense layers need a flat vector — insert a Flatten for image inputs.',
  E_PAIRING: 'The loss and the output activation must be a valid pair (BCE↔Sigmoid, CCE↔Softmax).',
  E_SIZE_MISMATCH: 'The final layer size must match the loss/dataset output requirement.',
  E_MISSING_CONFIG: 'The loss has an empty config socket (optimizer / epoch loop).',
  E_UNSUPPORTED: 'This block is not supported by engine v1.',
  E_PARAM_RANGE: 'A parameter is outside its declared range.',
  W_UNCONNECTED: 'Block is not connected to the loss chain and will be ignored.'
};

function err(code, message, nodeId = null, socket = null) {
  return { code, message, nodeId, socket };
}

function nodeName(graph, nodeId) {
  const node = graph.getNode(nodeId);
  return node ? `${node.type} (${node.id})` : String(nodeId);
}

/** Parse a Class Filter keep-list ("0,1") into distinct ascending indices, or null when invalid. */
export function parseKeepList(text) {
  const parts = String(text || '')
    .split(',')
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
  if (parts.length === 0) return null;
  const indices = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return null;
    const value = Number(part);
    if (!indices.includes(value)) indices.push(value);
  }
  return indices.sort((a, b) => a - b);
}

/* ------------------------------------------------------- shape inference */

/**
 * Walk the graph in topological order and compute, for every node:
 *   shapes: Map(nodeId → dims array)      — the tensor leaving that node
 *   views:  Map(nodeId → dataset view)    — {type, numFeatures, numClasses, imageDim, name}
 * plus the tensor chain from its source to the loss, and every structural error.
 *
 * @param {GraphState} graph
 * @param {object} context { datasetInfoFor(node) → info|null }
 */
export function inferGraph(graph, context = {}) {
  const errors = [];
  const warnings = [];
  const shapes = new Map();
  const views = new Map();
  const datasetInfoFor = context.datasetInfoFor || (() => null);

  const sourceIds = graph.nodeList().filter((n) => SOURCE_NODE_TYPES.includes(n.type)).map((n) => n.id);
  const lossIds = graph.nodeList().filter((n) => LOSS_NODE_TYPES.includes(n.type)).map((n) => n.id);

  if (sourceIds.length === 0) errors.push(err('E_NO_SOURCE', 'The graph has no data source block.'));
  if (sourceIds.length > 1) {
    errors.push(err('E_MULTI_SOURCE', `Found ${sourceIds.length} source blocks (${sourceIds.join(', ')}) — exactly one dataset is allowed.`));
  }
  if (lossIds.length === 0) errors.push(err('E_NO_LOSS', 'The graph has no Loss block (the training terminal).'));
  if (lossIds.length > 1) {
    errors.push(err('E_MULTI_LOSS', `Found ${lossIds.length} loss blocks (${lossIds.join(', ')}) — exactly one is allowed.`));
  }

  const inputShapeOf = (nodeId, socketName) => {
    const conn = graph.incoming(nodeId, socketName);
    if (!conn) return null;
    return shapes.get(conn.from.nodeId) || null;
  };
  const viewOf = (nodeId, socketName) => {
    const conn = graph.incoming(nodeId, socketName);
    if (!conn) return null;
    return views.get(conn.from.nodeId) || null;
  };

  for (const nodeId of graph.topologicalOrder()) {
    const node = graph.getNode(nodeId);
    if (!node) continue;
    const def = getNodeType(node.type);
    if (!def) continue;

    // ---- unsupported (planned) blocks anywhere in a wired chain get a precise error
    if (def.supported === false) {
      const wired = graph.incoming(nodeId, 'in') || graph.outgoing(nodeId).length > 0;
      if (wired) {
        errors.push(err('E_UNSUPPORTED', `${nodeName(graph, nodeId)}: ${def.reason}`, nodeId));
      }
      continue;
    }

    // ---- data sources ----------------------------------------------------
    if (SOURCE_NODE_TYPES.includes(node.type)) {
      const info = datasetInfoFor(node);
      if (!info) {
        errors.push(err('E_DATA_UNRESOLVED', `${nodeName(graph, nodeId)} has no dataset attached yet.`, nodeId, 'out'));
        continue;
      }
      views.set(nodeId, { ...info });
      shapes.set(nodeId, info.imageDim ? [info.imageDim.height, info.imageDim.width] : [info.numFeatures]);
      continue;
    }

    // ---- class filter (remaps one-hot targets, keeps the input shape) ----
    if (node.type === 'class_filter') {
      const view = viewOf(nodeId, 'in');
      const shape = inputShapeOf(nodeId, 'in');
      if (!view || !shape) {
        errors.push(err('E_MISSING_INPUT', `${nodeName(graph, nodeId)} has no incoming data wire.`, nodeId, 'in'));
        continue;
      }
      if (view.type !== 'classification') {
        errors.push(err('E_KIND_MISMATCH', `${nodeName(graph, nodeId)} requires a classification dataset ("${view.name}" is regression).`, nodeId, 'in'));
        continue;
      }
      const keep = parseKeepList(node.params.keep);
      if (!keep) {
        errors.push(err('E_PARAM_RANGE', `${nodeName(graph, nodeId)}: keep list "${node.params.keep}" is not a comma list of class indices.`, nodeId));
        continue;
      }
      const bad = keep.find((index) => index < 0 || index >= view.numClasses);
      if (bad !== undefined) {
        errors.push(err('E_PARAM_RANGE', `${nodeName(graph, nodeId)}: class index ${bad} is out of range (dataset has ${view.numClasses} classes).`, nodeId));
        continue;
      }
      views.set(nodeId, { ...view, numClasses: keep.length });
      shapes.set(nodeId, shape);
      continue;
    }

    // ---- flatten ---------------------------------------------------------
    if (node.type === 'flatten') {
      const shape = inputShapeOf(nodeId, 'in');
      if (!shape) {
        errors.push(err('E_MISSING_INPUT', `${nodeName(graph, nodeId)} has no incoming data wire.`, nodeId, 'in'));
        continue;
      }
      const product = shape.reduce((acc, d) => acc * d, 1);
      shapes.set(nodeId, [product]);
      views.set(nodeId, viewOf(nodeId, 'in'));
      continue;
    }

    // ---- dense -----------------------------------------------------------
    if (node.type === 'dense') {
      const shape = inputShapeOf(nodeId, 'in');
      if (!shape) {
        errors.push(err('E_MISSING_INPUT', `${nodeName(graph, nodeId)} has no incoming data wire.`, nodeId, 'in'));
        continue;
      }
      if (shape.length !== 1) {
        errors.push(err(
          'E_RANK_MISMATCH',
          `${nodeName(graph, nodeId)} needs a flat vector but receives ${JSON.stringify(shape)} — insert a Flatten block (${JSON.stringify(shape)} → [${shape.reduce((a, d) => a * d, 1)}]).`,
          nodeId, 'in'
        ));
        continue;
      }
      const size = node.params.size;
      if (!Number.isInteger(size) || size < 1 || size > 512) {
        errors.push(err('E_PARAM_RANGE', `${nodeName(graph, nodeId)}: neuron count ${size} is outside 1–512.`, nodeId));
        continue;
      }
      shapes.set(nodeId, [size]);
      views.set(nodeId, viewOf(nodeId, 'in'));
      continue;
    }

    // ---- activations -----------------------------------------------------
    if (ACTIVATION_NODE_TYPES.includes(node.type)) {
      const shape = inputShapeOf(nodeId, 'in');
      if (!shape) {
        errors.push(err('E_MISSING_INPUT', `${nodeName(graph, nodeId)} has no incoming data wire.`, nodeId, 'in'));
        continue;
      }
      if (node.type === 'softmax' && shape.length !== 1) {
        errors.push(err('E_RANK_MISMATCH', `${nodeName(graph, nodeId)} requires a rank-1 vector, received ${JSON.stringify(shape)}.`, nodeId, 'in'));
        continue;
      }
      shapes.set(nodeId, shape);
      views.set(nodeId, viewOf(nodeId, 'in'));
      continue;
    }

    // ---- loss ------------------------------------------------------------
    if (LOSS_NODE_TYPES.includes(node.type)) {
      const shape = inputShapeOf(nodeId, 'in');
      if (!shape) {
        errors.push(err('E_MISSING_INPUT', `${nodeName(graph, nodeId)} has no incoming data wire (its "in" socket is empty).`, nodeId, 'in'));
        continue;
      }
      if (shape.length !== 1) {
        errors.push(err('E_RANK_MISMATCH', `${nodeName(graph, nodeId)} requires a flat output vector, received ${JSON.stringify(shape)}.`, nodeId, 'in'));
        continue;
      }
      const lossKind = def.loss;
      const outputSize = shape[0];
      const sourceConn = graph.incoming(nodeId, 'in');
      const upstream = sourceConn ? graph.getNode(sourceConn.from.nodeId) : null;
      const upstreamDef = upstream ? getNodeType(upstream.type) : null;
      const upstreamActivation = upstream && ACTIVATION_NODE_TYPES.includes(upstream.type)
        ? upstreamDef.activation
        : upstream && upstream.type === 'dense'
          ? 'linear'
          : null;

      // The loss must be fed by a Dense output (directly or through an Activation).
      if (!upstream || (upstream.type !== 'dense' && !ACTIVATION_NODE_TYPES.includes(upstream.type))) {
        errors.push(err(
          'E_PAIRING',
          `${nodeName(graph, nodeId)} must be fed by a Dense output (optionally through an Activation block), not by ${upstream ? upstream.type : 'nothing'}.`,
          nodeId, 'in'
        ));
        continue;
      }

      const view = viewOf(nodeId, 'in');

      if (lossKind === 'binary_crossentropy') {
        if (view && view.type === 'classification' && view.numClasses !== 2) {
          errors.push(err('E_PAIRING', `${nodeName(graph, nodeId)} is a two-class loss but the dataset has ${view.numClasses} classes — use Categorical Cross-Entropy.`, nodeId));
        }
        if (upstreamActivation !== 'sigmoid') {
          errors.push(err('E_PAIRING', `${nodeName(graph, nodeId)} requires a Sigmoid output activation (found "${upstreamActivation}").`, nodeId, 'in'));
        }
        if (outputSize !== 2) {
          errors.push(err('E_SIZE_MISMATCH', `${nodeName(graph, nodeId)} expects a 2-value output (binary one-hot) but the chain ends with ${outputSize} values.`, nodeId, 'in'));
        }
      } else if (lossKind === 'categorical_crossentropy') {
        if (upstreamActivation !== 'softmax') {
          errors.push(err('E_PAIRING', `${nodeName(graph, nodeId)} requires a Softmax output activation (found "${upstreamActivation}").`, nodeId, 'in'));
        }
        if (view && view.type === 'classification' && outputSize !== view.numClasses) {
          errors.push(err('E_SIZE_MISMATCH', `${nodeName(graph, nodeId)} needs ${view.numClasses} output values (one per class) but the chain ends with ${outputSize}.`, nodeId, 'in'));
        }
      }
      // MSE: any flat size is accepted; the natural regression pairing is Linear + size 1.
      continue;
    }

    // configs / monitor taps carry no tensor shape — nothing to infer here.
  }

  // ---- the tensor chain: source → … → loss (walk back from loss.in) -------
  const chain = [];
  let chainError = null;
  if (lossIds.length >= 1) {
    let currentId = lossIds[0];
    const guard = new Set();
    while (currentId) {
      if (guard.has(currentId)) { chainError = 'cycle'; break; }
      guard.add(currentId);
      chain.push(currentId);
      if (SOURCE_NODE_TYPES.includes(graph.getNode(currentId).type)) break;
      const conn = graph.incoming(currentId, 'in');
      if (!conn) break;
      currentId = conn.from.nodeId;
    }
    chain.reverse();
  }

  // Consecutive activations would silently override each other — refused here so every
  // validation layer (compile + Smart Assist) sees the same rule.
  for (let i = 1; i < chain.length; i++) {
    const prev = graph.getNode(chain[i - 1]);
    const cur = graph.getNode(chain[i]);
    if (prev && cur && ACTIVATION_NODE_TYPES.includes(prev.type) && ACTIVATION_NODE_TYPES.includes(cur.type)) {
      errors.push(err(
        'E_PAIRING',
        `Two Activation blocks in a row (${prev.id} → ${cur.id}) — the second would override the first. Keep one activation per Dense layer.`,
        cur.id, 'in'
      ));
    }
  }

  // ---- unconnected warnings ----------------------------------------------
  const chainSet = new Set(chain);
  const tapIds = graph.nodeList().filter((n) => ['metrics_gauge', 'confusion_matrix', 'prediction_view'].includes(n.type)).map((n) => n.id);
  for (const node of graph.nodeList()) {
    const def = getNodeType(node.type);
    if (!def || def.supported === false) continue;
    if (chainSet.has(node.id)) continue;
    if (OPTIMIZER_NODE_TYPES.includes(node.type) || node.type === 'epoch_loop' || node.type === 'lr_schedule') {
      if (graph.outgoing(node.id).length === 0) {
        warnings.push(err('W_UNCONNECTED', `${nodeName(graph, node.id)} is not wired — defaults apply.`, node.id));
      }
      continue;
    }
    if (tapIds.includes(node.id)) {
      const monitorWired = graph.incoming(node.id, 'in');
      if (!monitorWired) {
        warnings.push(err('W_UNCONNECTED', `${nodeName(graph, node.id)} has no monitor wire — it will show nothing.`, node.id));
      }
      continue;
    }
    if (sourceIds.includes(node.id) && node.id !== sourceIds[0]) continue; // already reported as multi-source
    warnings.push(err('W_UNCONNECTED', `${nodeName(graph, node.id)} is not connected to the loss chain — it will be ignored.`, node.id));
  }

  return { errors, warnings, shapes, views, chain, chainError, sourceIds, lossIds };
}

/** Validation only (strict, no repairs) — Advanced Mode's gate. */
export function validate(graph, context = {}) {
  const inference = inferGraph(graph, context);
  return { ok: inference.errors.length === 0, errors: inference.errors, warnings: inference.warnings };
}

/* ------------------------------------------------------------- compile */

/**
 * Compile a validated graph into the executable plan the run controller drives.
 * Returns { ok:false, errors, warnings } when the graph is not buildable — the canvas
 * never starts a run against a half-valid graph.
 */
export function compile(graph, context = {}) {
  const inference = inferGraph(graph, context);
  const errors = [...inference.errors];
  const warnings = [...inference.warnings];

  const lossId = inference.lossIds[0] || null;
  const sourceId = inference.sourceIds[0] || null;

  let network = null;
  let training = null;
  let data = null;
  let taps = { gauges: [], confusion: false, predict: false };

  if (errors.length === 0 && lossId && sourceId) {
    const lossNode = graph.getNode(lossId);
    const lossDef = getNodeType(lossNode.type);

    // ---- optimizer (required) -------------------------------------------
    const optConn = graph.incoming(lossId, 'opt');
    const optNode = optConn ? graph.getNode(optConn.from.nodeId) : null;
    if (!optNode || !OPTIMIZER_NODE_TYPES.includes(optNode.type)) {
      errors.push(err('E_MISSING_CONFIG', `${nodeName(graph, lossId)} has no Optimizer wired into its "opt" socket.`, lossId, 'opt'));
    }

    // ---- epoch loop (required) ------------------------------------------
    const loopConn = graph.incoming(lossId, 'loop');
    const loopNode = loopConn ? graph.getNode(loopConn.from.nodeId) : null;
    if (!loopNode || loopNode.type !== 'epoch_loop') {
      errors.push(err('E_MISSING_CONFIG', `${nodeName(graph, lossId)} has no Epoch Loop wired into its "loop" socket.`, lossId, 'loop'));
    }

    // ---- optional lr scheduler ------------------------------------------
    const schedConn = graph.incoming(lossId, 'schedule');
    const schedNode = schedConn ? graph.getNode(schedConn.from.nodeId) : null;
    const schedule = schedNode && schedNode.type === 'lr_schedule'
      ? { mode: schedNode.params.mode, factor: schedNode.params.factor, every: schedNode.params.every }
      : null;
    if (schedNode && schedNode.type !== 'lr_schedule') {
      errors.push(err('E_KIND_MISMATCH', `${nodeName(graph, lossId)} "schedule" socket is fed by ${nodeName(graph, schedNode.id)} — wire an LR Scheduler block.`, lossId, 'schedule'));
    }

    // ---- fuse the dense/activation chain into engine architecture --------
    const architecture = [];
    const chain = inference.chain;
    for (let i = 0; i < chain.length; i++) {
      const node = graph.getNode(chain[i]);
      if (!node) continue;
      if (node.type === 'dense') {
        let activation = 'linear';
        const next = i + 1 < chain.length ? graph.getNode(chain[i + 1]) : null;
        if (next && ACTIVATION_NODE_TYPES.includes(next.type)) {
          activation = getNodeType(next.type).activation;
          i++; // consumed by this dense layer
        }
        architecture.push({ size: node.params.size, activation });
      }
    }

    // Engine architecture convention: the FIRST entry is the input layer — the flattened
    // feature vector the engine actually consumes (e.g. Flatten's 784 for a 28×28 grid).
    const sourceShape = inference.shapes.get(sourceId) || null;
    if (sourceShape && architecture.length > 0) {
      const flatInputSize = sourceShape.reduce((acc, dim) => acc * dim, 1);
      architecture.unshift({ size: flatInputSize });
    }

    if (errors.length === 0) {
      if (architecture.length === 0) {
        errors.push(err('E_NO_LOSS', 'The chain has no Dense layers — a graph needs at least one Dense output.'));
      } else {
        // ---- dataset plan (source + optional class filter) ----------------
        const sourceNode = graph.getNode(sourceId);
        const filterNode = chain.map((id) => graph.getNode(id)).find((n) => n && n.type === 'class_filter');
        if (sourceNode.type === 'source') {
          data = { sourceType: 'source', datasetId: sourceNode.params.datasetId, filter: filterNode ? { keep: parseKeepList(filterNode.params.keep) } : null };
        } else if (sourceNode.type === 'sql_source') {
          data = { sourceType: 'sql', db: sourceNode.params.db, query: sourceNode.params.query, filter: filterNode ? { keep: parseKeepList(filterNode.params.keep) } : null };
        } else {
          data = { sourceType: 'api', url: sourceNode.params.url, filter: filterNode ? { keep: parseKeepList(filterNode.params.keep) } : null };
        }

        const view = inference.views.get(lossId) || null;
        network = {
          architecture,
          loss: lossDef.loss,
          optimizer: getNodeType(optNode.type).optimizer,
          learningRate: optNode.params.learningRate,
          weightDecay: optNode.params.weightDecay
        };

        training = {
          mode: loopNode.params.mode,
          maxEpochs: loopNode.params.mode === 'fixed' ? loopNode.params.maxEpochs : null,
          earlyStopPatience: loopNode.params.mode === 'auto_stop' ? loopNode.params.patience : null,
          batchSize: loopNode.params.batchSize,
          lrSchedule: schedule
        };

        taps = {
          gauges: graph.nodeList().filter((n) => n.type === 'metrics_gauge' && graph.incoming(n.id, 'in')).map((n) => n.id),
          confusion: graph.nodeList().some((n) => n.type === 'confusion_matrix' && graph.incoming(n.id, 'in')),
          predict: graph.nodeList().some((n) => n.type === 'prediction_view' && graph.incoming(n.id, 'in'))
        };

        const inputShape = inference.shapes.get(sourceId) || null;
        return {
          ok: true,
          errors: [],
          warnings,
          network,
          training,
          data,
          taps,
          chain,
          meta: {
            inputShape,
            flatInputSize: architecture[0].size,
            inputKind: inputShape && inputShape.length === 2 ? 'image' : 'table',
            outputSize: architecture[architecture.length - 1].size,
            effectiveClasses: view ? view.numClasses : null,
            datasetName: view ? view.name : null
          }
        };
      }
    }
  }

  return { ok: false, errors, warnings, network, training: training, data, taps, chain: inference.chain, meta: null };
}

/* ------------------------------------------------- class filtering */

/**
 * Keep a subset of classes and remap the one-hot targets — used by the Class Filter block.
 * Works on both split corpora (trainX/trainY/testX/testY, e.g. MNIST) and X/y tables.
 */
export function applyClassFilter(dataset, keep) {
  if (dataset.type !== 'classification') {
    throw new Error('Class Filter requires a classification dataset.');
  }
  const keepSet = new Set(keep);

  const filterSplit = (X, y) => {
    const outX = [];
    const outY = [];
    for (let i = 0; i < y.length; i++) {
      const row = y[i];
      let cls = 0;
      for (let c = 1; c < row.length; c++) if (row[c] > row[cls]) cls = c;
      if (!keepSet.has(cls)) continue;
      const mapped = new Array(keep.length).fill(0);
      mapped[keep.indexOf(cls)] = 1;
      outX.push(X[i]);
      outY.push(mapped);
    }
    return { outX, outY };
  };

  const filtered = {
    ...dataset,
    name: `${dataset.name} [${keep.join(',')}]`,
    numClasses: keep.length,
    targetNames: keep.map((i) => (dataset.targetNames && dataset.targetNames[i]) || `Class ${i}`)
  };

  if (Array.isArray(dataset.trainX) && Array.isArray(dataset.trainY)) {
    const train = filterSplit(dataset.trainX, dataset.trainY);
    const test = filterSplit(dataset.testX, dataset.testY);
    filtered.trainX = train.outX;
    filtered.trainY = train.outY;
    filtered.testX = test.outX;
    filtered.testY = test.outY;
  } else {
    const all = filterSplit(dataset.X, dataset.y);
    filtered.X = all.outX;
    filtered.y = all.outY;
  }
  return filtered;
}

/* ------------------------------------------------ lr schedule math */

/** Multiplier for the optimizer learning rate at a given epoch (1-based epoch). */
export function schedulerValue(schedule, epoch) {
  if (!schedule || schedule.mode === 'constant') return 1;
  if (schedule.mode === 'step') {
    const steps = Math.floor((Math.max(1, epoch) - 1) / Math.max(1, schedule.every));
    return Math.pow(schedule.factor, steps);
  }
  return 1;
}
