/**
 * Node Studio — Smart Assist (Beginner Mode).
 *
 * Every operation in this module is tolerant by contract: it repairs what can be repaired,
 * reports what it did (a human-readable action log the canvas prints to the repair console),
 * and NEVER throws into the UI. Structural violations that cannot be honestly fixed
 * (unsupported blocks, unresolved data) are returned as remaining errors — never faked.
 *
 * File: src/graph/smart_assist.js
 */

import { GraphState } from './graph_state.js';
import { getNodeType, SOURCE_NODE_TYPES, LOSS_NODE_TYPES, OPTIMIZER_NODE_TYPES, ACTIVATION_NODE_TYPES } from './node_types.js';
import { inferGraph, parseKeepList } from './node_engine.js';

/* ------------------------------------------------------------- utilities */

function action(code, message, nodeId = null) {
  return { code, message, nodeId };
}

function tensorOut(node) {
  const def = getNodeType(node.type);
  return (def && def.outputs ? def.outputs.find((s) => s.kind === 'tensor') : null) || null;
}

function tensorIn(node) {
  const def = getNodeType(node.type);
  return (def && def.inputs ? def.inputs.find((s) => s.kind === 'tensor') : null) || null;
}

/** Insert a Flatten block on the wire fromRef → toRef (toRef.in must exist). Rolls back on failure. */
function insertFlattenBetween(graph, fromRef, toRef, actions) {
  const flatten = graph.addNode('flatten', { x: 0, y: 0 });
  const originalConn = graph.incoming(toRef.nodeId, toRef.socket);
  if (originalConn) {
    graph.disconnect(originalConn.id);
  }

  const first = graph.connect(fromRef.nodeId, fromRef.socket, flatten.id, 'in');
  if (!first.ok) {
    graph.removeNode(flatten.id);
    if (originalConn) graph.connect(fromRef.nodeId, fromRef.socket, toRef.nodeId, toRef.socket);
    return null;
  }
  const second = graph.connect(flatten.id, 'out', toRef.nodeId, toRef.socket);
  if (!second.ok) {
    graph.disconnect(first.connection.id);
    graph.removeNode(flatten.id);
    if (originalConn) graph.connect(fromRef.nodeId, fromRef.socket, toRef.nodeId, toRef.socket);
    return null;
  }
  actions.push(action('A_INSERT_FLATTEN', `Inserted Flatten between ${fromRef.nodeId} and ${toRef.nodeId} (rank resolution).`, flatten.id));
  return flatten;
}

/* -------------------------------------------------------- assist connect */

/**
 * Beginner-mode connect: connect the two sockets, repairing rank violations by inserting
 * a Flatten, and replacing an existing wire into the target input when one is present.
 * Kind mismatches / cycles are refused with a precise reason (never repaired silently).
 */
export function assistConnect(graph, fromRef, toRef, context = {}) {
  const actions = [];
  try {
    const fromNode = graph.getNode(fromRef.nodeId);
    const toNode = graph.getNode(toRef.nodeId);
    if (!fromNode || !toNode) {
      return { ok: false, error: { code: 'E_BAD_NODE', message: 'Wire endpoints must exist.' }, actions };
    }

    const outDef = graph.socketDef(fromRef.nodeId, fromRef.socket, 'out');
    const inDef = graph.socketDef(toRef.nodeId, toRef.socket, 'in');
    if (!outDef || !inDef) {
      return { ok: false, error: { code: 'E_BAD_SOCKET', message: 'Unknown socket.' }, actions };
    }
    if (outDef.kind !== inDef.kind) {
      return {
        ok: false,
        error: { code: 'E_KIND_MISMATCH', message: `Cannot wire a ${outDef.kind} output into a ${inDef.kind} input.` },
        actions
      };
    }

    // Beginner mode replaces an occupied input (the newest gesture wins) — logged.
    const occupied = graph.incoming(toRef.nodeId, toRef.socket);
    if (occupied) {
      graph.disconnect(occupied.id);
      actions.push(action('A_REPLACED_WIRE', `Replaced the previous wire into ${toNode.type}.${toRef.socket}.`, toRef.nodeId));
    }

    // Rank repair: a rank-2 tensor into a rank-1 consumer (Dense or Loss input) is
    // flattened via an inserted Flatten block — the "28×28 → 784" auto-resolution.
    if ((toNode.type === 'dense' || LOSS_NODE_TYPES.includes(toNode.type)) && outDef.kind === 'tensor') {
      const inference = inferGraph(graph, context);
      const sourceShape = inference.shapes.get(fromRef.nodeId) || null;
      if (sourceShape && sourceShape.length !== 1) {
        const flatten = insertFlattenBetween(graph, fromRef, toRef, actions);
        if (flatten) return { ok: true, connection: graph.incoming(toRef.nodeId, toRef.socket), actions };
      }
    }

    const result = graph.connect(fromRef.nodeId, fromRef.socket, toRef.nodeId, toRef.socket);
    if (!result.ok) {
      // Occupied again (race) or cycle — surface the real reason.
      return { ok: false, error: result.error, actions };
    }
    return { ok: true, connection: result.connection, actions };
  } catch (err) {
    return { ok: false, error: { code: 'E_ASSIST_INTERNAL', message: err.message }, actions };
  }
}

/* ---------------------------------------------------------- drop wiring */

/** The chain end feeding nothing yet: the node the next layer should attach after. */
function chainEndFor(graph, context, targetRank = 1) {
  const inference = inferGraph(graph, context);
  const sourceIds = graph.nodeList().filter((n) => SOURCE_NODE_TYPES.includes(n.type)).map((n) => n.id);
  if (sourceIds.length === 0) return null;

  const lossIds = graph.nodeList().filter((n) => LOSS_NODE_TYPES.includes(n.type)).map((n) => n.id);
  const lossId = lossIds[0] || null;

  // Walk backwards from the loss to find the current chain head and tail.
  if (lossId) {
    const conn = graph.incoming(lossId, 'in');
    if (conn) return { nodeId: conn.from.nodeId, socket: conn.from.socket, shape: inference.shapes.get(conn.from.nodeId) };
  }

  // No loss attached: take the deepest reachable node from the source.
  let current = sourceIds[0];
  let guard = 0;
  while (guard++ < 100) {
    const outs = graph.outgoing(current);
    const consumer = outs.find((c) => graph.getNode(c.to.nodeId) && tensorIn(graph.getNode(c.to.nodeId)));
    if (!consumer) break;
    current = consumer.to.nodeId;
  }
  const d = getNodeType(graph.getNode(current).type);
  const outSocket = (d.outputs || []).find((s) => s.kind === 'tensor');
  return outSocket ? { nodeId: current, socket: outSocket.name, shape: inference.shapes.get(current) } : null;
}

/**
 * Wire a freshly dropped block into the graph the way a beginner expects:
 * layers extend the chain, loss/optimizer/loop attach to their sockets, gauges tap the loss.
 * Falls back to a full autoFixGraph pass; returns the union of actions.
 */
export function autoWireOnDrop(graph, nodeId, context = {}) {
  const actions = [];
  try {
    const node = graph.getNode(nodeId);
    if (!node) return { actions: [action('A_DROP_MISSING', 'Dropped node no longer exists.')] };

    const lossIds = graph.nodeList().filter((n) => LOSS_NODE_TYPES.includes(n.type)).map((n) => n.id);
    const lossId = lossIds[0] || null;

    if (SOURCE_NODE_TYPES.includes(node.type)) {
      // Feed the existing chain head (or the loss directly when the canvas is otherwise empty).
      const head = chainEndFor(graph, context);
      if (head) {
        const res = assistConnect(graph, { nodeId: node.id, socket: 'out' }, { nodeId: head.nodeId, socket: tensorIn(graph.getNode(head.nodeId))?.name || 'in' }, context);
        if (res.ok) actions.push(...res.actions, action('A_AUTO_WIRE', `Connected ${node.type} → ${head.nodeId}.`, node.id));
        return { actions, ...autoFixGraph(graph, context) };
      }
      if (lossId) {
        const res = assistConnect(graph, { nodeId: node.id, socket: 'out' }, { nodeId: lossId, socket: 'in' }, context);
        if (res.ok) actions.push(...res.actions, action('A_AUTO_WIRE', `Connected ${node.type} → loss.`, node.id));
      }
      return { actions, ...autoFixGraph(graph, context) };
    }

    const inSocket = tensorIn(node);
    if (inSocket) {
      const head = chainEndFor(graph, context);
      if (head) {
        const res = assistConnect(graph, { nodeId: head.nodeId, socket: head.socket }, { nodeId: node.id, socket: inSocket.name }, context);
        if (res.ok) actions.push(...res.actions, action('A_AUTO_WIRE', `Connected chain end → ${node.type}.`, node.id));
      }
      if (lossId && ACTIVATION_NODE_TYPES.includes(node.type)) {
        // An activation dropped standalone should sit right before the loss; assistConnect handles it.
        const upstream = graph.incoming(node.id, inSocket.name);
        if (!upstream) {
          const lossIn = graph.incoming(lossId, 'in');
          if (lossIn) {
            const res = assistConnect(graph, { nodeId: lossIn.from.nodeId, socket: lossIn.from.socket }, { nodeId: node.id, socket: 'in' }, context);
            if (res.ok) actions.push(...res.actions);
          }
        }
      }
      return { actions, ...autoFixGraph(graph, context) };
    }

    if (LOSS_NODE_TYPES.includes(node.type) && lossId === node.id) {
      const head = chainEndFor(graph, context);
      if (head) {
        const res = assistConnect(graph, { nodeId: head.nodeId, socket: head.socket }, { nodeId: node.id, socket: 'in' }, context);
        if (res.ok) actions.push(...res.actions, action('A_AUTO_WIRE', `Connected chain end → ${node.type}.`, node.id));
      }
      return { actions, ...autoFixGraph(graph, context) };
    }

    if (OPTIMIZER_NODE_TYPES.includes(node.type) || node.type === 'epoch_loop' || node.type === 'lr_schedule') {
      if (lossId) {
        const socket = OPTIMIZER_NODE_TYPES.includes(node.type) ? 'opt' : node.type === 'epoch_loop' ? 'loop' : 'schedule';
        const res = graph.connect(node.id, 'cfg', lossId, socket);
        if (res.ok) actions.push(action('A_AUTO_WIRE', `Attached ${node.type} to the loss "${socket}" socket.`, node.id));
      }
      return { actions, ...autoFixGraph(graph, context) };
    }

    if (['metrics_gauge', 'confusion_matrix', 'prediction_view'].includes(node.type)) {
      if (lossId) {
        const res = graph.connect(lossId, 'monitor', node.id, 'in');
        if (res.ok) actions.push(action('A_AUTO_WIRE', `Tapped ${node.type} onto the loss monitor.`, node.id));
      }
      return { actions };
    }

    return { actions, ...autoFixGraph(graph, context) };
  } catch (err) {
    return { actions: [...actions, action('E_ASSIST_INTERNAL', err.message)] };
  }
}

/* ------------------------------------------------------------- auto fix */

function addDefaultOptimizer(graph, lossId, actions) {
  const optimizer = graph.addNode('adam_optimizer', { x: 0, y: 0 });
  const res = graph.connect(optimizer.id, 'cfg', lossId, 'opt');
  if (res.ok) {
    actions.push(action('A_ADD_OPTIMIZER', 'Attached a default Adam optimizer.', optimizer.id));
  } else {
    graph.removeNode(optimizer.id);
  }
}

function addDefaultLoop(graph, lossId, actions) {
  const loop = graph.addNode('epoch_loop', { x: 0, y: 0 });
  const res = graph.connect(loop.id, 'cfg', lossId, 'loop');
  if (res.ok) {
    actions.push(action('A_ADD_LOOP', 'Attached a default Epoch Loop (auto-stop, batch 16).', loop.id));
  } else {
    graph.removeNode(loop.id);
  }
}

function defaultLossTypeFor(view) {
  if (!view) return 'mse_loss';
  if (view.type === 'regression') return 'mse_loss';
  return view.numClasses === 2 ? 'bce_loss' : 'cce_loss';
}

function requiredActivationFor(lossType) {
  if (lossType === 'bce_loss') return 'sigmoid';
  if (lossType === 'cce_loss') return 'softmax';
  return 'linear';
}

/**
 * One idempotent repair pass over the whole graph. Returns:
 *   { actions, remainingErrors, remainingWarnings, inference }
 * The canvas loops this until the action list stops changing (max 3 passes) or a pass
 * produces no actions — a convergent, predictable repair machine.
 */
export function autoFixGraph(graph, context = {}) {
  const actions = [];
  try {
    /* 1. input sockets accept exactly one wire — keep the first, drop the rest. */
    const perInput = new Map();
    for (const conn of graph.connections.values()) {
      const key = `${conn.to.nodeId}:${conn.to.socket}`;
      if (!perInput.has(key)) perInput.set(key, []);
      perInput.get(key).push(conn);
    }
    for (const [key, list] of perInput.entries()) {
      for (let i = 1; i < list.length; i++) {
        graph.disconnect(list[i].id);
        actions.push(action('A_DEDUP_WIRE', `Removed a duplicate wire into ${key}.`, list[i].to.nodeId));
      }
    }

    /* 2. exactly one source; none → add the MNIST default. */
    let sourceIds = graph.nodeList().filter((n) => SOURCE_NODE_TYPES.includes(n.type)).map((n) => n.id);
    if (sourceIds.length === 0) {
      const source = graph.addNode('source', { params: { datasetId: 'mnist_01' } });
      sourceIds = [source.id];
      actions.push(action('A_ADD_SOURCE', 'No data source found — added the MNIST 0-vs-1 corpus as the default.', source.id));
    } else if (sourceIds.length > 1) {
      for (const extra of sourceIds.slice(1)) {
        graph.removeNode(extra);
        actions.push(action('A_DROP_SOURCE', `Removed extra data source ${extra} (one dataset per graph).`, extra));
      }
      sourceIds = [sourceIds[0]];
    }
    const sourceId = sourceIds[0];

    /* 3. exactly one loss; none → create the pair that fits the dataset. */
    let lossIds = graph.nodeList().filter((n) => LOSS_NODE_TYPES.includes(n.type)).map((n) => n.id);
    if (lossIds.length === 0) {
      const sourceNode = graph.getNode(sourceId);
      const info = context.datasetInfoFor ? context.datasetInfoFor(sourceNode) : null;
      const lossType = defaultLossTypeFor(info);
      const loss = graph.addNode(lossType, { x: (sourceNode.x || 0) + 560, y: sourceNode.y || 0 });
      lossIds = [loss.id];
      actions.push(action('A_ADD_LOSS', `Added the matching ${lossType} terminal for this dataset.`, loss.id));
    } else if (lossIds.length > 1) {
      const connected = lossIds.find((id) => graph.incoming(id, 'in'));
      const keep = connected || lossIds[0];
      for (const extra of lossIds.filter((id) => id !== keep)) {
        graph.removeNode(extra);
        actions.push(action('A_DROP_LOSS', `Removed extra loss ${extra} (one terminal per graph).`, extra));
      }
      lossIds = [keep];
    }
    const lossId = lossIds[0];
    const lossNode = graph.getNode(lossId);

    /* 4. connect source side to the loss side. */
    if (!graph.canReach(sourceId, lossId)) {
      const lossIn = graph.incoming(lossId, 'in');
      if (lossIn) {
        // A partial chain hangs off the loss — attach the source to its root.
        let root = lossIn.from;
        const guard = new Set();
        while (guard.size < 200) {
          guard.add(root.nodeId);
          const incoming = graph.incoming(root.nodeId, 'in');
          if (!incoming) break;
          root = incoming.from;
        }
        if (root.nodeId !== sourceId) {
          const res = assistConnect(graph, { nodeId: sourceId, socket: 'out' }, { nodeId: root.nodeId, socket: 'in' }, context);
          if (res.ok) {
            actions.push(...res.actions, action('A_AUTO_WIRE', `Connected the source into the chain root (${root.nodeId}).`, sourceId));
          }
        }
      } else {
        const head = chainEndFor(graph, context);
        const res = head
          ? assistConnect(graph, { nodeId: head.nodeId, socket: head.socket }, { nodeId: lossId, socket: 'in' }, context)
          : assistConnect(graph, { nodeId: sourceId, socket: 'out' }, { nodeId: lossId, socket: 'in' }, context);
        if (res.ok) actions.push(...res.actions, action('A_AUTO_WIRE', 'Connected the chain into the loss.', lossId));
      }
    }

    /* 5. two activations in a row → the earlier one is removed (keep last = effective). */
    let inference = inferGraph(graph, context);
    for (const e of inference.errors) {
      if (e.code === 'E_PAIRING' && e.message.includes('Two Activation blocks in a row')) {
        const match = e.message.match(/\((\w+) → (\w+)\)/);
        if (match && graph.getNode(match[1])) {
          const earlier = graph.getNode(match[1]);
          const incoming = graph.incoming(earlier.id, 'in');
          const consumers = graph.outgoing(earlier.id);
          graph.removeNode(earlier.id);
          if (incoming) {
            for (const consumer of consumers) {
              graph.connect(incoming.from.nodeId, incoming.from.socket, consumer.to.nodeId, consumer.to.socket);
            }
          }
          actions.push(action('A_MERGE_ACTIVATION', `Removed overridden activation ${match[1]} (kept ${match[2]}).`, match[1]));
        }
      }
    }

    /* 6. loss pairing: ensure the correct final activation sits right before the loss. */
    inference = inferGraph(graph, context);
    const pairingErrors = inference.errors.filter((e) => e.code === 'E_PAIRING' && e.nodeId === lossId && (e.message.includes('requires a') || e.message.includes('must be fed')));
    for (const e of pairingErrors) {
      const needed = requiredActivationFor(lossNode.type);
      const lossIn = graph.incoming(lossId, 'in');
      if (!lossIn) break;
      const upstream = graph.getNode(lossIn.from.nodeId);
      if (upstream && upstream.type === 'dense') {
        // Insert the required activation between the dense and the loss.
        const act = graph.addNode(needed, { x: upstream.x + 80, y: upstream.y + 60 });
        const from = { nodeId: upstream.id, socket: lossIn.from.socket };
        const to = { nodeId: lossId, socket: 'in' };
        graph.disconnect(lossIn.id);
        const a = graph.connect(from.nodeId, from.socket, act.id, 'in');
        const b = a.ok ? graph.connect(act.id, 'out', to.nodeId, to.socket) : a;
        if (b.ok) {
          actions.push(action('A_ADD_ACTIVATION', `Inserted ${needed} before the loss (loss pairing).`, act.id));
        } else {
          if (a.ok) graph.disconnect(a.connection.id);
          graph.removeNode(act.id);
          graph.connect(from.nodeId, from.socket, to.nodeId, to.socket);
        }
      } else if (upstream && ACTIVATION_NODE_TYPES.includes(upstream.type) && getNodeType(upstream.type).activation !== needed) {
        // Wrong-kind activation: replace it with the required one. The old wires are
        // detached FIRST (the replacement must be able to claim the sockets), and the
        // original state is restored if any reconnect fails.
        const act = graph.addNode(needed, { x: upstream.x, y: upstream.y });
        const inConn = graph.incoming(upstream.id, 'in');
        const outConns = graph.outgoing(upstream.id);
        if (inConn) graph.disconnect(inConn.id);
        for (const c of outConns) graph.disconnect(c.id);
        let ok = true;
        if (inConn) {
          const r = graph.connect(inConn.from.nodeId, inConn.from.socket, act.id, 'in');
          ok = ok && r.ok;
        }
        for (const c of outConns) {
          const r = graph.connect(act.id, 'out', c.to.nodeId, c.to.socket);
          ok = ok && r.ok;
        }
        if (ok) {
          graph.removeNode(upstream.id);
          actions.push(action('A_REPLACE_ACTIVATION', `Replaced ${upstream.type} with ${needed} before the loss.`, act.id));
        } else {
          graph.removeNode(act.id);
          if (inConn) graph.connect(inConn.from.nodeId, inConn.from.socket, upstream.id, 'in');
          for (const c of outConns) graph.connect(upstream.id, 'out', c.to.nodeId, c.to.socket);
        }
      }
      break; // one pairing repair per pass — the loop re-runs inference next pass
    }

    /* 7. output size: set the last Dense to what the loss/dataset actually needs. */
    inference = inferGraph(graph, context);
    const sizeErrors = inference.errors.filter((e) => e.code === 'E_SIZE_MISMATCH' && e.nodeId === lossId);
    if (sizeErrors.length > 0) {
      const view = inference.views.get(lossId) || null;
      let expected = null;
      if (lossNode.type === 'bce_loss') expected = 2;
      else if (lossNode.type === 'cce_loss') expected = view ? view.numClasses : null;
      else if (lossNode.type === 'mse_loss' && view && view.type === 'regression') expected = 1;
      const lastDense = [...inference.chain].reverse().map((id) => graph.getNode(id)).find((n) => n && n.type === 'dense');
      if (expected !== null && lastDense && lastDense.params.size !== expected) {
        const old = lastDense.params.size;
        lastDense.params = { ...lastDense.params, size: expected };
        actions.push(action('A_FIX_SIZE', `Resized the output Dense from ${old} to ${expected} values (loss requirement).`, lastDense.id));
      }
    }

    /* 8. rank violations: insert Flatten wherever a Dense receives a rank-2 tensor. */
    inference = inferGraph(graph, context);
    for (const e of inference.errors) {
      if (e.code === 'E_RANK_MISMATCH' && e.nodeId) {
        const denseNode = graph.getNode(e.nodeId);
        if (denseNode && denseNode.type === 'dense') {
          const conn = graph.incoming(denseNode.id, 'in');
          if (conn) {
            const inserted = insertFlattenBetween(graph, conn.from, conn.to, actions);
            if (inserted) break; // one structural repair per pass
          }
        }
      }
    }

    /* 9. loss config sockets: attach defaults when empty. */
    inference = inferGraph(graph, context);
    if (!graph.incoming(lossId, 'opt')) addDefaultOptimizer(graph, lossId, actions);
    if (!graph.incoming(lossId, 'loop')) addDefaultLoop(graph, lossId, actions);

    /* 10. finally connect any still-dangling chain end into the loss. */
    inference = inferGraph(graph, context);
    const lossMissing = inference.errors.some((e) => e.code === 'E_MISSING_INPUT' && e.nodeId === lossId);
    if (lossMissing) {
      const head = chainEndFor(graph, context);
      if (head) {
        const res = assistConnect(graph, { nodeId: head.nodeId, socket: head.socket }, { nodeId: lossId, socket: 'in' }, context);
        if (res.ok) actions.push(...res.actions, action('A_AUTO_WIRE', 'Connected the chain into the loss.', lossId));
      }
    }

    inference = inferGraph(graph, context);
    return {
      actions,
      remainingErrors: inference.errors,
      remainingWarnings: inference.warnings,
      inference
    };
  } catch (err) {
    return {
      actions: [...actions, action('E_ASSIST_INTERNAL', `Assist error: ${err.message}`)],
      remainingErrors: [{ code: 'E_ASSIST_INTERNAL', message: err.message, nodeId: null, socket: null }],
      remainingWarnings: [],
      inference: null
    };
  }
}

/** Repeat autoFixGraph until it converges (bounded); returns the union of actions. */
export function autoFixGraphToConvergence(graph, context = {}, maxPasses = 3) {
  let actions = [];
  let lastResult = null;
  for (let pass = 0; pass < maxPasses; pass++) {
    const result = autoFixGraph(graph, context);
    actions = actions.concat(result.actions);
    lastResult = result;
    if (result.actions.length === 0) break;
  }
  return { ...lastResult, actions };
}

/* ------------------------------------------------------------ presets */

/**
 * One-click working graphs ("AI in 30 seconds"). Coordinates are real canvas layout
 * positions, connections are complete and compile cleanly with zero repairs.
 */
export function instantPreset(name) {
  const graph = new GraphState('assist');

  if (name === 'mnist_classifier') {
    const source = graph.addNode('source', { x: 40, y: 200, params: { datasetId: 'mnist_01' } });
    const flatten = graph.addNode('flatten', { x: 260, y: 200 });
    const d1 = graph.addNode('dense', { x: 420, y: 160, params: { size: 16 } });
    const a1 = graph.addNode('relu', { x: 560, y: 160 });
    const d2 = graph.addNode('dense', { x: 680, y: 160, params: { size: 8 } });
    const a2 = graph.addNode('relu', { x: 820, y: 160 });
    const d3 = graph.addNode('dense', { x: 940, y: 160, params: { size: 2 } });
    const a3 = graph.addNode('sigmoid', { x: 1080, y: 160 });
    const loss = graph.addNode('bce_loss', { x: 1220, y: 160 });
    const opt = graph.addNode('adam_optimizer', { x: 1220, y: 320 });
    const loop = graph.addNode('epoch_loop', { x: 1220, y: 460, params: { mode: 'auto_stop', patience: 150, batchSize: 16 } });
    const gauge = graph.addNode('metrics_gauge', { x: 1440, y: 120 });
    const conf = graph.addNode('confusion_matrix', { x: 1440, y: 280 });
    const probe = graph.addNode('prediction_view', { x: 1440, y: 440 });

    graph.connect(source.id, 'out', flatten.id, 'in');
    graph.connect(flatten.id, 'out', d1.id, 'in');
    graph.connect(d1.id, 'out', a1.id, 'in');
    graph.connect(a1.id, 'out', d2.id, 'in');
    graph.connect(d2.id, 'out', a2.id, 'in');
    graph.connect(a2.id, 'out', d3.id, 'in');
    graph.connect(d3.id, 'out', a3.id, 'in');
    graph.connect(a3.id, 'out', loss.id, 'in');
    graph.connect(opt.id, 'cfg', loss.id, 'opt');
    graph.connect(loop.id, 'cfg', loss.id, 'loop');
    graph.connect(loss.id, 'monitor', gauge.id, 'in');
    graph.connect(loss.id, 'monitor', conf.id, 'in');
    graph.connect(loss.id, 'monitor', probe.id, 'in');
    return graph;
  }

  if (name === 'xor_classifier') {
    const source = graph.addNode('source', { x: 40, y: 200, params: { datasetId: 'xor' } });
    const d1 = graph.addNode('dense', { x: 300, y: 160, params: { size: 8 } });
    const a1 = graph.addNode('relu', { x: 460, y: 160 });
    const d2 = graph.addNode('dense', { x: 600, y: 160, params: { size: 6 } });
    const a2 = graph.addNode('relu', { x: 760, y: 160 });
    const d3 = graph.addNode('dense', { x: 880, y: 160, params: { size: 2 } });
    const a3 = graph.addNode('sigmoid', { x: 1020, y: 160 });
    const loss = graph.addNode('bce_loss', { x: 1160, y: 160 });
    const opt = graph.addNode('adam_optimizer', { x: 1160, y: 320 });
    const loop = graph.addNode('epoch_loop', { x: 1160, y: 460, params: { mode: 'fixed', maxEpochs: 40, patience: 150, batchSize: 16 } });
    const gauge = graph.addNode('metrics_gauge', { x: 1380, y: 200 });

    graph.connect(source.id, 'out', d1.id, 'in');
    graph.connect(d1.id, 'out', a1.id, 'in');
    graph.connect(a1.id, 'out', d2.id, 'in');
    graph.connect(d2.id, 'out', a2.id, 'in');
    graph.connect(a2.id, 'out', d3.id, 'in');
    graph.connect(d3.id, 'out', a3.id, 'in');
    graph.connect(a3.id, 'out', loss.id, 'in');
    graph.connect(opt.id, 'cfg', loss.id, 'opt');
    graph.connect(loop.id, 'cfg', loss.id, 'loop');
    graph.connect(loss.id, 'monitor', gauge.id, 'in');
    return graph;
  }

  if (name === 'iris_multi_class') {
    const source = graph.addNode('source', { x: 40, y: 200, params: { datasetId: 'iris' } });
    const d1 = graph.addNode('dense', { x: 300, y: 160, params: { size: 12 } });
    const a1 = graph.addNode('relu', { x: 460, y: 160 });
    const d2 = graph.addNode('dense', { x: 600, y: 160, params: { size: 3 } });
    const a2 = graph.addNode('softmax', { x: 760, y: 160 });
    const loss = graph.addNode('cce_loss', { x: 900, y: 160 });
    const opt = graph.addNode('adam_optimizer', { x: 900, y: 320 });
    const loop = graph.addNode('epoch_loop', { x: 900, y: 460, params: { mode: 'auto_stop', patience: 150, batchSize: 16 } });
    const gauge = graph.addNode('metrics_gauge', { x: 1120, y: 200 });

    graph.connect(source.id, 'out', d1.id, 'in');
    graph.connect(d1.id, 'out', a1.id, 'in');
    graph.connect(a1.id, 'out', d2.id, 'in');
    graph.connect(d2.id, 'out', a2.id, 'in');
    graph.connect(a2.id, 'out', loss.id, 'in');
    graph.connect(opt.id, 'cfg', loss.id, 'opt');
    graph.connect(loop.id, 'cfg', loss.id, 'loop');
    graph.connect(loss.id, 'monitor', gauge.id, 'in');
    return graph;
  }

  return null;
}

export function presetList() {
  return [
    { id: 'mnist_classifier', label: 'MNIST 0 vs 1 classifier', hint: '28×28 → Flatten → 16 → 8 → 2, Sigmoid + BCE + Adam.' },
    { id: 'xor_classifier', label: 'XOR classifier', hint: '2 → 8 → 6 → 2, fixed 40-epoch budget.' },
    { id: 'iris_multi_class', label: 'Iris 3-class classifier', hint: '4 features → 12 → 3, Softmax + CCE.' }
  ];
}
