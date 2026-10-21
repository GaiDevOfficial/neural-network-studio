/**
 * Node Studio — graph state container (Node / Socket / Connection invariant keeper).
 *
 * Holds the visual graph and enforces the wiring rules at mutation time:
 *   - sockets must exist and match in kind (tensor / config / monitor),
 *   - every input socket accepts at most ONE incoming wire,
 *   - connections may not create cycles (the graph is a DAG by construction).
 *
 * Pure ES module — no DOM, no engine imports. Exercised by tests/test_graph_engine.js.
 *
 * File: src/graph/graph_state.js
 */

import { getNodeType, defaultParams } from './node_types.js';

export class GraphState {
  constructor(mode = 'assist') {
    this.mode = mode; // 'assist' (Beginner) | 'manual' (Advanced)
    this.nodes = new Map();
    this.connections = new Map();
    this.loadReport = null;
    this._idSeq = 1;
  }

  _nextId(prefix) {
    return `${prefix}${this._idSeq++}`;
  }

  /** Create a node of a declared type. Unknown types are refused loudly. */
  addNode(type, options = {}) {
    const def = getNodeType(type);
    if (!def) throw new Error(`Unknown block type "${type}".`);
    const node = {
      id: options.id || this._nextId('node_'),
      type,
      x: Number.isFinite(options.x) ? options.x : 0,
      y: Number.isFinite(options.y) ? options.y : 0,
      params: defaultParams(type, options.params || {})
    };
    this.nodes.set(node.id, node);
    return node;
  }

  getNode(id) {
    return this.nodes.get(id) || null;
  }

  nodeList() {
    return Array.from(this.nodes.values());
  }

  /** Remove a node and every wire touching it. */
  removeNode(id) {
    if (!this.nodes.has(id)) return false;
    this.nodes.delete(id);
    for (const [connId, conn] of Array.from(this.connections.entries())) {
      if (conn.from.nodeId === id || conn.to.nodeId === id) this.connections.delete(connId);
    }
    return true;
  }

  /** Resolve a socket declaration (or null when the socket does not exist). */
  socketDef(nodeId, socketName, direction) {
    const node = this.nodes.get(nodeId);
    if (!node) return null;
    const def = getNodeType(node.type);
    if (!def) return null;
    const list = direction === 'out' ? def.outputs : def.inputs;
    return (list || []).find((s) => s.name === socketName) || null;
  }

  incoming(nodeId, socketName) {
    for (const conn of this.connections.values()) {
      if (conn.to.nodeId === nodeId && conn.to.socket === socketName) return conn;
    }
    return null;
  }

  outgoing(nodeId, socketName = null) {
    const result = [];
    for (const conn of this.connections.values()) {
      if (conn.from.nodeId === nodeId && (socketName === null || conn.from.socket === socketName)) {
        result.push(conn);
      }
    }
    return result;
  }

  inputsOf(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node) return [];
    const def = getNodeType(node.type);
    return (def.inputs || []).map((spec) => ({ socket: spec, connection: this.incoming(nodeId, spec.name) }));
  }

  /** True when `fromNodeId` can reach `toNodeId` by following outgoing wires. */
  canReach(fromNodeId, toNodeId) {
    const seen = new Set([fromNodeId]);
    const queue = [fromNodeId];
    while (queue.length > 0) {
      const current = queue.shift();
      for (const conn of this.connections.values()) {
        if (conn.from.nodeId === current) {
          if (conn.to.nodeId === toNodeId) return true;
          if (!seen.has(conn.to.nodeId)) {
            seen.add(conn.to.nodeId);
            queue.push(conn.to.nodeId);
          }
        }
      }
    }
    return false;
  }

  /**
   * Attempt to wire `fromNode.outSocket → toNode.inSocket`.
   * Returns { ok: true, connection } or { ok: false, error: { code, message } } — never throws.
   */
  connect(fromNodeId, fromSocket, toNodeId, toSocket) {
    const fromNode = this.nodes.get(fromNodeId);
    const toNode = this.nodes.get(toNodeId);
    if (!fromNode || !toNode) {
      return { ok: false, error: { code: 'E_BAD_NODE', message: 'Both wire endpoints must exist on the canvas.' } };
    }

    const outDef = this.socketDef(fromNodeId, fromSocket, 'out');
    const inDef = this.socketDef(toNodeId, toSocket, 'in');
    if (!outDef || !inDef) {
      return {
        ok: false,
        error: {
          code: 'E_BAD_SOCKET',
          message: `Unknown socket (${fromNode.type}.${fromSocket} → ${toNode.type}.${toSocket}).`
        }
      };
    }

    if (outDef.kind !== inDef.kind) {
      return {
        ok: false,
        error: {
          code: 'E_KIND_MISMATCH',
          message: `Cannot wire ${outDef.kind} output into ${inDef.kind} input (${fromNode.type} → ${toNode.type}).`
        }
      };
    }

    if (fromNodeId === toNodeId) {
      return { ok: false, error: { code: 'E_SELF_LOOP', message: 'A block cannot feed itself.' } };
    }

    const occupied = this.incoming(toNodeId, toSocket);
    if (occupied) {
      return {
        ok: false,
        error: {
          code: 'E_DUP_INPUT',
          message: `${toNode.type}.${toSocket} already has an incoming wire — inputs accept a single connection.`
        }
      };
    }

    if (this.canReach(toNodeId, fromNodeId)) {
      return {
        ok: false,
        error: { code: 'E_CYCLE', message: 'That wire would create a cycle — the graph must stay a DAG.' }
      };
    }

    const connection = {
      id: this._nextId('wire_'),
      from: { nodeId: fromNodeId, socket: fromSocket },
      to: { nodeId: toNodeId, socket: toSocket }
    };
    this.connections.set(connection.id, connection);
    return { ok: true, connection };
  }

  disconnect(connectionId) {
    return this.connections.delete(connectionId);
  }

  /** Kahn topological order; appended run of any residual cycle nodes at the end (cannot happen via connect()). */
  topologicalOrder() {
    const indegree = new Map();
    for (const node of this.nodes.values()) indegree.set(node.id, 0);
    for (const conn of this.connections.values()) {
      if (indegree.has(conn.to.nodeId)) indegree.set(conn.to.nodeId, indegree.get(conn.to.nodeId) + 1);
    }
    const queue = [];
    for (const [id, deg] of indegree.entries()) if (deg === 0) queue.push(id);
    const order = [];
    while (queue.length > 0) {
      const id = queue.shift();
      order.push(id);
      for (const conn of this.connections.values()) {
        if (conn.from.nodeId === id && indegree.has(conn.to.nodeId)) {
          indegree.set(conn.to.nodeId, indegree.get(conn.to.nodeId) - 1);
          if (indegree.get(conn.to.nodeId) === 0) queue.push(conn.to.nodeId);
        }
      }
    }
    if (order.length !== this.nodes.size) {
      for (const id of this.nodes.keys()) if (!order.includes(id)) order.push(id);
    }
    return order;
  }

  clone() {
    const copy = GraphState.fromJSON(this.toJSON());
    copy.loadReport = null;
    return copy;
  }

  toJSON() {
    return {
      version: 1,
      mode: this.mode,
      seq: this._idSeq,
      nodes: this.nodeList().map((n) => ({ id: n.id, type: n.type, x: n.x, y: n.y, params: { ...n.params } })),
      connections: Array.from(this.connections.values()).map((c) => ({
        id: c.id,
        from: { ...c.from },
        to: { ...c.to }
      }))
    };
  }

  /**
   * Rebuild from JSON (graph files saved by the page). Unknown block types and dangling
   * wires are dropped, and the load report records exactly what was dropped — never silent.
   */
  static fromJSON(json) {
    const graph = new GraphState(json && json.mode === 'manual' ? 'manual' : 'assist');
    const droppedNodes = [];
    const droppedConnections = [];

    for (const raw of (json && json.nodes) || []) {
      if (!raw || !raw.id || !getNodeType(raw.type)) {
        droppedNodes.push(raw && raw.type ? `${raw.type}(${raw.id || '?'})` : String(raw && raw.id));
        continue;
      }
      try {
        graph.addNode(raw.type, { id: raw.id, x: raw.x, y: raw.y, params: raw.params });
      } catch (err) {
        droppedNodes.push(`${raw.type}(${raw.id})`);
      }
    }

    for (const raw of (json && json.connections) || []) {
      if (!raw || !raw.from || !raw.to) continue;
      const result = graph.connect(raw.from.nodeId, raw.from.socket, raw.to.nodeId, raw.to.socket);
      if (!result.ok) droppedConnections.push(`${raw.id || '?'} (${result.error.code})`);
    }

    graph._idSeq = Math.max(Number.isFinite(json && json.seq) ? json.seq : 1, graph._idSeq);
    graph.loadReport = { droppedNodes, droppedConnections };
    return graph;
  }
}
