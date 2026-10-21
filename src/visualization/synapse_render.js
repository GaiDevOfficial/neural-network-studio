/**
 * Real Synaptic Communication Visualizer (60 FPS Canvas Engine)
 * Features:
 * - Live Dynamic Weights (Thickness ~ |W_ij|, Color: Cyan > 0, Pink < 0)
 * - Real Node Activations (Glowing intensity proportional to activation a_i)
 * - Forward Signal Pulses along axons
 * - Backward Gradient Counter-Pulses (Reversed flow proportional to |dL/dW|)
 * - Adaptive layout with high-dimension node bundling for 784/64 inputs
 *
 * File: src/visualization/synapse_render.js
 */

export class SynapseParticle {
  constructor() {
    this.active = false;
    this.startX = 0;
    this.startY = 0;
    this.endX = 0;
    this.endY = 0;
    this.progress = 0;
    this.speed = 0.03;
    this.size = 2.5;
    this.color = '#00f0ff';
    this.isGradient = false;
  }

  spawn(x1, y1, x2, y2, color, speed, size, isGradient = false) {
    this.active = true;
    this.startX = x1;
    this.startY = y1;
    this.endX = x2;
    this.endY = y2;
    this.progress = 0;
    this.color = color;
    this.speed = speed;
    this.size = size;
    this.isGradient = isGradient;
  }

  update() {
    if (!this.active) return false;
    this.progress += this.speed;
    if (this.progress >= 1.0) {
      this.active = false;
      return true; // Reached destination
    }
    return false;
  }

  draw(ctx) {
    if (!this.active) return;
    const curX = this.startX + (this.endX - this.startX) * this.progress;
    const curY = this.startY + (this.endY - this.startY) * this.progress;

    ctx.save();
    ctx.shadowColor = this.color;
    ctx.shadowBlur = this.isGradient ? 10 : 7;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(curX, curY, this.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

export class SynapseRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');

    this.architecture = [];
    this.weights = [];
    this.activations = [];
    this.deltas = [];

    this.layers = [];
    this.synapses = [];

    // Particle pool for zero-garbage 60 FPS performance
    this.maxParticles = 500;
    this.particlePool = Array.from({ length: this.maxParticles }, () => new SynapseParticle());

    this.hoveredItem = null;
    this.mousePos = { x: -1000, y: -1000 };
    this.labelMode = 'technical'; // 'technical' | 'friendly'

    this.isLooping = false;
    this.animFrameId = null;

    this.setupEvents();
    this.resize();
  }

  setupEvents() {
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);

    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.mousePos.x = (e.clientX - rect.left);
      this.mousePos.y = (e.clientY - rect.top);
      this.checkHover();
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.mousePos.x = -1000;
      this.mousePos.y = -1000;
      this.hoveredItem = null;
    });
  }

  resize() {
    const parent = this.canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const width = parent ? parent.clientWidth : 800;
    const height = parent ? parent.clientHeight : 500;

    // While the host view is hidden (e.g. the playground before it opens), clientWidth/Height are
    // zero. Writing `0px` inline would then beat the CSS `width:100%;height:100%` and leave the
    // canvas permanently collapsed. Keep the CSS size and wait for a real layout pass instead.
    if (width <= 0 || height <= 0) {
      this.width = width;
      this.height = height;
      return;
    }

    this.width = width;
    this.height = height;
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);

    this.computeLayout();
  }

  setArchitecture(arch, weightsSnapshot = null) {
    this.architecture = arch;
    if (weightsSnapshot) this.weights = weightsSnapshot;
    this.computeLayout();
  }

  /** Beginner playground uses jargon-free labels; the technical studio keeps Input/Output names. */
  setLabelMode(mode) {
    this.labelMode = mode === 'friendly' ? 'friendly' : 'technical';
  }

  dispose() {
    this.stop();
    if (this._onResize) {
      window.removeEventListener('resize', this._onResize);
      this._onResize = null;
    }
    this.synapses = [];
    this.layers = [];
    this.architecture = [];
  }

  updateData(stepData) {
    if (stepData.weights) {
      this.weights = stepData.weights;
      this.syncSynapseWeights();
    }
    if (stepData.activations) {
      this.activations = stepData.activations;
      this.emitForwardPulses();
    }
    if (stepData.deltas) {
      this.deltas = stepData.deltas;
      this.emitBackwardPulses();
    }
  }

  computeLayout() {
    if (!this.architecture || this.architecture.length === 0) return;

    this.layers = [];
    this.synapses = [];

    const numLayers = this.architecture.length;
    const padX = Math.min(100, this.width * 0.12);
    const usableW = this.width - padX * 2;
    const layerSpacing = numLayers > 1 ? usableW / (numLayers - 1) : 0;

    for (let l = 0; l < numLayers; l++) {
      const layerCfg = this.architecture[l];
      const rawCount = layerCfg.size;

      // Adaptive compression for large layers (e.g. 784 inputs for 28x28 or 64 inputs for 8x8)
      const maxDisplayNodes = 14;
      const isCompressed = rawCount > maxDisplayNodes;
      const displayCount = isCompressed ? maxDisplayNodes : rawCount;

      const x = padX + l * layerSpacing;
      const nodeRadius = Math.max(8, Math.min(18, (this.height - 120) / (displayCount * 2.5)));
      const totalH = (displayCount - 1) * (nodeRadius * 2.5);
      const startY = (this.height - totalH) / 2;

      const layerNodes = [];
      for (let n = 0; n < displayCount; n++) {
        // Map display index back to raw index
        const rawIdx = isCompressed
          ? Math.min(rawCount - 1, Math.floor((n / (displayCount - 1)) * (rawCount - 1)))
          : n;

        layerNodes.push({
          layer: l,
          displayIndex: n,
          rawIndex: rawIdx,
          x,
          y: startY + n * (nodeRadius * 2.5),
          radius: nodeRadius,
          activation: 0.0,
          bias: 0.0,
          name: l === 0 ? `Input ${rawIdx}` : (l === numLayers - 1 ? `Output ${rawIdx}` : `H${l}_${rawIdx}`)
        });
      }
      this.layers.push({ nodes: layerNodes, rawCount, isCompressed });
    }

    // Build synapses between visible nodes
    for (let l = 0; l < numLayers - 1; l++) {
      const srcNodes = this.layers[l].nodes;
      const tgtNodes = this.layers[l + 1].nodes;

      for (let s = 0; s < srcNodes.length; s++) {
        for (let t = 0; t < tgtNodes.length; t++) {
          const src = srcNodes[s];
          const tgt = tgtNodes[t];

          let weight = 0.0;
          if (this.weights && this.weights[l] && this.weights[l].weights) {
            const wRow = this.weights[l].weights[tgt.rawIndex];
            if (wRow && wRow[src.rawIndex] !== undefined) {
              weight = wRow[src.rawIndex];
            }
          }

          this.synapses.push({
            layer: l,
            source: src,
            target: tgt,
            weight,
            gradient: 0.0
          });
        }
      }
    }
  }

  syncSynapseWeights() {
    if (!this.weights || this.weights.length === 0) return;
    for (const syn of this.synapses) {
      const l = syn.layer;
      const t = syn.target.rawIndex;
      const s = syn.source.rawIndex;
      if (this.weights[l] && this.weights[l].weights && this.weights[l].weights[t]) {
        syn.weight = this.weights[l].weights[t][s] || 0.0;
      }
    }
  }

  emitForwardPulses() {
    if (!this.activations || this.activations.length === 0) return;

    // Emit forward particles on active synapses
    let spawned = 0;
    for (const syn of this.synapses) {
      if (spawned >= 40) break;
      const actLevel = this.activations[syn.layer]
        ? (this.activations[syn.layer][syn.source.rawIndex] || 0.0)
        : 0.0;

      if (actLevel > 0.15 && Math.random() < 0.25) {
        const p = this.getFreeParticle();
        if (p) {
          p.spawn(
            syn.source.x,
            syn.source.y,
            syn.target.x,
            syn.target.y,
            '#00f0ff',
            0.025 + Math.random() * 0.02,
            2.0 + Math.min(3.0, actLevel * 2),
            false
          );
          spawned++;
        }
      }
    }
  }

  emitBackwardPulses() {
    if (!this.deltas || this.deltas.length === 0) return;

    let spawned = 0;
    for (const syn of this.synapses) {
      if (spawned >= 35) break;
      const nextDelta = this.deltas[syn.layer]
        ? (this.deltas[syn.layer][syn.target.rawIndex] || 0.0)
        : 0.0;

      const gradMag = Math.abs(nextDelta * (syn.weight || 0.5));
      if (gradMag > 0.05 && Math.random() < 0.3) {
        const p = this.getFreeParticle();
        if (p) {
          // Flow backward: target -> source
          p.spawn(
            syn.target.x,
            syn.target.y,
            syn.source.x,
            syn.source.y,
            '#ff007f', // Hot magenta gradient pulse
            0.03 + Math.random() * 0.025,
            2.5 + Math.min(3.0, gradMag * 3),
            true
          );
          spawned++;
        }
      }
    }
  }

  getFreeParticle() {
    for (let i = 0; i < this.particlePool.length; i++) {
      if (!this.particlePool[i].active) return this.particlePool[i];
    }
    return null;
  }

  checkHover() {
    this.hoveredItem = null;

    // Check nodes
    for (const layer of this.layers) {
      for (const node of layer.nodes) {
        const dx = this.mousePos.x - node.x;
        const dy = this.mousePos.y - node.y;
        if (dx * dx + dy * dy <= (node.radius + 4) * (node.radius + 4)) {
          this.hoveredItem = { type: 'node', data: node };
          return;
        }
      }
    }

    // Check synapses
    for (const syn of this.synapses) {
      const d = this.distToSegment(this.mousePos, syn.source, syn.target);
      if (d < 5) {
        this.hoveredItem = { type: 'synapse', data: syn };
        return;
      }
    }
  }

  distToSegment(p, v, w) {
    const l2 = (v.x - w.x) * (v.x - w.x) + (v.y - w.y) * (v.y - w.y);
    if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
    let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
    t = Math.max(0, Math.min(1, t));
    const projX = v.x + t * (w.x - v.x);
    const projY = v.y + t * (w.y - v.y);
    return Math.hypot(p.x - projX, p.y - projY);
  }

  start() {
    if (this.isLooping) return;
    this.isLooping = true;
    const loop = () => {
      this.render();
      if (this.isLooping) {
        this.animFrameId = requestAnimationFrame(loop);
      }
    };
    this.animFrameId = requestAnimationFrame(loop);
  }

  stop() {
    this.isLooping = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // 1. Draw Synapses
    for (const syn of this.synapses) {
      const w = syn.weight;
      const isPositive = w >= 0;
      const absW = Math.abs(w);

      const thickness = Math.max(0.6, Math.min(5.0, 1.0 + absW * 2.2));
      const alpha = Math.max(0.12, Math.min(0.85, 0.2 + absW * 0.4));

      ctx.save();
      ctx.lineWidth = thickness;
      ctx.strokeStyle = isPositive
        ? `rgba(0, 240, 255, ${alpha})`   // Cyan for positive
        : `rgba(255, 0, 127, ${alpha})`;  // Magenta for negative

      ctx.beginPath();
      ctx.moveTo(syn.source.x, syn.source.y);
      ctx.lineTo(syn.target.x, syn.target.y);
      ctx.stroke();
      ctx.restore();
    }

    // 2. Draw Synaptic Pulses
    for (const p of this.particlePool) {
      if (p.active) {
        p.update();
        p.draw(ctx);
      }
    }

    // 3. Draw Neurons
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const actLayer = this.activations[l];

      for (const node of layer.nodes) {
        let actVal = 0.0;
        if (actLayer && actLayer[node.rawIndex] !== undefined) {
          actVal = Math.max(0.0, Math.min(1.0, actLayer[node.rawIndex]));
        }

        ctx.save();
        // Glow on active nodes
        if (actVal > 0.1) {
          ctx.shadowColor = '#00f0ff';
          ctx.shadowBlur = 12 * actVal;
        }

        // Inner circle
        ctx.fillStyle = actVal > 0.05
          ? `rgba(0, 240, 255, ${0.3 + actVal * 0.7})`
          : '#131b2e';
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fill();

        // Border
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = actVal > 0.2 ? '#ffffff' : 'rgba(255, 255, 255, 0.3)';
        ctx.stroke();
        ctx.restore();

        // Label for first and last layers
        if (l === 0 || l === this.layers.length - 1) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
          ctx.font = '10px JetBrains Mono, monospace';
          ctx.textAlign = l === 0 ? 'right' : 'left';
          const textX = l === 0 ? node.x - node.radius - 6 : node.x + node.radius + 6;
          let label = node.name;
          if (this.labelMode === 'friendly') {
            label = l === 0 ? 'pictures in' : 'the answer';
          }
          ctx.fillText(label, textX, node.y + 3);
        }
      }
    }

    // 4. Draw Hover Tooltip
    if (this.hoveredItem) {
      this.drawTooltip(ctx, this.hoveredItem);
    }
  }

  drawTooltip(ctx, item) {
    ctx.save();
    const pad = 8;
    ctx.font = '11px JetBrains Mono, monospace';

    let text1 = '';
    let text2 = '';

    if (item.type === 'node') {
      const node = item.data;
      const act = this.activations[node.layer]
        ? (this.activations[node.layer][node.rawIndex] || 0.0).toFixed(4)
        : '0.0000';
      text1 = `${node.name}`;
      text2 = `Activation: ${act}`;
    } else {
      const syn = item.data;
      text1 = `Weight: ${syn.weight.toFixed(4)}`;
      text2 = `Layer ${syn.layer} -> ${syn.layer + 1}`;
    }

    const w1 = ctx.measureText(text1).width;
    const w2 = ctx.measureText(text2).width;
    const boxW = Math.max(w1, w2) + pad * 2;
    const boxH = 44;

    const boxX = Math.min(this.width - boxW - 10, this.mousePos.x + 12);
    const boxY = Math.max(10, this.mousePos.y - 30);

    ctx.fillStyle = 'rgba(10, 15, 30, 0.92)';
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.fillText(text1, boxX + pad, boxY + 18);
    ctx.fillStyle = 'rgba(0, 240, 255, 0.9)';
    ctx.fillText(text2, boxX + pad, boxY + 34);

    ctx.restore();
  }
}
