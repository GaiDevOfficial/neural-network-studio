/**
 * High-level synaptic canvas used by the Absolute-Beginner playground.
 *
 * Wraps the zero-garbage `SynapseRenderer` with a small lifecycle API:
 *   setNetwork(architecture, weights) · showStep({weights, activations, deltas})
 *   resize() · start() · stop() · dispose()
 *
 * Friendly label mode replaces "Input 0"/"Output 1" with "pictures in"/"the answer".
 * Only real engine data is ever passed in; with no weights snapshot the lines stay at zero.
 *
 * File: src/visualization/synaptic_canvas.js
 */

import { SynapseRenderer } from './synapse_render.js';

export class SynapticCanvas {
  constructor(canvasElement, options = {}) {
    if (!canvasElement) throw new Error('SynapticCanvas: a canvas element is required.');
    this.canvas = canvasElement;
    this.renderer = new SynapseRenderer(canvasElement);
    this.renderer.setLabelMode(options.friendly === false ? 'technical' : 'friendly');
    this.started = false;
    this._resizeObserver = null;
    if (typeof ResizeObserver !== 'undefined' && canvasElement.parentElement) {
      // The playground is display:none until it opens; a ResizeObserver catches the first real
      // layout pass (and later sidebar/grid changes) without relying on animation-frame timing.
      this._resizeObserver = new ResizeObserver(() => this.resize());
      this._resizeObserver.observe(canvasElement.parentElement);
    }
    if (options.autoStart !== false) this.start();
  }

  setNetwork(architecture, weights = null) {
    if (!Array.isArray(architecture) || architecture.length === 0) return;
    this.renderer.setArchitecture(architecture, weights);
  }

  /** Update synapse thickness without recomputing the node layout. */
  setWeights(weights) {
    if (!weights) return;
    this.renderer.weights = weights;
    this.renderer.syncSynapseWeights();
  }

  /** Feed one real engine payload: { weights?, activations?, deltas? }. */
  showStep(step) {
    if (!step) return;
    this.renderer.updateData(step);
  }

  setFriendly(friendly) {
    this.renderer.setLabelMode(friendly ? 'friendly' : 'technical');
  }

  resize() {
    this.renderer.resize();
  }

  start() {
    if (this.started) return;
    this.started = true;
    this.renderer.start();
  }

  stop() {
    if (!this.started) return;
    this.started = false;
    this.renderer.stop();
  }

  dispose() {
    this.stop();
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
    this.renderer.dispose();
  }
}

export function createSynapticCanvas(canvasElement, options = {}) {
  return new SynapticCanvas(canvasElement, options);
}
