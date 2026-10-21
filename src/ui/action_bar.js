/**
 * Streamlined action bar controller for the Node Studio header.
 *
 * The bar is one row: [ 3-tier mode toggle ] | [ dataset ] | [ training ] | [ live metrics ] | [ export ].
 * This module owns the state presentation only — every click is forwarded to the page controller, so
 * the bar can never mutate the graph or the worker directly.
 *
 * DOM-free at import time; safe to import in Node.
 *
 * File: src/ui/action_bar.js
 */

export const TIERS = ['playground', 'assist', 'manual'];

export class ActionBar {
  constructor(options = {}) {
    this.root = options.root || null;
    this.modeButtons = options.modeButtons || {};
    this.datasetButton = options.datasetButton || null;
    this.metricsButton = options.metricsButton || null;
    this.runButton = options.runButton || null;
    this.pauseButton = options.pauseButton || null;
    this.handlers = {
      mode: options.onMode || null,
      dataset: options.onDataset || null,
      metrics: options.onMetrics || null
    };
    this.cleanup = [];
    this.currentMode = null;
    this.currentRunning = false;

    for (const tier of TIERS) {
      const button = this.modeButtons[tier];
      if (!button) continue;
      const handler = () => {
        if (this.handlers.mode) this.handlers.mode(tier);
      };
      button.addEventListener('click', handler);
      this.cleanup.push(() => button.removeEventListener('click', handler));
    }

    if (this.datasetButton && this.handlers.dataset) {
      const handler = () => this.handlers.dataset();
      this.datasetButton.addEventListener('click', handler);
      this.cleanup.push(() => this.datasetButton.removeEventListener('click', handler));
    }
    if (this.metricsButton && this.handlers.metrics) {
      const handler = () => this.handlers.metrics();
      this.metricsButton.addEventListener('click', handler);
      this.cleanup.push(() => this.metricsButton.removeEventListener('click', handler));
    }
  }

  setMode(mode) {
    this.currentMode = mode;
    for (const tier of TIERS) {
      const button = this.modeButtons[tier];
      if (!button) continue;
      const active = tier === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
    if (this.root) this.root.dataset.mode = mode;
  }

  setTraining(running, { runLabel = '\u25b6 Run', pauseHidden = null } = {}) {
    this.currentRunning = Boolean(running);
    if (this.runButton) {
      this.runButton.disabled = Boolean(running);
      this.runButton.textContent = running ? 'Running\u2026' : runLabel;
    }
    if (this.pauseButton) {
      this.pauseButton.hidden = pauseHidden === null ? !running : pauseHidden;
    }
  }

  setDataset(shortText, fullText = '') {
    if (!this.datasetButton) return;
    this.datasetButton.textContent = shortText;
    this.datasetButton.title = fullText || shortText;
    this.datasetButton.classList.toggle('is-empty', !shortText || shortText === 'No dataset');
  }

  setPlan(text) {
    if (this.root) this.root.dataset.plan = text || '';
  }

  destroy() {
    for (const cleanup of this.cleanup) cleanup();
    this.cleanup = [];
  }
}
