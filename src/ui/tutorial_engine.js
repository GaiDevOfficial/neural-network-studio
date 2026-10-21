/**
 * Node Studio — guided tutorial engine.
 *
 * Two layers, deliberately separated:
 *   - TutorialRunner: the pure step machine (advance / back / skip / action-gated steps).
 *     Tested in Node by tests/test_graph_engine.js — no DOM required.
 *   - TutorialOverlay: the DOM presenter — a spotlight (box-shadow cutout) around the
 *     current target with a practical tooltip; the toolbar stays fully interactive during
 *     a tour so "click the button now" steps advance on the real gesture.
 *
 * File: src/ui/tutorial_engine.js
 */

import {
  isBackdropTarget,
  measureChrome,
  normalizeRect,
  placeFloatingPanel,
  STUDIO_CHROME_SELECTORS
} from '../tutorial/overlay_layout.js';
import { inlineMarkdown } from '../tutorial/inline_text.js';

/* ------------------------------------------------------------------ tours */

export const TOURS = {
  first_ai: {
    id: 'first_ai',
    title: 'Lesson 2 — Your First AI from a Preset',
    steps: [
      {
        id: 't1-welcome',
        title: 'Load a real 0-vs-1 digit classifier',
        text: 'This lesson starts from a working preset. Press the highlighted **MNIST preset** button. It creates a real graph: Dataset Source → Flatten → Dense Layer → Dense Layer → Dense Layer → Sigmoid, plus Binary Cross-Entropy, Adam and Epoch Loop. Nothing is simulated.',
        target: '#btn-preset-mnist',
        action: { id: 'preset_mnist', hint: 'Press MNIST preset.' }
      },
      {
        id: 't1-inspect',
        title: 'Trace the blocks from left to right',
        text: 'Follow the wires on the canvas. **Dataset Source** supplies 28×28 images. **Flatten** turns each image into 784 numbers. The three **Dense Layer** blocks learn the pattern. **Sigmoid** turns the final scores into two probabilities. **Binary Cross-Entropy** measures the error, **Adam** updates the weights, and **Epoch Loop** controls how long training runs.',
        target: '#studio-canvas'
      },
      {
        id: 't1-run',
        title: 'Press Run and let it train',
        text: 'Press **Run**. The worker performs real forward passes and backpropagation, then Adam updates every weight. Training auto-stops when the held-out loss stops improving. Your Run press is recorded, so this step stays complete even if the run finishes before you click Next.',
        target: '#btn-run',
        action: { id: 'training_started', hint: 'Press Run.' }
      },
      {
        id: 't1-metrics',
        title: 'Read the live metrics',
        text: 'The **Training** panel shows the current epoch and the measured training loss, held-out loss, accuracy and best checkpoint. Every number comes from the real epoch stream — there are no fake curves.',
        target: '#panel-metrics'
      },
      {
        id: 't1-confusion',
        title: 'Check which digits it confuses',
        text: 'The **Confusion Matrix** counts real held-out predictions: rows are the true class, columns are the predicted class. For 0-vs-1, the diagonal cells are correct and the off-diagonal cells are mistakes.',
        target: '#panel-confusion'
      },
      {
        id: 't1-export',
        title: 'Export the trained model',
        text: 'Open **Export** to save the graph and trained weights: JSON to reload here, standalone Python or JavaScript, ONNX for another runtime, or the exact packaging commands. Pick an option you understand — the lesson ends when the menu opens.',
        target: '#btn-export',
        action: { id: 'export_opened', hint: 'Open the Export menu.' }
      }
    ]
  },

  backprop: {
    id: 'backprop',
    title: 'Lesson 4 — Inside Backpropagation',
    steps: [
      {
        id: 't2-run',
        title: 'Start a real training run',
        text: 'Press **Run** on a valid graph. During each epoch the engine performs a forward pass, computes the loss, then backpropagates gradients through every **Dense Layer** before the optimizer updates the weights.',
        target: '#btn-run',
        action: { id: 'training_started', hint: 'Press Run on a valid graph.' }
      },
      {
        id: 't2-particles',
        title: 'Turn on wire signals',
        text: 'Tick **Particles**. Each moving dot is drawn from the engine’s real layer activity and gradient flow: forward values travel toward the loss, and backward signals travel toward the layers that must change.',
        target: '#toggle-particles',
        action: { id: 'particles_toggled', hint: 'Tick the Particles checkbox.' }
      },
      {
        id: 't2-slowmo',
        title: 'Slow the epochs down',
        text: 'Increase the **Epoch delay** slider. The delay only slows playback; the arithmetic is unchanged. This makes it easier to see the loss fall and the weights settle between epochs.',
        target: '#epoch-delay',
        action: { id: 'epoch_delay_changed', hint: 'Move the Epoch delay slider.' }
      },
      {
        id: 't2-metrics',
        title: 'Loss is the value being minimized',
        text: 'The **Training** panel plots the real loss. Backpropagation computes the gradient of that loss with respect to every weight. A falling curve means the optimizer is reducing the error — not that an animation is playing.',
        target: '#panel-metrics'
      },
      {
        id: 't2-inspector',
        title: 'Inspect the blocks that control learning',
        text: 'Select a **Dense Layer** to change its neuron count, an **Adam** block to change the learning rate, or the **Epoch Loop** to switch between auto-stop and a fixed epoch budget. Changes feed the real compiler and optimizer.',
        target: '#inspector-panel'
      }
    ]
  },

  database: {
    id: 'database',
    title: 'Lesson 6 — SQL Data & Live Predictions',
    steps: [
      {
        id: 't3-sql',
        title: 'Add a SQL Query source',
        text: 'Drag **SQL Query** from the Data palette onto the canvas. The local Data API reads your SQLite file read-only and streams rows. SQL Query replaces Dataset Source; you can only have one source block in a graph.',
        target: '#palette-sql_source',
        action: { id: 'source_added_sql', hint: 'Drag SQL Query onto the canvas.' }
      },
      {
        id: 't3-inspector',
        title: 'Point SQL Query at your table',
        text: 'Select the **SQL Query** block. In the Inspector, enter the `.sqlite` path and a single SELECT statement. The last selected column becomes the target; the other numeric columns become features. The same normalization and train/test split as CSV loading are applied.',
        target: '#inspector-panel'
      },
      {
        id: 't3-run',
        title: 'Train on the streamed rows',
        text: 'Press **Run**. Rows are fetched by the Data API, shaped by the compiler, and sent through the real training loop. If the query or schema is invalid, the validation panel shows the exact error instead of starting a fake run.',
        target: '#btn-run',
        action: { id: 'training_started', hint: 'Press Run on a valid SQL graph.' }
      },
      {
        id: 't3-predict',
        title: 'Probe the trained model',
        text: 'Open **Prediction View** and enter feature values or draw an image, then press **Predict**. Every probe is a real forward pass in the engine; the confidence and class come from the trained weights.',
        target: '#panel-predict',
        action: { id: 'predicted', hint: 'Press Predict in the Prediction View.' }
      }
    ]
  },

  export_deploy: {
    id: 'export_deploy',
    title: 'Lesson 10 — Export & Deploy a Trained Model',
    steps: [
      {
        id: 'e1-preset',
        title: 'Load a trained-ready graph',
        text: 'Press **MNIST preset** to create the verified classifier chain. You can also build your own graph; the export step only unlocks when the graph compiles and has trained weights.',
        target: '#btn-preset-mnist',
        action: { id: 'preset_mnist', hint: 'Press MNIST preset.' }
      },
      {
        id: 'e2-run',
        title: 'Train it once',
        text: 'Press **Run**. Exporting writes the current trained weights, so at least one real training run is required. Your press is recorded even if the run finishes quickly.',
        target: '#btn-run',
        action: { id: 'training_started', hint: 'Press Run.' }
      },
      {
        id: 'e3-export',
        title: 'Open the Export menu',
        text: 'Press **Export**. The menu offers JSON (reloadable here), HTML, JavaScript, Python, ONNX, a NumPy `.npz` weight archive, and the exact `.exe` packaging commands.',
        target: '#btn-export',
        action: { id: 'export_opened', hint: 'Open the Export menu.' }
      },
      {
        id: 'e4-json',
        title: 'Save the model as JSON',
        text: 'Choose **Model JSON**. The file contains the graph definition and trained weights, so it can be loaded back into Node Studio later. The download is only created after a successful compile and training run.',
        target: '#btn-export',
        action: { id: 'export_json', hint: 'Choose Model JSON from the Export menu.' }
      }
    ]
  }
};

/* ---------------------------------------------------------- state machine */

export class TutorialRunner {
  constructor(tour, options = {}) {
    this.tour = tour;
    this.index = 0;
    this.done = false;
    /* Live reference to the page action journal. It lets a step that was already performed
       (for example Run, which may finish before the learner clicks Next) stay satisfied. */
    this.actionLog = Array.isArray(options.actions) ? options.actions : [];
    this.satisfied = new Set();
  }

  get step() {
    return this.done ? null : this.tour.steps[this.index];
  }

  get total() {
    return this.tour.steps.length;
  }

  next() {
    if (this.done) return null;
    if (this.index >= this.tour.steps.length - 1) {
      this.done = true;
      return null;
    }
    this.index++;
    return this.step;
  }

  prev() {
    if (this.done) {
      this.done = false;
      this.index = this.tour.steps.length - 1;
      return this.step;
    }
    this.index = Math.max(0, this.index - 1);
    return this.step;
  }

  skip() {
    this.done = true;
    return null;
  }

  /** True when this step's real gesture is already in the page journal (or was just reported). */
  isSatisfied(step = this.step) {
    if (!step || !step.action) return true;
    if (this.satisfied.has(step.id)) return true;
    const id = step.action.id;
    return Boolean(id && this.actionLog.some((entry) => entry && entry.id === id));
  }

  /** Advance when the UI reports the gated action of the current step. */
  notify(actionId) {
    const step = this.step;
    if (step && step.action && step.action.id === actionId) {
      this.satisfied.add(step.id);
      return this.next();
    }
    return null;
  }
}

/* ------------------------------------------------------------- overlay */

export class TutorialOverlay {
  constructor(options = {}) {
    this.onComplete = options.onComplete || null;
    this.doc = options.document || (typeof document !== 'undefined' ? document : null);
    this.win = options.window || (typeof window !== 'undefined' ? window : null);
    this.runner = null;
    this.root = null;
    this.highlight = null;
    this.tooltip = null;
    this.manageEscape = options.manageEscape !== false;
    this._onScroll = () => this.refreshPosition();
  }

  get active() {
    return Boolean(this.runner && !this.runner.done);
  }

  /** Accept a tour object (curriculum tracks) or a legacy `TOURS` id. */
  start(tourOrId, options = {}) {
    const tour = typeof tourOrId === 'string' ? TOURS[tourOrId] : tourOrId;
    if (!tour || !Array.isArray(tour.steps) || tour.steps.length === 0) return false;
    if (this.root) this.finish(true);
    this.runner = new TutorialRunner(tour, { actions: options.actions });
    this.mount();
    this.render();
    return true;
  }

  /** Let the page report a real interaction (press, run start, …). */
  notify(actionId) {
    if (!this.runner || this.runner.done) return;
    const before = this.runner.index;
    this.runner.notify(actionId);
    if (this.runner.done) {
      this.finish();
      return;
    }
    if (this.runner.index !== before || this.runner.step) this.render();
  }

  mount() {
    if (!this.doc || !this.win) return;
    this.root = this.doc.createElement('div');
    this.root.className = 'tour-root';
    this.highlight = this.doc.createElement('div');
    this.highlight.className = 'tour-highlight';
    this.tooltip = this.doc.createElement('div');
    this.tooltip.className = 'tour-tooltip';
    this.root.appendChild(this.highlight);
    this.root.appendChild(this.tooltip);
    this.doc.body.appendChild(this.root);
    this._onResize = () => this._layout();
    this.win.addEventListener('resize', this._onResize);
    // Capture-phase scroll keeps the spotlight on a control inside a scrolling panel.
    this.win.addEventListener('scroll', this._onScroll, true);
    if (this.manageEscape) {
      this._onKeyDown = (event) => {
        if (event.key === 'Escape' || event.key === 'Esc') {
          event.preventDefault();
          this.finish();
        }
      };
      this.doc.addEventListener('keydown', this._onKeyDown, true);
    }
  }

  render() {
    if (!this.doc || !this.root) return;
    const step = this.runner && this.runner.step;
    if (!step) {
      this.finish();
      return;
    }

    const stepNo = this.runner.index + 1;
    const progress = `${stepNo} / ${this.runner.total}`;
    const pct = Math.round((stepNo / this.runner.total) * 100);
    const done = this.runner.isSatisfied(step);
    const hintText = step.action && step.action.hint
      ? `<div class="tour-hint ${done ? 'is-met' : ''}">${done ? '✓ Done — press Next to continue.' : inlineMarkdown(step.action.hint)}</div>`
      : '';
    const gated = Boolean(step.action) && !done;
    this.tooltip.innerHTML = `
      <button class="tour-close" data-tour="close" type="button" aria-label="Close tutorial" title="Close (Esc)">×</button>
      <div class="tour-kicker">Step-by-step lesson · Step ${stepNo}</div>
      <div class="tour-lesson">${this.runner.tour.title}</div>
      <div class="tour-title">${step.title}</div>
      <div class="tour-progress-track" aria-hidden="true"><span style="width:${pct}%"></span></div>
      <div class="tour-text">${inlineMarkdown(step.text)}</div>
      ${hintText}
      <div class="tour-actions">
        <button data-tour="back">Back</button>
        <button data-tour="skip">Skip lesson</button>
        <button data-tour="next" class="primary" ${gated ? 'disabled title="Perform the highlighted action to continue"' : ''}>${gated ? 'Do the action above' : (this.runner.index === this.runner.total - 1 ? 'Finish' : 'Next')}</button>
        <span class="tour-progress">${progress}</span>
      </div>`;

    this.tooltip.querySelector('[data-tour="close"]').onclick = () => this.finish();
    this.tooltip.querySelector('[data-tour="back"]').onclick = () => this.back();
    this.tooltip.querySelector('[data-tour="skip"]').onclick = () => this.finish();
    this.tooltip.querySelector('[data-tour="next"]').onclick = () => this.advance();
    this._layout();
  }

  /**
   * Reposition without rebuilding the tooltip (which would reset focus / button state). The page
   * calls this after node drags, pan/zoom and inspector re-renders; resize and captured scroll are
   * handled here.
   */
  refreshPosition() {
    if (!this.active || !this.root) return;
    this._layout();
  }

  _layout() {
    if (!this.doc || !this.win || !this.root || !this.tooltip) return;
    const step = this.runner && this.runner.step;
    if (!step) return;
    const winW = Number.isFinite(this.win.innerWidth) ? this.win.innerWidth : 1280;
    const winH = Number.isFinite(this.win.innerHeight) ? this.win.innerHeight : 800;
    const target = step.target ? this.doc.querySelector(step.target) : null;
    const rect = target && target.getBoundingClientRect ? normalizeRect(target.getBoundingClientRect()) : null;
    const hasTarget = Boolean(rect && rect.width >= 1 && rect.height >= 1);

    if (!hasTarget) {
      this.highlight.style.display = 'none';
      this.tooltip.style.left = '50%';
      this.tooltip.style.top = '50%';
      this.tooltip.style.transform = 'translate(-50%, -50%)';
      return;
    }

    const pad = 6;
    const left = Math.max(2, rect.left - pad);
    const top = Math.max(2, rect.top - pad);
    Object.assign(this.highlight.style, {
      display: 'block',
      left: `${left}px`,
      top: `${top}px`,
      width: `${Math.max(2, Math.min(winW - left - 2, rect.width + pad * 2))}px`,
      height: `${Math.max(2, Math.min(winH - top - 2, rect.height + pad * 2))}px`
    });

    // A control inside the palette / inspector / topbar should push the tooltip into the canvas,
    // not just to the other side of the control (which would still cover the panel).
    let avoidRect = rect;
    if (target && target.closest) {
      const anchorEl = target.closest('.ns-node')
        || target.closest('.ns-palette, .ns-right, .ns-topbar, .ns-statusbar');
      if (anchorEl && anchorEl !== target && anchorEl.getBoundingClientRect) {
        const anchorRect = normalizeRect(anchorEl.getBoundingClientRect());
        if (anchorRect && anchorRect.width > 0 && anchorRect.height > 0) avoidRect = anchorRect;
      }
    }

    const tooltipW = this.tooltip.offsetWidth || Math.min(380, Math.max(1, winW - 32));
    const tooltipH = this.tooltip.offsetHeight || 220;
    const placed = placeFloatingPanel({
      target: avoidRect,
      width: tooltipW,
      height: tooltipH,
      viewportWidth: winW,
      viewportHeight: winH,
      margin: 16,
      gap: 16,
      prefer: ['below', 'above', 'right', 'left'],
      hardAvoid: !(target.id === 'studio-canvas' || isBackdropTarget(avoidRect, winW, winH)),
      softAvoid: measureChrome(this.doc, STUDIO_CHROME_SELECTORS)
    });
    this.tooltip.style.left = `${placed.x}px`;
    this.tooltip.style.top = `${placed.y}px`;
    this.tooltip.style.transform = 'none';
  }

  advance() {
    if (!this.runner || !this.runner.step) return;
    // Action-gated steps advance from the real gesture, but a gesture already present in the
    // page journal counts too (this is what keeps a finished training run from blocking Next).
    if (this.runner.step.action && !this.runner.isSatisfied(this.runner.step)) return;
    this.runner.next();
    if (this.runner.done) this.finish();
    else this.render();
  }

  back() {
    if (!this.runner) return;
    this.runner.prev();
    this.render();
  }

  finish(silent = false) {
    if (this._onResize && this.win) {
      this.win.removeEventListener('resize', this._onResize);
      this._onResize = null;
    }
    if (this.win) this.win.removeEventListener('scroll', this._onScroll, true);
    if (this._onKeyDown && this.doc) {
      this.doc.removeEventListener('keydown', this._onKeyDown, true);
      this._onKeyDown = null;
    }
    if (this.root && this.root.parentNode) this.root.parentNode.removeChild(this.root);
    this.root = null;
    this.highlight = null;
    this.tooltip = null;
    if (this.runner) this.runner.skip();
    if (!silent && this.onComplete) this.onComplete();
  }
}
