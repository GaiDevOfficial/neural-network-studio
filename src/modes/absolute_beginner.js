/**
 * Tier 1 — Absolute-Beginner Playground.
 *
 * A jargon-free workspace over the same real engine: a pre-assembled MNIST 0-vs-1 graph, a
 * simplified drawing pad, a live synaptic canvas fed by real activations/gradients, and
 * good/wrong feedback that runs one real ONLINE_STEP and immediately re-predicts.
 *
 * The module owns presentation only. Every computation is delegated to the page controller,
 * which posts the same worker RPCs the block studio uses (prepare / startTraining / predict /
 * correct). Nothing here fabricates a number, a glow or an update.
 *
 * File: src/modes/absolute_beginner.js
 */

import { SynapticCanvas } from '../visualization/synaptic_canvas.js';

const PAD = 28;
const PAD_CSS = 280;
const STORY_KEYS = ['teach', 'draw', 'ask', 'feedback'];

export class AbsoluteBeginnerPlayground {
  constructor(options = {}) {
    this.controller = options.controller;
    if (!this.controller) throw new Error('AbsoluteBeginnerPlayground: a controller is required.');
    this.onAction = options.onAction || null;
    this.onExit = options.onExit || null;

    this.pixels = new Float64Array(PAD * PAD);
    this.ink = false;
    this.active = false;
    this.busy = false;
    this.training = false;
    this.weightsReady = false;
    this.prediction = null;
    this.classNames = ['0', '1'];
    this.story = { teach: false, draw: false, ask: false, feedback: false };
    this.padCtx = null;
    this.lastPoint = null;
    this.drawing = false;
    this.synaptic = null;
    this.dom = null;

    this._initDom();
  }

  /* ------------------------------------------------------------------ dom */

  _initDom() {
    const $ = (id) => document.getElementById(id);
    this.dom = {
      container: $('playground'),
      brain: $('playground-brain'),
      brainState: $('playground-brain-state'),
      pad: $('playground-pad'),
      story: $('playground-story'),
      answer: $('playground-answer'),
      feedback: $('playground-feedback'),
      loss: $('playground-loss'),
      ask: $('playground-ask'),
      clear: $('playground-clear'),
      teach: $('playground-teach'),
      stop: $('playground-stop'),
      progress: $('playground-progress'),
      accuracy: $('playground-accuracy'),
      right: $('playground-right'),
      wrong: $('playground-wrong'),
      exit: $('playground-exit')
    };

    this.synaptic = new SynapticCanvas(this.dom.brain, { friendly: true, autoStart: false });
    this._setupPad();

    this.dom.teach.addEventListener('click', () => this._teach());
    this.dom.stop.addEventListener('click', () => this._stop());
    this.dom.ask.addEventListener('click', () => this._ask());
    this.dom.clear.addEventListener('click', () => this._clear());
    this.dom.right.addEventListener('click', () => this._feedback(true));
    this.dom.wrong.addEventListener('click', () => this._feedback(false));
    this.dom.exit.addEventListener('click', () => {
      if (this.onExit) this.onExit();
    });

    this._updateStory();
    this._updateAskState();
  }

  _setupPad() {
    const canvas = this.dom.pad;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = PAD_CSS * dpr;
    canvas.height = PAD_CSS * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.padCtx = ctx;
    this._renderPad();

    const point = (event) => {
      const rect = canvas.getBoundingClientRect();
      const scale = PAD_CSS / (rect.width || PAD_CSS);
      return {
        x: (event.clientX - rect.left) * scale,
        y: (event.clientY - rect.top) * scale
      };
    };

    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      this.drawing = true;
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch {
        // Synthetic pointers / restricted webviews cannot be captured; `drawing` still tracks
        // the gesture so the pad keeps working.
      }
      this.lastPoint = point(event);
      this._paintSegment(this.lastPoint, this.lastPoint);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.drawing || !this.lastPoint) return;
      event.preventDefault();
      const next = point(event);
      this._paintSegment(this.lastPoint, next);
      this.lastPoint = next;
    });
    const end = (event) => {
      this.drawing = false;
      this.lastPoint = null;
      try {
        if (canvas.hasPointerCapture && canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      } catch {
        // Capture was never acquired.
      }
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('pointerleave', (event) => {
      // Without pointer capture, leaving the pad ends the stroke; with capture the browser
      // retargets the pointer and this event does not fire mid-draw.
      if (!canvas.hasPointerCapture || !canvas.hasPointerCapture(event.pointerId)) end(event);
    });
  }

  /* --------------------------------------------------------------- lifecycle */

  async open() {
    this.active = true;
    this.busy = true;
    this.dom.answer.textContent = 'Getting the brain ready…';
    this.dom.loss.textContent = '';
    this.dom.feedback.hidden = true;
    this.dom.accuracy.textContent = '';
    this._updateAskState();

    try {
      const snapshot = await this.controller.prepare();
      if (!snapshot || !snapshot.plan || !snapshot.plan.ok) {
        throw new Error('The pre-assembled brain could not be compiled.');
      }
      this.classNames = this.controller.getClassNames();
      this.weightsReady = Boolean(snapshot.weights);
      this.training = Boolean(snapshot.running);
      this.architecture = snapshot.plan.network.architecture;
      this.synaptic.setNetwork(this.architecture, snapshot.weights);
      this._setTrainingUi(this.training);
      this.dom.brainState.textContent = this.weightsReady ? 'trained' : 'not trained yet';
      this.dom.answer.textContent = this.weightsReady
        ? 'Draw a number and ask me what I see.'
        : 'Press “Teach the brain” first — it has to practice before it can answer.';
      if (!this.weightsReady) this.dom.progress.textContent = 'The brain is ready to practice on real handwritten digits.';
      else if (!this.training) this.dom.progress.textContent = 'The brain already knows a little. Keep teaching or draw a number.';
    } catch (error) {
      this.dom.answer.textContent = `Could not prepare the playground: ${error.message}`;
      this.dom.progress.textContent = 'Open the block studio and check the validation panel.';
    } finally {
      this.busy = false;
      this._updateAskState();
      // The section is visible by the time prepare() resolves, so resize synchronously now;
      // rAF/timers are only fallbacks for webviews that throttle animation frames while the
      // window is in the background.
      this.synaptic.resize();
      this.synaptic.start();
      requestAnimationFrame(() => this.synaptic.resize());
      setTimeout(() => this.synaptic.resize(), 150);
    }
  }

  close() {
    this.active = false;
    this.busy = false;
    this.prediction = null;
    this.drawing = false;
    this.lastPoint = null;
    this.synaptic.stop();
  }

  /* ---------------------------------------------------------------- training */

  async _teach() {
    if (this.training || this.busy) return;
    this.busy = true;
    this.dom.teach.disabled = true;
    this.dom.progress.textContent = 'Starting practice on real handwritten digits…';
    const started = await this.controller.startTraining();
    this.busy = false;
    if (!started) {
      this.dom.teach.disabled = false;
      this.dom.progress.textContent = 'Could not start — fix the graph in the block studio first.';
      return;
    }
    this.training = true;
    this._setTrainingUi(true);
    this._mark('teach');
    if (this.onAction) this.onAction('playground_training_started');
  }

  _stop() {
    this.controller.stopTraining();
    this.dom.progress.textContent = 'Stopping after the current batch…';
  }

  _setTrainingUi(training) {
    this.dom.teach.disabled = training;
    this.dom.teach.textContent = training ? 'Practicing…' : 'Teach the brain';
    this.dom.stop.hidden = !training;
    if (training) {
      this.dom.brainState.textContent = 'learning';
    } else if (this.weightsReady) {
      this.dom.brainState.textContent = 'ready';
    }
  }

  handleTrainingStep(payload) {
    if (payload.weights) {
      this.weightsReady = true;
      this.synaptic.setWeights(payload.weights);
    }
    if (this.active) {
      this.synaptic.showStep({
        activations: payload.activations,
        deltas: payload.deltas
      });
      this.dom.progress.textContent = `Practice round ${payload.epoch} · mistakes ${Number(payload.loss).toFixed(4)}`;
      if (!this.training) {
        this.training = true;
        this._setTrainingUi(true);
      }
      if (!this.story.teach) {
        this._mark('teach');
        if (this.onAction) this.onAction('playground_training_started');
      }
      this._updateAskState();
    }
  }

  handleTrainingComplete(payload) {
    this.training = false;
    if (!this.active) return;
    this._setTrainingUi(false);
    this.dom.progress.textContent = `Practice finished at round ${payload.epoch}.`;
    this._updateAskState();
  }

  handleTrainingPaused(payload) {
    this.training = false;
    if (!this.active) return;
    this._setTrainingUi(false);
    this.dom.progress.textContent = `Paused at round ${payload.epoch}.`;
    this._updateAskState();
  }

  handleEvaluation(payload) {
    if (!this.active || !payload || !payload.ok) return;
    if (typeof payload.accuracy === 'number') {
      this.dom.accuracy.textContent = `Held-out check: ${Math.round(payload.accuracy * 1000) / 10}% of unseen pictures answered correctly (${payload.count} pictures).`;
    } else if (typeof payload.loss === 'number') {
      this.dom.accuracy.textContent = `Held-out mistake score: ${Number(payload.loss).toFixed(4)} over ${payload.count} pictures.`;
    }
  }

  /* -------------------------------------------------------------- prediction */

  async _ask() {
    if (this.busy || !this.ink) return;
    if (!this.weightsReady) {
      this.dom.answer.textContent = 'The brain has not practiced yet. Press “Teach the brain” first.';
      return;
    }
    this.busy = true;
    this.dom.ask.disabled = true;
    this.dom.answer.textContent = 'Looking at your drawing…';
    this.dom.feedback.hidden = true;
    this.dom.loss.textContent = '';
    try {
      const payload = await this.controller.predict(this._inputVector());
      this._applyPrediction(payload, false);
      this._mark('ask');
      if (this.onAction) this.onAction('playground_predicted');
    } catch (error) {
      this.dom.answer.textContent = `The brain could not answer: ${error.message}`;
    } finally {
      this.busy = false;
      this._updateAskState();
    }
  }

  _applyPrediction(payload, afterLearning) {
    this.prediction = payload;
    if (payload.activations) {
      this.synaptic.showStep({ activations: payload.activations, deltas: payload.deltas });
    }
    const predicted = payload.predictedClass;
    const label = this.classNames[predicted] !== undefined ? this.classNames[predicted] : String(predicted);
    const confidence = Math.round((payload.confidence || 0) * 1000) / 10;
    const prefix = afterLearning ? 'After learning: ' : '';
    this.dom.answer.textContent = `${prefix}I think this is a ${label} — ${confidence}% sure.`;

    const other = predicted === 0 ? 1 : 0;
    const otherLabel = this.classNames[other] !== undefined ? this.classNames[other] : String(other);
    this.dom.wrong.textContent = `Nope — it's a ${otherLabel}`;
    this.dom.feedback.hidden = false;
    if (!afterLearning) this.dom.loss.textContent = '';
  }

  async _feedback(wasRight) {
    if (this.busy || !this.prediction) return;
    this._mark('feedback');
    if (this.onAction) this.onAction('playground_corrected');

    if (wasRight) {
      this.dom.feedback.hidden = true;
      this.dom.loss.textContent = 'Good — no correction was needed. Draw another one.';
      return;
    }

    const trueLabel = this.prediction.predictedClass === 0 ? 1 : 0;
    this.busy = true;
    this.dom.feedback.hidden = true;
    this.dom.loss.textContent = 'Teaching the brain with this drawing…';
    try {
      const result = await this.controller.correct(this._inputVector(), trueLabel);
      if (!result || !result.ok) throw new Error((result && result.error) || 'online step failed');

      if (result.weights) {
        this.weightsReady = true;
        this.synaptic.showStep({
          weights: result.weights,
          activations: result.activations,
          deltas: result.deltas
        });
      }

      const before = Number(result.initialLoss);
      const after = Number(result.postLoss);
      if (result.lossReduced) {
        const percent = Math.round((result.relativeReduction || 0) * 1000) / 10;
        this.dom.loss.textContent = `It learned from this drawing: mistakes ${before.toFixed(4)} → ${after.toFixed(4)} (${percent}% less on this one).`;
      } else {
        this.dom.loss.textContent = `Mistakes ${before.toFixed(4)} → ${after.toFixed(4)}; the update did not improve this ambiguous drawing.`;
      }

      const recheck = await this.controller.predict(this._inputVector());
      this._applyPrediction(recheck, true);
      this.dom.loss.textContent += ` Now it says: ${this.classNames[recheck.predictedClass] !== undefined ? this.classNames[recheck.predictedClass] : recheck.predictedClass} (${Math.round((recheck.confidence || 0) * 1000) / 10}%).`;
    } catch (error) {
      this.dom.loss.textContent = `The correction failed: ${error.message}`;
    } finally {
      this.busy = false;
      this._updateAskState();
    }
  }

  /* ------------------------------------------------------------------- pad */

  _paintSegment(from, to) {
    const minX = Math.min(from.x, to.x);
    const maxX = Math.max(from.x, to.x);
    const minY = Math.min(from.y, to.y);
    const maxY = Math.max(from.y, to.y);
    const radius = 13;
    const cell = PAD_CSS / PAD;
    const x0 = Math.max(0, Math.floor((minX - radius) / cell));
    const x1 = Math.min(PAD - 1, Math.ceil((maxX + radius) / cell));
    const y0 = Math.max(0, Math.floor((minY - radius) / cell));
    const y1 = Math.min(PAD - 1, Math.ceil((maxY + radius) / cell));

    for (let gy = y0; gy <= y1; gy++) {
      for (let gx = x0; gx <= x1; gx++) {
        const cx = (gx + 0.5) * cell;
        const cy = (gy + 0.5) * cell;
        const distance = this._distToSegment({ x: cx, y: cy }, from, to);
        if (distance > radius) continue;
        const strength = Math.min(1, (1 - distance / radius) * 1.25 + 0.15);
        const index = gy * PAD + gx;
        this.pixels[index] = Math.max(this.pixels[index], strength);
      }
    }

    this.ink = true;
    this._invalidateAnswer();
    this._mark('draw');
    this._renderPad();
    this._updateAskState();
  }

  _distToSegment(point, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq === 0) return Math.hypot(point.x - a.x, point.y - a.y);
    let t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy));
  }

  _renderPad() {
    const ctx = this.padCtx;
    if (!ctx) return;
    const cell = PAD_CSS / PAD;
    ctx.fillStyle = '#f4f1ea';
    ctx.fillRect(0, 0, PAD_CSS, PAD_CSS);
    for (let y = 0; y < PAD; y++) {
      for (let x = 0; x < PAD; x++) {
        const value = this.pixels[y * PAD + x];
        if (value <= 0) continue;
        const shade = Math.round(244 - value * 226);
        ctx.fillStyle = `rgb(${shade}, ${shade - 4}, ${shade - 8})`;
        ctx.fillRect(x * cell, y * cell, cell + 0.6, cell + 0.6);
      }
    }
    // A quiet one-pixel grid so the learner can see the 28×28 tensor.
    ctx.strokeStyle = 'rgba(20, 24, 30, 0.055)';
    ctx.lineWidth = 1;
    for (let i = 1; i < PAD; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, PAD_CSS);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(PAD_CSS, i * cell);
      ctx.stroke();
    }
  }

  _clear() {
    this.pixels.fill(0);
    this.ink = false;
    this.prediction = null;
    this.drawing = false;
    this.lastPoint = null;
    this._renderPad();
    this.dom.answer.textContent = this.weightsReady
      ? 'Draw another number and ask me.'
      : 'Press “Teach the brain” first.';
    this.dom.feedback.hidden = true;
    this.dom.loss.textContent = '';
    this._updateAskState();
  }

  _invalidateAnswer() {
    if (!this.prediction) return;
    this.prediction = null;
    this.dom.feedback.hidden = true;
    this.dom.loss.textContent = '';
    this.dom.answer.textContent = 'New drawing — ask me what it is.';
  }

  _inputVector() {
    return Array.from(this.pixels, (value) => Number(value.toFixed(4)));
  }

  _updateAskState() {
    this.dom.ask.disabled = this.busy || !this.ink || !this.weightsReady;
  }

  /* ----------------------------------------------------------------- story */

  _mark(key) {
    if (!STORY_KEYS.includes(key) || this.story[key]) return;
    this.story[key] = true;
    this._updateStory();
  }

  _updateStory() {
    if (!this.dom || !this.dom.story) return;
    let activeAssigned = false;
    for (const item of this.dom.story.querySelectorAll('[data-story]')) {
      const key = item.dataset.story;
      const done = Boolean(this.story[key]);
      const active = !done && !activeAssigned;
      if (active) activeAssigned = true;
      item.classList.toggle('is-done', done);
      item.classList.toggle('is-active', active);
    }
  }
}
