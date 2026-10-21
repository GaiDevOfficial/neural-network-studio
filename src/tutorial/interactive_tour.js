/**
 * Spotlight curriculum tracks (A/B/C) for the three-tier platform.
 *
 * The step machine and the spotlight presenter live in `src/ui/tutorial_engine.js`; this module owns
 * the curriculum content and the real action gates. Every gated step advances only when the page
 * reports the actual gesture (mode change, block added, training started, prediction made, …) —
 * the Next button is disabled while an action is pending.
 *
 * File: src/tutorial/interactive_tour.js
 */

import { TutorialOverlay, TutorialRunner } from '../ui/tutorial_engine.js';

export const CURRICULUM_TRACKS = {
  track_a: {
    id: 'track_a',
    title: 'Lesson 1 — Your First AI in 60 Seconds (Playground)',
    steps: [
      {
        id: 'a-open',
        title: 'Open the Playground',
        text: 'Press the highlighted **Playground** button. Playground is the no-node warm-up: a locked MNIST 0-vs-1 network, a drawing pad, and real prediction. Use it to see what a trained network does before you build one from blocks.',
        target: '#btn-mode-playground',
        action: { id: 'mode_playground', hint: 'Press Playground.' }
      },
      {
        id: 'a-brain',
        title: 'Read the real network',
        text: 'Each dot is a neuron. The lines are the actual weights, and the moving pulses are forward signals from the engine. The network is the same kind of **Dense Layer** stack you will wire later; nothing here is pre-rendered.',
        target: '#playground-brain'
      },
      {
        id: 'a-teach',
        title: 'Let the network practice',
        text: 'Press **Teach the brain**. It trains on real handwritten digits and updates its weights after every batch. Watch the weight lines change — that is learning, not a loading animation.',
        target: '#playground-teach',
        action: { id: 'playground_training_started', hint: 'Press Teach the brain.' }
      },
      {
        id: 'a-draw',
        title: 'Draw a digit',
        text: 'Draw a 0 or a 1 with the mouse, then press **What is it?**. The pad is normalized into the same 28×28 tensor shape used by Dataset Source, and the answer comes from a real forward pass.',
        target: '#playground-pad',
        action: { id: 'playground_predicted', hint: 'Draw a digit, then press What is it?' }
      },
      {
        id: 'a-feedback',
        title: 'Correct a wrong answer',
        text: 'Press **That’s right** or **Nope — it’s a 1**. A correction runs one real backpropagation step on your drawing and immediately predicts again. This is the smallest possible example of the training loop you will see in the node graph.',
        target: '#playground-feedback',
        action: { id: 'playground_corrected', hint: 'Press That’s right or Nope — it’s a 1.' }
      }
    ]
  },

  track_b: {
    id: 'track_b',
    title: 'Lesson 3 — Build a Digit Classifier from Blocks',
    steps: [
      {
        id: 'b-mode',
        title: 'Choose Beginner · Assist',
        text: 'Press **Beginner · Assist**. This tier auto-wires a block when you drop it and can repair shape mistakes with **Fix automatically**. The preview uses the same compiler as **Run**, so a green preview means the real engine accepts the graph.',
        target: '#btn-mode-assist',
        action: { id: 'mode_assist', hint: 'Press Beginner · Assist.' }
      },
      {
        id: 'b-preset',
        title: 'Load the MNIST classifier preset',
        text: 'Press **MNIST preset**. It builds a real pipeline: **Dataset Source** → **Flatten** (28×28 → 784) → **Dense Layer** → **Dense Layer** → **Dense Layer** → **Sigmoid**, with **Binary Cross-Entropy**, **Adam** and **Epoch Loop** attached. Every block stays editable.',
        target: '#btn-preset-mnist',
        action: { id: 'preset_mnist', hint: 'Press MNIST preset.' }
      },
      {
        id: 'b-blocks',
        title: 'Read the blocks and wires',
        text: 'Data flows left to right. Amber is data, blue is layers, green is activation, coral is loss and optimizers, and teal is monitors. The compiler checks tensor shapes and loss pairing before anything runs.',
        target: '#studio-canvas'
      },
      {
        id: 'b-run',
        title: 'Train the classifier',
        text: 'Press **Run**. The shared worker performs real forward and backward passes. Epochs, losses, confusion counts and predictions all come from those passes. The run may finish before you click Next; your Run action is still recorded.',
        target: '#btn-run',
        action: { id: 'training_started', hint: 'Press Run.' }
      },
      {
        id: 'b-metrics',
        title: 'Watch the measured metrics',
        text: 'The **Training** panel shows measured train loss, held-out loss, accuracy and the best checkpoint. The sparkline uses the same real epoch stream, so a flat curve means the model really stopped improving.',
        target: '#panel-metrics'
      }
    ]
  },

  track_c: {
    id: 'track_c',
    title: 'Lesson 5 — Manual Wiring & SQL Streaming',
    steps: [
      {
        id: 'c-mode',
        title: 'Switch to Advanced · Manual',
        text: 'Press **Advanced · Manual**. Automatic repairs are disabled: sockets connect exactly where you drop them, and every rejected connection reports a real validation code. Use this tier when you want full control.',
        target: '#btn-mode-manual',
        action: { id: 'mode_manual', hint: 'Press Advanced · Manual.' }
      },
      {
        id: 'c-sql',
        title: 'Add the SQL Query source',
        text: 'Drag **SQL Query** from the Data palette onto the canvas. It replaces **Dataset Source** and reads a SQLite table through the bundled read-only Data API. SELECT-only access and `mode=ro` are enforced by the server.',
        target: '#palette-sql_source',
        action: { id: 'source_added_sql', hint: 'Drag SQL Query onto the canvas.' }
      },
      {
        id: 'c-inspector',
        title: 'Configure the query',
        text: 'Select **SQL Query** and set the `.sqlite` path plus one SELECT statement in the Inspector. The last selected column becomes the target; the other numeric columns become features. Normalization and the train/test split match the CSV pipeline.',
        target: '#inspector-panel'
      },
      {
        id: 'c-run',
        title: 'Backpropagate through streamed batches',
        text: 'Press **Run** on a valid SQL graph. The engine performs real gradient updates on the streamed batches, including the same single-sample ONLINE_STEP path used by the Playground.',
        target: '#btn-run',
        action: { id: 'training_started', hint: 'Press Run on a valid SQL graph.' }
      },
      {
        id: 'c-export',
        title: 'Export the trained graph',
        text: 'Open **Export** to save the graph and trained weights as JSON, standalone Python or JavaScript, ONNX, or packaging commands. The export uses the same compiled plan and weights you just trained.',
        target: '#btn-export',
        action: { id: 'export_opened', hint: 'Open the Export menu.' }
      }
    ]
  }
};

/**
 * Thin coordinator injected into `initNodeStudio()`: starts a track through the shared overlay,
 * exposes `active`/`notify` to the page, and forwards completion to the activity log.
 */
export class CurriculumTour {
  constructor(options = {}) {
    this.onComplete = options.onComplete || null;
    /* Live action journal from the page; lets a gesture performed before this track opened count. */
    this.actions = Array.isArray(options.actions) ? options.actions : [];
    this.overlay = options.overlay || new TutorialOverlay({
      onComplete: () => {
        if (this.onComplete) this.onComplete();
      }
    });
  }

  get active() {
    return this.overlay.active;
  }

  get runner() {
    return this.overlay.runner;
  }

  start(trackId) {
    const track = CURRICULUM_TRACKS[trackId];
    if (!track) return false;
    return this.overlay.start(track, { actions: this.actions });
  }

  notify(actionId) {
    this.overlay.notify(actionId);
  }

  finish() {
    this.overlay.finish();
  }
}

/** Low-level helper for tests / embedders that only want the step machine. */
export function runnerFor(trackId, options = {}) {
  const track = CURRICULUM_TRACKS[trackId];
  if (!track) return null;
  return new TutorialRunner(track, options);
}

export { TutorialOverlay };
