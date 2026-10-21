# Neural Network Studio Platform: Bug Patching & Performance Optimization Manifest

## 1. Overview
During Phase 3 automated verification and stress testing, the test suites identified numerical edge cases, type-check boundary conditions, and execution bottlenecks. Below is the comprehensive log of identified issues, root cause analyses, and implemented resolutions.

---

## 2. Identified Bugs & Patches

### Bug 1: Float64Array Dimensionality Mismatch in Optimizers
- **Symptom**: During initial batch training, weights collapsed into `NaN` values, resulting in 0% convergence across XOR, Circles, and Iris benchmarks.
- **Root Cause**: In `src/engine/optimizers.js`, matrix dimensionality was checked using `Array.isArray(param[0])`. In modern ES6+, typed arrays like `Float64Array` return `false` under `Array.isArray()`. Consequently, the 2D weight matrix was mistakenly treated as a 1D bias vector, attempting element-wise subtraction on an array of typed arrays (`[Float64Array] - scalar = NaN`).
- **Resolution**: Updated the dimensional check in `SGDOptimizer`, `MomentumOptimizer`, `AdamOptimizer`, and `RMSpropOptimizer` to use the explicit `isWeight` boolean flag passed from the layer update loop.
- **Verification**: Verified via `node tests/test_engine_convergence.js`, achieving 100.0% convergence on XOR, 100.0% on Circles, and 100.0% on Iris.

### Bug 2: Unbound Derivative for Softmax with Non-CCE Losses
- **Symptom**: In the reference engine `python_core/neural_engine.py`, pairing `softmax` activation with Mean Squared Error (MSE) threw `TypeError: 'NoneType' object is not callable`.
- **Root Cause**: The activation derivative dispatch table returned `None` for softmax under the assumption that it would only ever be paired with Categorical Cross-Entropy (where $dL/dz = y_{pred} - y_{true}$). When users experimented with other losses, `out_layer.act_deriv` was `None`.
- **Resolution**: Added analytical `softmax_deriv(x) = s * (1 - s)` fallback and bound it inside `get_activation('softmax')`.
- **Verification**: Verified via `python -m unittest python_core.test_engine` (4/4 tests OK).

### Bug 3: Exploding Gradient Instability with Extreme Floating-Point Inputs
- **Symptom**: Extremely large numerical inputs ($> 10^7$) or steep loss gradients caused exponential divergence in exponential activations like Sigmoid and Softmax.
- **Root Cause**: Unbounded `Math.exp(x)` can overflow IEEE 754 float64 when $x > 709.78$, resulting in `Infinity` and cascading `NaN` weights.
- **Resolution**: Added numerical stability clamping `Math.max(-50, Math.min(50, x))` across `activations.js`, `loss_functions.js`, and `neural_engine.py`. Added gradient clipping `[-100, 100]` during backpropagation layer error summation.
- **Verification**: Verified in `tests/test_edge_cases.js` under input values of $\pm 10^8$.

### Bug 4: Memory Leak & Garbage Collection Spikes during 60 FPS Particle Animation
- **Symptom**: Spawning new JavaScript objects for firing particles on every frame or batch caused periodic frame rate stutter (micro-janks) when the browser garbage collector ran.
- **Root Cause**: Instantiating thousands of ephemeral particle objects per second `{ x, y, progress, trail: [] }`.
- **Resolution**: Implemented a fixed-size object pool (`ParticleEngine(800)`) in `src/visualizer/particle_engine.js`. Particles are pre-allocated upon initialization; particles are recycled by resetting an `active = false` flag, reducing heap allocations during rendering to zero.
- **Verification**: Smooth 60 FPS frame time maintained consistently in high-throughput training.

### Bug 5: Zero-Variance Feature Division by Zero in CSV Normalization
- **Symptom**: When users upload a CSV with a constant column (e.g. constant bias feature, all values = 5.0), $(\max - \min) = 0$, resulting in `0/0 = NaN` throughout the dataset.
- **Root Cause**: Unchecked division by span $(x - \min) / (\max - \min)$.
- **Resolution**: Added a zero-variance guard in `DatasetManager.normalize()`: if $|\max - \min| < 10^{-9}$, the column is marked `isConstant = true` and normalized directly to `0.0`.
- **Verification**: Verified in `tests/test_edge_cases.js`.

---

## 3. Active Learning & Real MNIST Phase (Bugs 6-12)

### Bug 6: Mis-Encoded One-Hot Targets in the Online Feedback Test Harness
- **Symptom**: The historical online feedback suite reported pervasive, contradictory failures (loss rising on corrections, `evaluate()` "disagreeing" with `online_step`); at peak 58 of 63 checks failed while the engine itself was mathematically correct.
- **Root Cause**: A ternary typo in the harness-local `oneHot()` produced `[1,1]` for class 0 and `[0,0]` for class 1. Every loss, gradient and accuracy measurement was computed against targets that were wrong for both classes.
- **Resolution**: Corrected the helper to `[c === 0 ? 1 : 0, c === 1 ? 1 : 0]` and added a regression fence that asserts exact one-hot parity before any measurement executes.
- **Verification**: `node tests/test_online_feedback.js` — 75/75 checks, stable across repeated runs; the engine's `evaluate()` now agrees with a fresh `forwardSample` on every probe sample (0 mismatches).

### Bug 7: MNIST Integrity Assertion Weakened by Test Subsetting
- **Symptom**: Subset loads (`maxTrain`/`maxTest`) of the shipped MNIST payload failed structural assertions, and the subset path skipped the full-file integrity check entirely.
- **Root Cause**: `MNISTLoader.buildDescriptor` applied the `maxTrain`/`maxTest` truncation before validation, so the in-memory buffers no longer matched the counts declared in `meta.json`.
- **Resolution**: `fromBuffers` now always validates the shipped payload against the FULL manifest counts before any subsetting; the subset window is applied only afterwards inside `buildDescriptor` (see the code comment at `mnist_loader.js`).
- **Verification**: Every suite that loads a subset (2000/400 in the feedback test) verifies SHA-256 digests and decodes through the identical production path.

### Bug 8: Degeneracy / Full-Frame Warnings Evaluated on the Padded Frame
- **Symptom**: Hostile inputs (a fully inked frame, uniform slabs, tiny specks) could dodge the empty/uniform degeneracy warnings because the zero-padded 28x28 border diluted the ink statistics — a user could receive a confident prediction on garbage without any warning.
- **Root Cause**: Degeneracy classification ran over the padded final tensor instead of the actual resized ink field.
- **Resolution**: Degeneracy detection and the new `FULL_FRAME_INK` warning are evaluated on the real ink content (the resized inner field) inside `src/pipeline/image_processor.js` stage 9.
- **Verification**: `tests/test_image_pipeline.js` (57/57) drives empty canvas, uniform slab, full-frame ink and tiny strokes through the same code path the browser runs.

### Bug 9: Unbounded Online Corrections Could Destructively Shift the Model
- **Symptom**: A human-in-the-loop correction using a single large step (or the batch training rate 0.03) could move the weights far enough in one update that held-out accuracy visibly collapsed — the model "unlearned" other digits to satisfy one drawing.
- **Root Cause**: The online path originally applied full-magnitude single-sample gradients at the training learning rate with no per-coordinate bound.
- **Resolution**: Bounded trust-region update (`applyOnlineUpdate`): the gradient is rescaled so its largest coordinate is at most `ONLINE_GRADIENT_LIMIT = 1`, meaning no weight position can move further than the learning-rate value per step. Corrections never inherit the training rate (default `DEFAULT_ONLINE_LEARNING_RATE = 0.005`), never touch the Adam batch-optimizer state, and fire the same iteration hook the batch trainer uses so the visualizer sees real data.
- **Verification**: `tests/test_online_feedback.js` guard (6 corrections at the studio default, mixed directions: 0.00 pp holdout drop, limit 2 pp) and `python_core/test_active_learning.py` part 6.

### Bug 10: Online-Step Visualizer Frame Shipped Post-Update Gradients
- **Symptom**: The backward gradient particles in the visualizer could disagree with the activations displayed at the same instant — gradient magnitudes were measured against a different weight state than the activations/deltas they were paired with.
- **Root Cause**: `executeOnlineStep` in `nn_worker.js` derived `gradientMagnitudes` from an extra probe forward pass executed AFTER the weights had already been updated.
- **Resolution**: The gradient field is now computed from the backward frame captured during the correction itself: `gradientMagnitudes[l][i][j] = capturedDeltas[l][i] * capturedActivations[l][j]`; the extra probe call was removed. Activations, deltas and gradients now describe one consistent moment.
- **Verification**: Structural identity asserted in `tests/test_online_feedback.js` — delta (x) activation holds for every layer at sampled coordinates, on both the JS and Python engines.

### Bug 11: Early Stop Could Freeze a Saturated-but-Wrong Prediction
- **Symptom**: A correction whose sample reached a numerically tiny loss on the WRONG class could stop early — the sample was "solved" by loss yet still misclassified.
- **Root Cause**: The early-stop condition tested only `loss <= SATURATED_SAMPLE_LOSS (1e-6)`.
- **Resolution**: Early stop now requires BOTH a solved loss AND the argmax matching the human-supplied label; otherwise the loop keeps running to the requested budget (still bounded and safe).
- **Verification**: `python_core/test_active_learning.py` part 5 freezes a saturated + correct sample (stops after 1 of 10 steps) and a saturated + wrong sample (runs the full 10 of 10 steps at loss 27.63). The JS engine exercises the same rule.

### Bug 12: Guard Budget Cliff — Wide-Budget Single-Direction Corrections Can Shift the Positional Prior
- **Symptom**: Stress testing showed repeated wide-budget corrections (6 drawings x 10 steps at 0.005) are bimodal across model seeds: most runs stay within a few points of the held-out baseline, while worst cases collapse held-out accuracy from 100% to 51.75% (a 48.25 pp drop) when the corrections consistently drift in one spatial direction.
- **Root Cause**: Each correction nudges the input-to-hidden positional weights in the drift direction; with a six-correction budget of `6 x 10 x 0.005 = 0.3` per coordinate there is no global mechanism redistributing that accumulated displacement.
- **Resolution**: The studio ships the measured-safe default budget (3 bounded steps per correction, default rate) in `src/ui/canvas_feedback.js`, and the automated guard corrects with mixed drift directions at exactly that budget: measured 0.00 pp worst-case drop across the 8 model configurations tested (5 steps measured up to 4.25 pp; 10 steps up to 48.25 pp).
- **Verification**: `tests/test_online_feedback.js` guard section and `python_core/test_active_learning.py` part 6 (holdout drop limit 2 pp, measured 0.00 pp).

---

## 4. UI Field-Report & Training Continuity Phase (Bugs 13-15)

### Bug 13: Digit Studio Fed 784-Value Tensors into Non-28x28 Networks
- **Symptom**: With a 2-feature dataset loaded (e.g. Two Spirals), drawing on the Digit Studio pad produced a nonsense prediction (`Spiral A 50.8%`) computed from the first two pixel values, and confirmation/labeling failed with the cryptic error `Could not add sample: Expected a 2-value input tensor.`
- **Root Cause**: The pad always emits a 784-value tensor and the worker RPC paths never validated tensor length against the input layer — `forwardSample` silently reads only `layer.inputSize` values, so mismatched predictions looked real and corrections were only caught at the add-sample step.
- **Resolution**: The worker now rejects every length mismatch with an explicit `Input tensor has 784 values but the active network expects 2.` message on all three RPC paths (`RUN_PREDICTION` / `ONLINE_STEP` / `ADD_TRAINING_SAMPLE`); `CanvasFeedbackStudio.setTaskCompatibility` disables prediction/correction/labeling when the active dataset is not 28×28, shows a persistent in-panel notice, and offers a one-click `Load MNIST 0-vs-1 corpus` button.
- **Verification**: `tests/test_training_continuity.js` Part 1 (6 checks) drives 784-into-2 tensors through the real worker on all three RPC paths and asserts the size-quoting errors, while correctly-shaped calls still succeed.

### Bug 14: Hard 500-Epoch Cap Stopped Training and Blocked Continuation
- **Symptom**: A Start run stopped after the fixed 500-epoch budget ("trains for a minute and then stops"), improvement ended at an arbitrary point, and pressing Start again completed instantly — `currentEpoch >= maxEpochs` short-circuited the loop, so the model could never train further.
- **Root Cause**: `nn_worker.js` shipped a hard `maxEpochs = 500` and the app sent the same static budget on every Start; no mechanism existed to train longer or to stop on measured convergence.
- **Resolution**: Runs are now unbounded (`maxEpochs: null`) with a convergence auto-stop: the worker tracks the best measured held-out loss and completes with `reason: "converged"` (plus `stallEpochs` / `bestValLoss`) once it has not improved for `earlyStopPatience` (150) epochs — evaluated per epoch, so the run stops exactly on patience. `Start` again continues the same network; finite `maxEpochs` still caps exactly for the test harness; a 100,000-epoch ceiling backstops pathological endless micro-improvement.
- **Verification**: `tests/test_training_continuity.js` Parts 2-3 (8 checks) sail a live run past epoch 500 (paused at 504), verify the auto-stop fires at exactly 30 flat epochs with a real `bestValLoss`, then confirm 100% held-out accuracy on the long-run model; the pre-existing worker suites remain green (75/75 feedback checks).

### Bug 15: Exported Models Could Not Be Reloaded — Saving Progress Was a One-Way Door
- **Symptom**: `Export JSON` produced a complete `neural_model.json`, but there was no way to load it back into the studio: closing the app (or reloading the page) lost the trained run, and the exported file was only usable outside the app. The export also omitted the training configuration (`loss` / `optimizer` / `learningRate` / `weightDecay`) and the epoch reached — and `NeuralNetwork.fromJSON` silently accepted shape-mismatched files, leaving random weights in place for missing layers.
- **Root Cause**: The worker supported `IMPORT_MODEL` and the app had a `MODEL_IMPORTED` handler, but no UI ever sent the command — there was no button, no file picker, and no restore path for the dataset an imported model needs.
- **Resolution**: New **Load Model** button + JSON file picker. The app validates the export envelope, restores the model's named built-in dataset first when shapes do not match (custom CSVs must be re-uploaded by hand), then the worker rebuilds the exact network. `NeuralNetwork.fromJSON` now validates every structural claim before applying a single weight (layer count, per-layer shapes, finite values), restores `loss` / `optimizer` / rates, resumes the epoch counter from the export metadata and restarts the convergence window. A failed import reports the real reason and leaves the live network untouched.
- **Verification**: `tests/test_training_continuity.js` Part 4 (9 checks) drives the exact export payload (`ModelExporter.buildExportPayload` → `JSON.parse(JSON.stringify(...))`) through the real worker `IMPORT_MODEL` RPC: weights are bit-identical, predictions match the reference to 1e-12, epoch 42 → training continues to 45, and truncated / shape-mismatched files are rejected without touching the live network.
