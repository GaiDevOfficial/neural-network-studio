# Node Studio — Scratch-Style Visual Model Builder

**File:** `NODE_STUDIO.md`
**Status:** Living specification. Part I = Phase 1 architecture; Part II = Phase 4 packaging, quickstart and the measured Phase-3/4 verification results. All four prompt phases are complete on disk.

Node Studio is the application workspace: an infinite pan/zoom canvas where a model is *wired from
blocks* (data → layers → activations → loss/optimizer → output gauges) instead of configured through
forms. It compiles the visual graph into the **verified engine** (`src/engine/neural_network.js`), the
**worker RPC protocol** (`src/engine/nn_worker.js`) and the **real datasets** (shipped MNIST corpus,
built-in generators, user CSVs, SQL/CSV/JSON/REST connectors). Nothing is duplicated, nothing is mocked.
The retired classic landing page is gone; `src/index.html` is only a redirect shim and the desktop host
opens `graph_studio.html` directly.

---

## Part I — Architecture (Phase 1)

### 1. Graph data structures

All graph code is dependency-free ES modules under `src/graph/` and is exercised in Node by
`tests/test_graph_engine.js` (pure logic, no DOM).

**Socket** — a typed connection point on a node.

```js
{ name: 'out', kind: 'tensor' }   // kinds: tensor | config | monitor
```

| Kind | Carried by | Carries |
|---|---|---|
| `tensor` | data/layer/activation blocks, loss `in` | a real tensor shape (dims array) |
| `config` | optimizer / epoch-loop → loss | hyper-parameters (not data) |
| `monitor` | loss → gauges / confusion / prediction view | a read-only tap on the live model |

**Node** — `{ id, type, x, y, params }`. Types are declared once, centrally, in
`src/graph/node_types.js` (sockets, params with ranges/defaults, category, palette order,
`supported: false` + `reason` for honestly-disabled blocks — see §7).

**Connection** — `{ id, from: { nodeId, socket }, to: { nodeId, socket } }`. One connection per input
socket; tensor sockets accept exactly one incoming wire.

**GraphState** (`src/graph/graph_state.js`) owns `nodes`, `connections`, a `mode` (`assist` | `manual`)
and the invariants:

- `connect()` validates socket existence, kind compatibility, the one-wire-per-input rule and
  **cycle-freedom** (a connection is refused if the target can already reach the source).
- `disconnect()`, `removeNode()` (removes incident connections), `incoming()/outgoing()` lookups.
- `topologicalOrder()` for compile-time traversal.
- `toJSON()/fromJSON()` — full round-trip persistence (graph files saved/loaded from the page).

### 2. Tensor shape model & auto-shape inference (Beginner Mode)

Shapes are real dims arrays: `[28,28]` (image), `[784]` (flattened), `[16]` (dense output).

**Source rules**

| Source | Emits |
|---|---|
| Image dataset (MNIST 0-vs-1, 8×8 digits, stroke fixture) | `[h, w]` (from `dataset.imageDim`) |
| Tabular dataset (XOR, Iris, CSV, SQL, REST) | `[numFeatures]` |

**Transform rules** (implemented in `src/graph/node_engine.js`)

| Block | In rank | Out | Rule |
|---|---|---|---|
| Flatten | any | 1 | `dims → [∏ dims]` (no-op on rank 1) |
| Dense | 1 only | 1 | `[n] → [size]`, `size ∈ [1, 512]` |
| Activation | any | same | passthrough; Softmax additionally requires rank 1 |
| Class filter | classification only | `[numFeatures]` | keeps `k` classes, remaps one-hot targets to `k` outputs |
| Loss | 1 | — (terminal) | see pairing matrix below |

**Loss pairing matrix** (the errors a beginner can never see because assist fixes them first):

| Loss | Requires preceding activation | Requires output size |
|---|---|---|
| Binary Cross-Entropy | Sigmoid | 2 (binary one-hot) |
| Categorical Cross-Entropy | Softmax | `numClasses ≥ 2` |
| Mean Squared Error | any (usual: Linear) | any (usual: 1, regression) |

**Auto-resolution**: when a `tensor` wire would violate the rank rule (e.g. `[28,28] → Dense`),
Beginner Mode inserts a **Flatten** block into the wire (visible on the canvas, logged in the repair
console) and proceeds — exactly the "28×28 → 784 before a Dense layer" requirement.

### 3. Validation (Advanced Mode)

`node_engine.validate(graph)` runs the same inference **without repairs** and returns structured errors:
`E_NO_SOURCE`, `E_NO_LOSS`, `E_DUP_INPUT`, `E_KIND_MISMATCH`, `E_CYCLE`, `E_RANK_MISMATCH`,
`E_PAIRING`, `E_SIZE_MISMATCH`, `E_UNSUPPORTED`, `E_PARAM_RANGE`, plus warnings (`W_UNCONNECTED`).
Every error quotes the offending node and socket. The compile gate refuses to build a network while
errors exist — no silent substitution, matching the platform's zero-mock policy.

### 4. Compilation: graph → executable plan

`compile(graph, context)` is pure and synchronous (data resolution is async and happens *before* it):

```
{BUILT-IN | CSV | SQL | REST | Canvas}   →  resolver (async, real fetches)
        ↓ dataset
GraphState  →  [assist.autoFixGraph]  →  compile()  →  plan:
{
  network:  { architecture, loss, optimizer, learningRate, weightDecay },   // → BUILD_NETWORK
  training: { maxEpochs|null, earlyStopPatience, batchSize, lrSchedule },   // → START_TRAINING + scheduler
  data:     { sourceNodeId, datasetId, filter },                            // → SET_DATASET
  taps:     { gauges, confusion, predict },                                 // → canvas widgets
  chain:    [ordered node ids]                                              // → node highlighting + codegen
}
```

The Dense/Activation chain is **fused** into `architecture` entries (`{size, activation}`) — the exact
structure `NeuralNetwork.buildFromArchitecture` accepts. The run controller posts the platform's
existing RPCs (`SET_DATASET`, `BUILD_NETWORK`, `START_TRAINING`, …) to the existing worker, so all
training semantics (unbounded runs, convergence auto-stop, input-shape guards, online corrections)
apply to node-built models unchanged. A `Learning Rate Scheduler` block is applied for real by the run
controller via `UPDATE_HYPERPARAMS` at epoch boundaries (step decay: `lr × factor` every N epochs).

### 5. Smart Assist (Beginner Mode)

`src/graph/smart_assist.js` — pure operations that never throw (the UI wraps every call):

- `assistConnect()` — connects or repairs (auto-inserts Flatten for rank violations; refuses only
  kind-mismatch / cycles, with a precise reason).
- `autoWireOnDrop()` — after dropping a block, snaps it into the nearest unconnected compatible chain
  position (UI geometry supplies candidate pairs; logic validates them).
- `autoFixGraph()` — idempotent repair pass: ensures one source, one loss, wires dangling chain ends,
  inserts missing Flattens, attaches default optimizer/epoch-loop when missing, resolves duplicate
  inputs (keeps the first wire, reports), and merges split chains. Returns a log of every repair.
- `instantPreset(name)` — one-click graphs (`mnist_classifier`, `xor_classifier`) laid out with real
  coordinates, compiling to measurable models in < 10 s of interaction (Tutorial 1's "AI in 30 seconds").

### 6. Data pipelines

| Source | Path |
|---|---|
| Built-in / CSV / Canvas | `DatasetManager` (the shared parser/normalizer used by the data blocks) |
| SQLite | `GET /api/data/sqlite?db=…&query=…&limit=…&offset=…` from the local Python Data API (`desktop/studio_data_api.py`, stdlib-only), rows → `DatasetManager.parseCSV`-equivalent normalization via the row→CSV bridge, so semantics are **identical** to a CSV upload |
| CSV file (path) | `GET /api/data/csv?path=…` |
| REST API | direct `fetch()` from the page → `{columns, rows}` → same row bridge (honest errors surface to the console) |

Batching: `streamBatches(dataset, {batchSize, shuffle, seed})` yields real mini-batches for the
streaming test and for future incremental training; the current run loop uses the worker's own
shuffled epochs used by the run loop.

### 7. Honest deferrals (declared, never faked)

- **Conv2D / MaxPool / Dropout blocks** are present in the palette under *Planned* — visibly disabled
  with the reason shown on hover (`engine v1 is a Dense-stack engine`). They are never wired into a
  compile path that would pretend to run them.
- **ONNX export** ships as `tools/export_onnx.py`, which exports through the `onnx` package when it is
  installed and fails with a clear message otherwise (no stub files).
- **Parquet** is deferred; CSV/JSON/SQLite/REST are implemented.

### 8. File map (prompt → delivered)

| Requested | Delivered |
|---|---|
| `src/graph/node_engine.py (or .js)` | `src/graph/node_engine.js` (shape inference, validation, compile) + `graph_state.js` + `node_types.js` |
| `src/modes/smart_assist.py (or .js)` | `src/graph/smart_assist.js` |
| `src/data/db_connectors.py (or .js)` | `src/graph/data_connectors.js` + `desktop/studio_data_api.py` (Python side; `src/data/` already holds the MNIST payload) |
| `src/ui/canvas_studio.py (or .js)` | `src/ui/canvas_studio.js` + `src/graph_studio.html` + `src/css/node_studio.css` |
| `src/ui/tutorial_engine.py (or .js)` | `src/ui/tutorial_engine.js` |
| `src/export/builder.py (or .js)` | `src/export/builder.js` + `codegen.js` + `npz_writer.js` |

---

## Part II — Packaging & Quickstart (Phase 4)

### 9. What ships where

| Artifact | Path | How to open |
|---|---|---|
| Node Studio page | `src/graph_studio.html` | `python server.py` → http://127.0.0.1:8080/graph_studio.html; `/` and `/index.html` redirect there |
| Static web bundle | `dist/web/graph_studio.html` | serve `dist/web/` from any static host; built-in datasets work with no backend |
| Portable Windows app | `dist/NeuralNetworkStudio/NeuralNetworkStudio.exe` | double-click the launcher (keep the folder beside it); it opens Node Studio directly in its window |
| Public distribution | `dist/NeuralNetworkStudio-Windows-x64.zip` | extract, keep the folder together, run `NeuralNetworkStudio.exe`; checksum in `dist/NeuralNetworkStudio-Windows-x64.sha256.txt` |
| Developer-only one-file exe | `dist/NeuralNetworkStudio.exe` | `python desktop/build_exe.py --onefile`; Defender flags this bootloader, so it is **not** for distribution |
| Developer browser smoke test | `tools/smoke_node_studio.py` | `python tools/smoke_node_studio.py` (drives the real page in WebView2 and measures the DOM) |

The SQL / CSV-file / REST blocks call the local Data API (`desktop/studio_data_api.py`), which is wired
into both `server.py` (`/api/data/*`) and the exe's built-in server. On a plain static host those three
source blocks show the honest fetch error; everything else (built-in datasets, CSV **upload**, training,
exports) is fully client-side.

### 10. Quickstart

**Build a model in 30 seconds**
1. Open the Node Studio, press **MNIST preset** in the top bar. The canvas fills with the real graph:
   `Dataset Source → Flatten → Dense 16 → ReLU → Dense 8 → ReLU → Dense 2 → Sigmoid → BCE`, with Adam,
   the Epoch Loop and the three output taps (metrics gauge, confusion matrix, prediction view).
2. Press **Run**. The status strip shows the compiled plan and the resolved corpus
   (`MNIST Handwritten Digits … (2500 rows)`); live train/held-out loss, accuracy and the sparkline
   update as the worker trains, the confusion matrix fills after the run converges, and the prediction
   panel becomes a 28×28 drawing pad you can probe live.
3. Press **Export** for the model JSON (loadable by both studios), standalone Python (NumPy),
   a standalone ES module, or a NumPy `.npz` weight archive; **Packaging commands** prints the real
   `npm run build:exe` / `--web-only` invocations.

**Beginner vs Advanced** — Beginner mode (Smart Assist) repairs wiring while you build: dropping a Dense
on an image source auto-inserts Flatten, occupied inputs get replaced, consecutive activations are
resolved, and **Fix automatically** runs the repair loop until the graph compiles. Advanced mode is
strict: every refused wire reports its structured error code and nothing is changed behind your back.

**Lessons** — the **Lessons** menu is a single numbered curriculum: ten step-by-step guides over the
live page. Lessons 1/3/5 are the Playground, Scratch-block and manual-wiring tracks; 2/4/6 are the
preset, backpropagation and SQL deep dives; 7/8/9 build XOR, the MNIST classifier and the advanced
regression pipeline from a blank canvas; 10 exports and deploys a trained model. Every card shows the
current step, progress and a real action gate; a small XOR run that finishes before the learner presses
Next remains satisfied because gates read the persistent action journal, not the momentary `running` flag.

**Data sources**
- *Dataset Source*: the built-in benchmarks + real MNIST, or **custom CSV upload** (inspector → file picker).
- *SQL Source*: local `.sqlite`/`.db` file + a `SELECT` query; **Test data fetch** pages the whole
  result (≤500 rows/page, 5000-row default ceiling) through the read-only Data API and reports rows/classes.
- *REST Source*: any URL returning a JSON array of objects or `{columns, rows}`.
- *Class Filter*: keep a subset of classes; one-hot targets are remapped before training.

**Build commands** (exact):
```powershell
python desktop/build_exe.py             # dist/web + portable app folder + public ZIP (recommended)
python desktop/build_exe.py --onefile   # developer-only single-file exe (Defender flags it; do not ship)
python desktop/build_exe.py --web-only  # static bundle only
python desktop/build_exe.py --exe-only  # desktop package only
```

**Verification commands** (exact):
```powershell
npm test            # 18 suites: engine, edge, MNIST, image, feedback, continuity, graph, nodes, db, exports, ONNX, UI, menus, drag, Playground, wires, quick-add, missions + examples
npm run test:python # engine + active learning + data connectors + desktop host (26 unittest cases)
python tools/smoke_node_studio.py   # real-browser UI smoke (needs a desktop session)
```

### 11. Measured results (quote these, not promises)

| Suite | Result |
|---|---|
| `tests/test_graph_engine.js` | **80/80** — registry vs real engine, wiring invariants, pairing matrix, compile, Smart Assist incl. 30-seed fuzz (no throw), presets, geometry, scheduler, tutorials |
| `tests/test_node_convergence.js` | **20/20** — node-built MNIST preset through the real worker RPC path: **100.00%** held-out (800 real digits) within the 10-epoch budget, confusion-matrix counts equal the evaluator exactly; XOR **100%**; custom-CSV graph **100%**; Class-Filter Iris **100%** |
| `tests/test_db_streaming.js` | **14/14** — spawns the real `studio_data_api.py`; SQLite windows + pagination, connector rows numerically identical to a CSV upload, CSV tail window, REST parsing, error contract, `streamBatches` coverage |
| `tests/test_node_exports.js` | **19/19** — the generated Python is executed under real Python/NumPy: max |Δ| vs engine **1.39e-17** (XOR, demo) / **3.89e-16** (arbitrary inputs) / **1.59e-21** (softmax probes); generated ES module ≤ **1.39e-17**; `.npz` opens with `zipfile`+`numpy` (shapes + ΣW match to 1e-9); model JSON reloads through `fromJSON` bit-identically |
| `python_core/test_data_connectors.py` | **12 tests OK** — endpoint contract, pagination/guards, live HTTP round-trip |
| `tools/smoke_node_studio.py` | **12/12** in real WebView2 — boot, palette (27 blocks), MNIST preset (14 blocks / 13 wires / valid compiled plan), real training advancing epochs with live measured losses, zero runtime errors, pause keeps measured state |
| Full `npm test` | **all 12 suites green** (4/4 engine, 7/7 edge, 99.88% real-MNIST legacy suite, 57/57 image, 75/75 feedback, 27/27 continuity, then the five node-studio suites) |

### 12. Bugs found by Phase-3/4 verification and fixed

1. **Worker emitted no live updates at full speed** — in the fast path (`epochDelayMs === 0`) the
   snapshot flag was computed *before* each epoch ran, so a chunk consisting of one slow epoch (e.g.
   MNIST ≈ 340 ms) never emitted a single `TRAINING_STEP`; the UI stayed frozen until the run completed.
   Fixed with a lazily-built pending snapshot: a chunk that never saw a flagged epoch now posts its
   last completed epoch (cost is paid only when the payload is actually sent).
2. **Node Studio page crashed during boot** — `renderValidation()` ran before any plan existed
   (`state.plan === null`) which aborted the rest of `initNodeStudio`; status stayed "Booting…" and the
   particle loop never started. Boot now runs `validateNow()` and the validator has an explicit
   empty-canvas state.
3. **Connector windows dropped `hasMore`** — found by the DB suite: `fetchSqliteWindow`/`fetchCsvWindow`
   returned no pagination flag, so callers could not tell a full window from the tail; both now relay it
   and `resolveSourceDataset` stops on the real server flag.

*(Smart Assist's own three implementation bugs — action-log wipe on convergence, activation replacement
without disconnecting, consecutive-activation detection — are recorded in Part I / the suite that caught
them.)*

---

## Part III — Three-Tier Platform Upgrade (Phase 1 design)

This part specifies the upgrade from the two-mode editor to the full three-tier platform: an
Absolute-Beginner playground, a universal menu lifecycle, drag-to-delete, ghost suggestions, the
three curriculum tracks, and the ONNX exporter. All numbers remain measured; nothing below is a mock.

### 13. The three tiers as one state machine

`state.mode ∈ { 'playground', 'assist', 'manual' }`. The graph, engine, worker RPCs and datasets are
identical in every tier — only the editing affordances and the presented vocabulary change.

| Tier | `mode` | Workspace | Wiring policy | Vocabulary |
|---|---|---|---|---|
| 1 — Visual playground | `playground` | `#playground` overlay (editor hidden) | Locked pre-assembled MNIST graph; training + probing only | Jargon-free ("pictures", "brain cells", "answer", "mistakes") |
| 2 — Smart Node Assist | `assist` | block canvas | Auto-wire on drop, auto-repair, ghost suggestions, snap assist | Block names + plain-language summaries |
| 3 — Full Manual | `manual` | block canvas | Raw sockets; strict validation; no automatic mutations | Technical (shapes, losses, optimizers) |

Transitions:
* `playground → assist|manual`: dispose the playground render loop (not the worker), re-show the editor,
  re-render from the same graph — training state is preserved.
* `assist ↔ manual`: only the wiring policy changes; an undo snapshot is pushed first.
* `assist|manual → playground`: ensure the MNIST preset is loaded (real corpus resolved), build/show the
  brain canvas; if a different graph is on the canvas, it is replaced through the same `applyPreset`
  path (undo snapshot kept).

### 14. Absolute-Beginner data flow (all real)

```
padding canvas → 28×28 pixels [0,1]
        │
        ▼
controller.predict(pixels) ── RUN_PREDICTION(source:'playground') ──► worker
        │                                                            │
        │            PREDICTION_RESULT {probabilities, activations} ◄┘
        ▼
friendly answer + confidence                  glowing cells (SynapticCanvas)
                                                   ▲
“Nope, it’s a 1” → controller.correct(pixels,1)    │ TRAINING_STEP {activations, deltas, weights}
        │                                          │
        ▼                                          │
ONLINE_STEP {sample,target,source:'playground'} ──► worker ── ONLINE_STEP_COMPLETE
        │                                          {initialLoss, postLoss, lossDelta, weights,
        ▼                                           activations, deltas}
“Before X → after Y” + immediate re-prediction
```

The playground only calls real worker RPCs exported by the controller (the same functions the main
Run button uses): `prepare`/`startTraining`/`pause`/`reset`/`predict`/`correct`. The training loop is
never duplicated and the drawn tensor is the exact `[0,1]` 28×28 vector the engine receives.

### 15. Universal menu lifecycle manager (`src/ui/menu_manager.js`)

One registry owns every popup, dropdown, modal and tour overlay:

* `register({ id, trigger, panel, kind: 'popover'|'modal'|'overlay', requestClose, isOpen })`
* `open(id)` / `close(id)` / `toggle(id)` / `closeAll()`; only one popover is open at a time.
* `Escape` closes the topmost open entry (a stack, so nested overlays unwind in order).
* A document-level pointerdown outside a popover closes it; clicking its trigger toggles it.
* Every registered popover gets an injected `×` close button (`data-menu-close`) when it does not
  declare one, so no panel can become unclosable.
* Modals/overlays expose `requestClose()` (the tutorial maps it to `finish()`, the playground story
  panel to its own close path); they register Esc + X like everything else.
* No DOM access happens at module import time (Node-safe), and `destroy()` removes every listener.

### 16. Drag-to-delete lifecycle

1. `pointerdown` on a block records the drag origin and moves the block normally.
2. While a node is being dragged, the palette's rectangle is tested on every `pointermove`
   (`NodePalette.isOver`). Inside it: the palette gains `.is-trash-target` (red border + "Drop to
   delete" overlay) and the dragged block gains `.is-trash-armed`.
3. `pointerup` over the palette calls `deleteNode(nodeId, {viaTrash:true})`:
   * push one undo snapshot,
   * `GraphState.removeNode(id)` deletes the node **and every wire touching it** (`connections` map),
   * delete its per-node caches (`sourceInfo`, `sourceData`, `featuresMeta`), clear `selectedId`,
   * re-render nodes/wires/inspector/validation; the DOM/SVG is rebuilt, so no detached element keeps
     the node alive,
   * log `Dropped <block> in the trash — N wire(s) severed.`
4. Every iteration is O(nodes + wires); repeated add/delete cycles leave both maps at zero (verified
   by `tests/test_drag_delete.js`).

### 17. Smart Assist ghost suggestions & snapping (`src/modes/smart_assist.js`)

The existing repair engine (`src/graph/smart_assist.js`) stays the algorithm core. The mode module adds:

* `suggestConnection(graph, selectedNodeId, context)` — clones the graph, tries every legal
  selected-node → free-input candidate, compiles each candidate, and returns the one that reduces the
  real error count. Returns `{ from, to, errorReduction }` or `null`.
* `snapTarget(ports, worldPoint, drag, radius)` — during a wire drag in Beginner mode, compatible
  sockets within 34 px are highlighted (`.ns-socket-snap`) and the nearest one is used on release;
  Advanced mode keeps the strict 18 px hit test.
* The ghost suggestion is drawn as a dashed wire (`.ns-wire-ghost`) and offered in the validation
  panel with an **Apply suggestion** button / `Enter` key. Applying it goes through `assistConnect`,
  never a direct mutation.

### 18. Spotlight curriculum tracks (`src/tutorial/interactive_tour.js`)

Three tracks, action-gated by the real UI (the `Next` button is disabled while a step has an action):

| Track | Steps (target → required action) |
|---|---|
| A — How Machines Think (60 s) | mode button → `mode_playground` · brain canvas → info · Teach button → `playground_training_started` · pad → `playground_predicted` · feedback → `playground_corrected` |
| B — Digit Classifier with Scratch Blocks | Beginner mode → `mode_assist` · palette → `node_added` · canvas → `node_connected` · Run → `training_started` · metrics → info |
| C — Custom Backprop & DB Streaming | Advanced mode → `mode_manual` · SQL palette block → `source_added` · inspector → info · Run → `training_started` · Export → `export_opened` |

The overlay keeps the page interactive so the gated gesture can happen, adds `Esc` + an explicit `×`
close path, and announces completion through the controller (logging + story progress), never by
pretending the action occurred.

### 19. Exporters added in this phase

* **ONNX** (`src/export/onnx_writer.js`): a dependency-free protobuf writer emitting a real ONNX
  inference graph — one `Gemm` per trained Dense layer (`transB = 1`, raw float initializers) followed
  by the layer's activation (`Relu`/`LeakyRelu`/`Sigmoid`/`Tanh`/`Softmax`); opset 13, IR 8.
  Verified by loading the bytes with `onnxruntime` and comparing every output against the engine.
* **Packaging commands** remain the exact repo commands (`npm run build:exe`, `--web-only`), and the
  canonical PyInstaller spec ships at `desktop/NeuralNetworkStudio.spec`.

### 20. Phase-3 verification plan

1. **Modal regression** — real WebView2: open Export → `Esc`; open Tours → outside click; click the
   injected `×`; start a tour → `Esc` removes `.tour-root`; every menu is closed afterwards.
2. **Drag-to-delete** — pure (GraphState + cache lifecycle, repeated cycles) and real browser
   (pointer drag over the palette removes the block and its wires).
3. **Active learning** — worker-level: predict a drawn sample, run `ONLINE_STEP` for the true class,
   re-predict; assert the corrected-class probability and sample loss improve on the immediate pass.
4. **ONNX parity** — generated bytes loaded by `onnxruntime`; max |Δ| vs engine measured.
5. Playground smoke — Tier-1 mode boots, teaches, predicts a synthetic drawing, corrects it, and
   reports the measured loss drop.

---

## Part IV — Three-Tier Upgrade: Implementation & Verification (Phase 3/4 complete)

Status: **implemented, verified and packaged** on 2026-10-19. Everything below is measured by the
suite named next to it; no metric in this part is estimated or mocked.

### 21. Files delivered by the upgrade

| File | Responsibility |
|---|---|
| `src/ui/menu_manager.js` | One Escape stack / outside-click owner / injected × for every popup, dropdown, modal and overlay. |
| `src/ui/action_bar.js` | Tier buttons, dataset chip, Live-metrics action, training-state labels. |
| `src/canvas/node_palette.js` | Registry-driven palette, real drag sources, `isOver()` trash geometry, red veil. |
| `src/modes/smart_assist.js` | Ghost-wire scoring by recompiling cloned graphs, 34 px snap assist. |
| `src/modes/absolute_beginner.js` | Tier-1 playground DOM, pad, Teach/Ask/Good/Wrong flow, story progress, brain canvas lifecycle. |
| `src/visualization/synaptic_canvas.js` | Friendly-label wrapper around `SynapseRenderer`, ResizeObserver + deterministic resize. |
| `src/tutorial/interactive_tour.js` | Curriculum tracks A/B/C, action-gated steps, deep-dive menu entries. |
| `src/export/onnx_writer.js` | Dependency-free protobuf ONNX writer (IR 8, opset 13). |
| `src/graph_studio.html` | Two node-mode tabs, dataset/metrics chips, Lessons/Graph/Export menus, relocated Playground button, playground markup. |
| `src/css/node_studio.css` | Menu close buttons, playground grid, trash veil, ghost wires, snap sockets, tour close, universal `[hidden]` rule. |
| `tests/test_menu_lifecycle.js` | Menu/overlay lifecycle regression, 28 assertions. |
| `tests/test_drag_delete.js` | Palette trash geometry + graph sanitisation, 209 assertions. |
| `tests/test_playground_flow.js` | Tier-1 data path on real MNIST, 34 assertions. |
| `tests/test_onnx_export.js` | Five ONNX models, official checker + ORT parity, 48 assertions. |
| `tools/onnx_parity_check.py` | Runs `onnx.checker` and `onnxruntime`, returns JSON for the JS suite. |

### 22. Verified behavior by tier

**Tier 1 — Absolute Beginner Playground.** `setMode('playground')` hides `#studio-main`, shows
`#playground` and reuses the same worker/controller. `prepare()` loads the `mnist_classifier` preset
unless `state.presetId` already matches, pauses any active run before replacing the graph, resolves
the real 2500/800 MNIST split, and returns the current plan/weights. The pad renders 280×280 and
converts strokes to the same 784-value `[0,1]` tensor the engine consumes. Predictions are real
`RUN_PREDICTION` calls (source `playground`); corrections are one real `ONLINE_STEP` with a one-hot
target, `steps:1`, `learningRate:0.05`, followed by an immediate re-prediction. The canvas receives
the returned activations/deltas/weights; no local inference path exists.

**Tier 2 — Smart Node Assist.** Drop a block: legal inputs are auto-wired, an occupied input is
replaced, image sources get Flatten, and a default Source/Loss/Optimizer/Loop chain is completed.
`suggestConnection()` clones the graph and compiles every candidate selected-node → free-input
wire; the candidate with the largest real error-count reduction becomes the dashed ghost wire.
Dragging a wire snaps to a compatible socket within 34 px (18 px in Full Manual); Enter applies
the suggestion through `assistConnect`, never by direct mutation.

**Tier 3 — Full Manual.** Raw sockets, strict hit test, no automatic repair; invalid wires keep the
structured error code and the validation panel explains the real shape/loss/optimizer mismatch.

### 23. Universal menu lifecycle (all three tiers)

`MenuManager` is the single owner. Its Escape priority is: **open menu → active overlay → nothing**.
The Tours and Export menus, every future popover, the curriculum overlay and the tutorial all register
with it. `TutorialOverlay` runs with `manageEscape:false`, injects its own × (`data-tour="close"`),
and starts with Next disabled on action-gated steps. The regression suite exercises open/switch/close,
Escape unwinding a popover before an overlay, outside vs. inside pointerdown, injected ×, trigger
toggle, unregister and `destroy()` listener cleanup.

A real WebView2 run confirms all three close paths for the shipped dropdowns, a Track-A tour with a
disabled Next, and Escape removing `.tour-root`. A global `[hidden]{display:none!important}` rule was
added because several author display rules were overriding the lifecycle attribute.

### 24. Drag-to-delete: exact behavior and verification

While a block is dragged, `NodePalette.isOver(clientX, clientY)` tests the palette's real bounding
rectangle. Inside it the palette gains `.is-trash-target` (red border + veil copy `Drop to delete`)
and the block gets `.is-trash-armed`. On release over the palette, `deleteNode(id,{viaTrash:true})`:

1. pushes one undo snapshot;
2. removes the node and every connection touching it;
3. drops `sourceInfo`, `sourceData`, `featuresMeta`, and the shared `dataCache` entry when the last
   source stops using it;
4. clears selection, re-renders nodes/wires/inspector/validation and logs
   `Dropped <block> in the trash — N wire(s) severed.`

`tests/test_drag_delete.js` deletes every block of XOR, Iris and MNIST one at a time and proves the
exact number of severed wires, zero dangling endpoints, no exception on recompile, and a full sweep
that ends with zero nodes and zero wires (209/209). The WebView2 smoke performs the same gesture with
real pointer events and observes 14 → 13 blocks, the armed state, the cleared trash state and the log.

### 25. Curriculum tracks

| Track | Real actions that unlock Next |
|---|---|
| A — How Machines Think (60 s) | switch to Playground → start real training → draw + ask → correct once. |
| B — Digit Classifier with Scratch Blocks | switch to Beginner → add a block → make a real connection → Run → inspect metrics. |
| C — Custom Backprop & DB Streaming | switch to Advanced → add a SQL block → inspect → Run → open Export. |

The DOM contract suite checks every track target against the real page (or the runtime palette ids)
and that all three track buttons exist. The smoke starts Track A and asserts the first step's gated
Next is disabled, the tooltip × exists, and Escape removes the overlay.

### 26. ONNX exporter verification

`src/export/onnx_writer.js` emits the protobuf by hand: one `Gemm` per Dense layer (`transB=1`,
float32 raw initializers for weights and biases), followed by the mapped activation
(`Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `Softmax`; `linear` emits no node). Model metadata is
IR 8 / opset 13 with a `[1, features] -> [1, classes]` graph.

| Model | Bytes | Nodes | max abs difference vs. JS engine |
|---|---:|---:|---:|
| XOR (trained, Relu → Sigmoid) | 995 | 6 | 8.378e-8 |
| Iris (trained, Relu → Softmax) | 923 | 4 | 9.775e-8 |
| MNIST 784→16→8→2 | 51,499 | 6 | 3.009e-8 |
| LeakyRelu → Tanh → Sigmoid | 872 | 6 | 2.973e-8 |
| Linear regression (MSE) | 511 | 3 | 2.055e-7 |

Every model passes the official `onnx.checker` (IR 8, opset 13, unique names, topological order)
and runs in `onnxruntime` 1.30.0. The exporter refuses empty/untrained snapshots and malformed weight
matrices. The UI's Export → ONNX path is exercised in the WebView2 smoke and logs the artifact.

### 27. Full verification run (2026-10-19)

| Command | Result |
|---|---|
| `npm test` | **15/15 JS suites, exit 0** (engine, edge, MNIST, image, feedback, continuity, graph, nodes, DB, exports, ONNX, DOM, menus, drag, playground) |
| `npm run test:python` | **22/22 tests OK** (engine, active learning incl. 50/50 checks, connectors) |
| `python tools/smoke_node_studio.py` | **52/52 checks, exit 0** in real WebView2 |
| ONNX suite | 48/48; parity 3.0e-8 to 2.1e-7 |
| Menus suite | 28/28 |
| Drag/delete suite | 209/209 |
| Playground data path | 34/34 |
| Static DOM contract | 135/135 |

Measured playground numbers: 2 practice epochs on 600 real digits reached **100.00% held-out**
on the 160-digit sampled split; a genuinely wrong digit's one-step correction moved the loss
**2.20528 → 1.71745** (22.12% reduction) and the true-class probability **0.0578 → 0.1066**; the
immediate re-prediction equals a fresh forward pass. The browser run reached a measured **99.9%**
held-out check over all 800 test images after real training, then completed draw → ask → Nope →
measured-loss trace → re-prediction.

Two real defects were found by this verification and fixed:

1. **Loss message wiped after correction.** `_applyPrediction()` cleared `#playground-loss` even on
   the post-correction re-prediction, erasing the real before/after loss. It now clears only on a
   fresh Ask, so the measured trace remains visible.
2. **Collapsed brain canvas.** The renderer ran once while `#playground` was hidden, writing
   `width/height:0px` inline; that beat the CSS size and left a 300×150 (or 0×0) backing store.
   Resize now refuses to write zero dimensions, a ResizeObserver watches the canvas parent, and
   `open()` resizes synchronously plus via rAF/timer fallbacks. The smoke asserts the backing
   store equals the CSS box (800×579 at 1× in the test window).

### 28. Packaging & quickstart

```powershell
# install test-only ONNX verification deps (not runtime deps)
python -m pip install onnx onnxruntime

# full verification
npm test
npm run test:python
python tools/smoke_node_studio.py

# rebuild static web + portable app (kill a running exe first)
python desktop/build_exe.py
.\dist\NeuralNetworkStudio\NeuralNetworkStudio.exe
```

The static bundle is `dist/web/` (open `graph_studio.html`); the desktop app embeds the same
`src/` tree through `desktop/NeuralNetworkStudio.spec`. The ONNX packages are only needed to run the
parity suite, not by the application. `dist/web` and the `.exe` are rebuilt from the final source
after this document was written; the build script reports the actual binary size.

### 29. Remaining deferrals / open thread

* Conv2D / MaxPool / Dropout remain *Planned* palette blocks; engine v1 is a Dense stack.
* Parquet/folder loaders remain unimplemented.
* The separately requested **10 example projects pack** (`example_projects.md` plus verified
  CSVs) is still pending the user's go-ahead; no placeholder files have been emitted.

---

## Part V — Interactive Wire Mechanics & Guided Construction (Phase 1 architecture)

This part specifies the interaction upgrade: right-click wire sniping with a context menu, drag-wire-to-empty-space
quick node creation, and a build-from-scratch, action-gated mission system across the three tiers. The
implementation lives in five new modules (`src/canvas/wire_interactions.js`, `src/canvas/quick_node_menu.js`,
`src/tutorial/step_by_step_engine.js`, `src/tutorial/explanations_catalog.js`, `src/modes/interactive_modes.js`)
and reuses the existing graph engine, compiler and worker unchanged — every visual action mutates the same
`GraphState` that the compiler and backpropagation run on.

### 30. Coordinate model and wire raycasting

Wires are cubic Béziers in **world coordinates** rendered as SVG paths inside the transformed `#graph-layer`
(`transform: translate(view.x·s, view.y·s) scale(s)`). The wire layer has `pointer-events: none`, so hit-testing
cannot use DOM events; it samples the Bézier and measures distance to the resulting polyline:

1. `screenToWorld(clientPoint, state.view)` → world point `p`.
2. For every connection, `socketPosition(fromNode, …, 'out')` → `A`, `socketPosition(toNode, …, 'in')` → `B`.
3. `flattenBezier(A, B, 28)` → 29-point polyline; `distanceToPolyline(p, polyline)` → `{ distance, t, point }`.
4. `hitTestWires(wires, p, radius)` returns the nearest wire whose distance ≤ radius.
   `radius = clamp(10 / view.scale, 5, 22)` world units, so the feel is constant in screen space.
5. Priority on the canvas: **socket hit → wire hit → background**. The existing 18 px socket hit test runs first,
   so a click near a port always wires; the wire menu only opens on the stroke itself.

All of (2)–(4) are pure functions (no DOM) and are unit-tested directly against real `GraphState` graphs.

### 31. Right-click wire disconnection lifecycle

`contextmenu` on `#studio-canvas` (left-button filtering untouched):

1. `preventDefault()` and compute the world point from `clientX/clientY`.
2. `hitTestWires` against the live connection list; a miss closes any open popover and returns.
3. A hit builds the **wire menu model**:
   `{ connectionId, fromLabel, toLabel, kind, detail, warning, items: [{ id: 'break', label, danger }] }`.
   The tier policy (Part V §35) decides friendly vs technical copy. In Advanced mode the model also clones
   the graph, disconnects the wire, re-runs `inferGraph`, and reports the **real** post-break error delta as
   the menu warning (never a guessed message).
4. The menu is a `MenuManager` popover (`wire-menu`) anchored at the cursor; Escape, outside-click, the injected
   `×` and the item itself close it.
5. **Break sequence:** `WireBreakAnimator.play(path)` adds `.is-breaking` (stroke fade + dash collapse, 180 ms),
   then calls the page's `onBreak(connectionId)`, which
   pushes one undo snapshot, `graph.disconnect(id)`, clears `presetId`, runs `afterGraphMutation({ resolve: true })`,
   logs `Broke <from> → <to>.`, and notifies the mission engine (`wire_broken`).

### 32. Drag-wire to empty space → quick node creation

Gesture state machine added to the existing `wireDrag` flow:

```
idle ──pointerdown on socket──▶ dragging ──pointerup on compatible socket──▶ wired (existing attemptConnect)
                                    │
                                    ├─ hover over empty canvas → canvas shows .is-wire-drop (dashed ring)
                                    └─ pointerup over empty canvas
                                          ▶ dropped-on-empty   (wireDrag → pendingDrop, graph untouched)
                                          ▶ quick menu opens at the cursor with the pending wire still drawn
                                          ▶ pick type → applyQuickAdd() → node placed at the release point and
                                            wired in the same gesture → idle
                                          ▶ Escape / outside / × → pendingDrop cleared, no mutation → idle
```

**Candidate generation (real compile scoring, zero guesses):**

1. `baseCandidates(graph, drag)` lists supported node types that own a socket with the dragged `kind` and the
   opposite `direction` (`out` drag → candidate `in` socket; `in` drag → candidate `out` socket).
2. `scoreCandidates(...)` clones the graph per candidate, adds the block at the release point, performs the wire
   (`assistConnect` in Beginner — including real Flatten insertion for rank repair — or strict `connect` in
   Advanced), re-runs `inferGraph`, and scores `errorsBefore − errorsAfter` minus warnings.
   Each row carries `{ applies, errorsAfter, reason }` measured on the real compiler.
3. `applyQuickAdd(graph, drag, type, position, { mode })` mutates the live graph: add node → best compatible
   socket → connect (assist repair or strict) → rollback on failure. Returns
   `{ ok, node, connection, actions, error }`.
4. **Smart filtering** is the menu itself: incompatible blocks are never listed; compile-clean candidates are
   badged; blocked ones show the exact error code the engine would raise.

The popover is searchable (label/summary/aliases), grouped (friendly intents in Tier 1, registry categories in
Tiers 2/3), keyboard driven (↑/↓/Enter), and cursor-anchored with viewport clamping. It is a `MenuManager`
popover (`quick-node-menu`) so the universal close contract applies.

### 33. Step-by-step mission engine schema

```
TutorialStep {
  id, title, concept,               // concept → explanations_catalog entry
  text,                             // one-sentence instruction
  target,                           // CSS selector, or { nodeType, socket, direction } resolved by the page
  action?: { id, hint, validate? }  // gate: a real event id and/or a graph predicate
  setup?: { id, payload }           // page command performed on step entry (e.g. clear_graph)
  optional?: boolean
}

Mission { id, tier, title, goal, steps: TutorialStep[] }

ValidationRule { evaluate(ctx) → { ok, reason }, describe(params) }

MissionRunner {
  index, done, completed[],
  enter(index), next(force?), prev(), skip(),
  notify(actionId, ctx),           // event gate from the page
  evaluateGate(ctx)                // predicate gate, re-run after every graph mutation
}
```

Built-in predicates (all read the live `GraphState`/state, never a scripted flag):
`modeIs`, `hasNode`, `wireExists(fromType,fromSocket,toType,toSocket)`, `wireBroken`, `nodeCount`,
`paramEquals(type,param,value)`, `compiles` (runs `inferGraph`), `trainingStarted`, `running`,
`quickMenuOpened`, `wireSniped`.

The page drives the engine from **one place**: `afterGraphMutation()`, mode changes, worker messages and
interaction callbacks all call `missions.check(ctx)`, so a step advances the instant the real graph satisfies
its predicate. While a mission is active, `state.missionSuppressAssist = true` disables the drop-time
auto-wiring so the construction is genuinely performed by the learner (the compiler and Fix automatically stay
fully real).

**Overlay:** `StepByStepOverlay` (same `TutorialOverlay` spirit) renders a mission header with progress, the
spotlight, a pulsing target ring, a dropzone animation on canvas targets, and an explanation card with
Plain / Analogy / Technical layers from the catalog. Next is disabled until the gate is met; Back, Skip step,
and Exit mission are always available; its own `×` and `MenuManager` registration give it the universal
Escape/outside/× contract.

### 34. Explanations catalog schema

```
{ key: {
  title, category, friendlyLabel, aliases[],
  plain,          // jargon-free explanation
  analogy,        // everyday metaphor (ReLU = light switch, …)
  technical,      // exact engine behavior (max(0,x), pairing rules, shapes)
  signals,        // what the moving particles / neuron glow represent for this block
  tips[]          // practical wiring advice
} }
```

Coverage: every supported block type, every activation, all three losses, all optimizers, plus the concepts
`wiring`, `forward_pass`, `backprop`, `loss`, `training`, `inference`, `quick_add`, `snip`. Helpers:
`explain(key)`, `friendlyLabel(type)`, `technicalLine(type)`, `searchTerms(entry)` — all pure and Node-safe.

### 35. Three-tier interaction policy (`interactive_modes.js`)

| Tier | Mode | Wire menu | Quick-add menu | Missions |
|---|---|---|---|---|
| 1 — Playground | `playground` (+ missions with beginner vocabulary) | “Cut this wire?” — friendly endpoints, no socket jargon | curated ≤ 8 blocks, friendly group names (“Make it think”, “Turn signals on/off”), no scores | `build_first_brain` |
| 2 — Guided | `assist` | endpoints + socket kinds; assists repair after reconnect | all compatible blocks, category groups, compile-clean badge, shape hints | `build_first_brain`, `build_digit_classifier` |
| 3 — Advanced | `manual` | raw sockets + real post-break topology warning | all blocks + socket kinds, shapes, param summaries, error codes | `build_custom_pipeline` |

`activeTier(ctx) = mission.tier ?? modeToTier(mode)`; the policy is a pure `{wireModel, quickModel}` decorator
consumed by both DOM menus and the test suites.

### 36. Page integration points (`canvas_studio.js`)

* `bindCanvasEvents`: contextmenu → `WireInteractions.openFromEvent`; pointerup with an unconsumed `wireDrag` →
  `QuickNodeMenu.open`; pointermove adds `.is-wire-drop` while over empty canvas.
* `renderWires`: draws `pendingDrop` as `.ns-wire-pending` while the quick menu is open.
* `renderNodes`: adds `data-type` to node elements so missions can target `[data-type="dense"]`.
* `afterGraphMutation`, `attemptConnect`, `deleteNode`, `setMode`, `startRun`, worker `TRAINING_COMPLETE`:
  call `missions.check(ctx)`.
* Tours menu: new **Build missions** section with `data-mission` buttons.
* `MenuManager`: registers `wire-menu`, `quick-node-menu` popovers and the `mission-overlay`.
* Undo: breaking a wire is one undo entry, like every other mutation.

### 37. Verification plan (Phase 3)

1. `tests/test_wire_interactions.js` — real `GraphState` + geometry: raycast finds the correct wire on straight
   and curved wires, respects radius/scale, misses cleanly; break mutates the graph and leaves no dangling
   endpoint; animator state machine; tier wire-menu copy.
2. `tests/test_quick_node_menu.js` — candidate lists per socket kind/direction on real presets; scoring uses
   real compiler errors; `applyQuickAdd` connects (including assist Flatten insertion and in-drag direction);
   rollback on refusal; search/group/keyboard model; popover DOM via `tests/fake_dom.js`.
3. `tests/test_step_by_step.js` — runner gate enforcement (wrong event does not advance), predicate gates read
   real graphs, every mission is proven solvable by applying the exact graph mutations each step describes and
   asserting `evaluateGate` passes; explanation coverage for every palette block.
4. WebView2 smoke additions — right-click a real preset wire and break it; drag a socket to empty space, pick a
   block, verify the new wire; start a mission and advance it with the real gestures.

---

## Part VI — Implementation record & measured results

Built exactly to the Part V design. The new modules are `src/canvas/wire_interactions.js`,
`src/canvas/quick_node_menu.js`, `src/tutorial/step_by_step_engine.js`, `src/tutorial/explanations_catalog.js`
and `src/modes/interactive_modes.js`; `src/ui/canvas_studio.js`, `src/graph_studio.html` and
`src/css/node_studio.css` carry the page wiring and styling. The graph engine, compiler and worker are unchanged.

### 38. What each subsystem really does

* **Wire raycasting** samples the real cubic Bézier into a 29-point polyline and returns the nearest wire within
  `clamp(10/scale, 5, 22)` world units; the socket hit test keeps priority. The right-click context menu opens as a
  `MenuManager` popover; **Break connection** runs a 180 ms `.is-breaking` collapse and only then performs one real
  `GraphState.disconnect` + re-inference, as a single undo entry.
* **Quick node creation**: a wire released over empty canvas opens a searchable popover anchored at the cursor. Every
  candidate is ranked by cloning the live graph, adding the block, connecting through `assistConnect` (assist) or
  `connect` (manual) and re-running the real `inferGraph` — `applies` literally means “the engine compiles it”.
  Picking a row places the block at the release point, wires it, and reports any repair actions (e.g.
  `A_INSERT_FLATTEN`) in one gesture; refused picks roll the node back out of the graph.
* **Tier policy**: tier 1 (playground/missions-for-beginners) shows curated friendly groups (≤ 8 rows) with no raw
  errors while search still reaches every compatible block; tier 2 shows all blocks by category with clean-compile
  badges; tier 3 adds socket kinds, shapes, param summaries and the real post-break consequence. The advanced wire
  menu pre-computes the cut on a clone and reports the actual `compile` delta (“This cut breaks the build:
  `E_MISSING_CONFIG`.” / “the network shrinks from 5 to 4 compiled layer(s).” / “still compiles”).
* **Missions** run from a blank canvas (`prepare_mission` replaces the graph, resets weights/history and re-renders).
  Every step is gated by a live predicate (`hasNode`, `wireExists`, `wireBroken`, `pathExists`, `socketWired`,
  `feedingParam`, `compiles`, `running`, …) evaluated after every graph mutation; a step with a `validate` predicate
  treats the graph itself as ground truth, and `next()` refuses while the gate is unmet. Snip steps require both the
  specific cut **and** the survival of the rest of the chain, so cutting the wrong wire cannot pass. Explanations come
  from `explanations_catalog.js` (plain language, analogy, technical detail, friendly labels used by tier-1 menus).

### 39. Real defects found by this verification and fixed

1. **The advanced topology preview disconnected nothing.** `GraphState.clone()` round-trips through JSON and
   `fromJSON` mints fresh connection ids, so the preview was calling `disconnect` with the original id on the clone
   and always reported “valid”. It now resolves the cloned wire by endpoints (id first, endpoints as the fallback)
   and reports real `compile` deltas, including layer-count shrinkage and warnings.
2. **JS-created popovers were never mounted.** `MenuManager` owns lifecycle (show/hide/Escape/outside-click), not
   DOM mounting; the quick-add and wire menus built their panels but never appended them, so in the browser the
   popovers could never appear (the Node tests accessed `menu.panel` directly and missed it). Both menus now append
   their panel to `document.body` in `_build()`; `tests/fake_dom.js` provides a `body`, and both suites assert the
   panel is mounted and reachable through `document.querySelector`.
3. **Overlay Next lost the live context.** Button-driven `next()` re-evaluated the gate with an empty context and
   refused after any click. `StepByStepEngine` now remembers the last live page context (`_ctx`) and reuses it for
   every button-driven `next`/`prev`/`skip`.

### 40. Measured results (2026-10-19)

| Check | Result |
|---|---|
| `tests/test_wire_interactions.js` | **71/71** — polyline/raycast math, radius scaling, nearest-wire priority, animator state machine, popover lifecycle, animated end-to-end break with real disconnect, graceful missing-path fallback |
| `tests/test_quick_node_menu.js` | **85/85** — socket filtering per kind/direction, unsupported blocks excluded, real compiler scoring (assist vs manual rank delta incl. `E_RANK_MISMATCH` reasons), assist Flatten insertion, in-drag direction, `E_NO_SOCKET`/`E_DUP_INPUT` rollback, tier models, search beyond the curated cap, popover DOM + keyboard + × |
| `tests/test_step_by_step.js` | **90/90** — curriculum integrity (every concept/target/rule), all validators on real graphs, runner gate enforcement, **full solvability simulation of all three missions with the real GraphState** (every gate flips false→true from the action; the final graphs compile: XOR 2→8→relu→2→sigmoid→bce with Adam + epoch loop; MNIST 28×28→flatten→dense→relu→dense(2)→sigmoid→bce with a real snip + socket-to-socket rewire; regression flatten→dense→relu→dense(1)→linear→mse with RMSprop + LR schedule + monitor tap), overlay target/explanation/gated Next/Escape/completion |
| `npm test` | **18/18 JS suites, exit 0** |
| `npm run test:python` | **22/22 tests OK** |
| `python tools/smoke_node_studio.py` | **73/73 checks, exit 0** in real WebView2 |

Browser-measured flows: dropping a dragged wire on empty canvas opened the popover with **12 compatible blocks**;
picking “Dense Layer” produced 14 → 15 blocks / 13 → 14 wires with the log line *“Placed Dense Layer and wired it in
one gesture.”*; right-clicking the new wire opened the menu (`Dense Layer · out (tensor) → Dense Layer · in
(tensor)`) and **Break connection** removed exactly that wire (14 → 13) with *“Broke Dense Layer.out → Dense
Layer.in.”* in the log; starting mission 1 blanked the canvas (0 blocks), refused Next on the source step until the
real Source block was placed (hint flipped to verified, Next enabled), and Escape exited the overlay.

### 41. Packaging note

`dist/web`, the portable app folder and the public ZIP are rebuilt from the final source after this record (see README
§Quickstart for the commands). The quick menu, wire menu and mission overlay ship in both targets.

### 42. Packaged-host boot failure and fix (2026-10-20)

**Report.** Launching `dist/NeuralNetworkStudio.exe` and opening Node Studio left the page frozen on the static
“Booting…” status.

**Diagnosis.** The exe was re-launched with WebView2 remote debugging
(`WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9333`) and the real page was navigated from
`index.html` to `graph_studio.html`. The browser console logged `net::ERR_CONNECTION_REFUSED` for
`ui/action_bar.js`, `tutorial/step_by_step_engine.js` and `modes/absolute_beginner.js`; the status text stayed
`Booting…` with 0 palette blocks. The packaged app serves `src/` from `desktop/desktop_app.py` with a
**single-threaded** `HTTPServer` (default listen backlog 5). Chromium opens its ES-module graph over many parallel
connections; while one response holds the accept loop (the 2 MB MNIST payload, the 83 KB `canvas_studio.js`), the
extra connections overflow the backlog and Windows refuses them. Module graphs fail atomically, so the boot code
never runs.

**Reproduction.** A 3 s response plus 12 parallel GETs against the same class: 7 refused with `HTTPServer`,
0 refused with `ThreadingHTTPServer`.

**Fix.**
- `desktop/desktop_app.py`: new `StudioHTTPServer(ThreadingHTTPServer)` (`daemon_threads`, `request_queue_size = 64`);
  `StudioHTTPHandler.handle_error` no longer depends on console streams (`console=False` builds have none), quietly
  ignores routine connection aborts and appends other failures to `%TEMP%\neural_network_studio_server.log`; the
  `/api/data/*` route answers a JSON 500 instead of dropping the connection.
- `server.py` and `tools/smoke_node_studio.py`: same threaded server class, so the dev host and the smoke host match
  the shipped host.

**Regression coverage.**
- `tests/test_desktop_host.py` (4 tests, now part of `npm run test:python`): threaded class + deep backlog, correct
  HTML/JS MIME, a stalled response with a 30+ module parallel burst (0 failures; the old host refused 7+), and a
  windowed `stdout`/`stderr=None` abort that must not take the server down.
- `tools/smoke_desktop_boot.py` (`npm run smoke:desktop`): serves `src/` through the real desktop host class, opens
  `graph_studio.html` in WebView2 and requires the status to reach “Ready — build a graph or load a preset.” with
  ≥ 20 rendered palette blocks and no failed resources.
- `tools/smoke_packaged_exe.mjs` (`npm run smoke:packaged`): launches the built `.exe` itself with
  `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=<free>`, waits for the desktop host's
  direct `graph_studio.html` page, and requires Ready + ≥ 20 palette blocks + zero failed resources before
  terminating the process tree.

**Verification after rebuild.** The rebuilt exe was probed again through the same CDP path: status
**“Ready — build a graph or load a preset.”**, **27** palette items, **0** failed requests and 0 console errors;
the desktop-host boot smoke passed; the full 98-check browser smoke passed; `npm test` exited 0 and
`npm run test:python` passed 27/27. `dist/web` matches all 47 `src` files by SHA-256.

### 43. Tutorial/mission overlay overlap, stale spotlights and stacking (2026-10-20)

**Report.** In the guided experience the mission card covered the palette and sometimes the very control the
learner was asked to press, so a mission could not be completed. Moving a node left the spotlight/ring at the old
screen position. Starting a tour and a mission together stacked two dim layers, and the title-bar menus could
render underneath an active guide.

**Diagnosis.**
- `StepByStepEngine._position()` ran only on render and `resize`. `renderNodes()` rebuilds every node element on a
  drag, so the presenter's cached `targetEl` became detached and its rectangle stale. The same applied to canvas
  pan/zoom and panel scrolling.
- The card was a fixed 384 px panel with an assumed 460 px height; when neither side fit it fell back to a position
  over the target. The tour tooltip made the same assumption at 380×220 and never reset its center transform.
- A palette/inspector target was treated as just the small button/field. Placing beside it still sliced through the
  surrounding panel because there was no notion of the app chrome.
- `.ns-menu-cursor` was `z-index: 60` (below mission 70 and tour 100), and `.ns-menu` was `40`; a newly opened menu
  was dimmed and could be covered by a guide. The Tours menu's mission entry was therefore not reliably reachable.
- Tour and mission startup had no mutual exclusion; both overlay roots could be mounted at once.

**Fix.**
- New `src/tutorial/overlay_layout.js`: pure, tested geometry. It measures the real panel, tries right/left/below/
  above, clamps to the viewport, ranks every target-clearing placement above every placement that clips the target
  (with a 10 px halo), then minimizes overlap with the fixed chrome (`.ns-topbar`, `.ns-palette`, `.ns-right`,
  `.ns-statusbar`), and treats `#studio-canvas` / full-workspace targets as backdrops that the card may sit over.
- `StepByStepEngine`: public `refreshPosition()`; re-resolves the target and updates spotlight/ring/card on every
  refresh; a socket target expands its avoidance rectangle to the whole `.ns-node`; palette/inspector/topbar targets
  expand to the whole panel; the card measures its rendered height, is capped by the viewport and scrolls internally;
  it stays put while still valid and re-places only when the target, size or viewport makes that necessary; captured
  scroll, resize and the technical-detail toggle all re-anchor it.
- `TutorialOverlay`: same measured placement and panel-aware avoidance; `refreshPosition()`; layout is separate from
  rendering so a refresh never rebuilds the buttons/focus; center transform is reset when a target appears; the
  tooltip is viewport-bounded and scrollable; captured scroll is tracked.
- `src/ui/canvas_studio.js`: `refreshGuideOverlays()` is called from `renderNodes()`, `applyView()`,
  `renderInspector()` (both paths) and `setMode()`. Starting a mission silently closes an active tour; starting a
  tour/track closes an active mission. `.ns-menu` and `.ns-menu-cursor` now use `z-index: 120`.
- `src/css/node_studio.css`: `.mission-card` and `.tour-tooltip` get `max-height: calc(100vh - …)`, `overflow-y:
  auto`, and `overscroll-behavior: contain`.

**Verification (measured).**
- `npm run test:missions`: **106/106** — the new geometry section checks palette/inspector anchoring, backdrop
  placement, exact intersection maths, moving-target spotlight follow, card dodging, captured-scroll re-anchoring,
  a taller card clamped on-screen, resize re-placement and listener cleanup.
- `npm run test:ui`: all static checks pass, including the viewport bounds, scroll rules, guide refresh hooks,
  mutual exclusion and both menu z-index contracts.
- `python tools/smoke_node_studio.py`: **98/98**, exit 0, in real WebView2. It now walks mission 1 to step 3, changes
  the real dataset to XOR, reaches step 4, drags the source node with real pointer events and asserts the socket,
  spotlight and ring move together while the card dodges the moved block; it also asserts the card/tooltip are fully
  in-window and never cover the highlighted palette/inspector/topbar target, and that switching from an active tour
  to a mission leaves exactly one guide mounted.
- `npm run smoke:packaged`: **PASS** against the rebuilt portable launcher — the host now opens
  `graph_studio.html` directly and reports **“Ready — build a graph or load a preset.”**, **27** palette items,
  **0** failed resources.
- `npm test`: exit 0, 18/18 JS suites plus the deterministic example certificate; `npm run test:python`: **27/27**
  (22 reference + 5 desktop-host). `dist/web` matches all **47** `src` files by SHA-256; the public ZIP is
  **13.66 MB** and the portable app folder is **28.77 MB**.

### 44. Ten-lesson curriculum, persistent action journal and guide polish (2026-10-21)

**Report.** Lesson 1 asked the learner to press **Run**, the run finished in under a second, and the card asked
for **Run** again. The copy also used metaphors instead of real block names, the curriculum was split across
three menus with only nine entries, the Playground sat in the node-mode tab strip, and the top bar felt cramped.

**Root cause.** `TutorialRunner`/`MissionRunner` re-evaluated the live `state.running` flag when the learner pressed
**Next**. A completed run legitimately sets `running = false`, so the action that had already happened could no
longer satisfy the gate.

**Fix.**
- `src/ui/canvas_studio.js`: every real gesture is appended to a page-level journal (`state.actions`) with
  `{ id, seq, at, detail }` (300-entry cap). `recordAction()` notifies both the tour and mission runners; the
  training-start, prediction, export, preset, mode, parameter, wiring, Playground and playback events all use it.
- `src/tutorial/step_by_step_engine.js`: `VALIDATORS.actionSeen` and the run steps (`fb-run`, `db-run`, `cp-run`)
  read that journal (and `MissionRunner`'s own event counts), not `state.running`. The old `running` validator stays
  for compatibility.
- `src/ui/tutorial_engine.js`: `TutorialRunner` keeps the journal reference, latches satisfied steps and exposes
  `isSatisfied()`. `TutorialOverlay.start()` forwards the page journal, `render()` enables **Next** for an action
  already in the log, and `advance()` accepts it. The same gate holds when a run ends before **Next** is pressed.
- The **Lessons** menu is now exactly ten numbered entries. Lessons 1/3/5 are curriculum tracks, 2/4/6 are deep
  dives, 7/8/9 are build missions, 10 is the new export/deploy lesson. All ten use the same step card, progress
  indicator and real gesture gates.
- Copy now names the actual registry blocks (`Dataset Source`, `Flatten`, `Dense Layer`, `ReLU`, `Sigmoid`,
  `Binary Cross-Entropy`, `Adam`, `Epoch Loop`, `SQL Query`, `LR Scheduler`, …). Metaphors remain as supporting
  analogies. `friendlyLabel()`/`explain()` return the registry label, so the beginner quick menu and wire menu use
  the same names as the canvas.
- The Playground is no longer a node-mode tab. `#btn-mode-playground` lives in the action bar, remains the target
  of Lesson 1 and still switches to the real Playground view. `Beginner · Assist` and `Advanced · Manual` are the
  only node-mode tabs.
- Lesson copy renders through `src/tutorial/inline_text.js`: HTML is escaped first, then `**bold**` and `` `code` ``
  become real emphasis. The mission card no longer shows literal `**Next**`, and the tour card no longer shows
  literal `**MNIST preset**`; the smoke asserts the rendered `<strong>` and the absence of `**`.
- The empty-canvas invitation now lives on the viewport (`#studio-canvas.is-empty::after`) instead of the
  transformed node layer, so it stays centered and readable at any pan/zoom.
- Visual polish: compact top bar, Graph dropdown for save/load/reset/undo, numbered Lessons menu with subtitle/tag
  rows, progress bars, lesson-title line, readable hint boxes, stronger mission ring and a compact tour card.
- The real-browser smoke now starts Lesson 2 on a fresh journal, loads the MNIST preset, swaps to a fast XOR corpus,
  starts a real run, goes **Back** to the Run step, and asserts **Next** is enabled and advances to the metrics step.
  This is the regression test for the reported replay bug.

**Verification (measured).**
- `npm test`: exit 0; graph 84/84, nodes 20/20, DB 14/14, exports 19/19, ONNX 48/48, all DOM checks, menu 28/28,
  drag 209/209, Playground 34/34, wire 71/71, quick-menu 85/85, missions 111/111, examples certificate PASS.
- `npm run test:python`: 27/27 (22 reference + 5 desktop-host).
- `python tools/smoke_node_studio.py`: **108/108**, exit 0 in real WebView2. The run-gate replay check passes:
  the Run step shows **Next enabled** and the hint is met after the run has stopped, then **Next** reaches
  **Read the live metrics**. The same smoke asserts both the tour and mission copy contain `<strong>` and no
  literal `**`.
- The Lessons menu is asserted to contain exactly ten entries numbered `01`–`10`; the Playground is asserted absent
  from `.ns-mode` but present in `.ns-actions`; the first lesson opens its Playground path.
- `dist/web` is synced from `src`; the public artifact is
  `dist/NeuralNetworkStudio-Windows-x64.zip`, built from the portable folder with its shipped `README.txt`.
  A Microsoft Defender custom scan of the extracted folder reports **no threats**, and
  `dist/NeuralNetworkStudio-Windows-x64.sha256.txt` publishes the ZIP checksum.
- `npm run smoke:packaged` boots the real WebView2 host from the extracted ZIP directly into
  `graph_studio.html` and reaches **Ready — build a graph or load a preset.** with 27 palette blocks and zero
  failed resources.
- The one-file launcher is developer-only and is not part of the distributable. It is the artifact Windows
  Defender falsely flags as `Trojan:Win32/Bearfoos.A!ml`; the one-folder build does not trigger that detection.

### 45. Node Studio is the only workspace; public repo and release workflow (2026-10-21)

**Request.** Remove the classic landing page and its flashy visualizer/dashboard, boot directly into Node
Studio, and publish the project to a new public GitHub repository with proper releases and one-version-behind
backups.

**Removed from the shipped app.**
- `src/css/style.css`, `src/css/components.css`.
- `src/ui/app.js`, `architecture_builder.js`, `prediction_panel.js`, `canvas_feedback.js`.
- `src/visualizer/` (network renderer, particle engine, decision boundary, metrics chart, audio synthesizer).
- The stale `src/static/` duplicate and the internal `CHAT.md` / `architecture_blueprint.md` working notes.
- The back-to-classic arrow in the Node Studio top bar.

**What remains.**
- `src/graph_studio.html` and `src/css/node_studio.css` are the only application page and theme.
- `src/index.html` is a minimal redirect shim (`location.replace('graph_studio.html')`) so static hosts still
  open the right page from `/`.
- `desktop/desktop_app.py` opens `graph_studio.html` directly and redirects `/` and `/index.html` to it.
- `server.py` does the same, and the browser batch file opens the Node Studio URL.
- Shared engine, worker, data, export and Playground modules remain untouched.

**Verification.**
- `tests/test_desktop_host.py` now has 5 tests: the new one asserts both `/` and `/index.html` resolve to
  `graph_studio.html` and contain the real status element.
- `tools/smoke_packaged_exe.mjs` waits directly for `graph_studio.html` instead of navigating from a classic
  landing page.
- `npm test`: exit 0 (graph 84/84, nodes 20/20, DB 14/14, exports 19/19, ONNX 48/48, DOM all-pass, menu 28/28,
  drag 209/209, Playground 34/34, wire 71/71, quick-menu 85/85, missions 111/111, example certificate PASS).
- `npm run test:python`: 27/27 (22 reference + 5 desktop-host).
- `python tools/smoke_node_studio.py`: 108/108 in real WebView2.
- `npm run smoke:packaged`: PASS on the rebuilt portable launcher, direct `graph_studio.html` ready state,
  27 palette blocks, zero failed resources.
- Microsoft Defender custom scans of the ZIP and extracted folder: no threats.
- `src` and `dist/web`: 47 files each, 0 SHA-256 mismatches.
- Public ZIP: 13.66 MB, SHA-256 `BEE2DBF5FE500E57678D62756EC2FAE7B0065CA72EFC14C87AA270966C8594A6`.

**Repository.** `https://github.com/GaiDevOfficial/neural-network-studio` (public). The release workflow is
`tools/push_with_backup.ps1`: tests, portable build, packaged smoke, commit, semantic tag, `backup` branch at
the previous release, local `backups/` ZIP, and a GitHub release with the ZIP and checksum attached. Release
notes live in `release-notes/`. Licensing is AGPL-3.0-or-later with a separate trademark policy and commercial
license path.