# Node Studio - The Full Tutorial

Node Studio is the visual neural-network builder in Neural Network Studio. You wire a model from real
blocks on an infinite canvas, then train it through the same engine, worker and datasets used everywhere
else in the app. Every loss, accuracy, confidence, gradient and pulse comes from a real forward or
backward pass. Nothing is mocked.

> The classic forms-and-charts landing page has been retired. The desktop launcher opens Node Studio
> directly, and `src/index.html` is now only a redirect shim for static hosts.

---

## 1. Get Node Studio running

### Option A - Portable Windows app (easiest)

The public build is `dist/NeuralNetworkStudio-Windows-x64.zip`. Extract it, keep the
`NeuralNetworkStudio` folder together, and run:

```powershell
.\dist\NeuralNetworkStudio\NeuralNetworkStudio.exe
```

No Python, Node.js or other runtime is required. The ZIP checksum is in
`dist/NeuralNetworkStudio-Windows-x64.sha256.txt`.

### Option B - Local browser

```powershell
python server.py
```

Open <http://127.0.0.1:8080/graph_studio.html>. The root URL and the old `/index.html` address both
redirect there. Always use the local server; browsers block the MNIST `fetch()` payload on `file://`.

### Option C - Static hosting

`dist/web/` is a complete static bundle. Deploy that folder to Netlify, Vercel or GitHub Pages. Built-in
datasets work without a backend; SQLite/CSV-path/REST blocks need the local Python Data API.

---

## 2. The three tiers

| Tier | Button | What it is for |
|---|---|---|
| Absolute Beginner Playground | **Playground** | A pre-built MNIST 0-vs-1 brain. Draw, ask, and correct it without touching blocks. |
| Smart Node Assist | **Beginner - Assist** | Drop blocks and let the editor suggest, auto-wire and repair legal connections. |
| Full Manual | **Advanced - Manual** | Raw sockets, strict validation, no automatic wiring. |

All three use the same graph compiler, worker RPCs and datasets. Changing tier never changes the model
that is actually trained.

---

## 3. Playground: your first real classifier

1. Press **Playground** in the toolbar.
2. Wait for the brain state to report the real MNIST 0-vs-1 corpus.
3. Press **Teach** and let it train. The progress and accuracy values are measured on held-out digits.
4. Draw a **0** or a **1** in the pad.
5. Press **What is it?**. The answer and confidence bars come from a real forward pass.
6. Press **That's right** if the answer is correct.
7. Press **Nope - it's a ...**, choose the correct digit, and watch one bounded online-backprop step update
   the exact drawing. The loss trace is measured before and after that step.
8. Press **Ask** again to see the corrected prediction.

The Playground is not a toy animation: it owns a real 784 -> 16 -> 8 -> 2 network backed by the shipped
MNIST payload.

---

## 4. Build a classifier with blocks

1. Press **MNIST** (or **XOR** / **Iris**) in the toolbar. A preset graph is laid out with real
   coordinates and compiles immediately.
2. Inspect the blocks:
   - **MNIST Images** or **XOR Data** - the data source.
   - **Flatten** - turns 28x28 images into 784 values when needed.
   - **Dense Layer** - a real weight matrix and bias vector.
   - **ReLU** / **Sigmoid** / **Softmax** - activation blocks.
   - **Cross Entropy** or **Mean Squared Error** - the loss block.
   - **Adam** / **SGD** / **RMSprop** - the optimizer block.
   - **Training Loop** - connects loss and optimizer to the compiled model.
   - **Accuracy** / **Loss Gauge** - live output monitors.
3. Press **Run**. The status bar, loss curve, held-out accuracy, confusion matrix and sparkline all update
   from real worker messages.
4. Use **Pause** to inspect the current epoch, or **Reset** to return to fresh weights.
5. Press **Predict** and draw a digit (image models) or move the feature sliders (tabular models). The
   prediction probe uses the live trained weights.

### Reading the validation panel

The inspector explains every block in plain language, with an analogy and an optional technical note.
The validation panel reports real compiler errors such as `E_RANK_MISMATCH`, `E_PAIRING` or
`E_MISSING_CONFIG`. In **Beginner - Assist**, press **Fix automatically** to let Smart Assist repair the
graph with legal blocks and wires.

---

## 5. Manual wiring

1. Switch to **Advanced - Manual**.
2. Drag a block from the palette onto the canvas. The **Place Node** popover opens at the cursor if you
   invoke it from a socket or empty canvas.
3. Drag from an output socket to an input socket. Compatible sockets highlight; invalid connections are
   refused with the real reason.
4. Drag a socket into empty space to open the searchable quick-add menu. It lists only blocks compatible
   with that socket, scored by actually compiling the candidate into a clone of your graph.
5. Right-click any wire and choose **Break connection**. The wire animates away, both sockets detach, the
   graph recompiles and the action lands as one undo entry.
6. Drag a block over the palette to delete it. A red **Drop to delete** veil appears; releasing severs
   every attached wire and clears the block's cached data.
7. Press **Ctrl+Z** or **Undo** to restore the last graph action.

---

## 6. Data sources

| Source | How to use it |
|---|---|
| Built-in | XOR, Concentric Circles, Two Moons, Two Spirals, Iris, Regression |
| Real MNIST | The shipped 0-vs-1 corpus (2500 train / 800 held-out in the web payload) |
| CSV upload | Drop a CSV, choose the target column and let the parser detect the delimiter/header |
| SQLite | Use a **SQLite Query** block; the local Data API runs `desktop/studio_data_api.py` |
| CSV path | Use a **CSV File** block with a local path |
| REST / JSON | Use a **REST Endpoint** block; the page fetches `{columns, rows}` |

A **Class Filter** block can remap one-hot targets for a subset of classes.

---

## 7. Exports

Open the **Export** menu. Every artifact is generated from the compiled, trained graph:

- **Graph JSON** - save and reload the block layout, parameters and wires.
- **Model JSON** - architecture plus learned weights.
- **Python (NumPy)** - a standalone inference script.
- **JavaScript ES module** - browser/Node inference code.
- **NumPy `.npz`** - weight archives.
- **ONNX** - a dependency-free inference graph (opset 13, IR 8), verified against `onnxruntime`.
- **Commands** - the exact local build/run commands.

Exports are verified numerically against the engine in the test suite; they are not templates.

---

## 8. Lessons and missions

Open **Lessons** for ten numbered step-by-step guides:

1. Playground
2. Preset classifier
3. Block building
4. Backpropagation
5. Manual wiring
6. SQL streaming
7. Build-from-scratch mission (Assist)
8. Build-from-scratch mission (Manual)
9. Regression mission
10. Export and deploy

The three build missions start from a blank canvas (13, 14 and 18 steps). A step advances only after the
real action: placing the exact block, making the wire, cutting it, re-wiring socket-to-socket, setting a
parameter, or completing a real training run. The pulsing target follows the node through drag, pan, zoom
and scroll; the card stays inside the window and never covers the target.

---

## 9. Honest deferrals

- **Conv2D**, **MaxPool** and **Dropout** appear in the palette as *Planned*, visibly disabled. The
  shipped engine is a Dense-stack engine; those blocks are never silently faked.
- **Parquet** and folder loaders are not implemented.
- SQLite, CSV-path and REST blocks need the local Python Data API; they are unavailable on a plain static
  host.

---

## 10. Verification

```powershell
npm test                  # all JavaScript suites + example-projects certificate
npm run test:python       # reference engine, active learning, data connectors, desktop host
python tools/smoke_node_studio.py   # 108-check real-WebView2 UI smoke
npm run smoke:desktop     # desktop host boots graph_studio.html to Ready
npm run smoke:packaged    # packaged launcher reaches Ready through WebView2 CDP
```

If Node Studio stays on **Booting...**, make sure you opened it through `server.py` or the packaged app,
not by double-clicking the HTML file. The browser must be allowed to fetch the dataset and ES modules
over HTTP.

---

## 11. Where to go next

- `example_projects.md` - ten buildable projects with measured certificates.
- `NODE_STUDIO.md` - architecture, graph compiler, worker protocol and packaging record.
- `README.md` - feature overview, building and distribution notes.
