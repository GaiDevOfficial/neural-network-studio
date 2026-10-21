# 🎓 Example Projects — Neural Network Studio

Ten ready-to-build projects for the Node Studio, from a first digit inspector to an advanced capstone.
Six of them ship with a small starter CSV in [`projects/`](projects/); the other four use the
built-in datasets.

Every number in this document is **measured by the real engine** — the same `DatasetManager` parser,
`NeuralNetwork` forward/backward passes and trainer that the studio runs. Nothing is simulated, and
the whole certificate is reproducible bit-for-bit:

```bash
node tools/verify_example_projects.mjs
```

> The studio draws fresh random initial weights on every **Run** (that is the honest product
> behavior), so your live numbers will wobble slightly around the certificate below. The certificate
> itself pins one deterministic seed per project so `tools/verify_example_projects.mjs` produces
> **byte-identical output on every run** — that was verified by running it twice in a row.
> The verification protocol: ten seeded 80/20 train/test splits per project, train with the recipe
> below, then score `network.evaluate()` / `predict()` on the held-out rows only.

---

## The certificate (measured)

| # | Project | Data | Setup | Measured held-out result |
|---|---|---|---|---|
| 1 | Digit QA Inspector | real MNIST 0-vs-1, 2500/800 payload | 784→16→8→2, Adam lr 0.05, batch 32, 10 epochs | **100%** on 800 images (train loss 0.0202) |
| 2 | Flower ID Field Guide | Iris, built-in (150 rows) | 4→16→8→3, softmax+CCE, Adam lr 0.03, batch 8, 600 epochs | median **100%**, 7/10 splits perfect, worst 91.7%¹ |
| 3 | Spam Screener | `projects/spam_screener.csv` (32 rows, 2 features) | 2→8→6→2, sigmoid+BCE, Adam lr 0.03, batch 8, 200 epochs | median **100%**, 9/10 splits ≥95%, worst 85.7%²; 95% by epoch 4–5 |
| 4 | Student Early-Warning | `projects/student_pass_fail.csv` (32 rows, 3 features) | 3→32→16→2, sigmoid+BCE, Adam lr 0.03, batch 8, 1000 epochs | median **100%**, 8/10 splits ≥95%, worst 85.7%²; 95% by epoch 4 |
| 5 | Home Energy Forecaster | `projects/energy_forecast.csv` (32 rows, 3 features, regression) | 3→16→16→1, linear+MSE, Adam lr 0.05, batch 8, 800 epochs | MAE median **0.158 kWh** (worst 0.35) on 3.88–7.77 kWh targets |
| 6 | Crop Yield Responder | `projects/crop_yield.csv` (32 rows, 4 features, regression) | 4→16→16→1, linear+MSE, Adam lr 0.05, batch 8, 800 epochs | MAE median **0.196 t** (worst 0.327) on 2.75–5.26 t targets |
| 7 | Shape Detective (depth lab) | Two Moons, built-in — see both configs | **A:** 2→2→2 · **B:** 2→16→16→2, sigmoid+BCE, Adam lr 0.05, batch 16, 300 epochs | **A (1 hidden neuron):** median 87.5%, crosses 95% in only 2/10 splits — it *struggles*. **B (deep):** **100% on all 10 splits**, 95% by epoch 3 |
| 8 | Workout Intensity Zones | `projects/workout_zones.csv` (30 rows, 3 features, 3 classes) | 3→10→6→3, softmax+CCE, Adam lr 0.03, batch 8, 200 epochs | **100% on all 10 splits**; 95% by epoch 4–5 |
| 9 | Mini OCR Lab | Template Digits 8×8, built-in (4 classes) | 64→16→4, softmax+CCE, Adam lr 0.03, batch 16, 200 epochs | **100% on all 10 splits**; threshold by epoch 1 |
| 10 | "Should I Bother?" capstone | `projects/bike_decision.csv` (30 rows, 3 features) | 3→32→16→2, sigmoid+BCE, Adam lr 0.03, batch 8, 1000 epochs | median **100%**, 8/10 splits ≥95%; 95% by epoch 2 |

¹ The classic Iris corpus contains one flower that sits between *versicolor* and *virginica*; on the
three splits where it lands in the test set, it is the single miss (12/13 = 92.3%).
² Starter CSVs have 30–32 rows, so a test split is 6–7 rows — one borderline row moves the
percentage a lot. Median across splits is the honest headline.

---

## How to use this pack

**CSV projects (3, 4, 5, 6, 8, 10).** Drop the CSV from `projects/` onto the Node Studio canvas.
The Dataset Source switches to **Uploaded CSV** and parses it for real: delimiter detection, the
last column becomes the target (numeric values → regression, a few repeated labels → classification
with automatic one-hot), features are min–max normalized. Then place the blocks from the recipe and
press **Run**.

**Built-in projects (1, 2, 7, 9).** Set the Dataset Source's dataset (or press the preset), or use
the Tier-1 playground for the MNIST flow.

Every recipe lists the exact chain in wiring order and the inspector values to set. In **Smart
Assist** you can drop blocks one by one and it wires them for you; **Full Manual** gives raw sockets
and strict validation. If you are new to the canvas, run **Tours → Build missions** first — the
missions teach exactly these moves (place, wire, snip, re-wire, set parameters, run).

---

## 1 · Digit QA Inspector (real MNIST, active learning)

*Story:* you are the QA inspector for a digit-recognition booth. The network is right most of the
time; when it is wrong you teach it on the spot and watch the loss drop.

- **Data:** built-in **MNIST 0 vs 1 (real 28×28 corpus)** — the published handwritten-digits corpus,
  digest-verified, 2500/800 in the shipped web payload.
- **Recipe:** this is the Tier-1 playground preset — Dataset Source → Dense(16)+ReLU →
  Dense(8)+ReLU → Dense(2)+Sigmoid → Binary Cross-Entropy → Adam(lr 0.05) → Epoch Loop(maxEpochs 10,
  batchSize 32).
- **Measured:** 100% held-out on all 800 test images after 10 epochs (final train loss 0.0202).
  In the live browser smoke the same flow scored 99.9% on all 800 images.
- **The active-learning loop to demonstrate:** press **Teach** (trains), then **Draw** a digit,
  press **Ask** (real prediction with probabilities), and when it is wrong press **Nope** and type
  the true digit — one real backpropagation step corrects the weights and re-predicts immediately.
  The test suite records a genuine correction taking the loss 2.20528 → 1.71745 (22.1% lower) in
  that single online step, and the on-page loss trace shows it live.

## 2 · Flower ID Field Guide (Iris)

*Story:* a botanist's pocket guide that identifies the species from four petal/sepal measurements.

- **Data:** built-in **Iris (4 features, 3 classes)**.
- **Recipe:** Source → Dense(16) → ReLU → Dense(8) → ReLU → Dense(3) → Softmax → Categorical
  Cross-Entropy → Adam(lr 0.03) → Epoch Loop(maxEpochs 600, batchSize 8).
- **Measured:** median 100% across the ten splits; 7 of 10 splits scored a perfect 100%; the worst
  split missed only the single ambiguous flower (91.7%). The 95% mark is crossed around epoch 4.
- **Bonus lesson:** open the Class Filter and keep only two species — feel how much easier a binary
  problem is.

## 3 · Spam Screener (starter CSV)

*Story:* a tiny mail-room assistant: two measurable signals — how many links and how many spammy
words — decide ham or spam.

- **Data:** `projects/spam_screener.csv` — 32 rows, columns `link_count, spam_word_hits, label`
  (`ham`/`spam`).
- **Recipe:** Source → Dense(8) → ReLU → Dense(6) → ReLU → Dense(2) → Sigmoid → Binary
  Cross-Entropy → Adam(lr 0.03) → Epoch Loop(maxEpochs 200, batchSize 8).
- **Measured:** median 100%; 9 of 10 splits ≥95% (the tenth missed one of seven test emails);
  95% reached by epoch 4–5.
- **Try it:** set the Loss/Accuracy Gauge after the loss block to watch the real curve, then push
  both inputs to the corners (0 links / 0 hits vs 9 links / 10 hits) in the Prediction View.

## 4 · Student Early-Warning (starter CSV)

*Story:* flag the students who are heading for a fail while there is still time to help.

- **Data:** `projects/student_pass_fail.csv` — 32 rows, columns
  `attendance_pct, study_hours_week, prior_grade, result` (`pass`/`fail`).
- **Recipe:** Source → Dense(32) → ReLU → Dense(16) → ReLU → Dense(2) → Sigmoid → Binary
  Cross-Entropy → Adam(lr 0.03) → Epoch Loop(maxEpochs 1000, batchSize 8).
- **Measured:** median 100%; 8 of 10 splits ≥95% (worst split misplaced one of seven students);
  95% reached by epoch 4.
- **Why the wide first layer:** the decision boundary is a combined rule (attendance *and* study
  time); 32 units give it room to carve it out. Try shrinking the first layer to 8 — measured in the
  same sweep, that variant crossed 95% in only 6 of 10 splits. Real evidence of capacity, measured
  live.

## 5 · Home Energy Forecaster (starter CSV, regression)

*Story:* predict tomorrow's electricity use from the weather and the household schedule. This is the
pack's regression warm-up — the output node is a single number, not a class.

- **Data:** `projects/energy_forecast.csv` — 32 rows, columns
  `outdoor_temp_c, occupancy, hour_of_day, kwh`.
- **Recipe:** Source → Dense(16) → ReLU → Dense(16) → ReLU → Dense(1) → Linear → MSE Loss →
  Adam(lr 0.05) → Epoch Loop(maxEpochs 800, batchSize 8).
- **Measured:** mean absolute error 0.158 kWh (median; worst split 0.35) against a target range of
  3.88–7.77 kWh — about 4% of the range.
- **Upload tip:** with a numeric last column the studio treats the CSV as regression automatically
  (no classes invented); the display shows raw kWh, so the gauge is directly readable.

## 6 · Crop Yield Responder (starter CSV, regression)

*Story:* from rainfall, temperature, soil pH and fertilizer, estimate the harvest — the classic
agronomy curve fit, but learned instead of hand-fitted.

- **Data:** `projects/crop_yield.csv` — 32 rows, columns
  `rainfall_mm, avg_temp_c, soil_ph, fertilizer_kg, yield_t` (target in tonnes per plot).
- **Recipe:** same regression stack as the energy project — Source → Dense(16) → ReLU → Dense(16) →
  ReLU → Dense(1) → Linear → MSE Loss → Adam(lr 0.05) → Epoch Loop(maxEpochs 800, batchSize 8).
- **Measured:** mean absolute error 0.196 t (median; worst 0.327) against 2.75–5.26 t targets —
  roughly 4–8% of the range on plots that yield ~3–5 tonnes.
- **Comparison lab:** swap ReLU for Tanh and keep everything else — then decide with the gauge which
  smoother curve suits the data. (Both train; the numbers differ — that is the point.)

## 7 · Shape Detective (depth lab — the star lesson)

*Story:* two interleaved crescents — the Two Moons dataset. Can a *single* tiny neuron separate
them? Build it, measure it, then give it depth and watch the boundary transform.

- **Data:** built-in **Two Moons (2 features)**.
- **Recipe A (the struggle):** Source → Dense(2) → ReLU → Dense(2) → Sigmoid → Binary
  Cross-Entropy → Adam(lr 0.05) → Epoch Loop(maxEpochs 300, batchSize 16).
- **Recipe B (the fix):** Source → Dense(16) → ReLU → Dense(16) → ReLU → Dense(2) → Sigmoid → same
  loss/optimizer/loop.
- **Measured:** A reached the 95% mark in only 2 of 10 splits (median 87.5%) — it genuinely
  underfits the crescent. B scored **100% on all ten splits**, crossing 95% by epoch 3.
- **Watch this:** run both and compare the live held-out accuracy and loss — A hovers around 88%
  and keeps wobbling between epochs, B locks onto 100% within a few epochs. Then use the
  **Prediction View** sliders (overview panel or block) to probe a handful of points inside each
  crescent for both runs: B answers with stable, confident probabilities, while A flips near the
  middle of the gap. Same data, same loss, only the depth changed.

## 8 · Workout Intensity Zones (starter CSV, 3 classes)

*Story:* your watch measures heart rate, HRV and pace — classify the effort as easy / moderate /
peak. The first multiclass project.

- **Data:** `projects/workout_zones.csv` — 30 rows, columns
  `avg_hr_bpm, hrv_ms, pace_min_per_km, zone` (`easy`/`moderate`/`peak`).
- **Recipe:** Source → Dense(10) → ReLU → Dense(6) → ReLU → Dense(3) → Softmax → Categorical
  Cross-Entropy → Adam(lr 0.03) → Epoch Loop(maxEpochs 200, batchSize 8).
- **Measured:** 100% on all ten splits; 95% reached by epoch 4–5.
- **Why softmax + CCE:** three mutually exclusive zones means the outputs should be a probability
  triple — check the Confusion Matrix and see all three classes stay on the diagonal.

## 9 · Mini OCR Lab (built-in 8×8 digits)

*Story:* the smallest OCR in town: 8×8 grayscale stamps of four digit shapes, classified in a few
hundred epochs.

- **Data:** built-in **Template Digits 8×8 (4 classes)** — note this is the *template* fixture
  (honestly labeled in the palette), while project 1 uses the real photographic MNIST corpus.
- **Recipe:** Source → Dense(16) → ReLU → Dense(4) → Softmax → Categorical Cross-Entropy →
  Adam(lr 0.03) → Epoch Loop(maxEpochs 200, batchSize 16).
- **Measured:** 100% on all ten splits; the 95% threshold is crossed by epoch 1.
- **Upgrade path:** after this, open project 1 and feel the difference real photographs make.

## 10 · "Should I Bother?" — capstone (starter CSV)

*Story:* the personal assistant every cyclist wants: given temperature, rain chance and wind, should
you ride today? The decision rule is really an OR of single-feature vetoes (freezing, storm, gale),
which makes it a satisfying final build.

- **Data:** `projects/bike_decision.csv` — 30 rows, columns
  `temp_c, rain_chance_pct, wind_kmh, decision` (`yes`/`no`).
- **Recipe:** Source → Dense(32) → ReLU → Dense(16) → ReLU → Dense(2) → Sigmoid → Binary
  Cross-Entropy → Adam(lr 0.03) → Epoch Loop(maxEpochs 1000, batchSize 8).
- **Measured:** median 100%; 8 of 10 splits ≥95%; 95% reached by epoch 2. The misses are single
  deliberate boundary days (e.g. 2 °C with otherwise perfect rain/wind) — the dataset *wants* you to
  find those and ask whether the rule or the data is wrong. That discussion is the capstone.
- **Extension:** add a fifth column of your own (e.g. `ride_minutes`) and drag the edited CSV back
  in — the parser picks up the new feature automatically.

---

## Files in this pack

| File | What it is |
|---|---|
| `example_projects.md` | this document |
| `projects/spam_screener.csv` | 32 rows · links + spam words → ham/spam |
| `projects/student_pass_fail.csv` | 32 rows · attendance + study + prior grade → pass/fail |
| `projects/energy_forecast.csv` | 32 rows · weather + schedule → kWh (regression) |
| `projects/crop_yield.csv` | 32 rows · rainfall + temp + pH + fertilizer → tonnes (regression) |
| `projects/workout_zones.csv` | 30 rows · heart rate + HRV + pace → easy/moderate/peak |
| `projects/bike_decision.csv` | 30 rows · temp + rain + wind → ride yes/no |
| `tools/verify_example_projects.mjs` | the certificate: reruns every measurement above, exits non-zero if any project stops being learnable |

## Reproducing the certificate

```bash
node tools/verify_example_projects.mjs
```

The script prints the per-split accuracy (or MAE) distribution for all ten projects plus the MNIST
run, and passes only when every classification project's **median** held-out accuracy is ≥95%, every
regression project's median MAE is ≤10% of the target range, and MNIST stays ≥95%. Because the
engine's weight init is random by design, the script temporarily substitutes a seeded PRNG around
each run — the training math is byte-for-byte the shared engine code, only the starting weights are
pinned, which makes the output fully reproducible (verified twice).

Want to go further? Every recipe above is also reachable from **Tours → Build missions**, which
teach the same canvas moves one real action at a time — then come back here and build a project
end-to-end by yourself.
