"""
PHASE 3 - TEST 2: Active Learning / Human-in-the-Loop verification (Python reference)

Mirrors tests/test_online_feedback.js against the Python parity engine
(python_core/matrix_nn.py). Everything measured here comes from real forward /
backward passes over the real binary MNIST corpus shipped as
python_core/data/mnist_01_full.npz (the complete 0-vs-1 store, verified against
its SHA-256 digests when it was exported by tools/fetch_mnist.py).

Coverage:
  1. > 95% held-out accuracy within a 10-epoch budget on the real corpus.
  2. online_step on a real misclassified image reduces the measured loss.
  3. A trained model correcting a real shifted-drawing mistake (the worker
     feedback case) measurably reduces the loss on the corrected image.
  4. Learning-rate guard rails: clamp range, no inheritance of the batch
     training rate, explicit override + restore, finite weights under an
     adversarial learning rate.
  5. Online-step trace semantics and the early-stop rule (solved loss AND
     correct class - both required).
  6. Multi-correction guard: six human corrections at the studio default
     budget (3 bounded steps, default rate) do not collapse held-out accuracy.

File: python_core/test_active_learning.py
"""

import copy
import os
import unittest

import numpy as np

from python_core.matrix_nn import (
    MatrixNeuralNetwork,
    DEFAULT_ONLINE_LR,
    MIN_ONLINE_LR,
    MAX_ONLINE_LR,
    SATURATED_SAMPLE_LOSS,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist_01_full.npz")

# The exact studio configuration for the 784-input / 2-class MNIST problem.
ARCHITECTURE = [
    {"size": 784, "activation": "linear"},
    {"size": 32, "activation": "relu"},
    {"size": 32, "activation": "relu"},
    {"size": 2, "activation": "sigmoid"},
]
EPOCH_BUDGET = 10
ACCURACY_TARGET = 0.95
BATCH_SIZE = 32
TRAIN_SEED = 20261018

CHECKS = {"passed": 0, "total": 0}

_STORE = None
_REFERENCE = None


# ---------------------------------------------------------------------------
# Data / training helpers (cached across tests - training is the slow part)
# ---------------------------------------------------------------------------

def load_store():
    """Load the real 0-vs-1 MNIST store; cache the decoded tensors."""
    global _STORE
    if _STORE is None:
        raw = np.load(DATA_PATH)
        _STORE = {
            "trainX": raw["trainX"].astype(np.float64) / 255.0,
            "trainY": np.eye(2, dtype=np.float64)[raw["trainY"]],
            "testX": raw["testX"].astype(np.float64) / 255.0,
            "testY": np.eye(2, dtype=np.float64)[raw["testY"]],
            "trainLabels": raw["trainY"].astype(int),
            "testLabels": raw["testY"].astype(int),
        }
    return _STORE


def evaluate(net, X, Y):
    """One honest pass over a partition: measured loss, accuracy, mistakes."""
    loss_sum = 0.0
    correct = 0
    misclassified = []
    for i in range(len(X)):
        pred, _, _ = net.forward_sample(X[i])
        loss = float(net.compute_loss(pred, Y[i]))
        loss_sum += loss
        pred_class = int(np.argmax(pred))
        true_class = int(np.argmax(Y[i]))
        if pred_class == true_class:
            correct += 1
        else:
            misclassified.append({
                "index": i,
                "predicted_class": pred_class,
                "true_class": true_class,
                "loss": loss,
            })
    return {
        "count": len(X),
        "loss": loss_sum / max(1, len(X)),
        "accuracy": correct / max(1, len(X)),
        "misclassified": misclassified,
    }


def train_reference_network():
    """Train the studio configuration on the full store, up to the 10-epoch budget."""
    global _REFERENCE
    if _REFERENCE is None:
        store = load_store()
        np.random.seed(TRAIN_SEED)
        net = MatrixNeuralNetwork(
            ARCHITECTURE,
            learning_rate=0.03,
            loss="binary_crossentropy",
            optimizer="adam",
            weight_decay=0.0001,
        )

        history = []
        achieved_epoch = None
        for epoch in range(1, EPOCH_BUDGET + 1):
            rng = np.random.RandomState(TRAIN_SEED + epoch)
            indices = rng.permutation(len(store["trainX"]))
            loss_sum = 0.0
            acc_sum = 0.0
            batches = 0
            for start in range(0, len(indices), BATCH_SIZE):
                batch = indices[start:start + BATCH_SIZE]
                loss, acc = net.train_batch(store["trainX"][batch], store["trainY"][batch])
                loss_sum += float(loss)
                acc_sum += float(acc)
                batches += 1

            evaluation = evaluate(net, store["testX"], store["testY"])
            history.append({
                "epoch": epoch,
                "train_loss": loss_sum / max(1, batches),
                "train_accuracy": acc_sum / max(1, batches),
                "test_loss": evaluation["loss"],
                "test_accuracy": evaluation["accuracy"],
            })
            if achieved_epoch is None and evaluation["accuracy"] > ACCURACY_TARGET:
                achieved_epoch = epoch
                break  # budget requirement measured as satisfied - stop for speed

        final = evaluate(net, store["testX"], store["testY"])
        _REFERENCE = {
            "net": net,
            "history": history,
            "achieved_epoch": achieved_epoch,
            "final": final,
        }
    return _REFERENCE


def shift_drawing(vec, dx, dy):
    """Move a 28x28 drawing by (dx, dy) pixels - models an off-centre human drawing."""
    field = np.array(vec, dtype=np.float64).reshape(28, 28)
    shifted = np.roll(np.roll(field, dy, axis=0), dx, axis=1)
    return shifted.reshape(-1)


def find_worst_mistake(net, X, Y, limit=None):
    """Return the real misclassification with the highest measured loss."""
    worst = None
    count = len(X) if limit is None else min(limit, len(X))
    for i in range(count):
        pred, _, _ = net.forward_sample(X[i])
        pred_class = int(np.argmax(pred))
        true_class = int(np.argmax(Y[i]))
        if pred_class == true_class:
            continue
        loss = float(net.compute_loss(pred, Y[i]))
        if worst is None or loss > worst["loss"]:
            worst = {
                "index": i,
                "predicted_class": pred_class,
                "true_class": true_class,
                "loss": loss,
            }
    return worst


def find_shifted_mistake(net, X, Y, shifts, limit=None):
    """Find a real misclassified sample among shifted (off-centre) drawings."""
    worst = None
    count = len(X) if limit is None else min(limit, len(X))
    for i in range(count):
        for (dx, dy) in shifts:
            drawing = shift_drawing(X[i], dx, dy)
            pred, _, _ = net.forward_sample(drawing)
            pred_class = int(np.argmax(pred))
            true_class = int(np.argmax(Y[i]))
            if pred_class == true_class:
                continue
            loss = float(net.compute_loss(pred, Y[i]))
            if worst is None or loss > worst["loss"]:
                worst = {
                    "index": i,
                    "dx": dx,
                    "dy": dy,
                    "predicted_class": pred_class,
                    "true_class": true_class,
                    "loss": loss,
                    "drawing": drawing,
                }
    return worst


def find_moderate_loss_samples(net, X, Y, count, low, high, scan_limit):
    """Real samples whose true-label loss sits in [low, high) - correctable mistakes."""
    found = []
    for i in range(min(scan_limit, len(X))):
        pred, _, _ = net.forward_sample(X[i])
        loss = float(net.compute_loss(pred, Y[i]))
        if low <= loss < high:
            found.append({"index": i, "loss": loss})
            if len(found) == count:
                break
    return found


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class ActiveLearningVerification(unittest.TestCase):
    def setUp(self):
        np.random.seed(TRAIN_SEED)

    def check(self, condition, label):
        CHECKS["total"] += 1
        if condition:
            CHECKS["passed"] += 1
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
        self.assertTrue(condition, label)

    # ------------------------------------------------------------------
    # 1. Convergence on the real corpus within the 10-epoch budget
    # ------------------------------------------------------------------
    def test_mnist_01_convergence_within_budget(self):
        print("\n==================================================================")
        print("  PART 1 - CONVERGENCE on the real 0-vs-1 store (full partition)")
        print("==================================================================")

        store = load_store()
        print(f"  store ............. {os.path.relpath(DATA_PATH)}")
        print(f"  train images ...... {len(store['trainX'])}")
        print(f"  held-out images ... {len(store['testX'])}")
        print(f"  input value range . [{store['trainX'].min():.1f}, {store['trainX'].max():.1f}] (real 0/1 pixel intensities)")

        self.check(set(np.unique(store["trainLabels"]).tolist()) == {0, 1}, "training store holds exactly the two real classes (0 and 1)")
        self.check(set(np.unique(store["testLabels"]).tolist()) == {0, 1}, "held-out partition holds exactly the two real classes (0 and 1)")

        reference = train_reference_network()
        print(f"\n  Training (max {EPOCH_BUDGET} epochs, architecture "
              f"{' -> '.join(str(l['size']) for l in ARCHITECTURE)})")
        print("  epoch | train loss | train acc | test loss | test acc")
        print("  ------+------------+-----------+-----------+----------")
        for m in reference["history"]:
            print(f"  {m['epoch']:5d} | {m['train_loss']:10.6f} | {m['train_accuracy'] * 100:8.2f}% | "
                  f"{m['test_loss']:9.6f} | {m['test_accuracy'] * 100:7.2f}%")

        final = reference["final"]
        print(f"\n  Final held-out evaluation: loss {final['loss']:.6f}, accuracy {final['accuracy'] * 100:.2f}%, "
              f"misclassified {len(final['misclassified'])} / {final['count']}")

        self.check(reference["achieved_epoch"] is not None,
                   f"network reached > {ACCURACY_TARGET * 100:.0f}% within the {EPOCH_BUDGET}-epoch budget "
                   f"(achieved at epoch {reference['achieved_epoch']})")
        self.check(final["accuracy"] > ACCURACY_TARGET,
                   f"final measured held-out accuracy {final['accuracy'] * 100:.2f}% exceeds the {ACCURACY_TARGET * 100:.0f}% target")

    # ------------------------------------------------------------------
    # 2. online_step reduces the measured loss on a real mistake
    # ------------------------------------------------------------------
    def test_online_step_reduces_loss_on_real_mistake(self):
        print("\n==================================================================")
        print("  PART 2 - online_step on a REAL misclassified image (untrained net)")
        print("==================================================================")

        store = load_store()
        net = MatrixNeuralNetwork(
            ARCHITECTURE, learning_rate=0.03, loss="binary_crossentropy", optimizer="adam"
        )

        baseline = evaluate(net, store["testX"], store["testY"])
        print(f"  untrained network held-out accuracy: {baseline['accuracy'] * 100:.2f}% "
              f"({len(baseline['misclassified'])} mistakes)")
        self.check(baseline["accuracy"] < ACCURACY_TARGET, "untrained network is genuinely unlearned (baseline below target)")

        # Remember the training rate before the human correction.
        training_lr_before = net.lr

        mistake = find_worst_mistake(net, store["testX"], store["testY"])
        self.check(mistake is not None, "untrained network misclassified at least one real digit")
        print(f"  correcting: held-out image #{mistake['index']} (predicted {mistake['predicted_class']}, "
              f"truth {mistake['true_class']}, loss {mistake['loss']:.5f})")

        x = store["testX"][mistake["index"]]
        y = store["testY"][mistake["index"]]
        result = net.online_step(x, y, steps=25)

        reduction = (result["initial_loss"] - result["post_loss"]) / max(1e-12, result["initial_loss"])
        print(f"  loss: {result['initial_loss']:.6f} -> {result['post_loss']:.6f} "
              f"({reduction * 100:.2f}% reduction over {result['steps']} bounded step(s), "
              f"lr {result['learning_rate_used']})")
        print(f"  prediction: class {int(np.argmax(result['initial_pred']))} -> class {int(np.argmax(result['post_pred']))} "
              f"(human says class {mistake['true_class']})")

        self.check(result["learning_rate_used"] == DEFAULT_ONLINE_LR,
                   f"correction used the conservative default online rate ({DEFAULT_ONLINE_LR}), not the training rate")
        self.check(net.lr == training_lr_before,
                   f"training learning rate left untouched ({net.lr})")
        self.check(1 <= result["steps"] <= 25, f"performed between 1 and 25 bounded steps ({result['steps']})")
        self.check(len(result["trace"]) == result["steps"] + 1,
                   f"loss trace length ({len(result['trace'])}) equals steps + 1")
        self.check(result["trace"][0]["loss"] == result["initial_loss"], "trace starts at the measured pre-correction loss")
        self.check(result["trace"][-1]["loss"] == result["post_loss"], "trace ends at the measured post-correction loss")
        self.check(all(np.isfinite(point["loss"]) for point in result["trace"]), "every trace point is a finite measured loss")
        self.check(result["post_loss"] < result["initial_loss"], "online_step reduced the loss on the corrected image")
        self.check(reduction > 0.25, f"loss reduction is decisive ({reduction * 100:.2f}% > 25%)")

        improvements = sum(
            1 for s in range(len(result["trace"]) - 1)
            if result["trace"][s + 1]["loss"] > result["trace"][s]["loss"] + 1e-12
        )
        print(f"  argmax flipped to the human class: {int(np.argmax(result['post_pred'])) == mistake['true_class']} "
              f"(flip is not asserted: bounded updates lower the loss even while argmax may need further steps)")
        self.check(improvements == 0, f"loss curve is monotone ({improvements} step(s) of {result['steps']} ticked up)")

        finite = all(np.all(np.isfinite(layer.W)) and np.all(np.isfinite(layer.b)) for layer in net.layers)
        self.check(finite, "no non-finite parameter after the online update")
        self.check(result["sanitized_parameters"] == 0, "nothing needed repairing after the update")

        max_weight = max(float(np.max(np.abs(layer.W))) for layer in net.layers)
        print(f"  weights stayed sane (max |w| = {max_weight:.4f})")
        self.check(max_weight < 10.0, "weights stayed within a sane magnitude after the correction")

        adam_untouched = all(
            np.all(layer.m_w == 0.0) and np.all(layer.v_w == 0.0)
            and np.all(layer.m_b == 0.0) and np.all(layer.v_b == 0.0)
            for layer in net.layers
        )
        self.check(adam_untouched, "online correction left the Adam batch-optimizer moments untouched (bounded trust region)")

        # Re-correcting an already-corrected image must not make it worse.
        again = net.online_step(x, y, steps=5)
        print(f"  re-correcting the same image: {again['initial_loss']:.6f} -> {again['post_loss']:.6f}")
        self.check(again["post_loss"] <= again["initial_loss"] + 1e-9,
                   "re-correcting an already-corrected image does not worsen it")

    # ------------------------------------------------------------------
    # 3. Trained model + shifted drawing (the worker feedback case)
    # ------------------------------------------------------------------
    def test_trained_model_corrects_shifted_drawing(self):
        print("\n==================================================================")
        print("  PART 3 - trained model correcting a real shifted-drawing mistake")
        print("==================================================================")

        store = load_store()
        reference = train_reference_network()
        net = copy.deepcopy(reference["net"])

        shifts = [(2, 2), (-2, 2), (2, -2), (-2, -2)]
        mistake = find_shifted_mistake(net, store["testX"], store["testY"], shifts, limit=500)
        self.check(mistake is not None,
                   "found a real off-centre drawing the trained model misclassifies")
        print(f"  correcting: held-out image #{mistake['index']} shifted by ({mistake['dx']},{mistake['dy']}) "
              f"- predicted {mistake['predicted_class']}, truth {mistake['true_class']}, loss {mistake['loss']:.5f}")

        result = net.online_step(mistake["drawing"], store["testY"][mistake["index"]], steps=25)
        reduction = (result["initial_loss"] - result["post_loss"]) / max(1e-12, result["initial_loss"])
        print(f"  loss: {result['initial_loss']:.6f} -> {result['post_loss']:.6f} "
              f"({reduction * 100:.2f}% reduction over {result['steps']} step(s), lr {result['learning_rate_used']})")
        print(f"  prediction now: class {int(np.argmax(result['post_pred']))} (human label {mistake['true_class']})")

        self.check(result["learning_rate_used"] == DEFAULT_ONLINE_LR,
                   "correction did not inherit the 0.03 batch training rate")
        self.check(result["post_loss"] < result["initial_loss"],
                   "loss decreased on the exact corrected drawing")
        self.check(reduction > 0.10, f"loss reduction is decisive ({reduction * 100:.2f}% > 10%)")
        self.check(1 <= result["steps"] <= 25, f"bounded step count respected ({result['steps']} <= 25)")

    # ------------------------------------------------------------------
    # 4. Learning-rate guard rails
    # ------------------------------------------------------------------
    def test_learning_rate_guardrails(self):
        print("\n==================================================================")
        print("  PART 4 - learning-rate guard rails")
        print("==================================================================")

        store = load_store()
        net = MatrixNeuralNetwork(
            ARCHITECTURE, learning_rate=0.03, loss="binary_crossentropy", optimizer="adam"
        )
        x = store["testX"][0]
        y = store["testY"][0]

        exploding = net.set_learning_rate(1e9)
        self.check(exploding == MAX_ONLINE_LR, f"explosive rate 1e9 clamped to {exploding}")

        negative = net.set_learning_rate(-5)
        self.check(negative == MIN_ONLINE_LR, f"negative rate -5 clamped to {negative}")

        zero = net.set_learning_rate(0)
        self.check(zero == MIN_ONLINE_LR, f"zero rate clamped to {zero}")

        nan = net.set_learning_rate(float("nan"))
        self.check(nan == MIN_ONLINE_LR, f"NaN rate clamped to {nan}")

        junk = net.set_learning_rate("abc")
        self.check(junk == MIN_ONLINE_LR, f"non-numeric rate clamped to {junk}")

        # Restore the training rate and verify the default correction does not touch it.
        net.set_learning_rate(0.03)
        default_step = net.online_step(x, y)
        self.check(default_step["learning_rate_used"] == DEFAULT_ONLINE_LR,
                   f"default correction used {DEFAULT_ONLINE_LR}, training rate untouched ({net.lr})")

        override = net.online_step(x, y, custom_lr=0.2)
        self.check(override["learning_rate_used"] == 0.2, f"explicit override honoured ({override['learning_rate_used']})")
        self.check(net.lr == 0.03, f"training rate restored after the override ({net.lr})")

        adversarial = net.online_step(x, y, custom_lr=1e9)
        finite = all(np.all(np.isfinite(layer.W)) and np.all(np.isfinite(layer.b)) for layer in net.layers)
        self.check(adversarial["learning_rate_used"] == MAX_ONLINE_LR,
                   f"adversarial 1e9 override clamped to {adversarial['learning_rate_used']}")
        self.check(finite and np.isfinite(adversarial["post_loss"]),
                   f"adversarial bypass kept every parameter finite (loss {adversarial['post_loss']:.4f})")
        self.check(net.lr == 0.03, "training rate restored after the adversarial override")

    # ------------------------------------------------------------------
    # 5. Trace semantics and the early-stop rule
    # ------------------------------------------------------------------
    def test_trace_semantics_and_early_stop(self):
        print("\n==================================================================")
        print("  PART 5 - trace semantics and early-stop rule")
        print("==================================================================")

        store = load_store()
        x = store["testX"][0]
        y = store["testY"][0]

        net = MatrixNeuralNetwork(
            ARCHITECTURE, learning_rate=0.03, loss="binary_crossentropy", optimizer="adam"
        )

        zero_steps = net.online_step(x, y, steps=0)
        self.check(zero_steps["requested_steps"] == 1, f"steps=0 clamped to a requested 1 ({zero_steps['requested_steps']})")

        many_steps = net.online_step(x, y, steps=999)
        self.check(many_steps["requested_steps"] == 50, f"steps=999 capped at 50 ({many_steps['requested_steps']})")
        self.check(len(many_steps["trace"]) == many_steps["steps"] + 1, "trace length equals performed steps + 1")
        self.check("confidence" in many_steps["trace"][-1], "every performed step records its measured confidence")

        # Saturated + correct: the sample is numerically solved and classified right,
        # so the loop must stop immediately (only one update is ever performed).
        saturated = MatrixNeuralNetwork(
            ARCHITECTURE, learning_rate=0.03, loss="binary_crossentropy", optimizer="adam"
        )
        for i, layer in enumerate(saturated.layers):
            layer.W.fill(0.0)
            layer.b.fill(0.0)
            if i == len(saturated.layers) - 1:
                layer.b[0] = 30.0
                layer.b[1] = -30.0

        solved = saturated.online_step(x, [1.0, 0.0], steps=10)
        self.check(solved["initial_loss"] <= SATURATED_SAMPLE_LOSS,
                   f"saturated sample starts numerically solved (loss {solved['initial_loss']:.3e} <= {SATURATED_SAMPLE_LOSS:.0e})")
        self.check(solved["steps"] == 1,
                   f"early stop fired for a solved + correctly classified sample ({solved['steps']} of {solved['requested_steps']} steps)")
        self.check(solved["trace"][-1]["loss"] <= SATURATED_SAMPLE_LOSS, "early stop happened on a measured solved loss")

        # Solved numerically but classified WRONG: the early stop must NOT fire.
        wrong = MatrixNeuralNetwork(
            ARCHITECTURE, learning_rate=0.03, loss="binary_crossentropy", optimizer="adam"
        )
        for i, layer in enumerate(wrong.layers):
            layer.W.fill(0.0)
            layer.b.fill(0.0)
            if i == len(wrong.layers) - 1:
                layer.b[0] = -30.0
                layer.b[1] = 30.0

        stuck = wrong.online_step(x, [1.0, 0.0], steps=10)
        self.check(stuck["steps"] == stuck["requested_steps"],
                   f"saturated-but-wrong sample ran the full budget ({stuck['steps']} of {stuck['requested_steps']}) - "
                   f"loss {stuck['trace'][-1]['loss']:.4f} alone does not stop the loop")
        self.check(stuck["trace"][-1]["loss"] > SATURATED_SAMPLE_LOSS, "the loop kept correcting because the class was still wrong")
        self.check(stuck["sanitized_parameters"] == 0, "no parameter needed repair during the stuck correction")

    # ------------------------------------------------------------------
    # 6. Multi-correction guard: 6 corrections at the studio default budget
    # ------------------------------------------------------------------
    def test_multi_correction_guard_preserves_holdout(self):
        print("\n==================================================================")
        print("  PART 6 - guard: 6 human corrections at the studio default budget")
        print("==================================================================")

        store = load_store()
        reference = train_reference_network()
        net = copy.deepcopy(reference["net"])

        before = evaluate(net, store["testX"], store["testY"])

        # The studio default correction covers drawings with a real moderate mistake
        # (a human only corrects images the model got meaningfully wrong).
        candidates = find_moderate_loss_samples(
            net, store["trainX"], store["trainY"], count=6, low=0.7, high=6.0, scan_limit=4000
        )
        self.check(len(candidates) >= 3, f"found {len(candidates)} real moderate-loss drawings to guard with (>= 3)")

        reduced = 0
        for candidate in candidates:
            x = store["trainX"][candidate["index"]]
            y = store["trainY"][candidate["index"]]
            result = net.online_step(x, y, steps=3)  # studio default: 3 bounded steps, default rate
            print(f"    guard drawing #{candidate['index']} (truth {int(np.argmax(y))}): "
                  f"loss {result['initial_loss']:.4f} -> {result['post_loss']:.4f}")
            if result["post_loss"] < result["initial_loss"]:
                reduced += 1

        self.check(reduced == len(candidates), f"every one of {len(candidates)} corrections reduced its own loss ({reduced} reduced)")

        after = evaluate(net, store["testX"], store["testY"])
        drop = before["accuracy"] - after["accuracy"]
        print(f"  held-out accuracy: {before['accuracy'] * 100:.2f}% -> {after['accuracy'] * 100:.2f}% ({-drop * 100:+.2f}%)")
        self.check(drop <= 0.02, f"held-out accuracy dropped at most 2.00% (actual {drop * 100:.2f}%)")


def tearDownModule():
    print("\n============================================================")
    print(f"  checks passed: {CHECKS['passed']}/{CHECKS['total']}")
    if CHECKS["total"] and CHECKS["passed"] == CHECKS["total"]:
        print("  [PASS] Python active-learning verification complete")
    print("============================================================")


if __name__ == "__main__":
    unittest.main()
