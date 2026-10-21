"""
Vectorized Matrix Neural Network Engine (NumPy Core)
Implements Multi-Layer Perceptrons, Online Backpropagation, Adam/SGD Optimizers,
Loss Functions, and Iteration Hooks.

File: python_core/matrix_nn.py
"""

import numpy as np

# Online (human-in-the-loop) update guard rails - mirror of the JS engine constants.
DEFAULT_ONLINE_LR = 0.005
MIN_ONLINE_LR = 1e-6
MAX_ONLINE_LR = 0.5
ONLINE_GRADIENT_LIMIT = 1.0
SATURATED_SAMPLE_LOSS = 1e-6


class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0.0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0.0, x, alpha * x)

    @staticmethod
    def leaky_relu_deriv(x, alpha=0.01):
        return np.where(x > 0.0, 1.0, alpha)

    @staticmethod
    def sigmoid(x):
        clamped = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clamped))

    @staticmethod
    def sigmoid_deriv(x):
        s = Activations.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def tanh(x):
        clamped = np.clip(x, -50.0, 50.0)
        return np.tanh(clamped)

    @staticmethod
    def tanh_deriv(x):
        t = np.tanh(np.clip(x, -50.0, 50.0))
        return 1.0 - t ** 2

    @staticmethod
    def softmax(x):
        shift = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(np.clip(shift, -50.0, 50.0))
        sum_e = np.sum(exps, axis=-1, keepdims=True)
        return exps / np.maximum(1e-12, sum_e)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_deriv(x):
        return np.ones_like(x)


def get_activation(name="relu"):
    n = str(name).lower()
    if n == "relu":
        return Activations.relu, Activations.relu_deriv
    if n == "leaky_relu":
        return Activations.leaky_relu, Activations.leaky_relu_deriv
    if n == "sigmoid":
        return Activations.sigmoid, Activations.sigmoid_deriv
    if n == "tanh":
        return Activations.tanh, Activations.tanh_deriv
    if n == "softmax":
        return Activations.softmax, Activations.linear_deriv
    return Activations.linear, Activations.linear_deriv


class MatrixLayer:
    def __init__(self, in_size, out_size, activation="relu", layer_idx=0):
        self.in_size = in_size
        self.out_size = out_size
        self.activation_name = activation.lower()
        self.act_fn, self.act_deriv = get_activation(activation)
        self.layer_idx = layer_idx

        # He or Glorot initialization
        if self.activation_name in ["relu", "leaky_relu"]:
            scale = np.sqrt(2.0 / in_size)
            self.W = np.random.randn(out_size, in_size) * scale
        else:
            limit = np.sqrt(6.0 / (in_size + out_size))
            self.W = np.random.uniform(-limit, limit, (out_size, in_size))

        self.b = np.zeros(out_size, dtype=float)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Adam moments
        self.m_w = np.zeros_like(self.W)
        self.v_w = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def zero_gradients(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)


class MatrixNeuralNetwork:
    def __init__(self, architecture, learning_rate=0.03, loss="binary_crossentropy", optimizer="adam", weight_decay=0.0001):
        self.architecture = architecture
        self.lr = learning_rate
        self.loss_type = loss.lower()
        self.optimizer_type = optimizer.lower()
        self.weight_decay = weight_decay

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.time_step = 0

        self.layers = []
        for i in range(len(architecture) - 1):
            in_s = architecture[i]["size"]
            out_s = architecture[i + 1]["size"]
            act = architecture[i + 1].get("activation", "sigmoid" if i == len(architecture) - 2 else "relu")
            self.layers.append(MatrixLayer(in_s, out_s, act, i))

        self.on_iteration = None  # Hook: on_iteration(weights, activations, gradients)

    def forward_sample(self, x):
        activations = [np.array(x, dtype=float)]
        pre_activations = []
        cur = activations[0]

        for layer in self.layers:
            z = np.dot(layer.W, cur) + layer.b
            pre_activations.append(z)
            a = layer.act_fn(z)
            activations.append(a)
            cur = a

        return cur, activations, pre_activations

    def predict(self, X):
        return np.array([self.forward_sample(x)[0] for x in X])

    def compute_loss(self, y_pred, y_true):
        y_pred = np.array(y_pred, dtype=float)
        y_true = np.array(y_true, dtype=float)

        if self.loss_type == "mse":
            return 0.5 * np.mean((y_pred - y_true) ** 2)

        if self.loss_type in ["binary_crossentropy", "bce"]:
            p = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
            return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

        # Categorical Cross-Entropy
        p = np.clip(y_pred, 1e-12, 1.0)
        return -np.sum(y_true * np.log(p))

    def _compute_deltas(self, output, pre_activations, y):
        """
        Backpropagated deltas (dL/dz) for every layer - the single source of truth for
        gradient computation. Batch training and the online/active-learning update both
        call this, so a drawn correction and a trained epoch use identical math.
        """
        num_layers = len(self.layers)
        deltas = [None] * num_layers
        if num_layers == 0:
            return deltas

        y_pred = np.asarray(output, dtype=float)
        y_true = np.asarray(y, dtype=float)
        out_layer = self.layers[-1]
        last_z = pre_activations[-1]

        if (out_layer.activation_name == "softmax" and self.loss_type in ["categorical_crossentropy", "cce"]) or \
           (out_layer.activation_name == "sigmoid" and self.loss_type in ["binary_crossentropy", "bce"]):
            # Canonical merged derivative: softmax+CCE and sigmoid+BCE both give dL/dz = y_pred - y_true
            deltas[-1] = y_pred - y_true
        else:
            dL_da = y_pred - y_true
            da_dz = out_layer.act_deriv(last_z)
            deltas[-1] = dL_da * da_dz

        for l_idx in range(num_layers - 2, -1, -1):
            layer = self.layers[l_idx]
            next_layer = self.layers[l_idx + 1]
            next_delta = deltas[l_idx + 1]
            z = pre_activations[l_idx]

            sum_err = np.dot(next_layer.W.T, next_delta)
            clipped = np.clip(sum_err, -100.0, 100.0)
            deltas[l_idx] = clipped * layer.act_deriv(z)

        return deltas

    def train_batch(self, X_batch, y_batch):
        batch_size = len(X_batch)
        if batch_size == 0:
            return 0.0, 0.0

        for l in self.layers:
            l.zero_gradients()

        total_loss = 0.0
        correct = 0
        last_acts = None
        last_deltas = []

        for b in range(batch_size):
            x = X_batch[b]
            y = y_batch[b]

            y_pred, acts, zs = self.forward_sample(x)
            last_acts = acts

            loss = self.compute_loss(y_pred, y)
            total_loss += loss

            if len(y) > 1:
                if np.argmax(y_pred) == np.argmax(y):
                    correct += 1
            else:
                if abs(y_pred[0] - y[0]) < 0.25:
                    correct += 1

            # Backpropagation (shared with the online/active-learning update path)
            deltas = self._compute_deltas(y_pred, zs, y)
            last_deltas = deltas

            # Accumulate gradients
            for l_idx in range(len(self.layers)):
                l = self.layers[l_idx]
                d = deltas[l_idx]
                prev_a = acts[l_idx]
                l.dW += np.outer(d, prev_a) / batch_size
                l.db += d / batch_size

        # Optimizer update
        self.step_optimizer()

        if self.on_iteration is not None:
            self.on_iteration(self.get_weights_snapshot(), last_acts, last_deltas)

        return total_loss / batch_size, correct / batch_size

    def step_optimizer(self):
        self.time_step += 1
        t = self.time_step

        for l in self.layers:
            if self.optimizer_type == "adam":
                l.m_w = self.beta1 * l.m_w + (1.0 - self.beta1) * (l.dW + self.weight_decay * l.W)
                l.v_w = self.beta2 * l.v_w + (1.0 - self.beta2) * ((l.dW + self.weight_decay * l.W) ** 2)

                l.m_b = self.beta1 * l.m_b + (1.0 - self.beta1) * l.db
                l.v_b = self.beta2 * l.v_b + (1.0 - self.beta2) * (l.db ** 2)

                m_hat_w = l.m_w / (1.0 - self.beta1 ** t)
                v_hat_w = l.v_w / (1.0 - self.beta2 ** t)
                m_hat_b = l.m_b / (1.0 - self.beta1 ** t)
                v_hat_b = l.v_b / (1.0 - self.beta2 ** t)

                l.W -= (self.lr * m_hat_w) / (np.sqrt(v_hat_w) + self.epsilon)
                l.b -= (self.lr * m_hat_b) / (np.sqrt(v_hat_b) + self.epsilon)
            else:
                l.W -= self.lr * (l.dW + self.weight_decay * l.W)
                l.b -= self.lr * l.db

    def set_learning_rate(self, lr):
        """
        Learning-rate guard rail: NaN, negatives, zero and absurd magnitudes are clamped
        into a range in which gradient descent is still a contraction. Returns the value
        that was actually installed.
        """
        try:
            safe = float(lr)
        except (TypeError, ValueError):
            safe = MIN_ONLINE_LR
        if not np.isfinite(safe) or safe <= 0.0:
            safe = MIN_ONLINE_LR
        if safe > MAX_ONLINE_LR:
            safe = MAX_ONLINE_LR
        self.lr = safe
        return safe

    def sanitize_weights(self):
        """Replace every non-finite parameter (NaN/Inf) with 0.0; returns a real repair count."""
        repaired = 0
        for layer in self.layers:
            mask_w = ~np.isfinite(layer.W)
            count_w = int(np.count_nonzero(mask_w))
            if count_w:
                layer.W[mask_w] = 0.0
                repaired += count_w
            mask_b = ~np.isfinite(layer.b)
            count_b = int(np.count_nonzero(mask_b))
            if count_b:
                layer.b[mask_b] = 0.0
                repaired += count_b
        self.non_finite_repairs = getattr(self, "non_finite_repairs", 0) + repaired
        return {"repaired": repaired, "total": self.non_finite_repairs}

    def apply_online_update(self, x, y, lr):
        """
        One bounded single-sample gradient step (the engine behind human-in-the-loop
        corrections). Real backprop: forward pass, shared _compute_deltas, gradient =
        delta (x) previous activation per layer. The trust-region scale bounds the largest
        gradient coordinate to ONLINE_GRADIENT_LIMIT, so one update moves any weight by at
        most `lr` and - unlike the batch Adam optimizer - leaves coordinates this sample
        does not activate untouched.
        """
        y_pred, acts, zs = self.forward_sample(x)
        deltas = self._compute_deltas(y_pred, zs, y)

        max_gradient = 0.0
        for l_idx, layer in enumerate(self.layers):
            delta = deltas[l_idx]
            prev_a = acts[l_idx]
            layer.dW = np.outer(delta, prev_a)
            layer.db = np.array(delta, dtype=float)
            if layer.dW.size:
                max_gradient = max(max_gradient, float(np.max(np.abs(layer.dW))))
            if layer.db.size:
                max_gradient = max(max_gradient, float(np.max(np.abs(layer.db))))

        scale = ONLINE_GRADIENT_LIMIT / max_gradient if max_gradient > ONLINE_GRADIENT_LIMIT else 1.0

        for layer in self.layers:
            layer.W -= lr * layer.dW * scale
            layer.b -= lr * layer.db * scale

        # Same hook the batch trainer fires, so a Python visualizer receives the real
        # activations / deltas / post-update weights for the correction it just watched.
        if self.on_iteration is not None:
            self.on_iteration(self.get_weights_snapshot(), acts, deltas)

        return {"max_gradient": max_gradient, "scale": scale}

    def online_step(self, x, y, custom_lr=None, steps=1):
        """
        Online Backpropagation Step on a single sample (Human-in-the-Loop Active Learning).

        Immediately adjusts weights to rectify a mistake on x with true label y, and reports
        the REAL measured loss before and after the update plus a per-step trace so a UI can
        show a genuine curve. The step is bounded (see apply_online_update) and uses the
        conservative online learning rate unless the caller explicitly overrides it; the
        loop stops early only when the sample is numerically solved AND classified correctly.

        Kept return keys (initial_loss/post_loss/initial_pred/post_pred/loss_reduced/
        loss_delta) plus steps (performed), requested_steps, trace, sanitized_parameters
        and learning_rate_used.
        """
        saved_lr = self.lr

        try:
            steps_value = float(steps)
        except (TypeError, ValueError):
            steps_value = 1.0
        if not np.isfinite(steps_value):
            steps_value = 1.0
        requested_steps = max(1, min(50, int(np.floor(steps_value))))

        # A human correction never inherits the batch training rate: without an explicit
        # override it uses the conservative online fine-tuning rate.
        if custom_lr is not None:
            effective_lr = self.set_learning_rate(custom_lr)
        else:
            effective_lr = DEFAULT_ONLINE_LR

        init_pred, _, _ = self.forward_sample(x)
        initial_loss = float(self.compute_loss(init_pred, y))
        initial_pred = init_pred.tolist()

        trace = [{
            "step": 0,
            "loss": initial_loss,
            "predicted_class": int(np.argmax(init_pred)),
        }]

        target_class = int(np.argmax(np.asarray(y, dtype=float)))
        sanitized = 0
        for s in range(requested_steps):
            self.apply_online_update(x, y, effective_lr)
            step_pred, _, _ = self.forward_sample(x)
            step_loss = float(self.compute_loss(step_pred, y))
            step_class = int(np.argmax(step_pred))
            trace.append({
                "step": s + 1,
                "loss": step_loss,
                "predicted_class": step_class,
                "confidence": float(np.max(step_pred)),
            })
            sanitized += self.sanitize_weights()["repaired"]

            # Stop only when this sample is genuinely solved - solved loss AND correct class.
            if step_loss <= SATURATED_SAMPLE_LOSS and step_class == target_class:
                break
        performed_steps = len(trace) - 1

        # Forward pass immediately after the update train
        post_pred, _, _ = self.forward_sample(x)
        post_loss = float(self.compute_loss(post_pred, y))
        post_pred_list = post_pred.tolist()

        if custom_lr is not None:
            self.set_learning_rate(saved_lr)

        return {
            "initial_loss": initial_loss,
            "post_loss": post_loss,
            "initial_pred": initial_pred,
            "post_pred": post_pred_list,
            "loss_reduced": bool(post_loss < initial_loss),
            "loss_delta": float(initial_loss - post_loss),
            "steps": performed_steps,
            "requested_steps": requested_steps,
            "trace": trace,
            "sanitized_parameters": sanitized,
            "learning_rate_used": effective_lr,
        }

    def get_weights_snapshot(self):
        return [{
            "layer_index": l.layer_idx,
            "input_size": l.in_size,
            "output_size": l.out_size,
            "activation": l.activation_name,
            "weights": l.W.tolist(),
            "biases": l.b.tolist()
        } for l in self.layers]
