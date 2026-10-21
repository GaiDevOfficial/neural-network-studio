"""
Reference Python Neural Network Engine with Step-by-Step Hook Listeners
Neural Network Studio Platform
"""

import numpy as np
import json

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_deriv(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

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
        # Numeric stability shift
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(np.clip(shift_x, -50.0, 50.0))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def softmax_deriv(x):
        s = Activations.softmax(x)
        return s * (1.0 - s)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_deriv(x):
        return np.ones_like(x)

def get_activation(name):
    n = name.lower()
    if n == 'relu': return Activations.relu, Activations.relu_deriv
    if n == 'leaky_relu': return Activations.leaky_relu, Activations.leaky_relu_deriv
    if n == 'sigmoid': return Activations.sigmoid, Activations.sigmoid_deriv
    if n == 'tanh': return Activations.tanh, Activations.tanh_deriv
    if n == 'softmax': return Activations.softmax, Activations.softmax_deriv
    return Activations.linear, Activations.linear_deriv

class Layer:
    def __init__(self, in_size, out_size, activation='relu', layer_idx=0):
        self.in_size = in_size
        self.out_size = out_size
        self.activation_name = activation.lower()
        self.act_fn, self.act_deriv = get_activation(activation)
        self.layer_idx = layer_idx
        
        # He or Glorot initialization
        if self.activation_name in ['relu', 'leaky_relu']:
            scale = np.sqrt(2.0 / in_size)
            self.W = np.random.randn(out_size, in_size) * scale
        else:
            limit = np.sqrt(6.0 / (in_size + out_size))
            self.W = np.random.uniform(-limit, limit, (out_size, in_size))
            
        self.b = np.zeros(out_size)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

class NeuralNetworkEngine:
    def __init__(self, architecture, learning_rate=0.03, loss='mse', optimizer='adam'):
        self.architecture = architecture
        self.lr = learning_rate
        self.loss_type = loss.lower()
        self.optimizer_type = optimizer.lower()
        self.layers = []
        
        # Build layers
        for i in range(len(architecture) - 1):
            in_s = architecture[i]['size']
            out_s = architecture[i + 1]['size']
            act = architecture[i + 1].get('activation', 'relu' if i < len(architecture) - 2 else 'linear')
            self.layers.append(Layer(in_s, out_s, act, i))
            
        # Optimizer buffers (Adam)
        self.m_w = [np.zeros_like(l.W) for l in self.layers]
        self.v_w = [np.zeros_like(l.W) for l in self.layers]
        self.m_b = [np.zeros_like(l.b) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.iter_count = 0
        
        # Hook listeners
        self.hook_forward = None
        self.hook_backward = None

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
            
        if self.hook_forward:
            self.hook_forward(activations, pre_activations)
            
        return cur, activations, pre_activations

    def predict(self, X):
        return np.array([self.forward_sample(x)[0] for x in X])

    def train_batch(self, X_batch, y_batch):
        batch_size = len(X_batch)
        if batch_size == 0:
            return 0.0, 0.0
            
        # Zero gradients
        for l in self.layers:
            l.dW.fill(0.0)
            l.db.fill(0.0)
            
        total_loss = 0.0
        correct = 0
        last_acts = None
        last_deltas = []
        
        for b in range(batch_size):
            x = X_batch[b]
            y = y_batch[b]
            
            y_pred, acts, zs = self.forward_sample(x)
            last_acts = acts
            
            # Loss computation
            diff = y_pred - y
            sample_loss = np.mean(diff ** 2) if self.loss_type == 'mse' else -np.sum(y * np.log(np.clip(y_pred, 1e-12, 1.0)))
            total_loss += sample_loss
            
            # Accuracy metric
            if len(y) > 1:
                if np.argmax(y_pred) == np.argmax(y):
                    correct += 1
            else:
                if abs(y_pred[0] - y[0]) < 0.25:
                    correct += 1
                    
            # Backward propagation
            num_layers = len(self.layers)
            deltas = [None] * num_layers
            
            out_layer = self.layers[-1]
            last_z = zs[-1]
            
            if out_layer.activation_name == 'softmax' and self.loss_type in ['categorical_crossentropy', 'cce']:
                deltas[-1] = y_pred - y
            elif out_layer.activation_name == 'sigmoid' and self.loss_type in ['binary_crossentropy', 'bce']:
                deltas[-1] = y_pred - y
            else:
                # Chain rule
                loss_grad = 2.0 * diff / len(y) if self.loss_type == 'mse' else (y_pred - y)
                act_deriv = out_layer.act_deriv(last_z)
                deltas[-1] = loss_grad * act_deriv
                
            for l_idx in range(num_layers - 2, -1, -1):
                layer = self.layers[l_idx]
                next_layer = self.layers[l_idx + 1]
                next_delta = deltas[l_idx + 1]
                z = zs[l_idx]
                
                sum_err = np.dot(next_layer.W.T, next_delta)
                deltas[l_idx] = sum_err * layer.act_deriv(z)
                
            last_deltas = deltas
            
            # Accumulate batch gradients
            for l_idx in range(num_layers):
                l = self.layers[l_idx]
                d = deltas[l_idx]
                prev_a = acts[l_idx]
                l.dW += np.outer(d, prev_a) / batch_size
                l.db += d / batch_size
                
        # Optimizer step (Adam / SGD)
        self.iter_count += 1
        t = self.iter_count
        
        for i, l in enumerate(self.layers):
            if self.optimizer_type == 'adam':
                self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * l.dW
                self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * (l.dW ** 2)
                
                self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * l.db
                self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (l.db ** 2)
                
                m_hat_w = self.m_w[i] / (1.0 - self.beta1 ** t)
                v_hat_w = self.v_w[i] / (1.0 - self.beta2 ** t)
                m_hat_b = self.m_b[i] / (1.0 - self.beta1 ** t)
                v_hat_b = self.v_b[i] / (1.0 - self.beta2 ** t)
                
                l.W -= (self.lr * m_hat_w) / (np.sqrt(v_hat_w) + self.epsilon)
                l.b -= (self.lr * m_hat_b) / (np.sqrt(v_hat_b) + self.epsilon)
            else:
                # Vanilla SGD
                l.W -= self.lr * l.dW
                l.b -= self.lr * l.db
                
        if self.hook_backward:
            self.hook_backward(last_acts, last_deltas, self.get_weights_snapshot())
            
        return total_loss / batch_size, correct / batch_size

    def get_weights_snapshot(self):
        return [{
            'layer_index': l.layer_idx,
            'input_size': l.in_size,
            'output_size': l.out_size,
            'activation': l.activation_name,
            'weights': l.W.tolist(),
            'biases': l.b.tolist()
        } for l in self.layers]
