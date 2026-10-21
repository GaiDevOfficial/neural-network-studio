"""
Automated Verification Suite for Python Reference Engine
Tests Convergence on XOR, Iris, and Linear Regression
Neural Network Studio Platform
"""

import unittest
import numpy as np
from python_core.neural_engine import NeuralNetworkEngine
from python_core.datasets import get_xor_dataset, get_iris_dataset, get_linear_dataset

class TestNeuralEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_xor_convergence(self):
        """Verify network solves non-linear XOR problem with > 90% accuracy"""
        X, y = get_xor_dataset(count=120, noise=0.04)
        arch = [
            {'size': 2},
            {'size': 8, 'activation': 'relu'},
            {'size': 6, 'activation': 'relu'},
            {'size': 2, 'activation': 'softmax'}
        ]
        
        net = NeuralNetworkEngine(arch, learning_rate=0.08, loss='cce', optimizer='adam')
        
        # Train for 150 epochs
        initial_loss, _ = net.train_batch(X, y)
        for epoch in range(180):
            loss, acc = net.train_batch(X, y)
            
        final_preds = net.predict(X)
        pred_classes = np.argmax(final_preds, axis=1)
        true_classes = np.argmax(y, axis=1)
        final_acc = np.mean(pred_classes == true_classes)
        
        print(f"\n[Python Test] XOR Final Accuracy: {final_acc * 100:.1f}%, Final Loss: {loss:.4f}")
        self.assertGreater(final_acc, 0.90, "XOR failed to achieve >= 90% convergence")
        self.assertLess(loss, initial_loss, "XOR loss did not decrease")

    def test_iris_convergence(self):
        """Verify multiclass convergence on Fisher's Iris dataset"""
        X, y = get_iris_dataset()
        arch = [
            {'size': 4},
            {'size': 8, 'activation': 'relu'},
            {'size': 3, 'activation': 'softmax'}
        ]
        
        net = NeuralNetworkEngine(arch, learning_rate=0.05, loss='cce', optimizer='adam')
        for epoch in range(150):
            loss, acc = net.train_batch(X, y)
            
        final_preds = net.predict(X)
        pred_classes = np.argmax(final_preds, axis=1)
        true_classes = np.argmax(y, axis=1)
        final_acc = np.mean(pred_classes == true_classes)
        
        print(f"[Python Test] Iris Final Accuracy: {final_acc * 100:.1f}%, Final Loss: {loss:.4f}")
        self.assertGreater(final_acc, 0.85, "Iris failed to achieve >= 85% convergence")

    def test_linear_regression_convergence(self):
        """Verify continuous function approximation on regression dataset"""
        X, y = get_linear_dataset(count=100)
        arch = [
            {'size': 1},
            {'size': 12, 'activation': 'tanh'},
            {'size': 8, 'activation': 'tanh'},
            {'size': 1, 'activation': 'linear'}
        ]
        
        net = NeuralNetworkEngine(arch, learning_rate=0.04, loss='mse', optimizer='adam')
        initial_loss, _ = net.train_batch(X, y)
        for epoch in range(150):
            loss, _ = net.train_batch(X, y)
            
        print(f"[Python Test] Regression Initial Loss: {initial_loss:.4f} -> Final Loss: {loss:.4f}")
        self.assertLess(loss, initial_loss * 0.35, "Regression loss did not reduce sufficiently")

    def test_hook_emission(self):
        """Verify step hooks capture step-by-step activations and gradients"""
        captured_activations = []
        captured_weights = []
        
        X, y = get_xor_dataset(count=10)
        arch = [{'size': 2}, {'size': 4, 'activation': 'relu'}, {'size': 2, 'activation': 'softmax'}]
        net = NeuralNetworkEngine(arch, learning_rate=0.05)
        
        net.hook_forward = lambda acts, zs: captured_activations.append(acts)
        net.hook_backward = lambda acts, deltas, weights: captured_weights.append(weights)
        
        net.train_batch(X, y)
        
        self.assertGreater(len(captured_activations), 0, "Forward hook did not fire")
        self.assertGreater(len(captured_weights), 0, "Backward hook did not fire")
        self.assertEqual(len(captured_weights[0]), 2, "Weight snapshot layer count mismatch")

if __name__ == '__main__':
    unittest.main()
