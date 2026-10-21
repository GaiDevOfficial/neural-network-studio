"""
Benchmark Datasets Generator for Python Reference Engine
Neural Network Studio Platform
"""

import numpy as np

def get_xor_dataset(count=120, noise=0.05):
    X = []
    y = []
    base = [
        ([-0.8, -0.8], [1.0, 0.0]),
        ([0.8, 0.8], [1.0, 0.0]),
        ([-0.8, 0.8], [0.0, 1.0]),
        ([0.8, -0.8], [0.0, 1.0])
    ]
    for pt, lbl in base:
        X.append(pt)
        y.append(lbl)
        
    per_cluster = (count - 4) // 4
    for pt, lbl in base:
        for _ in range(per_cluster):
            X.append([pt[0] + np.random.uniform(-noise, noise), pt[1] + np.random.uniform(-noise, noise)])
            y.append(lbl)
            
    return np.array(X), np.array(y)

def get_iris_dataset():
    raw_data = [
        [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0], [4.6, 3.1, 1.5, 0.2, 0],
        [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0], [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0],
        [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1], [5.5, 2.3, 4.0, 1.3, 1],
        [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1], [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1],
        [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2], [6.3, 2.9, 5.6, 1.8, 2],
        [6.5, 3.0, 5.8, 2.2, 2], [7.6, 3.0, 6.6, 2.1, 2], [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2]
    ]
    raw_data = np.array(raw_data)
    X = raw_data[:, :4]
    # Min-max normalize to [-1, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = ((X - X_min) / (X_max - X_min)) * 2.0 - 1.0
    
    labels = raw_data[:, 4].astype(int)
    y = np.zeros((len(labels), 3))
    for i, lbl in enumerate(labels):
        y[i, lbl] = 1.0
        
    return X, y

def get_linear_dataset(count=100):
    x = np.random.uniform(-1.0, 1.0, (count, 1))
    y = 0.5 * np.sin(np.pi * x) + 0.3 * (x ** 2)
    return x, y
