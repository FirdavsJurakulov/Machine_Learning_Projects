import numpy as np


class Dense:
    """Fully connected layer: out = x @ W + b"""

    def __init__(self, input_dim: int, output_dim: int, rng=None):
        rng = rng if rng is not None else np.random.default_rng()

        # He initialization: keeps activation variance stable through ReLU-like layers
        self.params = {
            "W": rng.standard_normal((input_dim, output_dim)) * np.sqrt(2 / input_dim),
            "b": np.zeros((1, output_dim)),
        }
        self.grads = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  # cached for the backward pass
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad is dL/d(out); store dL/dW and dL/db, return dL/dx for the previous layer
        self.grads["W"] = self.x.T @ grad
        self.grads["b"] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params["W"].T
