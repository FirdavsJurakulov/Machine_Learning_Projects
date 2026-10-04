import numpy as np


class Activation:
    """An activation is a layer with no parameters: it only transforms its input."""

    params = {}
    grads = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return grad * np.where(self.x > 0, 1.0, self.alpha)


class Sigmoid(Activation):
    def forward(self, x):
        # exp(-|x|) never overflows, so this is stable for large positive and negative x
        e = np.exp(-np.abs(x))
        self.out = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Softmax(Activation):
    def forward(self, x):
        # subtracting the row max doesn't change the result but prevents overflow
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exp / np.sum(exp, axis=-1, keepdims=True)
        return self.out

    def backward(self, grad):
        # Jacobian-vector product of softmax: s * (g - sum(g * s))
        s = self.out
        return s * (grad - np.sum(grad * s, axis=-1, keepdims=True))
