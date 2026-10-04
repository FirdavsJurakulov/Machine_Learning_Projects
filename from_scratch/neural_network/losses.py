import numpy as np


class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        # clip so log(0) never happens
        self.y_hat = np.clip(y_hat, self.epsilon, 1 - self.epsilon)
        self.y = y
        loss = -(y * np.log(self.y_hat) + (1 - y) * np.log(1 - self.y_hat))
        return np.mean(loss)

    def backward(self) -> np.ndarray:
        # dL/dy_hat. Followed by Sigmoid.backward (which multiplies by y_hat * (1 - y_hat)),
        # the denominator cancels and the gradient w.r.t. the logits becomes (y_hat - y) / m.
        m = self.y.shape[0]
        return (self.y_hat - self.y) / (self.y_hat * (1 - self.y_hat)) / m


class CategoricalCrossEntropy:
    """Multi-class loss. Expects y one-hot encoded and y_hat to be softmax probabilities."""

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        self.y_hat = np.clip(y_hat, self.epsilon, 1 - self.epsilon)
        self.y = y
        # only the log-probability of the correct class survives the sum
        return np.mean(-np.sum(y * np.log(self.y_hat), axis=1))

    def backward(self) -> np.ndarray:
        # dL/dy_hat. Followed by Softmax.backward, this simplifies to (y_hat - y) / m,
        # the same shape of gradient as sigmoid + binary cross-entropy.
        m = self.y.shape[0]
        return -self.y / self.y_hat / m


class MeanSquaredError:
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        self.y_hat = y_hat
        self.y = y
        return np.mean((y_hat - y) ** 2) / 2

    def backward(self) -> np.ndarray:
        return (self.y_hat - self.y) / self.y_hat.size
