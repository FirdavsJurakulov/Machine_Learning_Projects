import numpy as np


class NeuralNetwork:
    def __init__(self, layers, loss, learning_rate=0.1):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self) -> None:
        # chain rule: pass the gradient back through the layers in reverse order
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self) -> None:
        # gradient descent: move every parameter against its gradient
        for layer in self.layers:
            for name in layer.params:
                layer.params[name] -= self.learning_rate * layer.grads[name]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size=None, log_every=0, rng=None) -> list:
        """Train with mini-batch gradient descent. batch_size=None uses the whole dataset per step.
        Returns the average loss of each epoch."""
        rng = rng if rng is not None else np.random.default_rng()
        m = X.shape[0]
        batch_size = batch_size or m
        loss_history = []

        for epoch in range(epochs):
            # shuffle every epoch so batches differ and the model can't learn the order
            order = rng.permutation(m)
            epoch_loss = 0.0

            for start in range(0, m, batch_size):
                idx = order[start:start + batch_size]
                X_batch, y_batch = X[idx], y[idx]

                y_hat = self.forward(X_batch)
                epoch_loss += self.loss.forward(y_hat, y_batch) * len(idx)

                self.backward()
                self.step()

            loss_history.append(epoch_loss / m)

            if log_every and epoch % log_every == 0:
                print(f"epoch {epoch:>5}  loss {loss_history[-1]:.6f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
