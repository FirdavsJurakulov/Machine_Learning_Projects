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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, log_every=0) -> list:
        loss_history = []

        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.loss.forward(y_hat, y)
            loss_history.append(loss)

            self.backward()
            self.step()

            if log_every and epoch % log_every == 0:
                print(f"epoch {epoch:>5}  loss {loss:.6f}")

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
