import numpy as np
from activations import Activations
from dataclasses import dataclass

act = Activations()

reLU = act.reLU
leaky_relu = act.leaky_relu
leaky_relu_der = act.leaky_relu_der
reLU_der = act.reLU_der
sigmoid = act.sigmoid
stable_sigmoid = act.stable_sigmoid

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def forward(self, X, W1, b1, W2, b2):
        Z1 = X @ W1 + b1
        A1 = leaky_relu(Z1)

        Z2 = A1 @ W2 + b2
        y_hat = stable_sigmoid(Z2)

        return Z1, A1, Z2, y_hat
    def backward(self, X, y, W1, W2, Z1, A1, y_hat):
        m = X.shape[0]
        dZ2 = (y_hat - y) / m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * leaky_relu_der(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2
    def binary_cross_entropy(self, y_hat, y, epsilon=1e-15):
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        loss = -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

        return np.mean(loss)
    def mean_squared_error(self, y_hat, y, n):
        return (y_hat - y)**2/(2*n)
    def update(self, parameter, gradient):
        parameter -= self.learning_rate * gradient

        return parameter