import numpy as np
import matplotlib.pyplot as plt

from activations import LeakyReLU, Sigmoid
from layers import Dense
from losses import BinaryCrossEntropy
from neural_network import NeuralNetwork

rng = np.random.default_rng(42)

# XOR: not linearly separable, so it needs a hidden layer
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
])

hidden_units = 5

nn = NeuralNetwork(
    layers=[
        Dense(X.shape[1], hidden_units, rng),
        LeakyReLU(),
        Dense(hidden_units, y.shape[1], rng),
        Sigmoid(),
    ],
    loss=BinaryCrossEntropy(),
    learning_rate=0.1,
)

loss_history = nn.fit(X, y, epochs=10000, log_every=1000)

y_hat = nn.predict(X)
predictions = (y_hat >= 0.5).astype(int)

print("\nProbabilities:\n", y_hat.round(4))
print("Predicted classes:", predictions.ravel())
print("True labels:      ", y.ravel())

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss on XOR")
plt.show()
