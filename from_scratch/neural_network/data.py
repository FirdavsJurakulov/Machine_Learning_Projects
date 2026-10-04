import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

nn = NeuralNetwork(0.1)
hidden_units = 5

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
    [0]
])

num_epochs = 10000

input_dim = X.shape[1]
output_dim = y.shape[1]

# np.random.seed(42)

W1 = np.random.randn(
    input_dim,
    hidden_units
) * np.sqrt(2 / input_dim)

W2 = np.random.randn(
    hidden_units,
    output_dim
) * np.sqrt(2 / hidden_units)

b1 = np.zeros((1, hidden_units))
b2 = np.zeros((1, output_dim))

# ----------------- TRAINING LOOP -----------------

loss_history = []

for epoch in range(num_epochs):
    Z1, A1, Z2, y_hat = nn.forward(X, W1, b1, W2, b2)

    loss = nn.binary_cross_entropy(y_hat, y)
    loss_history.append(loss)

    dW1, db1, dW2, db2 = nn.backward(
        X, y, W1, W2, Z1, A1, y_hat
    )

    W1 = nn.update(W1, dW1)
    b1 = nn.update(b1, db1)
    W2 = nn.update(W2, dW2)
    b2 = nn.update(b2, db2)

    if epoch % 100 == 0:
        print(epoch, loss)

print(Z1, A1, Z2, y_hat, dW1, db1, dW2, db2)
print("----------------")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# # ----------------- VISUALIZATION -----------------
# # 1. Plot the original ground-truth data points
# plt.scatter(X[:, 0], X[:, 1], s=100, c=y.ravel())

# # 2. Generate a dense line of inputs across the domain of X (from 1 to 4)
# X_line = np.linspace(1, 4, 100).reshape(-1, 1)

# # 3. Get the network's predictions for this dense set of inputs
# _, _, _, y_line_pred = nn.forward(X_line, W1, b1, W2, b2)

# # 4. Plot the fitted curve as a line
# plt.plot(X_line, y_line_pred, c='red', linewidth=2, label='NN Fit Line')

# # 5. Add labels, legend, and show the plot
# plt.title('Neural Network Fit vs Actual Data')
# plt.suptitle(f"Final Weights and Biases: {W1, b1, W2, b2}")
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

Z1, A1, Z2, y_hat = nn.forward(
    X, W1, b1, W2, b2
)

predictions = (y_hat >= 0.5).astype(int)

print("Predictions:")
print(y_hat)

print("Predicted classes:")
print(predictions)

print("True labels:")
print(y)
