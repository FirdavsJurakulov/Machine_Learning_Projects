import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from activations import ReLU, Softmax
from layers import Dense
from losses import CategoricalCrossEntropy
from neural_network import NeuralNetwork
from utils import accuracy, one_hot

rng = np.random.default_rng(42)

# ----------------- DATA -----------------

# 70,000 handwritten digits, 28x28 pixels flattened to 784 values (downloaded once, then cached)
X, labels = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", return_X_y=True)

X = X / 255.0  # pixels 0..255 -> 0..1, so the initial weighted sums stay small
labels = labels.astype(int)

# standard split: first 60,000 for training, last 10,000 for testing
X_train, X_test = X[:60000], X[60000:]
labels_train, labels_test = labels[:60000], labels[60000:]

y_train = one_hot(labels_train, 10)

# ----------------- MODEL -----------------

nn = NeuralNetwork(
    layers=[
        Dense(784, 128, rng),
        ReLU(),
        Dense(128, 10, rng),
        Softmax(),
    ],
    loss=CategoricalCrossEntropy(),
    learning_rate=0.1,
)

# ----------------- TRAINING -----------------

epochs = 10
loss_history, test_acc_history = [], []

for epoch in range(epochs):
    loss_history += nn.fit(X_train, y_train, epochs=1, batch_size=64, rng=rng)
    test_acc_history.append(accuracy(nn.predict(X_test), labels_test))

    print(f"epoch {epoch + 1:>2}  loss {loss_history[-1]:.4f}  test accuracy {test_acc_history[-1]:.2%}")

# keep the trained weights so the web demo (export_web.py) can use them without retraining
nn.save("mnist_model.npz")

# ----------------- RESULTS -----------------

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
ax_loss.plot(range(1, epochs + 1), loss_history)
ax_loss.set(xlabel="Epoch", ylabel="Training loss", title="Loss")
ax_acc.plot(range(1, epochs + 1), test_acc_history)
ax_acc.set(xlabel="Epoch", ylabel="Test accuracy", title="Accuracy on unseen digits")
plt.tight_layout()
plt.show()

# a few test digits with the network's guess
predictions = np.argmax(nn.predict(X_test[:10]), axis=1)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for ax, image, pred, label in zip(axes, X_test[:10], predictions, labels_test[:10]):
    ax.imshow(image.reshape(28, 28), cmap="gray")
    ax.set_title(f"pred {pred}", color="green" if pred == label else "red")
    ax.axis("off")
plt.show()
