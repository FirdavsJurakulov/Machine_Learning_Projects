"""Train the from-scratch network and sklearn's MLPClassifier on MNIST under the same settings and compare them."""
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from activations import ReLU, Softmax
from layers import Dense
from losses import CategoricalCrossEntropy
from neural_network import NeuralNetwork
from utils import accuracy, one_hot

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1

rng = np.random.default_rng(42)

# ----------------- DATA -----------------

X, labels = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", return_X_y=True)
X = X / 255.0
labels = labels.astype(int)

X_train, X_test = X[:60000], X[60000:]
labels_train, labels_test = labels[:60000], labels[60000:]
y_train = one_hot(labels_train, 10)

# ----------------- FROM SCRATCH -----------------

nn = NeuralNetwork(
    layers=[Dense(784, 128, rng), ReLU(), Dense(128, 10, rng), Softmax()],
    loss=CategoricalCrossEntropy(),
    learning_rate=LEARNING_RATE,
)

scratch_acc, scratch_time = [], 0.0
for epoch in range(EPOCHS):
    start = time.perf_counter()
    nn.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, rng=rng)
    scratch_time += time.perf_counter() - start
    scratch_acc.append(accuracy(nn.predict(X_test), labels_test))
    print(f"scratch  epoch {epoch + 1:>2}  test accuracy {scratch_acc[-1]:.2%}")

# ----------------- SKLEARN, SAME SETTINGS -----------------

# plain mini-batch SGD with no momentum and no L2 penalty, so the only differences are the
# weight initialization (sklearn uses Glorot, we use He) and the implementation itself
sk = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation="relu",
    solver="sgd",
    learning_rate="constant",
    learning_rate_init=LEARNING_RATE,
    momentum=0.0,
    alpha=0.0,
    batch_size=BATCH_SIZE,
    max_iter=1,
    shuffle=True,
    random_state=42,
)

sk_acc, sk_time = [], 0.0
for epoch in range(EPOCHS):
    start = time.perf_counter()
    sk.partial_fit(X_train, labels_train, classes=np.arange(10))  # partial_fit = exactly one epoch
    sk_time += time.perf_counter() - start
    sk_acc.append(sk.score(X_test, labels_test))
    print(f"sklearn  epoch {epoch + 1:>2}  test accuracy {sk_acc[-1]:.2%}")

# ----------------- SKLEARN, DEFAULTS (ADAM) -----------------

# what most people get by typing MLPClassifier(hidden_layer_sizes=(128,)): Adam optimizer, L2 penalty
sk_adam = MLPClassifier(hidden_layer_sizes=(128,), max_iter=EPOCHS, random_state=42)
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)  # 10 epochs is deliberately short
    sk_adam.fit(X_train, labels_train)
adam_time = time.perf_counter() - start
adam_acc = sk_adam.score(X_test, labels_test)

# ----------------- RESULTS -----------------

scratch_pred = np.argmax(nn.predict(X_test), axis=1)
sk_pred = sk.predict(X_test)
both_wrong = np.sum((scratch_pred != labels_test) & (sk_pred != labels_test))

print("\n" + "-" * 58)
print(f"{'model':<28}{'test accuracy':>15}{'train time':>15}")
print("-" * 58)
print(f"{'from scratch (NumPy)':<28}{scratch_acc[-1]:>15.2%}{scratch_time:>14.1f}s")
print(f"{'sklearn, same settings':<28}{sk_acc[-1]:>15.2%}{sk_time:>14.1f}s")
print(f"{'sklearn, defaults (Adam)':<28}{adam_acc:>15.2%}{adam_time:>14.1f}s")
print("-" * 58)
print(f"predictions agree on {np.mean(scratch_pred == sk_pred):.2%} of the 10,000 test digits")
print(f"digits both models get wrong: {both_wrong}")

fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))
epochs = range(1, EPOCHS + 1)
ax_curve.plot(epochs, scratch_acc, marker="o", label="from scratch (NumPy)")
ax_curve.plot(epochs, sk_acc, marker="o", label="sklearn, same settings")
ax_curve.set(xlabel="Epoch", ylabel="Test accuracy", title="Accuracy on unseen digits")
ax_curve.legend()

names = ["from scratch", "sklearn\n(same settings)", "sklearn\n(Adam defaults)"]
accs = [scratch_acc[-1], sk_acc[-1], adam_acc]
bars = ax_bar.bar(names, accs, color=["tab:blue", "tab:orange", "tab:gray"])
ax_bar.bar_label(bars, labels=[f"{a:.2%}" for a in accs])
ax_bar.set(ylim=(min(accs) - 0.02, 1.0), ylabel="Test accuracy", title=f"After {EPOCHS} epochs")
plt.tight_layout()
plt.show()

# digits where the two models disagree: the interesting cases
disagree = np.flatnonzero(scratch_pred != sk_pred)[:10]
if len(disagree):
    fig, axes = plt.subplots(1, len(disagree), figsize=(1.2 * len(disagree), 2.2), squeeze=False)
    for ax, i in zip(axes[0], disagree):
        ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"true {labels_test[i]}\nus {scratch_pred[i]} | sk {sk_pred[i]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Digits where the models disagree")
    plt.tight_layout()
    plt.show()
