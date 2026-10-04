"""Convert the trained model (mnist_model.npz, written by mnist.py) into web/model.js for the browser demo."""
import base64
import json

import numpy as np
from sklearn.datasets import fetch_openml

from utils import accuracy

NUM_SAMPLES = 30


def to_base64(array: np.ndarray, dtype) -> str:
    # raw bytes are much smaller than a JSON list of numbers; JS reads them back with a typed array
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


model = np.load("mnist_model.npz")

X, labels = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", return_X_y=True)
X_test, labels_test = X[60000:] / 255.0, labels[60000:].astype(int)

# rebuild the forward pass from the saved arrays, both to export it and to measure test accuracy
layers = []
out = X_test
for i, kind in enumerate(model["layers"]):
    kind = str(kind)
    layer = {"type": kind}

    if kind == "Dense":
        W, b = model[f"{i}_W"], model[f"{i}_b"]
        layer.update(inputs=W.shape[0], outputs=W.shape[1], W=to_base64(W, np.float32), b=to_base64(b, np.float32))
        out = out @ W + b
    elif kind == "ReLU":
        out = np.maximum(0, out)
    elif kind == "Softmax":
        exp = np.exp(out - out.max(axis=1, keepdims=True))
        out = exp / exp.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"export_web.py doesn't know how to export a {kind} layer")

    layers.append(layer)

test_accuracy = accuracy(out, labels_test)

# a few real test digits so the page can show what MNIST looks like
samples = (X_test[:NUM_SAMPLES] * 255).round()

export = {
    "layers": layers,
    "testAccuracy": round(float(test_accuracy), 4),
    "samples": {"pixels": to_base64(samples, np.uint8), "labels": labels_test[:NUM_SAMPLES].tolist()},
}

with open("web/model.js", "w") as f:
    f.write("window.MNIST_MODEL = " + json.dumps(export) + ";\n")

print(f"wrote web/model.js  ({' -> '.join(l['type'] for l in layers)}, test accuracy {test_accuracy:.2%})")
