from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from gaussian_naive_bayes import Gaussian_Naive_Bayes
from sklearn.naive_bayes import GaussianNB
import numpy as np

features, labels = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=0)

custom_gnb = Gaussian_Naive_Bayes(X_train, y_train, 1e-9)
custom_gnb.fit()
predictions = custom_gnb.predict(X_test)

gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train,)
sk_preds = gnb.predict(X_test)


def report(name, y_true, preds):
    acc = np.mean(preds == y_true)
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="macro")
    print(f"{name:>10} | acc={acc:.4f}  precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")

report("Custom", y_test, predictions)
report("sklearn", y_test, sk_preds)

mislabeled = 0
for pred_label, true_label in zip(y_test, predictions):
    if pred_label != true_label:
        mislabeled += 1
print(f"Mislabeled: {mislabeled}/75")