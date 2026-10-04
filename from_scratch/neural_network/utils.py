import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """[3, 0] -> [[0,0,0,1,...], [1,0,0,0,...]]"""
    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1
    return encoded


def accuracy(y_hat: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows where the most probable class equals the true label."""
    return np.mean(np.argmax(y_hat, axis=1) == labels)
