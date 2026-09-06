import numpy as np
from dataclasses import dataclass

@dataclass
class Gaussian_Naive_Bayes:
    features: np.ndarray
    labels: np.ndarray
    var_smoothing: float

    def fit(self) -> None:
        """Fits Gaussian Naive Bayes Model"""
        self.unique_labels = np.unique(self.labels)
        self.params = []

        for label in self.unique_labels:
            label_features = self.features[self.labels == label]
            self.params.append([ (col.mean(), col.var()) for col in label_features.T])

    def likelihood(self, data: float, mean: float, var: float, eps=1e-4) -> float:

        #eps - a small value added to denominator to prevent divisions by zero

        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(- ((data - mean) ** 2 / (2 * var + eps)))

        return coeff * exponent

        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Performs inference using Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)"""
        num_samples, _ = features.shape

        predictions = np.empty(num_samples)
        for idx, feature in enumerate(features):
            posteriors = []
            for label_idx, label in enumerate(self.unique_labels):
                prior = (self.labels == label).mean()

                likelihood = np.prod([self.likelihood(f, m, v, self.var_smoothing) for f, (m, v) in zip(feature, self.params[label_idx])])
                posteriors.append(prior * likelihood)
            predictions[idx] = self.unique_labels[np.argmax(posteriors)]

        return predictions