import numpy as np
from helpers import build_tree, traverse_tree, print_tree, gini, majority_class, mse, mean_result

class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.root = build_tree(X, y, criterion=gini, leaf_value_func=majority_class, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

        return self

    def predict(self, X):
        X = np.asarray(X)

        return np.array([traverse_tree(x, self.root) for x in X])
    def print_tree(self, feature_names=None, class_names=None, **kwargs):
        if feature_names is None:
            feature_names = self.feature_names_
        return print_tree(
            self.root,
            feature_names=feature_names,
            class_names=class_names,
            is_classifier=True,
            impurity_name="gini",
            **kwargs,
        )

    def _print_tree(self, feature_names=None, class_names=None, **kwargs):
        return self.print_tree(feature_names, class_names, **kwargs)

class CustomDecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X)
        y = np.asarray(y)

        self.root = build_tree(X, y, criterion=mse, leaf_value_func=mean_result, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

        return self
    def predict(self, X):
        X = np.asarray(X)

        return np.array([traverse_tree(x, self.root) for x in X])
    def print_tree(self, feature_names=None, **kwargs):
        if feature_names is None:
            feature_names = self.feature_names_
        return print_tree(
            self.root,
            feature_names=feature_names,
            is_classifier=False,
            impurity_name="squared_error",
            **kwargs,
        )

    def _print_tree(self, feature_names=None, **kwargs):
        return self.print_tree(feature_names, **kwargs)
