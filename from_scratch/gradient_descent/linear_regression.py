import numpy as np

from gradient_descent import OptimizationResult, gradient_descent

Array = np.ndarray

def _validate_regression_data(X: Array, y: Array | None = None) -> tuple[Array, Array | None]:
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional array of shape [n_samples, n_features]."
        )
    if X.shape[0] == 0:
        raise ValueError(
            "samples cannot be empty."
        )
    if X.shape[1] == 0:
        raise ValueError(
            "features cannot be empty."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains too NaN of infinite values"
        )

    if y is None:
        return X, None

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if y.ndim != 1:
        raise ValueError(
            "y must be a one-dimensional array."
        )

    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y contains NaN or infinite values."
        )

    return X, y

def unpack_parameters(parameters: Array) -> tuple[Array, float]:
    """
    Separate the weight vector and bias.

    Parameter layout:
        [weight_1, weight_2, ..., weight_n, bias]
    """

    parameters = np.asarray(parameters, dtype=np.float64)

    if parameters.ndim != 1:
        raise ValueError(
            "parameters must be 1-dimensional."
        )
    if parameters.size < 2:
        raise ValueError(
            "parameters must contain at least one weight and 1 bias"
        )
    weights = parameters[:-1]
    bias = parameters[-1]

    return weights, bias

def predict_from_parameters(
    X: Array,
    parameters: Array
) -> Array:
    weights, bias = unpack_parameters(parameters)

    if X.shape[1] != weights.shape[0]:
        raise ValueError(
            "number of features in X does not match the number of weights."
        )

    return X @ weights + bias

def predict_linear(
    X: Array,
    weights: Array,
    bias: float
) -> Array:
    X, _ = _validate_regression_data(X)

    if weights.ndim != 1:
        raise ValueError(
            "weights must be one-dimensional."
        )

    if X.shape[1] != weights.shape[0]:
        raise ValueError(
            "number of features in X does not match the number of weights."
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights contains NaN or infinite values."
        )

    if not np.isfinite(bias):
        raise ValueError(
            "bias must be finite."
        )

    """Return linear-regression predictions."""
    return X @ weights + bias


def mean_squared_error(y_true: Array, y_pred: Array) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be one-dimensional."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "Target arrays cannot be empty."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )

    errors = y_true - y_pred
    
    return float(np.mean(errors ** 2))

def linear_regression_objective(
    parameters: Array,
    X: Array,
    y: Array,
) -> float:
    """
    Mean squared error objective for linear regression.
    """

    predictions = predict_from_parameters(
        X,
        parameters,
    )

    return mean_squared_error(
        y,
        predictions,
    )

def linear_regression_gradient(
    parameters: Array,
    X: Array,
    y: Array,
) -> Array:
    """
    Gradient of MSE with respect to all weights and the bias.
    """

    weights, bias = unpack_parameters(
        parameters
    )

    predictions = X @ weights + bias
    errors = predictions - y

    n_samples = X.shape[0]

    weight_gradient = (
        2.0 / n_samples
    ) * (X.T @ errors)

    bias_gradient = (
        2.0 / n_samples
    ) * np.sum(errors)

    return np.concatenate(
        [
            weight_gradient,
            np.array(
                [bias_gradient],
                dtype=np.float64,
            ),
        ]
    )


class LinearRegressionGD:
    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iterations: int = 10_000,
        gradient_tolerance: float | None = 1e-6,
        parameter_tolerance: float | None = None,
        loss_tolerance: float | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.gradient_tolerance = gradient_tolerance
        self.parameter_tolerance = parameter_tolerance
        self.loss_tolerance = loss_tolerance

        self.coef_: Array | None = None
        self.intercept_: float | None = None
        self.optimization_result_: OptimizationResult | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: Array, y: Array):

        X, y = _validate_regression_data(X, y)

        assert y is not None

        self.n_features_in_ = X.shape[1]

        initial_parameters = np.zeros(self.n_features_in_ + 1, dtype=np.float64)

        def objective(parameters: Array) -> float:
            return linear_regression_objective(parameters, X, y)
        def gradient(parameters: Array) -> Array:
            return linear_regression_gradient(parameters, X, y)

        result = gradient_descent(
            objective=objective, 
            gradient=gradient, 
            initial_parameters=initial_parameters, 
            learning_rate=self.learning_rate, 
            max_iterations=self.max_iterations, 
            gradient_tolerance=self.gradient_tolerance, 
            parameter_tolerance=self.parameter_tolerance, 
            loss_tolerance=self.loss_tolerance
        )

        self.coef_ = result.parameters[:-1].copy()
        self.intercept_ = float(result.parameters[-1])
        self.optimization_result_ = result

        return self
    def predict(
        self,
        X: Array
    ) -> Array:
        self._check_is_fitted()

        X, _ = _validate_regression_data(X)


        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features "
                "than the training data."
            )

        assert self.coef_ is not None
        assert self.intercept_ is not None

        return predict_linear(
            X,
            self.coef_,
            self.intercept_,
        )

    def score(self, X: Array, y:Array) -> float:
        X, y = _validate_regression_data(X, y)

        assert y is not None

        predictions = self.predict(X)

        residual_sum_of_squares = np.sum(
            (y - predictions) ** 2
        )

        total_sum_of_squares = np.sum(
            (y - np.mean(y)) ** 2
        )

        if total_sum_of_squares == 0:
            raise ValueError(
                "R² is undefined when all target values "
                "are identical."
            )

        return float(1 - residual_sum_of_squares / total_sum_of_squares)

    @property
    def loss_history_(self) -> Array:
        "Return recorded training losses"

        self._check_is_fitted()

        assert self.optimization_result_ is not None

        return self.optimization_result_.loss_history

    def _check_is_fitted(self) -> None:
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.optimization_result_ is None
        ):
            raise RuntimeError(
                "The model has not been fitted yet. "
                "Call fit() before predict() or score()."
            )