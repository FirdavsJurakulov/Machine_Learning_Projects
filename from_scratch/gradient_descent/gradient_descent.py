from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
ObjectiveFunction = Callable[[Array], float]
GradientFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class OptimizationResult:
    """
    Result returned by gradient_descent().
    """

    parameters: Array
    loss: float
    gradient: Array
    gradient_norm: float

    iterations: int
    converged: bool
    stopping_reason: str

    parameter_history: Array
    loss_history: Array
    gradient_history: Array
    gradient_norm_history: Array


def _validate_optimizer_arguments(
    initial_parameters: Array,
    learning_rate: float,
    max_iterations: int,
    gradient_tolerance: float | None,
    parameter_tolerance: float | None,
    loss_tolerance: float | None,
) -> Array:
    """
    Validate optimizer inputs and return a float64 parameter vector.
    """

    parameters = np.asarray(
        initial_parameters,
        dtype=np.float64,
    ).copy()

    if parameters.ndim != 1:
        raise ValueError(
            "initial_parameters must be a one-dimensional array."
        )

    if parameters.size == 0:
        raise ValueError(
            "initial_parameters cannot be empty."
        )

    if not np.all(np.isfinite(parameters)):
        raise ValueError(
            "initial_parameters contains NaN or infinite values."
        )

    if not np.isfinite(learning_rate):
        raise ValueError(
            "learning_rate must be finite."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if not isinstance(max_iterations, int):
        raise TypeError(
            "max_iterations must be an integer."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    tolerances = {
        "gradient_tolerance": gradient_tolerance,
        "parameter_tolerance": parameter_tolerance,
        "loss_tolerance": loss_tolerance,
    }

    for name, value in tolerances.items():
        if value is None:
            continue

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    if all(value is None for value in tolerances.values()):
        raise ValueError(
            "At least one stopping tolerance must be provided."
        )

    return parameters


def _evaluate_objective_and_gradient(
    objective: ObjectiveFunction,
    gradient: GradientFunction,
    parameters: Array,
) -> tuple[float, Array]:
    """
    Evaluate and validate the objective and gradient.
    """

    loss = float(objective(parameters))

    gradient_value = np.asarray(
        gradient(parameters),
        dtype=np.float64,
    )

    if not np.isfinite(loss):
        raise FloatingPointError(
            "Objective returned NaN or infinity."
        )

    if gradient_value.shape != parameters.shape:
        raise ValueError(
            "Gradient shape must match parameter shape. "
            f"Expected {parameters.shape}, "
            f"received {gradient_value.shape}."
        )

    if not np.all(np.isfinite(gradient_value)):
        raise FloatingPointError(
            "Gradient returned NaN or infinite values."
        )

    return loss, gradient_value


def gradient_descent(
    objective: ObjectiveFunction,
    gradient: GradientFunction,
    initial_parameters: Array,
    learning_rate: float = 0.1,
    max_iterations: int = 10_000,
    gradient_tolerance: float | None = 1e-6,
    parameter_tolerance: float | None = None,
    loss_tolerance: float | None = None,
) -> OptimizationResult:
    """
    Minimize an objective function using batch gradient descent.

    Parameters
    ----------
    objective:
        Function that accepts a parameter vector and returns
        one scalar loss.

    gradient:
        Function that accepts the same parameter vector and
        returns an array with the same shape.

    initial_parameters:
        One-dimensional initial parameter vector.

    learning_rate:
        Gradient-descent step size.

    max_iterations:
        Maximum number of parameter updates.

    gradient_tolerance:
        Stop when the gradient norm is below this value.

    parameter_tolerance:
        Stop when the parameter update norm is below this value.

    loss_tolerance:
        Stop when the absolute loss change is below this value.

    Returns
    -------
    OptimizationResult
        Final parameters, convergence information, and history.
    """

    parameters = _validate_optimizer_arguments(
        initial_parameters=initial_parameters,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        parameter_tolerance=parameter_tolerance,
        loss_tolerance=loss_tolerance,
    )

    parameter_history: list[Array] = []
    loss_history: list[float] = []
    gradient_history: list[Array] = []
    gradient_norm_history: list[float] = []

    loss, gradient_value = _evaluate_objective_and_gradient(
        objective=objective,
        gradient=gradient,
        parameters=parameters,
    )

    def record_state() -> None:
        gradient_norm = float(
            np.linalg.norm(gradient_value)
        )

        parameter_history.append(
            parameters.copy()
        )

        loss_history.append(loss)

        gradient_history.append(
            gradient_value.copy()
        )

        gradient_norm_history.append(
            gradient_norm
        )

    # Record the initial state before any updates.
    record_state()

    converged = False
    stopping_reason = "maximum_iterations"
    updates_completed = 0

    for _ in range(max_iterations):
        current_gradient_norm = float(
            np.linalg.norm(gradient_value)
        )

        # Check gradient at the current parameter location.
        if (
            gradient_tolerance is not None
            and current_gradient_norm <= gradient_tolerance
        ):
            converged = True
            stopping_reason = "gradient_tolerance"
            break

        new_parameters = (
            parameters
            - learning_rate * gradient_value
        )

        if not np.all(np.isfinite(new_parameters)):
            raise FloatingPointError(
                "Parameter update produced NaN or infinity. "
                "The learning rate may be too large."
            )

        new_loss, new_gradient = (
            _evaluate_objective_and_gradient(
                objective=objective,
                gradient=gradient,
                parameters=new_parameters,
            )
        )

        parameter_change = float(
            np.linalg.norm(
                new_parameters - parameters
            )
        )

        loss_change = abs(
            new_loss - loss
        )

        parameters = new_parameters
        loss = new_loss
        gradient_value = new_gradient

        updates_completed += 1
        record_state()

        if (
            parameter_tolerance is not None
            and parameter_change <= parameter_tolerance
        ):
            converged = True
            stopping_reason = "parameter_tolerance"
            break

        if (
            loss_tolerance is not None
            and loss_change <= loss_tolerance
        ):
            converged = True
            stopping_reason = "loss_tolerance"
            break

    final_gradient_norm = float(
        np.linalg.norm(gradient_value)
    )

    return OptimizationResult(
        parameters=parameters.copy(),
        loss=loss,
        gradient=gradient_value.copy(),
        gradient_norm=final_gradient_norm,
        iterations=updates_completed,
        converged=converged,
        stopping_reason=stopping_reason,
        parameter_history=np.asarray(
            parameter_history,
            dtype=np.float64,
        ),
        loss_history=np.asarray(
            loss_history,
            dtype=np.float64,
        ),
        gradient_history=np.asarray(
            gradient_history,
            dtype=np.float64,
        ),
        gradient_norm_history=np.asarray(
            gradient_norm_history,
            dtype=np.float64,
        ),
    )