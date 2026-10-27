"""Polynomial regression with transparent, float64 NumPy reference implementations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty finite 1-D numeric array")
    return a


def _check_degree(degree: int) -> None:
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)):
        raise ValueError("degree must be a nonnegative integer")
    if degree < 0:
        raise ValueError("degree must be a nonnegative integer")


def design_matrix(x: ArrayLike, degree: int) -> NDArray[np.float64]:
    """N by (degree+1) Vandermonde design with ascending powers and intercept."""
    _check_degree(degree)
    return np.vander(_vector(x, "x"), N=int(degree) + 1, increasing=True)


def _training_data(x: ArrayLike, t: ArrayLike, degree: int):
    phi = design_matrix(x, degree)
    target = _vector(t, "t")
    if len(phi) != len(target):
        raise ValueError("x and t must have the same number of observations")
    return phi, target


def fit_least_squares(x: ArrayLike, t: ArrayLike, degree: int) -> NDArray[np.float64]:
    """Compute least-squares coefficients without explicitly inverting X'X."""
    phi, target = _training_data(x, t, degree)
    return np.linalg.lstsq(phi, target, rcond=None)[0]


def fit_ridge(
    x: ArrayLike, t: ArrayLike, degree: int, penalty: float
) -> NDArray[np.float64]:
    """L2-penalized least squares; unlike the Bayesian prior below, intercept is free."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    phi, target = _training_data(x, t, degree)
    regularizer = np.eye(phi.shape[1])
    regularizer[0, 0] = 0.0
    design = np.vstack((phi, np.sqrt(penalty) * regularizer))
    y = np.concatenate((target, np.zeros(phi.shape[1])))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def predict(x: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
    """Continuous predictions for coefficients ordered [intercept, x, x**2, ...]."""
    w = _vector(weights, "weights")
    return design_matrix(x, len(w) - 1) @ w


@dataclass(frozen=True)
class BayesianPosterior:
    """Gaussian weight posterior: alpha and beta represent precision parameters."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    beta: float


def fit_bayesian(
    x: ArrayLike, t: ArrayLike, degree: int, alpha: float, beta: float
) -> BayesianPosterior:
    """Gaussian prior N(0, alpha^-1 I), Gaussian noise precision beta."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    phi, target = _training_data(x, t, degree)
    k = phi.shape[1]
    precision = alpha * np.eye(k) + beta * phi.T @ phi
    covariance = np.linalg.solve(precision, np.eye(k))
    covariance = (covariance + covariance.T) / 2
    mean = np.linalg.solve(precision, beta * phi.T @ target)
    return BayesianPosterior(mean=mean, covariance=covariance, beta=float(beta))


def predict_bayesian(
    x: ArrayLike, posterior: BayesianPosterior, include_noise: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return marginal Gaussian predictive mean and variance for each query."""
    phi = design_matrix(x, len(posterior.mean) - 1)
    mean = phi @ posterior.mean
    var = np.einsum("ni,ij,nj->n", phi, posterior.covariance, phi)
    var = np.maximum(var, 0.0)  # Counter tiny negative floating-point roundoff.
    if include_noise:
        var += 1.0 / posterior.beta
    return mean, var


def mse_and_gradient(
    x: ArrayLike, t: ArrayLike, weights: ArrayLike
) -> tuple[float, NDArray[np.float64]]:
    """Mean squared error and its exact analytic gradient."""
    w = _vector(weights, "weights")
    phi, target = _training_data(x, t, len(w) - 1)
    residual = phi @ w - target
    return float(np.mean(residual**2)), 2.0 / len(target) * phi.T @ residual
