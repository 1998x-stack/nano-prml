"""Finite discrete probability and numerically robust Bayesian updating (PRML §1.2)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_nonnegative_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative")
    return arr


def normalize_weights(weights: ArrayLike) -> NDArray[np.float64]:
    """Normalize nonnegative finite weights, scaling first to avoid sum overflow."""
    arr = _finite_nonnegative_1d(weights, "weights")
    peak = float(np.max(arr))
    if peak == 0.0:
        raise ValueError("weights must have positive mass")
    scaled = arr / peak
    return scaled / np.sum(scaled)


def logsumexp(log_values: ArrayLike) -> float:
    """Stable log(sum(exp(x))), including all -inf (zero total mass)."""
    v = np.asarray(log_values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or np.any(np.isnan(v)) or np.any(np.isposinf(v)):
        raise ValueError("log_values must be a nonempty 1-D vector without NaN/+inf")
    vmax = float(np.max(v))
    if np.isneginf(vmax):
        return float("-inf")
    return float(vmax + np.log(np.sum(np.exp(v - vmax))))


def bayes_posterior(
    prior: ArrayLike, log_likelihood: ArrayLike
) -> NDArray[np.float64]:
    """Categorical p(H | D) from p(H) and log p(D | H); zeros are allowed.

    The prior is an unnormalized nonnegative weight vector. Log-likelihoods
    may be -inf. At least one hypothesis must have positive joint mass.
    """
    probs = normalize_weights(prior)
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if ll.shape != probs.shape or np.any(np.isnan(ll)) or np.any(np.isposinf(ll)):
        raise ValueError("log_likelihood must match prior and contain no NaN/+inf")
    with np.errstate(divide="ignore"):
        log_joint = np.log(probs) + ll
    evidence = logsumexp(log_joint)
    if np.isneginf(evidence):
        raise ValueError("observed evidence has zero probability under every hypothesis")
    return np.exp(log_joint - evidence)


def joint_to_marginals(joint: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return p(X) and p(Y) from a nonnegative X-by-Y joint table."""
    matrix = np.asarray(joint, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.all(np.isfinite(matrix)):
        raise ValueError("joint must be a nonempty finite 2-D matrix")
    if np.any(matrix < 0):
        raise ValueError("joint must be nonnegative")
    peak = float(np.max(matrix))
    if peak == 0.0:
        raise ValueError("joint must contain positive mass")
    scaled = matrix / peak
    mass = scaled / np.sum(scaled)
    return np.sum(mass, axis=1), np.sum(mass, axis=0)


def condition_on_column(joint: ArrayLike, column: int) -> NDArray[np.float64]:
    """p(X | Y=column) from joint mass, rejecting impossible events."""
    matrix = np.asarray(joint, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.all(np.isfinite(matrix)):
        raise ValueError("joint must be a nonempty finite 2-D matrix")
    if np.any(matrix < 0):
        raise ValueError("joint must be nonnegative")
    if isinstance(column, (bool, np.bool_)) or not isinstance(column, (int, np.integer)):
        raise TypeError("column must be an integer")
    if column < 0 or column >= matrix.shape[1]:
        raise ValueError("column is outside the support")
    return normalize_weights(matrix[:, column])


def binary_positive_posterior(prior_positive: float, sensitivity: float, false_positive: float) -> float:
    """P(H=1 | test=1) with conditional independence of a single test.

    An impossible positive observation is rejected instead of yielding NaN.
    """
    p, tpr, fpr = (float(v) for v in (prior_positive, sensitivity, false_positive))
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (p, tpr, fpr)):
        raise ValueError("all probabilities must be finite and in [0, 1]")
    likelihood = p * tpr + (1 - p) * fpr
    if likelihood == 0:
        raise ValueError("positive observation has zero probability")
    return float(p * tpr / likelihood)
