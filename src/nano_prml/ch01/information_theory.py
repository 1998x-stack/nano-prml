"""Discrete Shannon entropy, cross-entropy, KL and mutual information (§1.6)."""

from math import log

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _base_divisor(base: float) -> float:
    b = float(base)
    if not np.isfinite(b) or b <= 0 or b == 1:
        raise ValueError("log base must be positive, finite, and different from 1")
    # Entropy in base < 1 changes sign; reject because teaching convention is base > 1.
    if b < 1:
        raise ValueError("log base must exceed 1")
    return log(b)


def _probability(values: ArrayLike, *, ndim: int, name: str) -> NDArray[np.float64]:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != ndim or any(n == 0 for n in p.shape) or np.any(~np.isfinite(p)):
        raise ValueError(f"{name} must be a finite nonempty {ndim}-D probability array")
    if np.any(p < 0) or not np.isclose(np.sum(p), 1.0, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be nonnegative and sum to one")
    return p / np.sum(p)


def entropy(probabilities: ArrayLike, *, base: float = np.e) -> float:
    """H(P)=-Σ p log(p); convention 0*log(0)=0."""
    p = _probability(probabilities, ndim=1, name="probabilities")
    divisor = _base_divisor(base)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive])) / divisor)


def cross_entropy(p: ArrayLike, q: ArrayLike, *, base: float = np.e) -> float:
    """H(P,Q)=-Σ p log(q), +inf when P has mass where Q is zero."""
    left = _probability(p, ndim=1, name="p")
    right = _probability(q, ndim=1, name="q")
    divisor = _base_divisor(base)
    if left.shape != right.shape:
        raise ValueError("p and q must have the same shape")
    positive = left > 0
    if np.any(right[positive] == 0):
        return float("inf")
    return float(-np.sum(left[positive] * np.log(right[positive])) / divisor)


def kl_divergence(p: ArrayLike, q: ArrayLike, *, base: float = np.e) -> float:
    """D_KL(P||Q); does not symmetrize and may be +infinity."""
    left = _probability(p, ndim=1, name="p")
    right = _probability(q, ndim=1, name="q")
    divisor = _base_divisor(base)
    if left.shape != right.shape:
        raise ValueError("p and q must have the same shape")
    mask = left > 0
    if np.any(right[mask] == 0):
        return float("inf")
    value = float(np.sum(left[mask] * (np.log(left[mask]) - np.log(right[mask]))) / divisor)
    if value < -1e-12:
        raise ArithmeticError("nonnegative information divergence invariant failed")
    return max(value, 0.0)


def mutual_information(joint: ArrayLike, *, base: float = np.e) -> float:
    """I(X;Y)=Σ p(x,y) log[p(x,y)/(p(x)p(y))], discrete finite joint."""
    p = _probability(joint, ndim=2, name="joint")
    divisor = _base_divisor(base)
    px = np.sum(p, axis=1, keepdims=True)
    py = np.sum(p, axis=0, keepdims=True)
    mask = p > 0
    product = px * py
    value = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(product[mask]))) / divisor)
    if value < -1e-12:
        raise ArithmeticError("nonnegative information divergence invariant failed")
    return max(value, 0.0)
