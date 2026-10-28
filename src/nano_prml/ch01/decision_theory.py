"""Posterior expected loss and Bayes actions (Bishop PRML §1.5)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _posterior_rows(posterior: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    arr = np.asarray(posterior, dtype=np.float64)
    is_vector = arr.ndim == 1
    if is_vector:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or 0 in arr.shape or np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("posterior must contain finite nonnegative probabilities, shape (N, C)")
    if not np.all(np.isclose(np.sum(arr, axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("each posterior row must sum to one")
    return arr, is_vector


def expected_risk(posterior: ArrayLike, loss: ArrayLike) -> NDArray[np.float64]:
    """Risk R[a]=sum_c L[a,c] p(c|x).

    Loss rows represent actions; columns represent true classes.  For a
    single posterior vector returns (A,), otherwise (N,A).
    """
    probs, single = _posterior_rows(posterior)
    matrix = np.asarray(loss, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != probs.shape[1]:
        raise ValueError("loss shape must be (n_actions, n_classes)")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("loss must be finite and nonnegative")
    risks = probs @ matrix.T
    return risks[0] if single else risks


def minimum_risk_action(posterior: ArrayLike, loss: ArrayLike) -> int | NDArray[np.int64]:
    """Lowest-index action resolves exact risk ties deterministically."""
    risks = expected_risk(posterior, loss)
    selected = np.argmin(risks, axis=-1)
    return int(selected) if risks.ndim == 1 else selected


def zero_one_loss(n_classes: int) -> NDArray[np.float64]:
    """Square 0/1 loss: action a predicts class a."""
    if isinstance(n_classes, (bool, np.bool_)) or not isinstance(n_classes, (int, np.integer)):
        raise TypeError("n_classes must be an integer")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    return np.ones((n_classes, n_classes)) - np.eye(n_classes)


def reject_option_loss(n_classes: int, reject_cost: float) -> NDArray[np.float64]:
    """0/1 loss with extra final action 'reject', costing r regardless of class."""
    base = zero_one_loss(n_classes)
    cost = float(reject_cost)
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("reject_cost must be finite and nonnegative")
    return np.vstack((base, np.full((1, n_classes), cost)))
