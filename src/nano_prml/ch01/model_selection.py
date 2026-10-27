"""Leakage-resistant polynomial model selection via K-fold CV (PRML §1.3)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .polynomial import fit_least_squares, predict


@dataclass(frozen=True)
class CVCandidate:
    degree: int
    fold_mse: tuple[float, ...]
    mean_mse: float
    std_mse: float


@dataclass(frozen=True)
class CVSelection:
    selected_degree: int
    candidates: tuple[CVCandidate, ...]


def make_folds(n_samples: int, n_folds: int, seed: int = 0) -> tuple[NDArray[np.int64], ...]:
    """Deterministic disjoint exhaustive validation folds."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be an integer")
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)):
        raise TypeError("n_folds must be an integer")
    if n_samples < 2 or not 2 <= n_folds <= n_samples:
        raise ValueError("require 2 <= n_folds <= n_samples")
    return tuple(np.asarray(a, dtype=np.int64) for a in np.array_split(
        np.random.default_rng(seed).permutation(n_samples), n_folds
    ))


def cross_validate_polynomials(
    x: ArrayLike,
    target: ArrayLike,
    degrees: tuple[int, ...] = (1, 3, 5, 9),
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> CVSelection:
    """Choose the model using only the supplied development data.

    Held-out test data are deliberately absent from this interface. All
    candidate hypotheses share the same validation folds (paired comparison).
    Ties go to the lowest degree; no re-scoring on final test observations.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(target, dtype=np.float64)
    if xs.ndim != 1 or xs.size < 2 or ys.shape != xs.shape:
        raise ValueError("x and target must be equal-length 1-D vectors of size >= 2")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("training samples and labels must be finite")
    if not degrees or len(set(degrees)) != len(degrees):
        raise ValueError("degrees must be a nonempty unique list")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
           or d < 0 for d in degrees):
        raise ValueError("each degree must be a nonnegative integer")
    folds = make_folds(len(xs), n_folds, seed)
    all_indices = np.arange(len(xs))
    candidates = []
    for degree in sorted(degrees):
        fold_errors = []
        for validation_idx in folds:
            training_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=True)
            weights = fit_least_squares(xs[training_idx], ys[training_idx], degree)
            residual = predict(xs[validation_idx], weights) - ys[validation_idx]
            mse = float(np.mean(residual**2))
            if not np.isfinite(mse):
                raise ValueError("cross-validation MSE is nonfinite; rescale features")
            fold_errors.append(mse)
        candidates.append(CVCandidate(degree, tuple(fold_errors),
                                      float(np.mean(fold_errors)),
                                      float(np.std(fold_errors, ddof=0))))
    winner = min(candidates, key=lambda row: (row.mean_mse, row.degree))
    return CVSelection(selected_degree=winner.degree, candidates=tuple(candidates))
