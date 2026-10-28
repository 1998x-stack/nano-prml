"""Geometric and sampling consequences of dimension (Bishop PRML §1.4)."""

from dataclasses import dataclass
from math import exp, expm1, lgamma, log, pi

import numpy as np


def _positive_dimension(dimension: int) -> int:
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)):
        raise TypeError("dimension must be an integer")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return int(dimension)


def log_unit_ball_volume(dimension: int) -> float:
    """Logarithm of the Euclidean d-ball volume, π^(d/2)/Γ(d/2+1)."""
    d = _positive_dimension(dimension)
    return float(0.5 * d * log(pi) - lgamma(0.5 * d + 1))


def unit_ball_volume(dimension: int) -> float:
    """Unit d-ball volume. Use log_unit_ball_volume for high-dimensional underflow."""
    return exp(log_unit_ball_volume(dimension))


def unit_ball_shell_mass(dimension: int, radial_width: float) -> float:
    """Uniform volume share at radii (1 - width, 1] of the unit Euclidean ball.

    In a d-dimensional ball the inner ball occupies (1-width)**d of
    its volume.  expm1 avoids cancellation for very narrow shells.
    """
    d = _positive_dimension(dimension)
    width = float(radial_width)
    if not np.isfinite(width) or not 0 <= width <= 1:
        raise ValueError("radial_width must be finite and in [0, 1]")
    if width == 1:
        return 1.0
    return float(-expm1(d * np.log1p(-width)))


@dataclass(frozen=True)
class PairwiseDistanceStats:
    dimension: int
    n_pairs: int
    mean_distance: float
    std_distance: float
    coefficient_of_variation: float
    mean_squared_distance: float
    expected_squared_distance: float


def pairwise_distance_stats(
    dimension: int, n_pairs: int, *, seed: int = 0
) -> PairwiseDistanceStats:
    """Independent uniform cube pairs; E||X-Y||² = d/6, not E||X-Y||.

    This is a Monte Carlo study of distances in [0,1]^d, **not** a theorem
    about all high-dimensional data distributions or about nearest neighbours.
    """
    d = _positive_dimension(dimension)
    if isinstance(n_pairs, (bool, np.bool_)) or not isinstance(n_pairs, (int, np.integer)):
        raise TypeError("n_pairs must be an integer")
    if n_pairs < 2:
        raise ValueError("n_pairs must be >= 2")
    rng = np.random.default_rng(seed)
    distances = np.empty(n_pairs, dtype=np.float64)
    # Bounded working set even if experiments use many pairs or dimensions.
    for start in range(0, n_pairs, 512):
        count = min(512, n_pairs - start)
        first = rng.random((count, d))
        second = rng.random((count, d))
        distances[start:start + count] = np.linalg.norm(first - second, axis=1)
    mean = float(np.mean(distances))
    sd = float(np.std(distances, ddof=0))
    return PairwiseDistanceStats(
        dimension=d,
        n_pairs=int(n_pairs),
        mean_distance=mean,
        std_distance=sd,
        coefficient_of_variation=sd / mean,
        mean_squared_distance=float(np.mean(distances**2)),
        expected_squared_distance=d / 6.0,
    )
