"""Closed-form geometry and independent Monte Carlo checks for PRML §1.4."""

import math

import numpy as np
import pytest

from nano_prml.ch01.dimensionality import (
    log_unit_ball_volume,
    pairwise_distance_stats,
    unit_ball_shell_mass,
    unit_ball_volume,
)


def test_unit_ball_closed_form_oracle():
    np.testing.assert_allclose([unit_ball_volume(d) for d in (1, 2, 3)],
                               [2.0, math.pi, 4.0 * math.pi / 3], rtol=1e-14)


def test_log_volume_stays_finite_when_volume_underflows():
    assert np.isfinite(log_unit_ball_volume(1000))
    assert log_unit_ball_volume(1000) < log_unit_ball_volume(100)
    assert unit_ball_volume(1000) == 0.0


def test_shell_exact_and_monotone():
    np.testing.assert_allclose(unit_ball_shell_mass(1, 0.1), 0.1, atol=1e-15)
    np.testing.assert_allclose(unit_ball_shell_mass(2, 0.1), 0.19, atol=1e-15)
    assert unit_ball_shell_mass(100, 0.1) > unit_ball_shell_mass(10, 0.1)
    assert unit_ball_shell_mass(8, 0) == 0
    assert unit_ball_shell_mass(8, 1) == 1


def test_thin_shell_precision():
    # For small epsilon, 1-(1-epsilon)^d ≈ d*epsilon.
    width = 1e-13
    assert unit_ball_shell_mass(4, width) == pytest.approx(4e-13, rel=1e-12)


def test_cube_pair_distance_exact_second_moment():
    result = pairwise_distance_stats(10, 20000, seed=7)
    assert result.expected_squared_distance == pytest.approx(10 / 6)
    assert result.mean_squared_distance == pytest.approx(10 / 6, rel=0.025)
    assert result.mean_distance**2 < result.mean_squared_distance


def test_reproducibility_and_concentration():
    a = pairwise_distance_stats(40, 2000, seed=11)
    assert a == pairwise_distance_stats(40, 2000, seed=11)
    assert a != pairwise_distance_stats(40, 2000, seed=12)
    low = pairwise_distance_stats(2, 2000, seed=0)
    assert a.coefficient_of_variation < low.coefficient_of_variation


@pytest.mark.parametrize("dimension", [0, -1, 2.5, True])
def test_invalid_dimension(dimension):
    with pytest.raises((ValueError, TypeError)):
        log_unit_ball_volume(dimension)


@pytest.mark.parametrize("width", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_shell_width(width):
    with pytest.raises(ValueError):
        unit_ball_shell_mass(3, width)


@pytest.mark.parametrize("n_pairs", [0, 1, False, 1.5])
def test_invalid_pair_count(n_pairs):
    with pytest.raises((ValueError, TypeError)):
        pairwise_distance_stats(3, n_pairs)
