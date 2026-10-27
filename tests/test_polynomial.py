"""Analytical, numerical, property, and edge-case verification."""

import numpy as np
import pytest

from nano_prml.ch01.experiment import run_one
from nano_prml.ch01.polynomial import (
    design_matrix,
    fit_bayesian,
    fit_least_squares,
    fit_ridge,
    mse_and_gradient,
    predict,
    predict_bayesian,
)


def test_exact_polynomial_coefficients():
    x = np.linspace(-1, 1, 32)
    known = np.array([1.25, -2.0, 3.5, 0.75])
    actual = fit_least_squares(x, predict(x, known), 3)
    np.testing.assert_allclose(actual, known, rtol=1e-10, atol=1e-10)


def test_repeated_input_has_finite_minimum_norm_solution():
    x = np.array([0.2, 0.2, 0.2])
    w = fit_least_squares(x, [1.0, 1.0, 1.0], 5)
    assert np.isfinite(w).all()
    np.testing.assert_allclose(predict(x, w), np.ones(3), atol=1e-10)


def test_ridge_intercept_is_not_penalized():
    x = np.linspace(-1, 1, 31)
    w = fit_ridge(x, np.full(len(x), 2.5), 4, 1e6)
    np.testing.assert_allclose(w[0], 2.5, atol=1e-9)
    np.testing.assert_allclose(w[1:], 0, atol=1e-7)


def test_zero_penalty_is_ols():
    x = np.linspace(-1, 1, 20)
    a = fit_least_squares(x, np.sin(x), 3)
    b = fit_ridge(x, np.sin(x), 3, 0.0)
    np.testing.assert_allclose(predict(x, a), predict(x, b), atol=1e-11)


def test_bayesian_degree_zero_conjugate_oracle():
    x = np.array([0.0, 0.3, 0.8])
    t = np.array([1.0, 1.4, 0.6])
    alpha, beta = 2.0, 4.0
    post = fit_bayesian(x, t, 0, alpha, beta)
    expected_var = 1 / (alpha + beta * len(t))
    expected_mean = beta * np.sum(t) * expected_var
    np.testing.assert_allclose(post.mean[0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(post.covariance[0, 0], expected_var, atol=1e-12)


def test_bayesian_noise_variance_and_symmetry():
    x = np.linspace(0, 1, 15)
    post = fit_bayesian(x, np.sin(2 * np.pi * x), 3, 1.0, 25.0)
    grid = np.linspace(0, 1, 50)
    mean_f, var_f = predict_bayesian(grid, post, include_noise=False)
    mean_y, var_y = predict_bayesian(grid, post, include_noise=True)
    np.testing.assert_allclose(mean_f, mean_y)
    np.testing.assert_allclose(var_y - var_f, np.full(50, 1 / 25.0), atol=1e-12)
    assert np.all(var_f >= 0)
    np.testing.assert_allclose(post.covariance, post.covariance.T, atol=1e-12)


def test_mse_gradient_central_difference():
    x = np.linspace(-0.7, 0.8, 17)
    t = np.cos(x)
    w = np.array([0.1, -0.4, 0.2])
    _, gradient = mse_and_gradient(x, t, w)
    eps = 1e-6
    numerical = np.empty_like(w)
    for k in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[k] += eps
        minus[k] -= eps
        numerical[k] = (
            mse_and_gradient(x, t, plus)[0] - mse_and_gradient(x, t, minus)[0]
        ) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("degree", [-1, 1.5, True])
def test_invalid_degree(degree):
    with pytest.raises(ValueError, match="degree"):
        design_matrix([0, 1], degree)


@pytest.mark.parametrize("x,t", [([0, 1], [1]), ([], []), ([0, np.nan], [1, 2])])
def test_invalid_training_data(x, t):
    with pytest.raises(ValueError):
        fit_least_squares(x, t, 1)


@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan")])
def test_invalid_precision(value):
    with pytest.raises(ValueError):
        fit_bayesian([0, 1], [0, 1], 1, value, 2)
    with pytest.raises(ValueError):
        fit_bayesian([0, 1], [0, 1], 1, 1, value)


def test_experiment_seed_reproducibility():
    a, b, c = run_one(17), run_one(17), run_one(18)
    assert a == b and a != c
    assert {row["degree"] for row in a} == {1, 3, 9}
    assert all(np.isfinite(row["test_mse"]) for row in a)
