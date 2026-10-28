"""Independent analytical-oracle, normalization, and simulation tests for §1.2."""

import math

import numpy as np
import pytest

from nano_prml.ch01.probability import (
    bayes_posterior,
    binary_positive_posterior,
    condition_on_column,
    joint_to_marginals,
    logsumexp,
    normalize_weights,
)
from nano_prml.ch01.probability_experiment import run_one


def test_exact_enumeration_of_binary_bayes():
    # Enumerate the four cells p(H, test) and compare posterior with log-space API.
    prior = 0.1
    joint = np.array([[prior * 0.8, prior * 0.2],
                      [(1 - prior) * 0.05, (1 - prior) * 0.95]])
    np.testing.assert_allclose(joint.sum(), 1.0, atol=1e-15)
    expected = joint[0, 0] / joint[:, 0].sum()
    assert math.isclose(expected, 0.64, abs_tol=1e-15)
    assert binary_positive_posterior(prior, 0.8, 0.05) == pytest.approx(expected)
    posterior = bayes_posterior([prior, 1 - prior], np.log([0.8, 0.05]))
    np.testing.assert_allclose(posterior, [expected, 1 - expected], atol=1e-15)
    np.testing.assert_allclose(condition_on_column(joint, 0), posterior, atol=1e-15)


def test_marginalization_identity_and_total_probability():
    matrix = [[2, 1, 0], [3, 4, 0]]
    marginal_x, marginal_y = joint_to_marginals(matrix)
    np.testing.assert_allclose(marginal_x, [0.3, 0.7])
    np.testing.assert_allclose(marginal_y, [0.5, 0.5, 0.0])
    assert marginal_x.sum() == pytest.approx(1.0)
    assert marginal_y.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive mass"):
        condition_on_column(matrix, 2)


def test_logspace_prevents_underflow_and_large_weight_overflow():
    prior = normalize_weights([1e308, 1e308, 0])
    np.testing.assert_allclose(prior, [0.5, 0.5, 0])
    log_likelihoods = np.array([-10000.0, -10001.0, -np.inf])
    posterior = bayes_posterior(prior, log_likelihoods)
    np.testing.assert_allclose(posterior[:2], [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    assert posterior[2] == 0
    assert posterior.sum() == pytest.approx(1.0)
    assert logsumexp([-10000, -10001]) == pytest.approx(-10000 + np.log1p(np.exp(-1)))
    assert logsumexp([-np.inf, -np.inf]) == -np.inf


def test_permutation_invariance_of_bayes_posterior():
    prior = np.array([0.2, 0.3, 0.5])
    ll = np.log([0.8, 0.1, 0.03])
    permutation = [2, 0, 1]
    np.testing.assert_allclose(bayes_posterior(prior, ll)[permutation],
                               bayes_posterior(prior[permutation], ll[permutation]))


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 2], [], [[1, 2]]])
def test_invalid_weight_mass(weights):
    with pytest.raises(ValueError):
        normalize_weights(weights)


@pytest.mark.parametrize("prior,likelihood", [([0, 0], [0, 0]),
                                                ([0.4, 0.6], [-np.inf, -np.inf]),
                                                ([0.2, 0.8], [0]),
                                                ([0.2, 0.8], [np.nan, 0])])
def test_impossible_evidence_and_bad_shapes(prior, likelihood):
    with pytest.raises(ValueError):
        bayes_posterior(prior, likelihood)


def test_invalid_joint_and_binary_probabilities():
    with pytest.raises(ValueError):
        joint_to_marginals([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        joint_to_marginals([[1, -1]])
    with pytest.raises(ValueError):
        binary_positive_posterior(0.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        binary_positive_posterior(0.3, -0.2, 0.1)
    with pytest.raises(TypeError):
        condition_on_column([[1, 2]], True)


def test_simulation_is_reproducible_and_approaches_oracle():
    a = run_one(7)
    assert a == run_one(7) and a != run_one(8)
    assert len(a) == 3
    assert a[-1]["n_positive_observations"] > 0
    assert a[-1]["absolute_error"] < 0.03
