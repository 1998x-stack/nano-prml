"""Analytic information identities and support-mismatch boundary cases."""

import math

import numpy as np
import pytest

from nano_prml.ch01.information_theory import (
    cross_entropy,
    entropy,
    kl_divergence,
    mutual_information,
)


def test_fair_coin_entropy_in_nats_and_bits():
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2), abs=1e-15)
    assert entropy([0.5, 0.5], base=2) == pytest.approx(1, abs=1e-15)


def test_zero_probability_term_is_defined():
    assert entropy([1, 0]) == 0
    assert cross_entropy([1, 0], [1, 0]) == 0
    assert kl_divergence([1, 0], [1, 0]) == 0


def test_entropy_cross_entropy_kl_identity():
    p, q = [0.3, 0.7], [0.8, 0.2]
    assert cross_entropy(p, q) == pytest.approx(entropy(p) + kl_divergence(p, q), abs=1e-14)
    assert kl_divergence(p, q) > 0
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_support_mismatch_yields_infinite_divergence():
    assert math.isinf(kl_divergence([0.5, 0.5], [1, 0]))
    assert math.isinf(cross_entropy([0.5, 0.5], [1, 0]))
    assert kl_divergence([0, 1], [0.1, 0.9]) < float("inf")


def test_mutual_information_independent_is_zero():
    px = np.array([0.2, 0.8])
    py = np.array([0.6, 0.4, 0.0])
    assert mutual_information(np.outer(px, py)) == pytest.approx(0, abs=1e-14)


def test_mutual_information_perfect_correlation_is_log2():
    joint = np.array([[0.5, 0], [0, 0.5]])
    assert mutual_information(joint) == pytest.approx(math.log(2), abs=1e-14)
    assert mutual_information(joint, base=2) == pytest.approx(1, abs=1e-14)
    assert mutual_information(joint.T) == pytest.approx(mutual_information(joint), abs=1e-14)


def test_mutual_information_matches_joint_entropy_identity():
    joint = np.array([[0.4, 0.1], [0.2, 0.3]])
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    expected = entropy(px) + entropy(py) - entropy(joint.flatten())
    assert mutual_information(joint) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("invalid", [[0.4, 0.4], [-0.1, 1.1], [np.nan, 1], [], [np.inf, 0]])
def test_invalid_probabilities(invalid):
    with pytest.raises(ValueError):
        entropy(invalid)


@pytest.mark.parametrize("base", [0, 1, -2, 0.5, np.nan])
def test_invalid_log_base(base):
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=base)


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        kl_divergence([0.2, 0.8], [0.2, 0.3, 0.5])
    with pytest.raises(ValueError):
        cross_entropy([0.2, 0.8], [0.2, 0.3, 0.5])
    with pytest.raises(ValueError):
        mutual_information([[0.2, 0.3], [0.2, 0.2]])
