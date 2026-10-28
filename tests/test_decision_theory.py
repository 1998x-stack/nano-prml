"""Risk-minimizing actions, MAP special case, rejection, and invalid inputs."""

import numpy as np
import pytest

from nano_prml.ch01.decision_theory import (
    expected_risk,
    minimum_risk_action,
    reject_option_loss,
    zero_one_loss,
)


def test_zero_one_loss_equivalent_to_map():
    posterior = [0.2, 0.5, 0.3]
    np.testing.assert_allclose(expected_risk(posterior, zero_one_loss(3)), [0.8, 0.5, 0.7])
    assert minimum_risk_action(posterior, zero_one_loss(3)) == 1


def test_asymmetric_loss_changes_decision_from_map():
    # p(class=1)=0.3; MAP picks 0, asymmetric false-negative loss picks 1.
    posterior = [0.7, 0.3]
    cost = [[0, 5], [1, 0]]
    assert minimum_risk_action(posterior, zero_one_loss(2)) == 0
    assert minimum_risk_action(posterior, cost) == 1
    np.testing.assert_allclose(expected_risk(posterior, cost), [1.5, 0.7])


def test_rejection_region_and_action_index():
    loss = reject_option_loss(2, 0.2)
    assert loss.shape == (3, 2)
    assert minimum_risk_action([0.5, 0.5], loss) == 2
    assert minimum_risk_action([0.95, 0.05], loss) == 0
    assert minimum_risk_action([0.05, 0.95], loss) == 1


def test_batch_and_tie_breaking():
    rows = [[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]]
    np.testing.assert_array_equal(minimum_risk_action(rows, zero_one_loss(2)), [0, 0, 1])
    np.testing.assert_allclose(expected_risk(rows, zero_one_loss(2)),
                               [[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])


@pytest.mark.parametrize("posterior", [[0.4, 0.4], [-0.1, 1.1], [np.nan, 1], [], [[0, 1], [0.1, 0.8]]])
def test_invalid_posterior(posterior):
    with pytest.raises(ValueError):
        expected_risk(posterior, zero_one_loss(2))


@pytest.mark.parametrize("loss", [[[-1, 1], [1, 0]], [[0, np.inf]], [[0, 1, 0]]])
def test_invalid_loss(loss):
    with pytest.raises(ValueError):
        expected_risk([0.5, 0.5], loss)


def test_invalid_loss_generators():
    with pytest.raises(ValueError):
        zero_one_loss(1)
    with pytest.raises(TypeError):
        zero_one_loss(True)
    with pytest.raises(ValueError):
        reject_option_loss(2, -1)
