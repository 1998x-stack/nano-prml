"""Cross-validation oracle, invariants, leakage and failure cases for §1.3."""

import numpy as np
import pytest

from nano_prml.ch01.model_selection import cross_validate_polynomials, make_folds
from nano_prml.ch01.model_selection_experiment import run_one


def test_folds_partition_data_exactly_and_are_deterministic():
    folds = make_folds(29, 5, seed=42)
    flattened = np.concatenate(folds)
    assert sorted(flattened) == list(range(29))
    assert max(len(fold) for fold in folds) - min(len(fold) for fold in folds) <= 1
    assert all(np.array_equal(a, b) for a, b in zip(folds, make_folds(29, 5, 42)))
    assert not np.array_equal(flattened, np.concatenate(make_folds(29, 5, 43)))


def test_leave_one_out_constant_mean_matches_closed_form_oracle():
    # LOOCV: predicted mean excluding i = (sum(y)-y_i)/(n-1).
    y = np.array([1.5, -2.0, 0.3, 4.0, 3.2, 1.9])
    x = np.arange(len(y), dtype=float) / len(y)
    actual = cross_validate_polynomials(x, y, degrees=(0,),
                                        n_folds=len(y), seed=17)
    loo_predictions = (y.sum() - y) / (len(y) - 1)
    oracle = float(np.mean((y - loo_predictions) ** 2))
    assert actual.selected_degree == 0
    assert actual.candidates[0].mean_mse == pytest.approx(oracle, rel=1e-12)
    assert oracle == pytest.approx(float(np.mean((y - y.mean()) ** 2))
                                   * (len(y) / (len(y) - 1)) ** 2)


def test_cv_selects_correct_linear_model_with_noiseless_data():
    x = np.linspace(-1, 1, 50)
    target = 1.3 - 2.1 * x
    choice = cross_validate_polynomials(x, target, degrees=(3, 1, 0), seed=1)
    assert choice.selected_degree == 1  # 3 may tie numerically but floating-point noise differs
    assert min(c.mean_mse for c in choice.candidates) < 1e-25


def test_fold_change_with_seed_and_same_seed_reproducibility():
    rng = np.random.default_rng(123)
    x = rng.uniform(-1, 1, 33)
    target = x**3 + rng.normal(0, 0.1, len(x))
    a = cross_validate_polynomials(x, target, (1, 3), seed=1)
    b = cross_validate_polynomials(x, target, (1, 3), seed=1)
    c = cross_validate_polynomials(x, target, (1, 3), seed=2)
    assert a == b
    assert a.candidates != c.candidates


def test_test_set_cannot_influence_selection():
    # Selection API accepts training data only. Changing external test arrays is irrelevant.
    train_x = np.linspace(-1, 1, 30)
    train_y = np.cos(np.pi * train_x)
    first = cross_validate_polynomials(train_x, train_y, (0, 2, 4), seed=1)
    test_x = np.linspace(-1, 1, 100)
    test_y = 1e10 * np.cos(np.pi * test_x)
    assert len(test_y) == 100
    second = cross_validate_polynomials(train_x, train_y, (0, 2, 4), seed=1)
    assert first == second


@pytest.mark.parametrize("bad_degrees", [(), (1, 1), (-1, 2), (True, 1), (1.1, 2)])
def test_reject_invalid_candidate_grids(bad_degrees):
    with pytest.raises(ValueError):
        cross_validate_polynomials([0, 1, 2, 3], [0, 1, 2, 3], bad_degrees)


def test_reject_invalid_folds_and_nonfinite_training_values():
    with pytest.raises(ValueError):
        make_folds(4, 5)
    with pytest.raises(TypeError):
        make_folds(True, 3)
    with pytest.raises(ValueError):
        cross_validate_polynomials([0, 1, np.nan], [0, 1, 2], (0,), n_folds=2)
    with pytest.raises(ValueError):
        cross_validate_polynomials([0, 1, 2], [0, 1, 2], (0,), n_folds=4)


def test_reproducible_three_seed_experiments():
    first = run_one(7)
    assert first == run_one(7) and first != run_one(8)
    assert first["selected_degree"] in {1, 3, 5, 9}
    assert len(first["cv_candidates"]) == 4
    assert np.isfinite(first["selected_test_mse"])
    assert np.isfinite(first["degree1_baseline_test_mse"])
