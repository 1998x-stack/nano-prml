"""Reproducibility and expected-risk oracles of the new experiment driver."""

import math

import pytest

from nano_prml.ch01.theory_experiment import DIMENSIONS, run_one


def test_reproducible_complete_report():
    a = run_one(4, n_pairs=100, n_decisions=200)
    b = run_one(4, n_pairs=100, n_decisions=200)
    c = run_one(5, n_pairs=100, n_decisions=200)
    assert a == b and a != c
    assert [v["dimension"] for v in a["geometry"]] == list(DIMENSIONS)
    assert a["information"]["fair_coin_entropy_bits"] == pytest.approx(1)
    assert a["information"]["mutual_information_independent_nats"] == pytest.approx(0)
    assert a["information"]["mutual_information_correlated_nats"] > 0
    assert a["decision"]["bayes_weighted_expected_risk"] <= (
        a["decision"]["map_weighted_expected_risk"]
    )
    assert all(math.isfinite(x["mean_squared_distance"]) for x in a["geometry"])


def test_invalid_experiment_counts():
    with pytest.raises(ValueError):
        run_one(0, n_pairs=1)
    with pytest.raises(ValueError):
        run_one(0, n_decisions=1)


def test_figure_artifacts_are_real_images(tmp_path):
    pytest.importorskip("matplotlib")
    from nano_prml.ch01.theory_experiment import save_figures

    report = {"results": [run_one(0, n_pairs=25, n_decisions=50)]}
    save_figures(report, tmp_path)
    for name in ("shell_mass.png", "distance_concentration.png", "decision_cost.png"):
        path = tmp_path / name
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
