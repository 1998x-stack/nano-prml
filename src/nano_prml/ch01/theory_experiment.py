"""Reproducible Stage 01 Batch 03 experiments: dimensions, decisions, information."""

import argparse
import json
import logging
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .decision_theory import minimum_risk_action, zero_one_loss
from .dimensionality import (
    log_unit_ball_volume,
    pairwise_distance_stats,
    unit_ball_shell_mass,
)
from .information_theory import entropy, kl_divergence, mutual_information

LOGGER = logging.getLogger(__name__)
DIMENSIONS = (1, 2, 5, 10, 20, 50, 100)
COST_MATRIX = np.array([[0.0, 5.0], [1.0, 0.0]])
CORRELATED_JOINT = np.array([[0.45, 0.05], [0.05, 0.45]])


def run_one(seed: int, *, n_pairs: int = 4000, n_decisions: int = 20000) -> dict:
    """One seeded study. Decision rules use known posteriors, no test-label tuning.

    This is a synthetic, oracle-posterior evaluation, not an estimator study.
    The Monte Carlo draw is independent from the deterministic Bayes actions.
    """
    if n_decisions < 2 or n_pairs < 2:
        raise ValueError("n_pairs and n_decisions must both be >= 2")
    distances = []
    for d in DIMENSIONS:
        summary = asdict(pairwise_distance_stats(d, n_pairs, seed=seed + 7919 * d))
        summary["log_unit_ball_volume"] = log_unit_ball_volume(d)
        summary["outer_tenth_shell_mass"] = unit_ball_shell_mass(d, 0.1)
        distances.append(summary)

    rng = np.random.default_rng(seed)
    # Posterior p(Y=1|X) is known by construction; no fitting or data leakage.
    p_true_one = rng.uniform(0.0, 1.0, size=n_decisions)
    posteriors = np.column_stack((1.0 - p_true_one, p_true_one))
    truth = rng.binomial(1, p_true_one)
    weighted = minimum_risk_action(posteriors, COST_MATRIX)
    map_actions = minimum_risk_action(posteriors, zero_one_loss(2))
    weighted_realized = COST_MATRIX[weighted, truth]
    map_realized = COST_MATRIX[map_actions, truth]
    weighted_expected = np.sum(posteriors * COST_MATRIX[weighted], axis=1)
    map_expected = np.sum(posteriors * COST_MATRIX[map_actions], axis=1)
    # The weighted Bayes risk is pointwise <= MAP's expected cost for this matrix.
    if np.any(weighted_expected > map_expected + 1e-12):
        raise AssertionError("Bayes action must minimize expected cost pointwise")
    decision = {
        "n_decisions": int(n_decisions),
        "loss_rows_actions_columns_true_classes": COST_MATRIX.tolist(),
        "posterior_generator": "p1 ~ Uniform(0,1), Y|p1 ~ Bernoulli(p1)",
        "bayes_weighted_expected_risk": float(np.mean(weighted_expected)),
        "map_weighted_expected_risk": float(np.mean(map_expected)),
        "bayes_weighted_empirical_loss": float(np.mean(weighted_realized)),
        "map_weighted_empirical_loss": float(np.mean(map_realized)),
        "bayes_positive_action_rate": float(np.mean(weighted == 1)),
    }
    independent_joint = np.full((2, 2), 0.25)
    info = {
        "fair_coin_entropy_nats": entropy([0.5, 0.5]),
        "fair_coin_entropy_bits": entropy([0.5, 0.5], base=2),
        "kl_fair_to_biased_nats": kl_divergence([0.5, 0.5], [0.75, 0.25]),
        "mutual_information_independent_nats": mutual_information(independent_joint),
        "mutual_information_correlated_nats": mutual_information(CORRELATED_JOINT),
    }
    return {"seed": int(seed), "geometry": distances, "decision": decision, "information": info}


def save_figures(report: dict, output: Path) -> None:
    """Opt-in matplotlib plots; deterministic labels and unaltered observations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires: pip install -e '.[viz]'") from exc

    rows = report["results"]
    dims = [r["dimension"] for r in rows[0]["geometry"]]
    shell = [r["outer_tenth_shell_mass"] for r in rows[0]["geometry"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dims, shell, marker="o")
    ax.set(xlabel="Dimension d", ylabel="Fraction of unit-ball volume",
           title="Outer radial 10% of unit ball (exact)", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output / "shell_mass.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rows:
        ax.plot(dims, [v["coefficient_of_variation"] for v in row["geometry"]],
                marker="o", label=f"seed={row['seed']}")
    ax.set(xlabel="Dimension d", ylabel="SD(distance) / mean(distance)",
           title="Concentration of independent uniform cube-pair distances")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "distance_concentration.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    seeds = [str(row["seed"]) for row in rows]
    xpos = np.arange(len(rows), dtype=np.float64)
    ax.bar(xpos - 0.2, [v["decision"]["bayes_weighted_empirical_loss"] for v in rows],
           width=0.4, label="Bayes minimum-risk")
    ax.bar(xpos + 0.2, [v["decision"]["map_weighted_empirical_loss"] for v in rows],
           width=0.4, label="0-1 MAP baseline")
    ax.set(xticks=xpos, xticklabels=seeds, xlabel="Seed", ylabel="Realized cost / case",
           title="Asymmetric-cost decision policies (same samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "decision_cost.png", dpi=160)
    plt.close(fig)
    LOGGER.info("Saved three figures under %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ch01/theory"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--figures", action="store_true", help="requires optional matplotlib")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    reports = [run_one(seed) for seed in args.seeds]
    result = {
        "topic": "PRML §§1.4–1.6: high dimensions, Bayes risk, information",
        "configuration": {
            "seeds": args.seeds, "dimensions": list(DIMENSIONS), "pairs_per_dimension": 4000,
            "decision_trials": 20000, "unit_ball_shell_radial_width": 0.1,
            "protocol": "fixed models and independent Monte Carlo; no fitted classifiers",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "utc_timestamp": datetime.now(timezone.utc).isoformat()},
        "results": reports,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "metrics.json"
    metrics.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d seeds)", metrics, len(reports))
    if args.figures:
        save_figures(result, args.output)


if __name__ == "__main__":
    main()
