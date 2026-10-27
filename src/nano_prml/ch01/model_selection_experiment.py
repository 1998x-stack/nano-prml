"""Train-only CV and one-time independent test evaluation for polynomial models."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model_selection import cross_validate_polynomials
from .polynomial import fit_least_squares, predict


DEGREES = (1, 3, 5, 9)


def run_one(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1, 1, 60)
    y_train = np.sin(np.pi * x_train) + rng.normal(0, 0.1, len(x_train))
    # Test set is created only after all hyperparameters have been selected.
    selection = cross_validate_polynomials(x_train, y_train, DEGREES, n_folds=5, seed=seed)
    weights = fit_least_squares(x_train, y_train, selection.selected_degree)
    x_test = rng.uniform(-1, 1, 300)
    y_test = np.sin(np.pi * x_test) + rng.normal(0, 0.1, len(x_test))
    test_mse = float(np.mean((predict(x_test, weights) - y_test) ** 2))
    baseline_weights = fit_least_squares(x_train, y_train, 1)
    baseline_test_mse = float(np.mean((predict(x_test, baseline_weights) - y_test) ** 2))
    return {"seed": int(seed), "selected_degree": selection.selected_degree,
            "cv_candidates": [{"degree": row.degree, "mean_mse": row.mean_mse,
                               "std_mse": row.std_mse, "fold_mse": list(row.fold_mse)}
                              for row in selection.candidates],
            "selected_test_mse": test_mse, "degree1_baseline_test_mse": baseline_test_mse}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ch01/model-selection"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    report = {"topic": "PRML 1.3 model selection: train-only 5-fold CV",
              "configuration": {"seeds": args.seeds, "candidate_degrees": list(DEGREES),
                                "folds": 5, "n_train": 60, "n_test": 300,
                                "noise_std": 0.1, "signal": "sin(pi*x)",
                                "protocol": "train-only CV then single independent held-out test"},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "utc_timestamp": datetime.now(timezone.utc).isoformat()},
              "results": [run_one(seed) for seed in args.seeds]}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}: {len(report['results'])} results")


if __name__ == "__main__":
    main()
