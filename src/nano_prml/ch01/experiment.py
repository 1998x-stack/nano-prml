"""Reproducible held-out polynomial curve-fitting experiments."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .polynomial import design_matrix, fit_least_squares, predict


def run_one(seed: int, degrees: tuple[int, ...] = (1, 3, 9)) -> list[dict]:
    """Evaluate fixed hypotheses; no model selection on the held-out test set."""
    rng = np.random.default_rng(seed)
    x_train = np.linspace(0, 1, 12)
    t_train = np.sin(2 * np.pi * x_train) + rng.normal(0, 0.1, len(x_train))
    x_test = rng.uniform(0, 1, 250)
    t_test = np.sin(2 * np.pi * x_test) + rng.normal(0, 0.1, len(x_test))
    rows = []
    for degree in degrees:
        weights = fit_least_squares(x_train, t_train, degree)
        train_mse = float(np.mean((predict(x_train, weights) - t_train) ** 2))
        test_mse = float(np.mean((predict(x_test, weights) - t_test) ** 2))
        rows.append(
            {
                "seed": int(seed),
                "degree": int(degree),
                "train_mse": train_mse,
                "test_mse": test_mse,
                "design_condition_number": float(
                    np.linalg.cond(design_matrix(x_train, degree))
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ch01"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds may not contain duplicates")
    rows = [item for seed in args.seeds for item in run_one(seed)]
    report = {
        "study": "PRML section 1.1: polynomial curve fitting",
        "configuration": {
            "seeds": args.seeds,
            "degrees": [1, 3, 9],
            "n_train": 12,
            "n_test": 250,
            "noise_std": 0.1,
            "signal": "sin(2*pi*x)",
            "protocol": "fixed hypotheses and independent held-out test",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "metrics.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}: {len(rows)} results")


if __name__ == "__main__":
    main()
