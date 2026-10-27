"""Three-seed Bayes frequency calibration simulation without external data."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .probability import binary_positive_posterior


def run_one(seed: int, sizes: tuple[int, ...] = (1000, 10000, 100000)) -> list[dict]:
    if not sizes or any(n <= 0 for n in sizes) or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sizes must be strictly increasing positive integers")
    prior, tpr, fpr = 0.1, 0.8, 0.05
    oracle = binary_positive_posterior(prior, tpr, fpr)
    rng = np.random.default_rng(seed)
    h = rng.random(sizes[-1]) < prior
    observed = rng.random(sizes[-1]) < np.where(h, tpr, fpr)
    rows = []
    for n in sizes:
        count = int(np.count_nonzero(observed[:n]))
        estimate = float(np.count_nonzero(h[:n] & observed[:n]) / count) if count else None
        rows.append({"seed": int(seed), "n_samples": n, "n_positive_observations": count,
                     "posterior_oracle": oracle, "conditional_frequency": estimate,
                     "absolute_error": abs(estimate - oracle) if estimate is not None else None})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ch01/probability"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    report = {"topic": "PRML 1.2 Bayes theorem: empirical conditional frequency",
              "configuration": {"seeds": args.seeds, "sizes": [1000, 10000, 100000],
                                "prior_positive": 0.1, "sensitivity": 0.8, "false_positive": 0.05},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "utc_timestamp": datetime.now(timezone.utc).isoformat()},
              "results": [row for seed in args.seeds for row in run_one(seed)]}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}: {len(report['results'])} results")


if __name__ == "__main__":
    main()
