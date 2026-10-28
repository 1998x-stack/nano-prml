# AGENTS.md · nano-prml

## Purpose
Build an **original** Chinese executable companion to Bishop PRML (2006), with each mathematically justified lesson independently testable. Do not reproduce copyrighted textbook pages, figures or exercises.

## Read before making changes
1. `README.md`, `docs/ROADMAP.md`, `docs/AUTHORING.md`, `docs/ACCEPTANCE.md`, `docs/REFERENCES.md`.
2. Existing `docs/STAGE-01-BATCH-*.md`, open PRs and latest CI conclusions.
3. Original-source table of contents and errata; keep stage numbering mapped to PRML chapters.

## Non-negotiable boundaries
- New batch => branch from latest reviewable predecessor; **stacked PR**, never force-push / merge unfinished prerequisite PRs.
- Prefer clear float64 NumPy reference functions to opaque framework wrappers. Include assumptions, shapes, units, numerical caveats and invalid-input errors.
- Each core result: >=2 independent correctness strategies (analytic oracle, invariant, property, finite difference, independent numerical experiment).
- Three fixed seeds for stochastic experiments; record parameters, versions and result files. Do not tune on final test labels, and do not cherry-pick seeds.
- Optional visualization only under `[viz]`; default dependency should remain NumPy. Save charts with labeled axes; never claim qualitative trends without measurements.
- Use Python stdlib `logging`; prohibit silent exception handling, fabricated metrics or declaring tests successful before actual execution.
- Keep README concise and link detailed bilingual/original Chinese lessons. Update acceptance and CI in the same PR as code.

## Verification and publishing
~~~bash
python -m pip install -e '.[dev,viz]'
ruff check .
python -m pytest -q
python -m nano_prml.ch01.theory_experiment --output artifacts/ch01/theory --seeds 0 1 2 --figures
~~~

PR remains Draft until remote Python 3.10 and 3.12 CI, math review, copyright review, and predecessor dependencies are resolved. Report the exact commit and URLs, never invent them.
