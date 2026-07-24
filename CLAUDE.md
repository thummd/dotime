# CLAUDE.md

## Project Overview

DoTime (`dotime` on PyPI, v0.1.x) is a synthetic benchmark generator for interventional and
counterfactual time series. It samples multivariate temporal SCMs, applies hard/soft/time-varying/
continuous-time interventions, and produces paired observational–interventional trajectories with
exact ground truth (NOTE the semantics: the continuous-time generator shares one noise realisation
across arms — true counterfactuals; the discrete generators draw the interventional arm with
independent noise — paired interventional twins, not counterfactuals; verified 2026-07-24). Four frozen suites (dot-Identifiability-v1, dot-RegimeSwitch-v1,
dot-Continuous-v1, dot-Generic-100k) are published on Zenodo (citable DOIs) and Hugging Face, and a
NeurIPS-style benchmark paper is in progress (`paper_outline.md`; LaTeX lives in a separate
Overleaf repo, not here).

## Domain & Conventions

- src/ layout, hatchling build. Core code in `src/dotime/`; CLI entry points `dotime-generate`,
  `dotime-benchmark`, `dotime-eval-*` defined in `pyproject.toml`.
- RNG-stream stability is load-bearing: released suites were generated from specific seeds.
  Do not "modernize" `np.random.RandomState` code (ruff NPY002 is deliberately ignored) or
  otherwise perturb sampling order in prior/SCM modules.
- Frozen suites are immutable once released; new content means a new versioned suite via
  `scripts/build_release.py` + `scripts/release_config.yaml`, then Zenodo/HF upload scripts.
- pytest markers: `slow`, `gpu`, `snapshot`. Warnings are errors; "SCM diverged" warnings are
  expected and filtered. mypy is non-strict; consolidated research modules are exempted.

## Build & Test Commands

Dev venv (editable install of this repo, has pytest/ruff/mypy/sphinx):
`PY=/home/dennis/repos/ctp/causal_time_env/bin`

- Tests (no coverage, fast): `$PY/python -m pytest tests/ -q -p no:cacheprovider -o addopts= -m "not slow and not gpu"`
- Full CI-style tests: `$PY/python -m pytest -m "not slow and not gpu"` (coverage on via addopts)
- Lint: `$PY/ruff check .` and `$PY/ruff format --check .`
- Types: `$PY/mypy` (config in pyproject; checks `src/dotime`)
- Docs: `make -C docs html SPHINXBUILD=$PY/sphinx-build` (sphinx `-W`, warnings fatal)

Do NOT use the `dotime-rocm` conda env for this repo: its editable `dotime` points at the old
`/home/dennis/repos/continuous-time-causal-pfn` copy. System python has no dotime installed.

## Verification Requirements

- After code changes: smoke tests green, `ruff check` no new errors, and any change touching
  sampling/priors must show unchanged outputs for fixed seeds (see `tests/test_vectorized_equiv.py`,
  `tests/test_dopfn_equivalence.py` for the equivalence-test pattern).
- Paper claims (tables, DOIs, suite sizes, novelty statements) must be traced to code output or
  released artifacts before being asserted — this repo has an established claim-audit workflow.
- Never claim a benchmark result without rerunning the relevant `dotime-eval-*` harness or citing
  the frozen reference results in `results/reference/`.

## Recommended Skills

- `/paper-verify-experiments` — re-verify paper tables/claims against code and artifacts.
- `/paper-review` — pre-submission review of the benchmark paper.
- `/paper-references` — check citations (baselines: PCMCI, CausalPy, TabPFN/TabICL lineage).

## Workflow Notes

- Companion repo `/home/dennis/repos/dotime-reactor`: applies the (vendored) continuous-time
  Do-Over-Time-PFN to real chemical-reactor event logs, zero-shot. Raw reactor CSVs sit in
  `chemical_reactor/` here; the shared contract is `docs/reactor_event_parser_spec.md`.
- History: repo evolved from CausalTimePrior / `continuous-time-causal-pfn`; some experiments
  (causal chamber, pharmacokinetic multi-seed) still run in that repo. `HANDOVER.md` tracks the
  release checklist; past sessions covered claim audits, TabICLv2 prior ideas, external
  contributor evaluation, and running Do-Over-Time-PFN on user-supplied datasets.
