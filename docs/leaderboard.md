# Leaderboard

A public, versioned leaderboard for interventional / counterfactual time-series
estimation on the DoTime suites. Anyone can submit via a reproducible
path; entries are seeded with the reference baselines below.

## Metrics

Lower RMSE / NMSE / MAE is better; higher direction accuracy (`dir_acc`) and R²
are better. `Oracle` (the true SCM counterfactual) is the upper bound on the
synthetic suites; intervention-unaware baselines (`Zero`, `Mean`, `AR1`,
`VAR-OLS`) are the naive lower references.

## Reference results (release scale)

Pooled RMSE ± bootstrap std (direction accuracy in parentheses) on the **full
frozen suites** — 10,800 / 9,999 / 9,999 / 100,000 episodes. Reproduce with
`scripts/eval_reference_table.py`; the backing JSONs (with 95% episode-cluster
bootstrap CIs) live under [`results/reference/`](https://github.com/thummd/dotime/tree/main/results/reference).

| Baseline | Identifiability | RegimeSwitch | Continuous | Generic |
|---|---|---|---|---|
| Oracle (upper bound) | 0.00 (1.00) | 0.00 (1.00) | 0.00 (1.00) | 0.00 (1.00) |
| Zero | 0.72±0.02 (0.00) | 1.82±0.04 (0.00) | 2.75±0.61 (0.00) | 46.2±1.4 (0.00) |
| Mean (TrajMean) | 0.62±0.01 (0.67) | 1.84±0.03 (0.50) | 2.74±0.61 (0.50) | 46.6±1.4 (0.66) |
| AR1 | 0.74±0.02 (0.60) | 2.54±0.05 (0.49) | 3.40±0.67 (0.51) | 49.5±1.5 (0.62) |
| VAR-OLS | 0.66±0.01 (0.64) | 1.92±0.04 (0.50) | 3.50±0.56 (0.50) | 62.1±4.6 (0.65) |
| BackDoorOLS | 0.78±0.02 (0.65) | 1.84±0.03 (0.50) | 2.77±0.60 (0.53) | 46.6±1.4 (0.66) |
| IV2SLS | 0.88±0.06 (0.67) | 1.84±0.03 (0.50) | 2.76±0.60 (0.51) | 46.6±1.4 (0.66) |
| DoOverTimePFN (obs) | 0.67±0.01 (0.57) | 2.07±0.04 (0.51) | 2.97±0.70 (0.49) | 46.3±1.5 (0.62) |
| DoOverTimePFN (int) | 0.67±0.01 (**0.66**) | 1.99±0.04 (0.49) | 4.07±0.87 (0.51) | 48.2±1.4 (**0.66**) |

Generic RMSE is on the raw (un-normalized) scale, hence the larger magnitudes.
The PFN rows use the general `s9ho_all_causal` / `s9ho_all_obs` checkpoints
(HF `thummd/do-over-time-pfn`) evaluated with `scripts/eval_pfn_reference.py`
(interpolation causal-masking; observational mode zeroes the intervention
features). Direction-accuracy standard errors are ≤ 0.01 on every full-suite row;
a constant prediction (Zero) has no sign and scores 0 by convention.

Foundation-model int-vs-obs comparisons (TabPFN, Chronos-2) and the
structure-matched OSC/BTM gap sweeps across trajectory lengths are in
`results/reference/server/` and `results/reference/structure_matched/`.

Per-tier and per-structure breakdowns are available in each submission JSON.

## Submitting

A submission is a JSON file conforming to the `ctp-submission/1` schema, produced
by the reproducible evaluator:

```bash
# a registered baseline
python scripts/eval_submission.py --suite dot-Identifiability-v1 \
    --baseline VAR-OLS --out submission.json

# your own model (anything with predict(episode) -> Tensor)
python scripts/eval_submission.py --suite dot-Identifiability-v1 \
    --model mypkg.models:MyModel --name MyModel --out submission.json
```

### Submission schema (`ctp-submission/1`)

```json
{
  "schema": "ctp-submission/1",
  "suite": "dot-Identifiability-v1",
  "model": "MyModel",
  "package_version": "0.1.0",
  "n_episodes": 10800,
  "n_queries": 10800,
  "pooled":        {"rmse": 0.0, "mae": 0.0, "nmse": 0.0, "r2": 0.0, "dir_acc": 0.0},
  "per_structure": {"back_door": {"rmse": 0.0, "...": 0.0}, "...": {}}
}
```

Open a pull request adding your `submission.json` under `leaderboard/<suite>/`,
or submit through the project's leaderboard space.
