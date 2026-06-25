# Leaderboard

A public, versioned leaderboard for interventional / counterfactual time-series
estimation on the DoTime suites. Anyone can submit via a reproducible
path; entries are seeded with the reference baselines below.

## Metrics

Lower RMSE / NMSE / MAE is better; higher direction accuracy (`dir_acc`) and R²
are better. `Oracle` (the true SCM counterfactual) is the upper bound on the
synthetic suites; intervention-unaware baselines (`Zero`, `Mean`, `AR1`,
`VAR-OLS`) are the naive lower references.

## Reference results (preliminary)

Pooled RMSE / direction accuracy. **Preliminary** — regenerate at full scale with
`scripts/build_release.py` before camera-ready; numbers below are from a reduced
build for illustration.

RMSE (direction accuracy in parentheses); lower RMSE / higher dir-acc is better.

| Baseline | Identifiability | RegimeSwitch | Continuous | Generic |
|---|---|---|---|---|
| Oracle (upper bound) | 0.00 (1.00) | 0.00 (1.00) | 0.00 (1.00) | 0.00 (1.00) |
| Zero | 0.65 (0.00) | 1.81 (0.00) | 1.55 (0.00) | 34.00 (0.00) |
| Mean (TrajMean) | 0.63 (0.64) | 1.82 (0.55) | 1.54 (0.54) | 34.30 (0.65) |
| AR1 | 0.67 (0.60) | 2.33 (0.45) | 4.66 (0.53) | 33.99 (0.63) |
| VAR-OLS | 0.65 (0.62) | 1.93 (0.51) | 2.64 (0.57) | 51.93 (0.66) |
| DoOverTimePFN | _pending checkpoint mapping (Phase 8)_ | | | |

Generic RMSE is on the raw (un-normalized) scale, hence the larger magnitudes.
Numbers come from a reduced build (n ≈ 200–2000/suite); regenerate at release
scale for camera-ready.

Per-tier and per-structure breakdowns are available in each submission JSON.

## Submitting

A submission is a JSON file conforming to the `ctp-submission/1` schema, produced
by the reproducible evaluator:

```bash
# a registered baseline
python scripts/eval_submission.py --suite CTP-Identifiability-v1 \
    --baseline VAR-OLS --out submission.json

# your own model (anything with predict(episode) -> Tensor)
python scripts/eval_submission.py --suite CTP-Identifiability-v1 \
    --model mypkg.models:MyModel --name MyModel --out submission.json
```

### Submission schema (`ctp-submission/1`)

```json
{
  "schema": "ctp-submission/1",
  "suite": "CTP-Identifiability-v1",
  "model": "MyModel",
  "package_version": "0.1.0",
  "n_episodes": 10800,
  "n_queries": 10800,
  "pooled":        {"rmse": 0.0, "mae": 0.0, "nmse": 0.0, "r2": 0.0, "dir_acc": 0.0},
  "per_structure": {"back_door": {"rmse": 0.0, "...": 0.0}, "...": {}}
}
```

Open a pull request adding your `submission.json` under `leaderboard/<suite>/`,
or submit through the project's leaderboard space. Since the D&B track is
single-blind, submissions may use real names.
