# Frozen Benchmark Suites

DoTime ships four versioned, immutable suites for reproducible evaluation. Each has a Zenodo DOI and Croissant metadata.

## Suites

- **`dot-Identifiability-v1`** — ~10.8k trajectories across **eight** named structures: `back_door`, `observed_confounder`, `confounder_mediator` (back-door family); `front_door`, `mediator` (front-door family); `instrumental_variable` (IV); `bi_variate` (trivially identified); `unobserved_confounder` (non-identifiable, robustness check). Counterfactuals are exact.
- **`dot-RegimeSwitch-v1`** — regime-switching trajectories with controllable break density.
- **`dot-Continuous-v1`** — continuous-time intervention windows, multiple query offsets.
- **`dot-Generic-100k`** — 100 000 trajectories from the full diverse prior. Training-scale.

## Loader

```python
from dotime.benchmarks import load_benchmark

suite = load_benchmark("dot-Identifiability-v1", version="1.0.0")
```

On first access the suite is fetched into `~/.cache/dotime/` — from the Hugging Face
mirror ([`thummd/dot-*`](https://huggingface.co/thummd)) by default, falling back to the
Zenodo archive of record (concept DOIs `10.5281/zenodo.20846063`, `.20846073`, `.20845980`,
`.20845982`, each resolving to the latest archived version) — and md5-verified
against the manifest. Pass `force_download=True` to
re-fetch. Override the cache with `$DOTIME_CACHE` or `cache_dir=`.

## v1.0.0 field semantics and known issues (erratum)

The archived v1.0.0 files are frozen; the issues below are **documented, not
silently patched**. They are fixed in the generator for any v1.1+ build.

| Suite | Issue | Consequence | Workaround on v1 data |
|---|---|---|---|
| Identifiability | `metadata.y_causal_effect` stores the **interventional level** (`y_true`), not the effect | any effect-based analysis using the field is wrong | use the released realignment sidecar's `y_effect_corrected` |
| Identifiability | released `x_obs` columns are in **topological** order while `x_int`/`query_target`/`intervention_target` are canonical (identity only for `bi_variate`) | baselines reading `x_obs[:, query_target]` touch the wrong variable on 6/8 structures | realignment sidecar maps each episode's canonical→topo permutation |
| Identifiability | hidden variables (`front_door`, `instrumental_variable`, `unobserved_confounder`) are **not zeroed** in the released `x_obs` | the "unobserved" confounder is readable from the data | zero the sidecar's `hidden_canonical` columns after realigning |
| Generic-100k / Identifiability | diverged episodes are stored as all-zero trajectories with **no flag** (28.7% / 4.6%) | included in RMSE; silently excluded from direction accuracy by the near-zero filter | filter `x_int.abs().max() == 0`; v1.1 adds a `diverged` metadata flag |
| Continuous | documentation said query offsets `{1,2,3,5,10}`; actual query times are **uniform over [onset, T-1]** (observed offsets 0–138) | protocol description only — data and scoring are self-consistent | none needed |
| all | reported direction accuracy scores the sign of the **interventional level**, not the causal effect | see the paper erratum; `--dir-target effect` re-scores | `dotime-eval-reference --dir-target effect [--realignment <sidecar>]` |

`x_int`, `y_true`, `query_target`, and `intervention_*` are correct and mutually
consistent in all four v1 suites; `dot-Continuous-v1` and `dot-RegimeSwitch-v1`
carry none of the column/field issues above.

## Evaluation protocol

The default evaluation reports RMSE, NMSE, MAE, direction accuracy, lift-over-naive, and effect-error correlation, computed per-structure and pooled.

```python
from dotime.evaluation import evaluate

results = evaluate(model, suite)
```

See the {doc}`api` reference for the full `benchmarks`, `baselines`, and
`evaluation` module documentation.
