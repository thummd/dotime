# Frozen Benchmark Suites

DoTime ships four versioned, immutable suites for reproducible evaluation. Each has a Zenodo DOI and Croissant metadata.

## Suites

- **`dot-Identifiability-v1`** — ~10.8k trajectories across **eight** named structures: `back_door`, `observed_confounder`, `confounder_mediator` (back-door family); `front_door`, `mediator` (front-door family); `instrumental_variable` (IV); `rct_no_confounding` (trivially identified); `unobserved_confounder` (non-identifiable, robustness check). Counterfactuals are exact.
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
Zenodo archive of record (DOIs `10.5281/zenodo.20846064`, `.20846074`, `.20845981`,
`.20845983`) — and md5-verified against the manifest. Pass `force_download=True` to
re-fetch. Override the cache with `$DOTIME_CACHE` or `cache_dir=`.

## Evaluation protocol

The default evaluation reports RMSE, NMSE, MAE, direction accuracy, lift-over-naive, and effect-error correlation, computed per-structure and pooled.

```python
from dotime.evaluation import evaluate

results = evaluate(model, suite)
```

See the {doc}`api` reference for the full `benchmarks`, `baselines`, and
`evaluation` module documentation.
