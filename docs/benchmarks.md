# Frozen Benchmark Suites

CausalTime ships four versioned, immutable suites for reproducible evaluation. Each has a Zenodo DOI and Croissant metadata.

## Suites

- **`CTP-Identifiability-v1`** — ~10.8k trajectories across **eight** named structures: `back_door`, `observed_confounder`, `confounder_mediator` (back-door family); `front_door`, `mediator` (front-door family); `instrumental_variable` (IV); `rct_no_confounding` (trivially identified); `unobserved_confounder` (non-identifiable, robustness check). Counterfactuals are exact.
- **`CTP-RegimeSwitch-v1`** — regime-switching trajectories with controllable break density.
- **`CTP-Continuous-v1`** — continuous-time intervention windows, multiple query offsets.
- **`CTP-Generic-100k`** — 100 000 trajectories from the full diverse prior. Training-scale.

## Loader

```python
from causaltime.benchmarks import load_benchmark

suite = load_benchmark("CTP-Identifiability-v1", version="1.0.0")
```

On first access, the suite is downloaded from Zenodo into `~/.cache/causaltime/`. Pass `force_download=True` to redownload.

## Evaluation protocol

The default evaluation reports RMSE, NMSE, MAE, direction accuracy, lift-over-naive, and effect-error correlation, computed per-structure and pooled.

```python
from causaltime.evaluation import evaluate

results = evaluate(model, suite)
```

See the {doc}`api` reference for the full `benchmarks`, `baselines`, and
`evaluation` module documentation.
