# DoTime

**A synthetic benchmark generator for interventional and counterfactual time series.**

`dotime` samples multivariate temporal structural causal models (SCMs),
applies interventions, and produces paired observational / interventional
trajectories with exact counterfactual targets. It ships four frozen evaluation
suites, reference baselines, and an evaluation harness — the artifact behind the
KDD 2027 Datasets & Benchmarks paper.

```{toctree}
:maxdepth: 2
:caption: Getting started

quickstart
benchmarks
leaderboard
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
troubleshoot
```

## Install

```bash
pip install dotime              # core generator + suite loaders (CPU)
pip install 'dotime[baselines]' # classical / Bayesian baselines
pip install 'dotime[models]'    # the Do-Over-Time-PFN model
pip install 'dotime[all]'       # everything except dev tooling
```

## Quickstart

```python
from dotime import DoTime

prior = DoTime(seed=42)
X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
```

See {doc}`quickstart` for the benchmark-loading and evaluation walkthrough.

## Indices

- {ref}`genindex`
- {ref}`modindex`
