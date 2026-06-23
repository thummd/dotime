# CausalTimePrior

**A synthetic benchmark generator for interventional and counterfactual time series.**

`causaltimeprior` samples multivariate temporal structural causal models (SCMs),
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
pip install causaltimeprior              # core generator + suite loaders (CPU)
pip install 'causaltimeprior[baselines]' # classical / Bayesian baselines
pip install 'causaltimeprior[models]'    # the Do-Over-Time-PFN model
pip install 'causaltimeprior[all]'       # everything except dev tooling
```

## Quickstart

```python
from causaltimeprior import CausalTimePrior

prior = CausalTimePrior(seed=42)
X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
```

See {doc}`quickstart` for the benchmark-loading and evaluation walkthrough.

## Indices

- {ref}`genindex`
- {ref}`modindex`
