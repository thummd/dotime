<!-- Badges (wired in Phase 6: arXiv, docs, HF dataset, license, CI) -->
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://causaltime.readthedocs.io)

# CausalTime

**A synthetic benchmark generator for interventional and counterfactual time series.**

`causaltime` samples multivariate temporal structural causal models (SCMs),
applies interventions to them, and produces paired observational / interventional
trajectories together with exact counterfactual targets. It is the data engine and
evaluation harness behind the KDD 2027 Datasets & Benchmarks paper of the same name.

Where most time-series causal benchmarks are *observational* (recover a graph from
passive dynamics), CausalTime is *interventional / counterfactual*: it answers
`do(...)` queries over time and ships frozen suites to measure how well a model
estimates interventional effects.

## Highlights

- **Four intervention regimes** — hard, soft, time-varying, and continuous-time
  intervention *windows*.
- **Four counterfactual sampling modes** — `prior`, `observed_discrete`,
  `observed_normal`, `observed_uniform`, with a positivity guard.
- **Regime-switching SCMs** as a strict generalization of interrupted-time-series.
- **Eight named identification structures** (back-door, front-door, IV, …) with
  exact counterfactual ground truth.
- **Reference baselines** and **four frozen evaluation suites** (released with
  Zenodo DOIs and Croissant metadata).

## Install

```bash
pip install causaltime            # core generator + suite loaders (CPU, no GPU needed)
pip install 'causaltime[baselines]'   # classical / Bayesian baselines
pip install 'causaltime[all]'         # everything except dev tooling
```

Requires Python ≥ 3.10. The core install runs on CPU with no GPU dependency.

## Quickstart

```python
from causaltime import CausalTime

prior = CausalTime(seed=42)
X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
# X_obs, X_int: (T, N) tensors;  intervention: InterventionSpec;  scm: TemporalSCM
```

Command-line:

```bash
ct-generate -n 1000 -T 200 -o data/sample.pt   # sample paired trajectories
ct-benchmark --list                            # list frozen benchmark suites
```

## Documentation

Full docs at <https://causaltime.readthedocs.io>. See `docs/quickstart.md` and
`docs/benchmarks.md` to get started locally.

## Citation

If you use CausalTime, please cite the paper and the software (see
[`CITATION.cff`](CITATION.cff)).

## License

Apache-2.0 (code). Released benchmark suites are distributed under CC-BY-4.0.
