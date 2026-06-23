# Quick Start

A five-minute tour through the three things you'll most often do with CausalTimePrior: **generate paired data**, **load a frozen benchmark**, and **evaluate a baseline**.

## 1. Generate paired observational / interventional data

```python
from causaltimeprior import CausalTimePrior

prior = CausalTimePrior(seed=42)

X_obs, X_int, intervention, scm = prior.generate_pair(T=100)

print(f"Obs:  {tuple(X_obs.shape)}")            # (T, N)
print(f"Int:  {tuple(X_int.shape)}")            # (T, N)
print(f"Type: {intervention.intervention_type}")
print(f"At:   t = {intervention.times}")
```

## 2. Generate a dataset

```python
dataset = prior.generate_dataset(n_scms=1000, T=200)
```

Returns a list of `(X_obs, X_int, intervention)` tuples — useful for amortized training or quick sanity checks. For large runs, prefer the on-the-fly streaming loader:

```python
from causaltimeprior.data import TemporalInterventionDataLoader

loader = TemporalInterventionDataLoader(
    num_steps=10_000,
    batch_size=32,
    t_range=(50, 200),
    intervention_source="observed_normal",  # counterfactual mode
)
for batch in loader:
    ...  # train your model
```

## 3. Load a frozen benchmark

```python
from causaltimeprior.benchmarks import load_benchmark

suite = load_benchmark("CTP-Identifiability-v1")
print(suite)

# CTP-Identifiability-v1
# CTP-Identifiability-v1
#   8 structures (back_door, observed_confounder, confounder_mediator,
#                 front_door, mediator, instrumental_variable,
#                 rct_no_confounding, unobserved_confounder)
#   1 350 episodes per structure
#   T = 200, N_max = 10
```

## 4. Evaluate a baseline

```python
from causaltimeprior.baselines import VAROLSBaseline
from causaltimeprior.evaluation import evaluate

baseline = VAROLSBaseline(lag=3)
results = evaluate(baseline, suite, metrics=["rmse", "direction_accuracy", "lift_over_naive"])
print(results.to_dataframe())
```

## Where to go next

- {doc}`../user_guide/data_generation` — full prior configuration.
- {doc}`../user_guide/interventions` — hard, soft, time-varying, and continuous-window interventions.
- {doc}`../user_guide/counterfactuals` — the four counterfactual sampling modes.
- {doc}`../user_guide/benchmarks` — the four frozen suites and their evaluation protocols.
- {doc}`../tutorials/index` — end-to-end notebook walkthroughs.
