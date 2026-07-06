# Evaluating on Your Own Data

The Do-Over-Time-PFN inference path ships with the package, so you can run the
model on your own trajectories — not just the frozen suites. There is no
CSV-style loader; instead you wrap your data in the same `Episode` format the
suites use, and everything downstream (prediction, metrics) works unchanged.

## 1. Install and get a checkpoint

```bash
pip install 'dotime[models]'
```

Trained checkpoints are hosted at
[`thummd/do-over-time-pfn`](https://huggingface.co/thummd/do-over-time-pfn).
Checkpoints are passed by local path, not auto-downloaded:

```bash
huggingface-cli download thummd/do-over-time-pfn --local-dir checkpoints/
```

## 2. Wrap your data as episodes

An `Episode` bundles one observational trajectory with the intervention that
was (or would be) applied and the query you want answered:

```python
import torch
from dotime.benchmarks import Episode
from dotime.interventions import InterventionSpec, InterventionType

x_obs = torch.as_tensor(my_trajectory, dtype=torch.float32)  # (T, N)

episode = Episode(
    x_obs=x_obs,
    x_int=x_obs,  # not used at predict time; pass ground truth here if you have it
    intervention=InterventionSpec(
        targets=[2],                                # intervened variable index
        times=[50],                                 # onset step
        intervention_type=InterventionType.HARD,    # do(X_2 := 1.5)
        values=1.5,
    ),
    y_true=torch.tensor([observed_outcome]),        # required for metrics
    query_target=torch.tensor([4]),                 # variable to predict
    query_time=torch.tensor([60.0]),                # step to predict at
)
```

`y_true` is the interventional outcome at the query — on real data you only
have this when the intervention actually happened (an experiment, an A/B test,
a policy change). Without it you can still get predictions, but not metrics.

## 3. Predict

The registered `DoOverTimePFN` baseline handles all preprocessing the model
expects — causal masking from the intervention onset, per-variable
normalization over the pre-intervention window, padding, and mapping the
prediction back to the raw scale:

```python
from dotime import baselines

model = baselines.get("DoOverTimePFN", checkpoint="checkpoints/best.pt")
prediction = model.predict(episode)  # raw-scale point estimate, shape (n_queries,)
```

Do not build the model's batch dict by hand — `predict(episode)` mirrors the
training-time preprocessing exactly.

## 4. Evaluate a whole dataset

`evaluate` runs over any `BenchmarkSuite`, and a suite is just metadata plus a
list of episodes, so you can build one around your own data:

```python
from dotime.benchmarks import BenchmarkSuite, SuiteMetadata
from dotime.evaluation import evaluate

meta = SuiteMetadata(
    name="my-dataset", version="1.0.0", zenodo_record_id="", doi="",
    description="in-house evaluation", n_episodes=len(my_episodes),
)
suite = BenchmarkSuite(meta, my_episodes)

results = evaluate(model, suite)
print(results.summary())
```

This reports the same pooled (and, if you set `Episode.structure`,
per-structure) metrics as the frozen suites, and plugs directly into
`scripts/eval_submission.py` for leaderboard-format output.

## Constraints and caveats

- **Input shape.** The model assumes a single intervention target per episode,
  a known onset step, a scalar intervention value, and at most `n_max`
  variables (41 for the released checkpoints). Longer variable lists need to be
  subset; multi-target interventions are out of scope.
- **Zero-shot transfer.** The PFN is trained on the synthetic SCM prior;
  running it on real data is zero-shot transfer — the same setting as the
  CausalChamber task shipped in the `chamber` extra.
- **Reference numbers.** Reproducing the paper's reference results requires the
  checkpoint trained for the corresponding suite/structure and the matched
  evaluation protocol; on arbitrary external data expect indicative, not
  paper-level, numbers.
