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
`dotime-eval-submission` for leaderboard-format output.

## Single-stream data (interrupted time series)

You do not need paired observational / interventional trajectories — the
pairing in the suites is an artifact of the synthetic generator, which can
produce both arms from the same SCM and noise. At predict time the model never
reads `x_int`, and `x_obs` is causally masked from the intervention onset
onward, so its actual input is the pre-event history plus the intervention
spec and the query. A single real-world stream with an embedded event is
therefore the native deployment shape (the interrupted-time-series setting the
benchmark generalizes). Slice it per event:

- **Pre-event window → context.** Everything before the onset is treated as
  the undisturbed observational regime. The normalization stats come from this
  window, so give it enough length (the encoder's default context is 200
  steps).
- **The event → `InterventionSpec`** — onset step, affected variable, imposed
  level.
- **Post-event observations → `y_true`.** After the event, the stream *is* the
  interventional trajectory: the value actually observed at the query time is
  a valid ground-truth target for the model's prediction.
- **Pass the stream as both `x_obs` and `x_int`.** Pre-onset the two arms
  coincide by definition, and the post-onset part of `x_obs` is masked away.
  The one thing a single stream cannot give you is the *no-intervention*
  counterfactual — and the model is not asked to predict that.

```python
onset, t_query = 480, 520   # event at step 480, evaluate 40 steps later
j, k = 2, 4                 # intervened variable, outcome variable

episode = Episode(
    x_obs=stream,           # (T, N); masked from `onset` internally
    x_int=stream,
    intervention=InterventionSpec(
        targets=[j], times=[onset],
        intervention_type=InterventionType.HARD,
        values=float(stream[onset, j]),      # the level the event imposed
    ),
    y_true=stream[t_query, k].reshape(1),    # what actually happened
    query_target=torch.tensor([k]),
    query_time=torch.tensor([float(t_query)]),
)
```

For the resulting metrics to mean anything, three things must hold. The event
has to be do-like — imposed exogenously (a policy rollout, a forced setpoint
change, an A/B switch), not triggered by the system's own state, which
confounds the realized outcome unless the drivers of the decision are
themselves observed channels in the context. An event that is not a clamp on
an observed variable should be encoded as an explicit 0/1 treatment column in
the `(T, N)` matrix, with the event as `do(treatment := 1)`. And one event is
one episode: a stream with *k* events yields *k* episodes, each of whose
pre-onset windows should be clear of the previous event's transient, since the
model reads everything before onset as the undisturbed regime.

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
