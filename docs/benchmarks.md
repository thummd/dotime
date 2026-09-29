# Frozen Benchmark Suites

DoTime ships four versioned, immutable suites for reproducible evaluation. Each has a Zenodo DOI and Croissant metadata.

## Suites

- **`dot-Identifiability-v1`** — ~10.8k trajectories across **eight** named structures: `back_door`, `observed_confounder`, `confounder_mediator` (back-door family); `front_door`, `mediator` (front-door family); `instrumental_variable` (IV); `bi_variate` (trivially identified); `unobserved_confounder` (null-effect control: a hidden U drives A and Y, and there is no A→Y edge at any lag, so the effect is identified and equals zero). Targets are exact interventional outcomes. In v1.0.0 the two arms are independent noise draws from the same SCM (interventional twins, so `y_int - y_obs` is not a per-episode counterfactual effect). From v1.1.0 one pre-drawn noise stream is shared across arms and the pair is a true counterfactual.
- **`dot-RegimeSwitch-v1`** — regime-switching trajectories with controllable break density. In v1.0.0 no regime mechanism reads its parents, so every variable is independent noise. See the erratum below.
- **`dot-Continuous-v1`** — continuous-time intervention windows, multiple query offsets.
- **`dot-Generic-100k`** — 100 000 trajectories from the full diverse prior. Training-scale.

**`dot-Identifiability-v1` 1.1.0 (2026-09)** regenerates the suite with one exogenous-noise realisation per episode shared across both arms (`pair_mode="counterfactual"`): the arms agree exactly before the intervention onset, `y_true - y_obs` is a per-episode counterfactual effect, the released `x_obs` is canonically aligned with hidden variables zeroed, `y_causal_effect` is correct, and diverged episodes are resampled (0 zeroed episodes, verified on all 10 800 episodes). It is a new artifact with new trajectories and targets. Pin `version="1.0.0"` to load the frozen original.

The generator also builds a ninth structure that no frozen suite contains yet,
`bow_graph`: a hidden U drives A and Y, and A drives Y. It is
`unobserved_confounder` plus the causal edge A→Y, and it is the structure that
is **not identifiable**. Nothing observed blocks the back-door path A←U→Y and
there is no mediator, so two SCMs can agree on every observational
distribution and still differ in the effect of `do(A)`.

```python
ExtendedDoTime(tscm_structure="bow_graph", pair_mode="counterfactual")
```

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
| Generic-100k / Identifiability | diverged arms are stored as all-zero trajectories with **no flag**. Both arms are zeroed in 28.7% / 4.6% of episodes (28,734 / 497); **only one arm** is zeroed in a further 1,377 / 7 (Generic: 682 observational, 695 interventional), so 30.1% / 4.7% have a zeroed arm | both-arm and interventional-arm zeroing store `y_true == 0`: included in RMSE, excluded from direction accuracy by the near-zero filter. An observational-arm zeroing leaves a nonzero target behind an all-zero history: 657 of the 682 Generic cases are scored for direction, and every history-based baseline predicts exactly 0 there | drop an episode when **either** arm is all-zero (`x_obs.abs().max() == 0` or `x_int.abs().max() == 0`); builds from v1.1 on set a `diverged` metadata flag on the same either-arm rule |
| Continuous | documentation said query offsets `{1,2,3,5,10}`; actual query times are **uniform over [onset, T-1]** (observed offsets 0–138) | protocol description only — data and scoring are self-consistent | none needed |
| all | reported direction accuracy scores the sign of the **interventional level**, not the causal effect | see the paper erratum; `--dir-target effect` re-scores | `dotime-eval-reference --dir-target effect [--realignment <sidecar>]` |
| Continuous (harness, fixed) | `query_time` is normalized time, `index / (T - 1)`, while Identifiability stores `index / T`; `evaluation.query_obs_levels` and `dotime-eval-chronos` decoded every fraction as `index / T` | the effect-scored `y_obs` (and the Chronos forecast horizon) sat one step past the query on **92.5%** of episodes, every query at index 100 to 198; level-scored rows of the other baselines are unaffected | each suite declares its encoding (`SuiteMetadata.query_time_encoding`) and consumers read `Episode.query_time_idx` |
| Continuous (1.0.0) | the query target is drawn uniformly over the three observable variables, treatment included, so **34.0%** of episodes are **self-queries** (query on the intervened variable); **16.6%** of all episodes are in-window self-queries whose target equals the do-value (the suite is hard-intervention only) and 17.4% are post-window self-queries (relaxation after release); 33.7% of in-window queries are self-queries | the published Continuous rows **include** them, which favours any model that receives the intervention value as input on the in-window subset | `Episode.is_self_query`; `--exclude-self-queries` on the eval harness; per-episode `self_query` / `in_window` / `window_end_idx` in `results/reference/dot-Continuous-v1.0.0_query_sidecar.jsonl`; 1.1.0+ builds tag them in metadata |
| Identifiability (1.0.0 and 1.1.0) | every episode is queried at **offset 0** (the onset step); the paper's per-structure offset protocol was used by the model's loaders, not by the release build | with true counterfactuals (1.1.0) the effect at the query is exactly 0 for `mediator`, `observed_confounder`, `unobserved_confounder` and ~1/3 of `front_door`/`confounder_mediator` episodes, so the effect metric is defined only for structures with an instantaneous A→Y edge | filter `|effect| >= 0.1` (done by `direction_accuracy`); a 1.2.0 protocol with path-length offsets is prepared |
| Identifiability / RegimeSwitch / Generic (1.0.0) | the two arms are **independent noise draws** from the same SCM (interventional twins), so `y_int - y_obs` is not a per-episode counterfactual effect | effect-based analyses on these files score a noisy twin difference | Identifiability 1.1.0 shares the noise across arms; the continuous suite always did |
| RegimeSwitch, Generic-100k (1.0.0) | every regime-switching SCM is **per-variable independent noise**. The builder renamed each regime's nodes to `X0..X{N-1}` but kept the mechanism weights under the old names, so no mechanism reads its parents. This is all 9,999 RegimeSwitch episodes and the 15,041 regime-switching episodes (15.0%) of Generic-100k. In the released RegimeSwitch `x_obs` the median absolute lag-1 autocorrelation is 0.048, 1 of 64,830 variables exceeds 0.3, and the median cross-correlation is 0.048 | these episodes have no temporal or causal structure. An intervention changes only the treated variable, so every other variable, the query target included, has the same distribution in both arms and a true effect of 0 | nothing to recover from v1 data. Drop the regime share of Generic-100k with the recipe below. New builds can opt into the fix |

`x_int`, `y_true`, `query_target`, and `intervention_*` are correct and mutually
consistent in all four v1 suites, apart from the zeroed (diverged) episodes in the
table above; `dot-Continuous-v1` and `dot-RegimeSwitch-v1` carry none of the
column/field issues above and have no zeroed arm.
The RegimeSwitch issue lies in the simulated dynamics, not in how the fields
are stored.

**Regime-switching episodes (1.0.0).** A Generic-100k episode comes from a
regime-switching SCM exactly when the first draw of a generator seeded with its
per-episode seed lies in `[0.15, 0.30)`:

```python
import torch


def in_regime_share(scm_id: int) -> bool:
    """True for the 15,041 dot-Generic-100k 1.0.0 episodes that are independent noise."""
    seed = (20264719 * 1_000_003 + scm_id) & 0x7FFFFFFF  # per-episode seed of the build
    g = torch.Generator()
    g.manual_seed(seed)
    return 0.15 <= torch.rand(1, generator=g).item() < 0.30
```

The unreleased opt-in `DoTime(config={"regime_canonical_weights": True})` re-keys
the weights and zeroes any arm whose values exceed 500, as for the other SCMs. It
draws the same random numbers as the default, and the default stays byte-identical,
so the v1.0.0 files still regenerate. With live weights most regime SCMs of the
current prior are unstable. At the default prior 64.5% of RegimeSwitch episodes and
62.7% of the Generic regime share diverge, and at `N_max=60, K_max=8` 94% and 97%
do (`results/reference/regime_weights/`). No suite has been rebuilt with the flag.

## Evaluation protocol

The default evaluation reports RMSE, NMSE, MAE, direction accuracy, lift-over-naive, and effect-error correlation, computed per-structure and pooled.

```python
from dotime.evaluation import evaluate

results = evaluate(model, suite)
```

See the {doc}`api` reference for the full `benchmarks`, `baselines`, and
`evaluation` module documentation.
