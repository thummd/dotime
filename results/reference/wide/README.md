# dot-Wide-v1 quality check (September 2026)

Quality check of the wide generic suite `dot-Wide-v1` 1.0.0 before its full
build. No model was trained.

## How it fits the pipeline

`scripts/release_config_wide.yaml` defines the suite. `scripts/build_release.py`
turns it into per-episode specs with `dotime._build.episode_specs`, and because
the suite sets generic-prior options, every spec has the kind
`generic_configured`. `dotime._build.make_episode` hands those specs to
`_make_configured_generic_episode`, which

1. builds `DoTime` with the suite's `prior_config` (`N_min: 12`, `N_max: 40`,
   `K_max: 8` and `RECOMMENDED_HARDENING`), `chain_prob: 0` and
   `regime_switching_prob: 0`,
2. draws the pair with `generate_pair(pair_mode="counterfactual")`, so both arms
   share one exogenous-noise realisation, and resamples it up to 10 times while
   an arm is zeroed,
3. drops the prior's hidden variables (`u*`) that are not intervened on, and
4. only then calls `episode_from_pair`, which picks the query among the
   released columns and, because the suite sets `query_row: window_end`,
   places it at the last step of the intervention window.

`qa.py` runs exactly this path through `episode_specs` and `make_episode`.
Episode seeds depend only on the suite seed and the episode index, so its 200
episodes are episodes 0 to 199 of the full build.

## What is asserted

- No episode diverged.
- Every simulated graph has 12 to 40 variables.
- Both arms agree exactly before the intervention onset.
- The released columns plus the dropped ones add up to the simulated graph.
- The per-arm target statistics of the reference harness
  (`dotime.reference.reference_table.target_qa` with `dir_target="effect"`):
  finite values, positive variance and a nonzero fraction of at least 0.5 in
  both level arms, and an effect that is not zero everywhere.

## Results on 200 episodes

| Quantity | Value |
|---|---|
| Diverged | 0 |
| Simulated variables | 12 to 40, median 28 |
| Released variables | 8 to 40, median 23 |
| Hidden variables dropped | 14.3% of all variables, median 3 per episode, none in 32 episodes, at most 15 |
| Tiers 1, 2 and 3 | 58, 61 and 81 episodes |
| Intervention types | 101 hard, 61 soft, 38 time varying |
| Values at the clip (\|x\| >= 999) | 0 episodes |
| Observational level at the query | nonzero 100%, mean 0.340, variance 0.98 |
| Interventional level at the query (`y_true`) | nonzero 100%, mean 0.472, variance 1.81 |
| Effect at the query (`y_true - y_obs`) | nonzero 98%, mean 0.132, variance 0.72 |
| Time per episode | mean 2.11 s, median 1.66 s, 90th percentile 4.4 s. Tier means 0.7, 1.8 and 3.4 s |
| Projected full build | 10 000 episodes on 15 workers in about 0.39 h, inside the 2 h budget, so `K_max` stays 8 |

The times cover sampling, hardening, simulating both arms, retries and the
latent drop. They were measured with 4 workers on a lightly loaded machine
(load average 1.0 before and 4.8 after the run on 16 cores), so the projection
is approximate.

## The query sits at the last step of the intervention window

By default `episode_from_pair` queries the last step of the trajectory, as for
`dot-Generic-100k`. The generic intervention sampler draws a window that almost
never reaches that step, and the hardened dynamics are contractive, so under
shared noise the effect has mostly decayed by the time the default query reads
it. On these 200 episodes the default rule put 4.5% of the queries inside the
window, at a median of 36 steps after it, and only 13.5% of the effects at the
query reached 0.1 in absolute value (median 3.3e-6). Those numbers are in the
version of `wide_qa.json` committed before the rule below was adopted (commit
af7968b).

The suite therefore sets `query_row: window_end`, an opt-in build option that
places the query at the last step of the intervention window. The query variable
is still the most affected released variable that is not intervened on, now
judged at that row, and the tensors are unchanged. `metadata["query_row"]`
records the rule.

| Quantity | Value |
|---|---|
| Queries inside the intervention window | 100% |
| Steps from the window's last step to the query | 0 |
| \|effect\| at the query | median 0.48, 90th percentile 1.35 |
| Queries with \|effect\| >= 0.01 | 97.5% |
| Queries with \|effect\| >= 0.1 | 88.5% |
| Queries with \|effect\| >= 0.1 pre-onset standard deviations | 93.0% |

The level target `y_true` is exact, and the arms are true counterfactual twins
over the whole trajectory. With the query at the window's last step, direction
accuracy on the effect sign is defined on 88.5% of the episodes.

`wide_qa.json` holds every number above, the histograms, the effect record of
each episode, and the machine load during the run.

## The full build (2026-09-30)

`scripts/build_release.py --config scripts/release_config_wide.yaml --workers 7
--target-qa enforce` built the suite in 83 minutes on a shared 16-core machine.
`wide_full_effect_share.json` summarises the built rows and
`wide_cpu_effect.json` holds the CPU reference rows (`dotime-eval-reference
--dir-target effect`, with NaiveOLS).

| Quantity | Value |
|---|---|
| Episodes | 10,000, none diverged |
| Simulated variables | 12 to 40, median 26 |
| Released variables | 6 to 40, median 22 |
| Tiers 1, 2 and 3 | 3,116, 3,541 and 3,343 episodes |
| \|effect\| at the query | median 0.45, 90th percentile 1.23 |
| Queries with \|effect\| >= 0.01 | 98.5% |
| Queries with \|effect\| >= 0.1 | 89.3% |
| Effect-sign accuracy, NaiveOLS / BackDoorOLS / IV2SLS | 0.624 / 0.612 / 0.612 |
| Effect-sign accuracy, best naive (Zero, Mean, AR1, VAR-OLS) | 0.612 |

The validity invariants of `../audit_2026-09/suite_invariants_2026_10.json` hold in
every episode: both arms agree before the onset, every hard intervention places its
do-value, and no arm is zeroed. The generic builder stores no effect field, so the
effect identity is not checked here.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src python results/reference/wide/qa.py --workers 4
```

## Per-lag breakdown (October 2026)

dot-Generic-100k cannot show how estimators fare across lags: its arms are
independent draws, so its effect field measures regression to the mean
(`docs/benchmarks.md`, "Per-lag scores"). dot-Wide-v1 can. Its arms share one
noise realisation, its query sits at the end of the intervention window, and
every episode records its lagged graph (`metadata["graph"]`).
`lag_breakdown.py` rebuilds the full suite from the release config, asserts the
per-arm target statistics, and gates on `wide_cpu_effect.json`. `dir_n_valid`
and `dir_acc` of Zero, Mean, AR1, VAR-OLS and NaiveOLS must match exactly, and
pooled RMSE to a relative 1e-4, the tolerance of the frozen fingerprints. The
hardening's spectral scaling and the time-varying interventions go through the
platform's linear algebra, sin and exp. On this Windows rebuild against the
Linux build, counts and accuracies matched exactly, and the largest RMSE
difference was 2.9e-6. The results are in `wide_lag_breakdown.json`, scored as
`run_baseline` scores (effect sign, float32).

By the smallest summed lag from an intervened column to the query, with
effect-sign accuracy ± binomial SE:

| Min lag | Episodes | \|effect\| ≥ 0.1 | Median \|effect\| | Mean | NaiveOLS | TimeOLS |
|---|---|---|---|---|---|---|
| 0 | 5,040 | 0.919 | 0.513 | 0.629 ± 0.007 | 0.679 ± 0.007 | 0.632 ± 0.007 |
| 1 | 4,009 | 0.891 | 0.407 | 0.596 ± 0.008 | 0.568 ± 0.008 | 0.546 ± 0.008 |
| 2 | 633 | 0.855 | 0.377 | 0.567 ± 0.021 | 0.529 ± 0.021 | 0.486 ± 0.021 |
| 3 | 144 | 0.764 | 0.276 | 0.627 ± 0.046 | 0.555 ± 0.047 | 0.555 ± 0.047 |
| ≥ 4 | 102 | 0.706 | 0.221 | 0.597 ± 0.058 | 0.597 ± 0.058 | 0.514 ± 0.059 |
| unreachable | 72 | 0 | 0 | n/a | n/a | n/a |

The JSON also splits by the sampled maximum lag `k_sampled` and the lag order
the graph uses, `k_eff` (1 to 8, about 1,250 episodes each), and has the Zero,
AR1 and VAR-OLS rows.

Findings:

1. Effects shrink with the lag between treatment and query. The median |effect|
   falls from 0.51 at lag 0 to 0.22 at lag 4 or more, and the share that can be
   scored from 0.92 to 0.71. The 72 unreachable queries have an exact zero effect
   and are not scored.
2. The unadjusted regression helps only for contemporaneous effects. NaiveOLS
   beats the pre-onset mean at lag 0 (0.679 against 0.629) and falls below it at
   lags 1 and 2 (0.568 against 0.596, 0.529 against 0.567). It regresses `Y_t` on
   `A_t`, so it sees only paths with no lag. This is the generic counterpart of
   the lagged mediator of dot-Identifiability-v1 1.2.0, where only an estimator
   that models the lag separates.
3. The sampled lag order barely matters. Across `k_sampled` from 1 to 8 the
   accuracies stay between 0.57 and 0.66, because most queries are reached at
   lag 0 or 1 whatever the graph's maximum lag. The lag that matters is that of
   the path from treatment to query, which is why the graph metadata records
   it.
4. No packaged estimator models lagged effects in wide graphs. The lag-aware
   structural VAR of `../detection_power_2026-10` runs only on the named
   structures. This is an open entry for submissions.
5. TimeOLS, which targets confounding by time, is worse than NaiveOLS here
   (0.587 against 0.624 pooled). dot-Wide-v1 has no time driver, so its trend,
   seasonal term and roll-forward add variance without removing bias. It is a
   specialist estimator, not a general replacement.

To reproduce (about 30 minutes to build on 16 workers, cached with `--cache`):

```bash
PYTHONPATH=src python results/reference/wide/lag_breakdown.py --workers 16 --cache wide.npz
```
