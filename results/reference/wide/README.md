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

## Reproduce

From the repository root:

```bash
PYTHONPATH=src python results/reference/wide/qa.py --workers 4
```
