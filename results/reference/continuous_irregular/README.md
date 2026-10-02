# Irregular observation grids (September 2026)

Measurement behind the observation schedules of
`scripts/release_config_continuous_irregular.yaml` (`dot-ContinuousIrregular-v1`
1.0.0, prepared but not yet built when this ran). No model was trained.

`measure_irregular_grids.py` takes, for each structure of `dot-Continuous-v1`,
the first 300 episode indices of that structure in the suite (suite seed
20263719) and builds each of them on every candidate schedule. The candidates
therefore share seeds, SCMs and interventions, and the comparison is paired.
Candidates that pass `check_schedule` are built by `dotime._build.make_episode`,
as a release build would build them. The two unguarded candidates allow Euler
sub-steps above 1.0, which `check_schedule` refuses. They are built with the
same calls past the guard. The released 1.0.0 statistics come from all 9,999
cached episodes. The per-arm target statistics (nonzero fraction, mean and
variance of the observational level, the interventional level and the effect at
the query) are asserted for every group before any number is used. Both level
arms are nonzero in every episode of every group.

All 900 regular episodes have the same `x_obs` bytes as the released rows with
the same index.

Pooled over the three structures (900 episodes per schedule):

| Schedule | Largest sub-step | Finite | Max \|x\| above 10 | Median max \|x\| | p99 max \|x\| | Largest \|x\| | Zero effect at the query | Time per episode |
|---|---|---|---|---|---|---|---|---|
| Released 1.0.0 (9,999 episodes) | 1 | 100% | 3.9% | 2.70 | 22.3 | 504 | 42.6% | |
| `regular` | 1 | 100% | 3.7% | 2.65 | 22.4 | 210 | 41.4% | 0.10 s |
| `jittered`, jitter 0.5, 2 sub-steps | 0.75 | 100% | 0% | 1.72 | 5.29 | 6.88 | 53.7% | 0.15 s |
| `poisson`, rate 1, cap 4, 4 sub-steps | 1.0 | 100% | 0% | 1.71 | 5.29 | 7.22 | 52.0% | 0.34 s |
| Jittered, jitter 0.9, 2 sub-steps | 0.95 | 100% | 0% | 1.86 | 5.32 | 6.93 | 53.9% | 0.16 s |
| Poisson, rate 1, cap 3, 3 sub-steps | 1.0 | 100% | 0% | 1.77 | 5.29 | 6.81 | 51.9% | 0.17 s |
| Regular grid, 2 sub-steps | 0.5 | 100% | 0% | 1.68 | 5.29 | 6.90 | 53.2% | 0.14 s |
| Regular grid, 4 sub-steps | 0.25 | 100% | 0% | 1.65 | 5.29 | 6.93 | 50.7% | 0.34 s |
| Unguarded jittered, 1 sub-step | 1.5 | 100% | 29.6% | 4.91 | 323 | 5,080 | 48.6% | 0.11 s |
| Unguarded poisson, 1 sub-step | 4.0 | 100% | 82.6% | 44.7 | 42,200 | 7.45e6 | 42.3% | 0.12 s |

A zero effect is `|y_int - y_obs| < 1e-6` at the query. The per-structure rows
are in `irregular_grids_measurement.json`. They show the same pattern. Times were taken with four workers on a shared machine under load
and are only comparable within this table.

**Sub-steps.** With every Euler sub-step at or below 1.0, no episode exceeds
|x| = 10 and the largest value is 7.2. Without sub-steps, the jittered gaps (up
to 1.5) put 29.6% of the episodes above 10 and the poisson gaps (up to 4) put
82.6% there. The mean-reversion rates of the prior reach 2, and an Euler step
of size h keeps a variable bounded only while rate × h ≤ 2. Continuous episodes
are never retried, so the guard is what keeps these values out of a suite.

**The regular third differs through its integration, not its grid.** The
regular grid integrated with 2 or 4 sub-steps matches the irregular candidates
in every column. The regular third keeps the single Euler step of
`dot-Continuous-v1` (h = 1), which sits on the stability edge for rates near 2.
Its amplitudes are larger (median max |x| 2.65 against about 1.7, and 3.7% of
episodes above 10), and the variance of the observational level at the query is
42.8 against 0.11 to 0.17. Its effects also decay more slowly after the window,
so fewer queries see a zero effect (41% against 51 to 54%). Scores
compared across the three thirds mix the grid with this difference. A `jittered`
entry with `jitter: 0` and `num_substeps: 2` is the regular grid with the
integration of the irregular thirds and needs no code change. It would isolate
the effect of the grid, but it is not in the prepared config.

**Chosen values.** `jittered` with `dt` 1, `jitter` 0.5 and 2 sub-steps gives
gaps in [0.5, 1.5] and sub-steps of at most 0.75, a moderate irregularity next
to the strong one of the poisson grid. Jitter 0.9 behaves the same. `poisson`
with rate 1, `max_gap` 4 and 4 sub-steps truncates 1.8% of the Exp(1) mass,
against 5.0% for a cap of 3, at about twice the cost per episode. Its mean gap
is 0.93, so its span averages 185 time units against 199 on the regular grid.

Reproduce with

```bash
PYTHONPATH=src python results/reference/continuous_irregular/measure_irregular_grids.py \
    --cache ~/.cache/dotime/dot-Continuous-v1-1.0.0 --workers 4
```

## Reference rows on the built suite

`continuous_irregular_cpu_effect.json` holds the pooled and per-structure rows of the
eight CPU baselines on the built `dot-ContinuousIrregular-v1` 1.0.0 (9,999 episodes),
scored with the packaged protocol (`dotime-eval-reference --dir-target effect`), and
`continuous_irregular_per_schedule.json` splits six of them by schedule
(`../audit_2026-09/scripts/per_group_baselines.py --group schedule`). Effect-sign
accuracy, 3,333 episodes per schedule, scored on the episodes with an effect of at
least 0.1 in absolute value:

| Schedule | n | Mean | AR1 | VAR-OLS | NaiveOLS | BackDoorOLS | IV2SLS |
|---|---|---|---|---|---|---|---|
| `regular` | 3333 | 0.606 | 0.584 | 0.567 | 0.733 | 0.652 | 0.625 |
| `jittered` | 3333 | 0.545 | 0.544 | 0.531 | 0.811 | 0.626 | 0.563 |
| `poisson` | 3333 | 0.544 | 0.546 | 0.561 | 0.829 | 0.634 | 0.567 |

The baselines treat the rows as equally spaced. The `regular` third scores lower for
NaiveOLS and higher for the naive baselines than the two irregular thirds. As the
measurement above explains, it keeps the single Euler step of `dot-Continuous-v1` and
its larger amplitudes, so the thirds differ in their integration as well as in their
grid, and scores should be compared within a third.
