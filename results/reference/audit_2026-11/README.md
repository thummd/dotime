# Audit measurements (2026-11)

Reference rows that the published paper reports and that no earlier folder held.
No model was trained. Every run logs and asserts the per-arm target statistics
(nonzero fraction, mean and variance of the observational level, the
interventional level and the effect) before it scores anything.

## How it fits the pipeline

`dotime-eval-reference` pools a suite. `scripts/group_rows.py` calls the same
functions (`dotime.reference.reference_table.run_baseline` and `target_qa`) per
level of one design factor, so a pooled row here and a row of the reference table
are the same computation. With `--reference` the script first checks that its
pooled rows reproduce an existing `dotime-eval-reference` file and stops if they
do not. A level without a scored episode is written as `null`.

| File | Contents | Command |
|---|---|---|
| `wide_per_lag.json` | dot-Wide-v1 1.0.0 by the lag of the shortest path from the treatment to the query (`metadata["graph"]["path"]`) | `python scripts/group_rows.py --suite dot-Wide-v1 --version 1.0.0 --group min_lag --reference ../wide/wide_cpu_effect.json --out wide_per_lag.json` |
| `continuous_cpu_effect.json` | dot-Continuous-v1 1.0.0, effect-scored, every query | `dotime-eval-reference --suite dot-Continuous-v1 --version 1.0.0 --dir-target effect --out continuous_cpu_effect.json` |
| `continuous_cpu_effect_noself.json` | The same without queries on the treated variable | the command above with `--exclude-self-queries` |
| `continuous_irregular_per_schedule_noself.json` | dot-ContinuousIrregular-v1 1.0.0 by observation schedule, without queries on the treated variable | `python scripts/group_rows.py --suite dot-ContinuousIrregular-v1 --version 1.0.0 --group schedule --exclude-self-queries --out continuous_irregular_per_schedule_noself.json` |

The level error of `continuous_cpu_effect.json` equals `../continuous.json` for
every baseline that file holds. NaiveOLS is new in this file.

## dot-Wide-v1 by path lag

Effect-sign accuracy. "Best naive" is the highest of Zero, Mean, AR1 and VAR-OLS.
The pooled row reproduces `../wide/wide_cpu_effect.json`.

| Path lag | Episodes | Scored | Best naive | NaiveOLS |
|---|---|---|---|---|
| 0 | 5,040 | 4,633 | 0.629 | 0.679 |
| 1 | 4,009 | 3,571 | 0.596 | 0.568 |
| 2 | 633 | 541 | 0.567 | 0.529 |
| 3 | 144 | 110 | 0.627 | 0.555 |
| 4 or more | 102 | 72 | 0.597 | 0.597 |
| Unreachable | 72 | 0 | not defined | not defined |
| All | 10,000 | 8,927 | 0.612 | 0.624 |

The contemporaneous regression leads the naive forecasters only when the shortest
path has lag zero. From lag one on it is at or below the pre-onset mean. An
unreachable query has an effect of exactly zero and is never scored.

## Self-queries inflate the continuous-time rows

One third of the queries of the two continuous-time suites target the treated
variable. Inside the intervention window such a target equals the intervention
value, so any estimator that reads that value is right by construction.

| Suite | Queries | Scored | Best naive | NaiveOLS |
|---|---|---|---|---|
| dot-Continuous-v1 | all | 3,751 | 0.591 | 0.730 |
| dot-Continuous-v1 | without self-queries | 1,953 | 0.584 | 0.525 |
| dot-ContinuousIrregular-v1 | without self-queries | 1,672 | 0.547 | 0.593 |

By schedule, without self-queries, NaiveOLS scores 0.612 on the jittered grid,
0.649 on the Poisson grid and 0.532 on the regular grid. The rows of
`../continuous_irregular/continuous_irregular_per_schedule.json` include
self-queries and read 0.811, 0.829 and 0.733. Effect-sign rows of the
continuous-time suites should be quoted without self-queries.
