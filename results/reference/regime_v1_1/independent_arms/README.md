# The repair with independent arms (superseded, never released)

These files record the first candidate of dot-RegimeSwitch-v1 version 1.1.0, built on
2026-11-27 SGT. It repaired the mechanisms of version 1.0.0 and kept its pairing: the two arms of
an episode were independent draws. The candidate was valid and did not separate estimators, which
is why version 1.1.0 pairs its arms under shared noise instead (`../README.md`). Nothing here
was released.

The candidate, its config and its scripts are those of commit `488ebc7`. The release config
there has the hash `338d087ab99e29ef`, and the two shards had the digests
`27102dd4dfc6427f3a007f05281bb51b` and `7dfb513c3f6d54ec164fcbefa3e1c326` in two builds with 12
and 13 workers. To reproduce a file of this folder, check out that commit and run the command
that its `results/reference/regime_v1_1/README.md` gives.

| File | Content |
|---|---|
| `design_pilot.json` | `effect_persistence.py` of `488ebc7` on the first 200 episodes of each regime density |
| `build_validation.json` | `validate_build.py` of `488ebc7` on the candidate |
| `regime_v1_1_cpu.json` | Reference rows of the candidate |

## What held

`build_validation.json` asserts all of this on the 9,999 stored episodes.

- **Targets.** The per-arm target statistics passed `dotime.qa.target_qa`.
- **No divergence.** No episode had a zeroed arm.
- **Repair.** Every episode recorded `reads_parents: true`. 31.2% of the variables had a lag-one
  autocorrelation above 0.3 in the observational arm. Without the repair no variable of 600
  measured episodes does (`../../regime_weights/`).
- **Query.** Every query sat at the last step of its intervention window, on an untreated
  variable. The window closed a median 33 steps before the last step.
- **Reproducibility.** 300 randomly sampled episodes regenerated exactly from the config and the
  seeds. 299 of them were first attempts and had the regime graphs and the intervention of the
  version 1.0.0 episode with the same seed.

## What the pilot measured

With independent arms the gap between the arms mixes the effect of the intervention with noise.
The pilot therefore simulated each interventional arm twice under common random numbers, once
with the intervention and once with a reference. A soft intervention was contrasted with the
observational run, a hard or time-varying one with a hard intervention at the treated
variable's pre-onset mean. On 600 episodes, with the effect threshold 0.1 of the benchmark:

| Share of episodes in which the intervention moved ... by at least 0.1 | At the last step | At the window's end |
|---|---|---|
| the queried variable | 6.0% | 47.7% |
| some untreated variable | 9.5% | 78.3% |

The queried variable was the one with the largest gap between the two independent arms, and it
was the most affected variable in 29.0% of the episodes.

## Why it was superseded

No reference estimator beat the pre-onset mean (`regime_v1_1_cpu.json`, level scoring, 9,999
episodes, 95% cluster-bootstrap intervals).

| Estimator | RMSE | 95% interval | Sign of the level |
|---|---|---|---|
| Zero | 2.350 | 2.094 to 2.761 | |
| Mean | 2.277 | 2.037 to 2.663 | 0.605 |
| AR1 | 2.851 | 2.611 to 3.215 | 0.567 |
| VAR-OLS | 2.456 | 2.219 to 2.832 | 0.583 |
| NaiveOLS | 3.067 | 2.764 to 3.437 | 0.589 |
| BackDoorOLS, IV2SLS | 2.277 | 2.037 to 2.663 | 0.605 |

BackDoorOLS and IV2SLS do not apply to these structures and return the mean. The estimator that
reads the intervention value, NaiveOLS, was the worst. The reason is the pairing, not the
repair. With independent arms the observational history tells a model which SCM it faces and
nothing about the noise of the interventional arm. The best possible prediction is the
interventional mean, and the intervention shifts that mean by little against the spread of the
target (variance 5.4). A per-episode effect needs arms that share their noise.
