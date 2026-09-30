# dot-Observed-v1: reference rows (September 2026)

Reference rows of the CPU baselines on `dot-Observed-v1` 1.0.0 (10,800 episodes: 12
observation cells, 9 structures, 100 latent episodes per structure and cell), scored with
the packaged protocol. `dotime-eval-reference --dir-target effect` imputes the missing
cells (forward fill, then the pre-onset mean) for models that cannot read masks, and the
effect sign is scored against the latent factual level `y_obs_latent`. No model was
trained.

| File | Contents | Script |
|---|---|---|
| `observed_cpu_effect.json` | Pooled and per-structure rows of the eight CPU baselines, with the per-arm target statistics | `dotime.reference.reference_table` |
| `observed_per_cell.json` | Effect-sign accuracy of six baselines per observation cell | `../audit_2026-09/scripts/per_group_baselines.py --group obs_cell` |

Effect-sign accuracy per cell, 900 episodes each, scored on the episodes with an
effect of at least 0.1 in absolute value:

| Cell | n | Mean | AR1 | VAR-OLS | NaiveOLS | BackDoorOLS | IV2SLS |
|---|---|---|---|---|---|---|---|
| `none+none` | 900 | 0.559 | 0.585 | 0.540 | 0.796 | 0.623 | 0.575 |
| `none+mcar10` | 900 | 0.559 | 0.597 | 0.559 | 0.773 | 0.617 | 0.565 |
| `none+block` | 900 | 0.553 | 0.575 | 0.537 | 0.796 | 0.613 | 0.575 |
| `none+mnar` | 900 | 0.534 | 0.550 | 0.514 | 0.716 | 0.565 | 0.537 |
| `snr10+none` | 900 | 0.556 | 0.575 | 0.550 | 0.783 | 0.623 | 0.578 |
| `snr10+mcar10` | 900 | 0.556 | 0.575 | 0.546 | 0.754 | 0.613 | 0.565 |
| `snr10+block` | 900 | 0.550 | 0.569 | 0.537 | 0.770 | 0.613 | 0.569 |
| `snr10+mnar` | 900 | 0.530 | 0.543 | 0.521 | 0.690 | 0.562 | 0.527 |
| `snr3+none` | 900 | 0.550 | 0.575 | 0.546 | 0.757 | 0.610 | 0.556 |
| `snr3+mcar10` | 900 | 0.556 | 0.578 | 0.543 | 0.738 | 0.604 | 0.562 |
| `snr3+block` | 900 | 0.543 | 0.556 | 0.537 | 0.738 | 0.610 | 0.553 |
| `snr3+mnar` | 900 | 0.537 | 0.550 | 0.543 | 0.652 | 0.562 | 0.530 |

Findings:

1. The clean cell `none+none` reproduces the 1.2.0 rows of the same base episodes
   (`../audit_2026-09/identity_2026_10.json`), and its NaiveOLS accuracy (0.796) is the
   ceiling of the design.
2. Missingness that is not at random costs the most. Under `mnar` NaiveOLS loses 8 to
   11 points against the matching cell without missingness, while `mcar10` and `block`
   cost 0 to 3 points at the same or a lower missing rate.
3. Measurement noise costs 1 to 6 points for NaiveOLS at a signal-to-noise ratio of 10
   and 4 to 6 points at 3. The two factors add up: the hardest cell, `snr3+mnar`, scores
   0.652 against 0.796 on the clean cell.
4. BackDoorOLS moves from 0.623 to 0.562 over the same cells, and the naive baselines
   stay between 0.51 and 0.60 throughout, so the ranking of the estimator classes does
   not change with the observation model.

Reproduce from the repository root:

```
PYTHONPATH=src python -m dotime.reference.reference_table --suite dot-Observed-v1 --version 1.0.0 --dir-target effect --out results/reference/observed/observed_cpu_effect.json
PYTHONPATH=src python results/reference/audit_2026-09/scripts/per_group_baselines.py --suite dot-Observed-v1 --version 1.0.0 --group obs_cell --out results/reference/observed/observed_per_cell.json
```
