# Lagged back-door set: graded predictions

Graded by `scripts/grade_lagged.py` from `ident_v1_2_predictions.parquet`. Differences are a minus b, paired over episodes, percentile bootstrap with 2000 replicates and seed 20261001. `false_effect` is the rate of `|pred(do v) - pred(do a_ref)| >= 0.1`, `effect_sign` the effect-sign accuracy on the episodes with a scorable effect.

| Id | Structure | Statistic | a | b | n | a | b | a minus b | 95% CI | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | `observed_confounder` | false_effect | 0.46444444444444444 | 0.4785185185185185 | 1350 | +0.464 | +0.479 | -0.014 | [-0.034, +0.007] | not confirmed |
| L1b | `observed_confounder` | false_effect | 0.46444444444444444 | 0.3637037037037037 | 1350 | +0.464 | +0.364 | +0.101 | [+0.078, +0.124] | not confirmed |
| L2 | `back_door` | effect_sign | 0.8060836501901141 | 0.8098859315589354 | 789 | +0.806 | +0.810 | -0.004 | [-0.019, +0.011] | confirmed |
| L3 | `confounder_mediator` | effect_sign | 0.6723646723646723 | 0.6780626780626781 | 351 | +0.672 | +0.678 | -0.006 | [-0.031, +0.020] | not confirmed |
| X1a | `observed_confounder` | false_effect | 0.4562962962962963 | 0.4785185185185185 | 1350 | +0.456 | +0.479 | -0.022 | [-0.047, +0.002] | reported |
| X1b | `observed_confounder` | false_effect | 0.4562962962962963 | 0.3637037037037037 | 1350 | +0.456 | +0.364 | +0.093 | [+0.071, +0.113] | reported |
| X1c | `back_door` | effect_sign | 0.7946768060836502 | 0.8098859315589354 | 789 | +0.795 | +0.810 | -0.015 | [-0.033, +0.004] | reported |
| X1d | `confounder_mediator` | effect_sign | 0.6666666666666666 | 0.6780626780626781 | 351 | +0.667 | +0.678 | -0.011 | [-0.049, +0.028] | reported |
| X1e | `back_door` | effect_sign | 0.7946768060836502 | 0.8060836501901141 | 789 | +0.795 | +0.806 | -0.011 | [-0.027, +0.004] | reported |
| X2a | `back_door` | effect_sign | 0.8060836501901141 | 0.7756653992395437 | 789 | +0.806 | +0.776 | +0.030 | [+0.003, +0.058] | reported |
| X2b | `confounder_mediator` | effect_sign | 0.6723646723646723 | 0.6581196581196581 | 351 | +0.672 | +0.658 | +0.014 | [-0.026, +0.054] | reported |
| X2c | `confounder_mediator` | effect_sign | 0.6723646723646723 | 0.7122507122507122 | 351 | +0.672 | +0.712 | -0.040 | [-0.083, +0.003] | reported |
| X3a | `observed_confounder` | false_effect | 0.46444444444444444 |  | 1350 | +0.464 |  |  |  | reported |
| X3b | `observed_confounder` | false_effect | 0.4562962962962963 |  | 1350 | +0.456 |  |  |  | reported |

Statements:

- L1: LaggedBackDoorOLS makes false effects on observed_confounder at a lower rate than BackDoorOLS (paired 95% CI of the difference below 0)
- L1b: LaggedBackDoorOLS makes false effects on observed_confounder at a lower rate than do-SVAR (paired 95% CI of the difference below 0)
- L2: on back_door the effect-sign accuracy of LaggedBackDoorOLS is not below BackDoorOLS's by more than 0.03 (lower bound of the paired 95% CI above -0.03)
- L3: on confounder_mediator the effect-sign accuracy of LaggedBackDoorOLS is not below BackDoorOLS's by more than 0.03 (lower bound of the paired 95% CI above -0.03)
- X1a: lag 3 against BackDoorOLS
- X1b: lag 3 against do-SVAR
- X1c: lag 3 against BackDoorOLS
- X1d: lag 3 against BackDoorOLS
- X1e: lag 3 against lag 1
- X2a: lag 1 against do-SVAR
- X2b: lag 1 against do-SVAR
- X2c: lag 1 against FrontDoorOLS
- X3a: lag 1 rate against the study's 0.10 bar
- X3b: lag 3 rate against the study's 0.10 bar
