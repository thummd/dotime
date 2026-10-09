# Signature-space diagnostic, Stage I-a: DoTime 1.2.0 primary set (October 2026)

A diagnostic for the "magnitude transfer" thread: which coordinates of a truncated path
signature of the pre-onset history carry the counterfactual effect of a hard intervention, and
how far does a sparse linear model on them get against the in-context estimators on the same
episodes? Script `scripts/signature_stage1a.py`, output `stage1a_dotime.json`, both under the
public `dotime` package with no new dependencies (signatures by Chen's relation in numpy, LASSO
from scikit-learn). Data: the 3,528 primary episodes of `dot-Identifiability-v1` 1.2.0 (six
structures, true effect at least 0.1, digest `f35883465575c702`, the set of the pre-registered
s13 re-test).

## Design

* **Pre-onset path only.** The (X, I) cross term of a signature over the whole window equals the
  do-value times the outcome's post-onset displacement, which is the target, so the path ends at
  the onset.
* **The do-value is an interaction.** The effect in this prior is `slope(history) × (v − a_ref)`,
  so every feature enters twice, alone and multiplied by `dv = (v − a_ref) / sd_A`. The sparse
  model then selects which geometry coordinates modulate the dose response.
* **End state as a feature.** Signatures describe increments and carry no absolute location, so
  `(x_last − mean) / sd` is added per variable.
* **Target** `(y_true − y_obs) / sd_Y`. Effect-sign accuracy from 5-fold cross-validated
  predictions of a LASSO at the alpha LassoCV picks on the full set (a mild optimism, the same for
  every variant). Stable coordinates: selection frequency at least 0.75 over 40 half-sample refits.
* **Variants.** State only; signature of the time-augmented normalised (t, A, Y) path at levels 2
  and 3; the same at level 3 without time; the raw (unnormalised) path; the log-signature; and,
  per structure, the full observed path (all non-zero columns) at levels 2 and 3.

## Results

Pooled over the six structures, out-of-sample effect-sign accuracy (binomial SE 0.008):

| Features of the pre-onset history | Accuracy | RMSE of the normalised effect | Non-zero / features |
|---|---|---|---|
| State only (`dv`, end state, their product) | 0.547 | 1.924 | 0 / 5 |
| Normalised signature of (t, A, Y), level 2 | 0.582 | 1.873 | 21 / 29 |
| **Normalised signature of (t, A, Y), level 3** | **0.676** | **1.730** | 35 / 83 |
| Same, level 3, without the time coordinate | 0.577 | 1.872 | 26 / 33 |
| Raw (unnormalised) path, level 3, with time | 0.659 | 1.762 | 29 / 83 |
| Normalised log-signature, level 3, with time | 0.638 | 1.791 | 4 / 83 |

References on the same episodes (`detection_power_2026-10/ident_v1_2_predictions.parquet`):
naive forecasters 0.522 to 0.534, NaiveOLS 0.764, do-SVAR 0.762, GraphRouter 0.724, BackDoorOLS
0.612. The slope-blind ceiling (a model that reads sign(v − a_ref) perfectly but not the slope) is
0.515. The s13 PFN arms score 0.538 and 0.544.

Per structure, level-3 (t, A, Y) signature versus the full observed path at level 3:

| Structure | n | (t, A, Y) level 3 | Full path level 3 | NaiveOLS | do-SVAR |
|---|---|---|---|---|---|
| bi_variate | 846 | 0.704 | 0.704 | 0.849 | 0.823 |
| back_door | 789 | 0.700 | 0.649 | 0.788 | 0.776 |
| instrumental_variable | 837 | 0.716 | 0.689 | 0.812 | 0.798 |
| mediator (queried one step after onset) | 334 | 0.781 | 0.781 | 0.536 | 0.701 |
| front_door | 371 | 0.526 | 0.539 | 0.698 | 0.663 |
| confounder_mediator | 351 | 0.527 | 0.513 | 0.672 | 0.658 |

### Which coordinates

On bi_variate, back_door and instrumental_variable the same four coordinates are selected in
every half-sample refit, with the same sign and similar standardised coefficients: `dv × S(tAY)`,
`dv × S(AYt)`, `dv × S(YAt)`, `dv × S(tYA)` (coefficients 1.3 to 2.8), and two small negative ones,
`dv × S(YtA)` and `dv × S(AtY)`. These are the level-3 words that order the treatment's and the
outcome's increments against time: time-weighted co-movement of A and Y, a nonparametric lagged
slope. Without the time coordinate the level-2 words `dv × S(AY)`, `dv × S(YA)` remain and the
accuracy falls from 0.676 to 0.577, so the time weighting is what carries the slope. The
log-signature compresses the same information into three coordinates (`dv × log(AtY)` dominant)
at 0.638. On the mediator structure, whose effect is lagged by construction, `dv × S(AY)` alone
gives 0.781, above both NaiveOLS (contemporaneous, 0.536) and do-SVAR (0.701).

### Where the linear functional fails

front_door and confounder_mediator stay at chance in every variant, including the full observed
path at level 3, where the LASSO selects nothing. The effect there is the product of two slopes
(A→M and M→Y), a degree-2 functional of the path statistics. A linear functional of a signature
truncated at level 3 cannot represent a product of two level-2 or level-3 coordinates; by the
shuffle-product identity that needs words of length 4 to 6, or a nonlinear readout. FrontDoorOLS
computes exactly that product and scores 0.720 on front_door.

## Reading against the hypotheses of the signature note

1. **Scale.** Within this one domain, normalising the path changes little (0.676 against 0.659),
   as expected: scale enters through `dv`, and the pipeline already works in pre-onset standard
   deviations. The diagnostic says nothing yet about cross-domain scale; that is Stage I-b.
2. **Geometry carries the slope, and the same geometry across structures.** The selected words
   and their signs are the same on the three direct-effect structures. That is the "same features,
   same signs" outcome the note describes, here across identification structures of one prior.
3. **The gap to the explicit regression is real but bounded.** A structure-agnostic linear
   functional of (t, A, Y) geometry recovers 0.676 where NaiveOLS, which is told to regress Y on A
   and its own lag, reaches 0.764. It beats the contemporaneous regression where the effect is
   lagged (mediator) and fails where the effect is composed (front_door, confounder_mediator).
4. **For wave W3 of the recovery program.** The statistic tokens of W3 (b) are a hand-picked
   member of this feature class (the OLS slope, which is what NaiveOLS computes). A "signature
   tokens" variant (level-3 words of (t, A, X_j) per variable, projected and added to the tokens)
   is the basis-free generalisation and the natural next variant if W3 (b) passes its gate; the
   PFN's nonlinear readout would also lift the degree-2 limitation that stops the linear
   functional on the mediated structures.

## Caveats

The LASSO alpha is chosen on the full set, so the cross-validated accuracies carry a mild optimism
shared by all variants. Only the (t, A, Y) path was run pooled, because column layouts differ
across structures; full-path models are per structure. Level 4 was not run. This is DoTime only:
the cross-domain comparison with the reactor logs and the market simulator (Stage I-b) has to
state that its targets are factual post-intervention displacements, not counterfactual effects.
