# Pre-registration s14: a redesigned Do-Over-Time-PFN against the do-information ablation and the unadjusted in-context regression

Registered on 2026-10-15 by the commit that adds this file to the public dotime repository.
The timestamp is that commit and its GitHub push event, saved as push_event.json. Unlike s13, the
nine training runs of this re-test started before the registration, on 2026-10-14 at 18:17
Singapore time (SGT), because the training takes days on a shared server and the author chose to
start it while this document was written. What this registration therefore fixes before any
benchmark number exists is the scoring set, the prediction protocol, the endpoints, the decision
rules and the summary sentences, and it pins the nine checkpoints by digest (Section 10) so that
no run can be swapped, added or dropped afterwards. No released suite was scored by any of the
nine checkpoints before the push, and every training-prior read of them before the push is listed
in Section 10. Authors: the DoTime authors.

## 1. Question
The pre-registered re-test s13 (results/reference/prereg_2026-09) found that, under the published
recipe, a model trained with the intervention visible does not predict the sign of the
counterfactual effect more often than the same model trained with the intervention withheld
(label REVERSED, D = −0.007, 95% interval −0.012 to −0.001). The withdrawn claim of the submitted
paper stays withdrawn whatever this document finds.

A diagnostic programme after s13 (Section 2) found three defects of the published model that
make the effect sign unreadable by construction, and a pilot on the training prior found a
redesign that reads it on direct-effect structures. This document fixes the design, statistics and
decision rules of one confirmatory test of that redesign on the public benchmark. Two questions,
with one registered label each:

P1. Does the redesigned model trained with the intervention visible predict the sign of the
counterfactual effect more often than the same model trained on the same target with the
intervention withheld?

P2. Where does the redesigned interventional model stand against an unadjusted in-context
regression that reads the do-value (NaiveOLS) on the same episodes?

A positive P1 supports the narrower claim that the model uses the do-value. P2 says whether that
use reaches the simplest estimator that reads the do-value. Neither restores the withdrawn claim,
which concerned the published model and recipe.

## 2. Prior knowledge
1. s13: PFN_int 0.538, the ablation 0.544, PFN_obs 0.535 effect-sign accuracy on the 3,528
   primary episodes of dot-Identifiability-v1 1.2.0; NaiveOLS 0.764 and do-SVAR 0.762 on the same
   episodes.
2. Diagnostics (do-over-time-pfn results/diag_2026-10, October 2026): the do-value entered the
   published model uncentred (v / sd) while the history was centred, the encoder window ends one
   row before the onset so the treatment's factual onset value never reached the network, and the
   encoder pools each variable's history on its own, so no cross-variable statistic exists before
   the mixer. The effect sign in this prior is sign(slope) × sign(v − a_ref) with a random slope
   sign per system, so it cannot be read without that statistic. The slope-blind ceiling is 0.515
   on the benchmark's primary set and 0.55 on the training prior; 76% of the benchmark's primary
   do-values lie beyond 3 pre-onset standard deviations of the treatment, outside the published
   training clip.
3. Pilot (do-over-time-pfn PILOT_S14.md, committed before its first run, ten training runs on the
   training prior only, scored on a fixed 1,920-episode training-prior set with registered digests;
   no released suite was scored). Pooled effect-sign accuracy on the 796 scored episodes, floor
   0.607 (pre-onset mean), NaiveOLS 0.775, do-SVAR 0.802 on the same rows:

   | Rows | Wave | int | B | Flip rate | Anchor ratio | Slope agreement |
   |---|---|---|---|---|---|---|
   | 1, 2 | W1: centred do-value, onset value on the query, 6 sd training clip | 0.614 | 0.609 | 0.034 | 1.080 | 0.520 |
   | 5, 6 | W3 (a): joint cross-section token | 0.612 | 0.608 | 0.031 | 1.041 | 0.530 |
   | 3, 4 | W3 (b): regression statistics on the treatment per token | 0.657 | 0.608 | 0.130 | 1.051 | 0.658 |
   | 7, 8 | W3 (c): role vectors, path statistics, auxiliary loss 0.1 | 0.773 | 0.612 | 0.234 | 1.026 | 0.763 |
   | 9, 10 | W3 (c) with auxiliary loss 0.5 | 0.783 | 0.609 | 0.265 | 1.010 | 0.766 |

   The pilot's registered accuracy gate (pooled ≥ 0.75, ≥ B + 0.10, no structure below 0.60,
   slope agreement ≥ 0.65) was met by rows 7 and 9. Its registered mechanism gate was not: it
   required a mirror flip rate ≥ 0.80 and an anchor error ratio ≤ 1.0, and the gate records show
   that NaiveOLS itself scores 0.722 and 0.962 on those probes on the same rows, because for any
   model linear in the do-value the flip rate is bounded by its level error at the anchor. The pilot
   therefore ended, by its own rule, without a passing wave, and the author chose row 9's recipe
   for this confirmatory test with that record disclosed. The gain of the recipe on the training
   prior is confined to back_door (0.927 against NaiveOLS 0.917); front_door (0.633) and
   instrumental_variable (0.657) stay near the floor, and the slope agreement on front_door is 0.478.
4. The reason W3 (b) stopped at 0.70 on back_door and what W3 (c) changes: the encoder applies the
   same weights and positional encodings to every column and nothing adds a column identity, so the
   mixer's one-hot indices of the treated and the queried variable cannot address any token and the
   model averages the statistic tokens of every non-treated variable. W3 (c) marks the treated and
   the queried token with two learned vectors and gives every token the queried variable's
   regression on it given the treatment, so a mediated effect is a product of two features of one
   token.
5. Expectation stated in advance. If the training-prior behaviour carries to the benchmark, P1 is
   SUPPORTED and P2 is AT_BAR or BELOW_BAR, with the direct-effect structures above NaiveOLS and
   the mediated structures below it. We make no directional prediction for the non-identified
   structure. Before this registration the trainer's own evaluation loss on its seeded
   training-prior set was visible every 500 steps for every run; those values are in the run logs
   and do not involve the benchmark.
6. Smoke test of the scoring scripts, 2026-10-14 18:27 to 18:35 SGT: the two scripts of this folder
   ran on the first 40 episodes of 1.2.0 with two pilot checkpoints (rows 7 and 9 of the pilot)
   standing in for two seeds, to check the strict load, the probe columns, the analysis paths and
   the output files. None of the nine s14 checkpoints was involved, and the smoke outputs were
   discarded as results.

## 3. Hypotheses
H1. The interventional arm has a higher pooled effect-sign accuracy than the do-information
ablation (Arm B) on the primary scoring set.
H0. It does not.
P2 carries no hypothesis; it receives one of three labels (Section 7b).

## 4. Arms
All arms share recipe, seed, initialisation and training data stream. They differ only in the
training target and in whether the intervention is visible.

| Arm | Extra flags | Target | Intervention in the input |
|---|---|---|---|
| int | none | Y_true | visible |
| B, do-information ablation | --observational-only | Y_true | withheld |
| A, natural continuation | --observational-only --obs-only-target Y_obs | Y_obs | withheld |

Withheld means that intervention_target, intervention_type, intervention_value,
intervention_time_start and intervention_time_end are zeroed in training, in the trainer's
evaluation and at scoring time. Every arm receives the treatment's factual onset value as a scalar
on the query token (a_ref_norm, the onset value in the treatment's pre-onset frame); it is not an
intervention field and is not withheld. Because the zeroed intervention_target points at column 0,
the learned role vector of the treated variable marks column 0 in the withheld arms; this is an
artefact of the withheld protocol shared by B and A and is stated here rather than changed. Arm B
is the primary control because it shares the target of the int arm. The int minus A contrast is
the published one, and it also changes the target.

## 5. Recipe
Pilot row 9 of PILOT_S14.md, with the step count as the only change
(do-over-time-pfn scripts/run_s14_train.sh at the training commit of Section 10):
--config configs/server_s9_hardened_oscillatory.yaml --sim-device cpu --total-steps 20000
--batch-size 16 --head-type quantile --target-key Y_true --n-queries 10 --query-mode all_pairs
--causal-mask interpolation --intervention-source positivity_aware --early-stop-patience 0
--num-workers 4 --prefetch 4 --no-tscm-lag
--tscm-structures back_door front_door instrumental_variable --pair-mode counterfactual
--lr 2e-5 --token-lags 4 --readout mean_last --pos-init-std 1.0 --horizon-embed scalar
--center-intervention-value --query-a-ref --positivity-clip-sd 6 --eval-positivity-clip-sd 3
--stat-tokens ols_path --role-embed --value-aux-weight 0.5 --value-aux-anchor --seed S

The joint loader queries back_door at offset 0, front_door at offsets 1 to 5 and
instrumental_variable at offsets 0 to 5, one query per trajectory. Warmup and weight decay come
from the configuration file. The model has 26.25 million parameters (embedding width 512, six
encoder layers). The auxiliary loss applies to the int arm only, since the withheld arms have no
do-value to anchor. Seeds are 42, 43 and 44. The nine runs are s14ho_all_{int,B,A}_seed{42,43,44},
launched 2026-10-14 18:17 SGT with the primary pairs first. We score do_over_time_pfn_last.pt, the
weights at step 20,000.

Out of scope: the published recipe and model, structure-matched models, the break-trajectory-mean
prior, lagged edges, the trajectory-length sweep, and transfer to real data. A positive result
supports joint models of this redesign on the hardened oscillatory prior, scored by the effect sign
at the registered offsets on dot-Identifiability-v1 1.2.0, and nothing wider.

## 6. Scoring set
Suite: dot-Identifiability-v1 version 1.2.0 as released (Zenodo record 23095083); the manifest MD5
digests are recorded in the results. Primary set: every episode whose structure is bi_variate,
back_door, confounder_mediator, front_door, instrumental_variable or mediator and whose effect
satisfies |y_true − y_obs| ≥ 0.1, the same 3,528 episodes as s13. The factual level y_obs comes from
dotime.evaluation.query_obs_levels, and 0.1 is DIR_ACC_EPS. The set depends only on the suite.

Prediction: dotime.reference.pfn.PFNRef(checkpoint, observational = arm is not int) of the public
dotime package at the commit of Section 10, which builds the batch under the input convention
recorded in the checkpoint (the centred do-value and the onset value) and refuses a checkpoint
whose weights do not match its configuration. An episode is correct when sign(pred − y_obs) equals
sign(y_true − y_obs). A prediction equal to y_obs or not finite counts as incorrect. For the int
arms the scorer also records the prediction at the mirrored do-value 2 a_ref − v and at the anchor
a_ref, and for every episode the treatment's pre-onset mean and standard deviation; these feed
S10 and S11 only.

## 7. Primary endpoint and decision rule
acc(arm, s) is the fraction of correct episodes of the primary set, pooled over episodes, for seed
s. D_s = acc(int, s) − acc(B, s). The estimate D is the mean of D_42, D_43 and D_44.

Seed interval: D ± t·SD(D_s)/√3, with t = 4.303 for 95% and 2.920 for 90% (two degrees of freedom).
Episode interval: 10,000 paired bootstrap resamples of the primary set (numpy default_rng, seed
20261001). Each resample is applied to both arms and all seeds, and D is recomputed. Percentiles
2.5 and 97.5 give the 95% interval, and 5 and 95 give the 90% interval.

Exactly one label applies:
- SUPPORTED: the lower ends of both 95% intervals are above 0.
- REVERSED: the upper ends of both 95% intervals are below 0.
- EQUIVALENT: neither of the above, and both 90% intervals lie inside (−0.02, +0.02).
- INCONCLUSIVE: anything else.
If SUPPORTED, we also state whether D ≥ 0.02. If some structure in S1 differs significantly in the
opposite direction after Holm correction, we add the qualifier MIXED.

## 7b. Co-primary bar
E_s = acc(int, s) − acc(NaiveOLS) on the same primary episodes, where the NaiveOLS predictions are
the rows of the frozen detection-power file
results/reference/detection_power_2026-10/ident_v1_2_predictions.parquet (SHA-256
a83cb4d81b05aaac78c315016c6ac1d15d0102e65e02a25af1706c175722f65a, estimator NaiveOLS), graded by
the same rule. E is the mean over the three seeds. The seed interval uses the spread of E_s over the
int seeds; the episode interval is the same paired bootstrap, resampling the estimator's and the
model's correctness together. Exactly one label applies:
- ABOVE_BAR: the lower ends of both 95% intervals are above 0.
- AT_BAR: not ABOVE_BAR, and the lower ends of both 95% intervals are at or above −0.02.
- BELOW_BAR: anything else.
The same contrast against do-SVAR, GraphRouter, BackDoorOLS, FrontDoorOLS, IV2SLS and the pre-onset
mean is reported without a label.

## 8. Secondary endpoints
These are reported with intervals and carry no confirmatory claims.
S1. D per structure of the primary set, and pooled over the three training structures. Paired
    bootstrap p-values with Holm correction over six structures.
S2. int minus A, the published contrast.
S3. B minus A, the effect of the training target alone.
S4. Level root mean squared error (RMSE) with an episode-cluster bootstrap, and level-sign accuracy.
S5. Each CPU estimator of the detection-power analysis on the same primary episodes.
S6. Negative control: the non-identified structure bow_graph. Accuracy per arm and D.
S7. Training-prior check: the pilot's gate script on each int checkpoint (the fixed training-prior
    set of PILOT_S14.md), reported for comparability with the pilot.
S8. Sensitivity: do_over_time_pfn_best.pt, and effect thresholds of 0.05 and 0.2.
S9. E per structure of the primary set, with Holm correction over six structures.
S10. Mechanism probes of each int checkpoint on the primary set: the mirror flip rate (the sign of
     pred − y_obs flips when the do-value is reflected around a_ref), the median anchor error ratio
     (|pred(a_ref) − y_obs| over |mean_Y − y_obs|) and the slope-sign agreement
     (sign((pred(v) − pred(a_ref)) / (v − a_ref)) against the true sign).
S11. P1 and P2 on the in-support subset: primary episodes whose do-value lies within 3 pre-onset
     standard deviations of the treatment's pre-onset mean.
S12. Context: the published Chronos-2 result on this suite with and without the do-value as a
     covariate (results/reference/v1_2/chronos_ident_v1_2.json), pooled numbers only, since no
     per-episode Chronos-2 predictions exist.

## 9. Failures, exclusions and missing data
No episode is excluded beyond Section 6. A run is invalid only if its step-zero target check aborts,
if it crashes or produces non-finite losses before its final save, or if its code digest differs
from Section 10. An invalid run is relaunched once from scratch with identical flags under the
suffix _r2, and every relaunch is reported. No run is relaunched or excluded because of its scores.
If a relaunch fails again, that seed has no pair. The primary analysis then uses the complete int
and B pairs, with degrees of freedom equal to the number of pairs minus one. With fewer than two
complete pairs the primary endpoint is reported as not completed.

## 10. Provenance
Training code: do-over-time-pfn commit 2a37d3f (tag s14-train), a private repository.
code_manifest.sha256 lists the SHA-256 digest of every tracked file at that commit except the
directory of historical checkpoints the repository tracks, which plays no part in training.
ctp_manifest.sha256 is the s13 manifest of the causal_time_prior copy; every one of its 132 files
was re-hashed on the training host on 2026-10-14 and none had changed. Each run writes its command,
host, start time and code commit to results/<tag>/cmd.txt.

Checkpoints pinned at registration (SHA-256 of do_over_time_pfn_last.pt; all nine runs finished
between 2026-10-15 01:00 and 14:34 SGT, before this push):

| Run | Start (SGT) | SHA-256 |
|---|---|---|
| s14ho_all_int_seed42 | 2026-10-14 18:16 | 460cb1c325de86fcb2ce91e780b4864c04946e4cfbfaaeef2f756f467e1894db |
| s14ho_all_int_seed43 | 2026-10-14 18:17 | 96fbf741609571e5ffc188e9eeede9f2d75cb0648c634d98d4170bd0e98ab1f1 |
| s14ho_all_int_seed44 | 2026-10-15 01:08 | 79e3362df26c0afe2dbf23169444a3c5984f668b55726cb6318662b04724a7f8 |
| s14ho_all_B_seed42 | 2026-10-14 18:17 | 7f9234a38c6e6fe45eef713c238f0484dd582374a338c37067eb14b2994c6dab |
| s14ho_all_B_seed43 | 2026-10-15 01:02 | 8510f9a1fdbd589d7f7c11a26f55a2493fd3349badf4cb44878edff61b28094a |
| s14ho_all_B_seed44 | 2026-10-15 01:17 | 966244c7efd993d4f9cb8e25a491813f09648459e90a738a8eeec13a93394502 |
| s14ho_all_A_seed42 | 2026-10-15 07:39 | 39a209c2350c266bad1074f908782394c9a725bf5128ba8b802f1b47520f4789 |
| s14ho_all_A_seed43 | 2026-10-15 07:50 | dcff4214ffea4c30ce972d30ee2dca32166f739a2f96f069db88cb977d38faa6 |
| s14ho_all_A_seed44 | 2026-10-15 07:50 | bb09ebc28618afd81ae687a5e80c9dc3846fc9f6dd24a26b57cf58cb64a54068 |

Training-prior reads of these checkpoints before the push: the pilot's gate script (do-over-time-pfn scripts/pilot_gate.py at commit ac8a4ac) was run on the training-prior set of PILOT_S14.md for each finished int checkpoint on 2026-10-15, 13:24 to 13:28 SGT, before this push. s14ho_all_int_seed42 (training_prior_reads/s14ho_all_int_seed42.json): pooled effect sign 0.789 (back_door 0.915, front_door 0.633, instrumental_variable 0.700), flip rate 0.277, anchor ratio 1.019, slope agreement 0.778 s14ho_all_int_seed43 (training_prior_reads/s14ho_all_int_seed43.json): pooled effect sign 0.783 (back_door 0.920, front_door 0.628, instrumental_variable 0.674), flip rate 0.255, anchor ratio 1.023, slope agreement 0.768 s14ho_all_int_seed44 (training_prior_reads/s14ho_all_int_seed44.json): pooled effect sign 0.789 (back_door 0.930, front_door 0.622, instrumental_variable 0.683), flip rate 0.269, anchor ratio 1.010, slope agreement 0.769 These are secondary endpoint S7 and touch no released suite.

Scoring uses scripts/score_prereg.py and scripts/analyze_prereg.py in this folder, together with
the public dotime package at the commit of this registration (the scorer records the hash it ran on
in s14_scoring_provenance.json; a git archive of that commit carries it in a COMMIT file). Neither
script changes after registration. A
bug fix after unblinding is a deviation, and both versions are reported. The checkpoints go to the
public Hugging Face repository thummd/do-over-time-pfn under s14.

## 11. Freeze and reporting
The primary analysis runs once all six int and B checkpoints exist, and no later than 2026-10-20 12:00 SGT
on whatever is complete. results.json and README.md report every endpoint, including the
unfavourable ones. The summary sentence is composed of four parts in this order: the P1 sentence
of the label, the P2 sentence of the label, the claim sentence, and the closing sentence. L and U
are the outer ends of the seed interval and the episode interval of the contrast in question.

P1 sentences:
- SUPPORTED: "A redesigned Do-Over-Time-PFN with the intervention visible predicts the effect sign
  more often than the same model with it withheld, by D with a 95% confidence interval from L to U."
- REVERSED: "With the intervention visible the redesigned model predicts the effect sign less often
  than the same model with it withheld, by D with a 95% confidence interval from L to U."
- EQUIVALENT: "Interventional training changes the redesigned model's effect-sign accuracy by D,
  with a 95% confidence interval from L to U, inside the registered equivalence margin."
- INCONCLUSIVE: "Interventional training changes the redesigned model's effect-sign accuracy by D,
  with a 95% confidence interval from L to U, which is inconclusive."

P2 sentences:
- ABOVE_BAR: "It is E above an unadjusted in-context regression on the same episodes, with a 95%
  confidence interval from L to U."
- AT_BAR: "It is within the registered margin of an unadjusted in-context regression on the same
  episodes (E, 95% confidence interval from L to U)."
- BELOW_BAR: "It remains E below an unadjusted in-context regression on the same episodes, with a
  95% confidence interval from L to U."

Claim sentence: SUPPORTED, "This supports the narrower claim that the model uses the do-value.";
REVERSED or EQUIVALENT, "This does not support the narrower claim that the model uses the
do-value."; INCONCLUSIVE, "This neither supports nor rejects the narrower claim that the model uses
the do-value." Closing sentence, in every cell: "It does not restore the withdrawn claim." Not
completed: "The re-test did not complete by the registered deadline. It does not restore the
withdrawn claim."

## 12. Design sensitivity
Each seed scores about 3,500 episodes. If the arms disagree on a fraction d of episodes, the
within-seed standard error of D_s is about √(d/3500), which is 0.004 at d = 0.05 and 0.008 at
d = 0.2. With a seed standard deviation of 0.008, the 95% seed interval has a half-width of 0.02.
The training-prior gap of the pilot (0.17) would be SUPPORTED by a wide margin; a gap of 0.02 would
be at the edge. For P2 the seed spread of the int arm alone sets the seed interval, and a
difference of 0.02 from NaiveOLS cannot be told from zero with three seeds, which is why AT_BAR
exists as a label.

## 13. Deviations
DEVIATIONS.md records every departure from this document with a timestamp and a reason.
