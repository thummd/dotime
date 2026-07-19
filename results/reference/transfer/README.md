# Zero-shot transfer probes (paper §7.1)

Per-seed evaluation JSONs for the representativeness/transfer results reported in
the paper's Representativeness section — a synthetic-trained Do-Over-Time-PFN
applied zero-shot to two real interventional systems:

- `chamber_*_rpm_in.json` — CausalChamber wind-tunnel (`wt_intake_impulse_v1`),
  predicting `rpm_in` under an intake impulse, 5 training seeds (linear mechanism).
- `warfarin_*.json` — Warfarin plasma-concentration / PD trajectories, 5 seeds.
- `transfer_multiseed_aggregate.json` — across-seed mean ± std for every row.

Headline (linear, from the aggregate): chamber `rpm_in` RMSE 660.8 vs. naive
last-value 1264.8 (+47.8% ± 0.3% lift over 5 seeds), per-episode Pearson
r = 0.12 ± 0.17; Warfarin concentration Pearson r = 0.49 ± 0.69 (4/5 seeds in
[0.72, 0.89], one seed at −0.87).

Provenance: produced by the FMSD/continuous-time line (companion work
thumm2026towards); the full transfer pipeline lives in that codebase. These JSONs
are copied here so the §7.1 numbers are reproducible from a released artifact.
