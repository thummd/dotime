# Erratum artifacts (v1.0.0 ledger, 2026-08)

Corrected evaluations behind `edit.tex` Appendix "Erratum ledger". The
published tables' sources are in `../structure_matched/` (unchanged).

| file / dir | what | backs |
|---|---|---|
| `ident_cpu_{level,effect}_realigned.json` | CPU baselines on `dot-Identifiability-v1` with realigned `x_obs` (sidecar), level- and effect-scored | erratum table, items 3+5 |
| `pfn_ident_{level,effect}_realigned.json` | published `s9ho_all_causal` vs `s9ho_all_obs` (degenerate-target obs arm), realigned inputs | erratum table decomposition |
| `pfn_ident_{level,effect}_realigned_s10obs.json` | `s9ho_all_causal` vs **retrained** `s10ho_all_obs` | corrected tab:results / fmgap rows |
| `s10_sweep/` | OSC structure-matched sweep: `s9ho_*_causal` vs retrained `s10ho_*_obs`; seed 42, T in {200,500,1000,2000}, n_batches=40, OSC hardening (`analyze_s10.py`) | corrected tab:gap (level + effect) |
| `s10btm_sweep/` | same for BTM, BTM hardening config | corrected tab:gapbtm |
| `gap_tables_corrected.json` | per structure x T: published gap, corrected level/effect gaps + SE, per-arm accuracies, n_queries; pooled rows query-weighted | the four corrected tables |

Retrained obs arms (`s10ho/s10btm x {bd,fd,iv,all}_obs`): exact s9 hyperparameters,
`--obs-only-target Y_obs` with the corrected (unmasked) `Y_obs`; the step-zero
target-QA banner in each `train.log` records nonzero fraction / mean / variance
per arm. Checkpoints live in the do-over-time-pfn repo (`checkpoints/s10*`).

Headline: with the observational arms trained on a non-degenerate target, no
structure x T cell in either prior shows an int-vs-obs gap beyond its SE.
