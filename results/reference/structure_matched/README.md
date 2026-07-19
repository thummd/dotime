# Structure-matched OSC/BTM gap results (paper §6.5, Tables 5–6, App. E)

Interventional-vs-observational direction-accuracy gap for the structure-matched
Do-Over-Time-PFN checkpoints, produced with the upstream do-over-time-pfn eval
harness (`analyze_s9ho.py` -> `analyze_s7.evaluate_checkpoint`). Paper name map:
**OSC = `s9ho`**, **BTM = `s9btm`**.

Each `<tag>/T<t>.json` holds per-checkpoint metrics with `rmse`, `rmse_se`
(bootstrap), `direction_accuracy`, `direction_accuracy_se` (binomial),
`n_queries`.

| directory        | what                                                        | backs |
|------------------|-------------------------------------------------------------|-------|
| `s9ho_sweep/`    | seed 42, T∈{200,500,1000,2000}, n_batches=40                | Table 5 |
| `s9btm_sweep/`   | seed 42, T∈{200,500,1000,2000}, n_batches=40                | Table 6 (App. E) |
| `s9ho_scale/`    | seed 42, T=200, n_batches=60 (tighter single point)         | §6.5 T=200 prose (0.63/0.78); seed-42 point of the seed study |
| `s9btm_scale/`   | seed 42, T=200, n_batches=60                                 | BTM T=200 cross-check |
| `s9ho_seedvar/`  | seeds 43+44, all 6 OSC arms, T=200 (`seed_eval.json`)        | the three-seed result (pooled +0.089, seed std 0.005) |

Three-seed gap = combine `s9ho_scale` (seed 42) with the two seeds in
`s9ho_seedvar/seed_eval.json`; positive in all 9 structure×seed cells.

## Exploratory: two more identification structures (reviewer follow-up)

`s9ho_extra/extra_eval.json` — int-vs-obs gap for `observed_confounder`
(identified, back-door family) and `unobserved_confounder` (non-identifiable by
design), seed 42, T=200, n_batches=60, offset (0,0). Finding: the gap is small
and similar on both (+0.028±0.028 and +0.032±0.024) — it does **not** vanish
under non-identifiability. Both structures are near-saturated in direction
accuracy (int/obs ≈ 0.89–0.92), which compresses the achievable gap.

## Lagged replication (reviewer follow-up)

`s9ho_lag/lag_eval.json` — the same structure-matched int/obs protocol with
**lagged edges enabled in both training and evaluation** (checkpoints
`s9ho_{bd,fd,iv}_{causal,obs}_lag`, seed 42, T=200, n_batches=60). The gap
survives DoTime's temporal axis: back_door +0.116±0.026, front_door
+0.042±0.023, instrumental_variable +0.055±0.023, pooled +0.071 (vs +0.084
in the primary no-lag study at the same protocol point).
