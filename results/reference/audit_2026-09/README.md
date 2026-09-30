# Audit measurements (2026-09)

CPU-only measurements from the September 2026 audit of the v1 suites and the paper's
claims. No model was trained. The scripts that score targets log and assert per-arm
target statistics (nonzero fraction, mean, variance) before using them, and the scripts
that re-score released data first reproduce the published numbers, as noted per file.

| file | what | script |
|---|---|---|
| `ident_v1_1_per_structure.json` | CPU baselines on `dot-Identifiability-v1` 1.1.0 (shared-noise counterfactual targets): pooled and per-structure direction accuracy, both effect-scored and level-scored. Also level/effect sign agreement and the fraction of exactly-zero effects per structure. Gate: the pooled numbers reproduce `../v1_1/ident_cpu_{effect,level}.json` exactly. Computed with the old BackDoorOLS adjustment set (see the next row) | `scripts/ident_analysis.py` on the pre-fix package, with the released rows as `--ref-effect`/`--ref-level` |
| `ident_v1_1_per_structure_backdoor_fix.json` | The same breakdown after BackDoorOLS stopped adjusting for the mediator of `confounder_mediator`. Only the BackDoorOLS row differs: its effect sign on `confounder_mediator` rises from 0.499 to 0.678 (n = 351) and pooled from 0.601 to 0.621. Gate: the pooled numbers reproduce `../v1_1/ident_cpu_{effect,level}_backdoor_fix.json` exactly | `scripts/ident_analysis.py` |
| `ident_validity.json` | Invariant pass rates and per-arm target stats for Identifiability 1.0.0 (realigned with the sidecar; misalignment and hidden-variable leak measured on the raw files) and 1.1.0 | `scripts/ident_analysis.py` |
| `continuous_validity.json` | The same invariants for `dot-Continuous-v1` 1.0.0, read at the query index `round(query_time * (T - 1))` | `scripts/continuous_validity.py` |
| `scaling_lag.json` | Generic prior at (N_max, K_max) = (10, 3), (10, 8) and (60, 8), 1,000 episodes each, with `stability_retries` 0 and 3. Diverged fraction (both arms zeroed) overall and by realised K, generation time, realised N, and baseline scores by K. The per-K baseline scores are too noisy to cite | `scripts/scaling_lag.py` |
| `transfer_facts.json` | Re-derivation of the paper's §7.1 transfer-probe facts from `../transfer/` plus the mixed-mechanism chamber JSONs of the training runs (`--extra-dir`) | `scripts/transfer_analysis.py` |
| `half_diverged_released_scan.json` | Per-arm zeroed counts in the md5-verified v1.0.0 files of all four suites | `scripts/scan_released.py` |
| `half_diverged_regen_release.json` | Bit-exact regeneration of released Generic and RegimeSwitch rows (every one-arm row plus a random sample) and their retry sequences | `scripts/arm_sequences.py --mode release` |
| `half_diverged_test_config.json` | The same accounting for the 200-episode configuration of `tests/test_build_release.py` | `scripts/arm_sequences.py --mode testcfg` |
| `half_diverged_stationarity.json` | Divergence by spectral radius on the `dotime-diagnose-stationarity` sample | `scripts/arm_sequences.py --mode stationarity` |
| `half_diverged_scaling.json` | The `scaling_lag.py` configurations recounted under the both-arm and either-arm rules | `scripts/arm_sequences.py --mode scaling` |
| `half_diverged_generic_metric_impact.json` | Published `dot-Generic-100k` CPU rows re-scored without zeroed episodes. Gate: the unfiltered rows reproduce `../generic.json` | `scripts/generic_metric_impact.py` |
| `half_diverged_generic_target_qa.json` | Per-arm target statistics of `dot-Generic-100k` by zeroed-arm category | `scripts/target_qa.py` |
| `half_diverged_summary.json` | Digest of the `half_diverged_*` files above | |
| `frozen_regeneration.json` | Every row of `dot-Continuous-v1` 1.0.0 (9,999) and `dot-Identifiability-v1` 1.1.0 (10,800) regenerated with the current package and compared column by column with the md5-verified release files. All 11 data columns match on every row. `metadata_json` differs on every row because the releases predate keys such as `query_time_idx`, and every released key is present with an equal value | repository `scripts/fingerprint_frozen_suites.py --full` |
| `frozen_target_qa.json` | `dotime.qa.target_qa` with the default thresholds and `dir_target="effect"` on all five cached frozen suite versions, pooled and per structure (regime density for RegimeSwitch). Identifiability 1.0.0 reads its observational level from the realignment sidecar after checking every row against its episode. All five pass. The tightest effect cells are Identifiability 1.1.0 `confounder_mediator` and `front_door` (0.66 and 0.67 nonzero, saturating mechanisms) and Continuous `back_door` and `instrumental_variable` (0.51, queries of variables A does not reach). The script writes nothing if a version fails | `scripts/frozen_target_qa.py` |

To reproduce, run from the repository root with the dev environment. The scripts read
the cached v1.0.0 suites (`~/.cache/dotime/`) and, for 1.1.0, a local build in
`output/v1_1/`.

```
python results/reference/audit_2026-09/scripts/ident_analysis.py
python results/reference/audit_2026-09/scripts/continuous_validity.py
python results/reference/audit_2026-09/scripts/scaling_lag.py --n 1000 --workers 12
python results/reference/audit_2026-09/scripts/transfer_analysis.py --extra-dir <training-runs>/results/phase14c_multiseed
python scripts/fingerprint_frozen_suites.py --full
python results/reference/audit_2026-09/scripts/frozen_target_qa.py
```

## Findings fixed since

- `evaluation.query_obs_levels` rounded `query_time * T`, but the continuous suite encodes
  `query_time` as `idx / (T - 1)`. On `dot-Continuous-v1` it read one step late for query
  indices at or above 100. Fixed through `SuiteMetadata.query_time_encoding` (see
  `CHANGELOG.md`).
- The `dotime._build` divergence flag and retry gate counted a pair as diverged only when
  both arms were zeroed. The either-arm fractions of the v1.0.0 files are 30.1% (Generic)
  and 4.7% (Identifiability), not 28.7% and 4.6%. Fixed to the either-arm rule.
