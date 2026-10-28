# Deviations from PREREG.md

Every departure from the pre-registration is listed here with a timestamp in Singapore time (SGT)
and a reason, in the order it happened.

1. **2026-10-28 16:35 SGT, registered fallback used.** The first invocation of `scripts/run.sh`
   failed gate 3 on last-digit arithmetic differences of this environment in eight RMSE cells of
   registered estimators (README.md, "Execution record"); nothing of the extensions was involved.
   PREREG.md Section 4 registers the fallback `--limit-per-structure 1350` for exactly this case,
   and it was run next. Listed here so that the record of the run is complete: the failed gate
   record and its log are kept, and the fallback changes no number of the registered analysis.
