#!/usr/bin/env bash
# Registered invocation of the lagged back-door scoring (PREREG.md, Section 4).
# Run from the repository root. PYTHON selects the interpreter.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
OUT=$ROOT/results/reference/lagged_backdoor_2026-10
STUDY=$ROOT/results/reference/detection_power_2026-10
PY=${PYTHON:-python}
export PYTHONPATH=$ROOT/src
# Gate 3 reads the study's gate records from the output directory, so the two
# registered records are copied there unchanged (never overwritten).
for f in ident_v1_1_gate.json self_test.json; do
  [ -e "$OUT/$f" ] || cp "$STUDY/$f" "$OUT/$f"
done
# Every registered row plus the two extensions, so that gate 3 compares the
# registered rows with the 1.1.0 table exactly as the study did.
"$PY" "$STUDY/scripts/detection_power.py" --version 1.2.0 --out "$OUT" --workers "${WORKERS:-8}" \
  --compare-to "$STUDY/ident_v1_1.json" \
  --estimators Zero Mean AR1 VAR-OLS NaiveOLS do-SVAR BackDoorOLS IV2SLS FrontDoorOLS \
    LaggedBackDoorOLS LaggedBackDoorOLS-L3 GraphRouter Oracle "$@" 2>&1 | tee "$OUT/run.log"
"$PY" "$OUT/scripts/grade_lagged.py" --predictions "$OUT/ident_v1_2_predictions.parquet" --out "$OUT" 2>&1 | tee "$OUT/grade.log"
