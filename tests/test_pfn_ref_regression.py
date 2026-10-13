"""The reference PFN scorer reproduces the pre-registered s13 predictions.

``results/reference/prereg_2026-09/s13_predictions.parquet`` holds every
prediction of the s13 scoring (``scripts/score_prereg.py`` of that folder),
made with :class:`dotime.reference.pfn.PFNRef` on dot-Identifiability-v1
1.2.0. Any later change to the scorer, the vendored model or the batch
construction must reproduce a stratified sample of those rows to float32 precision, which
pins the legacy input convention while the recovery-program options are added.
The test needs the cached suite and the public s13 checkpoints, and skips
cleanly without them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("pfns", reason="model package needs the [models] extra")
pd = pytest.importorskip("pandas")

from dotime.benchmarks import _cache_root

_PARQUET = Path(__file__).resolve().parents[1] / (
    "results/reference/prereg_2026-09/s13_predictions.parquet"
)
_CKPT_DIR = Path(os.path.expanduser("~/.cache/dotime-tools/s13_checkpoints_public"))
_SUITE_DIR = _cache_root(None) / "dot-Identifiability-v1-1.2.0"
_ROWS_PER_STRUCTURE = 3


@pytest.mark.slow
@pytest.mark.parametrize("tag", ["s13ho_all_int_seed42", "s13ho_all_B_seed42"])
def test_pfn_ref_reproduces_the_registered_s13_predictions(tag):
    from dotime.benchmarks import load_benchmark
    from dotime.reference.pfn import PFNRef

    ckpt = _CKPT_DIR / tag / "do_over_time_pfn_last.pt"
    if not (ckpt.exists() and _SUITE_DIR.exists() and _PARQUET.exists()):
        pytest.skip("needs the s13 checkpoints, the cached 1.2.0 suite and the predictions")
    rows = pd.read_parquet(_PARQUET)
    rows = rows[rows["tag"] == tag]
    # A fixed stratified sample: the first rows of every structure in file order.
    sample = rows.groupby("structure", sort=True).head(_ROWS_PER_STRUCTURE)
    assert len(sample) == 9 * _ROWS_PER_STRUCTURE
    suite = load_benchmark("dot-Identifiability-v1", version="1.2.0")
    by_id = {int(ep.scm_id): ep for ep in suite if int(ep.scm_id) in set(sample["scm_id"])}
    model = PFNRef(str(ckpt), observational="_int_" not in tag)
    assert model.convention == {}, "s13 checkpoints predate the data block: legacy convention"
    for _, r in sample.iterrows():
        ep = by_id[int(r["scm_id"])]
        assert ep.structure == r["structure"]
        pred = float(torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1)[0])
        # The registered predictions were made on a GPU; CPU float32 accumulation
        # differs in the eighth digit (about 3e-8 here). A convention change moves
        # a prediction by orders of magnitude more than this tolerance.
        assert pred == pytest.approx(float(r["pred"]), abs=1e-6, rel=0.0), (
            f"{tag} scm {r['scm_id']}: {pred!r} != registered {r['pred']!r}"
        )
