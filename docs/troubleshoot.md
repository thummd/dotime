# Troubleshooting

Common install / runtime snags and their fixes.

## Install

**`torch` / CUDA mismatch.** The core package depends on `torch>=2.0` but does not
pin a CUDA build. If `import torch` fails or silently runs on CPU when you expect
GPU, install the matching wheel from the
[PyTorch site](https://pytorch.org/get-started/locally/) *before* installing
`dotime`.

**`ModuleNotFoundError: pyarrow`.** Reading/writing frozen suites (parquet) needs
the `evaluation` extra:

```bash
pip install 'dotime[evaluation]'
```

**`The 'gdp' encoder backend requires the optional [gdp] extra`.** The default
encoder backend is `transformer` and runs on CPU with no extra dependency. The
`gdp` (GatedDeltaProduct / TempoPFN) backend is GPU-only and needs:

```bash
pip install 'dotime[gdp]'   # requires a CUDA GPU + flash-linear-attention
```

Unless you explicitly pass `backend="gdp"`, you never need this.

## Runtime

**`DoOverTimePFN baseline needs a trained checkpoint`.** Pass a checkpoint path:

```python
from dotime import baselines

model = baselines.get("DoOverTimePFN", checkpoint="/path/to/best.pt")
```

The `[models]` extra (`pfns`) must be installed for the model to import.

**`RuntimeWarning: SCM diverged ... returning zeros`.** The diverse prior
occasionally samples an unstable SCM. The observational and interventional arms
are simulated separately and each diverged arm is replaced by zeros, so an
episode can have one arm or both zeroed. In the released v1.0.0 suites both arms
are zeroed in **28.7% of `dot-Generic-100k`** (28,734 of 100,000) and **4.6% of
`dot-Identifiability-v1`** (497 of 10,800). Counting episodes with either arm
zeroed, the fractions are **30.1%** (30,111) and **4.7%** (504). RegimeSwitch and
Continuous have no zeroed arm. A zeroed interventional arm stores `y_true == 0`,
and a zeroed observational arm leaves an all-zero history in front of a nonzero
target, so filter on **both** arms (`x_obs.abs().max() > 0 and
x_int.abs().max() > 0`) rather than ignore them. The v1.0.0 files carry no flag;
suites built with the current generator flag either case with
`metadata["diverged"]`. To build a divergence-free suite instead, pass
`--stability-retries 20` to `scripts/build_release.py` (opt-in deterministic
resampling of any episode with a zeroed arm; it produces a *different*, non-v1
suite). The warnings themselves are safe to filter with
`warnings.simplefilter("ignore")`.

**Diverged samples in training batches.** For a named `tscm_structure`,
`ExtendedDoTime.generate_batch` (and so `TemporalInterventionDataLoader`) replaces
every diverged sample. Without hardening that is 3.5 to 6.2% of samples. For
interventional pairs the default, `divergence_fallback="sequential"`, keeps the v1
behaviour that released checkpoints were trained with: the replacement comes from
`generate_sample`, whose per-sample simulator differs from the batched one, ignores
`hardening` and draws its own intervention time. Pass `divergence_fallback="batched"`
to redraw diverged samples with the batch's own simulator instead. This produces
different training data, so it does not reproduce the released checkpoints.
Counterfactual batches (`pair_mode="counterfactual"`) always redraw this way.

## Benchmark cache

`load_benchmark` caches downloaded suites under
`~/.cache/dotime` (override with `$DOTIME_CACHE` or the
`cache_dir=` argument). If a cached suite is corrupt you will see a
`checksum mismatch` error — delete the suite directory and reload with
`force_download=True`.
