# Troubleshooting

Common install / runtime snags and their fixes.

## Install

**`torch` / CUDA mismatch.** The core package depends on `torch>=2.0` but does not
pin a CUDA build. If `import torch` fails or silently runs on CPU when you expect
GPU, install the matching wheel from the
[PyTorch site](https://pytorch.org/get-started/locally/) *before* installing
`causaltime`.

**`ModuleNotFoundError: pyarrow`.** Reading/writing frozen suites (parquet) needs
the `evaluation` extra:

```bash
pip install 'causaltime[evaluation]'
```

**`The 'gdp' encoder backend requires the optional [gdp] extra`.** The default
encoder backend is `transformer` and runs on CPU with no extra dependency. The
`gdp` (GatedDeltaProduct / TempoPFN) backend is GPU-only and needs:

```bash
pip install 'causaltime[gdp]'   # requires a CUDA GPU + flash-linear-attention
```

Unless you explicitly pass `backend="gdp"`, you never need this.

## Runtime

**`DoOverTimePFN baseline needs a trained checkpoint`.** Pass a checkpoint path:

```python
from causaltime import baselines
model = baselines.get("DoOverTimePFN", checkpoint="/path/to/best.pt")
```

The `[models]` extra (`pfns`) must be installed for the model to import.

**`RuntimeWarning: SCM diverged ... returning zeros`.** The diverse prior
occasionally samples an unstable SCM; those trajectories are zeroed and flagged
rather than dropped. The empirical divergence rate is < 1% on the released suites;
the warnings are safe to ignore (or filter with `warnings.simplefilter("ignore")`).

## Benchmark cache

`load_benchmark` caches downloaded suites under
`~/.cache/causaltime` (override with `$CAUSALTIME_CACHE` or the
`cache_dir=` argument). If a cached suite is corrupt you will see a
`checksum mismatch` error — delete the suite directory and reload with
`force_download=True`.
