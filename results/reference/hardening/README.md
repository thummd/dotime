# Generic-prior hardening (September 2026)

Measurement behind the opt-in `hardening` option of the generic prior
(`dotime.hardening`, `DoTime(config={"hardening": ...})`). No model was trained.

`measure_hardening.py` samples episodes exactly as the release build samples the
generic suite: a per-episode seed, the global torch seed, then
`DoTime(...).generate_pair(T=200)`. It does this under three variants:

- `base`: no hardening, which is the released v1.0.0 behaviour.
- `weights_only`: `unit_norm_rows` and `spectral_rho=0.9`.
- `recommended`: `RECOMMENDED_HARDENING`, which also sets `bounded_square`.

Hardening draws no random numbers, so all three variants see the same graphs,
interventions and noise, and the comparison is paired. An episode counts as
diverged when either arm is returned all-zero. The effect size is the true
causal effect under shared noise, taken as the largest post-onset
`|x_int - x_obs|` over non-target variables in units of that variable's standard
deviation. Per-arm target statistics (nonzero fraction, mean, variance) are
asserted before any number is used.

| Config | Variant | Diverged | Saturated survivors | Median N (all / survivors) | Effect >= 0.1 sd | Time per episode |
|---|---|---|---|---|---|---|
| N_max=10, K_max=3, 500 episodes | base | 29.6% | 0.9% | 6 / 6 | 87% | 0.33 s |
| | weights_only | 9.2% | 0% | 6 / 6 | 90% | 0.41 s |
| | recommended | 0.2% | 0% | 6 / 6 | 90% | 0.44 s |
| N_max=60, K_max=8, 300 episodes | base | 67.3% | 1.0% | 27 / 7 | 86% | 3.7 s |
| | weights_only | 12.7% | 0.8% | 27 / 26 | 94% | 11.2 s |
| | recommended | 0.0% | 0% | 27 / 27 | 95% | 11.5 s |

The full output, including divergence by SCM class and the per-arm target
statistics, is in `hardening_measurement.json`. Unhardened, 85% of the episodes
with more than 20 variables diverge and the survivors are small graphs (median
7 variables). With the recommended hardening, none diverge and the survivors
keep the size distribution of the prior.

The time column covers sampling and simulating one episode. Diverged episodes
stop early, so the unhardened time is not comparable at N_max=60. The
hardening step alone costs a median 2.5 ms at N_max=10 and 112 ms at N_max=60,
at most about 1 s, which is small against a full 60-variable simulation.

Regime-switching SCMs show 0% divergence in every variant because of a separate
defect. Their mechanisms read no parent weights, since the weights are keyed by
pre-remap node names, so each variable is independent noise. Hardening only
touches weights that a mechanism reads, so it is correct for them either way.

To reproduce, run from the repository root:

```
PYTHONPATH=src python results/reference/hardening/measure_hardening.py --workers 12
```
