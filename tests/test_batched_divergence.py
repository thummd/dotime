"""Diverged samples and noise pairing in ``generate_batch`` for named TSCM structures.

``divergence_fallback="sequential"`` (the default for interventional pairs) replaces a
diverged sample with ``generate_sample``, a different simulator that ignores
``hardening`` and draws its own intervention time. ``"batched"`` must instead (a)
redraw only the diverged slots, with the batch's own simulator, intervention time and
validity check, (b) leave every other slot and every later batch unchanged, and (c)
fail loudly when a configuration keeps diverging. ``pair_mode="counterfactual"`` must
give shared-noise pairs in batches too, and the loader must pass both options on.
"""

from __future__ import annotations

import pytest
import torch

import dotime.extended as extended
from dotime.batched_tscm import DIVERGENCE_THRESHOLD, BatchedTSCMSimulator
from dotime.data import TemporalInterventionDataLoader
from dotime.extended import ExtendedDoTime
from dotime.tscm_sampler import TSCMStructure


class _PairsSpy:
    """Stand-in for ``generate_pairs`` that records calls and can force divergence."""

    def __init__(self, sim: BatchedTSCMSimulator, force=(), keep=(), always: bool = False):
        """Wrap ``sim.generate_pairs``.

        Args:
            sim: The simulator whose ``generate_pairs`` is wrapped.
            force: Sample indices marked diverged in the first call only.
            keep: Sample indices marked valid in the first call only, so a test does
                not depend on which samples diverge naturally.
            always: Mark every sample of every call diverged.

        Returns:
            None.

        Raises:
            Nothing.
        """
        self.real = sim.generate_pairs
        self.force = tuple(force)
        self.keep = tuple(keep)
        self.always = always
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        """Run the real simulator, then apply the forced validity.

        Args:
            *args: Positional arguments for ``generate_pairs``.
            **kwargs: Keyword arguments for ``generate_pairs``.

        Returns:
            The simulator's pair dict, with ``valid`` edited as configured.

        Raises:
            Nothing beyond what the real simulator raises.
        """
        out = self.real(*args, **kwargs)
        if self.always:
            out["valid"][:] = False
        elif not self.calls:
            for i in self.force:
                out["valid"][i] = False
            for i in self.keep:
                out["valid"][i] = True
        self.calls.append(dict(kwargs, valid=out["valid"].clone()))
        return out


def _forbid_per_sample(*args, **kwargs):
    """Stand-in for ``generate_sample`` that fails the test if the per-sample path runs.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        Never returns.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError("divergence_fallback='batched' must not call generate_sample")


def _slot(batch: dict, s: int) -> dict:
    """Every field of trajectory ``s``, including its query rows in a multi-query batch.

    Args:
        batch: A collated ``generate_batch`` output.
        s: Trajectory index.

    Returns:
        A dict mapping each field to the part that belongs to trajectory ``s``.

    Raises:
        KeyError: If the batch lacks a field the multi-query layout requires.
    """
    if "_traj_idx" not in batch:
        return {k: v[s] for k, v in batch.items()}
    rows = batch["_traj_idx"] == s
    return {k: (v[s] if v.shape[0] != rows.numel() else v[rows]) for k, v in batch.items()}


def _slots_equal(a: dict, b: dict) -> bool:
    """Whether two per-trajectory dicts hold bit-identical tensors.

    Args:
        a: Output of :func:`_slot`.
        b: Output of :func:`_slot`.

    Returns:
        ``True`` if both have the same keys and equal tensors.

    Raises:
        Nothing.
    """
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _pairs(**kwargs) -> dict:
    """One ``generate_pairs`` call on a fresh back-door simulator.

    Args:
        **kwargs: Extra keyword arguments for ``generate_pairs``.

    Returns:
        The pair dict for ``B=8``, ``T=60``, ``seed=5``.

    Raises:
        ValueError: If ``generate_pairs`` rejects the arguments.
    """
    sim = BatchedTSCMSimulator(TSCMStructure.BACK_DOOR)
    return sim.generate_pairs(B=8, T=60, seed=5, **kwargs)


def test_int_time_override_moves_only_the_onset():
    """``int_time`` replaces the drawn onset and changes nothing drawn before it.

    Returns:
        None.

    Raises:
        AssertionError: If the onset, values or pre-onset trajectories differ wrongly.
    """
    base = _pairs()
    t0 = int(base["int_time"][0])
    t1 = t0 + 7
    moved = _pairs(int_time=t1)
    assert (moved["int_time"] == t1).all()
    assert torch.equal(moved["int_value"], base["int_value"])
    assert torch.equal(moved["X_obs"], base["X_obs"])
    # Same mechanisms and interventional noise: identical until the earlier onset.
    assert torch.equal(moved["X_int"][:, :t0], base["X_int"][:, :t0])
    a = int(moved["int_target"][0])
    ok = moved["valid"]
    assert torch.equal(moved["X_int"][ok, t1, a], moved["int_value"][ok])


@pytest.mark.parametrize("int_time", [-1, 60])
def test_int_time_outside_the_trajectory_raises(int_time):
    """An onset the simulation would never reach is rejected.

    Args:
        int_time: An intervention time outside ``[0, T)`` for ``T=60``.

    Returns:
        None.

    Raises:
        AssertionError: If no ``ValueError`` is raised.
    """
    with pytest.raises(ValueError, match="int_time"):
        _pairs(int_time=int_time)


def test_shared_noise_pairs_agree_before_onset():
    """``shared_noise=True`` gives counterfactual pairs and leaves ``X_obs`` untouched.

    Returns:
        None.

    Raises:
        AssertionError: If the arms differ before the onset, or the observational arm
            or the values changed.
    """
    twins = _pairs()
    shared = _pairs(shared_noise=True)
    t0 = int(shared["int_time"][0])
    ok = shared["valid"]
    assert ok.any()
    assert torch.equal(shared["X_obs"], twins["X_obs"])
    assert torch.equal(shared["int_value"], twins["int_value"])
    assert torch.equal(shared["X_obs"][ok, :t0], shared["X_int"][ok, :t0])
    assert not torch.equal(twins["X_obs"][ok, :t0], twins["X_int"][ok, :t0])


@pytest.mark.parametrize("spike", [50.0, float("nan"), float("inf")])
def test_check_recorded_window_flags_values_the_periodic_check_misses(monkeypatch, spike):
    """A tail value the 50-step check never sees invalidates the sample only on request.

    Args:
        monkeypatch: Pytest fixture used to inject the value.
        spike: The value written at the last recorded step of one valid sample.

    Returns:
        None.

    Raises:
        AssertionError: If the flag ignores the value, or other samples change.
    """
    sim = BatchedTSCMSimulator(TSCMStructure.BACK_DOOR)
    i = int(torch.nonzero(sim.generate_pairs(B=8, T=60, seed=5)["valid"])[0])
    real = sim.simulate

    def spiking(*args, **kwargs):
        """Run the real simulation and spike the interventional arm's last step.

        Args:
            *args: Positional arguments for ``simulate``.
            **kwargs: Keyword arguments for ``simulate``.

        Returns:
            The recorded trajectories and the unchanged validity mask.

        Raises:
            Nothing beyond what the real simulation raises.
        """
        X, valid = real(*args, **kwargs)
        if kwargs.get("int_target") is not None:
            X = X.clone()
            X[i, -1, 0] = spike
        return X, valid

    monkeypatch.setattr(sim, "simulate", spiking)
    lax = sim.generate_pairs(B=8, T=60, seed=5)
    strict = sim.generate_pairs(B=8, T=60, seed=5, check_recorded_window=True)
    others = torch.arange(8) != i
    assert bool(lax["valid"][i])
    assert not bool(strict["valid"][i])
    assert torch.equal(strict["valid"][others], lax["valid"][others])
    kept = strict["valid"]
    for X in (strict["X_obs"][kept], strict["X_int"][kept]):
        assert torch.isfinite(X).all()
        assert X.abs().max() <= DIVERGENCE_THRESHOLD


def test_redraw_seed_is_pure_and_separates_rounds():
    """Redraw seeds are reproducible, in range and never reuse a neighbour's generator.

    Returns:
        None.

    Raises:
        AssertionError: If a seed repeats, leaves ``[0, 2**31)``, lands on another
            round's ``seed + 1`` or disturbs a global RNG.
    """
    torch_state = torch.get_rng_state()
    seeds = [extended._redraw_seed(b, a) for b in (0, 1, 2**31 - 1) for a in range(1, 21)]
    assert seeds == [extended._redraw_seed(b, a) for b in (0, 1, 2**31 - 1) for a in range(1, 21)]
    assert all(0 <= s < 2**31 for s in seeds)
    # generate_pairs seeds generators with seed and seed + 1: no overlap across rounds.
    footprint = [s for seed in seeds for s in (seed, seed + 1)]
    assert len(set(footprint)) == len(footprint)
    assert torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(ValueError):
        extended._redraw_seed(-1, 1)


def test_divergence_fallback_defaults_and_validation():
    """``None`` resolves per pair mode, and inconsistent combinations are rejected.

    Returns:
        None.

    Raises:
        AssertionError: If a default resolves wrongly or a combination is accepted.
    """
    assert ExtendedDoTime(tscm_structure="back_door").divergence_fallback == "sequential"
    cf = ExtendedDoTime(tscm_structure="back_door", pair_mode="counterfactual")
    assert cf.divergence_fallback == "batched"
    opt_in = ExtendedDoTime(tscm_structure="back_door", divergence_fallback="batched")
    assert opt_in.divergence_fallback == "batched"
    with pytest.raises(ValueError, match="divergence_fallback"):
        ExtendedDoTime(tscm_structure="back_door", divergence_fallback="drop")
    with pytest.raises(ValueError, match="counterfactual"):
        ExtendedDoTime(
            tscm_structure="back_door", pair_mode="counterfactual", divergence_fallback="sequential"
        )
    with pytest.raises(NotImplementedError, match="named"):
        ExtendedDoTime(tscm_structure=None, divergence_fallback="batched")


@pytest.mark.parametrize("n_queries", [1, 3])
def test_batched_redraw_replaces_only_diverged_slots(monkeypatch, n_queries):
    """A diverged slot is redrawn at the batch onset, and nothing else changes.

    Non-degenerate query offsets make every slot draw from ``gen.rng``. The legacy
    per-sample replacement draws the same offsets, so equal RNG states after one call
    in each mode show that later batches keep the T and seed they have today.

    Args:
        monkeypatch: Pytest fixture used to force slot 2 to diverge.
        n_queries: Queries per trajectory. ``3`` exercises the multi-query collate.

    Returns:
        None.

    Raises:
        AssertionError: If a kept slot, the keys or the RNG state differ from the
            references, or the redrawn slot is off-onset, reseeded from ``gen.rng``
            or came from generate_sample.
    """
    kwargs = dict(tscm_structure="front_door", seed=0, query_offset_range=(1, 5))
    call = dict(batch_size=8, T=60, n_queries=n_queries)
    reference_gen = ExtendedDoTime(**kwargs, divergence_fallback="batched")
    reference_spy = _PairsSpy(reference_gen.batched_sim)
    monkeypatch.setattr(reference_gen.batched_sim, "generate_pairs", reference_spy)
    reference = reference_gen.generate_batch(**call)

    legacy_gen = ExtendedDoTime(**kwargs, divergence_fallback="sequential")
    legacy_spy = _PairsSpy(legacy_gen.batched_sim, force=[2], keep=[0])
    monkeypatch.setattr(legacy_gen.batched_sim, "generate_pairs", legacy_spy)
    legacy_gen.generate_batch(**call)

    gen = ExtendedDoTime(**kwargs, divergence_fallback="batched")
    spy = _PairsSpy(gen.batched_sim, force=[2], keep=[0])
    monkeypatch.setattr(gen.batched_sim, "generate_pairs", spy)
    monkeypatch.setattr(gen, "generate_sample", _forbid_per_sample)
    batch = gen.generate_batch(**call)

    assert set(batch) == set(reference)
    assert gen.rng.get_state()[2] == legacy_gen.rng.get_state()[2]
    assert (gen.rng.get_state()[1] == legacy_gen.rng.get_state()[1]).all()
    assert spy.calls[1]["seed"] == extended._redraw_seed(spy.calls[0]["seed"], 1)
    both_valid = spy.calls[0]["valid"] & reference_spy.calls[0]["valid"]
    for s in range(8):
        if both_valid[s]:
            assert _slots_equal(_slot(batch, s), _slot(reference, s))
    assert not torch.equal(batch["X_int"][2], reference["X_int"][2])
    assert (batch["int_onset_idx"] == batch["int_onset_idx"][0]).all()
    assert spy.calls[0]["check_recorded_window"]
    assert not spy.calls[0]["shared_noise"]
    redraws = spy.calls[1:]
    assert redraws
    onset = int(batch["int_onset_idx"][0])
    assert all(c["int_time"] == onset and c["check_recorded_window"] for c in redraws)


def test_batched_redraw_keeps_the_batch_hardening(monkeypatch):
    """With hardening, replacements come from the hardened batched simulator.

    Args:
        monkeypatch: Pytest fixture used to force divergence and spy on calls.

    Returns:
        None.

    Raises:
        AssertionError: If the per-sample path runs, or a redraw targets another
            simulator or another onset.
    """
    hardening = dict(max_lag=2, spectral_rho=0.9, add_self_memory_lags=True, noise_std=0.05)
    gen = ExtendedDoTime(
        tscm_structure="back_door", seed=1, hardening=hardening, divergence_fallback="batched"
    )
    spy = _PairsSpy(gen.batched_sim, force=[0, 5])
    monkeypatch.setattr(gen.batched_sim, "generate_pairs", spy)
    monkeypatch.setattr(gen, "generate_sample", _forbid_per_sample)
    batch = gen.generate_batch(8, T=60)
    assert gen.batched_sim.max_lag == 2
    assert gen.batched_sim.spectral_rho == 0.9
    assert (batch["int_onset_idx"] == batch["int_onset_idx"][0]).all()
    assert all(c["int_time"] == int(batch["int_onset_idx"][0]) for c in spy.calls[1:])
    assert sum(int(c["valid"].sum()) for c in spy.calls[1:]) >= 2


def test_batched_redraw_raises_when_it_keeps_diverging(monkeypatch):
    """A configuration that never produces a valid sample fails instead of looping.

    Args:
        monkeypatch: Pytest fixture used to make every sample diverge.

    Returns:
        None.

    Raises:
        AssertionError: If no ``RuntimeError`` is raised or the round count is off.
    """
    gen = ExtendedDoTime(tscm_structure="bi_variate", seed=0, divergence_fallback="batched")
    spy = _PairsSpy(gen.batched_sim, always=True)
    monkeypatch.setattr(gen.batched_sim, "generate_pairs", spy)
    with pytest.raises(RuntimeError, match="still diverged"):
        gen.generate_batch(4, T=30)
    assert len(spy.calls) == 1 + extended._MAX_REDRAW_ROUNDS


def test_sequential_default_keeps_the_per_sample_fallback(monkeypatch):
    """The default still replaces a diverged interventional sample via generate_sample.

    Slot 0 is kept valid so the test does not depend on the separate ``X_obs_full``
    collate fix for a diverged first sample.

    Args:
        monkeypatch: Pytest fixture used to force divergence and count fallbacks.

    Returns:
        None.

    Raises:
        AssertionError: If the fallback count or the redraw count is off.
    """
    gen = ExtendedDoTime(tscm_structure="back_door", seed=0)
    assert gen.divergence_fallback == "sequential"
    spy = _PairsSpy(gen.batched_sim, force=[1], keep=[0])
    monkeypatch.setattr(gen.batched_sim, "generate_pairs", spy)
    real = gen.generate_sample
    calls = []

    def counting(*args, **kwargs):
        """Count the per-sample fallback and delegate to it.

        Args:
            *args: Positional arguments for ``generate_sample``.
            **kwargs: Keyword arguments for ``generate_sample``.

        Returns:
            The real ``generate_sample`` output.

        Raises:
            Nothing beyond what ``generate_sample`` raises.
        """
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gen, "generate_sample", counting)
    gen.generate_batch(8, T=60)
    assert len(calls) == int((~spy.calls[0]["valid"]).sum()) >= 1
    assert len(spy.calls) == 1
    # The legacy call: no window check and independent noise, as before the option existed.
    assert not spy.calls[0]["check_recorded_window"]
    assert not spy.calls[0]["shared_noise"]


@pytest.mark.parametrize("struct", ["back_door", "front_door", "instrumental_variable"])
def test_counterfactual_batches_share_noise_before_onset(struct):
    """Every slot of a counterfactual batch agrees with its twin before the onset.

    Args:
        struct: A named structure, with and without a hidden confounder.

    Returns:
        None.

    Raises:
        AssertionError: If any slot's arms differ before the onset, or the effect is
            not the difference of the two arms at the query.
    """
    gen = ExtendedDoTime(tscm_structure=struct, seed=2, pair_mode="counterfactual")
    batch = gen.generate_batch(8, T=80)
    for s in range(8):
        onset, n = int(batch["int_onset_idx"][s]), int(batch["num_vars"][s])
        assert torch.equal(batch["X_obs"][s, :onset, :n], batch["X_int"][s, :onset, :n])
    effect = batch["Y_true"] - batch["Y_obs"]
    assert torch.allclose(batch["Y_causal_effect"], effect, atol=1e-6)

    # "batched" keeps this comparison off the per-sample path, whose diverged-slot-0
    # collate crash is fixed separately.
    twins = ExtendedDoTime(tscm_structure=struct, seed=2, divergence_fallback="batched")
    twin_batch = twins.generate_batch(8, T=80)
    onset, n = int(twin_batch["int_onset_idx"][0]), int(twin_batch["num_vars"][0])
    assert not torch.equal(twin_batch["X_obs"][:, :onset, :n], twin_batch["X_int"][:, :onset, :n])


@pytest.mark.parametrize(
    ("structure_kwargs", "pair_mode", "fallback", "resolved"),
    [
        ({"tscm_structure": "back_door"}, "interventional", "batched", "batched"),
        ({"tscm_structures": ["back_door", "front_door"]}, "counterfactual", None, "batched"),
    ],
    ids=["single-opt-in", "multi-counterfactual"],
)
def test_loader_passes_pair_mode_and_divergence_fallback(
    structure_kwargs, pair_mode, fallback, resolved
):
    """The loader hands both options to every prior and still yields batches.

    Args:
        structure_kwargs: Single- or multi-structure loader arguments.
        pair_mode: The loader's ``pair_mode``.
        fallback: The loader's ``divergence_fallback``.
        resolved: The fallback every prior must end up with.

    Returns:
        None.

    Raises:
        AssertionError: If a prior has other settings or no batch comes out.
    """
    loader = TemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        t_range=(40, 60),
        prefetch=0,
        pair_mode=pair_mode,
        divergence_fallback=fallback,
        **structure_kwargs,
    )
    for prior in loader.priors or [loader.prior]:
        assert prior.pair_mode == pair_mode
        assert prior.divergence_fallback == resolved
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0]["X_obs"].shape[0] == 2
