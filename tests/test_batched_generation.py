"""``generate_batch`` for named TSCM structures: supported modes and diverged samples.

For a named ``tscm_structure``, ``ExtendedDoTime.generate_batch`` builds batches with
the vectorized simulator, which implements only ``prior`` and ``positivity_aware``.
The ``observed_*`` modes used to be ignored there, so prior values came back
silently. They must now (a) raise before any random number is drawn, (b) raise when
``TemporalInterventionDataLoader`` is constructed, since an error in its prefetch
thread would hang the training loop, and (c) keep working on the per-sample paths.
A diverged sample is replaced by ``generate_sample``, and (d) that replacement must
not crash the collate step when it lands at index 0.
"""

from __future__ import annotations

import re

import pytest
import torch

from dotime.data import TemporalInterventionDataLoader
from dotime.extended import BATCHED_INTERVENTION_SOURCES, ExtendedDoTime

UNBATCHED_MODES = ("observed", "observed_discrete", "observed_normal", "observed_uniform")


def _rng_states(gen: ExtendedDoTime) -> tuple:
    """Snapshot every RNG stream a call on a named-structure generator can draw from.

    Args:
        gen: A generator built with a named ``tscm_structure``.

    Returns:
        A tuple of the NumPy ``RandomState`` state, the prior's torch generator state
        and the global torch RNG state, with arrays as bytes so tuples compare with ``==``.

    Raises:
        AttributeError: If ``gen`` wraps the generic prior, which has no ``gen`` stream.
    """
    kind, keys, pos, has_gauss, cached = gen.rng.get_state()
    return (
        kind,
        keys.tobytes(),
        pos,
        has_gauss,
        cached,
        gen.prior.gen.get_state().numpy().tobytes(),
        torch.get_rng_state().numpy().tobytes(),
    )


@pytest.mark.parametrize("mode", UNBATCHED_MODES)
def test_generate_batch_rejects_unbatched_modes_before_any_draw(mode):
    """An observed mode with a named structure raises and leaves every RNG stream intact.

    Args:
        mode: An ``intervention_source`` outside ``BATCHED_INTERVENTION_SOURCES``.

    Returns:
        None.

    Raises:
        AssertionError: If no error is raised, or if a stream advanced first.
    """
    gen = ExtendedDoTime(tscm_structure="back_door", seed=0, intervention_source=mode)
    before = _rng_states(gen)
    # T=None makes generate_batch call sample_T, so this also pins the check ahead of it.
    with pytest.raises(NotImplementedError, match=re.escape(f"intervention_source={mode!r}")):
        gen.generate_batch(4)
    assert _rng_states(gen) == before


@pytest.mark.parametrize("mode", UNBATCHED_MODES)
def test_per_sample_paths_keep_every_mode(mode):
    """generate_sample applies an observed mode, and the generic prior still batches it.

    Args:
        mode: An ``intervention_source`` outside ``BATCHED_INTERVENTION_SOURCES``.

    Returns:
        None.

    Raises:
        AssertionError: If the per-sample value equals the prior draw, or the generic
            prior's batch has the wrong size.
    """
    values = []
    for source in ("prior", mode):
        torch.manual_seed(0)
        gen = ExtendedDoTime(tscm_structure="back_door", seed=0, intervention_source=source)
        values.append(float(gen.generate_sample(T=80)["intervention_value_raw"]))
    # Same seeds, so the episode is shared and only the resampled value can differ.
    assert values[1] != values[0]

    batch = ExtendedDoTime(seed=0, intervention_source=mode).generate_batch(2, T=40)
    assert batch["X_int"].shape[0] == 2


@pytest.mark.parametrize(
    "structure_kwargs",
    [{"tscm_structure": "back_door"}, {"tscm_structures": ["back_door", "front_door"]}],
    ids=["single", "multi"],
)
def test_dataloader_rejects_unbatched_modes_at_construction(structure_kwargs):
    """The loader raises in the caller's thread, before its prefetch thread exists.

    Args:
        structure_kwargs: The single-structure or multi-structure loader arguments.

    Returns:
        None.

    Raises:
        AssertionError: If construction does not raise ``NotImplementedError``.
    """
    with pytest.raises(NotImplementedError, match="observed_normal"):
        TemporalInterventionDataLoader(
            num_steps=1, batch_size=2, intervention_source="observed_normal", **structure_kwargs
        )


@pytest.mark.parametrize(
    ("structure_kwargs", "mode"),
    [({"tscm_structure": "back_door"}, m) for m in BATCHED_INTERVENTION_SOURCES]
    + [({}, "observed_normal")],
    ids=[*BATCHED_INTERVENTION_SOURCES, "generic-observed_normal"],
)
def test_dataloader_accepts_supported_configurations(structure_kwargs, mode):
    """Batched modes with a named structure, and any mode on the generic prior, still load.

    Args:
        structure_kwargs: Loader arguments selecting a named structure, or none for
            the generic prior (the quickstart configuration).
        mode: The ``intervention_source`` to load with.

    Returns:
        None.

    Raises:
        AssertionError: If the loader does not yield one batch of the requested size.
    """
    loader = TemporalInterventionDataLoader(
        num_steps=1,
        batch_size=2,
        t_range=(40, 60),
        prefetch=0,
        intervention_source=mode,
        **structure_kwargs,
    )
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0]["X_obs"].shape[0] == 2


@pytest.mark.parametrize("n_queries", [1, 3])
def test_batch_survives_a_diverged_first_sample(monkeypatch, n_queries):
    """A diverged sample 0 is replaced per sample without breaking the batch's keys.

    Args:
        monkeypatch: Pytest fixture used to force the first sample to diverge.
        n_queries: Queries per trajectory. ``3`` exercises the multi-query collate.

    Returns:
        None.

    Raises:
        AssertionError: If collation fails, keys differ from an undiverged batch, or
            anything other than sample 0 changed.
    """
    torch.manual_seed(0)
    reference = ExtendedDoTime(tscm_structure="back_door", seed=0).generate_batch(
        4, T=60, n_queries=n_queries
    )

    gen = ExtendedDoTime(tscm_structure="back_door", seed=0)
    real_generate_pairs = gen.batched_sim.generate_pairs

    def first_sample_diverged(*args, **kwargs):
        """Run the real simulator, then mark sample 0 as diverged.

        Args:
            *args: Positional arguments for ``BatchedTSCMSimulator.generate_pairs``.
            **kwargs: Keyword arguments for ``BatchedTSCMSimulator.generate_pairs``.

        Returns:
            The simulator's pair dict with ``valid[0]`` set to ``False``.

        Raises:
            Nothing beyond what the real simulator raises.
        """
        pairs = real_generate_pairs(*args, **kwargs)
        pairs["valid"][0] = False
        return pairs

    monkeypatch.setattr(gen.batched_sim, "generate_pairs", first_sample_diverged)
    torch.manual_seed(0)
    batch = gen.generate_batch(4, T=60, n_queries=n_queries)

    assert set(batch) == set(reference)
    assert "X_obs_full" not in batch
    # Only sample 0 was regenerated. Samples 1-3 are the same vectorized draws.
    assert not torch.equal(batch["X_int"][0], reference["X_int"][0])
    assert torch.equal(batch["X_int"][1:], reference["X_int"][1:])
    assert torch.equal(batch["X_obs"][1:], reference["X_obs"][1:])
