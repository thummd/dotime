"""Background prefetching in ``TemporalInterventionDataLoader``.

The default ``prefetch=2`` path generates batches in a daemon thread. A
generation error must reach the consumer as the original exception, after the
batches generated before it, instead of killing the thread and leaving
``queue.get()`` blocked forever. A consumer that stops early must not strand the
producer on ``queue.put``. Neither change may perturb the RNG stream, so the
prefetch path must yield exactly the synchronous path's batches.
"""

from __future__ import annotations

import threading

import pytest
import torch

import dotime.extended
from dotime.data import TemporalInterventionDataLoader

# Generous: a healthy run finishes in well under a second, and a regression
# otherwise blocks forever, so the margin only has to absorb a loaded CI box.
_HANG_TIMEOUT_S = 15.0
# Avoids the unrelated vectorized-collate KeyError ('X_obs_full', raised when
# the first sample of a batch diverges) for the few batches these tests draw.
_SEED = 0


def _make_loader(num_steps: int = 3, prefetch: int = 2) -> TemporalInterventionDataLoader:
    """Build the small back-door loader shared by these tests.

    Args:
        num_steps: Number of batches the loader yields.
        prefetch: Queue depth of the background producer. 0 is synchronous.

    Returns:
        A loader with batch size 4 on the vectorized back-door path.

    Raises:
        ValueError: Propagated from the loader if the configuration is invalid.
    """
    return TemporalInterventionDataLoader(
        num_steps=num_steps,
        batch_size=4,
        tscm_structure="back_door",
        prefetch=prefetch,
        seed=_SEED,
    )


def _assert_same_batches(got: list[dict], want: list[dict]) -> None:
    """Assert two batch lists match key for key and bit for bit.

    Args:
        got: Batches under test.
        want: Reference batches.

    Returns:
        None.

    Raises:
        AssertionError: If the lists differ in length, keys, or any tensor.
    """
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.keys() == w.keys()
        assert all(torch.equal(g[k], w[k]) for k in w)


def _consume_with_timeout(loader: TemporalInterventionDataLoader) -> tuple[list, BaseException]:
    """Drain ``loader`` in a helper thread so a hang fails the test instead of CI.

    Args:
        loader: Loader expected to raise partway through iteration.

    Returns:
        The batches yielded before the failure, and the exception raised by
        the iteration.

    Raises:
        AssertionError: If the consumer is still blocked after
            ``_HANG_TIMEOUT_S`` or the iteration finished without raising.
    """
    batches: list = []
    outcome: dict[str, BaseException] = {}

    def _consume() -> None:
        """Iterate the loader, recording batches and the terminating exception.

        Returns:
            None. Results are written to the enclosing ``batches`` and ``outcome``.

        Raises:
            Nothing. Every exception is recorded for the main thread to assert on.
        """
        try:
            for batch in loader:
                batches.append(batch)
        except BaseException as exc:
            outcome["exc"] = exc

    consumer = threading.Thread(target=_consume, daemon=True)
    consumer.start()
    consumer.join(_HANG_TIMEOUT_S)
    assert not consumer.is_alive(), "consumer blocked: prefetch thread died silently"
    assert "exc" in outcome, "iteration finished without raising"
    return batches, outcome["exc"]


def _fail_on_call(monkeypatch: pytest.MonkeyPatch, fail_at: int) -> RuntimeError:
    """Make ``ExtendedDoTime.generate_batch`` raise on its ``fail_at``-th call.

    Earlier calls run the real generator, so the batches before the failure
    are genuine.

    Args:
        monkeypatch: Pytest fixture that restores the method after the test.
        fail_at: 1-based index of the call that raises.

    Returns:
        The exception instance the patched method will raise, so callers can
        check that the consumer receives this very object.

    Raises:
        Nothing itself. The patched method raises the returned exception.
    """
    real = dotime.extended.ExtendedDoTime.generate_batch
    boom = RuntimeError("boom")
    calls = {"n": 0}

    def _generate_batch(self, *args, **kwargs):
        """Delegate to the real generator until call ``fail_at``, then raise.

        Args:
            self: The ``ExtendedDoTime`` instance.
            *args: Positional arguments forwarded to the real method.
            **kwargs: Keyword arguments forwarded to the real method.

        Returns:
            The real method's batch dictionary for calls before ``fail_at``.

        Raises:
            RuntimeError: The ``boom`` instance, on call number ``fail_at``.
        """
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise boom
        return real(self, *args, **kwargs)

    monkeypatch.setattr(dotime.extended.ExtendedDoTime, "generate_batch", _generate_batch)
    return boom


def test_generation_error_reaches_consumer(monkeypatch):
    """The producer's own exception object is re-raised in the consumer.

    Args:
        monkeypatch: Pytest fixture used to make generation fail.

    Returns:
        None.

    Raises:
        AssertionError: If the consumer hangs, receives a batch, or sees any
            exception other than the one raised in the producer thread.
    """
    boom = _fail_on_call(monkeypatch, fail_at=1)
    batches, exc = _consume_with_timeout(_make_loader(prefetch=2))
    assert batches == []
    assert exc is boom


def test_batches_before_error_are_delivered_in_order(monkeypatch):
    """Batches generated before the failure are yielded first, unaltered.

    This matches the synchronous path, which yields them before raising.

    Args:
        monkeypatch: Pytest fixture used to make the third call fail.

    Returns:
        None.

    Raises:
        AssertionError: If the prefetch path drops, reorders, or alters the
            batches that precede the failure, or raises something else.
    """
    expected = list(_make_loader(prefetch=0))[:2]
    boom = _fail_on_call(monkeypatch, fail_at=3)
    batches, exc = _consume_with_timeout(_make_loader(prefetch=2))
    assert exc is boom
    _assert_same_batches(batches, expected)


def test_early_stop_releases_producer():
    """Closing the iterator mid-epoch stops the producer instead of leaking it.

    With ``num_steps`` far above the queue depth, the producer is parked on a
    full queue when the consumer stops, which used to block it forever.

    Returns:
        None.

    Raises:
        AssertionError: If iteration did not start exactly one producer
            thread, or that thread is still alive after the iterator closes.
    """
    before = set(threading.enumerate())
    iterator = iter(_make_loader(num_steps=50, prefetch=1))
    next(iterator)
    started = set(threading.enumerate()) - before
    assert len(started) == 1
    (producer,) = started
    iterator.close()
    assert not producer.is_alive()


@pytest.mark.parametrize("prefetch", [1, 2])
def test_prefetch_matches_synchronous_batches(prefetch):
    """Prefetching is a pure throughput knob: same seed, bit-identical batches.

    Args:
        prefetch: Queue depth of the background producer.

    Returns:
        None.

    Raises:
        AssertionError: If any batch differs from the synchronous path.
    """
    sync = list(_make_loader(prefetch=0))
    assert len(sync) == 3
    _assert_same_batches(list(_make_loader(prefetch=prefetch)), sync)
