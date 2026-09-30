"""On-the-fly temporal intervention dataloader.

Generates batches by sampling from the extended DoTime,
following the pattern of Do-PFN's ObservationalDataLoader.

Supports background prefetching to overlap data generation with GPU compute.
"""

import logging
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread

import torch

from dotime.extended import ExtendedDoTime, check_batched_intervention_source
from dotime.normalization import normalize_batch
from dotime.qa import batch_target_qa

_LOG = logging.getLogger(__name__)

# Per-TSCM-structure canonical query offset range. Matches the protocol behind
# the paper's structure-matched gap tables (results/reference/structure_matched/;
# the training/eval drivers live in the upstream do-over-time-pfn codebase).
PER_STRUCT_OFFSET_RANGE: dict[str, tuple[int, int]] = {
    "back_door": (0, 0),
    "front_door": (1, 5),
    "instrumental_variable": (0, 5),
}

# Target QA checks the first queries of each structure once at least this many
# have been generated: one batch at common batch sizes, i.e. step zero, and
# enough queries for the nonzero fractions to mean something.
_QA_MIN_QUERIES = 64
# Query-level batch fields that target QA reads. They are copied, because the
# training loop owns the batch once it is yielded.
_QA_KEYS = ("Y_true", "Y_obs", "Y_causal_effect", "query_time", "int_onset_idx", "_traj_idx")

# How often a producer parked on a full queue re-checks whether the consumer has
# gone. Far below the 5 s join in _iter_prefetch, so an abandoned producer exits
# before the consumer stops waiting for it. A put with room returns at once, so
# this costs one wakeup per interval, and only while the producer is ahead.
_PUT_POLL_S = 0.1


@dataclass(frozen=True)
class _PrefetchError:
    """Carries an exception from the prefetch producer thread to the consumer.

    A dedicated envelope, rather than the bare exception, keeps the queue
    protocol explicit: the consumer re-raises only what the producer caught.

    Args:
        exc: The exception raised while generating a batch. The consumer
            re-raises this object unchanged.
    """

    exc: BaseException


class TemporalInterventionDataLoader:
    """Infinite dataloader that generates temporal intervention batches on-the-fly.

    With ``target_qa=True`` (the default) the raw targets of the first 64
    queries of every structure are checked by
    :func:`dotime.qa.batch_target_qa`: both level arms must be finite, varied and
    mostly nonzero, and with ``target_key="Y_causal_effect"`` the effect must be
    nonzero on queries that can carry one. The statistics are logged through
    ``logging`` at warning level, so they show without any logging setup. The
    check reads tensors only, so batches are bit-identical with it on or off.

    Raises:
        ValueError: At construction, if both ``tscm_structure`` and
            ``tscm_structures`` are given.
        NotImplementedError: At construction, if a named structure is combined with
            an ``intervention_source`` that ``ExtendedDoTime.generate_batch`` cannot
            apply (see :func:`dotime.extended.check_batched_intervention_source`).
        dotime.qa.TargetQAError: While iterating, when a structure's first
            targets fail target QA (also through the prefetch thread).
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        n_max: int = 41,
        n_max_prior: int = 10,
        t_range: tuple = (50, 200),
        burn_in: int = 50,
        downstream_prob: float = 0.7,
        seed: int = 42,
        normalize: bool = True,
        device: str = "cpu",
        num_workers: int = 0,
        prefetch: int = 2,
        target_key: str = "Y_true",
        n_queries: int = 1,
        query_mode: str = "single",
        intervention_source: str = "prior",
        tscm_structure: str | None = None,
        tscm_structures: list[str] | None = None,
        use_lagged_edges: bool = True,
        intervention_scale: float = 2.0,
        causal_mask_mode: str = "full",
        dynamics_burn_in: int = 0,
        sim_device: str | None = None,
        query_offset_range: tuple = (0, 0),
        hardening: dict | None = None,
        pair_mode: str = "interventional",
        divergence_fallback: str | None = None,
        target_qa: bool = True,
    ):
        # pair_mode and divergence_fallback go to every ExtendedDoTime unchanged. With a
        # named structure, divergence_fallback="batched" redraws diverged samples with
        # the batched simulator (hardened like the rest of the batch). The default None
        # keeps the per-sample replacement of interventional batches, which released
        # checkpoints were trained with, and redraws counterfactual ones.
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.target_key = target_key
        self.n_queries = n_queries
        self.query_mode = query_mode
        # Step-zero target QA state: raw-target snapshots per structure until
        # the structure is checked, then only its name in _qa_done.
        self.target_qa = target_qa
        self._qa_pending: dict[str | None, list[dict[str, torch.Tensor]]] = {}
        self._qa_counts: dict[str | None, int] = {}
        self._qa_done: set[str | None] = set()

        # Default sim_device to CPU. The BatchedTSCMSimulator's sequential T-loop
        # has too much kernel-launch overhead on GPU for typical batch sizes;
        # CPU is faster for B=16 with N<10 vars. Use sim_device='cuda:N' only if
        # you have very large batches where GPU saturates.
        if sim_device is None:
            sim_device = "cpu"

        if tscm_structures is not None and tscm_structure is not None:
            raise ValueError(
                "Pass either tscm_structure (single) or tscm_structures (list), not both."
            )

        if tscm_structures is not None:
            # Multi-structure: one prior per structure, each with its
            # canonical query_offset_range. _generate_batch picks one
            # structure uniformly at random per call so the model sees
            # all three identification strategies during training.
            self.priors = [
                ExtendedDoTime(
                    n_max=n_max,
                    n_max_prior=n_max_prior,
                    t_range=t_range,
                    burn_in=burn_in,
                    downstream_prob=downstream_prob,
                    seed=seed + i,
                    intervention_source=intervention_source,
                    tscm_structure=s,
                    use_lagged_edges=use_lagged_edges,
                    intervention_scale=intervention_scale,
                    causal_mask_mode=causal_mask_mode,
                    dynamics_burn_in=dynamics_burn_in,
                    sim_device=sim_device,
                    query_offset_range=PER_STRUCT_OFFSET_RANGE[s],
                    hardening=hardening,
                    pair_mode=pair_mode,
                    divergence_fallback=divergence_fallback,
                )
                for i, s in enumerate(tscm_structures)
            ]
            self._struct_names = list(tscm_structures)
            self._rng = random.Random(seed)
            self.prior = self.priors[0]  # default for any external readers
        else:
            self.prior = ExtendedDoTime(
                n_max=n_max,
                n_max_prior=n_max_prior,
                t_range=t_range,
                burn_in=burn_in,
                downstream_prob=downstream_prob,
                seed=seed,
                intervention_source=intervention_source,
                tscm_structure=tscm_structure,
                use_lagged_edges=use_lagged_edges,
                intervention_scale=intervention_scale,
                causal_mask_mode=causal_mask_mode,
                dynamics_burn_in=dynamics_burn_in,
                sim_device=sim_device,
                query_offset_range=query_offset_range,
                hardening=hardening,
                pair_mode=pair_mode,
                divergence_fallback=divergence_fallback,
            )
            self.priors = None

        # generate_batch is this loader's only generation path, and an exception raised
        # in the prefetch thread never reaches the training loop, which then blocks on
        # the queue forever. Checking here raises in the caller's thread instead.
        for prior in self.priors or [self.prior]:
            check_batched_intervention_source(prior.intervention_source, prior.tscm_structure)

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.prefetch > 0:
            yield from self._iter_prefetch()
        else:
            for _ in range(self.num_steps):
                yield self._generate_batch()

    def _iter_prefetch(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches generated ahead of time by a background producer thread.

        The producer fills a queue of depth ``self.prefetch`` so generation
        overlaps the consumer's compute, then ends the stream with a sentinel.
        If generation raises, the producer queues the exception in place of the
        sentinel, and the consumer re-raises it after the batches generated
        before it, exactly as the synchronous path would. If the consumer stops
        early (``break``, ``close()``, garbage collection or its own exception),
        a stop event makes the producer drop its pending batch and exit instead
        of blocking forever on a full queue.

        Yields:
            Batches in generation order, identical to the synchronous path.

        Raises:
            BaseException: Whatever ``_generate_batch`` raised in the producer
                thread, re-raised as the same object so callers can handle it
                exactly as with ``prefetch=0``.
        """
        queue: Queue = Queue(maxsize=self.prefetch)
        sentinel = object()
        stop = Event()

        def _put(item: object) -> bool:
            """Block until ``item`` is queued, unless the consumer stops first.

            Args:
                item: A batch, the sentinel, or a ``_PrefetchError``.

            Returns:
                True if the item was queued, False if the consumer stopped.

            Raises:
                Nothing. A timed-out attempt (``queue.Full``) is retried.
            """
            # A plain blocking put() never returns once the consumer is gone;
            # timed attempts give the producer a chance to see the stop event.
            while not stop.is_set():
                try:
                    queue.put(item, timeout=_PUT_POLL_S)
                except Full:
                    continue
                return True
            return False

        def _fill() -> None:
            """Producer loop: queue ``num_steps`` batches, then the sentinel.

            Returns:
                None. Batches and the terminal item travel through the queue.

            Raises:
                Nothing. An exception from ``_generate_batch`` is queued for the
                consumer instead of escaping, because a thread that dies without
                a terminal item leaves the consumer blocked on ``get()`` forever.
            """
            try:
                for _ in range(self.num_steps):
                    if not _put(self._generate_batch()):
                        return
            # BaseException, not Exception: anything that ends this thread without
            # a terminal item strands the consumer, including SystemExit, which
            # the threading module swallows silently.
            except BaseException as exc:
                _put(_PrefetchError(exc))
                return
            _put(sentinel)

        thread = Thread(target=_fill, daemon=True)
        thread.start()

        try:
            while True:
                item = queue.get()
                if item is sentinel:
                    break
                if isinstance(item, _PrefetchError):
                    # The original object keeps its type and its traceback into
                    # the producer frames, so a wrapper would only hide the cause.
                    raise item.exc
                yield item
        finally:
            # Also runs when the consumer abandons the generator mid-epoch
            # (GeneratorExit at the yield), which is what releases the producer.
            stop.set()
            # A generator still alive at interpreter shutdown is finalized after
            # daemon threads are frozen. Joining then cannot succeed, and on
            # Python 3.13 it stalls exit for the full timeout.
            if not sys.is_finalizing():
                thread.join(timeout=5)

    def _check_targets(self, structure: str | None, batch: dict[str, torch.Tensor]) -> None:
        """Collect a batch's raw targets and check a structure once it has enough.

        Args:
            structure: The ``tscm_structure`` of the prior that generated the
                batch, ``None`` for the generic prior.
            batch: The batch, before normalization or a device move.

        Raises:
            dotime.qa.TargetQAError: If the structure's first targets fail.
        """
        if structure in self._qa_done:
            return
        snap = {k: batch[k].detach().cpu().clone() for k in _QA_KEYS if k in batch}
        # Read for its length only, so a reference is enough.
        snap["X_obs"] = batch["X_obs"]
        self._qa_pending.setdefault(structure, []).append(snap)
        self._qa_counts[structure] = self._qa_counts.get(structure, 0) + batch["Y_true"].numel()
        if self._qa_counts[structure] >= _QA_MIN_QUERIES:
            self._qa_done.add(structure)
            batch_target_qa(
                self._qa_pending.pop(structure),
                structure=structure,
                target_key=self.target_key,
                log=_LOG.warning,
            )

    def _generate_batch(self) -> dict[str, torch.Tensor]:
        """Generate a single batch.

        Returns:
            The batch, normalized when ``normalize`` is set and moved to ``device``.

        Raises:
            dotime.qa.TargetQAError: If target QA is on and the first targets of
                this batch's structure fail it.
        """
        prior = self._rng.choice(self.priors) if self.priors is not None else self.prior
        batch = prior.generate_batch(
            self.batch_size,
            num_workers=self.num_workers,
            n_queries=self.n_queries,
            query_mode=self.query_mode,
        )
        if self.target_qa:
            self._check_targets(prior.tscm_structure, batch)

        if self.normalize:
            batch = normalize_batch(batch, target_key=self.target_key)

        # Move to device
        if self.device != "cpu":
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

        return batch
