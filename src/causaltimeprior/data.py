"""On-the-fly temporal intervention dataloader.

Generates batches by sampling from the extended CausalTimePrior,
following the pattern of Do-PFN's ObservationalDataLoader.

Supports background prefetching to overlap data generation with GPU compute.
"""

import random
from collections.abc import Iterator
from queue import Queue
from threading import Thread

import torch

from causaltimeprior.extended import ExtendedCausalTimePrior
from causaltimeprior.normalization import normalize_batch

# Per-TSCM-structure canonical query offset range (matches the per-structure
# eval pipeline in scripts/analyze_s9ho.py and the per-structure training
# launcher run_sanity9_hardened_oscillatory.sh).
PER_STRUCT_OFFSET_RANGE: dict[str, tuple[int, int]] = {
    "back_door": (0, 0),
    "front_door": (1, 5),
    "instrumental_variable": (0, 5),
}


class TemporalInterventionDataLoader:
    """Infinite dataloader that generates temporal intervention batches on-the-fly."""

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
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.target_key = target_key
        self.n_queries = n_queries
        self.query_mode = query_mode

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
                ExtendedCausalTimePrior(
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
                )
                for i, s in enumerate(tscm_structures)
            ]
            self._struct_names = list(tscm_structures)
            self._rng = random.Random(seed)
            self.prior = self.priors[0]  # default for any external readers
        else:
            self.prior = ExtendedCausalTimePrior(
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
            )
            self.priors = None

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.prefetch > 0:
            yield from self._iter_prefetch()
        else:
            for _ in range(self.num_steps):
                yield self._generate_batch()

    def _iter_prefetch(self) -> Iterator[dict[str, torch.Tensor]]:
        """Generate batches with background prefetching."""
        queue: Queue = Queue(maxsize=self.prefetch)
        sentinel = object()

        def _fill():
            for _ in range(self.num_steps):
                batch = self._generate_batch()
                queue.put(batch)
            queue.put(sentinel)

        thread = Thread(target=_fill, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item

        thread.join(timeout=5)

    def _generate_batch(self) -> dict[str, torch.Tensor]:
        """Generate a single batch."""
        prior = self._rng.choice(self.priors) if self.priors is not None else self.prior
        batch = prior.generate_batch(
            self.batch_size,
            num_workers=self.num_workers,
            n_queries=self.n_queries,
            query_mode=self.query_mode,
        )

        if self.normalize:
            batch = normalize_batch(batch, target_key=self.target_key)

        # Move to device
        if self.device != "cpu":
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

        return batch
