"""Frozen benchmark suites for DoTime.

This module exposes the *consumer* side of the released benchmarks: loading a
versioned, immutable suite (downloading + caching it from Zenodo on first use)
and iterating over its episodes for evaluation.

**Public surface**

- :class:`Episode`        — one trajectory: obs/int data, intervention, ground truth.
- :class:`BenchmarkSuite` — a named, versioned collection of episodes.
- :func:`load_benchmark`  — fetch a suite by name (cached under ``~/.cache``).
- :func:`available_suites`— list the registered suite names.

Notes for implementers
-----------------------
The download + parse path is stubbed where it touches real artifacts (marked
``TODO(release)``). The frozen on-disk format is a per-suite directory with a
``manifest.json`` plus one or more parquet shards in the tidy schema produced by
``scripts/build_release.py``. Wire :func:`_parse_suite_dir` to that schema.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from dotime.interventions import InterventionSpec

__all__ = [
    "BenchmarkSuite",
    "Episode",
    "SuiteMetadata",
    "available_suites",
    "load_benchmark",
]


# --------------------------------------------------------------------------- #
# Registry of released suites
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SuiteMetadata:
    """Static metadata for a released benchmark suite."""

    name: str
    version: str
    zenodo_record_id: str  # numeric Zenodo record id, e.g. "10567890"
    doi: str
    description: str
    n_episodes: int
    structures: tuple[str, ...] = ()
    license: str = "CC-BY-4.0"
    hf_repo_id: str = ""  # Hugging Face dataset repo, e.g. "thummd/dot-Identifiability-v1"

    @property
    def zenodo_files_url(self) -> str:
        return f"https://zenodo.org/api/records/{self.zenodo_record_id}"


# Suites are hosted on Hugging Face (mirror) + Zenodo (archive of record).
# Zenodo depositions are DRAFTS until published in the UI; HF is the default source.
_SUITE_REGISTRY: dict[str, SuiteMetadata] = {
    "dot-Identifiability-v1": SuiteMetadata(
        name="dot-Identifiability-v1",
        version="1.0.0",
        hf_repo_id="thummd/dot-Identifiability-v1",
        zenodo_record_id="20846064",
        doi="10.5281/zenodo.20846064",
        description="Named identification structures with exact counterfactuals.",
        n_episodes=10_800,
        structures=(
            "back_door",
            "observed_confounder",
            "confounder_mediator",
            "front_door",
            "mediator",
            "instrumental_variable",
            "rct_no_confounding",
            "unobserved_confounder",
        ),
    ),
    "dot-RegimeSwitch-v1": SuiteMetadata(
        name="dot-RegimeSwitch-v1",
        version="1.0.0",
        hf_repo_id="thummd/dot-RegimeSwitch-v1",
        zenodo_record_id="20846074",
        doi="10.5281/zenodo.20846074",
        description="Regime-switching SCMs (ITS generalization), break density in {2,3,5}.",
        n_episodes=10_000,
    ),
    "dot-Continuous-v1": SuiteMetadata(
        name="dot-Continuous-v1",
        version="1.0.0",
        hf_repo_id="thummd/dot-Continuous-v1",
        zenodo_record_id="20845981",
        doi="10.5281/zenodo.20845981",
        description="Continuous-time intervention windows, query offsets {1,2,3,5,10}.",
        n_episodes=10_000,
    ),
    "dot-Generic-100k": SuiteMetadata(
        name="dot-Generic-100k",
        version="1.0.0",
        hf_repo_id="thummd/dot-Generic-100k",
        zenodo_record_id="20845983",
        doi="10.5281/zenodo.20845983",
        description="100k trajectories from the full diverse prior (training scale).",
        n_episodes=100_000,
    ),
}


def available_suites() -> list[str]:
    """Return the names of all registered benchmark suites."""
    return sorted(_SUITE_REGISTRY)


# --------------------------------------------------------------------------- #
# Episode + suite containers
# --------------------------------------------------------------------------- #


@dataclass
class Episode:
    """A single benchmark trajectory and its associated queries.

    Attributes
    ----------
    x_obs:
        Observational trajectory, shape ``(T, N)``.
    x_int:
        Interventional trajectory under ``intervention``, shape ``(T, N)``.
    intervention:
        The applied intervention specification.
    y_true:
        Ground-truth interventional outcome(s) for the query/queries,
        shape ``(n_queries,)``.
    query_target:
        Index of the queried variable per query, shape ``(n_queries,)``.
    query_time:
        Query time (float in ``[0, 1]`` for continuous suites, or int step),
        shape ``(n_queries,)``.
    structure:
        Identification structure label (``"back_door"``, ...), if applicable.
    scm_id:
        Stable id of the generating SCM within the suite.
    metadata:
        Free-form per-episode metadata (effect magnitude, regime count, ...).
    """

    x_obs: torch.Tensor
    x_int: torch.Tensor
    intervention: InterventionSpec
    y_true: torch.Tensor
    query_target: torch.Tensor
    query_time: torch.Tensor
    structure: str | None = None
    scm_id: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return int(self.x_obs.shape[-1])

    @property
    def length(self) -> int:
        return int(self.x_obs.shape[0])


class BenchmarkSuite:
    """A named, versioned, immutable collection of :class:`Episode` objects."""

    def __init__(self, meta: SuiteMetadata, episodes: list[Episode]):
        self.meta = meta
        self._episodes = episodes

    # --- container protocol ------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self._episodes[idx]

    def __repr__(self) -> str:
        structs = f", {len(self.meta.structures)} structures" if self.meta.structures else ""
        return f"{self.meta.name} (v{self.meta.version}): {len(self._episodes)} episodes{structs}"

    # --- convenience views ------------------------------------------------- #

    def by_structure(self) -> Iterator[tuple[str, list[Episode]]]:
        """Yield ``(structure_name, episodes)`` groups.

        Episodes with ``structure is None`` are grouped under ``"_all"``.
        """
        groups: dict[str, list[Episode]] = {}
        for ep in self._episodes:
            groups.setdefault(ep.structure or "_all", []).append(ep)
        for name in sorted(groups):
            yield name, groups[name]

    def filter(self, structure: str) -> BenchmarkSuite:
        """Return a sub-suite containing only episodes of ``structure``."""
        eps = [e for e in self._episodes if e.structure == structure]
        return BenchmarkSuite(self.meta, eps)


# --------------------------------------------------------------------------- #
# Loading: cache -> download -> parse  (with local-generation fallback)
# --------------------------------------------------------------------------- #


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir)
    else:
        env = os.environ.get("DOTIME_CACHE")
        root = Path(env) if env else Path.home() / ".cache" / "dotime"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_benchmark(
    name: str,
    version: str = "latest",
    *,
    force_download: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
) -> BenchmarkSuite:
    """Load a frozen benchmark suite by name.

    On first use the suite is downloaded from Zenodo into the cache directory
    (``~/.cache/dotime`` by default, override with
    ``$DOTIME_CACHE`` or the ``cache_dir`` argument). Subsequent calls
    read from the cache.

    Parameters
    ----------
    name:
        Suite name, e.g. ``"dot-Identifiability-v1"``. See
        :func:`available_suites`.
    version:
        Suite version. ``"latest"`` resolves to the registered version.
    force_download:
        Re-download even if a cached copy exists.
    cache_dir:
        Override the cache root.

    Returns
    -------
    BenchmarkSuite
    """
    if name not in _SUITE_REGISTRY:
        raise KeyError(f"unknown benchmark suite {name!r}; available: {available_suites()}")
    meta = _SUITE_REGISTRY[name]
    if version not in ("latest", meta.version):
        raise ValueError(f"suite {name!r} has version {meta.version!r}, requested {version!r}")

    suite_dir = _cache_root(cache_dir) / f"{name}-{meta.version}"

    if force_download or not suite_dir.exists():
        # Prefer Hugging Face (faster mirror), fall back to Zenodo (archive of
        # record). If neither is configured yet, regenerate locally so downstream
        # code stays testable before the suites are hosted.
        if meta.hf_repo_id:
            _download_from_hf(meta, suite_dir, force=force_download)
        elif meta.zenodo_record_id not in ("TODO", "LOCAL"):
            _download_from_zenodo(meta, suite_dir, force=force_download)
        else:
            return _generate_fallback(meta)

    return _parse_suite_dir(meta, suite_dir)


def _download_from_hf(meta: SuiteMetadata, dest: Path, *, force: bool) -> None:
    """Download a suite from its Hugging Face dataset repo into ``dest``.

    Needs the ``hf`` extra (``huggingface_hub``). Uses ``snapshot_download`` to
    pull the suite directory (parquet shards + ``manifest.json``); md5 validation
    happens in :func:`_parse_suite_dir`.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "downloading suites from Hugging Face needs the 'hf' extra:\n"
            "    pip install 'dotime[hf]'"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=meta.hf_repo_id,
        repo_type="dataset",
        revision=f"v{meta.version}",
        local_dir=str(dest),
        force_download=force,
    )


def _download_from_zenodo(meta: SuiteMetadata, dest: Path, *, force: bool) -> None:
    """Download all files for a suite's Zenodo record into ``dest`` (stdlib only).

    Fetches the record JSON, streams each file, verifies its md5, and writes a
    small ``download.json`` marker. The suite's own ``manifest.json`` (one of the
    downloaded files) is what :func:`_parse_suite_dir` validates against.
    """
    import hashlib
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(meta.zenodo_files_url) as resp:
        record = json.loads(resp.read().decode())

    for entry in record.get("files", []):
        url = entry["links"]["self"]
        name = entry.get("key") or url.rsplit("/", 1)[-1]
        out = dest / name
        if out.exists() and not force:
            continue
        with urllib.request.urlopen(url) as resp, out.open("wb") as fh:
            h = hashlib.md5()
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                fh.write(chunk)
                h.update(chunk)
        checksum = entry.get("checksum", "")
        if checksum.startswith("md5:") and h.hexdigest() != checksum[4:]:
            raise ValueError(
                f"checksum mismatch for {name} from Zenodo record {meta.zenodo_record_id}"
            )

    (dest / "download.json").write_text(
        json.dumps({"record_id": meta.zenodo_record_id, "doi": meta.doi})
    )


def _parse_suite_dir(meta: SuiteMetadata, suite_dir: Path) -> BenchmarkSuite:
    """Parse a cached suite directory into a :class:`BenchmarkSuite`.

    Delegates to the canonical reader in :mod:`dotime._release_io`, which
    validates the manifest schema version and per-shard md5 checksums.
    """
    from dotime import _release_io

    return _release_io.read_suite(meta, suite_dir)


def _generate_fallback(meta: SuiteMetadata, n: int = 64) -> BenchmarkSuite:
    """Generate a tiny in-memory suite so code/tests run before Zenodo minting.

    This is NOT the released artifact — it is a development convenience that
    produces ``n`` episodes from the live prior. Remove once suites are hosted.
    """
    from dotime import DoTime

    prior = DoTime(seed=0)
    episodes: list[Episode] = []
    structures = meta.structures or (None,)
    for i in range(n):
        x_obs, x_int, intervention, _scm = prior.generate_pair(T=200)
        episodes.append(
            episode_from_pair(
                x_obs,
                x_int,
                intervention,
                structure=structures[i % len(structures)],
                scm_id=i,
                metadata={"fallback": True},
            )
        )
    return BenchmarkSuite(meta, episodes)


_INT_TYPE_BY_CODE = {0: "hard", 1: "soft", 2: "time_varying"}


def episode_from_sample(
    sample: dict,
    *,
    structure: str | None = None,
    scm_id: int | None = None,
    metadata: dict | None = None,
) -> Episode:
    """Build an :class:`Episode` from a generator ``generate_sample`` dict.

    Works for the structured generators (``ExtendedDoTime``,
    ``ContinuousExtendedPrior``) whose samples carry exact counterfactual targets
    and a per-structure query protocol. Trajectories padded to ``n_max`` are
    un-padded to clean ``(T, n_vars)`` here — this is the model-facing/release
    boundary for the padding, so released tensors carry no zero columns.
    """
    from dotime.interventions import InterventionType

    # Released episodes carry the FULL observational trajectory; causal masking
    # (zeroing post-onset) is a model-input transform applied by the baseline.
    x_obs = sample.get("X_obs_full", sample["X_obs"])
    x_int = sample["X_int"]
    # Un-pad to the true number of variables when the sample reports it.
    n_vars = int(sample["num_vars"].item()) if "num_vars" in sample else x_obs.shape[-1]
    x_obs = x_obs[:, :n_vars].clone()
    x_int = x_int[:, :n_vars].clone()

    int_type_code = int(sample["intervention_type"].item())
    raw_value = sample.get("intervention_value_raw", sample.get("intervention_value"))
    onset = sample.get("int_onset_idx")
    intervention = InterventionSpec(
        targets=[int(sample["intervention_target"].item())],
        times=[int(onset.item())] if onset is not None else [],
        intervention_type=InterventionType(_INT_TYPE_BY_CODE.get(int_type_code, "hard")),
        values=float(raw_value.item()) if raw_value is not None else 0.0,
    )

    y_true = torch.as_tensor(sample["Y_true"], dtype=torch.float32).reshape(-1)
    extra = {}
    if "Y_causal_effect" in sample:
        extra["y_causal_effect"] = torch.as_tensor(
            sample["Y_causal_effect"], dtype=torch.float32
        ).reshape(-1)
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=intervention,
        y_true=y_true,
        query_target=torch.as_tensor(sample["query_target"], dtype=torch.long).reshape(-1),
        query_time=torch.as_tensor(sample["query_time"], dtype=torch.float32).reshape(-1),
        structure=structure,
        scm_id=scm_id,
        metadata={**(metadata or {}), "y_oracle": y_true, **extra},
    )


def episode_from_pair(
    x_obs: torch.Tensor,
    x_int: torch.Tensor,
    intervention: InterventionSpec,
    *,
    structure: str | None = None,
    scm_id: int | None = None,
    metadata: dict | None = None,
) -> Episode:
    """Build an :class:`Episode` from a paired (obs, int) trajectory.

    The query targets the last step of the most intervention-affected variable
    that is not itself a treatment target — its interventional value is the exact
    counterfactual ground truth (also stored as ``y_oracle`` for the Oracle
    baseline). Shared by the local fallback suite and ``dotime-generate``.
    """
    t_query = x_int.shape[0] - 1
    effect = (x_int[t_query] - x_obs[t_query]).abs().clone()
    for tgt in intervention.targets:
        if 0 <= tgt < effect.numel():
            effect[tgt] = -1.0
    query_var = int(torch.argmax(effect).item())
    y_true = x_int[t_query, query_var].reshape(1).clone()
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=intervention,
        y_true=y_true,
        query_target=torch.tensor([query_var]),
        query_time=torch.tensor([float(t_query)]),
        structure=structure,
        scm_id=scm_id,
        metadata={**(metadata or {}), "y_oracle": y_true},
    )
