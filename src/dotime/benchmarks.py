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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from dotime.interventions import InterventionSpec

__all__ = [
    "QUERY_TIME_ENCODINGS",
    "BenchmarkSuite",
    "Episode",
    "SuiteMetadata",
    "available_suites",
    "load_benchmark",
    "query_time_to_index",
]


# --------------------------------------------------------------------------- #
# Query-time encodings
# --------------------------------------------------------------------------- #

#: How ``Episode.query_time`` encodes the queried row of a ``T``-step trajectory.
#: ``"step"`` stores the row index itself (generic and regime generators),
#: ``"index/T"`` stores ``index / T`` (``ExtendedDoTime``, the identifiability
#: suite) and ``"index/(T-1)"`` stores ``index / (T - 1)``, which is the continuous
#: prior's normalized observation time on its regular grid.
QUERY_TIME_ENCODINGS = ("step", "index/T", "index/(T-1)")


def query_time_to_index(
    query_time: torch.Tensor | Sequence[float], length: int, encoding: str | None = None
) -> list[int]:
    """Map encoded query times to row indices of a ``length``-step trajectory.

    Parameters
    ----------
    query_time:
        Encoded query time of each query.
    length:
        Number of rows ``T`` of the episode's trajectories.
    encoding:
        One of :data:`QUERY_TIME_ENCODINGS`. ``None`` infers the encoding per
        value, as the evaluation helpers did before suites declared one. Values
        up to 1 are then read as ``index / T`` and larger values as steps. That
        guess is wrong for ``dot-Continuous-v1``, which is why every registered
        suite declares its encoding.

    Returns
    -------
    list of int
        One row index per query, clamped to ``[0, length - 1]``.

    Raises
    ------
    ValueError
        If ``length`` is not positive, if ``encoding`` is unknown, or if a
        declared encoding does not land a query on a whole row, which is the
        signature of a suite declared with the wrong encoding.
    """
    if length < 1:
        raise ValueError(f"trajectory length must be positive, got {length}")
    if encoding is not None and encoding not in QUERY_TIME_ENCODINGS:
        raise ValueError(
            f"unknown query_time encoding {encoding!r}; expected one of {QUERY_TIME_ENCODINGS}"
        )
    scale = {"step": 1.0, "index/T": float(length), "index/(T-1)": float(length - 1)}
    # Positions are rebuilt from float32 values (relative error 2**-24), so a
    # correct declaration lands within ~length * 6e-8 of a whole row, while the
    # neighbouring encoding misses by up to half a row.
    tol = max(1e-3, length * 1e-6)
    out = []
    for v in torch.as_tensor(query_time, dtype=torch.float64).reshape(-1).tolist():
        if encoding is None:
            # Kept verbatim so episodes that declare nothing (e.g. hand-built
            # ones) resolve exactly as before.
            idx = round(v * length) if v <= 1.0 else int(v)
        else:
            pos = v * scale[encoding]
            idx = round(pos)
            if abs(pos - idx) > tol:
                raise ValueError(
                    f"query_time {v!r} is not a whole row under encoding {encoding!r} "
                    f"for T={length}; the declared encoding does not match the data"
                )
        out.append(min(max(idx, 0), length - 1))
    return out


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
    # Earlier released versions that stay loadable by pinning ``version=``:
    # ((version, zenodo_record_id), ...). The HF mirror serves them from the
    # matching ``v<version>`` tag; Zenodo needs the per-version record id.
    prior_versions: tuple[tuple[str, str], ...] = ()
    # How ``Episode.query_time`` maps to a row in this suite's frozen files (one
    # of QUERY_TIME_ENCODINGS). The generators disagree, and a fraction does not
    # say which one wrote it, so the loader resolves the row from this
    # declaration. ``None`` falls back to the per-value guess.
    query_time_encoding: str | None = None

    def for_version(self, version: str) -> SuiteMetadata:
        """Return the metadata for a specific released version of this suite.

        Parameters
        ----------
        version : str
            ``"latest"``, the registered version, or one of the
            ``prior_versions``.

        Returns
        -------
        SuiteMetadata
            ``self`` for the registered version, otherwise a copy whose
            ``version`` and ``zenodo_record_id`` point at the pinned release.

        Raises
        ------
        ValueError
            If ``version`` was never released for this suite.
        """
        if version in ("latest", self.version):
            return self
        for prior, record_id in self.prior_versions:
            if prior == version:
                return replace(self, version=prior, zenodo_record_id=record_id)
        known = [self.version, *[v for v, _ in self.prior_versions]]
        raise ValueError(f"suite {self.name!r} has versions {known}, requested {version!r}")

    @property
    def zenodo_files_url(self) -> str:
        return f"https://zenodo.org/api/records/{self.zenodo_record_id}"


# Suites are hosted on Hugging Face (mirror) + Zenodo (archive of record).
# Zenodo depositions are DRAFTS until published in the UI; HF is the default source.
_SUITE_REGISTRY: dict[str, SuiteMetadata] = {
    "dot-Identifiability-v1": SuiteMetadata(
        name="dot-Identifiability-v1",
        version="1.1.0",
        hf_repo_id="thummd/dot-Identifiability-v1",
        zenodo_record_id="22673322",  # 1.1.0 version record (concept DOI below is stable)
        doi="10.5281/zenodo.20846063",  # concept DOI (resolves to latest version)
        description=(
            "Named identification structures with exact shared-noise counterfactual "
            "targets (1.1.0). Version 1.0.0 paired independent noise draws."
        ),
        n_episodes=10_800,
        # 1.0.0 stays loadable via load_benchmark(..., version="1.0.0") so the
        # published numbers remain reproducible from the frozen artifact.
        prior_versions=(("1.0.0", "20919553"),),
        query_time_encoding="index/T",  # both versions: ExtendedDoTime stores index / T
        structures=(
            "back_door",
            "observed_confounder",
            "confounder_mediator",
            "front_door",
            "mediator",
            "instrumental_variable",
            "bi_variate",
            "unobserved_confounder",
        ),
    ),
    "dot-RegimeSwitch-v1": SuiteMetadata(
        name="dot-RegimeSwitch-v1",
        version="1.0.0",
        hf_repo_id="thummd/dot-RegimeSwitch-v1",
        zenodo_record_id="20919599",
        doi="10.5281/zenodo.20846073",  # concept DOI (resolves to latest version)
        description="Regime-switching SCMs (ITS generalization), break density in {2,3,5}.",
        n_episodes=9_999,
        query_time_encoding="step",
    ),
    "dot-Continuous-v1": SuiteMetadata(
        name="dot-Continuous-v1",
        version="1.0.0",
        hf_repo_id="thummd/dot-Continuous-v1",
        zenodo_record_id="20919057",
        doi="10.5281/zenodo.20845980",  # concept DOI (resolves to latest version)
        description="Continuous-time intervention windows; query times uniform over [onset, T-1].",
        n_episodes=9_999,
        # Normalized observation time on the regular dt=1 grid of the 1.0.0
        # build, i.e. index / (T - 1). Reading it as index / T is one step late
        # for every query in the second half of the trajectory.
        query_time_encoding="index/(T-1)",
    ),
    "dot-Generic-100k": SuiteMetadata(
        name="dot-Generic-100k",
        version="1.0.0",
        hf_repo_id="thummd/dot-Generic-100k",
        zenodo_record_id="20919177",
        doi="10.5281/zenodo.20845982",  # concept DOI (resolves to latest version)
        description="100k trajectories from the full diverse prior (training scale).",
        n_episodes=100_000,
        query_time_encoding="step",
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
        Encoded query time per query, shape ``(n_queries,)``. The encoding
        depends on the generator (see :data:`QUERY_TIME_ENCODINGS`): a step for
        the generic and regime suites, ``index / T`` for identifiability and
        ``index / (T - 1)`` for continuous. Read rows through
        :attr:`query_time_idx` rather than decoding this field.
    structure:
        Identification structure label (``"back_door"``, ...), if applicable.
    scm_id:
        Stable id of the generating SCM within the suite.
    metadata:
        Free-form per-episode metadata (effect magnitude, regime count, ...).
        ``query_time_idx`` holds the exact row of each query when the episode
        constructors or the suite loader recorded it.
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

    @property
    def query_time_idx(self) -> torch.Tensor:
        """Row of ``x_obs`` / ``x_int`` that each query refers to.

        The generators encode :attr:`query_time` differently and a fraction
        does not say which encoding wrote it, so the row is resolved from the
        ``query_time_idx`` metadata recorded by :func:`episode_from_sample`,
        :func:`episode_from_pair` and the frozen-suite loader. An episode that
        records none, such as one built by hand, falls back to the per-value
        guess of :func:`query_time_to_index`.

        Returns:
            ``torch.long`` tensor of shape ``(n_queries,)``.

        Raises:
            ValueError: If the recorded rows do not match the number of queries
                or fall outside the trajectory.
        """
        recorded = self.metadata.get("query_time_idx")
        if recorded is None:
            return torch.tensor(query_time_to_index(self.query_time, self.length), dtype=torch.long)
        idx = torch.as_tensor(recorded, dtype=torch.long).reshape(-1)
        n_queries = self.query_time.numel()
        if idx.numel() != n_queries or bool((idx < 0).any() or (idx >= self.length).any()):
            raise ValueError(
                f"episode {self.scm_id} records query_time_idx {idx.tolist()} for "
                f"{n_queries} queries on a {self.length}-step trajectory"
            )
        return idx

    @property
    def is_self_query(self) -> bool:
        """Whether the queried variable is the intervened variable itself.

        Self-queries arise in the continuous suite because its query target is
        drawn uniformly over the observable variables, treatment included, so
        about one third of its queries ask for the treated variable. Inside the
        intervention window of a hard intervention the answer is the do-value,
        which any intervention-aware model receives as an input, so evaluators
        should report whether self-queries are included. Structure-defined
        suites (identifiability) never produce them.

        Returns
        -------
        bool
            ``True`` when every query targets one of the intervention targets.
        """
        targets = set(int(t) for t in self.intervention.targets)
        qts = [int(q) for q in self.query_target.reshape(-1).tolist()]
        return bool(qts) and all(q in targets for q in qts)


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
    # Pinning an earlier release keeps published numbers reproducible after the
    # registry advances (v1.1 suites are new artifacts, not corrected copies).
    meta = _SUITE_REGISTRY[name].for_version(version)

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


def _sample_query_time_idx(sample: dict, query_time: torch.Tensor, t_len: int) -> list[int]:
    """Exact row of each query in a structured-generator sample.

    The two structured generators encode ``query_time`` differently, so the row
    is recovered from what each one emits rather than from a shared guess.
    ``ContinuousExtendedPrior`` samples carry their observation grid ``times``
    and the absolute query time ``t_query == times[row]``. Locating ``t_query``
    on that grid is exact for regular and irregular schedules alike, where no
    fraction-based encoding is. ``ExtendedDoTime`` samples store ``row / T``.

    Args:
        sample: A ``generate_sample`` dict.
        query_time: The sample's ``query_time`` as a flat tensor.
        t_len: Number of observations ``T``.

    Returns:
        One row index per query.

    Raises:
        ValueError: If ``t_query`` does not lie on the sample's time grid, or a
            sample without a grid does not store ``row / T``.
    """
    if "times" in sample and "t_query" in sample:
        times = torch.as_tensor(sample["times"]).reshape(1, -1)
        t_query = torch.as_tensor(sample["t_query"], dtype=times.dtype).reshape(-1, 1)
        dist, rows = (times - t_query).abs().min(dim=1)
        if float(dist.max()) > 1e-6 * max(1.0, float(times.abs().max())):
            raise ValueError("continuous sample has a t_query that is not on its time grid")
        return [int(r) for r in rows.tolist()]
    return query_time_to_index(query_time, t_len, "index/T")


def episode_from_sample(
    sample: dict,
    *,
    structure: str | None = None,
    scm_id: int | None = None,
    metadata: dict | None = None,
) -> Episode:
    """Build an :class:`Episode` from a generator ``generate_sample`` dict.

    Works for the structured generators (``ExtendedDoTime``,
    ``ContinuousExtendedPrior``) whose samples carry exact interventional targets
    (counterfactual when the generator shares noise across arms)
    and a per-structure query protocol. Trajectories padded to ``n_max`` are
    un-padded to clean ``(T, n_vars)`` here — this is the model-facing/release
    boundary for the padding, so released tensors carry no zero columns. The
    exact row of each query is recorded as ``metadata["query_time_idx"]``,
    because the two generators encode ``query_time`` differently. A sample of a
    driven structure (see :mod:`dotime.drivers`) also records its driver as
    ``metadata["driver"]``.
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
    query_time = torch.as_tensor(sample["query_time"], dtype=torch.float32).reshape(-1)
    extra: dict[str, object] = {}
    if "Y_causal_effect" in sample:
        extra["y_causal_effect"] = torch.as_tensor(
            sample["Y_causal_effect"], dtype=torch.float32
        ).reshape(-1)
    if "driver" in sample:
        # A hidden driver is zeroed in both released arms, so this record is the
        # only way a consumer can stratify by it or rebuild it
        # (dotime.drivers.released_driver_series).
        extra["driver"] = sample["driver"]
    # Recorded at build time, where the generator's own convention is still
    # known, so no consumer has to decode query_time later.
    extra["query_time_idx"] = _sample_query_time_idx(sample, query_time, int(x_int.shape[0]))
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=intervention,
        y_true=y_true,
        query_target=torch.as_tensor(sample["query_target"], dtype=torch.long).reshape(-1),
        query_time=query_time,
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
    baseline). ``query_time`` is that step, also recorded as
    ``metadata["query_time_idx"]``. Shared by the local fallback suite and
    ``dotime-generate``.
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
        metadata={**(metadata or {}), "y_oracle": y_true, "query_time_idx": [t_query]},
    )
